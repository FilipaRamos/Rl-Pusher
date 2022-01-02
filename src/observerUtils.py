#!/usr/bin/env python3
import math
import rospy
import rospkg
import numpy as np

from sklearn import cluster
from numpy.core.numeric import Inf

from sensor_msgs.msg import LaserScan

class TargetCylinder():
    def __init__(self, points, pos):
        self.points = points
        self.current_pos = np.asarray(pos)

    def set_points(self, points_n):
        self.points = points_n

    def set_cur_pos(self, newpos):
        self.current_pos = np.asarray(newpos)

    def dist_to(self):
        dists = []
        for point in self.points:
            dists.append(self.dist([0, 0], point))
        return min(dists)

    def dist(self, ref, point):
        return math.sqrt((point[0] - ref[0])**2 + (point[1] - ref[1])**2)

    def get_layout_dict(self):
        return { 'target_cyl': [self.current_pos] }

class Utils():

    def __init__(self):
        self.laser_sub = rospy.Subscriber("/scan", LaserScan, self.laser_callback, queue_size=1)

        self.precision = 0.01
        self.places = 2

        # Get ros package path
        ros_ = rospkg.RosPack()
        self.ros_path = ros_.get_path('deep-rl-pusher')

    def laser_callback(self, msg):
        self.x, self.y = self.generate_pc(msg.ranges, msg.range_min, msg.range_max, msg.angle_increment)
        self.clusters = self.segment_pc(self.x, self.y, msg.range_min)
        #self.plot_clusters(self.x, self.y, self.clusters)

    def generate_pc(self, ranges, range_min, range_max, angle_increment):
        ranges = np.array(ranges)
        angles = np.arange(0, 2*math.pi, angle_increment)

        # Inf ranges are spaces where no points exist
        x = [-math.sin(angle) * range for angle, range in zip(angles, ranges) if range != Inf]
        y = [math.cos(angle) * range for angle, range in zip(angles, ranges) if range != Inf]

        return x, y

    def exclude_walls(self, x_, y_, clusters_):
        x, y = [], []
        tmp_x, tmp_y = [], []
        clusters, tmp_clusters = [], []
        cur_cluster = clusters_[0]
        cur_cluster_size = 0
        for x_coord, y_coord, elem in zip(x_, y_, clusters_):
            if elem == cur_cluster:
                cur_cluster_size += 1
                tmp_x.append(x_coord)
                tmp_y.append(y_coord)
                tmp_clusters.append(elem)
            else:
                if cur_cluster_size < 15:
                    x.extend(tmp_x)
                    y.extend(tmp_y)
                    clusters.extend(tmp_clusters)
                tmp_x = []
                tmp_y = []
                tmp_clusters = []
                cur_cluster_size = 1
                cur_cluster = elem
        if cur_cluster_size < 15:
            x.extend(tmp_x)
            y.extend(tmp_y)
            clusters.extend(tmp_clusters)
        return x, y, clusters

    def transform_to_img(self, x, y, clusters):
        import math
        sizes = self.calculate_cluster_sizes(clusters)

        min_x = round(min(x), self.places)
        min_y = round(min(y), self.places)

        # Transform points to 0+ scale
        x_, y_, clusters_ = [], [], []
        for x_coord, y_coord, cluster in zip(x, y, clusters):
            x_coord = round(x_coord, self.places) + abs(min_x)
            y_coord = round(y_coord, self.places) + abs(min_y)
            x_.append(x_coord)
            y_.append(y_coord)
            clusters_.append(cluster)

        # Calculate length, height
        max_x_ = max(x_)
        max_y_ = max(y_)

        # Get up to desired precision
        length = int(max_x_ / self.precision)
        height = int(max_y_ / self.precision)
        image = np.zeros((length + 1, height + 1), dtype=np.float16)

        # Calculate buckets and image coords
        bucket_cap_l = int(length / max_x_)
        bucket_cap_h = int(height / max_y_)

        # Create a dictionary to save correspondence image coordinates -> pointcloud coordinates
        img_pc_coords = {}
        for x_coord, y_coord, c in zip(x_, y_, clusters_):
            # Get buckets
            x_bucket = int(x_coord)
            y_bucket = int(y_coord)

            x_idx = math.modf(x_coord)[0] * bucket_cap_l
            y_idx = math.modf(y_coord)[0] * bucket_cap_h

            x_img_coord = x_bucket * bucket_cap_l + int(x_idx)
            y_img_coord = y_bucket * bucket_cap_h + int(y_idx)
            
            assert type(x_img_coord) is int
            assert type(y_img_coord) is int
            image[x_img_coord, y_img_coord] = 255

            key = (y_img_coord, x_img_coord)
            x_img_pc = x_coord - abs(min_x)
            y_img_pc = y_coord - abs(min_y)
            img_pc_coords[key] = (x_img_pc, y_img_pc, sizes[c])

        return image, img_pc_coords

    def segment_pc(self, x, y, range_min):
        X = np.array((x, y)).T

        from sklearn.cluster import DBSCAN
        cluster = DBSCAN(eps=range_min*2, min_samples=3).fit(X)
        return cluster.labels_

    def find_cylinder(self, plot=False):
        if not self.x and not self.y and not self.clusters:
            raise Exception('find_cylinder', 'no data collected yet')
        x, y, clusters = self.exclude_walls(self.x, self.y, self.clusters)
        image, img_pc_coords = self.transform_to_img(x, y, clusters)
        
        if plot:
            self.plot_pc_img(image)
        cx, cy, radii = self.hough_transform(image, plot=plot)
        return self.transform_area_to_pc(cx, cy, radii, img_pc_coords)

    # Return True if circle, False if line
    def hough_transform(self, image, plot=False):
        from skimage.transform import hough_circle, hough_circle_peaks
        factor = 10**self.places
        hough_radii = np.arange(0.08*2*factor, 0.5*2*factor, 0.02*2*factor)
        res = hough_circle(image, hough_radii)
        # Extract the peak as we only want the closest cylinder 
        # (and the closest cylinder will have the largest amount of points and less noisy data)
        accums, cx, cy, radii = hough_circle_peaks(res, hough_radii,
                                        total_num_peaks=1)
        if plot:
            self.plot_hough_circle(image, cx, cy, radii)
        return cx, cy, radii

    def transform_area_to_pc(self, cx, cy, radii, img_pc_coords):
        x, y = [], []
        cluster_size = 0
        for img_coords, pc_coords in img_pc_coords.items():
            # Point is in the selected area
            if img_coords[0] in range(int(cx - radii), int(cx + radii)) and img_coords[1] in range(int(cy - radii), int(cy + radii)):
                x.append(pc_coords[0])
                y.append(pc_coords[1])
                cluster_size = pc_coords[2]
        X = np.array((x, y)).T
        return X, cluster_size

    def calculate_point_dist(self, cluster, p):
        dists = []
        for point in cluster:
            if (point[0] == p[0] and point[1] == p[1]) : continue
            dists.append(self.calculate_dist(point, p))
        return min(dists)

    def calculate_dist(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def calculate_nr_clusters(self, clusters):
        nr_clusters = (max(clusters) + 1)
        return nr_clusters - min(clusters)

    def calculate_cluster_sizes(self, clusters):
        from collections import Counter
        return Counter(clusters)

    def plot_clusters(self, x, y, clusters):
        import matplotlib.pyplot as plt
        import matplotlib
        #matplotlib.use('Agg')

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.scatter(x, y, s=.3, c=clusters)

        fig.savefig(self.ros_path + '/resources/clusters.png')
        plt.close(fig)

    def plot_hough_linear(self, image, theta, h, d):
        # Generating figure 1
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from skimage.transform import hough_line_peaks

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        ax = axes.ravel()

        ax[0].imshow(image, cmap=cm.gray)
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        angle_step = 0.5 * np.diff(theta).mean()
        d_step = 0.5 * np.diff(d).mean()
        bounds = [np.rad2deg(theta[0] - angle_step),
                np.rad2deg(theta[-1] + angle_step),
                d[-1] + d_step, d[0] - d_step]
        ax[1].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
        ax[1].set_title('Hough transform')
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].axis('image')

        ax[2].imshow(image, cmap=cm.gray)
        ax[2].set_ylim((image.shape[0], 0))
        ax[2].set_axis_off()
        ax[2].set_title('Detected lines')

        for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            ax[2].axline((x0, y0), slope=np.tan(angle + np.pi/2))

        plt.tight_layout()
        fig.savefig(self.ros_path + '/resources/lines.png')
        plt.close(fig)

    def plot_hough_circle(self, image, cx, cy, radii):
        # Generating figure 1
        import matplotlib.pyplot as plt
        from skimage.draw import circle_perimeter

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = circle_perimeter(center_y, center_x, int(math.ceil(radius)),
                                            shape=image.shape)
            image[circy, circx] = 50

        ax.set_title('Circle Hough')
        ax.imshow(image, cmap=plt.cm.gray)
        plt.tight_layout()
        fig.savefig(self.ros_path + '/resources/circles.png')
        plt.close(fig)

    def plot_pc_img(self, image):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
        ax.imshow(image)
        plt.tight_layout()
        fig.savefig(self.ros_path + '/resources/pc_image.png')
        plt.close(fig)
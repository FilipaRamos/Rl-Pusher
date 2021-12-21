#!/usr/bin/env python3
from numpy.core.numeric import Inf, NaN
from numpy.lib.function_base import _calculate_shapes, append
from sklearn import cluster
import rospy
import math
import numpy as np

from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped

class ObjectDetector:

    def __init__(self):

        self.laser_sub = rospy.Subscriber("/scan", LaserScan, self.laser_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback, queue_size=1)

        self.traj_pub = rospy.Publisher('trajectory', Path, queue_size=1000)
        self.traj_msg = Path()

        self.precision = 0.01
        self.places = 2

    def laser_callback(self, msg):
        self.x, self.y = self.generate_pc(msg.ranges, msg.range_min, msg.range_max, msg.angle_increment)
        self.clusters = self.segment_pc(self.x, self.y, msg.range_min)
        self.plot_clusters(self.x, self.y, self.clusters)

    def odom_callback(self, msg):
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        self.trajectory_publisher(msg, pose)

    def trajectory_publisher(self, msg, pose):
        self.traj_msg.header = msg.header
        self.traj_msg.poses.append(pose)

        self.traj_pub.publish(self.traj_msg)

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

    def transform_to_img(self, x, y):
        import math

        min_x = round(min(x), self.places)
        min_y = round(min(y), self.places)

        # Transform points to 0+ scale
        x_, y_ = [], []
        for x_coord, y_coord in zip(x, y):
            x_coord = round(x_coord, self.places) + abs(min_x)
            y_coord = round(y_coord, self.places) + abs(min_y)
            x_.append(x_coord)
            y_.append(y_coord)

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
        for x_coord, y_coord in zip(x_, y_):
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
            key = (x_img_coord, y_img_coord)
            img_pc_coords[key] = (x_coord, y_coord)

        return image, img_pc_coords

    def segment_pc(self, x, y, range_min):
        X = np.array((x, y)).T

        from sklearn.cluster import DBSCAN
        cluster = DBSCAN(eps=range_min*2, min_samples=3).fit(X)
        return cluster.labels_

    def find_cylinder(self, plot=False):
        x, y, clusters = self.exclude_walls(self.x, self.y, self.clusters)
        image, img_pc_coords = self.transform_to_img(x, y)
        
        if plot:
            self.plot_pc_img(image)
        cx, cy, radii = self.hough_transform(image)
        return self.transform_area_to_pc(cx, cy, radii, img_pc_coords)

    def detect_objs(self, x, y, clusters):
        nr_clusters = (max(clusters) + 1)
        classification = []
        for i in range(min(clusters), nr_clusters):
            x_c = np.array([x_ for x_, y_, c in zip(x, y, clusters) if c == i and x_ != Inf])
            y_c = np.array([y_ for x_, y_, c in zip(x, y, clusters) if c == i and y_ != Inf])
            # If the cluster has more than 20 points, it is probably the wall
            if x_c.shape[0] > 20 or x_c.shape[0] < 3:
                classification.append(False)
                continue
            cluster = np.array((x_c, y_c)).T
            #classification.append(self.hough_transform(cluster))
            classification.append(self.classify(cluster))
        return classification

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
        x = [], y = []
        for img_coords, pc_coords in img_pc_coords.items():
            # Point is in the selected area
            if img_coords[0] in range(cx - radii, cx + radii) and img_coords[1] in range(cy - radii, cy + radii):
                x.append(pc_coords[0])
                y.append(pc_coords[1])
        return np.array((x, y)).T        

    def calculate_point_dist(self, cluster, p):
        min_dist = 0
        for point in cluster:
            if (point[0] == p[0] and point[1] == p[1]) : continue
            dist = self.calculate_dist(point, p)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def calculate_dist(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def sample_cluster(self, cluster):
        import random
        if (cluster.shape[0] > 3) :
            p1 = random.randint(0, cluster.shape[0] - 1)
            p2 = random.choice([i for i in range(0, cluster.shape[0] - 1) if i not in [p1]])
            p3 = random.choice([i for i in range(0, cluster.shape[0] - 1) if i not in [p1, p2]])
            while (self.calculate_point_dist(cluster, cluster[p2]) == self.calculate_dist(cluster[p1], cluster[p2])) :
                p2_ = p2
                p2 = random.choice([i for i in range(0, cluster.shape[0] - 1) if i not in [p1, p2_]])
            while (self.calculate_point_dist(cluster, cluster[p3]) == self.calculate_dist(cluster[p2], cluster[p3]) 
                    and self.calculate_point_dist(cluster, cluster[p3]) == self.calculate_dist(cluster[p1], cluster[p3])) :
                p3_ = p3
                p3 = random.choice([i for i in range(0, cluster.shape[0] - 1) if i not in [p1, p2, p3_]])
            return cluster[p1], cluster[p2], cluster[p3]
        return cluster[0], cluster[1], cluster[2]

    def curvature(self, p1, p2, p3):
        area = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
        return abs(4*area / (self.calculate_dist(p1, p2) * self.calculate_dist(p2, p3) * self.calculate_dist(p3, p1)))

    # Binary classification
    # @return true if object is thought to be a cylinder
    def classify(self, edges):
        i = 0
        consensus = []
        threshold = 6
        while (i < 3):
            p1, p2, p3 = self.sample_cluster(edges)
            c = self.curvature(p1, p2, p3)
            print("----", c)
            # C = 1/R (in a line R->infinity, so C tends to 0)
            if (c < threshold): consensus.append(0)
            else: consensus.append(1)
            i += 1
        if (np.sum(consensus) > 1) : return True
        else : return False

    def calculate_nr_clusters(self, clusters):
        nr_clusters = (max(clusters) + 1)
        return nr_clusters - min(clusters)

    def plot_clusters(self, x, y, clusters):
    #def plot_clusters(self, x, y, clusters, classification):
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.scatter(x, y, s=.3, c=clusters)

        fig.savefig('./src/deep-rl-pusher/resources/clusters.png')
        plt.close(fig)

        # TODO: clusters look ok but classification not??
        '''markers = []
        nr_clusters = (max(clusters) + 1)
        for i in range(min(clusters), nr_clusters):
            marker = 'tab:orange' if classification[i] else 'tab:gray'
            c_marker = [marker for cluster in clusters if cluster == i]
            markers.extend(c_marker)
        #markers = [item for sublist in markers for item in sublist]
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_label('x')
        ax.set_label('y')
        ax.scatter(x, y, s=.3, c=markers)

        fig.savefig('./src/deep-rl-pusher/resources/objects.png')
        plt.close(fig)'''

        # TODO Filter out world end points

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
        fig.savefig('./src/deep-rl-pusher/resources/lines.png')
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
        fig.savefig('./src/deep-rl-pusher/resources/circles.png')
        plt.close(fig)

    def plot_pc_img(self, image):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
        ax.imshow(image)
        plt.tight_layout()
        fig.savefig('./src/deep-rl-pusher/resources/pc_image.png')
        plt.close(fig)
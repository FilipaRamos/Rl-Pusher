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

    def laser_callback(self, msg):
        x, y = self.generate_pc(msg.ranges, msg.range_min, msg.range_max, msg.angle_increment)
        clusters = self.segment_pc(x, y, msg.range_min)
        classification = self.detect_objs(x, y, clusters)
        self.plot_clusters(x, y, clusters, classification)

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

    def transform_to_img(self, x, y):
        max_x = int(max(x))
        min_x = int(min(x))
        max_y = int(max(y))
        min_y = int(min(y))

        length = max_x - min_x
        height = max_y - min_y

        image = np.zeros((length, height))

        for x_idx, y_idx in zip(x, y):
            # Indexes must be integers
            x_idx = int(x_idx)
            y_idx = int(y_idx)
            image[x_idx, y_idx] = 255

        return image

    def segment_pc(self, x, y, range_min):
        X = np.array((x, y)).T

        from skimage.transform import hough_line, hough_line_peaks
        image = self.transform_to_img(x, y)
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        h, theta, d = hough_line(image, theta=tested_angles)

        # Generating figure 1
        import matplotlib.pyplot as plt
        from matplotlib import cm
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
            ax[2].axhline((x0, y0), slope=np.tan(angle + np.pi/2))

        plt.tight_layout()
        fig.savefig('./src/deep-rl-pusher/resources/look.png')
        plt.close(fig)
        
        from sklearn.cluster import DBSCAN
        cluster = DBSCAN(eps=range_min*2, min_samples=3).fit(X)
        return cluster.labels_

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
    def hough_transform(self, edges):
        from skimage.transform import probabilistic_hough_line, hough_circle
        from skimage.feature import peak_local_max
        
        #lines = probabilistic_hough_line(edges, threshold=10, line_length=5, line_gap=3)
        hough_radii = np.arange(0.08, 1, 0.02)
        lines = hough_circle(edges, hough_radii)
        #print("Lines - {}".format(lines))
        # TODO: check peak?
        #print(len(list(lines)))
        if list(lines):
            #if len(lines) > 2:
            #    return False
            return True
        else:
            return False

        # Another idea:
        # Calculate curvature of the points in the cluster
        # 1st : Sample 3 points from the cluster (first, last, middle)?
        # 2nd : c (p1, p2, p3) = 4*A(area of the triangle created by the 3 points) / d(p1-p2)*d(p2-p3)*d(p3-p1)
        # Curvature larger than a threshold is a line (lines are infinite curvatures)

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
        threshold = 12
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

    def plot_clusters(self, x, y, clusters, classification):
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
        markers = []
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
        plt.close(fig)

        # TODO Filter out world end points
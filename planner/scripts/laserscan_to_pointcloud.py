#!/usr/bin/env python3

import sensor_msgs.point_cloud2 as pc2
import rospy
from sensor_msgs.msg import PointCloud2, LaserScan
import laser_geometry.laser_geometry as lg
import math
import numpy as np
import open3d as o3d
import open3d_conversions

rospy.init_node("laserscan_to_pointcloud")

lp = lg.LaserProjection()

pc_pub = rospy.Publisher("/pointcloud", PointCloud2, queue_size=1)

# parameters for front_laser to base_link frame pcd transform
translation = np.array([0.120, 0.000, 0.333])
xyz = np.array([0.000, -0.000, 0.000])

T = np.eye(4)
T[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz(xyz)
T[0, 3] = translation[0]
T[1, 3] = translation[1]
T[2, 3] = translation[2]

print("Transformation mtx: ")
print(T)

def scan_cb(msg):
    # convert the message of type LaserScan to a PointCloud2
    pc2_msg = lp.projectLaser(msg)

    lidar_pcd_o3d = open3d_conversions.from_msg(pc2_msg)
    lidar_pcd_o3d_t = lidar_pcd_o3d.transform(T)
    pc2_msg_t = open3d_conversions.to_msg(lidar_pcd_o3d_t, frame_id="base_link", stamp=pc2_msg.header.stamp)
    pc_pub.publish(pc2_msg_t)
    # pc_pub.publish(pc2_msg)
    

rospy.Subscriber("/front/scan", LaserScan, scan_cb, queue_size=1)
rospy.spin()
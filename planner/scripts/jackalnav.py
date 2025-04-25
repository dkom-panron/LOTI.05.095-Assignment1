#!/usr/bin/env python3

import rospy
import numpy as np
import jax
import jax.numpy as jnp

from tf import transformations
from message_filters import ApproximateTimeSynchronizer, Subscriber

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point, Twist, Vector3, Point
from visualization_msgs.msg import Marker, MarkerArray

import open3d as o3d
import open3d_conversions

from scipy.spatial.transform import Rotation as R
from gazebo_model_collision_plugin.msg import Contact
from plan.gradient_descent import gradient_descent
from plan.cem import CEM
from plan.gauss_newton import gauss_newton
from plan.cem_nesterov import cem_nesterov

class JackalNav:
    def __init__(self):
        rospy.init_node("jackal_navigation")
        print("[JACKAL NAVIGATION] Initializing the 'jackal_navigation' Node...")

        #----------------------------------------------------------------------------------

        ## Planner

        argv = rospy.myargv()
        if len(argv) < 4:
            print("USAGE: rosrun planner jackalnav.py [nesterov, cem, gauss_newton, cem_nesterov] [maxiter] [num_controls] [num_samples (for cem)]")
            exit(1)

        maxiter = int(argv[2])
        self.num_controls = int(argv[3])

        planner_type = argv[1]
        if planner_type == "nesterov":
            self.planner = gradient_descent(maxiter, self.num_controls)
        elif planner_type == "cem":
            if len(argv) != 5:
                print("USAGE: rosrun planner jackalnav.py [nesterov, cem, gauss_newton, cem_nesterov] [maxiter] [num_controls] [num_samples (if using cem)]")
                exit(1)

            num_samples = int(argv[4])
            self.planner = CEM(maxiter, self.num_controls, num_samples=num_samples, percentage_elite=0.1, stomp_like=True)
        elif planner_type == "gauss_newton":
            self.planner = gauss_newton(maxiter, self.num_controls)
        elif planner_type == "cem_nesterov":
            if len(argv) != 5:
                print("USAGE: rosrun planner jackalnav.py [nesterov, cem, gauss_newton, cem_nesterov] [maxiter] [num_controls] [num_samples]")
                exit(1)

            num_samples = int(argv[4])
            self.planner = cem_nesterov(maxiter, self.num_controls, cem_num_samples=num_samples, cem_percentage_elite=0.1, cem_stomp_like=True)
        else:
            print("!!ERROR: Invalid planner type. Choose from [nesterov, cem, gauss_newton, cem_nesterov]!!")
            exit(1)

        self.num = 0
        self.controls_init = 0.01*jnp.ones(2*self.num_controls)
        #----------------------------------------------------------------------------------

        # Observation topics
        self.cloud_topic = rospy.get_param('~cloud_topic','/pointcloud')
        self.odom_topic = rospy.get_param('~odom_topic','/ground_truth/odom')

        # Publishers
        self._vel_pub = rospy.Publisher("/mppi/cmd_vel", Twist,queue_size=10)
        
        # RViZ Visualization publishers
        self._linestrip_pub = rospy.Publisher('/trajectory_optimal', Marker, queue_size=10)
        self._arrowmarker_pub = rospy.Publisher('/goal_marker', Marker, queue_size=10)
        self._spheremarker_pub = rospy.Publisher('/goaltf_marker', Marker, queue_size=10)

        #----------------------------------------------------------------------------------

        # Parameters 

        # PointCloud Downsampling
        self.lidar_frame_id = rospy.get_param('~lidar_frame_id', 'base_link')
        self.L_max_lidar = 100
        self.voxel_size = 0.2  


        # class variables for storing observation data
        self.odom = None
        self.goal = None
        self.transformed_goal = None
        self.rotation_mtx = None

        # Odometry
        self.pose = None
        self.linear_vel = None
        self.linear_vel_mag = None
        self.prev_linear_vel_mag = 0.0
        self.angular_vel = None
        self.prev_angular_vel = 0.0
        self.prev_time = None
        self.prev_yaw_e = 0.0
        self.steer = 0.0
        self.steer_dot = 0.0

        # PCD
        self.cloud = None

        #----------------------------------------------------------------------------------

        # Observation subscribers
        odometry_sub = Subscriber(self.odom_topic, Odometry) 
        lidar_pcd_sub = Subscriber(self.cloud_topic, PointCloud2)

        ats = ApproximateTimeSynchronizer([odometry_sub, lidar_pcd_sub], queue_size=10, slop=0.1)
        ats.registerCallback(self.ats_callback)

        #----------------------------------------------------------------------------------

        # Collision subscribers
        collision_sub1 = Subscriber("/jackal/collision_1", Contact) 
        collision_sub2 = Subscriber("/jackal/collision_2", Contact) 
        collision_sub3 = Subscriber("/jackal/collision_3", Contact) 
        collision_sub4 = Subscriber("/jackal/collision_4", Contact) 
        collision_sub5 = Subscriber("/jackal/collision_5", Contact) 

        ats2 = ApproximateTimeSynchronizer([collision_sub1, collision_sub2, collision_sub3, collision_sub4, collision_sub5], queue_size=10, slop=0.1)
        ats2.registerCallback(self.ats2_callback)

        #----------------------------------------------------------------------------------

        # ############# Goal position ###################
        # print("#############[ Goal position ]###################")
        # print ("       ENTER a 2D NAV GOAL in RViz              ")
        # print("#################################################")
        # self.goal = rospy.wait_for_message('move_base_simple/goal', PoseStamped, timeout=30)
        # self.goal.pose.position.z = 2.0
        # #-----------------------------

        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.handle_goal)

        # #-----------------------------
        # ## Hard-Coding the Goal position
        # self.goal = PoseStamped()
        # self.goal.pose.position.x = -1.0
        # self.goal.pose.position.y = 10.0
        # self.goal.pose.position.z = 0.2
        # self.goal.pose.orientation.x = 0.0
        # self.goal.pose.orientation.y = 0.0
        # self.goal.pose.orientation.z = 0.7068252
        # self.goal.pose.orientation.w = 0.7073883
        # self.init_state = False
        # self.goal_arr = np.array([self.goal.pose.position.x, self.goal.pose.position.y, self.goal.pose.position.z])
        # #-----------------------------

        # print("GOAL SET AS :=")
        # print(" x: ", self.goal.pose.position.x)
        # print(" y: ", self.goal.pose.position.y)
        # print(" z: ", self.goal.pose.position.z)
        
        self.heading_aligned = False

        #----------------------------------------------------------------------------------

        # ROS Timer
        self.config_planner_frequency = 15       # Planner called every 0.1 seconds
        self.timer_save = rospy.Timer(rospy.Duration(1. / self.config_planner_frequency),
                                     self._call_planner)

        #----------------------------------------------------------------------------------
        
        print("[JACKAL NAVIGATION] Initialization completed!")

    def _call_planner(self, event):
        if (self.odom is not None) and \
            (self.linear_acc is not None) and \
            (self.angular_acc is not None) and \
            (self.goal is not None) and \
            (self.transformed_goal is not None) and \
            (self.cloud is not None) :

            v_optimal, omega_optimal, traj_optimal = self.planner.compute_controls(
                x_init = 0., y_init= 0., theta_init = 0.,
                v_init = self.linear_vel_mag.item(), omega_init = self.angular_vel.item(), 
                x_goal = self.transformed_goal[0], y_goal = self.transformed_goal[1],
                x_obs = self.cloud[:,0], y_obs = self.cloud[:,1],
                controls_init=self.controls_init
                )

            #print("v_optimal: ", v_optimal)

            self.controls_init = jnp.concatenate((v_optimal,omega_optimal))
            if v_optimal[1] != None and np.linalg.norm(self.pose[:2] - self.goal_arr[:2]) > 0.75:
                self.publish_cmd_vel_msg(v_optimal[1], omega_optimal[1], self.num)

            if np.linalg.norm(self.pose[:2] - self.goal_arr[:2]) < 0.75:
                print("!!REACHED GOAL!!")
                # rospy.signal_shutdown("You reached the goal")

            self._visualize_trajectories(traj_optimal)
            self._visualize_goal()
            self._visualize_goal_tf(self.transformed_goal)
            self.num += 1

    def handle_goal(self, goal_data):
        self.goal = goal_data
        self.goal.pose.position.z = 0.2
        self.goal_arr = np.array([self.goal.pose.position.x, self.goal.pose.position.y, self.goal.pose.position.z])

        print("GOAL SET AS :=")
        print(" x: ", self.goal.pose.position.x)
        print(" y: ", self.goal.pose.position.y)
        print(" z: ", self.goal.pose.position.z)

        dx = self.goal.pose.position.x - self.pose[0]
        dy = self.goal.pose.position.y - self.pose[1]
        target_yaw = np.arctan2(dy, dx)
        self.heading_aligned = False
        self.target_yaw = target_yaw 

    def ats2_callback(self, msg1, msg2, msg3, msg4, msg5):

        if msg1.objects_hit == ['collision'] or \
            msg2.objects_hit == ['collision'] or \
            msg3.objects_hit == ['collision'] or \
            msg4.objects_hit == ['collision'] or \
            msg5.objects_hit == ['collision']:
                
                print("!!CRASHED!!")
                rospy.signal_shutdown("You did not reach the goal")

    def ats_callback(self, odom_data, lidar_pcd_ros):
        
        if lidar_pcd_ros is None or \
            odom_data is None:
            return
        #------------------------------------------

        ## Odometry Processing

        self.odom = odom_data

        quaternion = (
            odom_data.pose.pose.orientation.x,
            odom_data.pose.pose.orientation.y,
            odom_data.pose.pose.orientation.z,
            odom_data.pose.pose.orientation.w
        )
        roll, pitch, yaw = transformations.euler_from_quaternion(quaternion)
        
        self.pose = np.array([
            odom_data.pose.pose.position.x, 
            odom_data.pose.pose.position.y, 
            odom_data.pose.pose.position.z,
            roll,
            pitch,
            yaw])

        linear_vel_odom = np.array([
            odom_data.twist.twist.linear.x,
            odom_data.twist.twist.linear.y,
            odom_data.twist.twist.linear.z
        ])
        
        angular_vel_odom = np.array([
           odom_data.twist.twist.angular.z
        ])

        self.rotation_mtx = R.from_quat(np.array([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w])).as_matrix()
        
        linear_vel = self.rotation_mtx.T @ linear_vel_odom
        self.linear_vel = linear_vel[:2]
        # self.linear_vel = linear_vel_odom   # Jackal odometry velocity already in local frame {for clearpath pkgs}
        self.linear_vel_mag = np.linalg.norm(self.linear_vel)
        self.angular_vel = angular_vel_odom

        if self.goal is not None:    
            self.transformed_goal = self.rotation_mtx.T @ (self.goal_arr - np.array([self.pose[0], self.pose[1], self.pose[2]]))
        
        # computing acceleration
        current_time = rospy.Time.now().to_sec()

        if self.prev_time is not None:
            self.linear_acc = (self.linear_vel_mag - self.prev_linear_vel_mag)/(current_time - self.prev_time + (1e-10))
            self.angular_acc = (self.angular_vel - self.prev_angular_vel)/(current_time - self.prev_time + (1e-10))

        
        self.prev_time = current_time
        self.prev_linear_vel_mag = self.linear_vel_mag
        self.prev_angular_vel = self.angular_vel

        #------------------------------------------

        # ## Hard-Coding the Goal position
        # if self.init_state == False:
        #     dx = self.goal.pose.position.x - self.pose[0]
        #     dy = self.goal.pose.position.y - self.pose[1]
        #     target_yaw = np.arctan2(dy, dx)
        #     self.heading_aligned = False
        #     self.target_yaw = target_yaw 
        #     self.init_state = True
        
        #------------------------------------------

        ## PCD Processing

        _lidar_o3d_pcd = open3d_conversions.from_msg(lidar_pcd_ros)                             # ROS PointCloud2 msgs to Open3D PointClouds

        _lidarpoints = np.asarray(_lidar_o3d_pcd.points)
        _lidarpoints[:,2] = 0

        lidar_o3d_pcd = o3d.geometry.PointCloud()
        lidar_o3d_pcd.points = o3d.utility.Vector3dVector(_lidarpoints)

        lidar_o3d_pcd = lidar_o3d_pcd.voxel_down_sample(voxel_size = self.voxel_size)    # Downsample the point clouds

        if np.asarray(lidar_o3d_pcd.points).shape[0] > self.L_max_lidar:
            # lidar_o3d_pcd = lidar_o3d_pcd.farthest_point_down_sample(self.L_max_lidar) # Downsample the point clouds
        
            # Searching for the nearest points in the pointcloud
            lidar_kdtree = o3d.geometry.KDTreeFlann(lidar_o3d_pcd) # Create KDTree for nearest-neighbour search
            k = self.L_max_lidar
            radius = 10
            query_point = np.array([0., 0., 0.])
            #------------------------------------------
            [lidar_k, lidar_idx, _] = lidar_kdtree.search_knn_vector_3d(query_point, k)
            #------------------------------------------
            # [lidar_k, lidar_idx, _] = lidar_kdtree.search_hybrid_vector_3d(query_point, radius, k)
            #------------------------------------------
            # [lidar_k, lidar_idx, _] = lidar_kdtree.search_radius_vector_3d(query_point, radius)
            #------------------------------------------
            lidar_nearest_points = np.asarray(lidar_o3d_pcd.points)[lidar_idx]     # nearest points' coordinates
            lidar_o3d_pcd.points = o3d.utility.Vector3dVector(lidar_nearest_points)

        if lidar_o3d_pcd.is_empty():
            obs_pt = np.array([self.pose[0] + 1e10, self.pose[1]  + 1e10, self.pose[2]  + 1e10])
            obs_pt = np.reshape(obs_pt, (1,3))
            lidar_o3d_pcd.points.extend(o3d.utility.Vector3dVector(obs_pt))

        lidar_pcd_ds = np.asarray(lidar_o3d_pcd.points)

        if lidar_pcd_ds.shape[0] < self.L_max_lidar:
            lidar_pcd_ds = np.vstack([lidar_pcd_ds, np.tile(lidar_pcd_ds[-1], (self.L_max_lidar - lidar_pcd_ds.shape[0], 1))]) # appending the last point          

        self.cloud = lidar_pcd_ds[: , :2]

    def compute_motion_cmd(self, vel, steer, num):

        if not self.heading_aligned:
            yaw_error = self.target_yaw - self.pose[5]

            if yaw_error > np.pi:
                yaw_error -= 2 * np.pi
            elif yaw_error < -np.pi:
                yaw_error += 2 * np.pi

            angular_z = np.clip(1.0 * yaw_error, -1.0, 1.0)
            cmd = Twist()
            cmd.angular.z = angular_z
            if np.abs(yaw_error) < 0.1:
                self.heading_aligned = True
                self.controls_init = 0.01*jnp.ones(2*self.num_controls)

        else: 
            cmd = Twist()

            #------------------------------------------
            ## velocity clipping
            v_mag_clip = np.clip(vel, -0.1, 2.0)
            # steer = np.clip(steer, -1.57, 1.57)
            omega = steer
            #------------------------------------------

            cmd.linear.x = v_mag_clip
            cmd.angular.z = omega
       
        return cmd

    def publish_cmd_vel_msg(self, vel, steer, num):
        try:
            cmd = self.compute_motion_cmd(vel, steer, num)
            self._vel_pub.publish(cmd)
        except Exception as e:
            print(f"ERROR: {e}")

    def _create_primitive_marker(self, idx, data, namespace, rgb):

        marker = Marker()
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale = Vector3(0.05, 0.01, 0)          # only scale.x used for line strip
        marker.color.r = rgb[0]                        # color 
        marker.color.g = rgb[1]                        
        marker.color.b = rgb[2]                         
        marker.color.a = 1.0                           # alpha - transparency parameter

        marker.ns = namespace
        marker.pose.orientation.w = 1.0

        for i in range(data.shape[0]):
            point = Point()
            point.x = data[i, 0]
            point.y = data[i, 1]
            point.z = 0.0
            marker.points.append(point)

        marker.id = idx
        marker.header.stamp = rospy.get_rostime()
        # marker.lifetime = rospy.Duration(5.0)
        marker.lifetime = rospy.Duration(0.0667)
        marker.header.frame_id = "base_link"

        return marker

    def _visualize_goal(self):

        if self.goal is None:
            return
        
        # scale.x is the arrow length, scale.y is the arrow width and scale.z is the arrow height.
        marker = Marker()
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale = Vector3(0.5, 0.075, 0.075)          
        marker.color.r = 1.0                               
        marker.color.g = 0.25                        
        marker.color.b = 0.25                         
        marker.color.a = 1.0                           

        marker.ns = "goal_marker"

        marker.pose.position.x = self.goal.pose.position.x
        marker.pose.position.y = self.goal.pose.position.y
        marker.pose.position.z = self.goal.pose.position.z
        marker.pose.orientation.x = self.goal.pose.orientation.x
        marker.pose.orientation.y = self.goal.pose.orientation.y
        marker.pose.orientation.z = self.goal.pose.orientation.z
        marker.pose.orientation.w = self.goal.pose.orientation.w

        marker.id = 0
        marker.header.stamp = rospy.get_rostime()
        marker.lifetime = rospy.Duration(0.0)
        marker.header.frame_id = "map"

        self._arrowmarker_pub.publish(marker)

    def _visualize_goal_tf(self, transformed_goal):

        if self.goal is None:
            return
        
        marker = Marker()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale = Vector3(0.1, 0.1, 0.1)              # sphere scaling
        marker.color.r = 0.0                               # color 
        marker.color.g = 0.0                       
        marker.color.b = 0.5                         
        marker.color.a = 1.0                               # alpha - transparency parameter

        marker.ns = "goaltf_marker"

        marker.pose.position.x = transformed_goal[0]
        marker.pose.position.y = transformed_goal[1]
        marker.pose.position.z = transformed_goal[2]

        marker.id = 0
        marker.header.stamp = rospy.get_rostime()
        marker.lifetime = rospy.Duration(0.0)
        marker.header.frame_id = "base_link"

        self._spheremarker_pub.publish(marker)

    def _visualize_trajectories(self, traj_optimal):
        
        trajoptimal_marker = self._create_primitive_marker( 0, traj_optimal, "traj_optimal", [0.0, 1.0, 0.0])
        self._linestrip_pub.publish(trajoptimal_marker)


if __name__ == '__main__':
    quadrotor_navigation = JackalNav()
    rospy.spin()
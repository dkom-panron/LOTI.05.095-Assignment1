# Optimization-for-Robot-Motion-Planning-and-Control

## WiFi

ssid - Robotiklubi24
passwd - ROBOOTIKA

## To connect to Jackal

### with ethernet connection
- ssh -X jackal@192.168.10.50
- passwd - clearpath

### over WiFi
- ssh -X jackal@192.168.1.170 
- passwd - clearpath

## Changes to .bashrc
```
# >>> Clearpath Jackal hardware >>>
export ROS_MASTER_URI=http://jackal:11311
## IP of your laptop as the ROS IP
#export ROS_IP=192.168.10.101 # for ethernet connection
#export ROS_IP=192.168.1.163  # for WiFi connection
# <<< Clearpath Jackal hardware <<<
```
## Changes to /etc/hosts

```
#192.168.10.50  jackal  ## for ethernet connection
#192.168.1.170   jackal ## for WiFi connection
```
## Jackal Hardware commands

```
roslaunch sick_scan sick_tim_5xx.launch
roslaunch realsense2_camera rs_camera.launch
roslaunch depth_to_pcd depth_to_pcd.launch
```

## Laptop commands

```
roslaunch planner jackal_custom.launch
rosrun planner jackalnav.py
```

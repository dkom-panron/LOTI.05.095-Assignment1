# Optimization-for-Robot-Motion-Planning-and-Control
Repository associated with the course Optimization for Robot Motion Planning and Control (LOTI.05.095)

## Installation Instructions:

Note that this code has only been tested with ROS-Noetic on Ubuntu 20.04. We assume that you already have a complete ROS installation.

1. (optional) Set up a catkin workspace

	```
	cd
	mkdir -p catkin_ws/src
	cd catkin_ws
	catkin init
	```

2. Clone this repository in the workspace 

	```
	cd catkin_ws/src
	git clone git@github.com:Arcane-01/Optimization-for-Robot-Motion-Planning-and-Control.git
	```

3. Build the workspace

	```
	cd ../
	catkin build
	source devel/setup.bash
	```
## Usage: 

1. To launch the Gazebo simulation with the Jackal, run the following command:

	```
	roslaunch planner jackal_custom.launch env_name:="jackal_custom"
	```
	* jackal_custom is a sparse environment. More cluttered 300 navigation environments from the [BARN](https://cs.gmu.edu/~xiao/Research/BARN/BARN.html) dataset, named `world_0` to `world_299`, could also be used.

2. To start the planner, run:
	```
	rosrun planner jackalnav.py [nesterov, cem, gauss_newton, cem_nesterov] [maxiter] [num_controls] [num_samples (if using cem)]
	```
	
3. Once the planner is running, you can begin navigation by setting a 2D Nav Goal in RViZ.
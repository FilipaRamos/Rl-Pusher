# Deep-Rl-Pusher
This repository holds the code for the Deep Rl Object Pusher. This code enables a robot to learn how to push cylinder-based objects to a pre-defined target location using q-learning.

The base code is in src/deepPusher.py. The standard algorithm is q-learning, however, I had also finished code for deep q-learning. I did not have time to test it though!

### Packages

OS: Ubuntu 20.04.3

Kernel: 5.13.0-27-generic

Python: 3.8.10 (my environment can be reproduced using requirements.txt)

ROS: Noetic

ROS Packages:
- roscpp
- rospy
- std_msgs
- std_srvs 

Turtlebot3 Packages:
    - turtlebot3_gazebo
        - simulation environment
    - turtlebot3_bringup
        - turtlebot sensor modelling
    - turtlebot3_description
        - turtlebot simulation model (URDF)

### Gazebo Models

*cube-1*
- size 0.2
- mass 0.02

*cube-2*
- size 0.5 0.2 1
- mass 0.3

*cube-3*
- size 0.4
- mass 0.15

*cylinder-1*
- radius .3
- height .35
- mass 0.03

*cylinder-2*
- radius .2
- height .6
- mass 0.05

*cylinder-2-light*
- radius .1
- height .8
- mass 0.005

*cylinder-3*
- radius .08
- height .2
- mass 0.05

### Directory Structure

This package is organised as recommended by ROS documentation. The structure is as follows:

```
deep-rl-pusher
│   README.md
│   package.xml
|   LICENSE
|   CMakeLists.txt
|   .gitignore
|   requirements.txt
|   
└───config
|   │   actions_mixed.config
|   │   actions.config
|   │   lidar.config
|   │   rewards.config
|   |   sim.config
|   │   
|  
└───include
│   │
│   └───deep-rl-pusher
│   
└───launch
|   │   cfg_gazebo.launch
|   │   deepPusher-v0.launch
|   │   
|  
└───models
|   │   cube-1
|   |   |   
|   |   └─── materials
|   |   |   |
|   |   |   └─── scripts
|   |   |   └─── textures
|   |   └─── model.config
|   |   └─── model.sdf
|   |   
|   |   cube-2
|   |   |
|   |   └─── ...
|   |   
|   |   ...
|   │   
└───resources
|   |   clusters.png
|   |   state.png
|   |   ...
│   
└───rviz
|   │   cfg-trajectory.rviz
|   │   
│   
└───src
|   │   envs
|   |   |
|   |   └─── __init__.py
|   |   └─── deepPusherEnv.py
|   |   └─── deepPusherMixedEnv.py
|   |   └─── mainEnv.py
|   |   
|   │   rl
|   |   |
|   |   └─── deep_qlearn.py
|   |   └─── qlearn_stats.py
|   |   └─── qlearn.py
|   |   └─── utils.py
|   |
|   │   __init__.py
|   │   deepPusher.py
|   |   observer.py
|   |   observerUtils.py
|   │   
│   
└───tmp
|   │   ...
|   │   
|
└───worlds
|   │   general.world
|   |   simple.world
|   │   
```

### Execution

In order to run, there is only the need to navigate to the source of the package (deep-rl-pusher/) and run the following command:

> python3 src/deepPusher.py qlearn

For inference, run (tables are saved on the tmp folder):

> python3 src/deepPusher.py qlearn qtable_filename

Take into consideration that Gazebo's client is automatically launched upon execution.

### Author

Filipa M. M. Ramos Ferreira

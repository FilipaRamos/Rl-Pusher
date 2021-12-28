cmake_minimum_required(VERSION 3.0.2)
project(deep-rl-pusher)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  std_srvs
)

catkin_python_setup()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS 
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
cmake_minimum_required(VERSION 3.0)
project(rospytorch)

find_package(catkin REQUIRED COMPONENTS
  roslint
  rospy
  sensor_msgs
  std_msgs
)

roslint_python()

catkin_package(
  CATKIN_DEPENDS
)

# install(DIRECTORY
#   launch
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

install(PROGRAMS
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

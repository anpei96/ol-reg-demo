#
# Project: lidar-camera system calibration based on
#          object-level 3d-2d correspondence
# Author:  anpei
# Data:    2023.03.07
# Email:   anpei@wit.edu.cn
#

'''
exp-3

localization experiment in self-collected dataset

it contains one data-processing oepration:

0) pre-processing of rgb image and lidar point cloud

it contains several sub-experiments:

1) experiment of optimization scheme
     - coarse    method
     - refined   method
     - iteration method
     - global    method
     under the various situations
2) experiment of visulization

registration metric:

1) rotation error    (unit: deg)
2) translation error (unit: cm)
3) mean registration loss (optional)

Q: how to find the ground truth?
A: human fine-tune :)

'''
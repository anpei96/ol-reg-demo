#
# Project: lidar-camera system calibration based on
#          object-level 3d-2d correspondence
# Author:  anpei
# Data:    2023.03.07
# Email:   anpei@wit.edu.cn
#

'''
exp-3

calibration experiment in self-collected dataset

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

import os
import cv2 as cv
import torch
import numpy as np
import open3d as o3d
import pandas as pd
import tqdm
import copy

from utils3dVisual import show_pcd
from pcdutils      import globalregistration
from utilsRecons   import load_calib
from utilsCalib    import make_camera_intrinsic_matrix
from calibSolution import object_3d_2d_match
from calibSolution import object_3d_2d_registration
from calibSolution import eva_err_rmat, eva_err_tvec
from utilsCalib    import projection
from utilsCalib    import ProjectionProcessor

base = "./self-collect-data/"

# HUAWEI P30 Pro camera intrinsic parameters
calib = load_calib(base+"calib_2.txt")
calib.fu = 726.2048#*0.99
calib.fv = 726.2781#*0.99
calib.cu = 456.8631
calib.cv = 340.6442
# calib.cu = 912/2
# calib.cv = 684/2
calib.P[0,0] = calib.fu
calib.P[1,1] = calib.fv
calib.P[0,2] = calib.cu
calib.P[1,2] = calib.cv
# calib.P[:,3] = 0
# print(calib.P)

def csv_2_pcd(csv_file):
     df = pd.read_csv(csv_file)
     x  = df['X'].values.reshape((-1,1))
     y  = df['Y'].values.reshape((-1,1))
     z  = df['Z'].values.reshape((-1,1))

     pts = np.concatenate((x,y,z), axis=1)
     leg = np.linalg.norm(pts, axis=1) 
     leg = leg.reshape((-1))

     valid_idx  = (leg <= 30.0) 
     filter_pts = pts[valid_idx, :]

     pcd = o3d.geometry.PointCloud()
     pcd.points = o3d.utility.Vector3dVector(filter_pts[:, 0:3])
     pcd_down = pcd.voxel_down_sample(voxel_size=0.05)
     return pcd_down

def csv_open_bin_file(scene_path, pcd_list, id):
     csv_path = scene_path + pcd_list[id] + ".csv"
     print("loading ", csv_path)
     pcd = csv_2_pcd(csv_path)
     pts = np.array(pcd.points).astype(np.float32)
     num = pts.shape[0]
     las = np.ones((num,1), dtype=np.float32)*0.2
     pts = np.concatenate((pts,las), axis=1)
     return pts

def jpg_open_png_file(scene_path, img_list, id):
     jpg_path = scene_path + img_list[id] + ".jpg"
     print("loading ", jpg_path)
     img = cv.imread(jpg_path)
     h, w = img.shape[0], img.shape[1]
     img = cv.resize(img, (w//4, h//4))
     return img

def rotation_tune(rr, th):
     thz = th[0] * np.pi/180
     dRz = np.zeros((3,3), dtype=np.float)
     dRz[2,2] = 1
     dRz[0,0] = np.cos(thz)
     dRz[1,1] = np.cos(thz)
     dRz[0,1] = np.sin(thz)
     dRz[1,0] = np.sin(thz) * (-1)
     rr = np.matmul(rr, dRz)

     thx = th[1] * np.pi/180
     dRx = np.zeros((3,3), dtype=np.float)
     dRx[0,0] = 1
     dRx[1,1] = np.cos(thx)
     dRx[2,2] = np.cos(thx)
     dRx[1,2] = np.sin(thx)
     dRx[2,1] = np.sin(thx) * (-1)
     rr = np.matmul(rr, dRx)

     thy = th[2] * np.pi/180
     dRy = np.zeros((3,3), dtype=np.float)
     dRy[1,1] = 1
     dRy[0,0] = np.cos(thy)
     dRy[2,2] = np.cos(thy)
     dRy[0,2] = np.sin(thy)
     dRy[2,0] = np.sin(thy) * (-1)
     rr = np.matmul(rr, dRy)

     return rr

def check_pipeline(scene_path, pcd_list, img_list):
     # ================================ #
     # prepare all inputs
     # ================================ #
     # 1 1 3
     # 2 3 1
     # 3 3 -3
     # 4 5 5
     # pts = csv_open_bin_file(scene_path, pcd_list, id=1)
     # rgb = jpg_open_png_file(scene_path, img_list, id=3)
     # pts = csv_open_bin_file(scene_path, pcd_list, id=3)
     # rgb = jpg_open_png_file(scene_path, img_list, id=1)
     # pts = csv_open_bin_file(scene_path, pcd_list, id=3)
     # rgb = jpg_open_png_file(scene_path, img_list, id=-3)
     pts = csv_open_bin_file(scene_path, pcd_list, id=5)
     rgb = jpg_open_png_file(scene_path, img_list, id=5)
     kk  = make_camera_intrinsic_matrix(calib.fu, calib.fv, calib.cu, calib.cv)
     rr  = np.load(scene_path+"c_r.npy")
     tt  = np.load(scene_path+"c_t.npy")

     '''
     th = [th_z, th_x, th_y]

     th_z view left-right
     th_x view 
     th_y view tilt
     '''
     th = [0.6, 0.75, 1.0+0.8] # 1 1 3
     # th = [-1.2, 0, 0] # 2 3 1
     th = [-5, 0, 1] # 3 3 -3
     th = [-0.6, 0, 0] # 4 5 5 
     rr = rotation_tune(rr, th)

     '''
     dt = [dt_x, dt_y, dt_z]
     dt_x left-right
     dt_y up-down
     dt_z front-back
     '''
     dt = [-0.15, -0.05, 0.56] # 1 1 3
     # dt = [0, +0.5, 0.3] # 2 3 1
     dt = [-0.2, 1.0, 0.15] # 3 3 -3
     dt = [-0.2, -0.3, 0.2] # 4 5 5

     tt[0,0] += dt[0]
     tt[1,0] += dt[1]
     tt[2,0] += dt[2]

     pixels, depths = projection(pts, kk, rr, tt)
     proj_tool = ProjectionProcessor(w=rgb.shape[1], h=rgb.shape[0])
     depth_img = proj_tool.getDepth(depths, pixels)
     depth_img_vis = proj_tool.getDepthVis(depth_img)
     mergeVis_a = cv.addWeighted(rgb, 0.30, depth_img_vis, 0.70, 0)
     cv.imshow("refined calibration result", mergeVis_a)
     cv.waitKey()
     cv.imwrite(scene_path+"merge.png", mergeVis_a)

     np.save(scene_path+"r_r", rr)
     np.save(scene_path+"r_t", tt)

     # add visulization of 3d color point cloud
     is_show_pcd = True
     # is_show_pcd = False
     if is_show_pcd == True:
          p, c = proj_tool.get_color_point(depths, pixels, pts, rgb)
          pc = o3d.geometry.PointCloud()
          pc.points = o3d.utility.Vector3dVector(p[:,:3])
          pc.colors = o3d.utility.Vector3dVector(c[:,:3])
          show_pcd(pc)
          o3d.io.write_point_cloud(scene_path+"color.pcd", pc)
     

if __name__ == '__main__':
     # ================================================== #
     # scene-1 experiments
     # ================================================== #
     scene_path = base + "scene_1/"
     pcd_list   = ['1-a', '1-b', '1-c', '1-d', '1-e', '1-f']
     img_list   = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
     # check_pipeline(scene_path, pcd_list, img_list)

     # ================================================== #
     # scene-2 experiments
     # ================================================== #
     scene_path = base + "scene_2/"
     pcd_list   = ['2-a', '2-b', '2-c', '2-d', '2-e', '2-f', '2-g']
     img_list   = ['1', '2', '3', '4', '5', '6', '7', '8']
     # check_pipeline(scene_path, pcd_list, img_list)

     # ================================================== #
     # scene-3 experiments
     # ================================================== #
     scene_path = base + "scene_3/"
     pcd_list   = ['3-a', '3-b', '3-c', '3-d', '3-e']
     img_list   = ['1', '2', '3', '4', '5', '6', '7']
     # check_pipeline(scene_path, pcd_list, img_list)

     # ================================================== #
     # scene-4 experiments
     # ================================================== #
     scene_path = base + "scene_4/"
     pcd_list   = ['4-a', '4-b', '4-c', '4-d', '4-e', '4-f', '4-g']
     img_list   = ['1', '2', '3', '4', '5', '6', '7', '8']
     check_pipeline(scene_path, pcd_list, img_list)
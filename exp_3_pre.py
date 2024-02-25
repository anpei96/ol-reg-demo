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

from argparse import ArgumentParser
from mmdet3d.apis import inference_detector, init_model, show_result_meshlab
from mmdet3d.apis import inference_mono_3d_detector, init_model, show_result_meshlab

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

'''
note-0420

     gap between scenes is too large, so that
     point cloud registration is failed 
'''
def csv_save_bin_file(scene_path, pcd_list, id):
     csv_path = scene_path + pcd_list[id] + ".csv"
     print("loading ", csv_path)
     pcd = csv_2_pcd(csv_path)
     pts = np.array(pcd.points).astype(np.float32)
     num = pts.shape[0]
     las = np.ones((num,1), dtype=np.float32)*0.2
     pts = np.concatenate((pts,las), axis=1)
     pts.tofile(scene_path+"pcd.bin")

def detect_obj_pcd(scene_path, is_show_detect = True):
     base_mmdet3d  = './mmdetection3d-master/'
     config_3d     = base_mmdet3d + 'configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py'
     checkpoint_3d = base_mmdet3d + 'model_zoos/hv_second_secfpn_6x8_80e_kitti-3d-3class_20210831_022017-ae782e87.pth'
     device = 'cuda:0'
     pcd_path      = scene_path + 'pcd.bin'

     print("")
     print("============")
     model_3d = init_model(config_3d, checkpoint_3d, device=device)
     result_3d, data = inference_detector(model_3d, pcd_path)
     pred_bboxes_3d = result_3d[0]['boxes_3d'].tensor.numpy()
     pred_scores_3d = result_3d[0]['scores_3d'].numpy()
     score_thr_3d = 0.30
     # score_thr_3d = 0
     inds = pred_scores_3d > score_thr_3d
     pred_bboxes_3d = pred_bboxes_3d[inds]

     corners_3d_pts = result_3d[0]['boxes_3d'].corners
     corners_3d_pts = corners_3d_pts[inds].numpy()
     num_bbox = corners_3d_pts.shape[0]
     print("corners_3d_pts.shape: ", corners_3d_pts.shape)
     # print(corners_3d_pts)
     print("result 3d object detection in point cloud")
     print(pred_bboxes_3d.shape)
     # add visulization to check the 3d detection result in point cloud
     # --- check ok
     if is_show_detect == True:
          from utils3dVisual import show_3d_detection_pts
          show_3d_detection_pts(pred_bboxes_3d, pcd_path, calib=None)
     return corners_3d_pts

def jpg_save_png_file(scene_path, img_list, id):
     jpg_path = scene_path + img_list[id] + ".jpg"
     print("loading ", jpg_path)
     img = cv.imread(jpg_path)
     h, w = img.shape[0], img.shape[1]
     img = cv.resize(img, (w//4, h//4))
     # img = cv.medianBlur(img, 3)
     
     # print("img: ", img.shape)
     # cv.imshow("image", img)
     # cv.waitKey()
     cv.imwrite(scene_path+"rgb.png", img)

def detect_obj_rgb(scene_path, calib, is_show_detect = True):
     base_mmdet3d  = './mmdetection3d-master/'
     config_3d     = base_mmdet3d + 'configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py'
     checkpoint_3d = base_mmdet3d + 'model_zoos/hv_second_secfpn_6x8_80e_kitti-3d-3class_20210831_022017-ae782e87.pth'
     device = 'cuda:0'
     
     config_2d     = base_mmdet3d + 'configs/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py'
     checkpoint_2d = base_mmdet3d + 'model_zoos/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_20210929_015553-d46d9bb0.pth'
     model_2d = init_model(config_2d, checkpoint_2d, device=device)
     image_path    = scene_path + 'rgb.png'
     ann_path      = scene_path + 'kitti.json'
     result_2d, data = inference_mono_3d_detector(model_2d, image_path, ann_path)
     pred_bboxes_2d = result_2d[0]['img_bbox']['boxes_3d'].tensor.numpy()
     pred_scores_2d = result_2d[0]['img_bbox']['scores_3d'].numpy()
     score_thr_2d = 0.10
     # score_thr_2d = 0
     inds = pred_scores_2d > score_thr_2d
     pred_bboxes_2d = pred_bboxes_2d[inds]

     corners_3d_img = result_2d[0]['img_bbox']['boxes_3d'].corners
     corners_3d_img = corners_3d_img[inds].numpy()
     num_bbox = corners_3d_img.shape[0]
     print("corners_3d_img.shape: ", corners_3d_img.shape)
     print("result 3d object detection in rgb image")
     print(pred_bboxes_2d.shape)

     from calibSolution import change_order_array
     '''
          note-0407

          bounding box corner points are not aligned
          we use change_order_array to fix this problem
     '''
     corners_3d_img_rec = []
     for o in range(corners_3d_img.shape[0]):
          corners_3d_cam2 = corners_3d_img[o]
          corners_2d_cam2 = calib.project_rect_to_image(corners_3d_cam2)
          x = change_order_array(corners_2d_cam2)
          corners_3d_img_rec.append(x)
          # print(corners_2d_cam2.shape)
          print(x.shape)
     corners_3d_img_rec = np.array(corners_3d_img_rec)

     # add visulization to check the 3d detection result in rgb image
     # --- check ok
     if is_show_detect == True:
          from utils3dVisual import show_3d_detection_img
          rgb = cv.imread(scene_path + 'rgb.png')
          show_3d_detection_img(rgb, corners_3d_img, calib)
     return corners_3d_img_rec

def calibration_pipeline(scene_path, pcd_list, img_list, calib):
     # ================================ #
     # prepare all inputs
     # ================================ #
     # 1 1 3
     # 4 5 5
     # 2 3 1(0)
     # 3 3 -3
     csv_save_bin_file(scene_path, pcd_list, id=5)
     jpg_save_png_file(scene_path, img_list, id=4)
     corners_3d_pts = detect_obj_pcd(scene_path, is_show_detect = True)
     corners_3d_img = detect_obj_rgb(scene_path, calib, is_show_detect = True)
     
     pts = np.fromfile(scene_path + "pcd.bin", dtype=np.float32).reshape(-1, 4)
     rgb = cv.imread(scene_path + "rgb.png")
     kk  = make_camera_intrinsic_matrix(calib.fu, calib.fv, calib.cu, calib.cv)

     # ================================ #
     # 3d-2d object match
     # ================================ #
     solver_match = object_3d_2d_match(pts=pts, rgb=rgb, kmat=kk)

     measured_object_info = {'num': None, 'obj_3d_pts': None, 'obj_2d_pts':None}
     measured_object_info["obj_3d_pts"] = corners_3d_pts.reshape((-1,3))
     measured_object_info["obj_2d_pts"] = corners_3d_img.reshape((-1,2))
    
     solver_match.load_3d_object_from_pts(
          obj_3d_pts=measured_object_info['obj_3d_pts'])
     solver_match.load_3d_object_from_rgb(
          obj_2d_pts=measured_object_info['obj_2d_pts'])
     # 1 0.10
     # 2 0.20
     best_match_res, sort_obj_3d_pts, sort_obj_2d_pts, sort_obj_scores = \
          solver_match.object_match(tunc_val=0.20, is_has_prior=False)
     
     pixels, depths = projection(
          pts, kk, best_match_res['rmat'], best_match_res['tvec'])
     proj_tool = ProjectionProcessor(w=rgb.shape[1], h=rgb.shape[0])
     depth_img = proj_tool.getDepth(depths, pixels)
     depth_img_vis = proj_tool.getDepthVis(depth_img)
     mergeVis_a = cv.addWeighted(rgb, 0.30, depth_img_vis, 0.70, 0)
     cv.imshow("coarse calibration result", mergeVis_a)
     cv.waitKey()

     # ================================ #
     # 3d-2d object registration
     # ================================ #
     # rr  = np.load(scene_path+"r_r.npy")
     # tt  = np.load(scene_path+"r_t.npy")
     # best_match_res['rmat'] = rr
     # best_match_res['tvec'] = tt

     solver_match = object_3d_2d_registration(
        pts=pts, rgb=rgb, kmat=kk)
     solver_match.load_sort_obj_info(
          sort_obj_3d_pts, sort_obj_2d_pts, sort_obj_scores)
     solver_match.load_coarse_calib_res(
          best_match_res['rmat'], best_match_res['tvec'])
     calib_res = solver_match.points_regiestration(
          is_use_pnp=True, is_use_history=False)
     
     # solver_match.load_coarse_calib_res(
     #    calib_res['rmat'], calib_res['tvec'])
     # calib_res = solver_match.points_regiestration(
     #    is_use_pnp=True, is_use_history=False)

     # solver_match.load_coarse_calib_res(
     #    calib_res['rmat'], calib_res['tvec'])
     # calib_res = solver_match.points_regiestration(
     #    is_use_pnp=True, is_use_history=False)

     pixels, depths = projection(
          pts, kk, calib_res['rmat'], calib_res['tvec'])
     proj_tool = ProjectionProcessor(w=rgb.shape[1], h=rgb.shape[0])
     depth_img = proj_tool.getDepth(depths, pixels)
     depth_img_vis = proj_tool.getDepthVis(depth_img)
     mergeVis_a = cv.addWeighted(rgb, 0.30, depth_img_vis, 0.70, 0)
     cv.imshow("refined calibration result", mergeVis_a)
     cv.waitKey()

     print("calib_res['rmat']: ")
     print(calib_res['rmat'])
     print("calib_res['tvec']: ")
     print(calib_res['tvec'])

     # np.save(scene_path+"c_r", calib_res['rmat'])
     # np.save(scene_path+"c_t", calib_res['tvec'])
     np.save(scene_path+"c_r", best_match_res['rmat'])
     np.save(scene_path+"c_t", best_match_res['tvec'])

if __name__ == '__main__':
     # ================================================== #
     # scene-1 experiments
     # ================================================== #
     scene_path = base + "scene_1/"
     pcd_list   = ['1-a', '1-b', '1-c', '1-d', '1-e', '1-f']
     img_list   = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
     # calibration_pipeline(scene_path, pcd_list, img_list, calib)
     
     # ================================================== #
     # scene-2 experiments
     # ================================================== #
     scene_path = base + "scene_2/"
     pcd_list   = ['2-a', '2-b', '2-c', '2-d', '2-e', '2-f', '2-g']
     img_list   = ['1', '2', '3', '4', '5', '6', '7', '8']
     # calibration_pipeline(scene_path, pcd_list, img_list, calib)

     # ================================================== #
     # scene-3 experiments
     # ================================================== #
     scene_path = base + "scene_3/"
     pcd_list   = ['3-a', '3-b', '3-c', '3-d', '3-e']
     img_list   = ['1', '2', '3', '4', '5', '6', '7']
     # calibration_pipeline(scene_path, pcd_list, img_list, calib)

     # ================================================== #
     # scene-4 experiments
     # ================================================== #
     scene_path = base + "scene_4/"
     pcd_list   = ['4-a', '4-b', '4-c', '4-d', '4-e', '4-f', '4-g']
     img_list   = ['1', '2', '3', '4', '5', '6', '7', '8']
     calibration_pipeline(scene_path, pcd_list, img_list, calib)
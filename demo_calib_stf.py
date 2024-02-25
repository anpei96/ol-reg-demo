#
# Project: lidar-camera system calibration based on
#          object-level 3d-2d correspondence
# Author:  anpei
# Data:    2023.03.07
# Email:   anpei@wit.edu.cn
#

import os
import cv2 as cv
import torch
import numpy as np
import open3d as o3d

from utilsKitti    import Calibration
from utilsVisual   import read_detection, plot_3dbox
from utilsVisual   import plot_3dbox_image
from utils3dVisual import show3DdetectionResults

# in stf dataset, we use both gt and pd 3d boxes for simulation

if __name__ == '__main__':
    # ================================================== #
    # Step one: prepare raw dataset and ground truth
    # ================================================== #
    # set save path
    save_path = "/media/anpei/DiskA/multi_calib_lidar_cam/"

    # basic information
    base_path    = "/media/anpei/DiskA/weather-transfer-anpei/"
    temp_path    = base_path + "anpei_visual_detection/"
    dataset_path = base_path + "data/seeingthroughfog/training/"
    pred_gt_path = dataset_path + "label_2/"
    pred_pd_path = dataset_path + "label_2/"
    lidar_path   = dataset_path + "velodyne/"
    image_path   = dataset_path + "image_2/"
    calib_path   = dataset_path + "calib/kitti_stereo_velodynehdl_calib.txt"

    # make sure that weather split and pd path are consistent
    pred_pd_path = temp_path + "sp-det/clear/data/"
    # pred_pd_path = temp_path + "sp-det/light-fog/data/"
    # pred_pd_path = temp_path + "sp-det/dense-fog/data/"
    # pred_pd_path = temp_path + "sp-det/snow/data/"

    weather_split = base_path + "data/seeingthroughfog/ImageSets/val_clear.txt"
    # weather_split = base_path + "data/seeingthroughfog/ImageSets/val_dense_fog.txt"
    # weather_split = base_path + "data/seeingthroughfog/ImageSets/val_light_fog.txt"
    # weather_split  = base_path + "data/seeingthroughfog/ImageSets/val_snow.txt"

    idx = 111+2+10+55

    file_name_gt_list = os.listdir(pred_gt_path)
    file_name_pd_list = os.listdir(pred_pd_path)
    gt_label_name = file_name_gt_list[idx] 
    pd_label_name = file_name_pd_list[idx]

    split_index   = []
    f=open(file=weather_split)
    for line in f.readlines():
        line = line.strip('\n')
        stra = line[:-6]
        strb = line[-5:]
        line = stra + '_' + strb
        split_index.append(line)
    gt_label_name = split_index[idx] + ".txt"
    pd_label_name = split_index[idx] + ".txt"

    label_gt_path = pred_gt_path + gt_label_name
    label_pd_path = pred_pd_path + pd_label_name
    lidar_da_path = lidar_path   + gt_label_name[:-4] + ".bin"
    image_da_path = image_path   + gt_label_name[:-4] + ".png"

    print("label_gt_path: ", label_gt_path)
    print("label_pd_path: ", label_pd_path)

    # prepare inputs
    pts, rgb, obj_3d_pts, obj_2d_pts = \
        show3DdetectionResults(
            lidar_da_path, 
            image_da_path,
            label_gt_path, 
            label_pd_path, 
            calib_path, 
            save_path, 
            is_has_gt=True, 
            is_black_bg=True, 
            is_save=False,
            is_save_img=True)
    
    # pipeline of calibration method
    # prepare ground truth
    from utilsCalib import make_camera_intrinsic_matrix
    from utilsCalib import projection
    from utilsCalib import ProjectionProcessor
    calib = Calibration(calib_path)
    kk  = make_camera_intrinsic_matrix(
         calib.f_u, calib.f_v, calib.c_u, calib.c_v)
    rr  = calib.C2V[:3,:3].T
    tt  = calib.C2V[:3,3:4]
    tt  = -np.matmul(rr,tt)
    pixels, depths = projection(pts, kk, rr, tt)

    dist_coeffs = np.zeros((4, 1)) 
    obj_3d_pts  = obj_3d_pts.reshape((-1,3))
    obj_2d_pts, _ = projection(obj_3d_pts, kk, rr, tt)
    success, rvec, tvec = cv.solvePnP(
        obj_3d_pts, obj_2d_pts, kk, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
    rmat = cv.Rodrigues(rvec)[0]
    pixels, depths = projection(pts, kk, rmat, tvec)

    proj_tool = ProjectionProcessor(w=rgb.shape[1], h=rgb.shape[0])
    depth_img = proj_tool.getDepth(depths, pixels)
    depth_img_vis = proj_tool.getDepthVis(depth_img)
    mergeVis = cv.addWeighted(rgb, 0.25, depth_img_vis, 0.75, 0)

    # cv.imshow("depth_img_vis", depth_img_vis)
    # cv.imshow("mergeVis", mergeVis)
    # cv.waitKey()
    # cv.imwrite("draw_img.png", mergeVis)

    # ================================================== #
    # Step two: simulation the measure noise
    # (include 3d object detection error, the random sort)
    # needs visulization
    # ================================================== #
    # todo

    # ================================================== #
    # Step three: calibration method
    # ================================================== #
    # todo
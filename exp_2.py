#
# Project: lidar-camera system calibration based on
#          object-level 3d-2d correspondence
# Author:  anpei
# Data:    2023.03.07
# Email:   anpei@wit.edu.cn
#

'''
exp-2

localization experiment in kitti odometry dataset

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

'''

from email.mime import image
import os
import cv2 as cv
import torch
import numpy as np
import open3d as o3d
import tqdm
import copy

from utils3dVisual import show_pcd
from utilsKitti    import Calibration
from utils3dVisual import show3DdetectionResults
from utilsCalib    import add_object_measure_noise
from utilsCalib    import add_object_measure_noise_sel_num
from calibSolution import object_3d_2d_match
from calibSolution import object_3d_2d_registration
from calibSolution import eva_err_rmat, eva_err_tvec

from utilsRecons   import load_pose, load_calib, point_cloud_merge_target
from utilsRecons   import generate_seg_color
from utilsRecons   import load_fuse_data_slide_window, load_fuse_data_slide_window_target

au_dataset_basedir = "/media/anpei/DiskA/auxiliary-transfer-anpei/auxiliary_data/"
au_id = "sequences/05/" # 04 05 07 15
au_pid = "05/"
au_id = "sequences/07/" # 04 05 07 15
au_pid = "07/"

def odometry_data_reconstruct(imageIdx, save_path):
    """
        merge point cloud and save it

        do not use predicted semantic information
    """
    curr_data_path = au_dataset_basedir + au_id
    pred_data_path = "/media/anpei/DiskA/auxiliary-transfer-anpei/auxiliary_data/rangenet_pred/"
    pred_data_path = pred_data_path + au_pid 

    pose_info = load_pose(curr_data_path +"poses.txt")
    calib     = load_calib(curr_data_path+"calib_2.txt")
    
    halfwin = 10
    halfwin = 3
    halfwin = 1
    # halfwin = 0
    pts_array_full, pcd_array_full, rgb_array_full, pos_array_full, seg_array_full, seg_pred_array_full = load_fuse_data_slide_window(
        curr_data_path, pred_data_path, imageIdx, pose_info, 
        halfwin=halfwin, stride=1, is_need_seg=True) # 3 20
    pts_array, pcd_array, rgb_array, pos_array, seg_array, seg_array_pred = load_fuse_data_slide_window_target(
        curr_data_path, pred_data_path, imageIdx, pose_info, 
        halfwin=halfwin, stride=1, is_need_seg=True) # 3 20

    merge_seg = np.concatenate(seg_array, axis=0)
    merge_seg_full = np.concatenate(seg_array_full, axis=0)
    merge_seg_pred = np.concatenate(seg_array_pred, axis=0)
    merge_seg_pred_full = np.concatenate(seg_pred_array_full, axis=0)

    merge_pts, merge_pts_full = point_cloud_merge_target(pts_array, pos_array, 
        calib, pts_array_full,
        is_need_prior=True, is_need_reg=True) # False
    
    merge_pcd = o3d.geometry.PointCloud()
    merge_pcd.points = o3d.utility.Vector3dVector(merge_pts[:, 0:3])
    merge_pcd_full = o3d.geometry.PointCloud()
    merge_pcd_full.points = o3d.utility.Vector3dVector(merge_pts_full[:, 0:3])

    pred_merge_pcd = o3d.geometry.PointCloud()
    pred_merge_pcd.points = o3d.utility.Vector3dVector(merge_pts[:, 0:3])
    pred_merge_pcd_full = o3d.geometry.PointCloud()
    pred_merge_pcd_full.points = o3d.utility.Vector3dVector(merge_pts_full[:, 0:3])

    '''
        note-0409

        down-sample the reconstruct point cloud
        to reduce the computation burden in 2d-3d registration
    '''
    pred_merge_pcd_full = pred_merge_pcd_full.voxel_down_sample(
        voxel_size=0.05)
    # pred_merge_pcd_full = pred_merge_pcd_full.voxel_down_sample(
    #     voxel_size=0.10)

    is_need_seg_vis = True
    # is_need_save    = False

    if is_need_seg_vis == True:
        merge_rgb      = generate_seg_color(merge_pts, merge_seg)
        merge_rgb_full = generate_seg_color(merge_pts_full, merge_seg_full)
        pred_merge_rgb      = generate_seg_color(merge_pts, merge_seg_pred)
        pred_merge_rgb_full = generate_seg_color(merge_pts_full, merge_seg_pred_full)
        
        merge_pcd.colors = o3d.utility.Vector3dVector(merge_rgb)
        merge_pcd_full.colors = o3d.utility.Vector3dVector(merge_rgb_full)
        pred_merge_pcd.colors = o3d.utility.Vector3dVector(pred_merge_rgb)
        pred_merge_pcd_full.colors = o3d.utility.Vector3dVector(pred_merge_rgb_full)

        # save
        rgb_array_full_np = np.array(rgb_array_full)
        np.save(save_path + "rgb.npy", rgb_array_full_np)
        o3d.io.write_point_cloud(save_path + "merge.pcd", merge_pcd)
        o3d.io.write_point_cloud(save_path + "merge_full.pcd", merge_pcd_full)
        # o3d.io.write_point_cloud(save_path + "pred_merge.pcd", pred_merge_pcd)
        # o3d.io.write_point_cloud(save_path + "pred_merge_full.pcd", pred_merge_pcd_full)
        
        # also save pts as bin file
        (merge_pts_full[:, 0:4]).tofile(save_path + "merge_full.bin")
        cv.imwrite(save_path + "rgb.png", rgb_array_full[0])

        cv.imshow("rgb image", rgb_array_full[0])
        cv.waitKey()

    # debug 
    is_allow_debug = True
    if is_allow_debug:
        # show_pcd(merge_pcd)
        show_pcd(merge_pcd_full)
        # show_pcd(pred_merge_pcd)
        # show_pcd(pred_merge_pcd_full)

def load_odometry_data_reconstruct():
    rgb_array_full = np.load(save_path + "rgb.npy")
    merge_pcd      = o3d.io.read_point_cloud(save_path + "merge.pcd")
    merge_pcd_full = o3d.io.read_point_cloud(save_path + "merge_full.pcd")

    curr_data_path = au_dataset_basedir + au_id
    calib = load_calib(curr_data_path+"calib_2.txt")

    merge_bin_full = np.fromfile(save_path + 'merge_full.bin', dtype=np.float32).reshape(-1,4)
    return rgb_array_full, merge_pcd, merge_pcd_full, calib, merge_bin_full

from utilsCalib import make_camera_intrinsic_matrix
from utilsCalib import projection
from utilsCalib import ProjectionProcessor

def pre_proceess(image_idx, is_need_precomp=True):
    # is_need_precomp = True
    # is_need_precomp = False
    if is_need_precomp == True:
        odometry_data_reconstruct(image_idx, save_path)
        rgb_array_full, merge_pcd, merge_pcd_full, calib, merge_bin_full = \
            load_odometry_data_reconstruct()
    else:
        rgb_array_full, merge_pcd, merge_pcd_full, calib, merge_bin_full = \
            load_odometry_data_reconstruct()
    
    pts = np.array(merge_pcd_full.points)
    rgb = rgb_array_full[0]

    calib.fu = 7.070912000000e+02
    calib.fv = 7.070912000000e+02
    calib.cu = 6.018873000000e+02
    calib.cv = 1.831104000000e+02

    kk  = make_camera_intrinsic_matrix(
         calib.fu, calib.fv, calib.cu, calib.cv)
    rr  = calib.C2V[:3,:3].T
    tt  = calib.C2V[:3,3:4]
    tt  = -np.matmul(rr,tt)

    # note-0331
    # need to fine-tune the calibration results
    is_need_hand_tune_gt = True
    is_need_hand_tune_gt = False
    if is_need_hand_tune_gt:
        thz = -0.3 * np.pi/180
        dRz = np.zeros((3,3), dtype=np.float)
        dRz[2,2] = 1
        dRz[0,0] = np.cos(thz)
        dRz[1,1] = np.cos(thz)
        dRz[0,1] = np.sin(thz)
        dRz[1,0] = np.sin(thz) * (-1)

        rr = np.matmul(rr, dRz)
        tt[0,0] += 0.00
        # tt[1,0] += -0.32
        tt[1,0] += -0.25
        tt[2,0] += -0.16

    calib_gt_info    = {'rr': rr, 'tt': tt}
    return pts, rgb, kk, calib_gt_info, calib

from argparse import ArgumentParser
from mmdet3d.apis import inference_detector, init_model, show_result_meshlab
from mmdet3d.apis import inference_mono_3d_detector, init_model, show_result_meshlab

def pre_detect_3d(calib, rgb, is_show_detect=True):
    base_mmdet3d  = './mmdetection3d-master/'
    config_3d     = base_mmdet3d + 'configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py'
    checkpoint_3d = base_mmdet3d + 'model_zoos/hv_second_secfpn_6x8_80e_kitti-3d-3class_20210831_022017-ae782e87.pth'
    device = 'cuda:0'
    pcd_path      = save_path + 'merge_full.bin'

    is_need_inference = True
    # is_need_inference = False
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
    print(corners_3d_pts)
    print("result 3d object detection in point cloud")
    print(pred_bboxes_3d.shape)
    # add visulization to check the 3d detection result in point cloud
    # --- check ok
    if is_show_detect == True:
        from utils3dVisual import show_3d_detection_pts
        show_3d_detection_pts(pred_bboxes_3d, pcd_path, calib)

    config_2d     = base_mmdet3d + 'configs/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py'
    checkpoint_2d = base_mmdet3d + 'model_zoos/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_20210929_015553-d46d9bb0.pth'
    model_2d = init_model(config_2d, checkpoint_2d, device=device)
    image_path    = save_path + 'rgb.png'
    ann_path      = save_path + 'kitti.json'
    result_2d, data = inference_mono_3d_detector(model_2d, image_path, ann_path)
    pred_bboxes_2d = result_2d[0]['img_bbox']['boxes_3d'].tensor.numpy()
    pred_scores_2d = result_2d[0]['img_bbox']['scores_3d'].numpy()
    score_thr_2d = 0.20
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
        show_3d_detection_img(rgb, corners_3d_img, calib)

    return corners_3d_pts, corners_3d_img_rec

if __name__ == '__main__':
    # ================================================== #
    # Step one: pre-process
    # ================================================== #
    save_path = "/media/anpei/DiskA/multi_calib_lidar_cam/dense/"

    image_idx = 97
    image_idx = 555+10+25+5
    # image_idx = 666+10+30+100 # fuse
    # image_idx = 200
    # image_idx = 200+50
    # image_idx = 300+80
    # image_idx = 100+80
    # image_idx = 600+10
    # image_idx = 50
    

    is_need_comp_vis = True
    # is_need_comp_vis = False

    if is_need_comp_vis == True:
        pts, rgb, kk, calib_gt_info, calib = pre_proceess \
            (image_idx, is_need_precomp=True)
        corners_3d_pts, corners_3d_img_rec = \
            pre_detect_3d(calib, rgb, is_show_detect=True)
    else:
        pts, rgb, kk, calib_gt_info, calib = pre_proceess \
            (image_idx, is_need_precomp=False)
        corners_3d_pts, corners_3d_img_rec = \
            pre_detect_3d(calib, rgb, is_show_detect=False)

    # ================================================== #
    # Step two: registration method
    # ================================================== #  
    # 3d-2d object matching from rgb and point cloud
    solver_match = object_3d_2d_match(pts=pts, rgb=rgb, kmat=kk)
    solver_match_his = object_3d_2d_match(pts=pts, rgb=rgb, kmat=kk)
    
    # prepare measured_object_info
    measured_object_info = {'num': None, 'obj_3d_pts': None, 'obj_2d_pts':None}
    measured_object_info["obj_3d_pts"] = corners_3d_pts.reshape((-1,3))
    measured_object_info["obj_2d_pts"] = corners_3d_img_rec.reshape((-1,2))
    
    '''
    note-0419 
    
        one trick: use the histiry object info to improve the 
                   stability of 3d-2d registration
    '''
    import copy
    is_use_history_info = True
    # is_use_history_info = False
    if is_use_history_info:
        h_obj_3d_pts = np.load("history_obj_3d_pts.npy")
        h_obj_2d_pts = np.load("history_obj_2d_pts.npy")
        a = corners_3d_pts.reshape((-1,3))
        b = corners_3d_img_rec.reshape((-1,2))
        measured_object_info_mix = copy.deepcopy(measured_object_info)
        measured_object_info_mix["obj_3d_pts"] = \
            np.concatenate((h_obj_3d_pts,h_obj_3d_pts,a), axis=0)
        measured_object_info_mix["obj_2d_pts"] = \
            np.concatenate((h_obj_2d_pts,h_obj_2d_pts,b), axis=0)
        solver_match_his.load_3d_object_from_pts(
            obj_3d_pts=measured_object_info_mix['obj_3d_pts'])
        solver_match_his.load_3d_object_from_rgb(
            obj_2d_pts=measured_object_info_mix['obj_2d_pts'])
        _, sort_obj_3d_pts, sort_obj_2d_pts, sort_obj_scores = \
            solver_match_his.object_match(is_has_prior=True)

    solver_match.load_3d_object_from_pts(
        obj_3d_pts=measured_object_info['obj_3d_pts'])
    solver_match.load_3d_object_from_rgb(
        obj_2d_pts=measured_object_info['obj_2d_pts'])
    best_match_res, sort_obj_3d_pts, sort_obj_2d_pts, sort_obj_scores = \
        solver_match.object_match(is_has_prior=True)
    
    err_rmat = eva_err_rmat(best_match_res['rmat'], calib_gt_info['rr'])
    err_tvec = eva_err_tvec(best_match_res['tvec'], calib_gt_info['tt'])
    print("=> err_rmat (coarse): ", err_rmat)
    print("=> err_tvec (coarse): ", err_tvec)

    '''
    note-0419 
    
        one trick: use the histiry object info to improve the 
                   stability of 3d-2d registration

        revise in solver_match.points_regiestration
    '''

    # 3d-2d registration from the matched objects
    solver_match = object_3d_2d_registration(
        pts=pts, rgb=rgb, kmat=kk)
    solver_match.load_sort_obj_info(
        sort_obj_3d_pts, sort_obj_2d_pts, sort_obj_scores)
    solver_match.load_coarse_calib_res(
        best_match_res['rmat'], best_match_res['tvec'])
    # solver_match.load_coarse_calib_res(
    #     calib_gt_info['rr'], calib_gt_info['tt'])
    calib_res = solver_match.points_regiestration(
        is_use_pnp=True, is_use_history=True)
    err_rmat = eva_err_rmat(calib_res['rmat'], calib_gt_info['rr'])
    err_tvec = eva_err_tvec(calib_res['tvec'], calib_gt_info['tt'])
    print("=> err_rmat (iter-1): ", err_rmat)
    print("=> err_tvec (iter-1): ", err_tvec)

    solver_match.load_coarse_calib_res(
        calib_res['rmat'], calib_res['tvec'])
    calib_res = solver_match.points_regiestration(
        is_use_pnp=True, is_use_history=True)
    err_rmat = eva_err_rmat(calib_res['rmat'], calib_gt_info['rr'])
    err_tvec = eva_err_tvec(calib_res['tvec'], calib_gt_info['tt'])
    print("=> err_rmat (iter-2): ", err_rmat)
    print("=> err_tvec (iter-2): ", err_tvec)

    solver_match.load_coarse_calib_res(
        calib_res['rmat'], calib_res['tvec'])
    calib_res = solver_match.points_regiestration(
        is_use_pnp=True, is_use_history=True)

    # w = 0.25
    # calib_res['rmat'] = best_match_res['rmat']
    # calib_res['tvec'] = calib_res['tvec']*w + best_match_res['tvec']*(1-w)

    err_rmat = eva_err_rmat(calib_res['rmat'], calib_gt_info['rr'])
    err_tvec = eva_err_tvec(calib_res['tvec'], calib_gt_info['tt'])
    print("=> err_rmat (iter-3): ", err_rmat)
    print("=> err_tvec (iter-3): ", err_tvec)

    # 3d-2d registration from the global scene
    # solver_match.load_coarse_calib_res(
    #     calib_res['rmat'], calib_res['tvec'])
    # calib_res = solver_match.points_regiestration_global()
    # err_rmat = eva_err_rmat(calib_res['rmat'], calib_gt_info['rr'])
    # err_tvec = eva_err_tvec(calib_res['tvec'], calib_gt_info['tt'])
    # print("=> err_rmat (global-1): ", err_rmat)
    # print("=> err_tvec (global-1): ", err_tvec)

    # solver_match.load_coarse_calib_res(
    #     calib_res['rmat'], calib_res['tvec'])
    # calib_res = solver_match.points_regiestration_global()
    # err_rmat = eva_err_rmat(calib_res['rmat'], calib_gt_info['rr'])
    # err_tvec = eva_err_tvec(calib_res['tvec'], calib_gt_info['tt'])
    # print("=> err_rmat (global-2): ", err_rmat)
    # print("=> err_tvec (global-2): ", err_tvec)

    # solver_match.load_coarse_calib_res(
    #     calib_res['rmat'], calib_res['tvec'])
    # calib_res = solver_match.points_regiestration_global()
    # err_rmat = eva_err_rmat(calib_res['rmat'], calib_gt_info['rr'])
    # err_tvec = eva_err_tvec(calib_res['tvec'], calib_gt_info['tt'])
    # print("=> err_rmat (global-3): ", err_rmat)
    # print("=> err_tvec (global-3): ", err_tvec)

    # ================================================== #
    # Step three: registration visulization
    # ================================================== # 
    is_need_vis = True
    # is_need_vis = False

    if is_need_vis == True:
        pixels, depths = projection(
            pts, kk, calib_res['rmat'], calib_res['tvec'])
        # pixels, depths = projection(
        #     pts, kk, best_match_res['rmat'], best_match_res['tvec'])
        proj_tool = ProjectionProcessor(w=rgb.shape[1], h=rgb.shape[0])
        depth_img = proj_tool.getDepth(depths, pixels)
        depth_img_vis = proj_tool.getDepthVis(depth_img)
        mergeVis_a = cv.addWeighted(rgb, 0.30, depth_img_vis, 0.70, 0)
        cv.imshow("refined calibration result", mergeVis_a)
        cv.waitKey()
        cv.imwrite("b.png", mergeVis_a)

        pixelsa, depthsa = projection(
            pts, kk, best_match_res['rmat'], best_match_res['tvec'])
        proj_tool = ProjectionProcessor(w=rgb.shape[1], h=rgb.shape[0])
        depth_img = proj_tool.getDepth(depthsa, pixelsa)
        depth_img_vis = proj_tool.getDepthVis(depth_img)
        mergeVis_a = cv.addWeighted(rgb, 0.30, depth_img_vis, 0.70, 0)
        cv.imshow("coarse calibration result", mergeVis_a)
        cv.waitKey()
        cv.imwrite("a.png", mergeVis_a)

        # add visulization of 3d color point cloud
        # p, c = proj_tool.get_color_point(depths, pixels, pts, rgb)
        # pc = o3d.geometry.PointCloud()
        # pc.points = o3d.utility.Vector3dVector(p[:,:3])
        # pc.colors = o3d.utility.Vector3dVector(c[:,:3])
        # show_pcd(pc)

    
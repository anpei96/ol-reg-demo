#
# Project: lidar-camera system calibration based on
#          object-level 3d-2d correspondence
# Author:  anpei
# Data:    2023.03.07
# Email:   anpei@wit.edu.cn
#

'''
exp-1

calibration experiment in kitti object dataset

it contains several sub-experiments:

1) experiment of bounding box noise 
2) experiment of point cloud noise            (box noise is fixed)
3) experiment of 2d-3d match and registration (all noise is fixed)
4) experiment of optimization scheme          (fuse or not)
5) experiment of 2d-3d matched object number  (all noise is fixed) 
6) experiment of iteration 

registration metric:

1) rotation error    (unit: deg)
2) translation error (unit: cm)
3) mean registration loss (optional)

'''

import os
import cv2 as cv
import torch
import numpy as np
import open3d as o3d
import tqdm
import copy

from utilsKitti    import Calibration
from utils3dVisual import show3DdetectionResults
from utilsCalib    import add_object_measure_noise
from utilsCalib    import add_object_measure_noise_sel_num
from calibSolution import object_3d_2d_match
from calibSolution import object_3d_2d_registration
from calibSolution import eva_err_rmat, eva_err_tvec

def exp_1d(input_basic_info, calib_gt_info, object_info):
    '''
    5) experiment of 2d-3d matched object number  (all noise is fixed) 
    '''
    num_trail = 250
    # num_trail = 10
    num_trail = 50
    iteration_time = 3

    err_mat_rr_all = np.zeros((4, num_trail, 3), dtype=np.float32) 
    err_mat_tt_all = np.zeros((4, num_trail, 3), dtype=np.float32)

    solver_match = object_3d_2d_match(
        pts=input_basic_info['pts'],
        rgb=input_basic_info['rgb'],
        kmat=input_basic_info['kk'])
    solver_reg   = object_3d_2d_registration(
        pts=input_basic_info['pts'],
        rgb=input_basic_info['rgb'],
        kmat=input_basic_info['kk'])
    
    noise_level = 0.100 # 0.050 0.100 0.150
    for obj_num in range(4):
        '''
        in each iteration, we conduct 100 random experiments
        '''
        for k in tqdm.trange(num_trail):
            object_info_this = copy.deepcopy(object_info)
            measured_object_info = add_object_measure_noise_sel_num(
                object_info_this, sigma=noise_level, select_num=obj_num+1)

            # proposed calibration pipeline
            # 2d-3d object match
            solver_match.load_3d_object_from_pts(
                obj_3d_pts=measured_object_info['mea_obj_3d_pts'])
            solver_match.load_3d_object_from_rgb(
                obj_2d_pts=measured_object_info['mea_obj_2d_pts'])
            best_match_res, sort_obj_3d_pts, sort_obj_2d_pts, sort_obj_scores = \
                solver_match.object_match(is_debug=False)
            
            # 2d-3d object registration
            solver_reg.load_sort_obj_info(
                sort_obj_3d_pts, sort_obj_2d_pts, sort_obj_scores)
            solver_reg.load_coarse_calib_res(
                best_match_res['rmat'], best_match_res['tvec'])
            calib_res = solver_reg.points_regiestration(is_use_pnp=True)

            # iteration scheme
            for iter in range(iteration_time):
                solver_reg.load_coarse_calib_res(
                    calib_res['rmat'], calib_res['tvec'])
                calib_res = solver_reg.points_regiestration(is_use_pnp=True)
        
            err_rmat = eva_err_rmat(calib_res['rmat'], calib_gt_info['rr'])
            err_tvec = eva_err_tvec(calib_res['tvec'], calib_gt_info['tt'])
            err_mat_rr_all[obj_num,k] = np.abs(err_rmat)
            err_mat_tt_all[obj_num,k] = np.abs(err_tvec)

        print("=> err_rmat (coarse): ", np.mean(err_mat_rr_all[obj_num], axis=0))
        print("=> err_tvec (coarse): ", np.mean(err_mat_tt_all[obj_num], axis=0))

    np.save('./exp_data/exp_1d_'+'r_c_250.npy', err_mat_rr_all)
    np.save('./exp_data/exp_1d_'+'t_c_250.npy', err_mat_tt_all)

def exp_1c(input_basic_info, calib_gt_info, object_info):
    '''
    6) experiment of iteration                    (iteration time)
    4) experiment of optimization scheme          (fuse or not)
                                                  (box noise is fixed)
    '''
    num_trail = 250
    num_trail = 10
    # num_trail = 50
    err_mat_rr_all = np.zeros((5, num_trail, 3), dtype=np.float32) 
    err_mat_tt_all = np.zeros((5, num_trail, 3), dtype=np.float32)

    solver_match = object_3d_2d_match(
        pts=input_basic_info['pts'],
        rgb=input_basic_info['rgb'],
        kmat=input_basic_info['kk'])
    solver_reg   = object_3d_2d_registration(
        pts=input_basic_info['pts'],
        rgb=input_basic_info['rgb'],
        kmat=input_basic_info['kk'])
    
    noise_level = 0.100 # 0.050 0.100 0.150
    for iteration_time in range(5):
        '''
        in each iteration, we conduct 100 random experiments
        '''
        for k in tqdm.trange(num_trail):
            measured_object_info = add_object_measure_noise(
                object_info, sigma=noise_level)

            # proposed calibration pipeline
            # 2d-3d object match
            solver_match.load_3d_object_from_pts(
                obj_3d_pts=measured_object_info['mea_obj_3d_pts'])
            solver_match.load_3d_object_from_rgb(
                obj_2d_pts=measured_object_info['mea_obj_2d_pts'])
            best_match_res, sort_obj_3d_pts, sort_obj_2d_pts, sort_obj_scores = \
                solver_match.object_match(is_debug=False)
            
            # 2d-3d object registration
            solver_reg.load_sort_obj_info(
                sort_obj_3d_pts, sort_obj_2d_pts, sort_obj_scores)
            solver_reg.load_coarse_calib_res(
                best_match_res['rmat'], best_match_res['tvec'])
            calib_res = solver_reg.points_regiestration(is_use_pnp=True)

            # iteration scheme
            for iter in range(iteration_time):
                solver_reg.load_coarse_calib_res(
                    calib_res['rmat'], calib_res['tvec'])
                calib_res = solver_reg.points_regiestration(is_use_pnp=True)
        
            err_rmat = eva_err_rmat(calib_res['rmat'], calib_gt_info['rr'])
            err_tvec = eva_err_tvec(calib_res['tvec'], calib_gt_info['tt'])
            err_mat_rr_all[iteration_time,k] = np.abs(err_rmat)
            err_mat_tt_all[iteration_time,k] = np.abs(err_tvec)

        print("=> err_rmat (coarse): ", np.mean(err_mat_rr_all[iteration_time], axis=0))
        print("=> err_tvec (coarse): ", np.mean(err_mat_tt_all[iteration_time], axis=0))

    np.save('./exp_data/exp_1c_'+'r_c_250.npy', err_mat_rr_all)
    np.save('./exp_data/exp_1c_'+'t_c_250.npy', err_mat_tt_all)

def exp_1b(input_basic_info, calib_gt_info, object_info):
    '''
    2) experiment of point cloud noise            (box noise is fixed)
    '''
    num_trail = 250
    iteration_time = 3
    err_mat_rr_all = np.zeros((9, num_trail, 3), dtype=np.float32) 
    err_mat_tt_all = np.zeros((9, num_trail, 3), dtype=np.float32)
    err_mat_pp_all = np.zeros((9, num_trail, 3), dtype=np.float32)

    solver_match = object_3d_2d_match(
        pts=input_basic_info['pts'],
        rgb=input_basic_info['rgb'],
        kmat=input_basic_info['kk'])
    # solver_reg   = object_3d_2d_registration(
    #     pts=input_basic_info['pts'],
    #     rgb=input_basic_info['rgb'],
    #     kmat=input_basic_info['kk'])

    noise_level = 0.100 # 0.050 0.100 0.150
    # print("noise level: ", noise_level)

    for i in range(9):
        '''
        in each level set, we conduct 100 random experiments
        '''
        pc_noise_level = 0.1*i
        print("point cloud noise level: ", pc_noise_level)
        tvec  = np.random.normal(0, pc_noise_level, (1,4))
        solver_reg = object_3d_2d_registration(
            pts=input_basic_info['pts'] + tvec,
            rgb=input_basic_info['rgb'],
            kmat=input_basic_info['kk'])

        for k in tqdm.trange(num_trail):
            measured_object_info = add_object_measure_noise(
                object_info, sigma=noise_level)

            # proposed calibration pipeline
            # 2d-3d object match
            solver_match.load_3d_object_from_pts(
                obj_3d_pts=measured_object_info['mea_obj_3d_pts'])
            solver_match.load_3d_object_from_rgb(
                obj_2d_pts=measured_object_info['mea_obj_2d_pts'])
            best_match_res, sort_obj_3d_pts, sort_obj_2d_pts, sort_obj_scores = \
                solver_match.object_match(is_debug=False)
        
            # 2d-3d object registration
            solver_reg.load_sort_obj_info(
                sort_obj_3d_pts, sort_obj_2d_pts, sort_obj_scores)
            solver_reg.load_coarse_calib_res(
                best_match_res['rmat'], best_match_res['tvec'])
            calib_res = solver_reg.points_regiestration(is_use_pnp=True)

            # iteration scheme
            for iter in range(iteration_time):
                solver_reg.load_coarse_calib_res(
                    calib_res['rmat'], calib_res['tvec'])
                calib_res = solver_reg.points_regiestration(is_use_pnp=True)
        
            err_rmat = eva_err_rmat(calib_res['rmat'], calib_gt_info['rr'])
            err_tvec = eva_err_tvec(calib_res['tvec'], calib_gt_info['tt'])
            err_mat_rr_all[i,k] = np.abs(err_rmat)
            err_mat_tt_all[i,k] = np.abs(err_tvec)

        print("=> err_rmat (coarse): ", np.mean(err_mat_rr_all[i], axis=0))
        print("=> err_tvec (coarse): ", np.mean(err_mat_tt_all[i], axis=0))
    
    np.save('./exp_data/exp_1b_'+'r_c_250.npy', err_mat_rr_all)
    np.save('./exp_data/exp_1b_'+'t_c_250.npy', err_mat_tt_all)

def exp_1a(input_basic_info, calib_gt_info, object_info):
    '''    
    1) experiment of bounding box noise 
    3) experiment of 2d-3d match and registration (all noise is fixed)
    6) experiment of iteration 

    sigma: 0.000 0.025 0.050 0.075 0.100 0.125 0.150 0.175 0.200
    '''
    num_trail = 250
    # num_trail = 50
    iteration_time = 3

    err_mat_rr_c_all = np.zeros((9, num_trail, 3), dtype=np.float32) # coarse
    err_mat_tt_c_all = np.zeros((9, num_trail, 3), dtype=np.float32)
    err_mat_pp_c_all = np.zeros((9, num_trail, 3), dtype=np.float32)

    err_mat_rr_r_all = np.zeros((9, num_trail, 3), dtype=np.float32) # refine
    err_mat_tt_r_all = np.zeros((9, num_trail, 3), dtype=np.float32)
    err_mat_pp_r_all = np.zeros((9, num_trail, 3), dtype=np.float32)

    err_mat_rr_i_all = np.zeros((9, num_trail, 3), dtype=np.float32) # iteration + refine
    err_mat_tt_i_all = np.zeros((9, num_trail, 3), dtype=np.float32)
    err_mat_pp_i_all = np.zeros((9, num_trail, 3), dtype=np.float32)

    solver_match = object_3d_2d_match(
        pts=input_basic_info['pts'],
        rgb=input_basic_info['rgb'],
        kmat=input_basic_info['kk'])
    solver_reg   = object_3d_2d_registration(
        pts=input_basic_info['pts'],
        rgb=input_basic_info['rgb'],
        kmat=input_basic_info['kk'])
    
    for i in range(9):
        '''
        in each level set, we conduct 100 random experiments
        '''
        noise_level = 0.025*i
        print("noise level: ", noise_level)
        for k in tqdm.trange(num_trail):
            measured_object_info = add_object_measure_noise(
                object_info, sigma=noise_level)
            
            # proposed calibration pipeline
            # 2d-3d object match
            solver_match.load_3d_object_from_pts(
                obj_3d_pts=measured_object_info['mea_obj_3d_pts'])
            solver_match.load_3d_object_from_rgb(
                obj_2d_pts=measured_object_info['mea_obj_2d_pts'])
            best_match_res, sort_obj_3d_pts, sort_obj_2d_pts, sort_obj_scores = \
                solver_match.object_match(is_debug=False)
        
            err_rmat = eva_err_rmat(best_match_res['rmat'], calib_gt_info['rr'])
            err_tvec = eva_err_tvec(best_match_res['tvec'], calib_gt_info['tt'])
            # err_rmat_c = np.mean(np.linalg.norm(err_rmat))
            # err_tvec_c = np.mean(np.linalg.norm(err_tvec))

            err_mat_rr_c_all[i,k] = np.abs(err_rmat)
            err_mat_tt_c_all[i,k] = np.abs(err_tvec)
            
            # 2d-3d object registration
            solver_reg.load_sort_obj_info(
                sort_obj_3d_pts, sort_obj_2d_pts, sort_obj_scores)
            solver_reg.load_coarse_calib_res(
                best_match_res['rmat'], best_match_res['tvec'])
            calib_res = solver_reg.points_regiestration(is_use_pnp=True)
            
            err_rmat = eva_err_rmat(calib_res['rmat'], calib_gt_info['rr'])
            err_tvec = eva_err_tvec(calib_res['tvec'], calib_gt_info['tt'])
            # err_rmat_r = np.mean(np.linalg.norm(err_rmat))
            # err_tvec_r = np.mean(np.linalg.norm(err_tvec))

            err_mat_rr_r_all[i,k] = np.abs(err_rmat)
            err_mat_tt_r_all[i,k] = np.abs(err_tvec)

            # iteration scheme
            for iter in range(iteration_time):
                solver_reg.load_coarse_calib_res(
                    calib_res['rmat'], calib_res['tvec'])
                calib_res = solver_reg.points_regiestration(is_use_pnp=True)
            
            err_rmat = eva_err_rmat(calib_res['rmat'], calib_gt_info['rr'])
            err_tvec = eva_err_tvec(calib_res['tvec'], calib_gt_info['tt'])
            err_mat_rr_i_all[i,k] = np.abs(err_rmat)
            err_mat_tt_i_all[i,k] = np.abs(err_tvec)

        print("=> err_rmat (coarse): ", np.mean(err_mat_rr_c_all[i],axis=0))
        print("=> err_tvec (coarse): ", np.mean(err_mat_tt_c_all[i],axis=0))
        print("=> err_rmat (refine): ", np.mean(err_mat_rr_r_all[i],axis=0))
        print("=> err_tvec (refine): ", np.mean(err_mat_tt_r_all[i],axis=0))
        print("=> err_rmat (iteration): ", np.mean(err_mat_rr_i_all[i],axis=0))
        print("=> err_tvec (iteration): ", np.mean(err_mat_tt_i_all[i],axis=0))

        print("")
    
    # save experiment results
    # np.save('./exp_data/exp_1a_'+'r_c_250.npy', err_mat_rr_c_all)
    # np.save('./exp_data/exp_1a_'+'t_c_250.npy', err_mat_tt_c_all)
    # np.save('./exp_data/exp_1a_'+'r_r_250.npy', err_mat_rr_r_all)
    # np.save('./exp_data/exp_1a_'+'t_r_250.npy', err_mat_tt_r_all)
    # np.save('./exp_data/exp_1a_'+'r_i_250.npy', err_mat_rr_i_all)
    # np.save('./exp_data/exp_1a_'+'t_i_250.npy', err_mat_tt_i_all)

    np.save('./exp_data/exp_1a_'+'r_c_50.npy', err_mat_rr_c_all)
    np.save('./exp_data/exp_1a_'+'t_c_50.npy', err_mat_tt_c_all)
    np.save('./exp_data/exp_1a_'+'r_r_50.npy', err_mat_rr_r_all)
    np.save('./exp_data/exp_1a_'+'t_r_50.npy', err_mat_tt_r_all)
    np.save('./exp_data/exp_1a_'+'r_i_50.npy', err_mat_rr_i_all)
    np.save('./exp_data/exp_1a_'+'t_i_50.npy', err_mat_tt_i_all)

    print("done. ")

if __name__ == '__main__':
    # ================================================== #
    # Step one: prepare raw dataset and ground truth
    # ================================================== #
    # set save path
    save_path = "/media/anpei/DiskA/multi_calib_lidar_cam/"

    # basic information
    base_path    = "/media/anpei/DiskA/weather-transfer-anpei/"
    dataset_path = base_path + "data/kitti/training/"
    pred_gt_path = dataset_path + "label_2/"
    pred_pd_path = dataset_path + "label_2/"
    lidar_path   = dataset_path + "velodyne/"
    image_path   = dataset_path + "image_2/"
    calib_path   = dataset_path + "calib/"

    img_id = 100+5+10+10+10+5
    lidar_da_path = lidar_path + '%06d.bin'%img_id
    image_da_path = image_path + '%06d.png'%img_id
    label_gt_path = pred_gt_path + '%06d.txt'%img_id
    label_pd_path = pred_pd_path + '%06d.txt'%img_id
    calib_path    = calib_path   + '%06d.txt'%img_id

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
            is_save_img=False,
            is_kitti_type=True)
    obj_2d_pts_ = obj_2d_pts.copy()

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

    # ================================================== #
    # Step two: simulation the measure noise
    # (include 3d object detection error, the random sort)
    # ================================================== #
    input_basic_info = {'pts': pts, 'rgb': rgb, 'kk': kk}
    calib_gt_info    = {'rr': rr, 'tt': tt}

    obj_num     = obj_3d_pts.shape[0]
    obj_3d_pts  = obj_3d_pts.reshape((-1,3))
    obj_2d_pts  = obj_2d_pts.reshape((-1,2))
    object_info = {'num': obj_num, 'obj_3d_pts': obj_3d_pts, 'obj_2d_pts':obj_2d_pts}

    # ================================================== #
    # Step three: conduct the simulated experiments
    # ================================================== #
    # exp_1a(input_basic_info, calib_gt_info, object_info)
    # exp_1b(input_basic_info, calib_gt_info, object_info)
    # exp_1c(input_basic_info, calib_gt_info, object_info)
    # exp_1d(input_basic_info, calib_gt_info, object_info)
    
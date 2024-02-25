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

# in kitti dataset, we use gt 3d boxes for simulation

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
    img_id = 100+100+50+100+17
    lidar_da_path = lidar_path + '%06d.bin'%img_id
    image_da_path = image_path + '%06d.png'%img_id
    label_gt_path = pred_gt_path + '%06d.txt'%img_id
    label_pd_path = pred_pd_path + '%06d.txt'%img_id
    calib_path    = calib_path   + '%06d.txt'%img_id

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
    pixels, depths = projection(pts, kk, rr, tt)

    # note-0314 
    # use result computed by 3d-2d bounding box
    # as gt calibration
    dist_coeffs = np.zeros((4, 1)) 
    obj_num     = obj_3d_pts.shape[0]
    obj_3d_pts  = obj_3d_pts.reshape((-1,3))
    obj_2d_pts  = obj_2d_pts.reshape((-1,2))
    # obj_2d_pts, _ = projection(obj_3d_pts, kk, rr, tt)
    success, rvec, tvec = cv.solvePnP(
        obj_3d_pts, obj_2d_pts, kk, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
    rmat = cv.Rodrigues(rvec)[0]
    pixels, depths = projection(pts, kk, rmat, tvec)

    proj_tool = ProjectionProcessor(w=rgb.shape[1], h=rgb.shape[0])
    depth_img = proj_tool.getDepth(depths, pixels)
    depth_img_vis = proj_tool.getDepthVis(depth_img)
    mergeVis = cv.addWeighted(rgb, 0.50, depth_img_vis, 0.50, 0)

    # cv.imshow("depth_img_vis", depth_img_vis)
    # cv.imshow("mergeVis", mergeVis)
    # cv.waitKey()
    # cv.imwrite("draw_img.png", mergeVis)

    # ================================================== #
    # Step two: simulation the measure noise
    # (include 3d object detection error, the random sort)
    # needs visulization
    # (visulization is good)
    # ================================================== #
    from utilsCalib import add_object_measure_noise
    object_info      = {'num': obj_num, 'obj_3d_pts': obj_3d_pts, 'obj_2d_pts':obj_2d_pts}
    input_basic_info = {'pts': pts, 'rgb': rgb, 'kk': kk}
    calib_gt_info    = {'rr': rr, 'tt': tt}
    measured_object_info = add_object_measure_noise(object_info, sigma=0.050)
    measured_object_info = add_object_measure_noise(object_info, sigma=0.005)
    # measured_object_info = add_object_measure_noise(object_info, sigma=0.000)

    from utils3dVisual import show_3d_bounding_box
    from utils3dVisual import show_2d_bounding_box
    PointsVis_1 = show_3d_bounding_box(
        pts = input_basic_info['pts'], 
        obj_num = object_info['num'], 
        obj_3d_pts = measured_object_info['obj_3d_pts'], 
        color=(0,1,0))
    PointsVis_2 = show_3d_bounding_box(
        pts = input_basic_info['pts'], 
        obj_num = object_info['num'], 
        obj_3d_pts = measured_object_info['mea_obj_3d_pts'], 
        color=(0,0,1))

    PointsVis = PointsVis_1 + PointsVis_2
    vis = o3d.visualization.Visualizer()
    vis.create_window("3d object detection visulization")
    render_options: o3d.visualization.RenderOption = vis.get_render_option()
    render_options.background_color = np.array([0,0,0])
    render_options.point_size = 3.0
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    R = mesh.get_rotation_matrix_from_xyz((-np.pi/3, 0, np.pi / 2))
    PointsVis.rotate(R, center=(0, 0, 0)) 
    vis.add_geometry(PointsVis)
    vis.poll_events()
    vis.update_renderer()
    vis.run() 

    vis_img_1 = show_2d_bounding_box(
        rgb = input_basic_info['rgb'], 
        obj_num = object_info['num'], 
        obj_2d_pts = measured_object_info['obj_2d_pts'],
        color=(0,255,0))
    vis_img_2 = show_2d_bounding_box(
        rgb = input_basic_info['rgb'], 
        obj_num = object_info['num'], 
        obj_2d_pts = measured_object_info['mea_obj_2d_pts'],
        color=(255,0,0))
    mergeVis = cv.addWeighted(vis_img_1, 0.50, vis_img_2, 0.50, 0)
    cv.imshow("mergeVis", mergeVis)
    cv.waitKey()
    cv.imwrite("object3d.png", mergeVis)

    # ================================================== #
    # Step three: calibration method
    # ================================================== #
    from calibSolution import object_3d_2d_match
    from calibSolution import object_3d_2d_registration
    from calibSolution import eva_err_rmat, eva_err_tvec
    
    # step 3.1
    # 3d-2d object matching from rgb and point cloud
    solver_match = object_3d_2d_match(
        pts=input_basic_info['pts'],
        rgb=input_basic_info['rgb'],
        kmat=input_basic_info['kk'])
    solver_match.load_3d_object_from_pts(
        obj_3d_pts=measured_object_info['mea_obj_3d_pts'])
    solver_match.load_3d_object_from_rgb(
        obj_2d_pts=measured_object_info['mea_obj_2d_pts'])
    best_match_res, sort_obj_3d_pts, sort_obj_2d_pts, sort_obj_scores = \
        solver_match.object_match()

    '''
    note-0419 
    
        one trick: use the histiry object info to improve the 
                   stability of 3d-2d registration
    '''
    np.save("history_obj_3d_pts.npy", measured_object_info['mea_obj_3d_pts'])
    np.save("history_obj_2d_pts.npy", measured_object_info['mea_obj_2d_pts'])
    np.save("history_obj_3d_pts.npy", object_info['obj_3d_pts'])
    np.save("history_obj_2d_pts.npy", object_info['obj_2d_pts'])
    print("save history object info ... ")
    # assert -1

    err_rmat = eva_err_rmat(best_match_res['rmat'], calib_gt_info['rr'])
    err_tvec = eva_err_tvec(best_match_res['tvec'], calib_gt_info['tt'])
    print("=> err_rmat (coarse): ", err_rmat)
    print("=> err_tvec (coarse): ", err_tvec)

    pixels, depths = projection(
        pts, kk, best_match_res['rmat'], best_match_res['tvec'])
    # pixels, depths = projection(
    #     pts, kk, calib_gt_info['rr'], calib_gt_info['tt'])
    depth_img = proj_tool.getDepth(depths, pixels)
    depth_img_vis = proj_tool.getDepthVis(depth_img)
    mergeVis_a = cv.addWeighted(rgb, 0.30, depth_img_vis, 0.70, 0)
    # cv.imshow("3d object detection", mergeVis)
    cv.imshow("coarse calibration result", mergeVis_a)
    cv.waitKey()
    cv.imwrite("coarse.png", mergeVis_a)

    # add visulization of 3d color point cloud
    from utils3dVisual import show_pcd
    p, c = proj_tool.get_color_point(depths, pixels, pts, rgb)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(p[:,:3])
    pc.colors = o3d.utility.Vector3dVector(c[:,:3])
    show_pcd(pc)

    # step 3.2
    # 3d-2d registration from the matched objects
    solver_match = object_3d_2d_registration(
        pts=input_basic_info['pts'],
        rgb=input_basic_info['rgb'],
        kmat=input_basic_info['kk'])
    solver_match.load_sort_obj_info(
        sort_obj_3d_pts, sort_obj_2d_pts, sort_obj_scores)
    solver_match.load_coarse_calib_res(
        best_match_res['rmat'], best_match_res['tvec'])
    # solver_match.load_coarse_calib_res(
    #     calib_gt_info['rr'], calib_gt_info['tt'])
    calib_res = solver_match.points_regiestration(is_use_history=True)
    err_rmat = eva_err_rmat(calib_res['rmat'], calib_gt_info['rr'])
    err_tvec = eva_err_tvec(calib_res['tvec'], calib_gt_info['tt'])
    print("=> err_rmat (iter-1): ", err_rmat)
    print("=> err_tvec (iter-1): ", err_tvec)

    # === compared with pnp algorithm === #
    # solver_match_pnp = object_3d_2d_registration(
    #     pts=input_basic_info['pts'],
    #     rgb=input_basic_info['rgb'],
    #     kmat=input_basic_info['kk'])
    # solver_match_pnp.load_sort_obj_info(
    #     sort_obj_3d_pts, sort_obj_2d_pts, sort_obj_scores)
    # solver_match_pnp.load_coarse_calib_res(
    #     best_match_res['rmat'], best_match_res['tvec'])
    # calib_res_pnp = solver_match_pnp.points_regiestration(is_use_pnp=True)
    # err_rmat = eva_err_rmat(calib_res_pnp['rmat'], calib_gt_info['rr'])
    # err_tvec = eva_err_tvec(calib_res_pnp['tvec'], calib_gt_info['tt'])
    # print("=> err_rmat (pnp-comp): ", err_rmat)
    # print("=> err_tvec (pnp-comp): ", err_tvec)
    # =================================== #

    # --- iterative 2d-3d registration from the matched objects 
    '''
        note-0329

        using iterative registration on objects,
        error of rmat and tvec are decreased.

        however, there is still inaccurate performance if the 
        initial calibration error is large. (low robustness)

        to solve this problem, we add a global 2d-3d registration. 
    '''
    solver_match.load_coarse_calib_res(
        calib_res['rmat'], calib_res['tvec'])
    calib_res = solver_match.points_regiestration(is_use_history=True)
    err_rmat = eva_err_rmat(calib_res['rmat'], calib_gt_info['rr'])
    err_tvec = eva_err_tvec(calib_res['tvec'], calib_gt_info['tt'])
    # print("=> err_rmat (iter-2): ", err_rmat)
    # print("=> err_tvec (iter-2): ", err_tvec)
    solver_match.load_coarse_calib_res(
        calib_res['rmat'], calib_res['tvec'])
    calib_res = solver_match.points_regiestration(is_use_history=True)
    err_rmat = eva_err_rmat(calib_res['rmat'], calib_gt_info['rr'])
    err_tvec = eva_err_tvec(calib_res['tvec'], calib_gt_info['tt'])
    # print("=> err_rmat (iter-3): ", err_rmat)
    # print("=> err_tvec (iter-3): ", err_tvec)
    solver_match.load_coarse_calib_res(
        calib_res['rmat'], calib_res['tvec'])
    calib_res = solver_match.points_regiestration(is_use_history=True)
    err_rmat = eva_err_rmat(calib_res['rmat'], calib_gt_info['rr'])
    err_tvec = eva_err_tvec(calib_res['tvec'], calib_gt_info['tt'])
    print("=> err_rmat (refined): ", err_rmat)
    print("=> err_tvec (refined): ", err_tvec)

    pixels, depths = projection(
        pts, kk, calib_res['rmat'], calib_res['tvec'])
    # pixels, depths = projection(
    #     pts, kk, calib_gt_info['rr'], calib_gt_info['tt'])
    depth_img = proj_tool.getDepth(depths, pixels)
    depth_img_vis = proj_tool.getDepthVis(depth_img)
    mergeVis_a = cv.addWeighted(rgb, 0.30, depth_img_vis, 0.70, 0)
    # cv.imshow("3d object detection", mergeVis)
    cv.imshow("refined calibration result", mergeVis_a)
    cv.waitKey()
    cv.imwrite("refine.png", mergeVis_a)

    # step 3.3
    # 3d-2d registration from the global scene
    '''
        note-0329

        aim to improve the calibration robustness
    '''
    # solver_match.load_coarse_calib_res(
    #     calib_res['rmat'], calib_res['tvec'])
    # calib_res = solver_match.points_regiestration_global()
    # err_rmat = eva_err_rmat(calib_res['rmat'], calib_gt_info['rr'])
    # err_tvec = eva_err_tvec(calib_res['tvec'], calib_gt_info['tt'])
    # print("=> err_rmat (global): ", err_rmat)
    # print("=> err_tvec (global): ", err_tvec)

    # solver_match.load_coarse_calib_res(
    #     calib_res['rmat'], calib_res['tvec'])
    # calib_res = solver_match.points_regiestration()
    # err_rmat = eva_err_rmat(calib_res['rmat'], calib_gt_info['rr'])
    # err_tvec = eva_err_tvec(calib_res['tvec'], calib_gt_info['tt'])
    # print("=> err_rmat (refined): ", err_rmat)
    # print("=> err_tvec (refined): ", err_tvec)

    # == calibration visulization reuslt
    # pixels, depths = projection(
    #     pts, kk, calib_res['rmat'], calib_res['tvec'])
    # # pixels, depths = projection(
    # #     pts, kk, calib_gt_info['rr'], calib_gt_info['tt'])
    # depth_img = proj_tool.getDepth(depths, pixels)
    # depth_img_vis = proj_tool.getDepthVis(depth_img)
    # mergeVis_2 = cv.addWeighted(rgb, 0.30, depth_img_vis, 0.70, 0)
    # cv.imshow("coarse calibration result", mergeVis_a)
    # cv.imshow("refine calibration result", mergeVis_2)
    # cv.waitKey()
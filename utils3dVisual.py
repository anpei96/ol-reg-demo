#
# Project: lidar-camera system calibration based on
#          object-level 3d-2d correspondence
# Author:  anpei
# Data:    2023.03.07
# Email:   anpei@wit.edu.cn
#

import os
import cv2
import torch
import numpy as np
import open3d as o3d

from utilsKitti  import Calibration
from utilsVisual import read_detection, plot_3dbox, read_detection_kitti
from utilsVisual import plot_3dbox_image
from utilsVisual import plot_3dbox_pts, plot_3dbox_image_per

def show_pcd(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window("3d object detection visulization")
    render_options: o3d.visualization.RenderOption = vis.get_render_option()
    render_options.background_color = np.array([0,0,0])
    render_options.point_size = 3.0
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.run() 

def show_3d_bounding_box(pts, obj_num, obj_3d_pts, color=(0,1,0)):
    obj_3d_pts_ = obj_3d_pts.reshape((obj_num, 8, 3))
    box_3d_pts  = np.zeros((0,6),dtype=float)
    for i in range(obj_num):
        corner_pts = obj_3d_pts_[i]
        save_pts   = plot_3dbox_pts(corner_pts.T)
        save_pts   = (save_pts.T)[:,:6]
        box_3d_pts = np.concatenate((box_3d_pts, save_pts), axis = 0)
    box_3d_pts[:,3] = color[0]
    box_3d_pts[:,4] = color[1]
    box_3d_pts[:,5] = color[2]

    lidar_pts = np.zeros((pts.shape[0], 6), dtype=np.float)
    lidar_pts[:,:3]  = pts[:,:3]
    lidar_pts[:,3:6] = 1.0
    
    visual_pts = np.concatenate((box_3d_pts, lidar_pts), axis = 0)
    PointsVis = o3d.geometry.PointCloud()
    PointsVis.points = o3d.utility.Vector3dVector(visual_pts[:,:3])
    PointsVis.colors = o3d.utility.Vector3dVector(visual_pts[:,3:]) 

    # print("box_3d_pts: ", box_3d_pts.shape)
    # print("pts: ", pts.shape)
    # vis = o3d.visualization.Visualizer()
    # vis.create_window("3d object detection visulization")
    # render_options: o3d.visualization.RenderOption = vis.get_render_option()
    # render_options.background_color = np.array([0,0,0])
    # render_options.point_size = 3.0
    # vis.add_geometry(PointsVis)
    # vis.poll_events()
    # vis.update_renderer()
    # vis.run() 

    return PointsVis

def show_2d_bounding_box(rgb, obj_num, obj_2d_pts, color=(255,0,0)):
    obj_2d_pts_ = obj_2d_pts.reshape((obj_num, 8, 2))
    vis_img     = rgb.copy()
    for i in range(obj_num):
        corner_pts = obj_2d_pts_[i]
        vis_img = plot_3dbox_image_per(
            vis_img, corner_pts, color=color, circ_radius=4, line_thick=2)
    
    # cv2.imshow("vis_img", vis_img)
    # cv2.waitKey()
    return vis_img

from open3d import geometry
def show_3d_detection_pts(
    pred_bboxes_3d, pcd_path, calib, is_black_bg=True):
    points = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
    points = points[:, :4]
    num    = points.shape[0]

    if is_black_bg:
        points = np.concatenate((points[:, :3],np.ones((num, 3))*0.75),  axis = 1)
    else:
        points = np.concatenate((points[:, :3],np.zeros((num, 3))), axis = 1)
    # # points[:, 5] = 0
    # points[:, 3] = 0

    PointsVis = o3d.geometry.PointCloud()
    PointsVis.points = o3d.utility.Vector3dVector(points[:,:3])
    PointsVis.colors = o3d.utility.Vector3dVector(points[:,3:]) 

    vis = o3d.visualization.Visualizer()
    vis.create_window("3d object detection visulization")
    render_options: o3d.visualization.RenderOption = vis.get_render_option()
    if is_black_bg:
        render_options.background_color = np.array([0,0,0])
    render_options.point_size = 1.0
    vis.add_geometry(PointsVis)

    bbox3d = pred_bboxes_3d
    num = bbox3d.shape[0]
    rot_axis = 2
    center_mode = 'lidar_bottom'
    bbox_color = (0, 1, 0)

    for i in range(num):
        center = bbox3d[i, 0:3]
        dim = bbox3d[i, 3:6]
        yaw = np.zeros(3)
        yaw[rot_axis] = bbox3d[i, 6]
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)
        if center_mode == 'lidar_bottom':
            center[rot_axis] += dim[
                rot_axis] / 2  # bottom center to gravity center
        elif center_mode == 'camera_bottom':
            center[rot_axis] -= dim[
                rot_axis] / 2  # bottom center to gravity center
        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)
        line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color(bbox_color)
        vis.add_geometry(line_set)
    
    vis.poll_events()
    vis.update_renderer()
    vis.run() 

from utilsVisual import plot_3dbox_image_light
def show_3d_detection_img(rgb, obj_corner, calib):
    draw_img = plot_3dbox_image_light(rgb, obj_corner, calib)
    is_save_img = True
    x,y = rgb.shape[0:2]
    if is_save_img:
        # vis_draw_img = cv2.resize(draw_img, (int(y / 2), int(x / 2)))
        cv2.imshow("vis_draw_img", draw_img)
        cv2.waitKey()
        # cv2.imwrite("draw_img.png", draw_img)

def show3DdetectionResults(
    lidar_da_path, 
    image_da_path,
    label_gt_path, 
    label_pd_path, 
    calib_path, 
    save_path, 
    is_has_gt=False, 
    is_black_bg=False, 
    is_save=False,
    is_save_img=False,
    is_kitti_type=False):

    # show 3d object detection in lidar point cloud
    if is_kitti_type == True:
        points = np.fromfile(lidar_da_path, dtype=np.float32).reshape(-1, 4)
        points = points[:, :4]
    else:
        points = np.fromfile(lidar_da_path, dtype=np.float32).reshape(-1, 5)
        points = points[:, :4]
    num = points.shape[0]
    pad = np.zeros((points.shape[0],2))
    raw_pts = points.copy()

    if is_kitti_type == True:
        df_pd = read_detection_kitti(label_pd_path)
        if is_has_gt: 
            df = read_detection_kitti(label_gt_path)
    else:
        df_pd = read_detection(label_pd_path)
        if is_has_gt: 
            df = read_detection(label_gt_path)

    if is_black_bg:
        points = np.concatenate((points[:, :3],np.ones((num, 3))), axis = 1)
    else:
        points = np.concatenate((points[:, :3],np.zeros((num, 3))), axis = 1)

    calib = Calibration(calib_path)

    # plot 3d box
    if is_has_gt:
        save_ = plot_3dbox(df, calib)
        save_points_velo = corners_3d_velo = calib.project_rect_to_velo(save_[0:3,:].T)
        save_[0:3,:] = save_points_velo.T
        points = np.concatenate((points,save_.T[:,:6]), axis = 0)

    save_pd = plot_3dbox(df_pd, calib)
    save_points_velo_pd = corners_3d_velo_pd = calib.project_rect_to_velo(save_pd[0:3,:].T)
    save_pd[0:3,:] = save_points_velo_pd.T
    save_pd_t = save_pd.T
    save_pd_t[:,4] = 0
    save_pd_t[:,5] = 1

    points = np.concatenate((points,save_pd_t[:,:6]), axis = 0)

    PointsVis = o3d.geometry.PointCloud()
    PointsVis.points = o3d.utility.Vector3dVector(points[:,:3])
    PointsVis.colors = o3d.utility.Vector3dVector(points[:,3:]) 

    # o3d.visualization.draw_geometries([PointsVis])
    if is_save:
        o3d.io.write_point_cloud(save_path + "results_comp.pcd", PointsVis)

    # view angel
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    R = mesh.get_rotation_matrix_from_xyz((-np.pi/3, 0, np.pi / 2))
    # R = mesh.get_rotation_matrix_from_xyz((0, 0, 0 / 2))
    PointsVis.rotate(R, center=(0, 0, 0))  

    # vis = o3d.visualization.Visualizer()
    # vis.create_window("3d object detection visulization")
    # render_options: o3d.visualization.RenderOption = vis.get_render_option()
    # if is_black_bg:
    #     render_options.background_color = np.array([0,0,0])
    # render_options.point_size = 3.0
    # vis.add_geometry(PointsVis)
    # vis.poll_events()
    # vis.update_renderer()
    # vis.run() 

    # show 3d object detection in rgb image
    rgb = cv2.imread(image_da_path)
    x,y = rgb.shape[0:2]
    draw_img = plot_3dbox_image(rgb, df, df_pd, calib, is_kitti=is_kitti_type)
    if is_save_img:
        vis_draw_img = cv2.resize(draw_img, (int(y / 2), int(x / 2)))
        # cv2.imshow("vis_draw_img", vis_draw_img)
        # cv2.waitKey()
        cv2.imwrite("draw_img.png", draw_img)

    # outputs object 3d bounding boxes in 3d and 2d coordinates
    # obj-3d-pts (n,8,3) (x,y,z)
    # obj-2d-pix (n,8,2) (u,v)
    from utilsVisual import compute_3d_box_cam2
    num = len(df)
    obj_3d_pts = np.zeros((num, 8, 3))
    obj_2d_pts = np.zeros((num, 8, 2))
    for o in range(len(df)):
        corners_3d_cam2 = compute_3d_box_cam2(*df.loc[o, ['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']])
        corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)
        corners_2d_cam2 = calib.project_rect_to_image(corners_3d_cam2.T)
        obj_3d_pts[o,:] = corners_3d_velo
        obj_2d_pts[o,:] = corners_2d_cam2
        # print("corners_3d_velo: ", corners_3d_velo.shape)
        # print("corners_2d_cam2: ", corners_2d_cam2.shape)
    return raw_pts, rgb, obj_3d_pts, obj_2d_pts
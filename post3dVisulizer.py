#
# Project: transfer learning based 3d object detection
#          in the adverse weather
# Author:  anpei
# Data:    2021.09.08
# Email:   22060402@wit.edu.cn
#

import os
import cv2
import torch
import numpy as np
import open3d as o3d

from utilsKitti  import Calibration
from utilsVisual import read_detection, plot_3dbox
from utilsVisual import plot_3dbox_image

def show3DdetectionResults(
    lidar_da_path, 
    image_da_path,
    label_gt_path, 
    label_pd_path, 
    calib_path, 
    save_path, 
    is_has_gt=False, 
    is_black_bg=False, 
    is_save=False):

    # show 3d object detection in lidar point cloud
    points = np.fromfile(lidar_da_path, dtype=np.float32).reshape(-1, 5)
    points = points[:, :4]
    num = points.shape[0]
    pad = np.zeros((points.shape[0],2))

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

    vis = o3d.visualization.Visualizer()
    vis.create_window("3d object detection visulization")
    render_options: o3d.visualization.RenderOption = vis.get_render_option()
    if is_black_bg:
        render_options.background_color = np.array([0,0,0])
    render_options.point_size = 3.0

    vis.add_geometry(PointsVis)
    vis.poll_events()
    vis.update_renderer()
    vis.run() 

    # show 3d object detection in rgb image
    rgb = cv2.imread(image_da_path)
    x,y = rgb.shape[0:2]
    draw_img = plot_3dbox_image(rgb, df, df_pd, calib)
    vis_draw_img = cv2.resize(draw_img, (int(y / 2), int(x / 2)))
    # cv2.imshow("vis_draw_img", vis_draw_img)
    # cv2.waitKey()
    cv2.imwrite("draw_img.png", draw_img)

if __name__ == '__main__':
    # basic information
    base_path    = "/media/anpei/DiskA/weather-transfer-anpei/"
    save_path    = base_path + "anpei_visual_detection/"
    dataset_path = base_path + "data/seeingthroughfog/training/"
    pred_gt_path = dataset_path + "label_2/"
    pred_pd_path = dataset_path + "label_2/"
    lidar_path   = dataset_path + "velodyne/"
    image_path   = dataset_path + "image_2/"
    calib_path   = dataset_path + "calib/kitti_stereo_velodynehdl_calib.txt"

    # pred_pd_path = save_path + "sp-det/clear/data/"
    pred_pd_path = save_path + "sp-det/light-fog/data/"
    # pred_pd_path = save_path + "sp-det/dense-fog/data/"
    pred_pd_path = save_path + "sp-det/snow/data/"

    # pred_pd_path = save_path + "pv-rcnn/light-fog/data/"
    # pred_pd_path = save_path + "voxel-rcnn/light-fog/data/"

    # pred_pd_path = save_path + "pv-rcnn/dense-fog/data/"
    # pred_pd_path = save_path + "voxel-rcnn/dense-fog/data/"

    # pred_pd_path = save_path + "pv-rcnn/snow/data/"
    # pred_pd_path = save_path + "voxel-rcnn/snow/data/"

    # if visualize all samples
    idx = 111
    file_name_gt_list = os.listdir(pred_gt_path)
    file_name_pd_list = os.listdir(pred_pd_path)
    gt_label_name = file_name_gt_list[idx] 
    pd_label_name = file_name_pd_list[idx]

    # or we only visulize the weather split
    clear_split = base_path + "data/seeingthroughfog/ImageSets/val_clear.txt"
    dense_fog_split = base_path + "data/seeingthroughfog/ImageSets/val_dense_fog.txt"
    light_fog_split = base_path + "data/seeingthroughfog/ImageSets/val_light_fog.txt"
    snow_split  = base_path + "data/seeingthroughfog/ImageSets/val_snow.txt"

    idx = 100
    idx = 20+5+20+20 # clear
    idx = 18+12+40+100 # light-fog
    idx = 2+5+12+20 # dense-fog
    idx = 2+10+20+22 # snow

    idx = 100+15 # light-fog-comp
    idx = 100+25+60 # dense-fog-comp
    idx = 100+25+60 # snow-comp

    #===
    idx = 18+12+25 # light-fog
    idx = 2+10+15 # snow
    #===

    # weather_split = clear_split
    weather_split = light_fog_split
    # weather_split = dense_fog_split
    weather_split = snow_split
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
    # print("lidar_da_path: ", lidar_da_path)
    # print("image_da_path: ", image_da_path)
    
    show3DdetectionResults(
        lidar_da_path, 
        image_da_path,
        label_gt_path, 
        label_pd_path, 
        calib_path, 
        save_path, 
        is_has_gt=True, 
        is_black_bg=True, 
        is_save=False)





from turtle import color
import numpy as np
import os
import matplotlib.pyplot as plt

import matplotlib.patches as patches
import yaml
import pandas as pd
import cv2
import open3d as o3d
import torch

from mpl_toolkits.mplot3d import Axes3D
from utilsKitti import *
from matplotlib.lines import Line2D

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

def compute_3d_box_cam2(h, w, l, x, y, z, yaw):
    """
    Return : 3xn in cam2 coordinate
    """
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d_cam2 += np.vstack([x, y, z])
    return corners_3d_cam2

def draw_box(ax, vertices, axes=[0, 1, 2], color='black'):
    """
    Draws a bounding 3D box in a pyplot axis.
    
    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
    ]
    for connection in connections:
        ax.plot(*vertices[:, connection], c=color, lw=0.5)


def read_detection(path):
    df = pd.read_csv(path, header=None, sep=' ')
    print(df)
    df.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y', ''] # 'score'
    df.loc[df.type.isin(['Truck', 'Van', 'Tram']), 'type'] = 'Car'
    df = df[df.type.isin(['Car', 'Pedestrian', 'Cyclist'])]
    # df = df[df['type']=='Car']
    # df = df[(df['type']=='Car') | (df['type']=='Van') | (df['type']=='Truck') | (df['type']=='Tram')]
    df.reset_index(drop=True, inplace=True)
    return df

def read_detection_kitti(path):
    df = pd.read_csv(path, header=None, sep=' ')
    print(df)
    df.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y'] # 'score'
    df.loc[df.type.isin(['Truck', 'Van', 'Tram']), 'type'] = 'Car'
    df = df[df.type.isin(['Car', 'Pedestrian', 'Cyclist'])]
    # df = df[df['type']=='Car']
    # df = df[(df['type']=='Car') | (df['type']=='Van') | (df['type']=='Truck') | (df['type']=='Tram')]
    df.reset_index(drop=True, inplace=True)
    return df

def read_detection_pd(path):
    df = pd.read_csv(path, header=None, sep=' ')
    df.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
                'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y', 'score']
    # df.loc[df.type.isin(['Truck', 'Van', 'Tram']), 'type'] = 'Car'
    df = df[df.type.isin(['Car', 'Pedestrian', 'Cyclist'])]
    # df = df[df['type']=='Car']
    df.reset_index(drop=True, inplace=True)
    return df

def read_detection_gt_stf(path):
    """
        only keep PassengerCar and Pedestrian
    """
    df = pd.read_csv(path, header=None, sep=' ')
    df.columns = ['type', 'truncated', 'occluded', 'alpha', 
    'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom', 
    'height', 'width', 'length', 
    'pos_x', 'pos_y', 'pos_z', 
    'rot_y', 'rx', 'ry', 'rz',
    'score',
    'qx', 'qy', 'qz', 'qw', 
    'visible_RGB', 'visible_Gated', 'visible_LiDAR', 'Empty']
    df = df[df.type.isin(['PassengerCar', 'Pedestrian'])]
    df.reset_index(drop=True, inplace=True)
    return df

# corners3d is [x, 8, 3] array
def plot_3dbox_easy(corners3d):
    save_ = np.zeros((7,0),dtype=float)
    num = corners3d.shape[0]
    for o in range(num):
        x0,y0,z0 = corners3d[o, 0, :3]
        x1,y1,z1 = corners3d[o, 1, :3]
        x2,y2,z2 = corners3d[o, 2, :3]
        x3,y3,z3 = corners3d[o, 3, :3]
        x4,y4,z4 = corners3d[o, 4, :3]
        x5,y5,z5 = corners3d[o, 5, :3]
        x6,y6,z6 = corners3d[o, 6, :3]
        x7,y7,z7 = corners3d[o, 7, :3]

        save_points1 = compute_3D_line(x0,x1, y0,y1, z0,z1)
        save_points2 = compute_3D_line(x1,x2, y1,y2, z1,z2)
        save_points3 = compute_3D_line(x2,x3, y2,y3, z2,z3)
        save_points4 = compute_3D_line(x3,x0, y3,y0, z3,z0)
        save_a = np.concatenate((save_points1,save_points2,save_points3,save_points4), axis = 1)
        
        save_points1 = compute_3D_line(x4,x5, y4,y5, z4,z5)
        save_points2 = compute_3D_line(x5,x6, y5,y6, z5,z6)
        save_points3 = compute_3D_line(x6,x7, y6,y7, z6,z7)
        save_points4 = compute_3D_line(x7,x4, y7,y4, z7,z4)
        save_b = np.concatenate((save_points1,save_points2,save_points3,save_points4), axis = 1)

        save_points1 = compute_3D_line(x4,x0, y4,y0, z4,z0)
        save_points2 = compute_3D_line(x5,x1, y5,y1, z5,z1)
        save_points3 = compute_3D_line(x6,x2, y6,y2, z6,z2)
        save_points4 = compute_3D_line(x7,x3, y7,y3, z7,z3)
        save_c = np.concatenate((save_points1,save_points2,save_points3,save_points4), axis = 1)

        save_ = np.concatenate((save_,save_a,save_b, save_c,), axis = 1)
    return save_

def plot_3dbox_image_per(
    draw_img, box_pts, color=(0,255,0), circ_radius=4, line_thick=2):
    for i in range(8):
        draw_img = cv2.circle(
            draw_img, (int(box_pts[i,0]), int(box_pts[i,1])), 
            radius=circ_radius, color=color, thickness=-1)
    for i in range(4):
        draw_img = cv2.line(draw_img, 
            (int(box_pts[i,0]), int(box_pts[i,1])),
            (int(box_pts[i+4,0]), int(box_pts[i+4,1])),
            color, line_thick)
    for i in range(3):
        draw_img = cv2.line(draw_img, 
            (int(box_pts[i,0]), int(box_pts[i,1])),
            (int(box_pts[i+1,0]), int(box_pts[i+1,1])),
            color, line_thick)
        draw_img = cv2.line(draw_img, 
            (int(box_pts[i+4,0]), int(box_pts[i+4,1])),
            (int(box_pts[i+5,0]), int(box_pts[i+5,1])),
            color, line_thick)
    draw_img = cv2.line(draw_img, 
            (int(box_pts[3,0]), int(box_pts[3,1])),
            (int(box_pts[0,0]), int(box_pts[0,1])),
            color, line_thick)
    draw_img = cv2.line(draw_img, 
            (int(box_pts[7,0]), int(box_pts[7,1])),
            (int(box_pts[4,0]), int(box_pts[4,1])),
            color, line_thick)
    return draw_img

def plot_3dbox_image_light(img, obj_corner, calib):
    draw_img = np.copy(img)
    circ_radius_gt = 4
    circ_radius_pd = 2
    line_thick     = 2
    for o in range(obj_corner.shape[0]):
        corners_3d_cam2 = obj_corner[o]
        corners_2d_cam2 = calib.project_rect_to_image(corners_3d_cam2)
        for i in range(8):
                draw_img = cv2.circle(
                    draw_img, (int(corners_2d_cam2[i,0]), int(corners_2d_cam2[i,1])), 
                    radius=circ_radius_pd, color=(255,0,0), thickness=-1)
        for i in range(4):
            draw_img = cv2.line(draw_img, 
                (int(corners_2d_cam2[i,0]), int(corners_2d_cam2[i,1])),
                (int(corners_2d_cam2[i+4,0]), int(corners_2d_cam2[i+4,1])),
                (255,0,0), line_thick)
        for i in range(3):
            draw_img = cv2.line(draw_img, 
                (int(corners_2d_cam2[i,0]), int(corners_2d_cam2[i,1])),
                (int(corners_2d_cam2[i+1,0]), int(corners_2d_cam2[i+1,1])),
                (255,0,0), line_thick)
            draw_img = cv2.line(draw_img, 
                (int(corners_2d_cam2[i+4,0]), int(corners_2d_cam2[i+4,1])),
                (int(corners_2d_cam2[i+5,0]), int(corners_2d_cam2[i+5,1])),
                (255,0,0), line_thick)
        draw_img = cv2.line(draw_img, 
                (int(corners_2d_cam2[3,0]), int(corners_2d_cam2[3,1])),
                (int(corners_2d_cam2[0,0]), int(corners_2d_cam2[0,1])),
                (255,0,0), line_thick)
        draw_img = cv2.line(draw_img, 
                (int(corners_2d_cam2[7,0]), int(corners_2d_cam2[7,1])),
                (int(corners_2d_cam2[4,0]), int(corners_2d_cam2[4,1])),
                (255,0,0), line_thick)
    return draw_img

def plot_3dbox_image(img, df_gt, df_pd, calib, is_kitti=False):
    draw_img = np.copy(img)

    if is_kitti == True:
        circ_radius_gt = 4#10//2
        circ_radius_pd = 2#6//2
        line_thick     = 2# 4//2
    else:
        circ_radius_gt = 10
        circ_radius_pd = 6
        line_thick     = 4

    for o in range(len(df_pd)):
        corners_3d_cam2 = compute_3d_box_cam2(*df_pd.loc[o, ['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']])
        corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)
        
        if np.abs(np.max(corners_3d_velo)) > 500:
            continue
        
        corners_2d_cam2 = calib.project_rect_to_image(corners_3d_cam2.T)

        for i in range(8):
            draw_img = cv2.circle(
                draw_img, (int(corners_2d_cam2[i,0]), int(corners_2d_cam2[i,1])), 
                radius=circ_radius_pd, color=(255,0,0), thickness=-1)
        for i in range(4):
            draw_img = cv2.line(draw_img, 
                (int(corners_2d_cam2[i,0]), int(corners_2d_cam2[i,1])),
                (int(corners_2d_cam2[i+4,0]), int(corners_2d_cam2[i+4,1])),
                (255,0,0), line_thick)
        for i in range(3):
            draw_img = cv2.line(draw_img, 
                (int(corners_2d_cam2[i,0]), int(corners_2d_cam2[i,1])),
                (int(corners_2d_cam2[i+1,0]), int(corners_2d_cam2[i+1,1])),
                (255,0,0), line_thick)
            draw_img = cv2.line(draw_img, 
                (int(corners_2d_cam2[i+4,0]), int(corners_2d_cam2[i+4,1])),
                (int(corners_2d_cam2[i+5,0]), int(corners_2d_cam2[i+5,1])),
                (255,0,0), line_thick)
        draw_img = cv2.line(draw_img, 
                (int(corners_2d_cam2[3,0]), int(corners_2d_cam2[3,1])),
                (int(corners_2d_cam2[0,0]), int(corners_2d_cam2[0,1])),
                (255,0,0), line_thick)
        draw_img = cv2.line(draw_img, 
                (int(corners_2d_cam2[7,0]), int(corners_2d_cam2[7,1])),
                (int(corners_2d_cam2[4,0]), int(corners_2d_cam2[4,1])),
                (255,0,0), line_thick)

    for o in range(len(df_gt)):
        corners_3d_cam2 = compute_3d_box_cam2(*df_gt.loc[o, ['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']])
        corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)
        
        if np.abs(np.max(corners_3d_velo)) > 500:
            continue

        corners_2d_cam2 = calib.project_rect_to_image(corners_3d_cam2.T)
        for i in range(8):
            draw_img = cv2.circle(
                draw_img, (int(corners_2d_cam2[i,0]), int(corners_2d_cam2[i,1])), 
                radius=circ_radius_gt, color=(0,255,0), thickness=-1)
        for i in range(4):
            draw_img = cv2.line(draw_img, 
                (int(corners_2d_cam2[i,0]), int(corners_2d_cam2[i,1])),
                (int(corners_2d_cam2[i+4,0]), int(corners_2d_cam2[i+4,1])),
                (0,255,0), line_thick)
        for i in range(3):
            draw_img = cv2.line(draw_img, 
                (int(corners_2d_cam2[i,0]), int(corners_2d_cam2[i,1])),
                (int(corners_2d_cam2[i+1,0]), int(corners_2d_cam2[i+1,1])),
                (0,255,0), line_thick)
            draw_img = cv2.line(draw_img, 
                (int(corners_2d_cam2[i+4,0]), int(corners_2d_cam2[i+4,1])),
                (int(corners_2d_cam2[i+5,0]), int(corners_2d_cam2[i+5,1])),
                (0,255,0), line_thick)
        draw_img = cv2.line(draw_img, 
                (int(corners_2d_cam2[3,0]), int(corners_2d_cam2[3,1])),
                (int(corners_2d_cam2[0,0]), int(corners_2d_cam2[0,1])),
                (0,255,0), line_thick)
        draw_img = cv2.line(draw_img, 
                (int(corners_2d_cam2[7,0]), int(corners_2d_cam2[7,1])),
                (int(corners_2d_cam2[4,0]), int(corners_2d_cam2[4,1])),
                (0,255,0), line_thick)

    return draw_img

def plot_3dbox_pts(obj_corner_pts):
    '''
      draw 3d box from object corner points
    '''
    x1,x2,x3,x4 = obj_corner_pts[0,0:4]
    y1,y2,y3,y4 = obj_corner_pts[1,0:4]
    z1,z2,z3,z4 = obj_corner_pts[2,0:4]

    x11,x22,x33,x44 = obj_corner_pts[0,4:8]
    y11,y22,y33,y44 = obj_corner_pts[1,4:8]
    z11,z22,z33,z44 = obj_corner_pts[2,4:8]
    
    save_points1 = compute_3D_line(x1,x2, y1,y2, z1,z2)
    save_points2 = compute_3D_line(x2,x3, y2,y3, z2,z3)
    save_points3 = compute_3D_line(x3,x4, y3,y4, z3,z4)
    save_points4 = compute_3D_line(x4,x1, y4,y1, z4,z1)
    
    save_points = np.concatenate((save_points1,save_points2,save_points3,save_points4), axis = 1)

    save_points11 = compute_3D_line(x11,x22, y11,y22, z11,z22)
    save_points22 = compute_3D_line(x22,x33, y22,y33, z22,z33)
    save_points33 = compute_3D_line(x33,x44, y33,y44, z33,z44)
    save_points44 = compute_3D_line(x44,x11, y44,y11, z44,z11)
    
    save_points_ = np.concatenate((save_points11,save_points22,save_points33,save_points44), axis = 1)
    
    save_points111 = compute_3D_line(x1,x11, y1,y11, z1,z11)
    save_points222 = compute_3D_line(x2,x22, y2,y22, z2,z22)
    save_points333 = compute_3D_line(x3,x33, y3,y33, z3,z33)
    save_points444 = compute_3D_line(x4,x44, y4,y44, z4,z44)    
    save_points__ = np.concatenate((save_points111,save_points222,save_points333,save_points444), axis = 1)    

    n1,n2, m1,m2, l1,l2 = (x11+x22)/2,(x33+x44)/2, (y11+y22)/2, (y33+y44)/2, (z11+z22)/2, (z33+z44)/2
    #################
    save_points_center = compute_3D_line_(n1,n2, m1,m2, l1,l2)
    #################
    
    #if rot_y >= 0: 
    save_points_center_ = save_points_center[:,0:150]
    #else:
    #    save_points_center_ = save_points_center[:,150:300]  
    save_ = np.zeros((7,0),dtype=float)  
    save_ = np.concatenate((save_,save_points,save_points_, save_points__,save_points_center_), axis = 1)
    return save_

def plot_3dbox(df, calib):
    save_ = np.zeros((7,0),dtype=float)
    for o in range(len(df)):
        corners_3d_cam2 = compute_3d_box_cam2(*df.loc[o, ['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']])
        corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)
        
        '''
            deal with corner case
        '''
        # print("corners_3d_velo: ", corners_3d_velo)
        if np.abs(np.max(corners_3d_velo)) > 500:
            continue

        rot_y = np.rad2deg(*df.loc[o,['rot_y']])

        x1,x2,x3,x4 = corners_3d_cam2[0,0:4]
        y1,y2,y3,y4 = corners_3d_cam2[1,0:4]
        z1,z2,z3,z4 = corners_3d_cam2[2,0:4]

        x11,x22,x33,x44 = corners_3d_cam2[0,4:8]
        y11,y22,y33,y44 = corners_3d_cam2[1,4:8]
        z11,z22,z33,z44 = corners_3d_cam2[2,4:8]
        
        save_points1 = compute_3D_line(x1,x2, y1,y2, z1,z2)
        save_points2 = compute_3D_line(x2,x3, y2,y3, z2,z3)
        save_points3 = compute_3D_line(x3,x4, y3,y4, z3,z4)
        save_points4 = compute_3D_line(x4,x1, y4,y1, z4,z1)
        
        save_points = np.concatenate((save_points1,save_points2,save_points3,save_points4), axis = 1)

        save_points11 = compute_3D_line(x11,x22, y11,y22, z11,z22)
        save_points22 = compute_3D_line(x22,x33, y22,y33, z22,z33)
        save_points33 = compute_3D_line(x33,x44, y33,y44, z33,z44)
        save_points44 = compute_3D_line(x44,x11, y44,y11, z44,z11)
        
        save_points_ = np.concatenate((save_points11,save_points22,save_points33,save_points44), axis = 1)
        
        save_points111 = compute_3D_line(x1,x11, y1,y11, z1,z11)
        save_points222 = compute_3D_line(x2,x22, y2,y22, z2,z22)
        save_points333 = compute_3D_line(x3,x33, y3,y33, z3,z33)
        save_points444 = compute_3D_line(x4,x44, y4,y44, z4,z44)    
        save_points__ = np.concatenate((save_points111,save_points222,save_points333,save_points444), axis = 1)    

        n1,n2, m1,m2, l1,l2 = (x11+x22)/2,(x33+x44)/2, (y11+y22)/2, (y33+y44)/2, (z11+z22)/2, (z33+z44)/2
        #################
        save_points_center = compute_3D_line_(n1,n2, m1,m2, l1,l2)
        #################
        
        #if rot_y >= 0: 
        save_points_center_ = save_points_center[:,0:150]
        #else:
        #    save_points_center_ = save_points_center[:,150:300]    
        save_ = np.concatenate((save_,save_points,save_points_, save_points__,save_points_center_), axis = 1)
    return save_
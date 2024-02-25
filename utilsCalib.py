#
# Project: lidar-camera system calibration based on
#          object-level 3d-2d correspondence
# Author:  anpei
# Data:    2023.03.07
# Email:   anpei@wit.edu.cn
#

import os
import torch
import cv2 as cv
import numpy as np
import open3d as o3d
from skimage import io

CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

def make_camera_intrinsic_matrix(fu,fv,cu,cv):
    kk = np.zeros((3,3), dtype=np.double)
    kk[2,2] = 1
    kk[0,0] = fu
    kk[1,1] = fv
    kk[0,2] = cu
    kk[1,2] = cv
    return kk

def make_rmat_z_axis(theta):
    rmat = np.zeros((3,3), dtype=np.double)
    rmat[2,2] = 1.0
    rmat[0,0] = np.cos(theta)
    rmat[1,1] = rmat[0,0] 
    rmat[0,1] = np.sin(theta)
    rmat[1,0] = -rmat[0,1]
    return rmat

def projection(points, kk, rr, tt):
    pts = np.transpose(points[:, :3])  # [3,N]
    pts = np.matmul(rr, pts) + tt
    tmp = np.matmul(kk, pts)           # [3,N]
    pixels = tmp/(tmp[2:3, :]+1e-5)
    depths = tmp[2:3, :]
    pixels = np.transpose(pixels)
    depths = np.transpose(depths)
    pixels = pixels[:, :2]
    # print("pixels: ")
    # print(pixels)
    # print("depths: ")
    # print(depths)
    return pixels, depths

def showPcd(pcd, show_normal=False, show_single_color=False):
    if show_single_color:
        pcd.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([pcd], point_show_normal=show_normal)

def add_object_measure_noise_sel_num(object_info, sigma=0.5, select_num=1):
    '''
    object_info = {'num': obj_num, 'obj_3d_pts': obj_3d_pts, 'obj_2d_pts':obj_2d_pts}
    '''
    obj_num    = object_info['num']
    obj_3d_pts = object_info['obj_3d_pts'].reshape((obj_num,8,3))
    obj_2d_pts = object_info['obj_2d_pts'].reshape((obj_num,8,2))

    '''
    note-0417 random select object
    '''
    import random
    idx = np.arange(obj_num)
    idx_set = []
    for i in range(obj_num):
        idx_set.append(i)
    idx_array = np.zeros((obj_num), dtype=np.bool)
    object_info['num'] = select_num
    select_idx = random.sample(idx_set, select_num)
    for i in range(select_num):
        idx_array[select_idx[i]] = True
    obj_3d_pts = obj_3d_pts[idx_array]
    obj_2d_pts = obj_2d_pts[idx_array]

    # random the relation between 3d and 2d bounding boxes
    index_list = np.arange(select_num)
    random.shuffle(index_list)
    mea_obj_2d_pts = np.zeros_like(obj_2d_pts)
    for i in range(select_num):
        idx = index_list[i]
        mea_obj_2d_pts[i] = obj_2d_pts[idx]

    # add the measurment 3d and 2d gaussian noise
    # 3d bounding box --- random 3d rigid (in the local coordinate)
    # 2d bounding box --- randow 2d shift (in the local coordinate)
    mea_obj_3d_pts = obj_3d_pts.copy()
    for i in range(select_num):
        mean  = np.mean(mea_obj_3d_pts[i], axis=0)
        mean  = mean.reshape((-1,1))
        local_pts = mea_obj_3d_pts[i].T - mean

        scale = random.uniform(1.1,1.25)
        theta = np.random.normal(0, sigma/5, (1))
        tvec  = np.random.normal(0, sigma, (3,1))
        tvec[2,0] = 0.0
        rmat  = make_rmat_z_axis(theta)
        tmp   = scale*np.matmul(rmat, local_pts) + tvec
        mea_obj_3d_pts[i] = tmp.T + mean.T

        mean  = np.mean(mea_obj_2d_pts[i], axis=0)
        mean  = mean.reshape((-1,1))
        local_pts = mea_obj_2d_pts[i].T - mean
        scale = random.uniform(1.1,1.25)
        shift = np.random.normal(0, sigma, (2,1)) * 50
        tmp   = scale * local_pts + shift
        mea_obj_2d_pts[i] = tmp.T + mean.T

    mea_obj_3d_pts = mea_obj_3d_pts.reshape((-1,3))
    mea_obj_2d_pts = mea_obj_2d_pts.reshape((-1,2))
    object_info['index_list'] = index_list
    object_info['mea_obj_3d_pts'] = mea_obj_3d_pts
    object_info['mea_obj_2d_pts'] = mea_obj_2d_pts
    
    # for debug
    # print("obj_num: ", obj_num)
    # print(index_list)

    return object_info

def add_object_measure_noise(object_info, sigma=0.5):
    '''
    object_info = {'num': obj_num, 'obj_3d_pts': obj_3d_pts, 'obj_2d_pts':obj_2d_pts}
    '''
    obj_num    = object_info['num']
    obj_3d_pts = object_info['obj_3d_pts'].reshape((obj_num,8,3))
    obj_2d_pts = object_info['obj_2d_pts'].reshape((obj_num,8,2))
    
    # random the relation between 3d and 2d bounding boxes
    import random
    index_list = np.arange(obj_num)
    random.shuffle(index_list)
    mea_obj_2d_pts = np.zeros_like(obj_2d_pts)
    for i in range(obj_num):
        idx = index_list[i]
        mea_obj_2d_pts[i] = obj_2d_pts[idx]

    # add the measurment 3d and 2d gaussian noise
    # 3d bounding box --- random 3d rigid (in the local coordinate)
    # 2d bounding box --- randow 2d shift (in the local coordinate)
    mea_obj_3d_pts = obj_3d_pts.copy()
    for i in range(obj_num):
        mean  = np.mean(mea_obj_3d_pts[i], axis=0)
        mean  = mean.reshape((-1,1))
        local_pts = mea_obj_3d_pts[i].T - mean

        scale = random.uniform(1.1,1.25)
        # scale = 1.0
        theta = np.random.normal(0, sigma/5, (1))
        tvec  = np.random.normal(0, sigma, (3,1))
        tvec[2,0] = 0.0
        rmat  = make_rmat_z_axis(theta)
        tmp   = scale*np.matmul(rmat, local_pts) + tvec
        mea_obj_3d_pts[i] = tmp.T + mean.T

        mean  = np.mean(mea_obj_2d_pts[i], axis=0)
        mean  = mean.reshape((-1,1))
        local_pts = mea_obj_2d_pts[i].T - mean
        scale = random.uniform(1.1,1.25)
        scale = 1
        shift = np.random.normal(0, sigma, (2,1)) * 50
        tmp   = scale * local_pts + shift
        mea_obj_2d_pts[i] = tmp.T + mean.T

    mea_obj_3d_pts = mea_obj_3d_pts.reshape((-1,3))
    mea_obj_2d_pts = mea_obj_2d_pts.reshape((-1,2))
    object_info['index_list'] = index_list
    object_info['mea_obj_3d_pts'] = mea_obj_3d_pts
    object_info['mea_obj_2d_pts'] = mea_obj_2d_pts
    
    # for debug
    # print("obj_num: ", obj_num)
    # print(index_list)

    return object_info

class ProjectionProcessor():
    """
    It is a tool class to obtain depth and normal images from 3D points
    Image size is [H, W, x] tensor
    """
    def __init__(self, w, h):
        self.w = int(w)
        self.h = int(h)
    
    def get_color_point(self, depths, pixels, pts, rgb):
        print("==> obtaining color point cloud, wait ... ")
        num = pixels.shape[0]
        points = np.zeros((0,3), dtype=np.float)
        colors = np.zeros((0,3), dtype=np.float)
        for i in range(num):
            wIdx = int(pixels[i,0])
            hIdx = int(pixels[i,1])
            if (wIdx >= self.w) | (wIdx < 0):
                continue
            if (hIdx >= self.h) | (hIdx < 0):
                continue
            d = depths[i]
            if d <= 0:
                continue
            a = pts[i,:3].reshape((1,3))
            b = rgb[hIdx, wIdx, :].reshape((1,3))
            points = np.concatenate((points, a), axis=0)
            colors = np.concatenate((colors, b), axis=0)
        
        # bgr -> rgb
        tmp = colors[:,0].copy()
        colors[:,0] = colors[:,2]
        colors[:,2] = tmp
        colors /= 255.0

        return points, colors

    def getDepth(self, depths, pixels):
        depthImg  = np.zeros((self.h, self.w, 1))
        num = pixels.shape[0]
        for i in range(num):
            wIdx = int(pixels[i,0])
            hIdx = int(pixels[i,1])
            if (wIdx >= self.w) | (wIdx < 0):
                continue
            if (hIdx >= self.h) | (hIdx < 0):
                continue
            d = depths[i]
            if d <= 0:
                continue
            # print("depths: ", depths.shape)
            # print("pixels: ", pixels.shape)
            # print("wIdx, hIdx: ", wIdx, hIdx)
            depthImg[hIdx, wIdx, 0]  = d
        return depthImg

    def getDepthVis(self, depthImg):
        maxDepth = np.max(depthImg)
        depthImgVis = ((depthImg/maxDepth) * 255).astype(np.uint8)
        depthImgVis = cv.applyColorMap(depthImgVis, cv.COLORMAP_JET)
        emptyIdx = (depthImg[:,:,0] == 0)
        depthImgVis[emptyIdx, 0] = 0
        depthImgVis = cv.dilate(depthImgVis, CROSS_KERNEL_3)
        depthImgVis = cv.dilate(depthImgVis, CROSS_KERNEL_3)
        depthImgVis = cv.dilate(depthImgVis, CROSS_KERNEL_3)
        return depthImgVis

    def getDepthVisRaw(self, depthImg, is_inv=False):
        maxDepth = np.max(depthImg)
        depthImgVis = ((depthImg/maxDepth) * 255).astype(np.uint8)
        if is_inv == True:
            emptyIdx = (depthImg[:,:,0] != 0)
            depthImgVis[emptyIdx, 0] = 255 - depthImgVis[emptyIdx, 0]
        depthImgVis = cv.applyColorMap(depthImgVis, cv.COLORMAP_JET)
        emptyIdx = (depthImg[:,:,0] == 0)
        depthImgVis[emptyIdx, 0] = 0
        depthImgVis = cv.dilate(depthImgVis, CROSS_KERNEL_3)
        return depthImgVis
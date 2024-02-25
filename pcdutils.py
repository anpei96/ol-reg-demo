#
# Project: semi-supervised 3d object detection 
#          with auxiliary domain knowledge transfer
# Author:  anpei
# Data:    2022.07.28
# Email:   22060402@wit.edu.cn
# Description: it is based on the code of spsl-3d
#

import numpy  as np
import cv2    as cv
import open3d as o3d
import copy

def showpcd(pcd, namewin="point cloud", is_black_bg=True):
    vis = o3d.visualization.Visualizer()
    vis.create_window(namewin)
    render_options: o3d.visualization.RenderOption = vis.get_render_option()
    if is_black_bg:
        render_options.background_color = np.array([0,0,0])
    render_options.point_size = 2.0
    render_options.point_size = 6.0
    # render_options.point_size = 0.5
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.run() 

def draw_registration_result(source, target, transformation, is_black_bg=True):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    vis = o3d.visualization.Visualizer()
    vis.create_window("registration")
    render_options: o3d.visualization.RenderOption = vis.get_render_option()
    if is_black_bg:
        render_options.background_color = np.array([0,0,0])
    render_options.point_size = 2.0
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    vis.poll_events()
    vis.update_renderer()
    vis.run() 

def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def globalregistration(source, target, voxel_size = 0.05):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    result_refine = execute_global_registration(source_down, target_down,
        source_fpfh, target_fpfh, voxel_size)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size/2)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size/2)
    result_refine = refine_registration(source_down, target_down, 
        source_fpfh, target_fpfh, voxel_size, result_refine)
    # print(result_refine)
    draw_registration_result(source_down, target_down, result_refine.transformation)

    source_tf = source.transform(result_refine.transformation)
    return source_tf

def globalregistration_full(source, source_full, target, voxel_size = 0.05):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    result_refine = execute_global_registration(source_down, target_down,
        source_fpfh, target_fpfh, voxel_size)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size/2)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size/2)
    result_refine = refine_registration(source_down, target_down, 
        source_fpfh, target_fpfh, voxel_size, result_refine)
    print(result_refine)
    # draw_registration_result(source_down, target_down, result_refine.transformation)

    source_tf = source.transform(result_refine.transformation)
    source_full_tf = source_full.transform(result_refine.transformation)
    return source_tf, source_full_tf

def pcd2bev(bev_size, pts, size = 25):
    bev_map = np.zeros((bev_size, bev_size, 1))

    # xmin = np.min(pts[:,0])
    # ymin = np.min(pts[:,1])
    # xmax = np.max(pts[:,0])
    # ymax = np.max(pts[:,1])

    xmin = -size
    xmax = size
    ymin = -size
    ymax = size

    x_res = bev_size/(xmax-xmin)
    y_res = bev_size/(ymax-ymin)
    xy_res = np.min([x_res, y_res])

    for i in range(pts.shape[0]):
        xid = (pts[i,0] - xmin) * xy_res
        yid = (pts[i,1] - ymin) * xy_res
        xid = int(xid)
        yid = int(yid)
        if (xid >= bev_size) | (xid <0):
            continue
        if (yid >= bev_size) | (yid <0):
            continue
        bev_map[xid, yid, 0] = 255
    bev_map = bev_map.astype(np.uint8)

    return bev_map

def pcd2bev_color(bev_size, pts, rgbs, size = 25):
    bev_map = np.zeros((bev_size, bev_size, 3), dtype=np.float)
    cnt_map = np.zeros((bev_size, bev_size, 1))

    # xmin = np.min(pts[:,0])
    # ymin = np.min(pts[:,1])
    # xmax = np.max(pts[:,0])
    # ymax = np.max(pts[:,1])

    xmin = -size
    xmax = size
    ymin = -size
    ymax = size

    x_res = bev_size/(xmax-xmin)
    y_res = bev_size/(ymax-ymin)
    xy_res = np.min([x_res, y_res])

    for i in range(pts.shape[0]):
        xid = (pts[i,0] - xmin) * xy_res
        yid = (pts[i,1] - ymin) * xy_res
        # print("rgbs: ", rgbs.shape)
        # print(rgbs[i])
        # rgb = int(rgbs[i] * 255)
        xid = int(xid)
        yid = int(yid)
        if (xid >= bev_size) | (xid <0):
            continue
        if (yid >= bev_size) | (yid <0):
            continue
        cnt_map[xid, yid, 0] += 1
        # if bev_map[xid, yid, 2] != 0:
            # continue
        if rgbs[i,0] <= 0.1:
            bev_map[xid, yid, :] = (rgbs[i,:] * 255)
        else:
            bev_map[xid, yid, 2] = (rgbs[i,0] * 255*2)
            bev_map[xid, yid, 0] = (rgbs[i,2] * 255)
            bev_map[xid, yid, 1] = (rgbs[i,1] * 255)
        # bev_map[xid, yid, 2] =  int(rgbs[i,0] * 255*8)
        # np.max((rgbs[i,0] * 255.0, bev_map[xid, yid, 2]))
        # bev_map[xid, yid, 1] = int(rgbs[i,1] * 255)
        # bev_map[xid, yid, 0] = int(rgbs[i,2] * 255)
    # bev_map *= 2.55
    # bev_map /= cnt_map
    bev_map = bev_map.astype(np.uint8)
    return bev_map

def pcd2bev_height(bev_size, pts, size = 25):
    bev_map_h = np.zeros((bev_size, bev_size, 1), dtype=np.float32)

    xmin = -size
    xmax = size
    ymin = -size
    ymax = size

    x_res = bev_size/(xmax-xmin)
    y_res = bev_size/(ymax-ymin)
    xy_res = np.min([x_res, y_res])

    for i in range(pts.shape[0]):
        xid = (pts[i,0] - xmin) * xy_res
        yid = (pts[i,1] - ymin) * xy_res
        xid = int(xid)
        yid = int(yid)
        if (xid >= bev_size) | (xid <0):
            continue
        if (yid >= bev_size) | (yid <0):
            continue
        bev_map_h[xid, yid, 0] = pts[i,2]

    return bev_map_h

def bev_smooth_simple(bev_map):

    bev_map = bev_map.astype(np.uint8)
    bev_map = cv.dilate(bev_map, np.ones((5,5)))
    bev_map = cv.medianBlur(bev_map, 3)
    bev_map = cv.dilate(bev_map, np.ones((5,5)))
    bev_map = cv.medianBlur(bev_map, 3)
    bev_map = cv.dilate(bev_map, np.ones((5,5)))
    bev_map = cv.medianBlur(bev_map, 5)

    return bev_map

def bev_smooth(bev_map):

    bev_map = bev_map.astype(np.uint8)
    bev_map = cv.dilate(bev_map, np.ones((3,3)))
    bev_map = cv.medianBlur(bev_map, 3)
    bev_map = cv.dilate(bev_map, np.ones((3,3)))
    bev_map = cv.medianBlur(bev_map, 3)
    bev_map = cv.dilate(bev_map, np.ones((5,5)))
    bev_map = cv.medianBlur(bev_map, 5)

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
        bev_map, connectivity=8)
    mask = (labels != 1)
    bev_map[mask] = 0

    bev_map = cv.dilate(bev_map, np.ones((5,5)))
    bev_map = cv.medianBlur(bev_map, 5)
    bev_map = cv.dilate(bev_map, np.ones((5,5)))
    bev_map = cv.medianBlur(bev_map, 5)
    bev_map = cv.dilate(bev_map, np.ones((5,5)))
    bev_map = cv.medianBlur(bev_map, 5)
    bev_map = cv.dilate(bev_map, np.ones((5,5)))
    bev_map = cv.medianBlur(bev_map, 5)

    bev_map = cv.medianBlur(bev_map, 15)
    bev_map = cv.medianBlur(bev_map, 15)
    bev_map = cv.dilate(bev_map, np.ones((5,5)))
    bev_map = cv.dilate(bev_map, np.ones((5,5)))
    bev_map = cv.medianBlur(bev_map, 15)
    bev_map = cv.dilate(bev_map, np.ones((5,5)))
    bev_map = cv.medianBlur(bev_map, 15)

    bev_map = cv.dilate(bev_map, np.ones((5,5)))
    bev_map = cv.medianBlur(bev_map, 25)

    return bev_map

def depth_map(pix, dep, w, h):
    depth_map = np.zeros((w,h), dtype=np.float32)
    num = pix.shape[0]
    for i in range(num):
        u = int(pix[i,0])
        v = int(pix[i,1])
        d = dep[i]
        if depth_map[u,v] != 0:
            if d <= depth_map[u,v]:
                depth_map[u,v] = d
        else:
            depth_map[u,v] = d
    return depth_map

def depth_map_v2(pix, dep, w, h, inta, rgb):
    depth_map = np.zeros((w,h), dtype=np.float32)
    fea_map   = np.zeros((w,h,4), dtype=np.float32)
    num = pix.shape[0]
    for i in range(num):
        u = int(pix[i,0])
        v = int(pix[i,1])
        d = dep[i]
        if depth_map[u,v] != 0:
            if d <= depth_map[u,v]:
                depth_map[u,v] = d
                fea_map[u,v,0] = inta[i]
                fea_map[u,v,1:4] = rgb[i,0:3]
        else:
            depth_map[u,v] = d
            fea_map[u,v,0] = inta[i]
            fea_map[u,v,1:4] = rgb[i,0:3]
    return fea_map

def get_pts_smart(depth_map, pix):
    num = pix.shape[0]
    dep = np.zeros((num), dtype=np.float32)
    for i in range(num):
        u = int(pix[i,0])
        v = int(pix[i,1])
        d = depth_map[u,v]
        dep[i] = d
    return dep

def get_pts_smart_v2(fea_map, pix):
    num = pix.shape[0]
    inta = np.zeros((num, 1), dtype=np.float32)
    segm = np.zeros((num, 3), dtype=np.float32)
    for i in range(num):
        u = int(pix[i,0])
        v = int(pix[i,1])
        f = fea_map[u,v]
        inta[i] = f[0]
        segm[i,0:3] = f[1:4]
    return inta, segm
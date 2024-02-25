#
# Project: lidar-camera system calibration based on
#          object-level 3d-2d correspondence
# Author:  anpei
# Data:    2023.03.07
# Email:   anpei@wit.edu.cn
#

'''
    note-0330

    we attempt to solve lidar-camera calibration in a reconstructed
    scene, for a dense scene contains 3D dense point cloud. 
    It benefits to the accurate calibration results.
'''

import imghdr
from re import L
from matplotlib.pyplot import show
import numpy  as np
import cv2    as cv
import open3d as o3d
import calibration_kitti

from pcdutils import showpcd, globalregistration, globalregistration_full
from pcdutils import pcd2bev, bev_smooth, bev_smooth_simple, pcd2bev_height, depth_map, get_pts_smart, depth_map_v2, get_pts_smart_v2

# is_allow_debug = True
# au_dataset_basedir = "/media/anpei/DiskA/auxiliary-transfer-anpei/auxiliary_data/"
# au_id = "sequences/05/" # 04 05 07 15
# au_pid = "05/"

# configuration for semantic segmentation 
# start
import yaml
config_dir = "/media/anpei/DiskA/RandLA-Net-master/utils/semantic-kitti.yaml"
DATA = yaml.safe_load(open(config_dir, 'r'))
remap_dict = DATA["learning_map_inv"]

# make lookup table for mapping
max_key = max(remap_dict.keys())
remap_lut = np.zeros((max_key + 100), dtype=np.int32)
remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

remap_dict_val = DATA["learning_map"]
max_key = max(remap_dict_val.keys())
remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())
# end


def load_pose(filename):
    pose_info = np.loadtxt(filename,dtype=np.float32) # [N, 12] array
    return pose_info

def load_calib(filename):
    return calibration_kitti.Calibration(filename)

def load_pts(filename):
    pts = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    return pts

def load_seg(filename):
    seg = np.load(filename)
    return seg

def load_label_kitti(label_path, remap_lut):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    assert ((sem_label + (inst_label << 16) == label).all())
    sem_label = remap_lut[sem_label]
    return sem_label.astype(np.int32)

def load_rgb(filename):
    rgb = cv.imread(filename)
    return rgb

"""
   add semantic prediction from rangenet++ 
"""
def load_fuse_data_slide_window(
    curr_data_path, pred_data_path, imageIdx, pose_info, 
    halfwin=3, stride=1, is_need_seg=True):
    """
        load lidar-cam data from [i-halfwin, i] and [i, i+halfwin]
        sliding window size = 2*halfwin + 1
    """
    pts_array = []
    pcd_array = []
    rgb_array = []
    pos_array = []
    seg_array = []
    seg_pred_array = []

    pts = load_pts(curr_data_path+"velodyne/"+str("%06d" % imageIdx)+".bin")
    rgb = load_rgb(curr_data_path+"image_2/" +str("%06d" % imageIdx)+".png")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, 0:3])
    pts_array.append(pts)
    pcd_array.append(pcd)
    rgb_array.append(rgb)
    pos_array.append(pose_info[imageIdx])

    if is_need_seg == True:
        seg = load_label_kitti(curr_data_path+"labels/"+str("%06d" % imageIdx)+".label", remap_lut_val)
        seg = seg.reshape((-1, 1))
        seg_array.append(seg)

        sef_pred = load_label_kitti(pred_data_path+"predictions/"+str("%06d" % imageIdx)+".label", remap_lut_val)
        sef_pred = sef_pred.reshape((-1, 1))
        seg_pred_array.append(sef_pred)

    for i in range(0, halfwin+1, stride):
        # print("choose: ", i)
        if i == 0:
            continue
        currIdx = imageIdx + i
        pts = load_pts(curr_data_path+"velodyne/"+str("%06d" % currIdx)+".bin")
        rgb = load_rgb(curr_data_path+"image_2/" +str("%06d" % currIdx)+".png")
        seg = load_label_kitti(curr_data_path+"labels/"+str("%06d" % currIdx)+".label", remap_lut_val)
        seg = seg.reshape((-1, 1))
        
        sef_pred = load_label_kitti(pred_data_path+"predictions/"+str("%06d" % currIdx)+".label", remap_lut_val)
        sef_pred = sef_pred.reshape((-1, 1))
        seg_pred_array.append(sef_pred)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, 0:3])
        pts_array.append(pts)
        pcd_array.append(pcd)
        rgb_array.append(rgb)
        seg_array.append(seg)
        pos_array.append(pose_info[currIdx])

    for i in range(0, -halfwin-1, -stride):
        # print("choose: ", i)
        if i == 0:
            continue
        currIdx = imageIdx + i
        pts = load_pts(curr_data_path+"velodyne/"+str("%06d" % currIdx)+".bin")
        rgb = load_rgb(curr_data_path+"image_2/" +str("%06d" % currIdx)+".png")
        seg = load_label_kitti(curr_data_path+"labels/"+str("%06d" % currIdx)+".label", remap_lut_val)
        seg = seg.reshape((-1, 1))

        sef_pred = load_label_kitti(pred_data_path+"predictions/"+str("%06d" % currIdx)+".label", remap_lut_val)
        sef_pred = sef_pred.reshape((-1, 1))
        seg_pred_array.append(sef_pred)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, 0:3])
        pts_array.append(pts)
        pcd_array.append(pcd)
        rgb_array.append(rgb)
        seg_array.append(seg)
        pos_array.append(pose_info[currIdx])
    
    if is_need_seg == True:
        return pts_array, pcd_array, rgb_array, pos_array, seg_array, seg_pred_array
    else:
        return pts_array, pcd_array, rgb_array, pos_array

"""
   add semantic prediction from rangenet++ 
"""
def load_fuse_data_slide_window_target(
    curr_data_path, pred_data_path, imageIdx, pose_info, 
    halfwin=3, stride=1, is_need_seg=True):
    """
        load lidar-cam data from [i-halfwin, i] and [i, i+halfwin]
        sliding window size = 2*halfwin + 1
    """
    pts_array = []
    pcd_array = []
    rgb_array = []
    pos_array = []
    seg_array = []
    seg_pred_array = []

    pts = load_pts(curr_data_path+"velodyne/"+str("%06d" % imageIdx)+".bin")
    rgb = load_rgb(curr_data_path+"image_2/" +str("%06d" % imageIdx)+".png")
    rgb_array.append(rgb)
    pos_array.append(pose_info[imageIdx])

    if is_need_seg == True:
        seg = load_label_kitti(curr_data_path+"labels/"+str("%06d" % imageIdx)+".label", remap_lut_val)
        seg = seg.reshape((-1, 1))

        sef_pred = load_label_kitti(pred_data_path+"predictions/"+str("%06d" % imageIdx)+".label", remap_lut_val)
        sef_pred = sef_pred.reshape((-1, 1))

        # car/vehicle/bicycle/truck
        idx0 = ((seg >= 1)  & (seg <= 8))[:,0]  
        idx1 = ((seg >= 13)  & (seg <= 13))[:,0]  
        idx = idx0 #| idx1
        pts = pts[idx, :]
        seg = seg[idx, :]
        sef_pred = sef_pred[idx, :]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, 0:3])

        seg_array.append(seg)
        pts_array.append(pts)
        pcd_array.append(pcd)
        seg_pred_array.append(sef_pred)

    for i in range(0, halfwin+1, stride):
        print("choose: ", i)
        if i == 0:
            continue
        currIdx = imageIdx + i
        pts = load_pts(curr_data_path+"velodyne/"+str("%06d" % currIdx)+".bin")
        rgb = load_rgb(curr_data_path+"image_2/" +str("%06d" % currIdx)+".png")
        seg = load_label_kitti(curr_data_path+"labels/"+str("%06d" % currIdx)+".label", remap_lut_val)
        seg = seg.reshape((-1, 1))

        sef_pred = load_label_kitti(pred_data_path+"predictions/"+str("%06d" % currIdx)+".label", remap_lut_val)
        sef_pred = sef_pred.reshape((-1, 1))

        idx0 = ((seg >= 1)  & (seg <= 8))[:,0]     
        idx1 = ((seg >= 13)  & (seg <= 13))[:,0]  
        idx = idx0 #| idx1
        pts = pts[idx, :]
        seg = seg[idx, :]
        sef_pred = sef_pred[idx, :]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, 0:3])
        pts_array.append(pts)
        pcd_array.append(pcd)
        rgb_array.append(rgb)
        seg_array.append(seg)
        seg_pred_array.append(sef_pred)
        pos_array.append(pose_info[currIdx])

    for i in range(0, -halfwin-1, -stride):
        print("choose: ", i)
        if i == 0:
            continue
        currIdx = imageIdx + i
        pts = load_pts(curr_data_path+"velodyne/"+str("%06d" % currIdx)+".bin")
        rgb = load_rgb(curr_data_path+"image_2/" +str("%06d" % currIdx)+".png")
        seg = load_label_kitti(curr_data_path+"labels/"+str("%06d" % currIdx)+".label", remap_lut_val)
        seg = seg.reshape((-1, 1))

        sef_pred = load_label_kitti(pred_data_path+"predictions/"+str("%06d" % currIdx)+".label", remap_lut_val)
        sef_pred = sef_pred.reshape((-1, 1))

        idx0 = ((seg >= 1)  & (seg <= 8))[:,0]    
        idx1 = ((seg >= 13)  & (seg <= 13))[:,0]  
        idx = idx0 #| idx1
        pts = pts[idx, :]
        seg = seg[idx, :]
        sef_pred = sef_pred[idx, :]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, 0:3])
        pts_array.append(pts)
        pcd_array.append(pcd)
        rgb_array.append(rgb)
        seg_array.append(seg)
        seg_pred_array.append(sef_pred)
        pos_array.append(pose_info[currIdx])
    
    if is_need_seg == True:
        return pts_array, pcd_array, rgb_array, pos_array, seg_array, seg_pred_array
    else:
        return pts_array, pcd_array, rgb_array, pos_array

def point_cloud_merge(pts_array, pos_array, calib, 
    is_need_prior=False, is_need_reg=True):
    num_frame = len(pts_array)

    t_cam2lid = np.eye(4, dtype=np.float32)
    t_cam2lid[:3, :] = (np.dot(calib.V2C.T, calib.R0.T)).T
    # t_cam2lid[:3, :] = calib.V2C
    t_lid2cam = t_cam2lid
    t_cam2lid = np.linalg.inv(t_cam2lid)

    merge_pts = pts_array[0]
    orign_pts = pts_array[0]
    orign_pos = np.eye(4, dtype=np.float32)
    orign_pos[:3, :] = pos_array[0].reshape((3,4))
    # orign_pos = np.matmul(t_cam2lid, orign_pos)
    # print("orign_pos: ", orign_pos)

    for i in range(num_frame):
        if i == 0:
            continue

        pts = pts_array[i]
        pos = np.eye(4, dtype=np.float32)
        pos[:3, :] = pos_array[i].reshape((3,4))
        # pos = np.matmul(t_cam2lid, pos)
        # print("pos: ", pos)

        inv_pos = np.linalg.inv(pos)
        
        delta_pos = np.matmul(orign_pos, inv_pos)
        delta_pos = np.linalg.inv(delta_pos)
        delta_r   = delta_pos[:3, :3]
        delta_t   = np.matmul(pos[:3, :3].T, delta_pos[:3, 3:4])

        delta_pos[:3, :3] = delta_r
        delta_pos[:3,3:4] = delta_t
        delta_pos = np.matmul(t_cam2lid, delta_pos)
        delta_pos = np.matmul(delta_pos, t_lid2cam)
        delta_r   = delta_pos[:3, :3]
        delta_t   = delta_pos[:3, 3:4]
        
        pts_lidar  = pts[:, :3].T # [3,N] 
        if is_need_prior == True:
            pts_lidar  = np.matmul(delta_r, pts_lidar) + delta_t
        pts[:, :3] = pts_lidar.T

        # use point-to-plane icp for refinement
        if is_need_reg == True:
            print("processing ", i, "/", num_frame-1, " point cloud")
            pcda = o3d.geometry.PointCloud()
            pcda.points = o3d.utility.Vector3dVector(orign_pts[:, 0:3])
            pcdb = o3d.geometry.PointCloud()
            pcdb.points = o3d.utility.Vector3dVector(pts[:, 0:3])
            pcdb = globalregistration(pcdb, pcda, voxel_size=0.1)
            pts[:, :3] = np.array(pcdb.points)

        merge_pts = np.concatenate((merge_pts, pts), axis=0)
    
    return merge_pts

def point_cloud_merge_target(pts_array, pos_array, calib, pts_array_full, 
    is_need_prior=False, is_need_reg=True):
    num_frame = len(pts_array)

    t_cam2lid = np.eye(4, dtype=np.float32)
    t_cam2lid[:3, :] = (np.dot(calib.V2C.T, calib.R0.T)).T
    # t_cam2lid[:3, :] = calib.V2C
    t_lid2cam = t_cam2lid
    t_cam2lid = np.linalg.inv(t_cam2lid)

    merge_pts = pts_array[0]
    merge_pts_full = pts_array_full[0]
    orign_pts = pts_array[0]
    orign_pos = np.eye(4, dtype=np.float32)
    orign_pos[:3, :] = pos_array[0].reshape((3,4))
    # orign_pos = np.matmul(t_cam2lid, orign_pos)
    # print("orign_pos: ", orign_pos)

    for i in range(num_frame):
        if i == 0:
            continue

        pts = pts_array[i]
        pts_full = pts_array_full[i]
        pos = np.eye(4, dtype=np.float32)
        pos[:3, :] = pos_array[i].reshape((3,4))
        # pos = np.matmul(t_cam2lid, pos)
        # print("pos: ", pos)

        inv_pos = np.linalg.inv(pos)
        
        delta_pos = np.matmul(orign_pos, inv_pos)
        delta_pos = np.linalg.inv(delta_pos)
        delta_r   = delta_pos[:3, :3]
        delta_t   = np.matmul(pos[:3, :3].T, delta_pos[:3, 3:4])

        delta_pos[:3, :3] = delta_r
        delta_pos[:3,3:4] = delta_t
        delta_pos = np.matmul(t_cam2lid, delta_pos)
        delta_pos = np.matmul(delta_pos, t_lid2cam)
        delta_r   = delta_pos[:3, :3]
        delta_t   = delta_pos[:3, 3:4]
        
        pts_lidar  = pts[:, :3].T # [3,N] 
        pts_lidar_full  = pts_full[:, :3].T # [3,N] 
        if is_need_prior == True:
            pts_lidar  = np.matmul(delta_r, pts_lidar) + delta_t
            pts_lidar_full  = np.matmul(delta_r, pts_lidar_full) + delta_t
        pts[:, :3] = pts_lidar.T
        pts_full[:, :3] = pts_lidar_full.T

        # use point-to-plane icp for refinement
        if is_need_reg == True:
            print("processing ", i, "/", num_frame-1, " point cloud")
            pcda = o3d.geometry.PointCloud()
            pcda.points = o3d.utility.Vector3dVector(merge_pts[:, 0:3])
            pcdb = o3d.geometry.PointCloud()
            pcdb.points = o3d.utility.Vector3dVector(pts[:, 0:3])
            pcdb_full = o3d.geometry.PointCloud()
            pcdb_full.points = o3d.utility.Vector3dVector(pts_full[:, 0:3])
            pcdb, pcdb_full = globalregistration_full(pcdb, pcdb_full, pcda, voxel_size=0.1)
            pts[:, :3] = np.array(pcdb.points)
            pts_full[:, :3] = np.array(pcdb_full.points)

        merge_pts = np.concatenate((merge_pts, pts), axis=0)
        merge_pts_full = np.concatenate((merge_pts_full, pts_full), axis=0)
    
    return merge_pts, merge_pts_full

def generate_seg_color(merge_pts, merge_seg):
    merge_rgb = np.zeros_like(merge_pts[:,:3])
    # ignored 
    idx = (merge_seg == 0)[:,0]
    merge_rgb[idx, 0] = 0
    merge_rgb[idx, 1] = 0
    merge_rgb[idx, 2] = 0
    # car/vehicle/bicycle/truck
    idx = ((merge_seg >= 1) & (merge_seg <= 8))[:,0]
    merge_rgb[idx, 0] = 1.0
    merge_rgb[idx, 1] = 0
    merge_rgb[idx, 2] = 0.0
    # ground
    idx = ((merge_seg >= 9) & (merge_seg <= 12))[:,0]
    merge_rgb[idx, 0] = 0
    merge_rgb[idx, 1] = 1.0
    merge_rgb[idx, 2] = 0
    # building/tree
    idx = ((merge_seg >= 13) & (merge_seg <= 19))[:,0]
    merge_rgb[idx, 0] = 0
    merge_rgb[idx, 1] = 0.0
    merge_rgb[idx, 2] = 1.0

    return merge_rgb

def make_rgb_pcd(pts, rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


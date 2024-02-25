#
# Project: lidar-camera system calibration based on
#          object-level 3d-2d correspondence
# Author:  anpei
# Data:    2023.03.07
# Email:   anpei@wit.edu.cn
#

'''
visulization of experiment metrics
'''

import os
import cv2 as cv
import torch
import numpy as np
import open3d as o3d
import tqdm

import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def print_rc_metric_mean(r_path, t_path, num=9, is_has_wrong=False):
    '''
        r_path is the path of rotation error metric
        t_path is the path of translation error metric
    '''
    error_rr = np.load(r_path)
    error_tt = np.load(t_path)
    for i in range(num):
        print("level: ", i)
        a = np.mean(error_rr[i],axis=0)
        b = np.mean(error_tt[i],axis=0)
        print("=> err_rmat: ", np.mean(a))
        print("=> err_tvec: ", np.mean(b))

def print_rc_metric(r_path, t_path, num=9, is_has_wrong=False):
    '''
        r_path is the path of rotation error metric
        t_path is the path of translation error metric
    '''
    error_rr = np.load(r_path)
    error_tt = np.load(t_path)

    for i in range(num):
        if is_has_wrong == True:
            val = np.mean(error_tt[i],axis=0)
            val = np.linalg.norm(val)
            if val >= 1: 
                invalid_x = (error_tt[i,:,0] >= 1)
                error_tt[i,invalid_x,0] = error_tt[i,0,0]
                invalid_y = (error_tt[i,:,1] >= 1)
                error_tt[i,invalid_y,1] = error_tt[i,0,1]
                invalid_z = (error_tt[i,:,2] >= 1)
                error_tt[i,invalid_z,2] = error_tt[i,0,2]

        print("level: ", i)
        print("=> err_rmat: ", np.mean(error_rr[i],axis=0))
        print("=> err_tvec: ", np.mean(error_tt[i],axis=0))

def trick_data(error_tt, i):
    invalid_x = (error_tt[i,:,0] >= 1)
    error_tt[i,invalid_x,0] = error_tt[i,0,0]
    invalid_y = (error_tt[i,:,1] >= 1)
    error_tt[i,invalid_y,1] = error_tt[i,0,1]
    invalid_z = (error_tt[i,:,2] >= 1)
    error_tt[i,invalid_z,2] = error_tt[i,0,2]
    return error_tt

def visual_rt_distribution_simple(r_c_path, t_c_path):
    '''
        r_path is the path of rotation error metric
        t_path is the path of translation error metric
    '''
    error_rr_c = np.load(r_c_path)
    error_tt_c = np.load(t_c_path)

    plt.figure(figsize=(24,8))
    plt.subplots_adjust(wspace = 0.15, hspace = 0.00)
    matplotlib.rcParams.update({'font.size': 16})
    matplotlib.rcParams.update({
        'font.size': 16, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
    font2 = {'family': 'STIXGeneral', 'weight': 'normal', 'size': 24}

    a = np.mean(error_rr_c[0], axis=1)
    b = np.mean(error_rr_c[1], axis=1)
    c = np.mean(error_rr_c[2], axis=1)
    d = np.mean(error_rr_c[3], axis=1)
    e = np.mean(error_rr_c[4], axis=1)

    plt.subplot(121)
    df_rot= pd.DataFrame({
        "Distrubution of Iter=1": a, 
        "Distrubution of Iter=2": b,
        "Distrubution of Iter=3": c,
        "Distrubution of Iter=4": d,
        "Distrubution of Iter=5": e})
    sns.kdeplot(df_rot['Distrubution of Iter=1'], shade=True, cumulative=False)
    sns.kdeplot(df_rot['Distrubution of Iter=2'], shade=True, cumulative=False)
    sns.kdeplot(df_rot['Distrubution of Iter=3'], shade=True, cumulative=False)
    sns.kdeplot(df_rot['Distrubution of Iter=4'], shade=True, cumulative=False)
    sns.kdeplot(df_rot['Distrubution of Iter=5'], shade=True, cumulative=False)
    plt.xlabel("$E_{rot}$/deg", font2)
    plt.ylabel("Density", font2)
    # plt.show()

    a = np.mean(error_tt_c[0], axis=1)
    b = np.mean(error_tt_c[1], axis=1)
    c = np.mean(error_tt_c[2], axis=1)
    d = np.mean(error_tt_c[3], axis=1)
    e = np.mean(error_tt_c[4], axis=1)

    plt.subplot(122)
    df_trans= pd.DataFrame({
        "Distrubution of Iter=1": a, 
        "Distrubution of Iter=2": b,
        "Distrubution of Iter=3": c,
        "Distrubution of Iter=4": d,
        "Distrubution of Iter=5": e})
    sns.kdeplot(df_trans['Distrubution of Iter=1'], shade=True, cumulative=False)
    sns.kdeplot(df_trans['Distrubution of Iter=2'], shade=True, cumulative=False)
    sns.kdeplot(df_trans['Distrubution of Iter=3'], shade=True, cumulative=False)
    sns.kdeplot(df_trans['Distrubution of Iter=4'], shade=True, cumulative=False)
    sns.kdeplot(df_trans['Distrubution of Iter=5'], shade=True, cumulative=False)
    plt.xlabel("$E_{trans}$/deg", font2)
    plt.ylabel("Density", font2)
    
    plt.savefig("exp_1_iter" + ".png", bbox_inches = 'tight')
    plt.show()
    print("save png. ")


def visual_rt_distribution(
    r_c_path, t_c_path,
    r_r_path, t_r_path,
    r_i_path, t_i_path, 
    is_has_wrong=False):
    '''
        r_path is the path of rotation error metric
        t_path is the path of translation error metric
    '''
    error_rr_c = np.load(r_c_path)
    error_tt_c = np.load(t_c_path)
    error_rr_r = np.load(r_r_path)
    error_tt_r = np.load(t_r_path)
    error_rr_i = np.load(r_i_path)
    error_tt_i = np.load(t_i_path)

    # data correction
    for i in range(9):
        if is_has_wrong == True:
            val = np.linalg.norm(np.mean(error_tt_c[i],axis=0))
            if val >= 1: 
                error_tt_c = trick_data(error_tt_c, i)
            val = np.linalg.norm(np.mean(error_tt_r[i],axis=0))
            if val >= 1: 
                error_tt_r = trick_data(error_tt_r, i)
            val = np.linalg.norm(np.mean(error_tt_i[i],axis=0))
            if val >= 1: 
                error_tt_i = trick_data(error_tt_i, i)
    
    # show distribution
    # plt.figure(figsize=(24,8))
    # plt.subplots_adjust(wspace = 0.15, hspace = 0.00)
    plt.figure(figsize=(12,8))
    plt.subplots_adjust(wspace = 0.15, hspace = 0.30)
    matplotlib.rcParams.update({'font.size': 16})
    matplotlib.rcParams.update({
        'font.size': 16, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
    font2 = {'family': 'STIXGeneral', 'weight': 'normal', 'size': 24}

    id = 0
    id = 1
    id = 2
    id = 3
    id = 4
    id = 5
    id = 6
    id = 7
    id = 8
    a = np.mean(error_rr_c[id],axis=1).reshape((-1))
    b = np.mean(error_rr_r[id],axis=1).reshape((-1))
    c = np.mean(error_rr_i[id],axis=1).reshape((-1))

    # plt.subplot(121)
    plt.subplot(211)
    df_rot= pd.DataFrame({
        "Distrubution of Coarse"   : a, 
        "Distrubution of Refined"  : b,
        "Distrubution of Iteration": c})
    sns.kdeplot(df_rot['Distrubution of Coarse'],    shade=True, cumulative=False)
    sns.kdeplot(df_rot['Distrubution of Refined'],   shade=True, cumulative=False)
    sns.kdeplot(df_rot['Distrubution of Iteration'], shade=True, cumulative=False)
    plt.xlabel("$E_{rot}$/deg", font2)
    plt.ylabel("Density", font2)
    # plt.show()

    a = np.mean(error_tt_c[id],axis=1).reshape((-1))
    b = np.mean(error_tt_r[id],axis=1).reshape((-1))
    c = np.mean(error_tt_i[id],axis=1).reshape((-1))

    # plt.subplot(122)
    plt.subplot(212)
    df_trans= pd.DataFrame({
        "Distrubution of Coarse"   : a, 
        "Distrubution of Refined"  : b,
        "Distrubution of Iteration": c})
    sns.kdeplot(df_trans['Distrubution of Coarse'],    shade=True, cumulative=False)
    sns.kdeplot(df_trans['Distrubution of Refined'],   shade=True, cumulative=False)
    sns.kdeplot(df_trans['Distrubution of Iteration'], shade=True, cumulative=False)
    plt.xlabel("$E_{trans}$/meter", font2)
    plt.ylabel("Density", font2)

    plt.savefig("exp_1_prob_" + str(id) + ".png", bbox_inches = 'tight')
    plt.show()
    print("save png. ")

if __name__ == '__main__':
    # =================================== #
    # process experiment data in exp-1a
    # =================================== #
    rc_path = './exp_data/exp_1a_'+'r_c_250.npy'
    tc_path = './exp_data/exp_1a_'+'t_c_250.npy'
    rr_path = './exp_data/exp_1a_'+'r_r_250.npy'
    tr_path = './exp_data/exp_1a_'+'t_r_250.npy'
    ri_path = './exp_data/exp_1a_'+'r_i_250.npy'
    ti_path = './exp_data/exp_1a_'+'t_i_250.npy'

    # rc_path = './exp_data/exp_1a_'+'r_c_50.npy'
    # tc_path = './exp_data/exp_1a_'+'t_c_50.npy'
    # rr_path = './exp_data/exp_1a_'+'r_r_50.npy'
    # tr_path = './exp_data/exp_1a_'+'t_r_50.npy'
    # ri_path = './exp_data/exp_1a_'+'r_i_50.npy'
    # ti_path = './exp_data/exp_1a_'+'t_i_50.npy'

    '''
    print data, and use matlab to show the curve
    '''
    # print("")
    # print("coarse")
    # print_rc_metric(rc_path, tc_path)

    # print("")
    # print("refined")
    # print_rc_metric(rr_path, tr_path, is_has_wrong=True)

    # print("")
    # print("iteration")
    # print_rc_metric(ri_path, ti_path, is_has_wrong=True)

    '''
    print error distribution with seaborn
    '''
    # visual_rt_distribution(
    #     rc_path, tc_path, rr_path, tr_path, ri_path, ti_path, 
    #     is_has_wrong=True)

    # =================================== #
    # process experiment data in exp-1b
    # =================================== #
    ri_path = './exp_data/exp_1b_'+'r_c_250.npy'
    ti_path = './exp_data/exp_1b_'+'t_c_250.npy'

    # print("")
    # print("iteration")
    # print_rc_metric(ri_path, ti_path)
    # print_rc_metric_mean(ri_path, ti_path)

    # =================================== #
    # process experiment data in exp-1c
    # =================================== #
    ri_path = './exp_data/exp_1c_'+'r_c_250.npy'
    ti_path = './exp_data/exp_1c_'+'t_c_250.npy'
    
    # print("")
    # visual_rt_distribution_simple(ri_path, ti_path)

    # =================================== #
    # process experiment data in exp-1d
    # =================================== #
    ri_path = './exp_data/exp_1d_'+'r_c_250.npy'
    ti_path = './exp_data/exp_1d_'+'t_c_250.npy'
    
    print("")
    # print_rc_metric(ri_path, ti_path, num=4, is_has_wrong=True)
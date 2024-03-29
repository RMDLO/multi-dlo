#!/usr/bin/env python3

import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField, Image
import sensor_msgs.point_cloud2 as pcl2
import std_msgs.msg

import struct
import time
import cv2
import numpy as np
import time
import os
import sys

import message_filters
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy import interpolate

import sensor_msgs.point_cloud2 as pcl2
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

nodes_per_dlo = 20
num_of_dlos = 0
use_first_frame_masks = sys.argv[1]
folder_name = ''
if use_first_frame_masks.lower() == 'true':
    folder_name = sys.argv[2]

def pt2pt_dis_sq(pt1, pt2):
    return np.sum(np.square(pt1 - pt2))

def pt2pt_dis(pt1, pt2):
    return np.sqrt(np.sum(np.square(pt1 - pt2)))

def register(pts, M, mu=0, max_iter=10):

    # initial guess
    X = pts.copy()
    Y = np.vstack((np.arange(0, 0.1, (0.1/M)), np.zeros(M), np.zeros(M))).T
    if len(pts[0]) == 2:
        Y = np.vstack((np.arange(0, 0.1, (0.1/M)), np.zeros(M))).T
    s = 1
    N = len(pts)
    D = len(pts[0])

    def get_estimates (Y, s):

        # construct the P matrix
        P = np.sum((X[None, :, :] - Y[:, None, :]) ** 2, axis=2)

        c = (2 * np.pi * s) ** (D / 2)
        c = c * mu / (1 - mu)
        c = c * M / N

        P = np.exp(-P / (2 * s))
        den = np.sum(P, axis=0)
        den = np.tile(den, (M, 1))
        den[den == 0] = np.finfo(float).eps
        den += c

        P = np.divide(P, den)  # P is M*N
        Pt1 = np.sum(P, axis=0)  # equivalent to summing from 0 to M (results in N terms)
        P1 = np.sum(P, axis=1)  # equivalent to summing from 0 to N (results in M terms)
        Np = np.sum(P1)
        PX = np.matmul(P, X)

        # get new Y
        P1_expanded = np.full((D, M), P1).T
        new_Y = PX / P1_expanded

        # get new sigma2
        Y_N_arr = np.full((N, M, 3), Y)
        Y_N_arr = np.swapaxes(Y_N_arr, 0, 1)
        X_M_arr = np.full((M, N, 3), X)
        diff = Y_N_arr - X_M_arr
        diff = np.square(diff)
        diff = np.sum(diff, 2)
        new_s = np.sum(np.sum(P*diff, axis=1), axis=0) / (Np*D)

        return new_Y, new_s

    prev_Y, prev_s = Y, s
    new_Y, new_s = get_estimates(prev_Y, prev_s)
    # it = 0
    tol = 0.0
    
    for it in range (max_iter):
        prev_Y, prev_s = new_Y, new_s
        new_Y, new_s = get_estimates(prev_Y, prev_s)

    # print(repr(new_x), new_s)
    return new_Y, new_s

# assuming Y is sorted
# for now, assume each wire has the same number of nodes
# k -- going left for k indices, going right for k indices. a total of 2k neighbors.
def get_nearest_indices (k, Y, idx):
    if idx - k < 0:
        # use more neighbors from the other side?
        indices_arr = np.append(np.arange(0, idx, 1), np.arange(idx+1, idx+k+1+np.abs(idx-k)))
        # indices_arr = np.append(np.arange(0, idx, 1), np.arange(idx+1, idx+k+1))
        return indices_arr
    elif idx + k >= len(Y):
        last_index = len(Y) - 1
        # use more neighbots from the other side?
        indices_arr = np.append(np.arange(idx-k-(idx+k-last_index), idx, 1), np.arange(idx+1, last_index+1, 1))
        # indices_arr = np.append(np.arange(idx-k, idx, 1), np.arange(idx+1, last_index+1, 1))
        return indices_arr
    else:
        indices_arr = np.append(np.arange(idx-k, idx, 1), np.arange(idx+1, idx+k+1, 1))
        return indices_arr

def calc_LLE_weights (k, X):
    W = np.zeros((len(X), len(X)))
    for i in range (0, len(X)):

        wire_index = int(i/nodes_per_dlo)
        offset = wire_index * nodes_per_dlo

        indices = get_nearest_indices(int(k/2), X[offset : offset+nodes_per_dlo], i-offset)
        indices += offset

        # print(i, indices)

        xi, Xi = X[i], X[indices, :]
        component = np.full((len(Xi), len(xi)), xi).T - Xi.T
        Gi = np.matmul(component.T, component)
        # Gi might be singular when k is large
        try:
            Gi_inv = np.linalg.inv(Gi)
        except:
            epsilon = 0.00001
            Gi_inv = np.linalg.inv(Gi + epsilon*np.identity(len(Gi)))
        wi = np.matmul(Gi_inv, np.ones((len(Xi), 1))) / np.matmul(np.matmul(np.ones(len(Xi),), Gi_inv), np.ones((len(Xi), 1)))
        W[i, indices] = np.squeeze(wi.T)

    return W

def indices_array(n):
    r = np.arange(n)
    out = np.empty((n,n,2),dtype=int)
    out[:,:,0] = r[:,None]
    out[:,:,1] = r
    return out


def cpd_lle (X, Y_0, beta, alpha, k, gamma, mu, max_iter, tol, use_decoupling=False, use_prev_sigma2=False, sigma2_0=None):

    M = len(Y_0)
    N = len(X)
    D = len(X[0])

    # initialization
    # faster G calculation
    diff = Y_0[:, None, :] - Y_0[None, :,  :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)
    G_orig = np.exp(-diff / (2 * beta**2))

    Y = Y_0.copy()

    # set up converted node dis
    converted_node_dis = []
    seg_dis = np.sqrt(np.sum(np.square(np.diff(Y_0, axis=0)), axis=1))
    converted_node_coord = []
    last_pt = 0
    converted_node_coord.append(last_pt)
    for i in range (0, num_of_dlos):
        if i != 0:
            last_pt += 10.0
        for j in range (0, nodes_per_dlo):
            if i == 0 and j == 0:
                continue
            last_pt += seg_dis[i*nodes_per_dlo+j-1]
            converted_node_coord.append(last_pt)
    converted_node_coord = np.array(converted_node_coord)
    converted_node_dis = np.abs(converted_node_coord[None, :] - converted_node_coord[:, None])
    converted_node_dis_sq = np.square(converted_node_dis)
    # print("len(converted_node_dis) = ", len(converted_node_dis))
    # print(converted_node_dis)

    # nodes on different wires should never interfere
    G = np.zeros((M, M))
    if use_decoupling:
        G_orig = np.exp(-converted_node_dis_sq / (2 * beta**2))
        for i in range (0, num_of_dlos):
            # copy information from the G matrix over
            start = i * nodes_per_dlo
            end = (i + 1) * nodes_per_dlo
            G[start:end, start:end] = G_orig[start:end, start:end] 
    else:
        G = G_orig.copy()

    # initialize sigma2
    if not use_prev_sigma2 or sigma2_0 == 0:
        (N, D) = X.shape
        (M, _) = Y.shape
        diff = X[None, :, :] - Y[:, None, :]
        err = diff ** 2
        sigma2 = np.sum(err) / (D * M * N)
    else:
        sigma2 = sigma2_0

    # get the LLE matrix
    L = calc_LLE_weights(k, Y_0)
    H = np.matmul((np.identity(M) - L).T, np.identity(M) - L)
    
    # loop until convergence or max_iter reached
    for it in range (0, max_iter):

        # faster P computation
        P = np.sum((X[None, :, :] - Y[:, None, :]) ** 2, axis=2)
        c = (2 * np.pi * sigma2) ** (D / 2) * mu/(1 - mu) * M/N
        P = np.exp(-P / (2 * sigma2))
        den = np.sum(P, axis=0)
        den = np.tile(den, (M, 1))
        den[den == 0] = np.finfo(float).eps
        den += c

        P = np.divide(P, den)
        max_p_nodes = np.argmax(P, axis=0)

        # if use_decoupling:
        #     potential_2nd_max_p_nodes_1 = max_p_nodes - 1
        #     potential_2nd_max_p_nodes_2 = max_p_nodes + 1
        #     potential_2nd_max_p_nodes_1 = np.where(potential_2nd_max_p_nodes_1 < 0, 1, potential_2nd_max_p_nodes_1)
        #     potential_2nd_max_p_nodes_2 = np.where(potential_2nd_max_p_nodes_2 > M-1, M-2, potential_2nd_max_p_nodes_2)
        #     potential_2nd_max_p_nodes_1_select = np.vstack((np.arange(0, N), potential_2nd_max_p_nodes_1)).T
        #     potential_2nd_max_p_nodes_2_select = np.vstack((np.arange(0, N), potential_2nd_max_p_nodes_2)).T
        #     potential_2nd_max_p_1 = P.T[tuple(map(tuple, potential_2nd_max_p_nodes_1_select.T))]
        #     potential_2nd_max_p_2 = P.T[tuple(map(tuple, potential_2nd_max_p_nodes_2_select.T))]
        #     next_max_p_nodes = np.where(potential_2nd_max_p_1 > potential_2nd_max_p_2, potential_2nd_max_p_nodes_1, potential_2nd_max_p_nodes_2)
        #     node_indices_diff = max_p_nodes - next_max_p_nodes
        #     max_node_smaller_index = np.arange(0, N)[node_indices_diff < 0]
        #     max_node_larger_index = np.arange(0, N)[node_indices_diff > 0]
        #     dis_to_max_p_nodes = np.sqrt(np.sum(np.square(Y[max_p_nodes]-X), axis=1))
        #     dis_to_2nd_largest_p_nodes = np.sqrt(np.sum(np.square(Y[next_max_p_nodes]-X), axis=1))
        #     converted_P = np.zeros((M, N)).T

        #     for idx in max_node_smaller_index:
        #         converted_P[idx, 0:max_p_nodes[idx]+1] = converted_node_dis[max_p_nodes[idx], 0:max_p_nodes[idx]+1] + dis_to_max_p_nodes[idx]
        #         converted_P[idx, next_max_p_nodes[idx]:M] = converted_node_dis[next_max_p_nodes[idx], next_max_p_nodes[idx]:M] + dis_to_2nd_largest_p_nodes[idx]

        #     for idx in max_node_larger_index:
        #         converted_P[idx, 0:next_max_p_nodes[idx]+1] = converted_node_dis[next_max_p_nodes[idx], 0:next_max_p_nodes[idx]+1] + dis_to_2nd_largest_p_nodes[idx]
        #         converted_P[idx, max_p_nodes[idx]:M] = converted_node_dis[max_p_nodes[idx], max_p_nodes[idx]:M] + dis_to_max_p_nodes[idx]

        #     converted_P = converted_P.T

        #     P = np.exp(-np.square(converted_P) / (2 * sigma2))

        #     # if not on the same dlo, converted_P has to be >= 10 since z = 10
        #     P = np.where(P < -10*10/(2 * sigma2), 0, P)

        #     den = np.sum(P, axis=0)
        #     den = np.tile(den, (M, 1))
        #     den[den == 0] = np.finfo(float).eps
        #     c = (2 * np.pi * sigma2) ** (D / 2)
        #     c = c * mu / (1 - mu)
        #     c = c * M / N
        #     den += c

        #     P = np.divide(P, den)

        Pt1 = np.sum(P, axis=0)
        P1 = np.sum(P, axis=1)
        Np = np.sum(P1)
        PX = np.matmul(P, X)
    
        # M step
        A_matrix = np.matmul(np.diag(P1), G) + alpha * sigma2 * np.identity(M) + sigma2 * gamma * np.matmul(H, G)
        B_matrix = PX - np.matmul(np.diag(P1) + sigma2*gamma*H, Y_0)
        W = np.linalg.solve(A_matrix, B_matrix)

        # update sigma2
        T = Y_0 + np.matmul(G, W)
        trXtdPt1X = np.trace(np.matmul(np.matmul(X.T, np.diag(Pt1)), X))
        trPXtT = np.trace(np.matmul(PX.T, T))
        trTtdP1T = np.trace(np.matmul(np.matmul(T.T, np.diag(P1)), T))

        sigma2 = (trXtdPt1X - 2*trPXtT + trTtdP1T) / (Np * D)

        # update Y
        if pt2pt_dis_sq(Y, Y_0 + np.matmul(G, W)) < tol:
            Y = Y_0 + np.matmul(G, W)
            break
        else:
            Y = Y_0 + np.matmul(G, W)

    return Y, sigma2

def sort_pts(Y_0):
    diff = Y_0[:, None, :] - Y_0[None, :,  :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)

    N = len(diff)
    G = diff.copy()

    selected_node = np.zeros(N,).tolist()
    selected_node[0] = True
    Y_0_sorted = []
        
    reverse = 0
    counter = 0
    reverse_on = 0
    insertion_counter = 0
    last_visited_b = 0
    while (counter < N - 1):
        
        minimum = 999999
        a = 0
        b = 0
        for m in range(N):
            if selected_node[m]:
                for n in range(N):
                    if ((not selected_node[n]) and G[m][n]):  
                        # not in selected and there is an edge
                        if minimum > G[m][n]:
                            minimum = G[m][n]
                            a = m
                            b = n

        if len(Y_0_sorted) == 0:
            Y_0_sorted.append(Y_0[a].tolist())
            Y_0_sorted.append(Y_0[b].tolist())
        else:
            if last_visited_b != a:
                reverse += 1
                reverse_on = a
                insertion_counter = 0

            if reverse % 2 == 1:
                # switch direction
                Y_0_sorted.insert(Y_0_sorted.index(Y_0[a].tolist()), Y_0[b].tolist())
            elif reverse != 0:
                Y_0_sorted.insert(Y_0_sorted.index(Y_0[reverse_on].tolist())+1+insertion_counter, Y_0[b].tolist())
                insertion_counter += 1
            else:
                Y_0_sorted.append(Y_0[b].tolist())

        last_visited_b = b
        selected_node[b] = True

        counter += 1

    return np.array(Y_0_sorted)

# original post: https://stackoverflow.com/a/59204638
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def ndarray2MarkerArray (Y, marker_frame, node_colors, line_colors, num_of_dlos, nodes_per_dlo):
    results = MarkerArray()

    for i in range (0, num_of_dlos):
        for j in range (0, nodes_per_dlo):
            cur_node_result = Marker()
            cur_node_result.header.frame_id = marker_frame
            cur_node_result.type = Marker.SPHERE
            cur_node_result.action = Marker.ADD
            cur_node_result.ns = "node_results" + str(i*nodes_per_dlo + j)
            cur_node_result.id = i

            cur_node_result.pose.position.x = Y[i*nodes_per_dlo + j, 0]
            cur_node_result.pose.position.y = Y[i*nodes_per_dlo + j, 1]
            cur_node_result.pose.position.z = Y[i*nodes_per_dlo + j, 2]
            cur_node_result.pose.orientation.w = 1.0
            cur_node_result.pose.orientation.x = 0.0
            cur_node_result.pose.orientation.y = 0.0
            cur_node_result.pose.orientation.z = 0.0

            cur_node_result.scale.x = 0.006
            cur_node_result.scale.y = 0.006
            cur_node_result.scale.z = 0.006
            cur_node_result.color.r = node_colors[i, 0]
            cur_node_result.color.g = node_colors[i, 1]
            cur_node_result.color.b = node_colors[i, 2]
            cur_node_result.color.a = node_colors[i, 3]

            results.markers.append(cur_node_result)

            if j == nodes_per_dlo-1:
                break

            cur_line_result = Marker()
            cur_line_result.header.frame_id = marker_frame
            cur_line_result.type = Marker.CYLINDER
            cur_line_result.action = Marker.ADD
            cur_line_result.ns = "line_results" + str(i*nodes_per_dlo + j)
            cur_line_result.id = i*nodes_per_dlo + j

            cur_line_result.pose.position.x = ((Y[i*nodes_per_dlo + j] + Y[i*nodes_per_dlo + j + 1])/2)[0]
            cur_line_result.pose.position.y = ((Y[i*nodes_per_dlo + j] + Y[i*nodes_per_dlo + j + 1])/2)[1]
            cur_line_result.pose.position.z = ((Y[i*nodes_per_dlo + j] + Y[i*nodes_per_dlo + j + 1])/2)[2]

            rot_matrix = rotation_matrix_from_vectors(np.array([0, 0, 1]), (Y[i*nodes_per_dlo + j + 1]-Y[i*nodes_per_dlo + j])/pt2pt_dis(Y[i*nodes_per_dlo + j + 1], Y[i*nodes_per_dlo + j])) 
            r = R.from_matrix(rot_matrix)
            x = r.as_quat()[0]
            y = r.as_quat()[1]
            z = r.as_quat()[2]
            w = r.as_quat()[3]

            cur_line_result.pose.orientation.w = w
            cur_line_result.pose.orientation.x = x
            cur_line_result.pose.orientation.y = y
            cur_line_result.pose.orientation.z = z
            cur_line_result.scale.x = 0.004
            cur_line_result.scale.y = 0.004
            cur_line_result.scale.z = pt2pt_dis(Y[i*nodes_per_dlo + j], Y[i*nodes_per_dlo + j + 1])
            cur_line_result.color.r = line_colors[i, 0]
            cur_line_result.color.g = line_colors[i, 1]
            cur_line_result.color.b = line_colors[i, 2]
            cur_line_result.color.a = line_colors[i, 3]

            results.markers.append(cur_line_result)
    
    return results

saved = False
initialized = False
init_nodes = []
nodes = []
# H = []
cur_time = time.time()
sigma2 = 0

def callback (rgb, depth, pc):
    global saved
    global initialized
    global init_nodes
    global nodes
    global cur_time
    global sigma2
    global num_of_dlos

    proj_matrix = np.array([[918.359130859375,              0.0, 645.8908081054688, 0.0], \
                            [             0.0, 916.265869140625,   354.02392578125, 0.0], \
                            [             0.0,              0.0,               1.0, 0.0]])

    # process rgb image
    cur_image = ros_numpy.numpify(rgb)
    hsv_image = cv2.cvtColor(cur_image.copy(), cv2.COLOR_RGB2HSV)

    # process depth image
    cur_depth = ros_numpy.numpify(depth)

    # process point cloud
    pc_data = ros_numpy.point_cloud2.pointcloud2_to_array(pc)
    cur_pc = ros_numpy.point_cloud2.get_xyz_points(pc_data)
    cur_pc = cur_pc.reshape((720, 1280, 3))

    # rope blue
    lower = (90, 90, 60)
    upper = (130, 255, 255)
    mask_binary = cv2.inRange(hsv_image, lower, upper)
    mask = cv2.cvtColor(mask_binary.copy(), cv2.COLOR_GRAY2BGR)

    # publish mask
    mask_img_msg = ros_numpy.msgify(Image, mask, 'rgb8')
    mask_img_pub.publish(mask_img_msg)

    mask = (mask/255).astype(int)

    # register nodes
    if not initialized:

        # for each object segment
        num_of_dlos = 0
        init_nodes_collection = []

        if use_first_frame_masks.lower() == 'false':
            # separate dlos (assume not entangled)
            gray = mask_binary.copy()
            contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            new_frame = np.zeros(gray.shape, np.uint8)

            for a, contour in enumerate(contours):
                c_area = cv2.contourArea(contour)

                if 1000 <= c_area:
                    num_of_dlos += 1

                    m = np.zeros(gray.shape, np.uint8)
                    m = cv2.drawContours(m, [contour], -1, 255, cv2.FILLED)
                    new_mask = cv2.bitwise_and(gray, m)
                    pt_frame = cv2.bitwise_or(new_frame, new_mask)
                    points = cv2.findNonZero(pt_frame)

                    cur_mask = cv2.cvtColor(pt_frame.copy(), cv2.COLOR_GRAY2BGR)
                    cur_mask = (cur_mask/255).astype(int)
                    filtered_pc = cur_pc * cur_mask
                    filtered_pc = filtered_pc[((filtered_pc[:, :, 0] != 0) | (filtered_pc[:, :, 1] != 0) | (filtered_pc[:, :, 2] != 0))]
                    filtered_pc = filtered_pc[filtered_pc[:, 2] > 0.2]

                    # downsample with open3d
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(filtered_pc)
                    downpcd = pcd.voxel_down_sample(voxel_size=0.007)
                    filtered_pc = np.asarray(downpcd.points)
                    print('filtered dlo {} pc = {}'.format(num_of_dlos, filtered_pc.shape))

                    init_nodes, sigma2 = register(filtered_pc, nodes_per_dlo, mu=0, max_iter=50)
                    init_nodes = np.array(sort_pts(init_nodes))
                    
                    tck, u = interpolate.splprep(init_nodes.T, s=0.0001)
                    # 1st fit, less points
                    u_fine = np.linspace(0, 1, 100) # <-- num fit points
                    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
                    spline_pts = np.vstack((x_fine, y_fine, z_fine)).T

                    # 2nd fit, higher accuracy
                    num_true_pts = int(np.sum(np.sqrt(np.sum(np.square(np.diff(spline_pts, axis=0)), axis=1))) * 1000)
                    u_fine = np.linspace(0, 1, num_true_pts) # <-- num true points
                    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
                    spline_pts = np.vstack((x_fine, y_fine, z_fine)).T
                    total_spline_len = np.sum(np.sqrt(np.sum(np.square(np.diff(spline_pts, axis=0)), axis=1)))

                    init_nodes = spline_pts[np.linspace(0, num_true_pts-1, nodes_per_dlo).astype(int)]
                    init_nodes_collection.append(init_nodes)

                    edges = np.empty((nodes_per_dlo-1, 2), dtype=np.uint32)
                    edges[:, 0] = range((num_of_dlos-1)*nodes_per_dlo, num_of_dlos*nodes_per_dlo - 1)
                    edges[:, 1] = range((num_of_dlos-1)*nodes_per_dlo + 1, num_of_dlos*nodes_per_dlo)
        else:
            all_mask_imgs = os.listdir(os.path.dirname(os.path.abspath(__file__)) + '/segmentation/first_frame_segmentations/' + folder_name)
            for mask_img in all_mask_imgs:
                num_of_dlos += 1

                cur_mask = cv2.imread(os.path.dirname(os.path.abspath(__file__)) + '/segmentation/first_frame_segmentations/' + folder_name + '/' + mask_img)
                cur_mask = (cur_mask/255).astype(int)
                filtered_pc = cur_pc * cur_mask
                filtered_pc = filtered_pc[((filtered_pc[:, :, 0] != 0) | (filtered_pc[:, :, 1] != 0) | (filtered_pc[:, :, 2] != 0))]
                filtered_pc = filtered_pc[filtered_pc[:, 2] > 0.2]

                # downsample with open3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(filtered_pc)
                downpcd = pcd.voxel_down_sample(voxel_size=0.005)
                filtered_pc = np.asarray(downpcd.points)
                print('filtered dlo {} pc = {}'.format(num_of_dlos, filtered_pc.shape))

                init_nodes, sigma2 = register(filtered_pc, nodes_per_dlo, mu=0, max_iter=300)
                init_nodes = np.array(sort_pts(init_nodes))

                tck, u = interpolate.splprep(init_nodes.T, s=0.0001)
                # 1st fit, less points
                u_fine = np.linspace(0, 1, 50) # <-- num fit points
                x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
                spline_pts = np.vstack((x_fine, y_fine, z_fine)).T

                # 2nd fit, higher accuracy
                num_true_pts = int(np.sum(np.sqrt(np.sum(np.square(np.diff(spline_pts, axis=0)), axis=1))) * 1000)
                u_fine = np.linspace(0, 1, num_true_pts) # <-- num true points
                x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
                spline_pts = np.vstack((x_fine, y_fine, z_fine)).T
                total_spline_len = np.sum(np.sqrt(np.sum(np.square(np.diff(spline_pts, axis=0)), axis=1)))

                init_nodes = spline_pts[np.linspace(0, num_true_pts-1, nodes_per_dlo).astype(int)]
                init_nodes_collection.append(init_nodes)

                edges = np.empty((nodes_per_dlo-1, 2), dtype=np.uint32)
                edges[:, 0] = range((num_of_dlos-1)*nodes_per_dlo, num_of_dlos*nodes_per_dlo - 1)
                edges[:, 1] = range((num_of_dlos-1)*nodes_per_dlo + 1, num_of_dlos*nodes_per_dlo)

        init_nodes = np.vstack(init_nodes_collection)
        initialized = True

    # continuous tracking
    if initialized:

        filtered_pc = cur_pc*mask
        filtered_pc = filtered_pc[((filtered_pc[:, :, 0] != 0) | (filtered_pc[:, :, 1] != 0) | (filtered_pc[:, :, 2] != 0))]
        filtered_pc = filtered_pc[filtered_pc[:, 2] > 0.2]

        # downsample with open3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_pc)
        downpcd = pcd.voxel_down_sample(voxel_size=0.007)
        filtered_pc = np.asarray(downpcd.points)
        print('number of points received =', len(filtered_pc))

        # add color
        pc_rgba = struct.unpack('I', struct.pack('BBBB', 255, 40, 40, 255))[0]
        pc_rgba_arr = np.full((len(filtered_pc), 1), pc_rgba)
        filtered_pc_colored = np.hstack((filtered_pc, pc_rgba_arr)).astype('O')
        filtered_pc_colored[:, 3] = filtered_pc_colored[:, 3].astype(int)

        nodes, sigma2 = cpd_lle(X=filtered_pc, 
                                Y_0 = init_nodes, 
                                beta=0.8, 
                                alpha=1, 
                                k=6, 
                                gamma=3, 
                                mu=0.05, 
                                max_iter=30, 
                                tol=0.00001, 
                                use_decoupling = True, 
                                use_prev_sigma2 = True, 
                                sigma2_0 = sigma2)

        init_nodes = nodes.copy()

        alpha = 1
        node_colors = np.array([[255, 0, 0, alpha], [255, 255, 0, alpha], [0, 255, 0, alpha]])
        line_colors = node_colors.copy()

        results = ndarray2MarkerArray(nodes, 'camera_color_optical_frame', node_colors, line_colors, num_of_dlos, nodes_per_dlo)
        results_pub.publish(results)

        # project and pub image
        # sort nodes and edges based on z values
        # ordered from small -> large
        adj_nodes_avg_z_values = np.zeros((len(nodes)-1, 2))  # [index, avg z value]
        adj_nodes_avg_z_values[:, 0] = np.arange(0, len(nodes)-1, 1)
        adj_nodes_avg_z_values[:, 1] = (nodes[0:len(nodes)-1, 2] + nodes[1:len(nodes), 2]) / 2.0
        z_sorted_idx = np.argsort(adj_nodes_avg_z_values[:,-1].copy())

        # we want to plot nodes and edges far away (larger z values) first
        # reverse ind
        z_sorted_idx = np.flip(z_sorted_idx)

        nodes_h = np.hstack((nodes, np.ones((len(nodes), 1))))
        # proj_matrix: 3*4; nodes_h.T: 4*M; result: 3*M
        image_coords = np.matmul(proj_matrix, nodes_h.T).T
        us = (image_coords[:, 0] / image_coords[:, 2]).astype(int)
        vs = (image_coords[:, 1] / image_coords[:, 2]).astype(int)

        tracking_img = cur_image.copy()
        # print('nodes shape = ', np.shape(nodes))

        for idx in z_sorted_idx:
            # if this pair is not from the same dlo, skip
            if int(idx / nodes_per_dlo) != int((idx+1) / nodes_per_dlo):
                continue

            # determine which dlo this is
            dlo_idx = int(idx / nodes_per_dlo)

            # color
            node_color = (255, 0, 0)
            if dlo_idx == 1:
                node_color = (255, 255, 0)
            elif dlo_idx == 2:
                node_color = (0, 255, 0)

            # draw circle 1
            uv_1 = (us[idx], vs[idx])
            cv2.circle(tracking_img, uv_1, 5, node_color, -1)
            # draw circle 2
            uv_2 = (us[idx+1], vs[idx+1])
            cv2.circle(tracking_img, uv_2, 5, node_color, -1)

            # draw line
            cv2.line(tracking_img, uv_1, uv_2, node_color, 10)

        header.stamp = rospy.Time.now()
        # converted_points = pcl2.create_cloud(header, fields, filtered_pc_colored)
        # pc_pub.publish(pc)
        
        tracking_img_msg = ros_numpy.msgify(Image, tracking_img, 'rgb8')
        tracking_img_pub.publish(tracking_img_msg)

        print('tracking step time =', time.time() - cur_time)
        cur_time = time.time()

if __name__=='__main__':
    rospy.init_node('multi_dlo_tracker', anonymous=True)

    rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
    depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
    pc_sub = message_filters.Subscriber('/camera/depth/color/points', PointCloud2)

    # header
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'camera_color_optical_frame'
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgba', 12, PointField.UINT32, 1)]
    pc_pub = rospy.Publisher ('/pts', PointCloud2, queue_size=10)

    tracking_img_pub = rospy.Publisher ('/results_img', Image, queue_size=10)
    mask_img_pub = rospy.Publisher('/mask', Image, queue_size=10)
    results_pub = rospy.Publisher ('/results_marker', MarkerArray, queue_size=10)

    ts = message_filters.TimeSynchronizer([rgb_sub, depth_sub, pc_sub], 10)
    ts.registerCallback(callback)

    rospy.spin()
#!/usr/bin/env python3

import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
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

from utils import ndarray2MarkerArray, register, sort_pts, indices_arr_pixel_coord

proj_matrix = None
def camera_info_callback (info):
    global proj_matrix
    proj_matrix = np.array(list(info.P)).reshape(3, 4)
    print('Received camera projection matrix:')
    print(proj_matrix)
    camera_info_sub.unregister()

def color_thresholding (hsv_image, cur_depth):
    # --- rope blue ---
    lower = (90, 90, 60)
    upper = (130, 255, 255)
    mask_dlo = cv2.inRange(hsv_image, lower, upper).astype('uint8')

    # --- tape red ---
    lower = (130, 60, 40)
    upper = (255, 255, 255)
    mask_red_1 = cv2.inRange(hsv_image, lower, upper).astype('uint8')
    lower = (0, 60, 40)
    upper = (10, 255, 255)
    mask_red_2 = cv2.inRange(hsv_image, lower, upper).astype('uint8')
    mask_marker = cv2.bitwise_or(mask_red_1.copy(), mask_red_2.copy()).astype('uint8')

    # combine masks
    mask = cv2.bitwise_or(mask_marker.copy(), mask_dlo.copy())

    # filter mask base on depth values
    mask[cur_depth < 0.58*1000] = 0

    return mask

def callback (rgb, depth):
    global lower, upper

    print("Initializing...")

    # process rgb image
    cur_image = ros_numpy.numpify(rgb)
    hsv_image = cv2.cvtColor(cur_image.copy(), cv2.COLOR_RGB2HSV)

    # process depth image
    cur_depth = ros_numpy.numpify(depth)

    if not multi_color_dlo:
        # color thresholding
        mask = cv2.inRange(hsv_image, lower, upper)
    else:
        # color thresholding
        mask = color_thresholding(hsv_image, cur_depth)

    start_time = time.time()
    mask_binary = mask.copy()
    mask = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)

    # get cur_pc from depth
    all_pixel_coords = indices_arr_pixel_coord(cur_depth.shape[0], cur_depth.shape[1])
    all_pixel_coords = all_pixel_coords.reshape(cur_depth.shape[0] * cur_depth.shape[1], 2)

    pc_z = cur_depth / 1000.0
    pc_z = pc_z.reshape(cur_depth.shape[0] * cur_depth.shape[1],)

    f = proj_matrix[0, 0]
    cx = proj_matrix[0, 2]
    cy = proj_matrix[1, 2]
    pixel_x = all_pixel_coords[:, 0]
    pixel_y = all_pixel_coords[:, 1]

    pc_x = (pixel_x - cx) * pc_z / f
    pc_y = (pixel_y - cy) * pc_z / f
    cur_pc = np.vstack((pc_x, pc_y))
    cur_pc = np.vstack((cur_pc, pc_z))
    cur_pc = cur_pc.T
    cur_pc = cur_pc.reshape(cur_depth.shape[0], cur_depth.shape[1], 3)

    # for each object segment
    num_of_dlos = 0
    init_nodes_collection = []

    if use_first_frame_masks == False:
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
        all_mask_imgs = os.listdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/segmentation/first_frame_segmentations/' + folder_name)
        for mask_img in all_mask_imgs:
            num_of_dlos += 1

            cur_mask = cv2.imread(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/segmentation/first_frame_segmentations/' + folder_name + '/' + mask_img)
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

    alpha = 1
    node_colors = np.array([[255, 0, 0, alpha], [255, 255, 0, alpha], [0, 255, 0, alpha]])
    line_colors = node_colors.copy()

    results = ndarray2MarkerArray(init_nodes, result_frame_id, node_colors, line_colors, num_of_dlos, nodes_per_dlo)
    results_pub.publish(results)

    # add color
    pc_rgba = struct.unpack('I', struct.pack('BBBB', 255, 40, 40, 255))[0]
    pc_rgba_arr = np.full((len(init_nodes), 1), pc_rgba)
    pc_colored = np.hstack((init_nodes, pc_rgba_arr)).astype(object)
    pc_colored[:, 3] = pc_colored[:, 3].astype(int)

    header.stamp = rospy.Time.now()
    converted_points = pcl2.create_cloud(header, fields, pc_colored)
    pc_pub.publish(converted_points)

    rospy.signal_shutdown('Finished initial node set computation.')

if __name__=='__main__':
    rospy.init_node('init_tracker', anonymous=True)

    nodes_per_dlo = rospy.get_param('/init_tracker/nodes_per_dlo')
    use_first_frame_masks = rospy.get_param('/init_tracker/use_first_frame_masks')
    folder_name = rospy.get_param('/init_tracker/folder_name')

    multi_color_dlo = rospy.get_param('/init_tracker/multi_color_dlo')
    camera_info_topic = rospy.get_param('/init_tracker/camera_info_topic')
    rgb_topic = rospy.get_param('/init_tracker/rgb_topic')
    depth_topic = rospy.get_param('/init_tracker/depth_topic')
    result_frame_id = rospy.get_param('/init_tracker/result_frame_id')
    visualize_initialization_process = rospy.get_param('/init_tracker/visualize_initialization_process')

    hsv_threshold_upper_limit = rospy.get_param('/init_tracker/hsv_threshold_upper_limit')
    hsv_threshold_lower_limit = rospy.get_param('/init_tracker/hsv_threshold_lower_limit')

    upper_array = hsv_threshold_upper_limit.split(' ')
    lower_array = hsv_threshold_lower_limit.split(' ')
    upper = (int(upper_array[0]), int(upper_array[1]), int(upper_array[2]))
    lower = (int(lower_array[0]), int(lower_array[1]), int(lower_array[2]))

    camera_info_sub = rospy.Subscriber(camera_info_topic, CameraInfo, camera_info_callback)
    rgb_sub = message_filters.Subscriber(rgb_topic, Image)
    depth_sub = message_filters.Subscriber(depth_topic, Image)

    # header
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = result_frame_id
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgba', 12, PointField.UINT32, 1)]
    pc_pub = rospy.Publisher ('/init_nodes', PointCloud2, queue_size=10)
    results_pub = rospy.Publisher ('/init_nodes_markers', MarkerArray, queue_size=10)

    ts = message_filters.TimeSynchronizer([rgb_sub, depth_sub], 10)
    ts.registerCallback(callback)

    rospy.spin()
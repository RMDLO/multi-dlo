#pragma once

#include "tracker.h"

#ifndef UTILS_H
#define UTILS_H

using Eigen::MatrixXd;
using cv::Mat;

void signal_callback_handler(int signum);

template <typename T> void print_1d_vector (const std::vector<T>& vec) {
    for (auto item : vec) {
        std::cout << item << " ";
        // std::cout << item << std::endl;
    }
    std::cout << std::endl;
}

double pt2pt_dis_sq (MatrixXd pt1, MatrixXd pt2);
double pt2pt_dis (MatrixXd pt1, MatrixXd pt2);

void reg (MatrixXd pts, MatrixXd& Y, double& sigma2, int M, double mu = 0, int max_iter = 50);
void remove_row(MatrixXd& matrix, unsigned int rowToRemove);
MatrixXd sort_pts (MatrixXd Y_0);

std::vector<MatrixXd> line_sphere_intersection (MatrixXd point_A, MatrixXd point_B, MatrixXd sphere_center, double radius);

visualization_msgs::MarkerArray MatrixXd2MarkerArray (MatrixXd Y,
                                                      std::string marker_frame, 
                                                      std::string marker_ns, 
                                                      std::vector<std::vector<int>> node_colors, 
                                                      std::vector<std::vector<int>> line_colors, 
                                                      double node_scale = 0.01,
                                                      double line_scale = 0.005,
                                                      int num_of_dlos = 1,
                                                      int nodes_per_dlo = 0,
                                                      std::vector<int> visible_nodes = {}, 
                                                      std::vector<float> occluded_node_color = {},
                                                      std::vector<float> occluded_line_color = {});

visualization_msgs::MarkerArray MatrixXd2MarkerArray (std::vector<MatrixXd> Y,
                                                      std::string marker_frame, 
                                                      std::string marker_ns,  
                                                      std::vector<float> node_color, 
                                                      std::vector<float> line_color,
                                                      double node_scale = 0.01,
                                                      double line_scale = 0.005,
                                                      std::vector<int> visible_nodes = {}, 
                                                      std::vector<float> occluded_node_color = {},
                                                      std::vector<float> occluded_line_color = {});

MatrixXd cross_product (MatrixXd vec1, MatrixXd vec2);
double dot_product (MatrixXd vec1, MatrixXd vec2);

#endif
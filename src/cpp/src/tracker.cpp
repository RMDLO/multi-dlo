#include "../include/utils.h"
#include "../include/tracker.h"

using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using cv::Mat;

tracker::tracker () {}

tracker::tracker(int num_of_nodes) {
    // default initialize
    Y_ = MatrixXd::Zero(num_of_nodes, 3);
    guide_nodes_ = Y_.replicate(1, 1);
    sigma2_ = 0.0;
    beta_ = 1.0;
    lambda_ = 1.0;
    alpha_ = 0.0;
    lle_weight_ = 1.0;
    k_vis_ = 0.0;
    mu_ = 0.05;
    max_iter_ = 50;
    tol_ = 0.00001;
    include_lle_ = true;
    use_geodesic_ = true;
    use_prev_sigma2_ = true;
    kernel_ = 1;
    geodesic_coord_ = {};
    correspondence_priors_ = {};
    visibility_threshold_ = 0.02;
}

tracker::tracker(int num_of_nodes,
                 double visibility_threshold,
                 double beta,
                 double lambda,
                 double alpha,
                 double lle_weight,
                 double k_vis,
                 double mu,
                 int max_iter,
                 double tol,
                 bool include_lle,
                 bool use_geodesic,
                 bool use_prev_sigma2,
                 int kernel) 
{
    Y_ = MatrixXd::Zero(num_of_nodes, 3);
    visibility_threshold_ = visibility_threshold;
    guide_nodes_ = Y_.replicate(1, 1);
    sigma2_ = 0.0;
    beta_ = beta;
    lambda_ = lambda;
    alpha_ = alpha;
    lle_weight_ = lle_weight;
    k_vis_ = k_vis;
    mu_ = mu;
    max_iter_ = max_iter;
    tol_ = tol;
    include_lle_ = include_lle;
    use_geodesic_ = use_geodesic;
    use_prev_sigma2_ = use_prev_sigma2;
    kernel_ = kernel;
    geodesic_coord_ = {};
    correspondence_priors_ = {};
}

double tracker::get_sigma2 () {
    return sigma2_;
}

MatrixXd tracker::get_tracking_result () {
    return Y_;
}

MatrixXd tracker::get_guide_nodes () {
    return guide_nodes_;
}

std::vector<MatrixXd> tracker::get_correspondence_pairs () {
    return correspondence_priors_;
}

void tracker::initialize_geodesic_coord (std::vector<double> geodesic_coord) {
    for (int i = 0; i < geodesic_coord.size(); i ++) {
        geodesic_coord_.push_back(geodesic_coord[i]);
    }
}

void tracker::initialize_nodes (MatrixXd Y_init) {
    Y_ = Y_init.replicate(1, 1);
    guide_nodes_ = Y_init.replicate(1, 1);
}

void tracker::set_sigma2 (double sigma2) {
    sigma2_ = sigma2;
}

std::vector<int> tracker::get_nearest_indices (int k, int M, int idx) {
    std::vector<int> indices_arr;
    if (idx - k < 0) {
        for (int i = 0; i <= idx + k; i ++) {
            if (i != idx) {
                indices_arr.push_back(i);
            }
        }
    }
    else if (idx + k >= M) {
        for (int i = idx - k; i <= M - 1; i ++) {
            if (i != idx) {
                indices_arr.push_back(i);
            }
        }
    }
    else {
        for (int i = idx - k; i <= idx + k; i ++) {
            if (i != idx) {
                indices_arr.push_back(i);
            }
        }
    }

    return indices_arr;
}

MatrixXd tracker::calc_LLE_weights (int k, MatrixXd X) {
    MatrixXd W = MatrixXd::Zero(X.rows(), X.rows());
    for (int i = 0; i < X.rows(); i ++) {
        std::vector<int> indices = get_nearest_indices(static_cast<int>(k/2), X.rows(), i);
        MatrixXd xi = X.row(i);
        MatrixXd Xi = MatrixXd(indices.size(), X.cols());

        // fill in Xi: Xi = X[indices, :]
        for (int r = 0; r < indices.size(); r ++) {
            Xi.row(r) = X.row(indices[r]);
        }

        // component = np.full((len(Xi), len(xi)), xi).T - Xi.T
        MatrixXd component = xi.replicate(Xi.rows(), 1).transpose() - Xi.transpose();
        MatrixXd Gi = component.transpose() * component;
        MatrixXd Gi_inv;

        if (Gi.determinant() != 0) {
            Gi_inv = Gi.inverse();
        }
        else {
            // std::cout << "Gi singular at entry " << i << std::endl;
            double epsilon = 0.00001;
            Gi.diagonal().array() += epsilon;
            Gi_inv = Gi.inverse();
        }

        // wi = Gi_inv * 1 / (1^T * Gi_inv * 1)
        MatrixXd ones_row_vec = MatrixXd::Constant(1, Xi.rows(), 1.0);
        MatrixXd ones_col_vec = MatrixXd::Constant(Xi.rows(), 1, 1.0);

        MatrixXd wi = (Gi_inv * ones_col_vec) / (ones_row_vec * Gi_inv * ones_col_vec).value();
        MatrixXd wi_T = wi.transpose();

        for (int c = 0; c < indices.size(); c ++) {
            W(i, indices[c]) = wi_T(c);
        }
    }

    return W;
}

bool tracker::cpd_lle (MatrixXd X,
                       MatrixXd& Y,
                       double& sigma2,
                       double beta,
                       double lambda,
                       double lle_weight,
                       double mu,
                       int max_iter,
                       double tol,
                       bool include_lle,
                       bool use_geodesic,
                       bool use_prev_sigma2,
                       int num_of_dlos,
                       int nodes_per_dlo,
                       bool use_ecpd,
                       std::vector<MatrixXd> correspondence_priors,
                       double alpha,
                       int kernel,
                       std::vector<int> visible_nodes,
                       double k_vis,
                       double visibility_threshold) 
{

    bool converged = true;

    if (correspondence_priors.size() == 0) {
        use_ecpd = false;
    }

    int M = Y.rows();
    int N = X.rows();
    int D = 3;

    MatrixXd Y_0 = Y.replicate(1, 1);

    MatrixXd diff_yy = MatrixXd::Zero(M, M);
    MatrixXd diff_yy_sqrt = MatrixXd::Zero(M, M);
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < M; j ++) {
            diff_yy(i, j) = (Y_0.row(i) - Y_0.row(j)).squaredNorm();
            diff_yy_sqrt(i, j) = (Y_0.row(i) - Y_0.row(j)).norm();
        }
    }

    MatrixXd converted_node_dis = MatrixXd::Zero(M, M); // this is a M*M matrix in place of diff_sqrt
    MatrixXd converted_node_dis_sq = MatrixXd::Zero(M, M);
    std::vector<double> converted_node_coord = {0.0};   // this is not squared

    MatrixXd G = MatrixXd::Zero(M, M);
    if (!use_geodesic) {
        if (kernel == 3) {
            G = (-diff_yy / (2 * beta * beta)).array().exp();
        }
        else if (kernel == 0) {
            G = (-diff_yy_sqrt / (2 * beta * beta)).array().exp();
        }
        else if (kernel == 1) {
            // G = 1/(2*beta * 2*beta) * (-sqrt(2)*diff_yy_sqrt/beta).array().exp() * (2*diff_yy_sqrt.array() + sqrt(2)*beta);
            G = 1/(2 * pow(beta, 2)) * (-2 *diff_yy_sqrt/beta).array().exp() * (2*diff_yy_sqrt.array() + sqrt(2)*beta);
        }
        else if (kernel == 2) {
            // G = 27 * 1/(72 * pow(beta, 3)) * (-sqrt(3)*diff_yy_sqrt/beta).array().exp() * (sqrt(3)*beta*beta + 3*beta*diff_yy_sqrt.array() + sqrt(3)*diff_yy.array());
            G = 3/(16 * pow(beta, 3)) * (-sqrt(6)*diff_yy_sqrt/beta).array().exp() * (2*sqrt(6)*diff_yy.array() + 6*beta*diff_yy_sqrt.array() + sqrt(6) * pow(beta, 2));
        }
        else { // default to gaussian
            G = (-diff_yy / (2 * beta * beta)).array().exp();
        }
    }
    else {
        double cur_sum = 0;
        for (int i = 0; i < M-1; i ++) {
            cur_sum += pt2pt_dis(Y_0.row(i+1), Y_0.row(i));
            converted_node_coord.push_back(cur_sum);
        }

        for (int i = 0; i < converted_node_coord.size(); i ++) {
            for (int j = 0; j < converted_node_coord.size(); j ++) {
                converted_node_dis_sq(i, j) = pow(converted_node_coord[i] - converted_node_coord[j], 2);
                converted_node_dis(i, j) = abs(converted_node_coord[i] - converted_node_coord[j]);
            }
        }

        if (kernel == 3) {
            G = (-converted_node_dis_sq / (2 * beta * beta)).array().exp();
        }
        else if (kernel == 0) {
            G = (-converted_node_dis / (2 * beta * beta)).array().exp();
        }
        else if (kernel == 1) {
            G = 1/(2*beta * 2*beta) * (-sqrt(2)*converted_node_dis/beta).array().exp() * (sqrt(2)*converted_node_dis.array() + beta);
        }
        else if (kernel == 2) {
            G = 27 * 1/(72 * pow(beta, 3)) * (-sqrt(3)*converted_node_dis/beta).array().exp() * (sqrt(3)*beta*beta + 3*beta*converted_node_dis.array() + sqrt(3)*converted_node_dis_sq.array());
        }
        else { // default to gaussian
            G = (-converted_node_dis_sq / (2 * beta * beta)).array().exp();
        }
    }

    // tracking multiple dlos
    if (use_geodesic && num_of_dlos > 1) {
        MatrixXd G_new = MatrixXd::Zero(M, M);
        for (int i = 0; i < num_of_dlos; i ++) {
            int start = i * nodes_per_dlo;
            int end = (i+1) * nodes_per_dlo;
            for (int idx = start; idx < end; idx ++) {
                G_new(idx, idx) = G(idx, idx);
            }
        }
        G = G_new.replicate(1, 1);
    }

    // get the LLE matrix
    MatrixXd L = calc_LLE_weights(6, Y_0);
    MatrixXd H = (MatrixXd::Identity(M, M) - L).transpose() * (MatrixXd::Identity(M, M) - L);

    // construct R and J
    MatrixXd priors = MatrixXd::Zero(correspondence_priors.size(), 3);
    MatrixXd J = MatrixXd::Zero(M, M);
    MatrixXd Y_extended = Y_0.replicate(1, 1);
    MatrixXd G_masked = MatrixXd::Zero(M, M);
    if (correspondence_priors.size() != 0) {
        int num_of_correspondence_priors = correspondence_priors.size();

        for (int i = 0; i < num_of_correspondence_priors; i ++) {
            MatrixXd temp = MatrixXd::Zero(1, 3);
            int index = correspondence_priors[i](0, 0);
            temp(0, 0) = correspondence_priors[i](0, 1);
            temp(0, 1) = correspondence_priors[i](0, 2);
            temp(0, 2) = correspondence_priors[i](0, 3);

            priors.row(i) = temp;
            J.row(index) = MatrixXd::Identity(M, M).row(index);
            Y_extended.row(index) = temp;
            G_masked.row(index) = G.row(index);

            // // enforce boundaries
            // if (i == 0 || i == num_of_correspondence_priors-1) {
            //     J.row(index) *= 5;
            // }
        }
    }

    // diff_xy should be a (M * N) matrix
    MatrixXd diff_xy = MatrixXd::Zero(M, N);
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < N; j ++) {
            diff_xy(i, j) = (Y_0.row(i) - X.row(j)).squaredNorm();
        }
    }

    // initialize sigma2
    if (!use_prev_sigma2 || sigma2 == 0) {
        sigma2 = diff_xy.sum() / static_cast<double>(D * M * N);
    }

    for (int it = 0; it < max_iter; it ++) {

        // update diff_xy
        std::map<int, double> shortest_node_pt_dists;
        for (int m = 0; m < M; m ++) {
            // for each node in Y, determine a point in X closest to it
            // for P_vis calculations
            double shortest_dist = 10000;
            for (int n = 0; n < N; n ++) {
                diff_xy(m, n) = (Y.row(m) - X.row(n)).squaredNorm();
                double dist = (Y.row(m) - X.row(n)).norm();
                if (dist < shortest_dist) {
                    shortest_dist = dist;
                }
            }
            // if close enough to X, the node is visible
            if (shortest_dist <= visibility_threshold) {
                shortest_dist = 0;
            }
            // push back the pair
            shortest_node_pt_dists.insert(std::pair<int, double>(m, shortest_dist));
        }

        MatrixXd P = (-0.5 * diff_xy / sigma2).array().exp();
        MatrixXd P_stored = P.replicate(1, 1);
        double c = pow((2 * M_PI * sigma2), static_cast<double>(D)/2) * mu / (1 - mu) * static_cast<double>(M)/N;
        P = P.array().rowwise() / (P.colwise().sum().array() + c);

        if (use_geodesic) {
            std::vector<int> max_p_nodes(P.cols(), 0);
            MatrixXd pts_dis_sq_geodesic = MatrixXd::Zero(M, N);

            // loop through all points
            for (int i = 0; i < N; i ++) {
                
                P.col(i).maxCoeff(&max_p_nodes[i]);
                int max_p_node = max_p_nodes[i];

                int potential_2nd_max_p_node_1 = max_p_node - 1;
                if (potential_2nd_max_p_node_1 == -1) {
                    potential_2nd_max_p_node_1 = 2;
                }

                int potential_2nd_max_p_node_2 = max_p_node + 1;
                if (potential_2nd_max_p_node_2 == M) {
                    potential_2nd_max_p_node_2 = M - 3;
                }

                int next_max_p_node;
                if (pt2pt_dis(Y.row(potential_2nd_max_p_node_1), X.row(i)) < pt2pt_dis(Y.row(potential_2nd_max_p_node_2), X.row(i))) {
                    next_max_p_node = potential_2nd_max_p_node_1;
                } 
                else {
                    next_max_p_node = potential_2nd_max_p_node_2;
                }

                // fill the current column of pts_dis_sq_geodesic
                pts_dis_sq_geodesic(max_p_node, i) = pt2pt_dis_sq(Y.row(max_p_node), X.row(i));
                pts_dis_sq_geodesic(next_max_p_node, i) = pt2pt_dis_sq(Y.row(next_max_p_node), X.row(i));

                if (max_p_node < next_max_p_node) {
                    for (int j = 0; j < max_p_node; j ++) {
                        pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[max_p_node]) + pt2pt_dis(Y.row(max_p_node), X.row(i)), 2);
                    }
                    for (int j = next_max_p_node; j < M; j ++) {
                        pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[next_max_p_node]) + pt2pt_dis(Y.row(next_max_p_node), X.row(i)), 2);
                    }
                }
                else {
                    for (int j = 0; j < next_max_p_node; j ++) {
                        pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[next_max_p_node]) + pt2pt_dis(Y.row(next_max_p_node), X.row(i)), 2);
                    }
                    for (int j = max_p_node; j < M; j ++) {
                        pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[max_p_node]) + pt2pt_dis(Y.row(max_p_node), X.row(i)), 2);
                    }
                }
            }

            // update P
            P = (-0.5 * pts_dis_sq_geodesic / sigma2).array().exp();
            // P = P.array().rowwise() / (P.colwise().sum().array() + c);
        }
        else {
            P = P_stored.replicate(1, 1);
        }

        
        // modified membership probability (adapted from cdcpd)
        if (visible_nodes.size() != Y.rows() && !visible_nodes.empty() && k_vis != 0) {
            MatrixXd P_vis = MatrixXd::Ones(P.rows(), P.cols());
            double total_P_vis = 0;

            for (int i = 0; i < Y.rows(); i ++) {
                double shortest_node_pt_dist = shortest_node_pt_dists[i];

                double P_vis_i = exp(-k_vis * shortest_node_pt_dist);
                total_P_vis += P_vis_i;

                P_vis.row(i) = P_vis_i * P_vis.row(i);
            }

            // normalize P_vis
            P_vis = P_vis / total_P_vis;

            // modify P
            P = P.cwiseProduct(P_vis);

            // modify c
            c = pow((2 * M_PI * sigma2), static_cast<double>(D)/2) * mu / (1 - mu) / N;
            P = P.array().rowwise() / (P.colwise().sum().array() + c);
        }
        else {
            P = P.array().rowwise() / (P.colwise().sum().array() + c);
        }

        // test code when not using pvis
        // P = P.array().rowwise() / (P.colwise().sum().array() + c);
        // std::cout << P.colwise().sum() << std::endl;

        MatrixXd Pt1 = P.colwise().sum();  // this should have shape (N,) or (1, N)
        MatrixXd P1 = P.rowwise().sum();
        double Np = P1.sum();
        MatrixXd PX = P * X;

        // M step
        MatrixXd A_matrix;
        MatrixXd B_matrix;
        if (include_lle) {
            if (use_ecpd) {
                A_matrix = P1.asDiagonal()*G + lambda*sigma2 * MatrixXd::Identity(M, M) + sigma2*lle_weight * H*G + alpha*J*G;
                B_matrix = PX - P1.asDiagonal()*Y_0 - sigma2*lle_weight * H*Y_0 + alpha*(Y_extended - Y_0);
            }
            else {
                A_matrix = P1.asDiagonal()*G + lambda*sigma2 * MatrixXd::Identity(M, M) + sigma2*lle_weight * H*G;
                B_matrix = PX - P1.asDiagonal()*Y_0 - sigma2*lle_weight * H*Y_0;
            }
        }
        else {
            if (use_ecpd) {
                A_matrix = P1.asDiagonal() * G + lambda * sigma2 * MatrixXd::Identity(M, M) + alpha*J*G;
                B_matrix = PX - P1.asDiagonal() * Y_0 + alpha*(Y_extended - Y_0);
            }
            else {
                A_matrix = P1.asDiagonal() * G + lambda * sigma2 * MatrixXd::Identity(M, M);
                B_matrix = PX - P1.asDiagonal() * Y_0;
            }
        }

        MatrixXd W = A_matrix.completeOrthogonalDecomposition().solve(B_matrix);

        MatrixXd T = Y_0 + G * W;
        double trXtdPt1X = (X.transpose() * Pt1.asDiagonal() * X).trace();
        double trPXtT = (PX.transpose() * T).trace();
        double trTtdP1T = (T.transpose() * P1.asDiagonal() * T).trace();

        sigma2 = (trXtdPt1X - 2*trPXtT + trTtdP1T) / (Np * D);

        if (pt2pt_dis(Y, Y_0 + G*W) / Y.rows() < tol) {
            Y = Y_0 + G*W;
            ROS_INFO_STREAM("Iteration until convergence: " + std::to_string(it+1));
            break;
        }
        else {
            Y = Y_0 + G*W;
        }

        if (it == max_iter - 1) {
            ROS_ERROR("optimization did not converge!");
            converged = false;
            break;
        }
    }
    
    return converged;
}

// alignment: 0 --> align with head; 1 --> align with tail
std::vector<MatrixXd> tracker::traverse_geodesic (std::vector<double> geodesic_coord, const MatrixXd guide_nodes, const std::vector<int> visible_nodes, int alignment) {
    std::vector<MatrixXd> node_pairs = {};

    // extreme cases: only one guide node available
    // since this function will only be called when at least one of head or tail is visible, 
    // the only node will be head or tail
    if (guide_nodes.rows() == 1) {
        MatrixXd node_pair(1, 4);
        node_pair << visible_nodes[0], guide_nodes(0, 0), guide_nodes(0, 1), guide_nodes(0, 2);
        node_pairs.push_back(node_pair);
        return node_pairs;
    }

    double guide_nodes_total_dist = 0;
    double total_seg_dist = 0;
    
    if (alignment == 0) {
        // push back the first pair
        MatrixXd node_pair(1, 4);
        node_pair << visible_nodes[0], guide_nodes(0, 0), guide_nodes(0, 1), guide_nodes(0, 2);
        node_pairs.push_back(node_pair);

        // initialize iterators
        int guide_nodes_it = 0;
        int seg_dist_it = 0;
        int last_seg_dist_it = seg_dist_it;

        // ultimate terminating condition: run out of guide nodes to use. two conditions that can trigger this:
        //   1. next visible node index - current visible node index > 1
        //   2. currenting using the last two guide nodes
        while (visible_nodes[guide_nodes_it+1] - visible_nodes[guide_nodes_it] == 1 && guide_nodes_it+1 <= guide_nodes.rows()-1 && seg_dist_it+1 <= geodesic_coord.size()-1) {
            guide_nodes_total_dist += pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it+1));
            // now keep adding segment dists until the total seg dists exceed the current total guide node dists
            while (guide_nodes_total_dist > total_seg_dist) {
                // break condition
                if (seg_dist_it == geodesic_coord.size()-1) {
                    break;
                }

                total_seg_dist += fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it+1]);
                if (total_seg_dist <= guide_nodes_total_dist) {
                    seg_dist_it += 1;
                }
                else {
                    total_seg_dist -= fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it+1]);
                    break;
                }
            }
            // additional break condition
            if (seg_dist_it == geodesic_coord.size()-1) {
                break;
            }
            // upon exit, seg_dist_it will be at the locaiton where the total seg dist is barely smaller than guide nodes total dist
            // the node desired should be in between guide_nodes[guide_nodes_it] and guide_node[guide_nodes_it + 1]
            // seg_dist_it will also be within guide_nodes_it and guide_nodes_it + 1
            if (guide_nodes_it == 0 && seg_dist_it == 0) {
                continue;
            }
            // if one guide nodes segment is not long enough
            if (last_seg_dist_it == seg_dist_it) {
                guide_nodes_it += 1;
                continue;
            }
            double remaining_dist = total_seg_dist - (guide_nodes_total_dist - pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it+1)));
            MatrixXd temp = (guide_nodes.row(guide_nodes_it + 1) - guide_nodes.row(guide_nodes_it)) * remaining_dist / pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it+1));
            node_pair(0, 0) = seg_dist_it;
            node_pair(0, 1) = temp(0, 0) + guide_nodes(guide_nodes_it, 0);
            node_pair(0, 2) = temp(0, 1) + guide_nodes(guide_nodes_it, 1);
            node_pair(0, 3) = temp(0, 2) + guide_nodes(guide_nodes_it, 2);
            node_pairs.push_back(node_pair);

            // update guide_nodes_it at the very end
            guide_nodes_it += 1;
            last_seg_dist_it = seg_dist_it;
        }
    }
    else {
        // push back the first pair
        MatrixXd node_pair(1, 4);
        node_pair << visible_nodes.back(), guide_nodes(guide_nodes.rows()-1, 0), guide_nodes(guide_nodes.rows()-1, 1), guide_nodes(guide_nodes.rows()-1, 2);
        node_pairs.push_back(node_pair);

        // initialize iterators
        int guide_nodes_it = guide_nodes.rows()-1;
        int seg_dist_it = geodesic_coord.size()-1;
        int last_seg_dist_it = seg_dist_it;

        // ultimate terminating condition: run out of guide nodes to use. two conditions that can trigger this:
        //   1. next visible node index - current visible node index > 1
        //   2. currenting using the last two guide nodes
        while (visible_nodes[guide_nodes_it] - visible_nodes[guide_nodes_it-1] == 1 && guide_nodes_it-1 >= 0 && seg_dist_it-1 >= 0) {
            guide_nodes_total_dist += pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it-1));
            // now keep adding segment dists until the total seg dists exceed the current total guide node dists
            while (guide_nodes_total_dist > total_seg_dist) {
                // break condition
                if (seg_dist_it == 0) {
                    break;
                }

                total_seg_dist += fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it-1]);
                if (total_seg_dist <= guide_nodes_total_dist) {
                    seg_dist_it -= 1;
                }
                else {
                    total_seg_dist -= fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it-1]);
                    break;
                }
            }
            // additional break condition
            if (seg_dist_it == 0) {
                break;
            }
            // upon exit, seg_dist_it will be at the locaiton where the total seg dist is barely smaller than guide nodes total dist
            // the node desired should be in between guide_nodes[guide_nodes_it] and guide_node[guide_nodes_it + 1]
            // seg_dist_it will also be within guide_nodes_it and guide_nodes_it + 1
            if (guide_nodes_it == 0 && seg_dist_it == 0) {
                continue;
            }
            // if one guide nodes segment is not long enough
            if (last_seg_dist_it == seg_dist_it) {
                guide_nodes_it -= 1;
                continue;
            }
            double remaining_dist = total_seg_dist - (guide_nodes_total_dist - pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it-1)));
            MatrixXd temp = (guide_nodes.row(guide_nodes_it - 1) - guide_nodes.row(guide_nodes_it)) * remaining_dist / pt2pt_dis(guide_nodes.row(guide_nodes_it), guide_nodes.row(guide_nodes_it-1));
            node_pair(0, 0) = seg_dist_it;
            node_pair(0, 1) = temp(0, 0) + guide_nodes(guide_nodes_it, 0);
            node_pair(0, 2) = temp(0, 1) + guide_nodes(guide_nodes_it, 1);
            node_pair(0, 3) = temp(0, 2) + guide_nodes(guide_nodes_it, 2);
            node_pairs.insert(node_pairs.begin(), node_pair);

            // update guide_nodes_it at the very end
            guide_nodes_it -= 1;
            last_seg_dist_it = seg_dist_it;
        }
    }

    return node_pairs;
}

std::vector<MatrixXd> tracker::traverse_euclidean (std::vector<double> geodesic_coord, const MatrixXd guide_nodes, const std::vector<int> visible_nodes, int alignment, int alignment_node_idx) {
    std::vector<MatrixXd> node_pairs = {};

    // extreme cases: only one guide node available
    // since this function will only be called when at least one of head or tail is visible, 
    // the only node will be head or tail
    if (guide_nodes.rows() == 1) {
        MatrixXd node_pair(1, 4);
        node_pair << visible_nodes[0], guide_nodes(0, 0), guide_nodes(0, 1), guide_nodes(0, 2);
        node_pairs.push_back(node_pair);
        return node_pairs;
    }

    if (alignment == 0) {
        // push back the first pair
        MatrixXd node_pair(1, 4);
        node_pair << visible_nodes[0], guide_nodes(0, 0), guide_nodes(0, 1), guide_nodes(0, 2);
        node_pairs.push_back(node_pair);

        std::vector<int> consecutive_visible_nodes = {};
        for (int i = 0; i < visible_nodes.size(); i ++) {
            if (i == visible_nodes[i]) {
                consecutive_visible_nodes.push_back(i);
            }
            else {
                break;
            }
        }

        int last_found_index = 0;
        int seg_dist_it = 0;
        MatrixXd cur_center = guide_nodes.row(0);

        // basically pure pursuit lol
        while (last_found_index+1 <= consecutive_visible_nodes.size()-1 && seg_dist_it+1 <= geodesic_coord.size()-1) {
            double look_ahead_dist = fabs(geodesic_coord[seg_dist_it+1] - geodesic_coord[seg_dist_it]);
            bool found_intersection = false;
            std::vector<double> intersection = {};

            for (int i = last_found_index; i+1 <= consecutive_visible_nodes.size()-1; i ++) {
                std::vector<MatrixXd> intersections = line_sphere_intersection(guide_nodes.row(i), guide_nodes.row(i+1), cur_center, look_ahead_dist);

                // if no intersection found
                if (intersections.size() == 0) {
                    continue;
                }
                else if (intersections.size() == 1 && pt2pt_dis(intersections[0], guide_nodes.row(i+1)) > pt2pt_dis(cur_center, guide_nodes.row(i+1))) {
                    continue;
                }
                else {
                    found_intersection = true;
                    last_found_index = i;

                    if (intersections.size() == 2) {
                        if (pt2pt_dis(intersections[0], guide_nodes.row(i+1)) <= pt2pt_dis(intersections[1], guide_nodes.row(i+1))) {
                            // the first solution is closer
                            intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                            cur_center = intersections[0];
                        }
                        else {
                            // the second one is closer
                            intersection = {intersections[1](0, 0), intersections[1](0, 1), intersections[1](0, 2)};
                            cur_center = intersections[1];
                        }
                    }
                    else {
                        intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                        cur_center = intersections[0];
                    }
                    break;
                }
            }

            if (!found_intersection) {
                break;
            }
            else {
                MatrixXd temp = MatrixXd::Zero(1, 4);
                temp(0, 0) = seg_dist_it + 1;
                temp(0, 1) = intersection[0];
                temp(0, 2) = intersection[1];
                temp(0, 3) = intersection[2];
                node_pairs.push_back(temp);

                seg_dist_it += 1;
            }
        }
    }
    else if (alignment == 1){
        // push back the first pair
        MatrixXd node_pair(1, 4);
        node_pair << visible_nodes.back(), guide_nodes(guide_nodes.rows()-1, 0), guide_nodes(guide_nodes.rows()-1, 1), guide_nodes(guide_nodes.rows()-1, 2);
        node_pairs.push_back(node_pair);

        std::vector<int> consecutive_visible_nodes = {};
        for (int i = 1; i <= visible_nodes.size(); i ++) {
            if (visible_nodes[visible_nodes.size()-i] == geodesic_coord.size()-i) {
                consecutive_visible_nodes.push_back(geodesic_coord.size()-i);
            }
            else {
                break;
            }
        }

        int last_found_index = guide_nodes.rows()-1;
        int seg_dist_it = geodesic_coord.size()-1;
        MatrixXd cur_center = guide_nodes.row(guide_nodes.rows()-1);

        // basically pure pursuit lol
        while (last_found_index-1 >= (guide_nodes.rows() - consecutive_visible_nodes.size()) && seg_dist_it-1 >= 0) {

            double look_ahead_dist = fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it-1]);

            bool found_intersection = false;
            std::vector<double> intersection = {};

            for (int i = last_found_index; i >= (guide_nodes.rows() - consecutive_visible_nodes.size() + 1); i --) {
                std::vector<MatrixXd> intersections = line_sphere_intersection(guide_nodes.row(i), guide_nodes.row(i-1), cur_center, look_ahead_dist);

                // if no intersection found
                if (intersections.size() == 0) {
                    continue;
                }
                else if (intersections.size() == 1 && pt2pt_dis(intersections[0], guide_nodes.row(i-1)) > pt2pt_dis(cur_center, guide_nodes.row(i-1))) {
                    continue;
                }
                else {
                    found_intersection = true;
                    last_found_index = i;

                    if (intersections.size() == 2) {
                        if (pt2pt_dis(intersections[0], guide_nodes.row(i-1)) <= pt2pt_dis(intersections[1], guide_nodes.row(i-1))) {
                            // the first solution is closer
                            intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                            cur_center = intersections[0];
                        }
                        else {
                            // the second one is closer
                            intersection = {intersections[1](0, 0), intersections[1](0, 1), intersections[1](0, 2)};
                            cur_center = intersections[1];
                        }
                    }
                    else {
                        intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                        cur_center = intersections[0];
                    }
                    break;
                }
            }

            if (!found_intersection) {
                break;
            }
            else {
                MatrixXd temp = MatrixXd::Zero(1, 4);
                temp(0, 0) = seg_dist_it - 1;
                temp(0, 1) = intersection[0];
                temp(0, 2) = intersection[1];
                temp(0, 3) = intersection[2];
                node_pairs.push_back(temp);

                seg_dist_it -= 1;
            }
        }
    }
    else {
        // push back the first pair
        MatrixXd node_pair(1, 4);
        node_pair << visible_nodes[alignment_node_idx], guide_nodes(alignment_node_idx, 0), guide_nodes(alignment_node_idx, 1), guide_nodes(alignment_node_idx, 2);
        node_pairs.push_back(node_pair);

        std::vector<int> consecutive_visible_nodes_2 = {visible_nodes[alignment_node_idx]};
        for (int i = alignment_node_idx+1; i < visible_nodes.size(); i ++) {
            if (visible_nodes[i] - visible_nodes[i-1] == 1) {
                consecutive_visible_nodes_2.push_back(visible_nodes[i]);
            }
            else {
                break;
            }
        }

        // traverse from the alignment node to the tail node
        int last_found_index = alignment_node_idx;
        int seg_dist_it = visible_nodes[alignment_node_idx];
        MatrixXd cur_center = guide_nodes.row(alignment_node_idx);

        // basically pure pursuit lol
        while (last_found_index+1 <= alignment_node_idx+consecutive_visible_nodes_2.size()-1 && seg_dist_it+1 <= geodesic_coord.size()-1) {
            double look_ahead_dist = fabs(geodesic_coord[seg_dist_it+1] - geodesic_coord[seg_dist_it]);
            bool found_intersection = false;
            std::vector<double> intersection = {};

            for (int i = last_found_index; i+1 <= alignment_node_idx+consecutive_visible_nodes_2.size()-1; i ++) {
                std::vector<MatrixXd> intersections = line_sphere_intersection(guide_nodes.row(i), guide_nodes.row(i+1), cur_center, look_ahead_dist);

                // if no intersection found
                if (intersections.size() == 0) {
                    continue;
                }
                else if (intersections.size() == 1 && pt2pt_dis(intersections[0], guide_nodes.row(i+1)) > pt2pt_dis(cur_center, guide_nodes.row(i+1))) {
                    continue;
                }
                else {
                    found_intersection = true;
                    last_found_index = i;

                    if (intersections.size() == 2) {
                        if (pt2pt_dis(intersections[0], guide_nodes.row(i+1)) <= pt2pt_dis(intersections[1], guide_nodes.row(i+1))) {
                            // the first solution is closer
                            intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                            cur_center = intersections[0];
                        }
                        else {
                            // the second one is closer
                            intersection = {intersections[1](0, 0), intersections[1](0, 1), intersections[1](0, 2)};
                            cur_center = intersections[1];
                        }
                    }
                    else {
                        intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                        cur_center = intersections[0];
                    }
                    break;
                }
            }

            if (!found_intersection) {
                break;
            }
            else {
                MatrixXd temp = MatrixXd::Zero(1, 4);
                temp(0, 0) = seg_dist_it + 1;
                temp(0, 1) = intersection[0];
                temp(0, 2) = intersection[1];
                temp(0, 3) = intersection[2];
                node_pairs.push_back(temp);

                seg_dist_it += 1;
            }
        }


        // traverse from alignment node to head node
        std::vector<int> consecutive_visible_nodes_1 = {visible_nodes[alignment_node_idx]};
        for (int i = alignment_node_idx-1; i >= 0; i ++) {
            if (visible_nodes[i+1] - visible_nodes[i] == 1) {
                consecutive_visible_nodes_1.push_back(visible_nodes[i]);
            }
            else {
                break;
            }
        }

        last_found_index = alignment_node_idx;
        seg_dist_it = visible_nodes[alignment_node_idx];
        cur_center = guide_nodes.row(alignment_node_idx);

        // basically pure pursuit lol
        while (last_found_index-1 >= alignment_node_idx-consecutive_visible_nodes_1.size() && seg_dist_it-1 >= 0) {
            double look_ahead_dist = fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it-1]);
            bool found_intersection = false;
            std::vector<double> intersection = {};

            for (int i = last_found_index; i-1 >= 0; i --) {
                std::vector<MatrixXd> intersections = line_sphere_intersection(guide_nodes.row(i), guide_nodes.row(i-1), cur_center, look_ahead_dist);

                // if no intersection found
                if (intersections.size() == 0) {
                    continue;
                }
                else if (intersections.size() == 1 && pt2pt_dis(intersections[0], guide_nodes.row(i-1)) > pt2pt_dis(cur_center, guide_nodes.row(i-1))) {
                    continue;
                }
                else {
                    found_intersection = true;
                    last_found_index = i;

                    if (intersections.size() == 2) {
                        if (pt2pt_dis(intersections[0], guide_nodes.row(i-1)) <= pt2pt_dis(intersections[1], guide_nodes.row(i-1))) {
                            // the first solution is closer
                            intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                            cur_center = intersections[0];
                        }
                        else {
                            // the second one is closer
                            intersection = {intersections[1](0, 0), intersections[1](0, 1), intersections[1](0, 2)};
                            cur_center = intersections[1];
                        }
                    }
                    else {
                        intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                        cur_center = intersections[0];
                    }
                    break;
                }
            }

            if (!found_intersection) {
                break;
            }
            else {
                MatrixXd temp = MatrixXd::Zero(1, 4);
                temp(0, 0) = seg_dist_it - 1;
                temp(0, 1) = intersection[0];
                temp(0, 2) = intersection[1];
                temp(0, 3) = intersection[2];
                node_pairs.push_back(temp);

                seg_dist_it -= 1;
            }
        }
    }

    return node_pairs;
}

void tracker::tracking_step (MatrixXd X, 
                             std::vector<int> visible_nodes, 
                             std::vector<int> visible_nodes_extended, 
                             MatrixXd proj_matrix, 
                             int img_rows, 
                             int img_cols) {
    // for later
}
#ifndef ARMOR_PROCESSOR__TRACKER_HPP_
#define ARMOR_PROCESSOR__TRACKER_HPP_

// Eigen
#include <Eigen/Eigen>

// STD
#include "KalmanFilter.hpp"
#include "KuhnMunkres.hpp"
#include "globalParam.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <cfloat>
#include <deque>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <string>
#include <vector>

enum class ArmorsNum { NORMAL_4 = 4, BALANCE_2 = 2, OUTPOST_3 = 3 };

class Tracker{
public:
    Tracker(GlobalParam &gp);
    void track(std::vector<Armor> &armors_curr, Translator &ts, double dt);
    void draw(const std::vector<Armor> armors_curr = std::vector<Armor>());
    void calc_armor_back(std::vector<Armor> &armors, Translator &ts);
private:

    GlobalParam *gp;
    double dt;
    double r_xy_correction[4];
    double r_yaw_corrected;
    std::vector<ExtendedKalmanFilter> ekf_list;
    std::vector<Eigen::VectorXd> z_vector_list;
    std::vector<int> lost_frame_count;
    std::vector<int> number_list;
    std::vector<Armor> armors_pred;
    bool have_number[8] = {false};
    int index = 0;

    void refine_zVector(int ekf_id);
    void create_new_ekf(Armor &armor);
    // double cost_threshold;
    
    using VecVecFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;
    using VecMatFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd &)>;
    using VoidMatFunc = std::function<Eigen::MatrixXd()>;

    // 保存了使用的EKF模型
    VecVecFunc f;           // State transition vector function
    VecVecFunc h;           // Observation nonlinear vector function
    VecMatFunc j_f;         // Jacobian of f()
    VecMatFunc j_h;         // Jacobian of h()
    VoidMatFunc u_q;        // Process noise covariance matrix
    VecMatFunc u_r;         // Measurement noise covariance matrix
    VecVecFunc nomolize_residual;  // Nomalize residual function
    //前哨站
    VecVecFunc f_outpost;           // State transition vector function
    VecVecFunc h_outpost;           // Observation nonlinear vector function
    VecMatFunc j_f_outpost;         // Jacobian of f()
    VecMatFunc j_h_outpost;         // Jacobian of h()
    VoidMatFunc u_q_outpost;        // Process noise covariance matrix
    VecMatFunc u_r_outpost;         // Measurement noise covariance matrix
};
#endif // ARMOR_PROCESSOR__TRACKER_HPP_
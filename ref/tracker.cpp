#include "tracker.hpp"
#include "globalParam.hpp"
#include "opencv2/core/mat.hpp"

// STD

// static inline double normalize_angle(double angle)
// {
//     const double result = fmod(angle + M_PI, 2.0 * M_PI);
//     if (result <= 0.0)
//         return result + M_PI;
//     return result - M_PI;
// }
// double shortest_angular_distance(double from, double to)
// {
//     return normalize_angle(to - from);
// }

// x << xc, vx, yc, vy, z1, z2, vz, r1, r2, yaw, vyaw;
#define OUTPOSE_R 553/2.0

inline double dyaw(double yaw1, double yaw2){
    double ans = fmod(abs(yaw1 - yaw2), 2*M_PI);
    return  min(ans, 2*M_PI - ans);
}

inline double cost(Armor a, Armor b){
    double ans = 0;
    Eigen::Vector3d p1 = a.position;
    Eigen::Vector3d p2 = b.position;
    ans = sqrt(pow(p1(0) - p2(0), 2) + pow(p1(1) - p2(1), 2) + pow(p1(2) - p2(2), 2));
    ans += 2000 * pow(dyaw(a.yaw, b.yaw), 2);
    ans += 10 * abs(p1(2) - p2(2));
    return ans;
}

inline Armor calcArmor(double xc, double yc, double z, double r, double yaw){
    Armor armor;
    armor.position = Eigen::Vector3d(xc - r * cos(yaw), yc - r * sin(yaw), z);
    armor.yaw = yaw;
    return armor;
}

void Tracker::track(std::vector<Armor> &armors_curr, Translator &ts, double dt){
    this->dt = dt;
    for (auto &zVector : z_vector_list) zVector.setZero();
    armors_pred.clear();
    for (int i = 0; i < ekf_list.size(); i++){
        auto x = ekf_list[i].predict();
        if (number_list[i] != 5){
            armors_pred.push_back(calcArmor(x(0), x(2), x(4), x(7), x(9)));
            armors_pred.push_back(calcArmor(x(0), x(2), x(5), x(8), x(9) + M_PI/2));
            armors_pred.push_back(calcArmor(x(0), x(2), x(4), x(7), x(9) + M_PI));
            armors_pred.push_back(calcArmor(x(0), x(2), x(5), x(8), x(9) + 3*M_PI/2));
        }else{
            armors_pred.push_back(calcArmor(x(0), x(2), x(4), x(7), x(9)));
            armors_pred.push_back(calcArmor(x(0), x(2), x(4), x(7), x(9) + M_PI/3*2));
            armors_pred.push_back(calcArmor(x(0), x(2), x(4), x(7), x(9) + M_PI/3*4));
            armors_pred.push_back(Armor{0});
        }
        
    };
    std::vector<int> matchX, matchY;
    int n, m;
    n = armors_curr.size();
    m = armors_pred.size();
    KuhnMunkres km;
    std::vector<std::vector<double>> w(n, std::vector<double>(m, -INF));
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            auto &u = armors_curr[i];
            auto &v = armors_pred[j];
            w[i][j] = cost(u, v);
            if (u.type != number_list[j/4]) w[i][j] = INF;
        }       
    }
    km.solve(w, matchX, matchY, gp->cost_threshold);
    for(int i = 0; i < n; i++)
        if(matchX[i] == -1)
            create_new_ekf(armors_curr[i]);
    for(int i = 0; i < n; i++){
        if(matchX[i] == -1) continue;
        int ekf_id  = matchX[i]/4;
        int armor_id= matchX[i]%4;
        auto &armor = armors_curr[i];
        z_vector_list[ekf_id].segment(armor_id*4, 4) << armor.position(0), armor.position(1), armor.position(2), armor.yaw;
    }
    for (int i = 0; i < ekf_list.size(); i++){
        if (z_vector_list[i].norm() == 0){
            lost_frame_count[i]++;
            int max_lost_frame = gp->max_lost_frame;
            if (number_list[i] == 5) max_lost_frame *= 3; // outpost
            if (lost_frame_count[i] > max_lost_frame){
                ekf_list.erase(ekf_list.begin() + i);
                z_vector_list.erase(z_vector_list.begin() + i);
                lost_frame_count.erase(lost_frame_count.begin() + i);
                have_number[number_list[i]] = false;
                number_list.erase(number_list.begin() + i);
                i--;
                continue;
            }
        } else {
            lost_frame_count[i] = 0;
            refine_zVector(i);
            if(number_list[i] != 5)ekf_list[i].update(z_vector_list[i]);
            else{
                ekf_list[i].update(z_vector_list[i].segment(0, 12));
                ekf_list[i].get_X()(7) = OUTPOSE_R;
                ekf_list[i].get_X()(5) = ekf_list[i].get_X()(4);
            }
        }
        if (ekf_list[i].get_X()(7)<100 || ekf_list[i].get_X()(8)<100 || ekf_list[i].get_X()(7)>450 || ekf_list[i].get_X()(8)>450 || abs(ekf_list[i].get_X()(10)) > 20){
            ekf_list.erase(ekf_list.begin() + i);
            z_vector_list.erase(z_vector_list.begin() + i);
            lost_frame_count.erase(lost_frame_count.begin() + i);
            have_number[number_list[i]] = false;
            number_list.erase(number_list.begin() + i);
            i--;
        }
    }
    if (ekf_list.size() > 0){
        float min_angle = INF;
        for(int i = 0; i < ekf_list.size(); i++){
            int num = number_list[i];
            if(num == 1){
                index = i;
                break;
            }
            auto x = ekf_list[i].get_X();
            float dangle = atan2(x(2),x(0)) - ts.message.yaw;
            dangle = abs(atan2(sin(dangle), cos(dangle)));
            if(dangle < min_angle){
                min_angle = dangle;
                index = i;
            }
        }
        auto x = ekf_list[index].get_X();        
        ts.message.crc = 1;
        ts.message.armor_flag = number_list[index];
        ts.message.x_c = x(0);
        ts.message.v_x = x(1);
        ts.message.y_c = x(2);
        ts.message.v_y = x(3);
        ts.message.z1 = x(4);
        ts.message.z2 = x(5);
        // ts.message.v_z = x(6);
        ts.message.r1 = x(7);
        ts.message.r2 = x(8);
        ts.message.yaw_a = x(9);
        ts.message.vyaw = x(10);
    }else{
        ts.message.armor_flag = 0;
        ts.message.x_c = 0;
        ts.message.v_x = 0;
        ts.message.y_c = 0;
        ts.message.v_y = 0;
        ts.message.z1 = 0;
        ts.message.z2 = 0;
        // ts.message.v_z = 0;
        ts.message.r1 = 0;
        ts.message.r2 = 0;
        ts.message.yaw_a = 0;
        ts.message.vyaw = 0;
        ts.message.crc = 0;
    }
}

void Tracker::refine_zVector(int ekf_id){
    auto x = ekf_list[ekf_id].get_X();
    double xc = x(0), yc = x(2), z1 = x(4), z2 = x(5), r1 = x(7), r2 = x(8), yaw = x(9);
    auto &z = z_vector_list[ekf_id];
    Armor armor;
    r_xy_correction[0] = r_xy_correction[1] = r_xy_correction[2] = r_xy_correction[3] = 1;
    r_yaw_corrected = gp -> r_yaw;
    if (number_list[ekf_id] != 5){
        // 看不见的装甲板的位姿由能看见的装甲板估计
        if (z.segment(0, 4) != Eigen::VectorXd::Zero(4)){
            armor = calcArmor(xc, yc, z(2), sqrt(pow(z(0)-xc, 2) + pow(z(1)-yc, 2)), z(3) + M_PI);
            z.segment(8, 4) << armor.position(0), armor.position(1), armor.position(2), armor.yaw;
            r_xy_correction[2] *= 10;
        }else if (z.segment(8, 4) != Eigen::VectorXd::Zero(4)){
            armor = calcArmor(xc, yc, z(10), sqrt(pow(z(8)-xc, 2) + pow(z(9)-yc, 2)), z(11) - M_PI);
            z.segment(0, 4) << armor.position(0), armor.position(1), armor.position(2), armor.yaw;
            r_xy_correction[0] *= 10;
        }
        if (z.segment(4, 4) != Eigen::VectorXd::Zero(4)){
            armor = calcArmor(xc, yc, z(6), sqrt(pow(z(4)-xc, 2) + pow(z(5)-yc, 2)), z(7) + M_PI);
            z.segment(12, 4) << armor.position(0), armor.position(1), armor.position(2), armor.yaw;
            r_xy_correction[3] *= 10;
        }else if (z.segment(12, 4) != Eigen::VectorXd::Zero(4)){
            armor = calcArmor(xc, yc, z(14), sqrt(pow(z(12)-xc, 2) + pow(z(13)-yc, 2)), z(15) - M_PI);
            z.segment(4, 4) << armor.position(0), armor.position(1), armor.position(2), armor.yaw;
            r_xy_correction[1] *= 10;
        }
        if (z.segment(0, 4) == Eigen::VectorXd::Zero(4) && z.segment(8, 4) == Eigen::VectorXd::Zero(4)){
            armor = calcArmor(xc, yc, z1, r1, z(7) - M_PI/2);
            z.segment(0, 4) << armor.position(0), armor.position(1), armor.position(2), armor.yaw;
            armor = calcArmor(xc, yc, z1, r1, z(7) + M_PI/2);
            z.segment(8, 4) << armor.position(0), armor.position(1), armor.position(2), armor.yaw;
            r_yaw_corrected *= 10;
            r_xy_correction[0] *= 10;
            r_xy_correction[2] *= 10;
        }
        if (z.segment(4, 4) == Eigen::VectorXd::Zero(4) && z.segment(12, 4) == Eigen::VectorXd::Zero(4)){
            armor = calcArmor(xc, yc, z2, r2, z(3) + M_PI/2);
            z.segment(4, 4) << armor.position(0), armor.position(1), armor.position(2), armor.yaw;
            armor = calcArmor(xc, yc, z2, r2, z(3) - M_PI/2);
            z.segment(12, 4) << armor.position(0), armor.position(1), armor.position(2), armor.yaw;
            r_yaw_corrected *= 10;
            r_xy_correction[1] *= 10;
            r_xy_correction[3] *= 10;
        }
    }else{  // 前哨站
        if (z.segment(0, 4) != Eigen::VectorXd::Zero(4)){
            armor = calcArmor(xc, yc, z(2), OUTPOSE_R, z(3) + M_PI / 3 *2);
            z.segment(4, 4) << armor.position(0), armor.position(1), armor.position(2), armor.yaw;
            armor = calcArmor(xc, yc, z(2), OUTPOSE_R, z(3) + M_PI / 3 *4);
            z.segment(8, 4) << armor.position(0), armor.position(1), armor.position(2), armor.yaw;
            r_xy_correction[1] *= 10;
            r_xy_correction[2] *= 10;
        }
        else if (z.segment(4, 4) != Eigen::VectorXd::Zero(4)){
            armor = calcArmor(xc, yc, z(6), OUTPOSE_R, z(7) + M_PI / 3 *2);
            z.segment(8, 4) << armor.position(0), armor.position(1), armor.position(2), armor.yaw;
            armor = calcArmor(xc, yc, z(6), OUTPOSE_R, z(7) + M_PI / 3 *4);
            z.segment(0, 4) << armor.position(0), armor.position(1), armor.position(2), armor.yaw;
            r_xy_correction[2] *= 10;
            r_xy_correction[0] *= 10;
        }
        else if (z.segment(8, 4) != Eigen::VectorXd::Zero(4)){
            armor = calcArmor(xc, yc, z(10), OUTPOSE_R, z(11) + M_PI / 3 *2);
            z.segment(0, 4) << armor.position(0), armor.position(1), armor.position(2), armor.yaw;
            armor = calcArmor(xc, yc, z(10), OUTPOSE_R, z(11) + M_PI / 3 *4);
            z.segment(4, 4) << armor.position(0), armor.position(1), armor.position(2), armor.yaw;
            r_xy_correction[0] *= 10;
            r_xy_correction[1] *= 10;
        }
        z.segment(12, 4) = Eigen::VectorXd::Zero(4);
    }
}

void Tracker::create_new_ekf(Armor &armor){
    if(have_number[armor.type]) return;
    if(armor.type != 5){
        Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(11, 11) * gp->s2p0xyr;
        P0(9, 9) = gp->s2p0yaw;
        Eigen::Vector3d c = calcArmor(armor.position(0), armor.position(1), armor.position(2), gp->r_initial, armor.yaw + M_PI).position; 
        Eigen::VectorXd x0(11);
        x0 << c(0), 0, c(1), 0, c(2), c(2), 0, gp->r_initial, gp->r_initial, armor.yaw, 0;
        ekf_list.push_back(ExtendedKalmanFilter(f, h, j_f, j_h, u_q, u_r, nomolize_residual, P0, x0));
        z_vector_list.push_back(Eigen::VectorXd::Zero(16));
        lost_frame_count.push_back(0);
        number_list.push_back(armor.type);
        have_number[armor.type] = true;
        armors_pred.push_back(calcArmor(x0(0), x0(2), x0(4), x0(7), x0(9)));
        armors_pred.push_back(calcArmor(x0(0), x0(2), x0(5), x0(8), x0(9) + M_PI/2));
        armors_pred.push_back(calcArmor(x0(0), x0(2), x0(4), x0(7), x0(9) + M_PI));
        armors_pred.push_back(calcArmor(x0(0), x0(2), x0(5), x0(8), x0(9) + 3*M_PI/2));
    }else{  // 前哨站
        Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(11, 11) * gp->s2p0xyr;
        P0(9, 9) = gp->s2p0yaw;
        Eigen::Vector3d c = calcArmor(armor.position(0), armor.position(1), armor.position(2), OUTPOSE_R, armor.yaw + M_PI).position; 
        Eigen::VectorXd x0(11);
        x0 << c(0), 0, c(1), 0, c(2), 0, 0, OUTPOSE_R, OUTPOSE_R, armor.yaw, 0;
        ekf_list.push_back(ExtendedKalmanFilter(f_outpost, h_outpost, j_f_outpost, j_h_outpost, u_q_outpost, u_r_outpost, nomolize_residual, P0, x0));
        z_vector_list.push_back(Eigen::VectorXd::Zero(16));
        lost_frame_count.push_back(0);
        number_list.push_back(armor.type);
        have_number[armor.type] = true;
        armors_pred.push_back(calcArmor(x0(0), x0(2), x0(4), x0(7), x0(9)));
        armors_pred.push_back(calcArmor(x0(0), x0(2), x0(4), x0(7), x0(9) + M_PI/3*2));
        armors_pred.push_back(calcArmor(x0(0), x0(2), x0(4), x0(7), x0(9) + M_PI/3*4));
        armors_pred.push_back(Armor{0});
    }   
}

void Tracker::draw(const std::vector<Armor> armor_curr){
    cv::Mat img(1080, 1440, CV_8UC3, cv::Scalar(255, 255, 255));
    double scale = 5, bias = 1000;
    for (auto ekf : ekf_list){
        auto x = ekf.get_X();
        double xc = x(0), yc = x(2);
        cv::circle(img, cv::Point(yc/scale + 720, (xc-bias)/scale), 5, cv::Scalar(0, 255, 0), -1);
    }
    for (auto z : z_vector_list){
        Armor armor;
        for (int i = 0; i < 16; i+=4){
            armor.position = z.segment(i,3), armor.yaw = z(i + 3);
            cv::Point a(armor.position(0) - 67.5 * sin(armor.yaw), armor.position(1) + 67.5 * cos(armor.yaw));
            cv::Point b(armor.position(0) + 67.5 * sin(armor.yaw), armor.position(1) - 67.5 * cos(armor.yaw));
            cv::Point c(armor.position(0) + 30 * cos(armor.yaw), armor.position(1) + 30 * sin(armor.yaw));
            cv::circle(img, cv::Point(armor.position(1)/scale + 720,(armor.position(0)-bias)/scale), 3, cv::Scalar(0, 0, 255), -1);
            cv::line(img, cv::Point(a.y/scale + 720, (a.x-bias)/scale), cv::Point(b.y/scale + 720, (b.x-bias)/scale), cv::Scalar(0, 0 ,255), 2);
            cv::line(img, cv::Point(c.y/scale + 720, (c.x-bias)/scale), cv::Point(armor.position(1)/scale + 720,(armor.position(0)-bias)/scale), cv::Scalar(0, 0 ,255), 2);
        }
    }
    for (auto armor : armors_pred){
        cv::Point a(armor.position(0) - 67.5 * sin(armor.yaw), armor.position(1) + 67.5 * cos(armor.yaw));
        cv::Point b(armor.position(0) + 67.5 * sin(armor.yaw), armor.position(1) - 67.5 * cos(armor.yaw));
        cv::Point c(armor.position(0) + 30 * cos(armor.yaw), armor.position(1) + 30 * sin(armor.yaw));
        cv::circle(img, cv::Point(armor.position(1)/scale + 720,(armor.position(0)-bias)/scale), 3, cv::Scalar(0, 255, 0), -1);
        cv::line(img, cv::Point(a.y/scale + 720, (a.x-bias)/scale), cv::Point(b.y/scale + 720, (b.x-bias)/scale), cv::Scalar(0, 255 ,0), 2);
        cv::line(img, cv::Point(c.y/scale + 720, (c.x-bias)/scale), cv::Point(armor.position(1)/scale + 720,(armor.position(0)-bias)/scale), cv::Scalar(0, 255 ,0), 2);
    }
    for (auto armor : armor_curr){
        cv::Point a(armor.position(0) - 67.5 * sin(armor.yaw), armor.position(1) + 67.5 * cos(armor.yaw));
        cv::Point b(armor.position(0) + 67.5 * sin(armor.yaw), armor.position(1) - 67.5 * cos(armor.yaw));
        cv::Point c(armor.position(0) + 30 * cos(armor.yaw), armor.position(1) + 30 * sin(armor.yaw));
        cv::circle(img, cv::Point(armor.position(1)/scale + 720,(armor.position(0)-bias)/scale), 3, cv::Scalar(255, 0, 0), -1);
        cv::line(img, cv::Point(a.y/scale + 720, (a.x-bias)/scale), cv::Point(b.y/scale + 720, (b.x-bias)/scale), cv::Scalar(255, 0 ,0), 2);
        cv::line(img, cv::Point(c.y/scale + 720, (c.x-bias)/scale), cv::Point(armor.position(1)/scale + 720,(armor.position(0)-bias)/scale), cv::Scalar(255, 0 ,0), 2);
    }
    cv::resize(img, img, cv::Size(img.size[1] * gp->resize, img.size[0] * gp->resize));
    // cv::imshow("track",img);
}

void Tracker::calc_armor_back(std::vector<Armor> &armors, Translator &ts){
    if (ekf_list.size() == 0)return;
    auto x = ekf_list[index].get_X();
    double xc = x(0), yc = x(2), z1 = x(4), z2 = x(5), r1 = x(7), r2 = x(8), yaw = x(9);
    if (number_list[index] == 5)
        armors = {
            calcArmor(xc, yc, z1, r1, yaw),
            calcArmor(xc, yc, z1, r1, yaw + M_PI/3*2),
            calcArmor(xc, yc, z1, r1, yaw + M_PI/3*4)
        };
    else
        armors = {
            calcArmor(xc, yc, z1, r1, yaw),
            calcArmor(xc, yc, z2, r2, yaw + M_PI/2),
            calcArmor(xc, yc, z1, r1, yaw + M_PI),
            calcArmor(xc, yc, z2, r2, yaw + 3*M_PI/2)
        };
    Eigen::Matrix3d m_pitch(3, 3);//pitch旋转矩阵
    Eigen::Matrix3d m_yaw(3, 3);//yaw旋转矩阵
    m_yaw << cos(ts.message.yaw), -sin(ts.message.yaw), 0, sin(ts.message.yaw), cos(ts.message.yaw), 0, 0, 0, 1;
    m_pitch << cos(ts.message.pitch), 0, -sin(ts.message.pitch), 0, 1, 0, sin(ts.message.pitch), 0, cos(ts.message.pitch);
    Eigen::MatrixXd r_mat = m_yaw * m_pitch;//旋转矩阵
    Eigen::Matrix3d rotation;
    rotation << 0, 0, 1,
                -1, 0, 0,
                0, -1, 0;
    Eigen::Matrix3d rMat = r_mat * rotation;
    Eigen::Vector3d tVec = r_mat * Eigen::Vector3d(gp->vector_x, gp->vector_y, gp->vector_z);
    for (auto &armor : armors){
        armor.type = number_list[index];
        armor.position = rMat.inverse() * (armor.position - tVec);
        armor.center = cv::Point3f(armor.position(0), armor.position(1), armor.position(2));
        double yaw = - armor.yaw;
        double pitch = (number_list[index] != 5 ? M_PI - (15 * M_PI / 180) : M_PI - (-15 * M_PI / 180));
        Eigen::Matrix<double, 3, 3> mat_x;
        mat_x << double(1), double(0), double(0),
                 double(0), cos(pitch), -sin(pitch),
                 double(0), sin(pitch), cos(pitch);
        Eigen::Matrix<double, 3, 3> mat_y;
        mat_y << cos(yaw), double(0), sin(yaw),
                double(0), double(1), double(0),
                -sin(yaw), double(0), cos(yaw);
        Eigen::Matrix<double, 3, 3> rotation_matrix = rMat.inverse() * rotation * mat_y * mat_x;
        cv::Mat rVec;
        cv::eigen2cv(rotation_matrix, rVec);
        cv::Rodrigues(rVec, rVec);
        armor.angle = cv::Point3f(rVec.at<double>(0), rVec.at<double>(1), rVec.at<double>(2));
    }
}

Tracker::Tracker(GlobalParam &gp){
    this->gp = &gp;

    // 定义状态转移函数
    f = [this](const Eigen::VectorXd &x)
    {
        // 更新状态：位置和速度
        Eigen::VectorXd x_new = x;
        x_new(0) += x(1) * dt; // 更新xc
        x_new(2) += x(3) * dt; // 更新yc
        x_new(9) += x(10)* dt; // 更新yaw
        return x_new;
    };

    // 状态转移函数的雅可比矩阵
    j_f = [this](const Eigen::VectorXd &)
    {
        Eigen::MatrixXd f(11, 11);
        // clang-format off
        //    xc   vx   yc   vy   z1   z2   vz   r1   r2   yaw  vyaw
        f <<  1,   dt,  0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   1,   dt,  0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   dt,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1;
        // clang-format on
        return f;
    };

    // 观测函数
    h = [](const Eigen::VectorXd &x)
    {
        Eigen::VectorXd z(16);
        double xc = x(0), yc = x(2), yaw = x(9), z1 = x(4), z2 = x(5), r1 = x(7), r2 = x(8);
        z(0) = xc - r1 * cos(yaw); 
        z(1) = yc - r1 * sin(yaw); 
        z(2) = z1;              
        z(3) = yaw;              

        yaw += M_PI/2;
        z(4) = xc - r2 * cos(yaw); 
        z(5) = yc - r2 * sin(yaw); 
        z(6) = z2;              
        z(7) = yaw;              

        yaw += M_PI/2;
        z(8) = xc - r1 * cos(yaw); 
        z(9) = yc - r1 * sin(yaw); 
        z(10)= z1;              
        z(11)= yaw;              

        yaw += M_PI/2;
        z(12)= xc - r2 * cos(yaw); 
        z(13)= yc - r2 * sin(yaw); 
        z(14)= z2;              
        z(15)= yaw;              
        return z;
    };

    // 观测函数的雅可比矩阵
    j_h = [](const Eigen::VectorXd &x)
    {
        Eigen::MatrixXd h(16, 11);
        double yaw = x(9), r1 = x(7), r2 = x(8);
        // clang-format off
        //    xc   vx   yc   vy   z1   z2   vz   r1              r2                  yaw                vyaw
        h <<  1,   0,   0,   0,   0,   0,   0,   -cos(yaw),      0,                  r1*sin(yaw),           0,
              0,   0,   1,   0,   0,   0,   0,   -sin(yaw),      0,                  -r1*cos(yaw),          0,
              0,   0,   0,   0,   1,   0,   0,   0,              0,                  0,                     0,
              0,   0,   0,   0,   0,   0,   0,   0,              0,                  1,                     0,

              1,   0,   0,   0,   0,   0,   0,   0,              -cos(yaw+M_PI/2),   r2*sin(yaw+M_PI/2),    0,
              0,   0,   1,   0,   0,   0,   0,   0,              -sin(yaw+M_PI/2),   -r2*cos(yaw+M_PI/2),   0,
              0,   0,   0,   0,   0,   1,   0,   0,              0,                  0,                     0,
              0,   0,   0,   0,   0,   0,   0,   0,              0,                  1,                     0,

              1,   0,   0,   0,   0,   0,   0,   -cos(yaw+M_PI), 0,                  r1*sin(yaw+M_PI),      0,
              0,   0,   1,   0,   0,   0,   0,   -sin(yaw+M_PI), 0,                  -r1*cos(yaw+M_PI),     0,
              0,   0,   0,   0,   1,   0,   0,   0,              0,                  0,                     0,
              0,   0,   0,   0,   0,   0,   0,   0,              0,                  1,                     0,
              
              1,   0,   0,   0,   0,   0,   0,   0,              -cos(yaw+3*M_PI/2), r2*sin(yaw+3*M_PI/2),  0,
              0,   0,   1,   0,   0,   0,   0,   0,              -sin(yaw+3*M_PI/2), -r2*cos(yaw+3*M_PI/2), 0,                 
              0,   0,   0,   0,   0,   1,   0,   0,              0,                  0,                     0,
              0,   0,   0,   0,   0,   0,   0,   0,              0,                  1,                     0;
              
        // clang-format on
        return h;
    };

    // 过程噪声协方差矩阵 u_q_0
    u_q = [this, &gp]()
    {
        Eigen::MatrixXd q(11, 11);
        double t{dt}, x{gp.s2qxyz}, y{gp.s2qyaw}, r{gp.s2qr};
        // 计算各种噪声参数
        double q_x_x{pow(t, 4) / 4 * x}, q_x_vx{pow(t, 3) / 2 * x}, q_vx_vx{pow(t, 2) * x};
        double q_y_y{pow(t, 4) / 4 * y}, q_y_vy{pow(t, 3) / 2 * y}, q_vy_vy{pow(t, 2) * y};
        double q_r{pow(t, 4) / 4 * r};
        //    xc      vx      yc      vy      z1      z2      vz      r1      r2      yaw     vyaw
        q <<  q_x_x,  q_x_vx, 0,      0,      0,      0,      0,      0,      0,      0,      0,
              q_x_vx, q_vx_vx,0,      0,      0,      0,      0,      0,      0,      0,      0,
              0,      0,      q_x_x,  q_x_vx, 0,      0,      0,      0,      0,      0,      0,
              0,      0,      q_x_vx, q_vx_vx,0,      0,      0,      0,      0,      0,      0,
              0,      0,      0,      0,      q_x_x,  0,      0,      0,      0,      0,      0,
              0,      0,      0,      0,      0,      q_x_x,  0,      0,      0,      0,      0,
              0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
              0,      0,      0,      0,      0,      0,      0,      q_r,    0,      0,      0,
              0,      0,      0,      0,      0,      0,      0,      0,      q_r,    0,      0,
              0,      0,      0,      0,      0,      0,      0,      0,      0,      q_y_y,  q_y_vy,
              0,      0,      0,      0,      0,      0,      0,      0,      0,      q_y_vy, q_vy_vy;
        // clang-format on
        return q;
    };

    // 观测噪声协方差矩阵 u_r_0
    u_r = [this, &gp](const Eigen::VectorXd &z)
    {
        Eigen::DiagonalMatrix<double, 16> r;
        double xy = gp.r_xy_factor;
        r.diagonal() << abs(xy * z[0]) * r_xy_correction[0],  abs(xy * z[1]) * r_xy_correction[0],  gp.r_z, r_yaw_corrected,
                        abs(xy * z[4]) * r_xy_correction[1],  abs(xy * z[5]) * r_xy_correction[1],  gp.r_z, r_yaw_corrected,
                        abs(xy * z[8]) * r_xy_correction[2],  abs(xy * z[9]) * r_xy_correction[2],  gp.r_z, r_yaw_corrected,
                        abs(xy * z[12]) * r_xy_correction[3], abs(xy * z[13]) * r_xy_correction[3], gp.r_z, r_yaw_corrected; // 定义观测噪声
        return r;
    };

    nomolize_residual = [](const Eigen::VectorXd &z)
    {
        Eigen::VectorXd nz = z;
        nz(3) = atan2(sin(nz(3)), cos(nz(3)));
        nz(7) = atan2(sin(nz(7)), cos(nz(7)));
        nz(11) = atan2(sin(nz(11)), cos(nz(11)));
        if(z.size() > 12) nz(15) = atan2(sin(nz(15)), cos(nz(15)));
        return nz;
    };

    // 前哨站
    f_outpost = [this](const Eigen::VectorXd &x)
    {
        // 更新状态：位置和速度
        Eigen::VectorXd x_new = x;
        x_new(7) = OUTPOSE_R;
        x_new(8) = OUTPOSE_R;        
        if(x_new(10) > 2.35) x_new(10) = 0.8 * M_PI;
        else if(x_new(10) < -2.35) x_new(10) = -0.8 * M_PI;
        x_new(9) += x(10)* dt; // 更新yaw
        return x_new;
    };

    // 状态转移函数的雅可比矩阵
    j_f_outpost = [this](const Eigen::VectorXd &)
    {
        Eigen::MatrixXd f(11, 11);
        // clang-format off
        //    xc   0    yc   0    z    0    0    r    0    yaw  vyaw
        f <<  1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   dt,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1;
        // clang-format on
        return f;
    };

    // 观测函数
    h_outpost = [this](const Eigen::VectorXd &x)
    {
        Eigen::VectorXd z(12);
        double xc = x(0), yc = x(2), yaw = x(9), z_ = x(4), r = x(7);
        z(0) = xc - r * cos(yaw); 
        z(1) = yc - r * sin(yaw); 
        z(2) = z_;              
        z(3) = yaw;              

        yaw += M_PI/3*2;
        z(4) = xc - r * cos(yaw); 
        z(5) = yc - r * sin(yaw); 
        z(6) = z_;              
        z(7) = yaw;              

        yaw += M_PI/3*2;
        z(8) = xc - r * cos(yaw); 
        z(9) = yc - r * sin(yaw); 
        z(10)= z_;              
        z(11)= yaw;

        return z;
    };

    // 观测函数的雅可比矩阵
    j_h_outpost = [](const Eigen::VectorXd &x)
    {
        Eigen::MatrixXd h(12, 11);
        double yaw = x(9), r = x(7);
        // clang-format off
        //    xc   0    yc   0    z    0    0    r                   0    yaw                    vyaw
        h <<  1,   0,   0,   0,   0,   0,   0,   -cos(yaw),          0,   r*sin(yaw),            0,
              0,   0,   1,   0,   0,   0,   0,   -sin(yaw),          0,   -r*cos(yaw),           0,
              0,   0,   0,   0,   1,   0,   0,   0,                  0,   0,                     0,
              0,   0,   0,   0,   0,   0,   0,   0,                  0,   1,                     0,

              1,   0,   0,   0,   0,   0,   0,   -cos(yaw+M_PI/3*2), 0,   r*sin(yaw+M_PI/3*2),   0,
              0,   0,   1,   0,   0,   0,   0,   -sin(yaw+M_PI/3*2), 0,   -r*cos(yaw+M_PI/3*2),  0,
              0,   0,   0,   0,   1,   0,   0,   0,                  0,   0,                     0,
              0,   0,   0,   0,   0,   0,   0,   0,                  0,   1,                     0,

              1,   0,   0,   0,   0,   0,   0,   -cos(yaw+M_PI/3*4), 0,   r*sin(yaw+M_PI/3*4),   0,
              0,   0,   1,   0,   0,   0,   0,   -sin(yaw+M_PI/3*4), 0,   -r*cos(yaw+M_PI/3*4),  0,
              0,   0,   0,   0,   1,   0,   0,   0,                  0,   0,                     0,
              0,   0,   0,   0,   0,   0,   0,   0,                  0,   1,                     0;
              
        // clang-format on
        return h;
    };

    // 过程噪声协方差矩阵 u_q_0
    u_q_outpost = [this, &gp]()
    {
        Eigen::MatrixXd q(11, 11);
        double t{dt}, x{gp.s2qxyz_outpost}, y{gp.s2qyaw_outpost};
        // 计算各种噪声参数
        double q_x_x{pow(t, 4) / 4 * x};
        double q_y_y{pow(t, 4) / 4 * y}, q_y_vy{pow(t, 3) / 2 * y}, q_vy_vy{pow(t, 2) * y};
        //    xc      0       yc      0       z       0       0       r       0       yaw     vyaw
        q <<  q_x_x,  0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
              0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
              0,      0,      q_x_x,  0,      0,      0,      0,      0,      0,      0,      0,
              0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
              0,      0,      0,      0,      q_x_x,  0,      0,      0,      0,      0,      0,
              0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
              0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
              0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
              0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
              0,      0,      0,      0,      0,      0,      0,      0,      0,      q_y_y,  q_y_vy,
              0,      0,      0,      0,      0,      0,      0,      0,      0,      q_y_vy, q_vy_vy;
        // clang-format on
        return q;
    };

    // 观测噪声协方差矩阵 u_r_0
    u_r_outpost = [this, &gp](const Eigen::VectorXd &z)
    {
        Eigen::DiagonalMatrix<double, 12> r;
        double xy = gp.r_xy_factor_outpost;
        
        r.diagonal() << abs(xy * z[0]) * r_xy_correction[0],  abs(xy * z[1]) * r_xy_correction[0],  gp.r_z_outpost * 10, r_yaw_corrected * 10,
                        abs(xy * z[4]) * r_xy_correction[1],  abs(xy * z[5]) * r_xy_correction[1],  gp.r_z_outpost * 10, r_yaw_corrected * 10,
                        abs(xy * z[8]) * r_xy_correction[2],  abs(xy * z[9]) * r_xy_correction[2],  gp.r_z_outpost * 10, r_yaw_corrected * 10; // 定义观测噪声
        return r;
    };
}
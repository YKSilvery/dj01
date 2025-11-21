#include "aim_auto/ekf_tracker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <rclcpp/logging.hpp>

using aim_auto::ArmorDetection;
using aim_auto::EkfTracker;
using aim_auto::ExtendedKalmanFilter;
using aim_auto::TrackState;

namespace {
constexpr double kCostInf = 1e12;
constexpr double kDefaultDt = 0.02;

inline double wrapAngle(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

inline double yawDistance(double lhs, double rhs)
{
  return std::abs(wrapAngle(lhs - rhs));
}

std::vector<int> solveHungarian(const std::vector<std::vector<double>> &cost_matrix,
                                double cost_threshold)
{
  const std::size_t n = cost_matrix.size();
  const std::size_t m = n > 0 ? cost_matrix.front().size() : 0;
  const std::size_t dim = std::max(n, m);

  std::vector<double> u(dim + 1, 0.0);
  std::vector<double> v(dim + 1, 0.0);
  std::vector<int> p(dim + 1, 0);
  std::vector<int> way(dim + 1, 0);

  for (std::size_t i = 1; i <= n; ++i) {
    p[0] = static_cast<int>(i);
    int j0 = 0;
    std::vector<double> minv(dim + 1, kCostInf);
    std::vector<char> used(dim + 1, false);
    do {
      used[j0] = true;
      const int i0 = p[j0];
      int j1 = 0;
      double delta = kCostInf;
      for (std::size_t j = 1; j <= dim; ++j) {
        if (used[j]) {
          continue;
        }
        double cur = kCostInf;
        if (j <= m) {
          cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j];
        }
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = static_cast<int>(j);
        }
      }
      for (std::size_t j = 0; j <= dim; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);
    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }

  std::vector<int> assignment(n, -1);
  for (std::size_t j = 1; j <= dim; ++j) {
    if (p[j] <= 0 || p[j] > static_cast<int>(n) || j > m) {
      continue;
    }
    const int row = p[j] - 1;
    const int col = static_cast<int>(j - 1);
    const double cost = cost_matrix[row][col];
    if (cost < cost_threshold && cost < kCostInf * 0.5) {
      assignment[row] = col;
    }
  }
  return assignment;
}
}  // namespace

ExtendedKalmanFilter::ExtendedKalmanFilter(const VecVecFunc &f,
                                           const VecVecFunc &h,
                                           const VecMatFunc &jacobian_f,
                                           const VecMatFunc &jacobian_h,
                                           const VoidMatFunc &update_q,
                                           const VecMatFunc &update_r,
                                           const VecVecFunc &normalize_residual,
                                           const Eigen::MatrixXd &p0,
                                           const Eigen::VectorXd &x0)
: f_(f),
  h_(h),
  jacobian_f_(jacobian_f),
  jacobian_h_(jacobian_h),
  update_q_(update_q),
  update_r_(update_r),
  normalize_residual_(normalize_residual),
  p_pri_(p0),
  p_post_(p0),
  x_pri_(x0),
  x_post_(x0),
  identity_(Eigen::MatrixXd::Identity(p0.rows(), p0.cols()))
{}

Eigen::VectorXd ExtendedKalmanFilter::predict()
{
  f_cache_ = jacobian_f_(x_post_);
  q_cache_ = update_q_();

  x_pri_ = f_(x_post_);
  p_pri_ = f_cache_ * p_post_ * f_cache_.transpose() + q_cache_;

  x_post_ = x_pri_;
  p_post_ = p_pri_;
  return x_pri_;
}

Eigen::VectorXd ExtendedKalmanFilter::update(const Eigen::VectorXd &z)
{
  h_cache_ = jacobian_h_(x_pri_);
  r_cache_ = update_r_(z);
  const Eigen::MatrixXd s = h_cache_ * p_pri_ * h_cache_.transpose() + r_cache_;
  k_cache_ = p_pri_ * h_cache_.transpose() * s.inverse();
  x_post_ = x_pri_ + k_cache_ * normalize_residual_(z - h_(x_pri_));
  p_post_ = (identity_ - k_cache_ * h_cache_) * p_pri_;
  return x_post_;
}

void ExtendedKalmanFilter::setState(const Eigen::VectorXd &x0)
{
  x_post_ = x0;
  x_pri_ = x0;
}

EkfTracker::EkfTracker(const EkfParams &params)
: params_(params)
{
  active_classes_.fill(false);
  setupModels();
}

void EkfTracker::reset()
{
  filters_.clear();
  measurements_.clear();
  lost_frames_.clear();
  class_ids_.clear();
  predicted_slots_.clear();
  active_classes_.fill(false);
  primary_index_ = -1;
  time_initialized_ = false;
  dt_ = kDefaultDt;
}

void EkfTracker::setParameters(const EkfParams &params)
{
  params_ = params;
}

void EkfTracker::setupModels()
{
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  f_regular_ = [this](const VectorXd &x) {
    VectorXd x_new = x;
    x_new(0) += x(1) * dt_;
    x_new(2) += x(3) * dt_;
    x_new(9) += x(10) * dt_;
    return x_new;
  };

  jf_regular_ = [this](const VectorXd &) {
    MatrixXd f = MatrixXd::Zero(11, 11);
    // clang-format off
    f << 1, dt_, 0,    0,    0, 0, 0, 0, 0, 0,    0,
         0, 1,   0,    0,    0, 0, 0, 0, 0, 0,    0,
         0, 0,   1,    dt_,  0, 0, 0, 0, 0, 0,    0,
         0, 0,   0,    1,    0, 0, 0, 0, 0, 0,    0,
         0, 0,   0,    0,    1, 0, 0, 0, 0, 0,    0,
         0, 0,   0,    0,    0, 1, 0, 0, 0, 0,    0,
         0, 0,   0,    0,    0, 0, 1, 0, 0, 0,    0,
         0, 0,   0,    0,    0, 0, 0, 1, 0, 0,    0,
         0, 0,   0,    0,    0, 0, 0, 0, 1, 0,    0,
         0, 0,   0,    0,    0, 0, 0, 0, 0, 1, dt_,
         0, 0,   0,    0,    0, 0, 0, 0, 0, 0,    1;
    // clang-format on
    return f;
  };

  h_regular_ = [](const VectorXd &x) {
    VectorXd z(16);
    const double xc = x(0);
    const double yc = x(2);
    double yaw = x(9);
    const double z1 = x(4);
    const double z2 = x(5);
    const double r1 = x(7);
    const double r2 = x(8);

    z(0) = xc - r1 * std::cos(yaw);
    z(1) = yc - r1 * std::sin(yaw);
    z(2) = z1;
    z(3) = yaw;

    yaw += M_PI_2;
    z(4) = xc - r2 * std::cos(yaw);
    z(5) = yc - r2 * std::sin(yaw);
    z(6) = z2;
    z(7) = yaw;

    yaw += M_PI_2;
    z(8) = xc - r1 * std::cos(yaw);
    z(9) = yc - r1 * std::sin(yaw);
    z(10) = z1;
    z(11) = yaw;

    yaw += M_PI_2;
    z(12) = xc - r2 * std::cos(yaw);
    z(13) = yc - r2 * std::sin(yaw);
    z(14) = z2;
    z(15) = yaw;
    return z;
  };

  jh_regular_ = [](const VectorXd &x) {
    MatrixXd h = MatrixXd::Zero(16, 11);
    const double yaw = x(9);
    const double r1 = x(7);
    const double r2 = x(8);
    // clang-format off
    h << 1, 0, 0, 0, 0, 0, 0, -std::cos(yaw),              0,  r1 * std::sin(yaw),              0,
         0, 0, 1, 0, 0, 0, 0, -std::sin(yaw),              0, -r1 * std::cos(yaw),              0,
         0, 0, 0, 0, 1, 0, 0,  0,                          0,  0,                              0,
         0, 0, 0, 0, 0, 0, 0,  0,                          0,  1,                              0,

         1, 0, 0, 0, 0, 0, 0,  0,             -std::cos(yaw + M_PI_2),  r2 * std::sin(yaw + M_PI_2), 0,
         0, 0, 1, 0, 0, 0, 0,  0,             -std::sin(yaw + M_PI_2), -r2 * std::cos(yaw + M_PI_2),0,
         0, 0, 0, 0, 0, 1, 0,  0,                          0,  0,                              0,
         0, 0, 0, 0, 0, 0, 0,  0,                          0,  1,                              0,

         1, 0, 0, 0, 0, 0, 0, -std::cos(yaw + M_PI),       0,  r1 * std::sin(yaw + M_PI),       0,
         0, 0, 1, 0, 0, 0, 0, -std::sin(yaw + M_PI),       0, -r1 * std::cos(yaw + M_PI),       0,
         0, 0, 0, 0, 1, 0, 0,  0,                          0,  0,                              0,
         0, 0, 0, 0, 0, 0, 0,  0,                          0,  1,                              0,

         1, 0, 0, 0, 0, 0, 0,  0,             -std::cos(yaw + 3.0 * M_PI_2), r2 * std::sin(yaw + 3.0 * M_PI_2), 0,
         0, 0, 1, 0, 0, 0, 0,  0,             -std::sin(yaw + 3.0 * M_PI_2),-r2 * std::cos(yaw + 3.0 * M_PI_2),0,
         0, 0, 0, 0, 0, 1, 0,  0,                          0,  0,                              0,
         0, 0, 0, 0, 0, 0, 0,  0,                          0,  1,                              0;
    // clang-format on
    return h;
  };

  q_regular_ = [this]() {
    Eigen::MatrixXd q = Eigen::MatrixXd::Zero(11, 11);
    const double t = dt_;
    const double x = params_.s2qxyz;
    const double y = params_.s2qyaw;
    const double r = params_.s2qr;

    const double q_x_x = std::pow(t, 4) / 4.0 * x;
    const double q_x_vx = std::pow(t, 3) / 2.0 * x;
    const double q_vx_vx = std::pow(t, 2) * x;
    const double q_y_y = std::pow(t, 4) / 4.0 * y;
    const double q_y_vy = std::pow(t, 3) / 2.0 * y;
    const double q_vy_vy = std::pow(t, 2) * y;
    const double q_r = std::pow(t, 4) / 4.0 * r;

    q(0, 0) = q_x_x;
    q(0, 1) = q_x_vx;
    q(1, 0) = q_x_vx;
    q(1, 1) = q_vx_vx;

    q(2, 2) = q_x_x;
    q(2, 3) = q_x_vx;
    q(3, 2) = q_x_vx;
    q(3, 3) = q_vx_vx;

    q(4, 4) = q_x_x;
    q(5, 5) = q_x_x;

    q(7, 7) = q_r;
    q(8, 8) = q_r;

    q(9, 9) = q_y_y;
    q(9, 10) = q_y_vy;
    q(10, 9) = q_y_vy;
    q(10, 10) = q_vy_vy;
    return q;
  };

  r_regular_ = [this](const VectorXd &z) {
    Eigen::DiagonalMatrix<double, 16> diag;
    const double xy = params_.r_xy_factor;

    diag.diagonal() <<
      std::abs(xy * z(0)) * r_xy_correction_buffer_[0],
      std::abs(xy * z(1)) * r_xy_correction_buffer_[0],
      params_.r_z,
      yaw_noise_correction_,
      std::abs(xy * z(4)) * r_xy_correction_buffer_[1],
      std::abs(xy * z(5)) * r_xy_correction_buffer_[1],
      params_.r_z,
      yaw_noise_correction_,
      std::abs(xy * z(8)) * r_xy_correction_buffer_[2],
      std::abs(xy * z(9)) * r_xy_correction_buffer_[2],
      params_.r_z,
      yaw_noise_correction_,
      std::abs(xy * z(12)) * r_xy_correction_buffer_[3],
      std::abs(xy * z(13)) * r_xy_correction_buffer_[3],
      params_.r_z,
      yaw_noise_correction_;
    return diag.toDenseMatrix();
  };

  normalize_residual_ = [](const VectorXd &z) {
    VectorXd nz = z;
    if (nz.size() >= 4) {
      nz(3) = wrapAngle(nz(3));
    }
    if (nz.size() >= 8) {
      nz(7) = wrapAngle(nz(7));
    }
    if (nz.size() >= 12) {
      nz(11) = wrapAngle(nz(11));
    }
    if (nz.size() >= 16) {
      nz(15) = wrapAngle(nz(15));
    }
    return nz;
  };

  f_outpost_ = [this](const VectorXd &x) {
    VectorXd x_new = x;
    x_new(7) = kOutpostRadius;
    x_new(8) = kOutpostRadius;
    if (x_new(10) > 2.35) {
      x_new(10) = 0.8 * M_PI;
    } else if (x_new(10) < -2.35) {
      x_new(10) = -0.8 * M_PI;
    }
    x_new(9) += x(10) * dt_;
    x_new(5) = x_new(4);
    return x_new;
  };

  jf_outpost_ = [this](const VectorXd &) {
    MatrixXd f = MatrixXd::Zero(11, 11);
    // clang-format off
    f << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0,    0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
         0, 0, 0, 0, 1, 0, 0, 0, 0, 0,    0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
         0, 0, 0, 0, 0, 0, 0, 1, 0, 0,    0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, dt_,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    1;
    // clang-format on
    return f;
  };

  h_outpost_ = [](const VectorXd &x) {
    VectorXd z(12);
    const double xc = x(0);
    const double yc = x(2);
    double yaw = x(9);
    const double z_val = x(4);
    const double r = x(7);

    z(0) = xc - r * std::cos(yaw);
    z(1) = yc - r * std::sin(yaw);
    z(2) = z_val;
    z(3) = yaw;

    yaw += 2.0 * M_PI / 3.0;
    z(4) = xc - r * std::cos(yaw);
    z(5) = yc - r * std::sin(yaw);
    z(6) = z_val;
    z(7) = yaw;

    yaw += 2.0 * M_PI / 3.0;
    z(8) = xc - r * std::cos(yaw);
    z(9) = yc - r * std::sin(yaw);
    z(10) = z_val;
    z(11) = yaw;
    return z;
  };

  jh_outpost_ = [](const VectorXd &x) {
    MatrixXd h = MatrixXd::Zero(12, 11);
    const double yaw = x(9);
    const double r = x(7);
    // clang-format off
    h << 1, 0, 0, 0, 0, 0, 0, -std::cos(yaw),              0,  r * std::sin(yaw),              0,
         0, 0, 1, 0, 0, 0, 0, -std::sin(yaw),              0, -r * std::cos(yaw),              0,
         0, 0, 0, 0, 1, 0, 0,  0,                          0,  0,                              0,
         0, 0, 0, 0, 0, 0, 0,  0,                          0,  1,                              0,

         1, 0, 0, 0, 0, 0, 0, -std::cos(yaw + 2.0 * M_PI / 3.0), 0,  r * std::sin(yaw + 2.0 * M_PI / 3.0), 0,
         0, 0, 1, 0, 0, 0, 0, -std::sin(yaw + 2.0 * M_PI / 3.0), 0, -r * std::cos(yaw + 2.0 * M_PI / 3.0), 0,
         0, 0, 0, 0, 1, 0, 0,  0,                                0,  0,                                   0,
         0, 0, 0, 0, 0, 0, 0,  0,                                0,  1,                                   0,

         1, 0, 0, 0, 0, 0, 0, -std::cos(yaw + 4.0 * M_PI / 3.0), 0,  r * std::sin(yaw + 4.0 * M_PI / 3.0), 0,
         0, 0, 1, 0, 0, 0, 0, -std::sin(yaw + 4.0 * M_PI / 3.0), 0, -r * std::cos(yaw + 4.0 * M_PI / 3.0), 0,
         0, 0, 0, 0, 1, 0, 0,  0,                                0,  0,                                   0,
         0, 0, 0, 0, 0, 0, 0,  0,                                0,  1,                                   0;
    // clang-format on
    return h;
  };

  q_outpost_ = [this]() {
    Eigen::MatrixXd q = Eigen::MatrixXd::Zero(11, 11);
    const double t = dt_;
    const double x = params_.s2qxyz_outpost;
    const double y = params_.s2qyaw_outpost;

    const double q_x_x = std::pow(t, 4) / 4.0 * x;
    const double q_y_y = std::pow(t, 4) / 4.0 * y;
    const double q_y_vy = std::pow(t, 3) / 2.0 * y;
    const double q_vy_vy = std::pow(t, 2) * y;

    q(0, 0) = q_x_x;
    q(2, 2) = q_x_x;
    q(4, 4) = q_x_x;

    q(9, 9) = q_y_y;
    q(9, 10) = q_y_vy;
    q(10, 9) = q_y_vy;
    q(10, 10) = q_vy_vy;
    return q;
  };

  r_outpost_ = [this](const VectorXd &z) {
    Eigen::DiagonalMatrix<double, 12> diag;
    const double xy = params_.r_xy_factor_outpost;
    diag.diagonal() <<
      std::abs(xy * z(0)) * r_xy_correction_buffer_[0],
      std::abs(xy * z(1)) * r_xy_correction_buffer_[0],
      params_.r_z_outpost * 10.0,
      yaw_noise_correction_ * 10.0,
      std::abs(xy * z(4)) * r_xy_correction_buffer_[1],
      std::abs(xy * z(5)) * r_xy_correction_buffer_[1],
      params_.r_z_outpost * 10.0,
      yaw_noise_correction_ * 10.0,
      std::abs(xy * z(8)) * r_xy_correction_buffer_[2],
      std::abs(xy * z(9)) * r_xy_correction_buffer_[2],
      params_.r_z_outpost * 10.0,
      yaw_noise_correction_ * 10.0;
    return diag.toDenseMatrix();
  };
}

ArmorDetection EkfTracker::calcArmor(double xc,
                                     double yc,
                                     double z,
                                     double r,
                                     double yaw,
                                     int class_id) const
{
  ArmorDetection armor;
  armor.class_id = class_id;
  armor.position = Eigen::Vector3d(xc - r * std::cos(yaw),
                                   yc - r * std::sin(yaw),
                                   z);
  armor.yaw = yaw;
  return armor;
}

double EkfTracker::matchingCost(const ArmorDetection &detection,
                                 const PredictedSlot &slot) const
{
  const double spatial = (detection.position - slot.armor.position).norm();
  const double yaw_err = yawDistance(detection.yaw, slot.armor.yaw);
  const double dz = std::abs(detection.position.z() - slot.armor.position.z());
  return 0.1 * spatial + 2000.0 * yaw_err * yaw_err + 10.0 * dz;  // Reduced spatial weight to 0.1
}

void EkfTracker::createTrack(const ArmorDetection &detection)
{
  if (detection.class_id < 0 || detection.class_id >= static_cast<int>(active_classes_.size())) {
    return;
  }
  if (active_classes_[detection.class_id]) {
    return;
  }

  Eigen::MatrixXd p0 = Eigen::MatrixXd::Identity(11, 11) * params_.s2p0xyr;
  p0(9, 9) = params_.s2p0yaw;
  Eigen::VectorXd x0(11);

  if (!isOutpost(detection.class_id)) {
    const Eigen::Vector3d center = detection.position +
      params_.r_initial * Eigen::Vector3d(std::cos(detection.yaw), std::sin(detection.yaw), 0.0);
    x0 << center.x(), 0.0, center.y(), 0.0,
      detection.position.z(), detection.position.z(), 0.0,
      params_.r_initial, params_.r_initial,
      detection.yaw, 0.0;
    filters_.emplace_back(f_regular_, h_regular_, jf_regular_, jh_regular_,
                          q_regular_, r_regular_, normalize_residual_, p0, x0);
  } else {
    const Eigen::Vector3d center = detection.position +
      kOutpostRadius * Eigen::Vector3d(std::cos(detection.yaw), std::sin(detection.yaw), 0.0);
    x0 << center.x(), 0.0, center.y(), 0.0,
      detection.position.z(), 0.0, 0.0,
      kOutpostRadius, kOutpostRadius,
      detection.yaw, 0.0;
    filters_.emplace_back(f_outpost_, h_outpost_, jf_outpost_, jh_outpost_,
                          q_outpost_, r_outpost_, normalize_residual_, p0, x0);
  }

  measurements_.push_back(Eigen::VectorXd::Zero(16));
  lost_frames_.push_back(0);
  class_ids_.push_back(detection.class_id);
  active_classes_[detection.class_id] = true;
}

void EkfTracker::removeTrack(std::size_t index)
{
  if (index >= filters_.size()) {
    return;
  }
  const int class_id = class_ids_[index];
  if (class_id >= 0 && class_id < static_cast<int>(active_classes_.size())) {
    active_classes_[class_id] = false;
  }
  filters_.erase(filters_.begin() + static_cast<long>(index));
  measurements_.erase(measurements_.begin() + static_cast<long>(index));
  lost_frames_.erase(lost_frames_.begin() + static_cast<long>(index));
  class_ids_.erase(class_ids_.begin() + static_cast<long>(index));
  if (filters_.empty()) {
    primary_index_ = -1;
  } else if (primary_index_ >= static_cast<int>(filters_.size())) {
    primary_index_ = static_cast<int>(filters_.size()) - 1;
  }
}

void EkfTracker::refineMeasurement(std::size_t index)
{
  r_xy_correction_buffer_.fill(1.0);
  yaw_noise_correction_ = params_.r_yaw;

  const Eigen::VectorXd x = filters_[index].state();
  const double xc = x(0);
  const double yc = x(2);
  const double z1 = x(4);
  const double z2 = x(5);
  const double r1 = x(7);
  const double r2 = x(8);
  const double yaw = x(9);

  Eigen::VectorXd &z = measurements_[index];
  auto has_segment = [&z](int seg) {
    return z.segment<4>(seg * 4).norm() > 1e-6;
  };
  auto set_segment = [&z](int seg, const ArmorDetection &armor) {
    const int base = seg * 4;
    z(base + 0) = armor.position.x();
    z(base + 1) = armor.position.y();
    z(base + 2) = armor.position.z();
    z(base + 3) = armor.yaw;
  };

  if (!isOutpost(class_ids_[index])) {
    if (has_segment(0)) {
  const Eigen::Vector4d seg = z.segment<4>(0);
      const double radius = std::hypot(seg(0) - xc, seg(1) - yc);
      const ArmorDetection armor = calcArmor(xc, yc, seg(2), radius, seg(3) + M_PI, class_ids_[index]);
      set_segment(2, armor);
      r_xy_correction_buffer_[2] *= 10.0;
    } else if (has_segment(2)) {
  const Eigen::Vector4d seg = z.segment<4>(8);
      const double radius = std::hypot(seg(0) - xc, seg(1) - yc);
      const ArmorDetection armor = calcArmor(xc, yc, seg(2), radius, seg(3) - M_PI, class_ids_[index]);
      set_segment(0, armor);
      r_xy_correction_buffer_[0] *= 10.0;
    }

    if (has_segment(1)) {
  const Eigen::Vector4d seg = z.segment<4>(4);
      const double radius = std::hypot(seg(0) - xc, seg(1) - yc);
      const ArmorDetection armor = calcArmor(xc, yc, seg(2), radius, seg(3) + M_PI, class_ids_[index]);
      set_segment(3, armor);
      r_xy_correction_buffer_[3] *= 10.0;
    } else if (has_segment(3)) {
  const Eigen::Vector4d seg = z.segment<4>(12);
      const double radius = std::hypot(seg(0) - xc, seg(1) - yc);
      const ArmorDetection armor = calcArmor(xc, yc, seg(2), radius, seg(3) - M_PI, class_ids_[index]);
      set_segment(1, armor);
      r_xy_correction_buffer_[1] *= 10.0;
    }

    if (!has_segment(0) && !has_segment(2)) {
      const ArmorDetection armor_a = calcArmor(xc, yc, z1, r1, yaw + M_PI_2, class_ids_[index]);
      const ArmorDetection armor_b = calcArmor(xc, yc, z1, r1, yaw - M_PI_2, class_ids_[index]);
      set_segment(0, armor_a);
      set_segment(2, armor_b);
      yaw_noise_correction_ *= 10.0;
      r_xy_correction_buffer_[0] *= 10.0;
      r_xy_correction_buffer_[2] *= 10.0;
    }

    if (!has_segment(1) && !has_segment(3)) {
      const ArmorDetection armor_a = calcArmor(xc, yc, z2, r2, yaw + M_PI_2, class_ids_[index]);
      const ArmorDetection armor_b = calcArmor(xc, yc, z2, r2, yaw - M_PI_2, class_ids_[index]);
      set_segment(1, armor_a);
      set_segment(3, armor_b);
      yaw_noise_correction_ *= 10.0;
      r_xy_correction_buffer_[1] *= 10.0;
      r_xy_correction_buffer_[3] *= 10.0;
    }
  } else {
    if (has_segment(0)) {
  const Eigen::Vector4d seg = z.segment<4>(0);
      const ArmorDetection armor_a = calcArmor(xc, yc, seg(2), kOutpostRadius, seg(3) + 2.0 * M_PI / 3.0, class_ids_[index]);
      const ArmorDetection armor_b = calcArmor(xc, yc, seg(2), kOutpostRadius, seg(3) + 4.0 * M_PI / 3.0, class_ids_[index]);
      set_segment(1, armor_a);
      set_segment(2, armor_b);
      r_xy_correction_buffer_[1] *= 10.0;
      r_xy_correction_buffer_[2] *= 10.0;
    } else if (has_segment(1)) {
  const Eigen::Vector4d seg = z.segment<4>(4);
      const ArmorDetection armor_a = calcArmor(xc, yc, seg(2), kOutpostRadius, seg(3) + 2.0 * M_PI / 3.0, class_ids_[index]);
      const ArmorDetection armor_b = calcArmor(xc, yc, seg(2), kOutpostRadius, seg(3) + 4.0 * M_PI / 3.0, class_ids_[index]);
      set_segment(2, armor_a);
      set_segment(0, armor_b);
      r_xy_correction_buffer_[2] *= 10.0;
      r_xy_correction_buffer_[0] *= 10.0;
    } else if (has_segment(2)) {
  const Eigen::Vector4d seg = z.segment<4>(8);
      const ArmorDetection armor_a = calcArmor(xc, yc, seg(2), kOutpostRadius, seg(3) + 2.0 * M_PI / 3.0, class_ids_[index]);
      const ArmorDetection armor_b = calcArmor(xc, yc, seg(2), kOutpostRadius, seg(3) + 4.0 * M_PI / 3.0, class_ids_[index]);
      set_segment(0, armor_a);
      set_segment(1, armor_b);
      r_xy_correction_buffer_[0] *= 10.0;
      r_xy_correction_buffer_[1] *= 10.0;
    }
    z.segment(12, 4).setZero();
  }
}

void EkfTracker::selectPrimaryTrack()
{
  if (filters_.empty()) {
    primary_index_ = -1;
    return;
  }
  primary_index_ = 0;
  int best_lost = lost_frames_[0];
  for (std::size_t i = 1; i < lost_frames_.size(); ++i) {
    if (lost_frames_[i] < best_lost) {
      best_lost = lost_frames_[i];
      primary_index_ = static_cast<int>(i);
    }
  }
}

void EkfTracker::updateDetections(const std::vector<ArmorDetection> &detections,
                                  const rclcpp::Time &stamp)
{
  if (!time_initialized_) {
    dt_ = kDefaultDt;
    last_stamp_ = stamp;
    time_initialized_ = true;
  } else {
    const double dt = (stamp - last_stamp_).seconds();
    last_stamp_ = stamp;
    if (std::isfinite(dt) && dt > 1e-6) {
      dt_ = std::clamp(dt, 1e-4, 0.2);
    } else {
      dt_ = kDefaultDt;
    }
  }

  for (auto &z : measurements_) {
    z.setZero();
  }
  predicted_slots_.clear();

  for (std::size_t i = 0; i < filters_.size(); ++i) {
    const Eigen::VectorXd x = filters_[i].predict();
    const int class_id = class_ids_[i];
    if (isOutpost(class_id)) {
      const ArmorDetection a0 = calcArmor(x(0), x(2), x(4), x(7), x(9), class_id);
      const ArmorDetection a1 = calcArmor(x(0), x(2), x(4), x(7), x(9) + 2.0 * M_PI / 3.0, class_id);
      const ArmorDetection a2 = calcArmor(x(0), x(2), x(4), x(7), x(9) + 4.0 * M_PI / 3.0, class_id);
      predicted_slots_.push_back({static_cast<int>(i), 0, a0});
      predicted_slots_.push_back({static_cast<int>(i), 1, a1});
      predicted_slots_.push_back({static_cast<int>(i), 2, a2});
      predicted_slots_.push_back({static_cast<int>(i), 3, ArmorDetection{-1, Eigen::Vector3d::Zero(), 0.0}});
    } else {
      const ArmorDetection a0 = calcArmor(x(0), x(2), x(4), x(7), x(9), class_id);
      const ArmorDetection a1 = calcArmor(x(0), x(2), x(5), x(8), x(9) + M_PI_2, class_id);
      const ArmorDetection a2 = calcArmor(x(0), x(2), x(4), x(7), x(9) + M_PI, class_id);
      const ArmorDetection a3 = calcArmor(x(0), x(2), x(5), x(8), x(9) + 3.0 * M_PI_2, class_id);
      predicted_slots_.push_back({static_cast<int>(i), 0, a0});
      predicted_slots_.push_back({static_cast<int>(i), 1, a1});
      predicted_slots_.push_back({static_cast<int>(i), 2, a2});
      predicted_slots_.push_back({static_cast<int>(i), 3, a3});
    }
  }

  std::vector<std::vector<double>> cost_matrix(detections.size(),
                                               std::vector<double>(predicted_slots_.size(), kCostInf));

  for (std::size_t i = 0; i < detections.size(); ++i) {
    for (std::size_t j = 0; j < predicted_slots_.size(); ++j) {
      if (predicted_slots_[j].armor.class_id != detections[i].class_id) {
        cost_matrix[i][j] = kCostInf;
      } else {
        cost_matrix[i][j] = matchingCost(detections[i], predicted_slots_[j]);
      }
    }
  }

  std::vector<int> assignment(detections.size(), -1);
  if (!detections.empty() && !predicted_slots_.empty()) {
    assignment = solveHungarian(cost_matrix, params_.cost_threshold);
  }

  // Debug: Print Hungarian matching results
  size_t matched_count = 0;
  for (size_t i = 0; i < assignment.size(); ++i) {
    if (assignment[i] >= 0) {
      const double cost = cost_matrix[i][assignment[i]];
      //RCLCPP_INFO(rclcpp::get_logger("ekf_tracker"), "Detection %zu (class %d) matched to slot %d with cost %.2f", i, detections[i].class_id, assignment[i], cost);
      ++matched_count;
    } else {
      // Find the minimum cost for this detection
      double min_cost = kCostInf;
      size_t best_j = 0;
      bool has_slots = !predicted_slots_.empty();
      for (size_t j = 0; j < cost_matrix[i].size(); ++j) {
        if (cost_matrix[i][j] < min_cost) {
          min_cost = cost_matrix[i][j];
          best_j = j;
        }
      }
      if (has_slots) {
        //RCLCPP_INFO(rclcpp::get_logger("ekf_tracker"), "Detection %zu (class %d) not matched, min cost %.2f to slot %zu (class %d)", i, detections[i].class_id, min_cost, best_j, predicted_slots_[best_j].armor.class_id);
      } else {
        //RCLCPP_INFO(rclcpp::get_logger("ekf_tracker"), "Detection %zu (class %d) not matched, no predicted slots", i, detections[i].class_id);
      }
    }
  }
  //RCLCPP_INFO(rclcpp::get_logger("ekf_tracker"), "Hungarian matching: %zu detections, %zu matched, %zu unmatched",
  //            detections.size(), matched_count, detections.size() - matched_count);

  for (std::size_t i = 0; i < detections.size(); ++i) {
    if (assignment[i] < 0) {
      createTrack(detections[i]);
      continue;
    }
    const auto &slot = predicted_slots_[assignment[i]];
    auto &z = measurements_[slot.track_index];
    const int base = slot.slot_index * 4;
    if (base + 3 < z.size()) {
      z(base + 0) = detections[i].position.x();
      z(base + 1) = detections[i].position.y();
      z(base + 2) = detections[i].position.z();
      z(base + 3) = detections[i].yaw;
    }
  }

  for (std::size_t i = 0; i < filters_.size();) {
    Eigen::VectorXd &z = measurements_[i];
    if (z.norm() < 1e-6) {
      lost_frames_[i] += 1;
      int max_lost = params_.max_lost_frame;
      if (isOutpost(class_ids_[i])) {
        max_lost *= 3;
      }
      if (lost_frames_[i] > max_lost) {
        removeTrack(i);
        continue;
      }
    } else {
      lost_frames_[i] = 0;
      refineMeasurement(i);
      if (!isOutpost(class_ids_[i])) {
        filters_[i].update(z);
      } else {
        filters_[i].update(z.head(12));
        Eigen::VectorXd &state = filters_[i].mutable_state();
        state(7) = kOutpostRadius;
        state(8) = kOutpostRadius;
        state(5) = state(4);
      }
      Eigen::VectorXd &state = filters_[i].mutable_state();
      // Debug: Log state values before sanity check
      //RCLCPP_DEBUG(rclcpp::get_logger("ekf_tracker"), "Tracker class %d state: r1=%.2f, r2=%.2f, vyaw=%.2f",
      //             class_ids_[i], state(7), state(8), state(10));
      const double old_r1 = state(7);
      const double old_r2 = state(8);
      state(7) = std::clamp(state(7), 100.0, 450.0);
      state(8) = std::clamp(state(8), 100.0, 450.0);
      if (old_r1 < 100.0 || old_r1 > 450.0 || old_r2 < 100.0 || old_r2 > 450.0) {
        std::string reason = "Tracker class " + std::to_string(class_ids_[i]) + " removed due to: r1/r2 out of range (r1=" + std::to_string(old_r1) + ", r2=" + std::to_string(old_r2) + ")";
        //RCLCPP_INFO(rclcpp::get_logger("ekf_tracker"), "%s", reason.c_str());
        removeTrack(i);
        continue;
      }
      if (std::abs(state(10)) > 20.0) {
        std::string reason = "Tracker class " + std::to_string(class_ids_[i]) + " removed due to: |vyaw| > 20 ";
        //RCLCPP_INFO(rclcpp::get_logger("ekf_tracker"), "%s", reason.c_str());
        removeTrack(i);
        continue;
      }
    }
    ++i;
  }

  selectPrimaryTrack();

  // Debug: Output lost frames and active trackers
  std::string tracker_info = "Active trackers: ";
  for (size_t i = 0; i < filters_.size(); ++i) {
    tracker_info += "class " + std::to_string(class_ids_[i]) + " (lost " + std::to_string(lost_frames_[i]) + ")";
    if (i < filters_.size() - 1) tracker_info += ", ";
  }
  if (filters_.empty()) {
    tracker_info += "none";
  }
  RCLCPP_INFO(rclcpp::get_logger("ekf_tracker"), "%s", tracker_info.c_str());
}

std::optional<TrackState> EkfTracker::currentState() const
{
  if (primary_index_ < 0 || primary_index_ >= static_cast<int>(filters_.size())) {
    return std::nullopt;
  }
  const Eigen::VectorXd &x = filters_[primary_index_].state();
  TrackState state;
  state.class_id = class_ids_[primary_index_];
  state.xc = x(0);
  state.vx = x(1);
  state.yc = x(2);
  state.vy = x(3);
  state.z1 = x(4);
  state.z2 = x(5);
  state.vz = x(6);
  state.r1 = x(7);
  state.r2 = x(8);
  state.yaw = x(9);
  state.vyaw = x(10);
  return state;
}

std::vector<ArmorDetection> EkfTracker::predictedArmors() const
{
  std::vector<ArmorDetection> armors;
  armors.reserve(filters_.size() * 4);
  for (std::size_t i = 0; i < filters_.size(); ++i) {
    const Eigen::VectorXd &x = filters_[i].state();
    const int class_id = class_ids_[i];
    if (isOutpost(class_id)) {
      armors.push_back(calcArmor(x(0), x(2), x(4), x(7), x(9), class_id));
      armors.push_back(calcArmor(x(0), x(2), x(4), x(7), x(9) + 2.0 * M_PI / 3.0, class_id));
      armors.push_back(calcArmor(x(0), x(2), x(4), x(7), x(9) + 4.0 * M_PI / 3.0, class_id));
    } else {
      armors.push_back(calcArmor(x(0), x(2), x(4), x(7), x(9), class_id));
      armors.push_back(calcArmor(x(0), x(2), x(5), x(8), x(9) + M_PI_2, class_id));
      armors.push_back(calcArmor(x(0), x(2), x(4), x(7), x(9) + M_PI, class_id));
      armors.push_back(calcArmor(x(0), x(2), x(5), x(8), x(9) + 3.0 * M_PI_2, class_id));
    }
  }
  return armors;
}

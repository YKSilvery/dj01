#ifndef AIM_AUTO__EKF_TRACKER_HPP_
#define AIM_AUTO__EKF_TRACKER_HPP_

#include <array>
#include <functional>
#include <optional>
#include <vector>

#include <Eigen/Dense>
#include <rclcpp/time.hpp>

namespace aim_auto {

class ExtendedKalmanFilter {
public:
  using VecVecFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;
  using VecMatFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd &)>;
  using VoidMatFunc = std::function<Eigen::MatrixXd()>;

  ExtendedKalmanFilter() = default;
  ExtendedKalmanFilter(const VecVecFunc &f,
                       const VecVecFunc &h,
                       const VecMatFunc &jacobian_f,
                       const VecMatFunc &jacobian_h,
                       const VoidMatFunc &update_q,
                       const VecMatFunc &update_r,
                       const VecVecFunc &normalize_residual,
                       const Eigen::MatrixXd &p0,
                       const Eigen::VectorXd &x0);

  Eigen::VectorXd predict();
  Eigen::VectorXd update(const Eigen::VectorXd &z);

  [[nodiscard]] const Eigen::VectorXd &state() const noexcept { return x_post_; }
  [[nodiscard]] Eigen::VectorXd &mutable_state() noexcept { return x_post_; }
  void setState(const Eigen::VectorXd &x0);

private:
  VecVecFunc f_;
  VecVecFunc h_;
  VecMatFunc jacobian_f_;
  VecMatFunc jacobian_h_;
  VoidMatFunc update_q_;
  VecMatFunc update_r_;
  VecVecFunc normalize_residual_;

  Eigen::MatrixXd p_pri_;
  Eigen::MatrixXd p_post_;
  Eigen::MatrixXd f_cache_;
  Eigen::MatrixXd h_cache_;
  Eigen::MatrixXd q_cache_;
  Eigen::MatrixXd r_cache_;
  Eigen::MatrixXd k_cache_;

  Eigen::VectorXd x_pri_;
  Eigen::VectorXd x_post_;
  Eigen::MatrixXd identity_;
};

struct EkfParams {
  double cost_threshold = 999999000.0;  // Increased from 1600.0 to handle armor switching during rotation
  int max_lost_frame = 15;
  double s2qxyz = 450.0;
  double s2qyaw = 2.0e-3;
  double s2qr = 100.0;
  double r_xy_factor = 1.0e-5;
  double r_z = 1.0e-3;
  double r_yaw = 5.0;
  double s2qxyz_outpost = 300.0;
  double s2qyaw_outpost = 0.0;
  double r_xy_factor_outpost = 6.0e-5;
  double r_z_outpost = 5.0e-3;
  double s2p0xyr = 1000.0;
  double s2p0yaw = 50.0;
  double r_initial = 200.0;
};

struct ArmorDetection {
  int class_id = 0;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  double yaw = 0.0;
};

struct TrackState {
  int class_id = 0;
  double xc = 0.0;
  double vx = 0.0;
  double yc = 0.0;
  double vy = 0.0;
  double z1 = 0.0;
  double z2 = 0.0;
  double vz = 0.0;
  double r1 = 0.0;
  double r2 = 0.0;
  double yaw = 0.0;
  double vyaw = 0.0;
};

class EkfTracker {
public:
  explicit EkfTracker(const EkfParams &params = EkfParams());

  void reset();
  void setParameters(const EkfParams &params);

  void updateDetections(const std::vector<ArmorDetection> &detections,
                        const rclcpp::Time &stamp);

  [[nodiscard]] std::optional<TrackState> currentState() const;
  [[nodiscard]] std::vector<ArmorDetection> predictedArmors() const;
  [[nodiscard]] std::size_t trackCount() const noexcept { return filters_.size(); }

private:
  struct PredictedSlot {
    int track_index = -1;
    int slot_index = -1;
    ArmorDetection armor;
  };

  static constexpr double kOutpostRadius = 553.0 / 2.0;

  ArmorDetection calcArmor(double xc, double yc, double z, double r, double yaw, int class_id) const;
  double matchingCost(const ArmorDetection &detection, const PredictedSlot &slot) const;
  void createTrack(const ArmorDetection &detection);
  void removeTrack(std::size_t index);
  void refineMeasurement(std::size_t index);
  void selectPrimaryTrack();
  bool isOutpost(int class_id) const noexcept { return class_id == 5 || class_id == 12; }

  void setupModels();

  EkfParams params_{};
  double dt_{0.02};
  bool time_initialized_{false};
  rclcpp::Time last_stamp_{};
  int primary_index_{-1};

  std::vector<ExtendedKalmanFilter> filters_;
  std::vector<Eigen::VectorXd> measurements_;
  std::vector<int> lost_frames_;
  std::vector<int> class_ids_;
  std::array<bool, 16> active_classes_{};
  std::vector<PredictedSlot> predicted_slots_;

  std::array<double, 4> r_xy_correction_buffer_{};
  double yaw_noise_correction_{0.0};

  ExtendedKalmanFilter::VecVecFunc f_regular_;
  ExtendedKalmanFilter::VecVecFunc h_regular_;
  ExtendedKalmanFilter::VecMatFunc jf_regular_;
  ExtendedKalmanFilter::VecMatFunc jh_regular_;
  ExtendedKalmanFilter::VoidMatFunc q_regular_;
  ExtendedKalmanFilter::VecMatFunc r_regular_;
  ExtendedKalmanFilter::VecVecFunc normalize_residual_;

  ExtendedKalmanFilter::VecVecFunc f_outpost_;
  ExtendedKalmanFilter::VecVecFunc h_outpost_;
  ExtendedKalmanFilter::VecMatFunc jf_outpost_;
  ExtendedKalmanFilter::VecMatFunc jh_outpost_;
  ExtendedKalmanFilter::VoidMatFunc q_outpost_;
  ExtendedKalmanFilter::VecMatFunc r_outpost_;
};

}  // namespace aim_auto

#endif  // AIM_AUTO__EKF_TRACKER_HPP_

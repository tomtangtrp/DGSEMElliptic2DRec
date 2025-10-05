#pragma once
#include <memory>
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace DG {

class JsonCase {
public:
  explicit JsonCase(const std::string& path);
  ~JsonCase();
  JsonCase(JsonCase&&) noexcept;
  JsonCase& operator=(JsonCase&&) noexcept;
  JsonCase(const JsonCase&) = delete;
  JsonCase& operator=(const JsonCase&) = delete;

  double alpha() const;

  // Volume fields on nodal grids (returns same shape as X/Y)
  Eigen::ArrayXXd f(const Eigen::ArrayXXd& X, const Eigen::ArrayXXd& Y) const;
  Eigen::ArrayXXd u(const Eigen::ArrayXXd& X, const Eigen::ArrayXXd& Y) const;
  bool has_exact() const;

  // Boundary data (vectors on 1D GLL nodes)
  Eigen::VectorXd g_left (double xLa, const Eigen::VectorXd& y) const;
  Eigen::VectorXd g_right(double xLb, const Eigen::VectorXd& y) const;
  Eigen::VectorXd g_bottom(const Eigen::VectorXd& x, double yLa) const;
  Eigen::VectorXd g_top   (const Eigen::VectorXd& x, double yLb) const;

  Eigen::VectorXd gN_left (double xLa, const Eigen::VectorXd& y) const;
  Eigen::VectorXd gN_right(double xLb, const Eigen::VectorXd& y) const;
  Eigen::VectorXd gN_bottom(const Eigen::VectorXd& x, double yLa) const;
  Eigen::VectorXd gN_top   (const Eigen::VectorXd& x, double yLb) const;

private:
  struct Impl;                    // hidden heavy stuff
  std::unique_ptr<Impl> p_;
};

} // namespace DG

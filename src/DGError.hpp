#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <limits>

namespace DG {

// Error metrics for DG on a rectangular, affine mesh (constant hx, hy per element).
// Assumptions:
//  - Per-element nodal layout is (k+1) x (k+1), rows=y, cols=x, row-major (x fastest).
//  - Global vectors u_h, u_exact are concatenations of element tiles in element-major order:
//      e = ey*Nelx + ex  (x fastest across elements)
//  - Quadrature: GLL nodes/weights; M1_ref is the (diagonal) 1D mass on reference [-1,1],
//    D1_ref is the 1D diff on reference; physical derivatives use 2/h scaling.
class DGError {
public:
  DGError(int k, const Eigen::MatrixXd& M1_ref, const Eigen::MatrixXd& D1_ref)
  : k_(k), k1_(k+1), W1_(M1_ref.diagonal()), D1_(D1_ref)
  {
    // 2D tensor weights Wy * Wx^T
    W2D_ = W1_ * W1_.transpose(); // (k1,k1)
    reset();
  }

  void reset() {
    sumL2_ = sumL2_exact_ = 0.0;
    sumH1_ = sumH1_exact_ = 0.0;
  }

  // Accumulate from one element tile (k1 x k1 each). hx,hy are THIS element's sizes.
  // Ue: numerical; Ux: exact on the same nodal grid.
  void add_element(const Eigen::Ref<const Eigen::MatrixXd>& Ue,
                   const Eigen::Ref<const Eigen::MatrixXd>& Ux,
                   double hx, double hy)
  {
    const double J = 0.25 * hx * hy;     // map jacobian [-1,1]^2 -> physical
    // L2 part
    Eigen::ArrayXXd Diff = (Ux - Ue).array();
    const double l2_e = (Diff.square() * W2D_.array()).sum() * J;
    const double l2_x = (Ux.array().square() * W2D_.array()).sum() * J;
    sumL2_ += l2_e; sumL2_exact_ += l2_x;

    // H1 seminorm: grad on reference with D1_, then scale to physical
    // U_eta = D * U, U_xi = U * D^T
    Eigen::MatrixXd Uy_num_ref = D1_ * Ue;
    Eigen::MatrixXd Ux_num_ref = Ue  * D1_.transpose();
    Eigen::MatrixXd Uy_ex_ref  = D1_ * Ux;
    Eigen::MatrixXd Ux_ex_ref  = Ux  * D1_.transpose();

    const double sx = 2.0 / hx, sy = 2.0 / hy; // ref -> physical
    Eigen::ArrayXXd dx_err = (sx*(Ux_ex_ref - Ux_num_ref)).array();
    Eigen::ArrayXXd dy_err = (sy*(Uy_ex_ref - Uy_num_ref)).array();

    const double h1_e = ((dx_err.square() + dy_err.square()) * W2D_.array()).sum() * J;

    // accumulate exact |grad u|^2 for relative H1
    Eigen::ArrayXXd dx_ex = (sx*Ux_ex_ref).array();
    Eigen::ArrayXXd dy_ex = (sy*Uy_ex_ref).array();
    const double h1_x = ((dx_ex.square() + dy_ex.square()) * W2D_.array()).sum() * J;

    sumH1_ += h1_e; sumH1_exact_ += h1_x;
  }

  // Accumulate for ALL elements from flat global vectors (row-major tiles).
  // u_h, u_exact: length Nel*(k+1)^2. Nelx,Nely element grid; hx,hy element sizes.
  void add_from_flat(const Eigen::VectorXd& u_h,
                     const Eigen::VectorXd& u_exact,
                     int Nelx, int Nely, double hx, double hy)
  {
    const int Nel = Nelx * Nely;
    const int locdim = k1_ * k1_;
    for (int e = 0; e < Nel; ++e) {
      Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>
          Uh(u_h.data()     + e*locdim, k1_, k1_),
          Ux(u_exact.data() + e*locdim, k1_, k1_);
      add_element(Uh, Ux, hx, hy);
    }
  }

  // Absolute norms
  double broken_L2() const { return std::sqrt(sumL2_); }
  double broken_H1() const { return std::sqrt(sumH1_); }

  // Relative norms (NaN if exact is zero)
  double broken_L2_rel() const {
    return (sumL2_exact_>0) ? std::sqrt(sumL2_/sumL2_exact_) : std::numeric_limits<double>::quiet_NaN();
  }
  double broken_H1_rel() const {
    return (sumH1_exact_>0) ? std::sqrt(sumH1_/sumH1_exact_) : std::numeric_limits<double>::quiet_NaN();
  }

  // Vector L2 (Euclidean) on the flat stacked vectors
  static double vector_L2(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    return (a-b).norm();
  }
  static double vector_L2_rel(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    const double denom = b.norm();
    return denom>0 ? (a-b).norm() / denom : std::numeric_limits<double>::quiet_NaN();
  }

private:
  int k_, k1_;
  Eigen::VectorXd W1_;     // (k+1) 1D weights on reference (assumed diagonal mass)
  Eigen::MatrixXd D1_;     // (k+1)x(k+1) 1D diff on reference
  Eigen::MatrixXd W2D_;    // (k+1)x(k+1) tensor weights Wy*Wx^T

  double sumL2_ = 0.0, sumL2_exact_ = 0.0;
  double sumH1_ = 0.0, sumH1_exact_ = 0.0;
};

} // namespace DG

#ifndef MESHGRID2DHEADERDEF
#define MESHGRID2DHEADERDEF

#include <Eigen/Dense> // include path: /usr/include/eigen3/Eigen/Dense
#include <unsupported/Eigen/CXX11/Tensor> // include path: /usr/include/eigen3/unsupported


template<typename Scalar = double>
class MeshGrid2D {
public:
  using Array1 = Eigen::Array<Scalar, Eigen::Dynamic, 1>;
  using Array2 = Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  // Construct with x (size Nx) and y (size Ny)
  MeshGrid2D(const Array1& x, const Array1& y)
    : x_(x), y_(y), computed_(false)
  {}

  // Accessors compute grid on first call
  const Array2& X() { compute(); return XX_; }
  const Array2& Y() { compute(); return YY_; }

private:
  // x_ to clearly indicate its a private (data-member) variable
  Array1 x_, y_;
  Array2 XX_, YY_;
  bool computed_;

  void compute() {
    if (computed_) return;
    // create row and column replicates
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> mx = x_.matrix().transpose();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> my = y_.matrix();
    // row major grid
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> mxx = mx.replicate(y_.size(), 1);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> myy = my.replicate(1, x_.size());
    XX_ = mxx.array();
    YY_ = myy.array();
    computed_ = true;
  }
};
#endif
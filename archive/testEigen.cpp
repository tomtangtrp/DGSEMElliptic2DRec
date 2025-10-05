#include <cmath>
#include <iostream>
#include <Eigen/Dense> // when installing the package, it creates a symlink /usr/include/Eigen -> /usr/include/eigen3/Eigen
#include <Eigen/Sparse>
#include "gllQuadrature.hpp"
#include <unsupported/Eigen/KroneckerProduct> 
// no symlink created for /usr/include/eigen3/unsupported at installation, so we need to 
// add -I/usr/include/eigen3 to g++
// in order to #include <unsupported/Eigen/KroneckerProduct>, the absolute path is 
// /usr/local/include/eigen3/unsupported/Eigen/

using namespace Eigen;

int main(int argc, char* argv[])
{
    // Check Eigen sum singe row methods
    MatrixXd M(2,3);
    M << 1, 2, 3,
         4, 5, 6;

    double sum_row_i = M.row(1).sum();
    std::cout << "sum of row 1 = " << sum_row_i << std::endl;

    int k = 2;
    int locdim_1d = k+1;
    GLL mGLL(k);
    VectorXd xi_1d = mGLL.getGLLNodes();
    VectorXd w_1d = mGLL.getGLLWeights();
    VectorXd bw_1d = mGLL.getBaryWeights();
    MatrixXd W1 = mGLL.get1dMass();
    MatrixXd D1 = mGLL.get1dDiff();

    // check Eigen outer product 
    VectorXd e0(locdim_1d);
    VectorXd en(locdim_1d);
    e0 << 1, 0, 0;
    en << 0, 0, 1;
    MatrixXd proj_0(locdim_1d, locdim_1d);
    // outer product (vectors)
    proj_0 = e0*(e0.transpose());
    std::cout << "proj_0 =  \n" << proj_0 << std::endl;


    // Check Eigen kronecker product
    // 2d Mass on reference element:
    MatrixXd M2d = kroneckerProduct(W1, W1).eval();

    std::cout << "M2d = \n" << M2d << std::endl;



    return 0;
}
#include <iostream>
#include <Eigen/Dense>
#include <type_traits>
#include "MeshGrid2D.hpp"
#include "gllQuadrature.hpp"

using namespace std;
using namespace Eigen;

VectorXd MapPhysical(VectorXd& xi_1d, double& a, double& b);

// evaluates exaxt_sol on an 2D square element then projected onto row-major vector
template<typename T, typename Func> T exact_sol(const T& x, const T& y, Func f);

double man_sol1(double x, double y) { return std::exp(-x - y * y);}

Eigen::ArrayXXd man_sol1(const Eigen::ArrayXXd& X,
                          const Eigen::ArrayXXd& Y)
{
    return (-X.array() - Y.array().square()).exp();
}
Eigen::ArrayXXd man_sol2(const Eigen::ArrayXXd& X,
    const Eigen::ArrayXXd& Y)
{
    return (M_PI*X.array()).sin() * (2*M_PI*Y.array()).sin();
}

int main(int argc, char* argv[])
{   
    int k = 2;
    GLL mGLL(k);
    VectorXd xi_1d = mGLL.getGLLNodes();
    VectorXd eta_1d = xi_1d;
    double ax = 0.0;
    double bx = 0.5;
    double ay = 0.0;
    double by = 0.5;
    VectorXd x_1d = MapPhysical(xi_1d, ax, bx);
    VectorXd y_1d = MapPhysical(eta_1d, ay, by);

    MeshGrid2D<double> mesh(x_1d, y_1d);
    // grids are built when you first request them
    // auto& X = mesh.X();
    // auto& Y = mesh.Y();
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>& X = mesh.X();
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>& Y = mesh.Y();
    
    std::cout << "X = \n" << X << std::endl;
    std::cout << "Y = \n" << Y << std::endl;

    // VectorXd exact_ary = exact_sol(X,Y, man_sol1);
    VectorXd exact_ary = exact_sol(X, Y, man_sol2);

    std::cout << "exact_sol = \n" << exact_ary << std::endl;
    
    return 0;
}


VectorXd MapPhysical(VectorXd& xi_1d, double& a, double& b)
{   
    int size = xi_1d.size();
    VectorXd x(size);
    for (int i=0; i<xi_1d.size(); i++)
    {
        x[i] = (b-a)*0.5*xi_1d[i] + (b+a)*0.5;
    } 
    return x;
}

// template<typename T> T exact_sol(const T& x, const T& y) {
//     // constexp and std::is_floating_point_v<T> both require -std=c++17
//     if constexpr (std::is_floating_point_v<T>) {
//         // Scalar path
//         return std::exp(-x - y * y);
//     } else {
//         // Eigen path: element-wise operations via .array()
//         MatrixXd u_mat = (-x.array() - y.array().square()).exp();  
//         // back to VectorXd
//         // Eigen is column major, so need to transfer to row major
//         using Scalar = typename T::Scalar;
//         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> u_rowmat = u_mat.matrix();
//         // Be careful about Map< const Matrix<...> >
//         return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(u_rowmat.data(), u_rowmat.size());
//     }
// }

template<typename T, typename Func> T exact_sol(const T& x, const T& y, Func f) {
    // constexp and std::is_floating_point_v<T> both require -std=c++17
    if constexpr (std::is_floating_point_v<T>) {
        // Scalar path
        return f(x,y);
    } else {
        // Eigen path: element-wise operations via .array()
        MatrixXd u_mat = f(x.array(), y.array()) ;
        // back to VectorXd
        // Eigen is column major, so need to transfer to row major
        using Scalar = typename T::Scalar;
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> u_rowmat = u_mat.matrix();
        // Be careful about Map< const Matrix<...> >
        return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(u_rowmat.data(), u_rowmat.size());
    }
}
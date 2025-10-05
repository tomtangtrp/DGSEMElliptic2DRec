#include <cmath>
#include <Eigen/Dense>
#include "gllQuadrature.hpp"


int main(int argc, char* argv[])
{   
    // GLL order k, so (k+1) points 
    int k = 5;
    GLL mGLL(k);
    VectorXd xi_1d = mGLL.getGLLNodes();
    VectorXd w_1d = mGLL.getGLLWeights();
    VectorXd bw_1d = mGLL.getBaryWeights();
    MatrixXd W1 = mGLL.get1dMass();
    MatrixXd D1 = mGLL.get1dDiff();

    int N = k+1;
    std::cout << "xi_1d = \n" << xi_1d << "\n";

    std::cout << "w_1d = \n" << w_1d << "\n";

    std::cout << "bw_1d = \n" << bw_1d << "\n";

    std::cout << "W1 = \n" << W1 << "\n";

    std::cout << "D1 = \n" << D1 << "\n";

    return 0;
}
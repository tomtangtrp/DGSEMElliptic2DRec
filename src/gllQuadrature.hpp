#ifndef GLLHEADERDEF
#define GLLHEADERDEF

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace Eigen;

// GLLHEADERDEF enables namespace in the class def
class GLL
{
    private:
        // Legendre polynonial $$P_N(x)$$ of order N; 
        // GLL nodes are {roots of P_N'(x) } + {-1,1}
        int mOrder;
        int locdim_1d;
        VectorXd xi_1d; // xi_1d: gllnodes;
        VectorXd w_1d;  // w_1d: gll weights
        VectorXd bw_1d;    // barycentric weights for Lagrange basis with GLL nodes
        MatrixXd W1;    // 1D Mass matrix on [-1, 1]
        MatrixXd D1;    //1D differentiation matrix [-1, 1]
    public:
        // Custome constructor
        GLL(int k);
        VectorXd getGLLNodes(){return xi_1d;};
        VectorXd getGLLWeights(){return w_1d;};
        VectorXd getBaryWeights(){return bw_1d;};
        MatrixXd get1dMass();
        MatrixXd get1dDiff();
};

#endif
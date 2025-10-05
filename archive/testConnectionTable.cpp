#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include "gllQuadrature.hpp"
#include "MeshGrid2D.hpp"
#include "SquareGrid.hpp"
#include "Connection.hpp"
#include "ConnectionTable.hpp"
#include "NumUtilities.hpp"
#include "SquareElement.hpp"

using namespace Eigen;


int main(int argc, char* argv[])
{      
    int k = 2;
    int locdim_1d = k+1;
    double alpha = 1.0;
    int eps;
    double sigma0;
    double sigma0b;
    int beta0;
    std::string method = "SIPG";
    if (method == "SIPG"){
        eps = -1;
        sigma0 = 2*k*k + 1;
        sigma0b = 2*sigma0;
        beta0 = 1.0;
    }
    else if (method == "IPG"){
        eps = 0;
        sigma0 = 2*k*k + 1;
        sigma0b = 2*sigma0;
        beta0 = 1.0;
    }
    else if (method == "NIPG"){
        eps = 1;
        sigma0 = 1.0;
        sigma0b = 1.0;
        beta0 = 1.0;
    }
    else if (method == "NIPG0"){
        eps = 1;
        sigma0 = 1.0;
        sigma0b = 0.0;
        beta0 = 0.0;
        alpha = 0.0; // no mass term
    }


    GLL mGLL(k);
    VectorXd xi_1d = mGLL.getGLLNodes();
    VectorXd eta_1d = xi_1d;
    VectorXd w_1d = mGLL.getGLLWeights();
    VectorXd bw_1d = mGLL.getBaryWeights();
    MatrixXd M1_ref = mGLL.get1dMass();
    MatrixXd D1_ref = mGLL.get1dDiff();

    // projection operators
    VectorXd e0 = VectorXd::Zero(locdim_1d);
    e0[0] = 1.0;
    VectorXd en = VectorXd::Zero(locdim_1d);
    en[k] = 1.0;
    VectorXd dphi_0 = D1_ref.row(0);
    VectorXd dphi_n = D1_ref.row(k);
    MatrixXd proj_0 = e0 * e0.transpose();
    MatrixXd proj_n = en * en.transpose();


    int locdim = locdim_1d*locdim_1d;
    int Nel_x = 2;
    int Nel_y = 2;
    SquareGrid mSquareGrid(Nel_x, Nel_y);
    ConnectionTable mSqaureConnectionTable = mSquareGrid.getSquareConnectionTable();
    mSqaureConnectionTable.printConnectionTable();

    Connection connect = mSqaureConnectionTable[0];
    int N_E1 = std::get<0>(connect.ElmConnect); 
    int N_E2 = std::get<1>(connect.ElmConnect);
    std::cout << "connect: Element" << N_E1 << "-Element"<< N_E2 << std::endl;

    return 0;
}


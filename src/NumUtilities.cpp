#include <iostream>
#include <Eigen/Dense>
#include "NumUtilities.hpp"

using namespace Eigen;

VectorXd NumUtilities::MapPhysical(VectorXd& xi_1d, double& a, double& b)
{
    int size = xi_1d.size();
    VectorXd x(size);
    for (int i=0; i<xi_1d.size(); i++)
    {
        x[i] = (b-a)*0.5*xi_1d[i] + (b+a)*0.5;
    } 
    return x;
}
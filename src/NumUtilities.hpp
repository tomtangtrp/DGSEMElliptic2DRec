#ifndef NUMUTILITIESHEADERDEF
#define NUMUTILITIESHEADERDEF

#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

class NumUtilities
{
    private:

    public:
        VectorXd MapPhysical(VectorXd& xi_1d, double& a, double& b);

};
#endif
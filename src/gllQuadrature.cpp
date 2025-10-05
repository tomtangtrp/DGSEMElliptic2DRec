#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cassert>
#include "gllQuadrature.hpp"
// #include "Legendre.hpp" // also works, but i moved Legendre.hpp to ./include
#include <Legendre.hpp>

using namespace Eigen;
using namespace std;

// Custom Constructor
GLL::GLL(int k)
{
    assert(k > 0);
    mOrder = k;
    locdim_1d = mOrder+1;
    xi_1d.resize(locdim_1d);
    w_1d.resize(locdim_1d);
    bw_1d.resize(locdim_1d);
    W1.resize(locdim_1d, locdim_1d);
    D1.resize(locdim_1d, locdim_1d);

    // Compute xi_1d:
    xi_1d[0] = -1.0;
    xi_1d[xi_1d.size()-1] = 1.0;
    // Number of interior nodes;
    int M = mOrder-1; 
    // N+1-2 = (mOrder-1) Interior nodes are roots of  dP_{mOrder}:=Q_{mOrder-1};
    // The roots of legendre dP_{mOrder} are found by Newton'w method with Cheby-Gauss initial guess;
    Legendre Legendre_n(mOrder);
    for (int i=1; i<=M; ++i) // pre-increment affecting for loops (but i initialised as 2 if a = ++i, but not happening in for loops);
    {
        // initial guess using chebyshev nodes of second kind, (mOder-i) so that the nodes are monotonically increasing
		double xi = std::cos(M_PI*(mOrder-i)/(mOrder));
        // double xi = std::cos((i+0.25)*M_PI/mOrder-3/(8*mOrder*M_PI)*(1/(i+0.25)));
        for (int iter=0; iter < 50; ++iter)
        {   
            std::pair <double, double> result = Legendre_n.getLegendreDeriv(xi);
            // unpack Legendre polynomial of order n's value P_n and dP_n at xi
            double P_n = result.first;
            double dP_n = result.second;
            // or: 
            // auto [P_n, dP_n] = L_n.getLegendreDeriv(xi);

            // root finding for Q_n = dP_n
            // x_tol = eps, sqrt of 1e-16;
            double eps = 1e-8;
            // Forward difference for Q_n' = (Q_n(x+\eps)-Q_n(x)) / eps
            double dP_n_eps =  Legendre_n.getLegendreDeriv(xi+eps).second;
            double ddP_n = (dP_n_eps - dP_n)/eps;
            // Check Division by zero:
            if (std::abs(ddP_n)<1e-15){break;}
            double dx = dP_n/ddP_n;
            xi -= dx;
            // Check step dx is not close to machine epsilon to avoid infinitely loops;
            if (std::abs(dx)< 1e-15){break;}
        }
        xi_1d[i] = xi;
    }
   // Compute gll w_1d;
   for (int i=0; i<=mOrder; ++i)
   {
        double P_n_xi = Legendre_n.getLegendreDeriv(xi_1d[i]).first;
        w_1d[i] = 2.0/(mOrder*locdim_1d*P_n_xi*P_n_xi);
   }

   // Compute barycentric weights bw_1d for lagrange basis with GLL nodes
   for (int i=0; i<locdim_1d; i++)
   {
		double prod = 1.0;
    	for (int j=0; j<locdim_1d; j++)
    	{   
        	if (j!=i){
            	prod *= xi_1d[i]-xi_1d[j];
        	}
    	}	
		bw_1d[i] = 1.0/prod;
   }
   }

// MatrixXd GLL::get1dMass()
// {   
//     for (int i=0; i<locdim_1d; i++)
//     {
//         W1(i,i) = w_1d[i];
//     }
//     return W1;
// }

MatrixXd GLL::get1dMass()
{   
    return w_1d.asDiagonal();
}

MatrixXd GLL::get1dDiff()
{   
    D1.setZero(); // Safe-guard only, not necessary since every entry is assigned below;
    // Barycentric formulation of 1D diff matrix of lagrange basis of lobatto nodes
    for (int i=0; i<locdim_1d; i++)
    {
        for (int j=0; j<locdim_1d; j++)
        {
            if (i!=j){
                D1(i,j) = bw_1d[j]/(bw_1d[i]*(xi_1d[i]-xi_1d[j]));
            }
        }
        D1(i,i) = -D1.row(i).sum();
    }
    return D1;
}




#ifndef LEGENDREHEADERDEF
#define LEGENDREHEADERDEF

#include <cmath>

// since this is headeronly:
// (i)  we can mv Legendre.hpp to /usr/include -> use #include <Legendre.hpp>
// (ii) mv Legendre.hpp to project include (current working directory for the project) ./include -> use #include <Legendre.hpp>
//      in the meantime, we need to add g++ I./include to let compiler know this package
// (iii) put Legendre.hpp in the same dir as a main.cpp, then use #include "Legendre.hpp"

class Legendre
{
    private:
        int mOrder;
    public:
        // Custom constructor
        Legendre(int k){mOrder = k;};
        std::pair<double,double> getLegendreDeriv(double x)
        {
            // P_0 = 1, P_1 = x
            double P_nm2 = 1.0; //P_{n-2}
            double P_nm1 = x; //P_{n-1}
            double P_n = 0.0; 
            if (mOrder == 0){P_n = 1.0;}
            else if (mOrder == 1){P_n = x;}
            else if (mOrder > 1){
            // Pre-increment ++n, increases n first
            for (int n = 2; n <= mOrder; ++n) {
                // Recursion formula:
                P_n = ((2.0*n - 1.0)*x*P_nm1 - (n - 1.0)*P_nm2)/n; // not symbolic, it is evaluated explicitly with input: double x
                P_nm2 = P_nm1;
                P_nm1 = P_n;
            }
        }
            // Derivative from Bonnet’s relation:
            // P_n'(x) = n/(1-x^2) [P_{n-1}(x) - x P_n(x)]
            double P_nm1_new = (mOrder>=1 ? P_nm2 : 1.0);
            double dP_n = (mOrder * (P_nm1_new - x*P_n)) / (1.0 - x*x);
            return {P_n, dP_n};
        };
};
#endif
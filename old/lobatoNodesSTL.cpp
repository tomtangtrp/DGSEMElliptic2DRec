#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <functional>

// Evaluate P_n(x) and P_n'(x) via recurrence
// returns {P_n(x), P_n'(x)}
std::pair<double,double> legendre_and_derivative(int N, double x) {
    // P_0 = 1, P_1 = x
    double Pnm2 = 1.0;
    double Pnm1 = x;
    double Pn = 0.0;
    // Pre-increment ++n, increases n first so that n initiased with 3
    for (int n = 2; n <= N; ++n) {
        // Recursion formula:
        Pn = ((2.0*n - 1.0)*x*Pnm1 - (n - 1.0)*Pnm2)/n; // not symbolic, it is evaluated explicitly with input: double x
        Pnm2 = Pnm1;
        Pnm1 = Pn;
    }
    if (N == 0)     Pn = 1.0;
    else if (N == 1) Pn = x;
    // Derivative from Bonnet’s relation:
    // P_n'(x) = n/(1-x^2) [P_{n-1}(x) - x P_n(x)]
    double Pn_1 = (N>=1 ? Pnm2 : 1.0);
    double dPn = (N * (Pn_1 - x*Pn)) / (1.0 - x*x);
    return {Pn, dPn};
}

void lobatto_1d(int N, std::vector<double> &x, std::vector<double> &w) {
    // N+1 points
    x.resize(N+1);
    w.resize(N+1);
    x.front() = -1.0;
    x.back()  =  1.0;
    if (N == 1) {
        w[0] = w[1] = 1.0;
        return;
    }
    // find interior roots of P_N'(x)
    int M = (N - 1);          // number of interior nodes
    for (int i = 1; i <= M; ++i) {
        // initial guess: cos(pi * i / N)
        double xi = std::cos(M_PI * i / N);
        // Newton iteration
        for (int iter = 0; iter < 50; ++iter) {
            auto [Pn, dPn] = legendre_and_derivative(N, xi);
            // We want roots of dPn = 0
            double dd = dPn;
            if (std::abs(dd) < 1e-14) break;
            // derivative of dPn is second derivative of Pn:
            double eps = 1e-14;
            double dPn_plus = legendre_and_derivative(N, xi + eps).second;
            double dd2 = (dPn_plus - dPn)/eps;
            xi -= dd / dd2;
            if (std::abs(dd) < 1e-13) break;
        }
        x[i] = xi;
    }
    // Compute weights: w_i = 2/[N(N+1) P_N(x_i)^2]
    for (int i = 0; i <= N; ++i) {
        auto [Pn, dPn] = legendre_and_derivative(N, x[i]);
        w[i] = 2.0 / (N*(N+1) * Pn*Pn);
    }
}

int main(){
    int N = 4;
    std::vector<double> x1d, w1d;
    lobatto_1d(N, x1d, w1d);

    // Tensor-product in 2D
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "2D Gauss–Lobatto (N="<<N<<"):\n";
    for(int i=0; i<=N; ++i){
      for(int j=0; j<=N; ++j){
        double xi = x1d[i], yi = x1d[j];
        double w2d = w1d[i]*w1d[j];
        std::cout << "("<<std::setw(8)<<xi<<","
                  <<           std::setw(8)<<yi<<") "
                  << " w="<<w2d<<"\n";
      }
    }
    return 0;
}
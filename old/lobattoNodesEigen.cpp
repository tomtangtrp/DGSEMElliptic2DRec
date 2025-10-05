// lobatto_eigen.cpp
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace Eigen;

// Compute Pₙ(x) and Pₙ′(x) via recurrence + analytic derivative
// Returns {Pₙ, Pₙ′}
std::pair<double,double> legendre_and_derivative(int N, double x) {
    double Pnm2 = 1.0;       // P₀(x)
    double Pnm1 = x;         // P₁(x)
    double Pn   = (N==0 ? 1.0 
                : N==1 ? x 
                : 0.0);
    // Build up to Pₙ
    for(int n=2; n<=N; ++n) {
        Pn = ((2*n - 1)*x*Pnm1 - (n - 1)*Pnm2) / n;
        Pnm2 = Pnm1;
        Pnm1 = Pn;
    }
    // Pₙ₋₁ for derivative formula
    // Tenary expression: if n>=1 true then Pn_1=Pnm2, if not Pn_1=1.0
    double Pn_1 = (N>=1 ? Pnm2 : 1.0);
    // Pₙ′(x) = n/(1−x²) [P_{n−1}(x) − x Pₙ(x)]
    double dPn = (N * (Pn_1 - x*Pn)) / (1 - x*x);
    return {Pn, dPn};
}

// Fill x, w as N+1 Lobatto pts + weights on [−1,1]
void lobatto1D(int N, VectorXd &x, VectorXd &w) {
    x.resize(N+1);
    w.resize(N+1);

    x(0)   = -1.0;
    x(N)   =  1.0;
    if(N==1) {
        w(0) = w(1) = 1.0;
        return;
    }

    // interior roots of Pₙ′
    for(int i=1; i< N; ++i) {
        // initial guess
        // double xi = std::cos(M_PI * i / N);
        double xi = std::cos((i+0.25)*M_PI/N-3/(8*N*M_PI)*(1/(i+0.25)));
        // Newton on f= Pₙ′(x)
        for(int it=0; it<50; ++it) {
            auto [Pn, dPn] = legendre_and_derivative(N, xi);
            // approximate second derivative via tiny shift
            double eps = 1e-8;
            double dPn_eps  = legendre_and_derivative(N, xi+eps).second;
            double ddPn     = (dPn_eps - dPn)/eps;
            if(std::abs(ddPn) < 1e-14) break;
            double dx = dPn / ddPn;
            xi -= dx;
            if(std::abs(dx) < 1e-14) break;
        }
        x(i) = xi;
    }

    // gll weights: w[i] = 2/[N(N+1) Pₙ(x_i)²]
    for(int i=0; i<=N; ++i) {
        double Pn = legendre_and_derivative(N, x(i)).first;
        w(i) = 2.0 / (N*(N+1) * Pn*Pn);
    }
}

int main(){
    int N = 5;                         // choose order
    VectorXd x1d, w1d;
    lobatto1D(N, x1d, w1d);

    // build tensor-product on square
    MatrixXd X = x1d.replicate(1, N+1);        // each column is x1d
    MatrixXd Y = x1d.transpose().replicate(N+1, 1);  // each row is x1d
    MatrixXd W = w1d * w1d.transpose();        // outer product

    // print points + weights
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "2D Gauss–Lobatto (N="<<N<<"):\n";
    for(int i=0; i<=N; ++i){
        for(int j=0; j<=N; ++j){
            std::cout 
              << "(" << std::setw(8) << X(i,j)
              << "," <<           std::setw(8) << Y(i,j)
              << ")  w=" << W(i,j) 
              << "\n";
        }
    }
    return 0;
}
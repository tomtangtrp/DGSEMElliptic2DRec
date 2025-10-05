#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <cmath>
#include <chrono>
#include <filesystem>
#include "gllQuadrature.hpp"
#include "MeshGrid2D.hpp"
#include "RecGrid.hpp"
#include "ConnectionTable.hpp"
#include "NumUtilities.hpp"
#include "RecElement.hpp"
#include <highfive/highfive.hpp>
#include <highfive/eigen.hpp>

using namespace Eigen;

// Kronecker product for Eigen matrices
template<typename DerivedA, typename DerivedB> Eigen::Matrix<typename DerivedA::Scalar, Eigen::Dynamic, Eigen::Dynamic> 
kron(const Eigen::MatrixBase<DerivedA>& A, const Eigen::MatrixBase<DerivedB>& B);

template<typename T, typename Func> T eval_2d(const T& x, const T& y, Func f);

Eigen::ArrayXXd exact_sol1(const Eigen::ArrayXXd& X, const Eigen::ArrayXXd& Y) {
    return (-X.array() - Y.array().square()).exp();};

Eigen::ArrayXXd source1(const Eigen::ArrayXXd& X, const Eigen::ArrayXXd& Y) {
    return (2-4*(Y.array()*Y.array()))*((-X.array() - Y.array().square()).exp());};

Eigen::ArrayXXd g_bottom(const Eigen::ArrayXXd& X, const double yLa) {
    return ((-X.array() - yLa*yLa).exp());};

Eigen::ArrayXXd g_top(const Eigen::ArrayXXd& X, const double yLb) {
    return ((-X.array() - yLb*yLb).exp());}; 

Eigen::ArrayXXd g_left(const double xLa, const Eigen::ArrayXXd& Y) {
    return ((-xLa - Y.array().square()).exp());};

Eigen::ArrayXXd g_right(const double xLb, const Eigen::ArrayXXd& Y) {
    return ((-xLb - Y.array().square()).exp());};

enum class Method { SIPG, IIPG, NIPG, NIPG0};

bool parseMethod(std::string_view s, Method& out);

int main(int argc, char* argv[])
{    
    // take command line inputs
    std::string write_h5 = argv[1];
    int k, Nel_x, Nel_y;
    // Defaults
    double xLa = 0.0, xLb = 1.0, yLa = 0.0, yLb = 1.0;

    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " METHOD k Nel_x Nel_y [xLa xLb yLa yLb]\n";
        return 1;
    }
    
    Method method;
    if (!parseMethod(argv[2], method)) {
        std::cerr << "Unknown METHOD: " << argv[1]
                  << " (expected SIPG|IIPG|NIPG|NIPG0)\n";
        return 1;
    }

    k      = std::atoi(argv[3]);
    Nel_x  = std::atoi(argv[4]);
    Nel_y  = std::atoi(argv[5]);

    if (argc >= 9) {
        xLa = std::atof(argv[6]);
        xLb = std::atof(argv[7]);
        yLa = std::atof(argv[8]);
        yLb = std::atof(argv[9]);
    }

    // Basic timing:
    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();

    int locdim_1d = k+1;
    double alpha = 1.0;
    int eps;
    double sigma0;
    double sigma0b;
    double beta0;
    
    // One-time method parameterization (cheap branch on enum)
    switch (method) {
        case Method::SIPG:
            eps = -1;
            sigma0 = 2.0 * k * k + 1.0;
            sigma0b = 2.0 * sigma0;
            beta0 = 1.0;
            alpha = 1.0;
            break;
        case Method::IIPG:
            eps = 0;
            sigma0 = 2.0 * k * k + 1.0;
            sigma0b = 2.0 * sigma0;
            beta0 = 1.0;
            alpha = 1.0;
            break;
        case Method::NIPG:
            eps = 1;
            sigma0 = 1.0;
            sigma0b = 1.0;
            beta0 = 1.0;
            alpha = 1.0;
            break;
        case Method::NIPG0:
            eps = 1;
            sigma0 = 1.0;
            sigma0b = 0.0;
            beta0 = 0.0;
            alpha = 0.0; // no mass term
            break;
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
    MatrixXd proj_0n = e0*en.transpose();
    MatrixXd proj_n0 = en*e0.transpose();
    MatrixXd dproj_0 = dphi_0*e0.transpose();
    MatrixXd dproj_0T = e0*dphi_0.transpose();
    MatrixXd dproj_n = dphi_n*en.transpose();
    MatrixXd dproj_nT = en*dphi_n.transpose();
    MatrixXd dproj_0n = dphi_0*en.transpose();
    MatrixXd dproj_0nT = en*dphi_0.transpose();
    MatrixXd dproj_n0T = e0*dphi_n.transpose();
    MatrixXd dproj_n0 = dphi_n*e0.transpose();

    int locdim = locdim_1d*locdim_1d;
    RecGrid mRecGrid(Nel_x, Nel_y);
    ConnectionTable mRecConnectionTable = mRecGrid.getRecConnectionTable();

    int Nel = Nel_x*Nel_y;
    int dim = Nel*locdim;

    double h_n_x = (xLb-xLa)/double(Nel_x);
    double h_n_y = (yLb-yLa)/double(Nel_y);
    double J_1d_x = h_n_x/2.0;
    double J_1d_y = h_n_y/2.0;
    double dfac_x = 2.0/h_n_x;
    double dfac_y = 2.0/h_n_y;

    // 1D operator: Mass and differentiation 
    MatrixXd M_x = J_1d_x * M1_ref;
    MatrixXd M_y = J_1d_y * M1_ref; 
    MatrixXd D_x = dfac_x * D1_ref;
    MatrixXd D_y = dfac_y * D1_ref;
    MatrixXd I = Eigen::MatrixXd::Identity(locdim_1d, locdim_1d);
    // 2D operator: Mass and differentiation matrix on reference element
    MatrixXd M = kron(M_y, M_x);
    MatrixXd D2_x = kron(I, D_x);
    MatrixXd D2_y = kron(D_y, I);
    MatrixXd Q_xx = D2_x.transpose() * M * D2_x;
    MatrixXd Q_yy = D2_y.transpose() * M * D2_y;
    MatrixXd Q =  Q_xx + Q_yy;
    VectorXd u_exact(dim);

    // === Sparse global matrix and assembly ===
    using Trip = Eigen::Triplet<double>;
    std::vector<Trip> triplets;
    triplets.reserve(std::size_t( (size_t)Nel * (size_t)locdim * (size_t)locdim * 6 / 4 )); // rough guess

    auto addDenseBlock = [&](int r0, int c0, const Eigen::Ref<const MatrixXd>& B){
        const int Br = (int)B.rows();
        const int Bc = (int)B.cols();
        for (int r = 0; r < Br; ++r) {
            for (int c = 0; c < Bc; ++c) {
                double v = B(r,c);
                if (v != 0.0) triplets.emplace_back(r0 + r, c0 + c, v);
            }
        }
    };

    VectorXd b = VectorXd::Zero(dim);

    MatrixXd C11_BT = - 0.5*J_1d_x*dfac_y*kron(dproj_nT, M1_ref)
                    + 0.5*double(eps)*J_1d_x*dfac_y*kron(dproj_n, M1_ref)
                    + J_1d_x*(sigma0/(std::pow(h_n_y, beta0)))*kron(proj_n, M1_ref);
    MatrixXd C22_BT = 0.5*J_1d_x*dfac_y*kron(dproj_0T, M1_ref)
                    - 0.5*double(eps)*J_1d_x*dfac_y*kron(dproj_0, M1_ref)
                    + J_1d_x*(sigma0/(std::pow(h_n_y, beta0)))*kron(proj_0, M1_ref);
    MatrixXd C12_BT = - 0.5*J_1d_x*dfac_y*kron(dproj_0nT, M1_ref)
                    - 0.5*double(eps)*J_1d_x*dfac_y*kron(dproj_n0, M1_ref)
                    - J_1d_x*(sigma0/(std::pow(h_n_y, beta0)))*kron(proj_n0, M1_ref);
    MatrixXd C21_BT = + 0.5*J_1d_x*dfac_y*kron(dproj_n0T, M1_ref)
                    + 0.5*double(eps)*J_1d_x*dfac_y*kron(dproj_0n, M1_ref)
                    - J_1d_x*(sigma0/(std::pow(h_n_y, beta0)))*kron(proj_0n, M1_ref);

    MatrixXd C11_LR = - 0.5*J_1d_y*dfac_x*kron(M1_ref, dproj_nT)
                    + 0.5*double(eps)*J_1d_y*dfac_x*kron(M1_ref, dproj_n)
                    + J_1d_y*(sigma0/(std::pow(h_n_x, beta0)))*kron(M1_ref, proj_n);
    MatrixXd C22_LR = 0.5*J_1d_y*dfac_x*kron(M1_ref, dproj_0T)
                    - 0.5*double(eps)*J_1d_y*dfac_x*kron(M1_ref, dproj_0)
                    + J_1d_y*(sigma0/(std::pow(h_n_x, beta0)))*kron(M1_ref, proj_0);
    MatrixXd C12_LR = - 0.5*J_1d_y*dfac_x*kron(M1_ref, dproj_0nT)
                    - 0.5*double(eps)*J_1d_y*dfac_x*kron(M1_ref,dproj_n0)
                    - J_1d_y*(sigma0/(std::pow(h_n_x, beta0)))*kron(M1_ref, proj_n0);
    MatrixXd C21_LR = + 0.5*J_1d_y*dfac_x*kron(M1_ref, dproj_n0T)
                    + 0.5*double(eps)*J_1d_y*dfac_x*kron(M1_ref,dproj_0n)
                    - J_1d_y*(sigma0/(std::pow(h_n_x, beta0)))*kron(M1_ref, proj_0n);

    int nsign1 = 1;
    int nsign2 = -1;

    MatrixXd BB = - double(nsign2)*J_1d_x*dfac_y*kron(dproj_0T, M1_ref)
                  + double(nsign2)*(double(eps))*J_1d_x*dfac_y*kron(dproj_0, M1_ref)
                  + J_1d_x*(sigma0b/(std::pow(h_n_y, beta0)))*kron(proj_0, M1_ref);

    MatrixXd BR = - double(nsign1)*J_1d_y*dfac_x*kron(M1_ref,dproj_nT)
                  + double(nsign1)*(double(eps))*J_1d_y*dfac_x*kron(M1_ref, dproj_n)
                  + J_1d_y*(sigma0b/(std::pow(h_n_x, beta0)))*kron(M1_ref, proj_n);

    MatrixXd BT = - double(nsign1)*J_1d_x*dfac_y*kron(dproj_nT, M1_ref)
                  + double(nsign1)*(double(eps))*J_1d_x*dfac_y*kron(dproj_n, M1_ref)
                  + J_1d_x*(sigma0b/(std::pow(h_n_y, beta0)))*kron(proj_n, M1_ref);
    
    MatrixXd BL = - double(nsign2)*J_1d_y*dfac_x*kron(M1_ref, dproj_0T)
                  + double(nsign2)*(double(eps))*J_1d_y*dfac_x*kron(M1_ref, dproj_0)
                  + J_1d_y*(sigma0b/(std::pow(h_n_x, beta0)))*kron(M1_ref, proj_0);

    // Volume terms
    for (int i=0; i<Nel; i++)
    {
        double a_x = (double (i%Nel_x))*h_n_x;
        double b_x = a_x + h_n_x;
        double a_y = (double (i/Nel_x))*h_n_y;
        double b_y = a_y + h_n_y;
        VectorXd x_1d = (b_x-a_x)/2.0*xi_1d.array() + (b_x+a_x)/2.0;
        VectorXd y_1d = (b_y-a_y)/2.0*eta_1d.array() + (b_y+a_y)/2.0;

        MeshGrid2D<double> mesh(x_1d, y_1d);
        const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>& X = mesh.X();
        const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>& Y = mesh.Y();

        // Assemble A with volume terms.
        addDenseBlock(i*locdim, i*locdim, Q + alpha*M);

        // Source contribution to b
        VectorXd bf_local = M*(eval_2d(X, Y, source1).matrix().transpose().reshaped());
        b.segment(i*locdim, locdim) = bf_local;

        // Compute u_exact
        VectorXd u_exact_local=eval_2d(X, Y, exact_sol1);
        u_exact.segment(i*locdim, locdim) = u_exact_local;

        RecElement elm_local(a_x, b_x, a_y, b_y);
        for (int ie=0; ie<4; ie++)
        {
            Edge edge = elm_local.getEdges()[ie];
            edge.check_bdry(xLa, xLb, yLa, yLb);
            if (bool bool_bdry = edge.get_is_bdry()) {
                int edge_lid = edge.get_edge_lid();
                if (edge_lid ==0) { // Bottom edge
                    addDenseBlock(i*locdim, i*locdim, BB);
                    VectorXd gB = g_bottom(x_1d, yLa);
                    VectorXd wgB = M1_ref*gB;
                    MatrixXd bB_mat = double(nsign2)*double(eps)*J_1d_x*dfac_y*(dphi_0*wgB.transpose()) + J_1d_x*(sigma0b/(std::pow(h_n_y, beta0)))*(e0*wgB.transpose());
                    VectorXd bB = bB_mat.transpose().reshaped();
                    b.segment(i*locdim, locdim) += bB;
                }
                if (edge_lid ==1) { // Right edge
                    addDenseBlock(i*locdim, i*locdim, BR);
                    VectorXd gR = g_right(xLb, y_1d);
                    VectorXd wgR = M1_ref*gR;
                    MatrixXd bR_mat = double(nsign1)*double(eps)*J_1d_y*dfac_x*(wgR*dphi_n.transpose()) + J_1d_y*(sigma0b/(std::pow(h_n_x, beta0)))*(wgR*en.transpose());
                    VectorXd bR = bR_mat.transpose().reshaped();
                    b.segment(i*locdim, locdim) += bR;
                }
                if (edge_lid ==2) { // Top edge
                    addDenseBlock(i*locdim, i*locdim, BT);
                    VectorXd gT = g_top(x_1d, yLb);
                    VectorXd wgT = M1_ref*gT;
                    MatrixXd bT_mat = double(nsign1)*double(eps)*J_1d_x*dfac_y*(dphi_n*wgT.transpose()) + J_1d_x*(sigma0b/(std::pow(h_n_y, beta0)))*(en*wgT.transpose());
                    VectorXd bT = bT_mat.transpose().reshaped();
                    b.segment(i*locdim, locdim) += bT;
                }
                if (edge_lid ==3) { // Left edge
                    addDenseBlock(i*locdim, i*locdim, BL);
                    VectorXd gL = g_left(xLa, y_1d);
                    VectorXd wgL = M1_ref*gL;
                    MatrixXd bL_mat = double(nsign2)*double(eps)*J_1d_y*dfac_x*(wgL*dphi_0.transpose()) + J_1d_y*(sigma0b/(std::pow(h_n_x, beta0)))*(wgL*e0.transpose());
                    VectorXd bL = bL_mat.transpose().reshaped();
                    b.segment(i*locdim, locdim) += bL;
                }   
            }
        }
    }

    // Contributions of A from Numerical flux from connection table
    for (int i=0; i<mRecConnectionTable.getSize(); i++)
    {
        Connection connect = mRecConnectionTable[i];
        int N_E1 = std::get<0>(connect.ElmConnect); 
        int N_E2 = std::get<1>(connect.ElmConnect);

        int edge_minus_id = std::get<0>(connect.EdgeConnect);
        if (edge_minus_id == 2){ // Bottom-Top connect
            addDenseBlock(N_E1*locdim, N_E1*locdim, C11_BT);
            addDenseBlock(N_E2*locdim, N_E2*locdim, C22_BT);
            addDenseBlock(N_E1*locdim, N_E2*locdim, C12_BT);
            addDenseBlock(N_E2*locdim, N_E1*locdim, C21_BT);      
        }
        else if (edge_minus_id == 1){ // Left-Right connect
            addDenseBlock(N_E1*locdim, N_E1*locdim, C11_LR);
            addDenseBlock(N_E2*locdim, N_E2*locdim, C22_LR);
            addDenseBlock(N_E1*locdim, N_E2*locdim, C12_LR);
            addDenseBlock(N_E2*locdim, N_E1*locdim, C21_LR);      
        }
    }

    // Build sparse matrix (sum duplicate triplets)
    using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>; // RowMajor to enable OpenMP in SpMV
    SpMat A(dim, dim);
#if EIGEN_VERSION_AT_LEAST(3,3,0)
    A.setFromTriplets(triplets.begin(), triplets.end(), std::plus<double>());
#else
    A.setFromTriplets(triplets.begin(), triplets.end());
#endif
    A.makeCompressed();

    // init numerical solution and solve with iterative sparse solvers
    VectorXd u_h(dim);
    if (method == Method::SIPG){
        // SPD -> Conjugate Gradient (multithreaded with OpenMP when compiled, esp. with Lower|Upper)
        Eigen::ConjugateGradient<SpMat, Eigen::Lower|Eigen::Upper, Eigen::DiagonalPreconditioner<double>> cg;
        cg.setMaxIterations(2000);
        // cg.setTolerance(1e-10); // Higher precision usually reqreuis good pre-conditioning 
        cg.setTolerance(1e-13); // Higher precision usually reqreuis good pre-conditioning 
        cg.compute(A);
        u_h = cg.solve(b);
        std::cout << "CG iters: " << cg.iterations() << ", error: " << cg.error() << std::endl;
    }
    else{
        // Non-symmetric variants -> BiCGSTAB (also can utilize OpenMP with row-major SpMV)
        Eigen::BiCGSTAB<SpMat, Eigen::IncompleteLUT<double>> solver;
        solver.setMaxIterations(4000);
        solver.setTolerance(1e-13); // Higher precision usually reqreuis good pre-conditioning 
        solver.compute(A);
        u_h = solver.solve(b);
        std::cout << "BiCGSTAB iters: " << solver.iterations() << ", error: " << solver.error() << std::endl;
    }

    std::string method_str = argv[2];
    std::cout << "Rectangle grid [ " << xLa << ", " << xLb << " ] x [ " << yLa << ", " << yLb << " ]" << " with Nel_x=" << Nel_x << ", Nel_y=" << Nel_y << ", and total Nel=" << Nel << " elements:" << "\n";
    std::cout << "DG method=" << method_str << ", order k=" << k << ", sigma0=" << sigma0 << "\n";
    std::cout << "Solver method=" <<  "Sparse matrix Iterative" << ", package=" << "Eigen sparse CG and BICG" << "\n";

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    double seconds = std::chrono::duration<double>(duration).count();
    std::cout << "DG Solver Wall time: " << seconds << " seconds" << std::endl;
	

    // Check the preliminary vector l2 error      
    VectorXd error = u_exact - u_h;
    double l2_error = std::sqrt(error.dot(error))/std::sqrt(u_exact.dot(u_exact));
    std::cout << "relative vector l2 error = " << l2_error << std::endl;

    // Write the solution to a HDF5 file
    if (write_h5 == "w"){
        std::string method_str = argv[2];
        namespace fs = std::filesystem;

        std::string output_dir  = "data";  // controls path
        fs::create_directories(output_dir);

        std::ostringstream fname;
        fname << "DG2DSquare_method=" << method_str
            << "_Nel=" << Nel
            << "_k=" << k
            << "_sigma0=" << sigma0
            << ".h5";

        fs::path filepath = fs::path(output_dir) /fname.str();
        std::cout << "Writing solution to file: " << filepath << std::endl;

        HighFive::File file(filepath.string(), HighFive::File::Overwrite);

        auto u_h_group = file.createGroup("NumericalSolution");
        u_h_group.createAttribute("method", method_str);
        u_h_group.createAttribute("Nel", Nel);
        u_h_group.createAttribute("k", k);
        u_h_group.createAttribute("sigma0", sigma0);
        u_h_group.createDataSet("u_h",  u_h);
        u_h_group.createDataSet("l2_error",  l2_error);
    }

    return 0;
}


template<typename DerivedA, typename DerivedB> Eigen::Matrix<typename DerivedA::Scalar, Eigen::Dynamic, Eigen::Dynamic> 
kron(const Eigen::MatrixBase<DerivedA>& A, const Eigen::MatrixBase<DerivedB>& B)
{
    using Scalar = typename DerivedA::Scalar;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> K(
        A.rows() * B.rows(), A.cols() * B.cols());

    for (Eigen::Index i = 0; i < A.rows(); ++i)
        for (Eigen::Index j = 0; j < A.cols(); ++j)
            K.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;

    return K;
}


template<typename T, typename Func> T eval_2d(const T& x, const T& y, Func f) {
    if constexpr (std::is_floating_point_v<T>) {
        // Scalar path
        return f(x,y);
    } else {
        // Eigen path: element-wise operations via .array()
        MatrixXd u_mat = f(x.array(), y.array()) ;
        using Scalar = typename T::Scalar;
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> u_rowmat = u_mat.matrix();
        return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(u_rowmat.data(), u_rowmat.size());
    }
}

bool parseMethod(std::string_view s, Method& out) {
    if (s == "SIPG")  { out = Method::SIPG;  return true; }
    if (s == "IIPG")  { out = Method::IIPG;  return true; }
    if (s == "NIPG")  { out = Method::NIPG;  return true; }
    if (s == "NIPG0") { out = Method::NIPG0; return true; }
    return false;
}


#include <iostream>
#include <Eigen/PardisoSupport>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SparseCholesky>
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
#include "JsonCase.hpp"
#include "DGError.hpp"

using namespace Eigen;

enum class BC { DC, NC }; // Dirichlet, Neumann

bool parseBCToken(std::string tok, BC& out) {
    // tok is DC or NC (!!!case-insensitive!!!), or long forms "dirichlet"/"neumann" or "Dirichlet"/"Neumann"
    for (auto& ch : tok) ch = std::tolower(ch); 
    if (tok=="dc" || tok=="dirichlet" || tok=="d") { out = BC::DC; return true; }
    if (tok=="nc" || tok=="neumann" || tok=="n")   { out = BC::NC;   return true; }
    return false;
}

// Parse a compact spec like: "L=D,R=N,B=D,T=N" (order/spacing flexible)
struct BCspec { BC left=BC::DC, right=BC::DC, bottom=BC::DC, top=BC::DC; };

BCspec parseBCspec(const std::string& s) {
    BCspec spec;
    std::string cur; cur.reserve(s.size());
    auto apply = [&](const std::string& kv){
        auto p = kv.find('=');
        if (p==std::string::npos) return;
        std::string k = kv.substr(0,p), v = kv.substr(p+1);
        // trim
        auto trim=[&](std::string& z){ z.erase(0, z.find_first_not_of(" \t")); z.erase(z.find_last_not_of(" \t")+1); };
        trim(k); trim(v);
        BC tmp;
        if (!parseBCToken(v, tmp)) return;
        if (k.size()) {
            char c = std::tolower(k[0]);
            if (c=='l') spec.left = tmp;
            else if (c=='r') spec.right = tmp;
            else if (c=='b') spec.bottom = tmp;
            else if (c=='t') spec.top = tmp;
        }
    };
    std::string acc;
    for (char c: s) { if (c==',' || c==';') { apply(acc); acc.clear(); } else acc.push_back(c); }
    if (!acc.empty()) apply(acc);
    return spec;
}
    
template <typename SparseMat>
std::size_t sparseMemoryBytes(const SparseMat& M) {
    using Scalar = typename SparseMat::Scalar;
    using StorageIndex = typename SparseMat::StorageIndex;

    std::size_t bytes = 0;
    bytes += sizeof(Scalar)      * M.nonZeros();       // values
    bytes += sizeof(StorageIndex)* M.nonZeros();       // inner indices
    bytes += sizeof(StorageIndex)* (M.outerSize() + 1);// outer ptrs
    return bytes;
}

// Kronecker product for Eigen matrices
template<typename DerivedA, typename DerivedB> Eigen::Matrix<typename DerivedA::Scalar, Eigen::Dynamic, Eigen::Dynamic> 
kron(const Eigen::MatrixBase<DerivedA>& A, const Eigen::MatrixBase<DerivedB>& B);

template<typename Derived>
Eigen::VectorXd flatten_transpose_rowmajor(const Eigen::MatrixBase<Derived>& m) {
  #if EIGEN_VERSION_AT_LEAST(3,4,0)
      return m.transpose().reshaped();
  #else
      Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> tmp = m.transpose();
      return Eigen::Map<const Eigen::VectorXd>(tmp.data(), tmp.size());
  #endif
}

enum class Method { SIPG, IIPG, NIPG, NIPG0};

bool parseMethod(std::string_view s, Method& out);

int main(int argc, char* argv[])
{    
    if (argc != 13) {
        std::cerr << "Usage: " << argv[0]
                  << " {w|nw} METHOD k sigma0 Nel_x Nel_y xLa xLb yLa yLb BC_SPEC CASE_JSON\n";
        std::exit(1);
    }
    
    std::string write_h5  = argv[1];          // "w" or "nw"
    Method method;
    if (!parseMethod(argv[2], method)) {
        std::cerr << "Unknown METHOD: " << argv[2]
                  << " (expected SIPG|IIPG|NIPG|NIPG0)\n";
        return 1;
    }
    int         k         = std::stoi(argv[3]);
    double      sigma0 = std::stod(argv[4]);   // numeric only, by request
    int         Nel_x     = std::stoi(argv[5]);
    int         Nel_y     = std::stoi(argv[6]);
    double      xLa       = std::stod(argv[7]);
    double      xLb       = std::stod(argv[8]);
    double      yLa       = std::stod(argv[9]);
    double      yLb       = std::stod(argv[10]);
    BCspec      mBC       = parseBCspec(argv[11]);
    std::string case_path = argv[12];
    
    if (!(xLb > xLa && yLb > yLa)) {
        std::cerr << "ERROR: require xLb>xLa and yLb>yLa.\n";
        std::exit(1);
    }
    
    // Build JSON case (unchanged)
    DG::JsonCase mCASE(case_path);
    double alpha = mCASE.alpha();  // if your code uses JSON alpha


    // Basic timing:
    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();

    int locdim_1d = k+1;
    //double alpha = 1.0;
    int eps;
    double sigma0b;
    double beta0;
    
    // One-time method parameterization (cheap branch on enum)
    switch (method) {
        case Method::SIPG:
            eps = -1;
            sigma0b = 2.0 * sigma0;
            beta0 = 1.0;
            //alpha = 1.0;
            break;
        case Method::IIPG:
            eps = 0;
            sigma0b = 2.0 * sigma0;
            beta0 = 1.0;
            //alpha = 1.0;
            break;
        case Method::NIPG:
            eps = 1;
            sigma0 = 1.0;
            sigma0b = 1.0;
            beta0 = 1.0;
            //alpha = 1.0;
            break;
        case Method::NIPG0:
            eps = 1;
            sigma0 = 0.0;
            sigma0b = 0.0;
            beta0 = 0.0;
            //alpha = 1.0; 
            // no mass term then alpha = 0.0;
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

    const double h_n_x = (xLb-xLa)/double(Nel_x);
    const double h_n_y = (yLb-yLa)/double(Nel_y);
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
        // VectorXd bf_local = M*(eval_2d(X, Y, source1).matrix().transpose().reshaped());
        // VectorXd bf_local = M*(mCASE.f(X, Y).matrix().transpose().reshaped());
        VectorXd bf_local = M*flatten_transpose_rowmajor(mCASE.f(X, Y).matrix());
        b.segment(i*locdim, locdim) = bf_local;

        // Compute u_exact
        //VectorXd u_exact_local=eval_2d(X, Y, exact_sol1);
        //u_exact.segment(i*locdim, locdim) = u_exact_local;

        // VectorXd u_exact_local = mCASE.u(X, Y).matrix().transpose().reshaped();
        VectorXd u_exact_local = flatten_transpose_rowmajor(mCASE.u(X, Y).matrix());
        u_exact.segment(i*locdim, locdim) = u_exact_local;

        RecElement elm_local(a_x, b_x, a_y, b_y);
        for (int ie=0; ie<4; ie++)
        {
            Edge edge = elm_local.getEdges()[ie];
            edge.check_bdry(xLa, xLb, yLa, yLb);
            if (bool bool_bdry = edge.get_is_bdry()) {
                int edge_lid = edge.get_edge_lid();
                if (edge_lid ==0 && alpha >=0.0) { // Bottom edge
                    if (mBC.bottom == BC::DC){
                    addDenseBlock(i*locdim, i*locdim, BB);
                    // VectorXd gB = g_bottom(x_1d, yLa);
                    VectorXd gB = mCASE.g_bottom(x_1d, yLa);
                    VectorXd wgB = M1_ref*gB;
                    MatrixXd bB_mat = double(nsign2)*double(eps)*J_1d_x*dfac_y*(dphi_0*wgB.transpose()) + J_1d_x*(sigma0b/(std::pow(h_n_y, beta0)))*(e0*wgB.transpose());
                    // VectorXd bB = bB_mat.transpose().reshaped();
                    VectorXd bB = flatten_transpose_rowmajor(bB_mat);
                    b.segment(i*locdim, locdim) += bB;
                    }
                    else if (mBC.bottom == BC::NC) {
                        // Neumann BC: No contribution to global matrix,  do nothing (natural for diffusion)
                        VectorXd gNB = mCASE.gN_bottom(x_1d, yLa);
                        // VectorXd gNB = gN_bottom(x_1d, yLa);
                        VectorXd wgNB = M1_ref * gNB;
                        MatrixXd bN_mat = J_1d_x * (e0 * wgNB.transpose());
                        // b.segment(i*locdim, locdim) += bN_mat.transpose().reshaped();
                        b.segment(i*locdim, locdim) += flatten_transpose_rowmajor(bN_mat);
                    }
                }
                if (edge_lid ==1) { // Right edge
                    if (mBC.right== BC::DC){
                    addDenseBlock(i*locdim, i*locdim, BR);
                    // VectorXd gR = g_right(xLb, y_1d);
                    VectorXd gR = mCASE.g_right(xLb, y_1d);
                    VectorXd wgR = M1_ref*gR;
                    MatrixXd bR_mat = double(nsign1)*double(eps)*J_1d_y*dfac_x*(wgR*dphi_n.transpose()) + J_1d_y*(sigma0b/(std::pow(h_n_x, beta0)))*(wgR*en.transpose());
                    // VectorXd bR = bR_mat.transpose().reshaped();
                    VectorXd bR = flatten_transpose_rowmajor(bR_mat);
                    b.segment(i*locdim, locdim) += bR;
                    }
                    else if (mBC.right == BC::NC) {
                        // Neumann BC: No contribution to global matrix,  do nothing (natural for diffusion)
                        // VectorXd gNR = gN_right(xLb, y_1d);
                        VectorXd gNR = mCASE.gN_right(xLb, y_1d);
                        VectorXd wgNR = M1_ref * gNR;
                        MatrixXd bN_mat = J_1d_y * (wgNR * en.transpose());
                        // b.segment(i*locdim, locdim) += bN_mat.transpose().reshaped();
                        b.segment(i*locdim, locdim) += flatten_transpose_rowmajor(bN_mat);
                    }
                }
                if (edge_lid ==2 && alpha >=0.0) { // Top edge
                    if (mBC.top == BC::DC){
                    addDenseBlock(i*locdim, i*locdim, BT);
                    //VectorXd gT = g_top(x_1d, yLb);
                    VectorXd gT = mCASE.g_top(x_1d, yLb);
                    VectorXd wgT = M1_ref*gT;
                    MatrixXd bT_mat = double(nsign1)*double(eps)*J_1d_x*dfac_y*(dphi_n*wgT.transpose()) + J_1d_x*(sigma0b/(std::pow(h_n_y, beta0)))*(en*wgT.transpose());
                    // VectorXd bT = bT_mat.transpose().reshaped();
                    VectorXd bT = flatten_transpose_rowmajor(bT_mat);
                    b.segment(i*locdim, locdim) += bT;
                    }
                    else if (mBC.top == BC::NC) {
                        // Neumann BC: No contribution to global matrix,  do nothing (natural for diffusion)
                        // VectorXd gNT = gN_top(x_1d, yLb);
                        VectorXd gNT = mCASE.gN_top(x_1d, yLb);
                        VectorXd wgNT = M1_ref * gNT;
                        MatrixXd bN_mat = J_1d_x * (en * wgNT.transpose());
                        // b.segment(i*locdim, locdim) += bN_mat.transpose().reshaped();
                        b.segment(i*locdim, locdim) += flatten_transpose_rowmajor(bN_mat);
                    }
                }
                if (edge_lid ==3) { // Left edge
                    if (mBC.left == BC::DC){
                    addDenseBlock(i*locdim, i*locdim, BL);
                    //VectorXd gL = g_left(xLa, y_1d);
                    VectorXd gL = mCASE.g_left(xLa, y_1d);
                    VectorXd wgL = M1_ref*gL;
                    MatrixXd bL_mat = double(nsign2)*double(eps)*J_1d_y*dfac_x*(wgL*dphi_0.transpose()) + J_1d_y*(sigma0b/(std::pow(h_n_x, beta0)))*(wgL*e0.transpose());
                    // VectorXd bL = bL_mat.transpose().reshaped();
                    VectorXd bL = flatten_transpose_rowmajor(bL_mat);
                    b.segment(i*locdim, locdim) += bL;
                }
                else if (mBC.left == BC::NC) {
                    // std::cout << "NC on left\n";
                    // Neumann BC: No contribution to global matrix,  do nothing (natural for diffusion)
                    //VectorXd gNL = gN_left(xLa, y_1d);
                    VectorXd gNL = mCASE.gN_left(xLa, y_1d);
                    VectorXd wgNL = M1_ref * gNL;
                    MatrixXd bN_mat = J_1d_y * (wgNL * e0.transpose());
                    // b.segment(i*locdim, locdim) += bN_mat.transpose().reshaped();
                    b.segment(i*locdim, locdim) += flatten_transpose_rowmajor(bN_mat);
                }
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
        if (edge_minus_id == 2 && alpha >=0.0){ // Bottom-Top connect
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
    SparseMatrix<double> A(dim, dim);
#if EIGEN_VERSION_AT_LEAST(3,3,0)
    A.setFromTriplets(triplets.begin(), triplets.end(), std::plus<double>());
#else
    A.setFromTriplets(triplets.begin(), triplets.end());
#endif
    A.makeCompressed();

    // // init numerical solution and solve with sparse solvers
    // VectorXd u_h;
    // if (method == Method::SIPG){
    //     // SPD -> Cholesky
    //     SimplicialLDLT<SparseMatrix<double>> solver;
    //     solver.compute(A);
    //     if (solver.info() != Success) {
    //         std::cerr << "SimplicialLDLT factorization failed, falling back to SparseLU" << std::endl;
    //         SparseLU<SparseMatrix<double>> lu;
    //         lu.analyzePattern(A);
    //         lu.factorize(A);
    //         u_h = lu.solve(b);
    //     } else {
    //         u_h = solver.solve(b);
    //     }
    // }
    // else{
    //     // Possibly nonsymmetric -> SparseLU
    //     SparseLU<SparseMatrix<double>> solver;
    //     solver.analyzePattern(A);
    //     solver.factorize(A);
    //     u_h = solver.solve(b);
    // }

    // init numerical solution and solve with MKL PARDISO
    VectorXd u_h;

    // PardisoLDLT is for SPD; PardisoLU is for general
    // alpha<0 is Helmholtz, even SIPG is indefinite
    if (method == Method::SIPG){
        if (alpha < 0.0) {
        // symmetric but indefinite -> LDLT cholesky
        Eigen::PardisoLDLT<SparseMatrix<double>> solver;
        std::cout << "SIPG and alpha<0, Symmetric+Indefinite, trying Eigen's MKL Pardiso's LDLT first"<< "\n";
        solver.compute(A);
        if (solver.info() != Eigen::Success) {
            std::cerr << "PardisoLDLT factorization failed, falling back to PardisoLU" << std::endl;
            Eigen::PardisoLU<SparseMatrix<double>> lu;
            lu.compute(A);
            if (lu.info() != Eigen::Success) {
                throw std::runtime_error("PARDISO factorization failed");
            }
            u_h = lu.solve(b);
        } else {
            u_h = solver.solve(b); 
        }
        }
        else{ 
        // SPD -> LLT cholesky
        Eigen::PardisoLLT<SparseMatrix<double>> solver;
        std::cout << "SIPG and alpha>0, SPD, using Eigen's MKL Pardiso's LLT"<< "\n";
        solver.compute(A);
        if (solver.info() != Eigen::Success) {
            std::cerr << "PardisoLDLT factorization failed, falling back to PardisoLU" << std::endl;
            Eigen::PardisoLU<SparseMatrix<double>> lu;
            lu.compute(A);
            if (lu.info() != Eigen::Success) {
                throw std::runtime_error("PARDISO factorization failed");
            }
            u_h = lu.solve(b);
        } else {
            u_h = solver.solve(b);
        }
    }
    } else {
        std::cout << "IIPG or NIPG or NIPG0, Non-Symmetric, using Eigen's MKL Pardiso's LU"<< "\n";
        // Possibly nonsymmetric -> MKL LU
        Eigen::PardisoLU<SparseMatrix<double>> solver;
        solver.compute(A);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("PARDISO factorization failed");
        }
        u_h = solver.solve(b);
    }

// (Optional) Check solve status
if ((A * u_h).isApprox(b) == false) {
    std::cerr << "Warning: solution residual is large\n";
}
    std::string method_str = argv[2];
    std::cout << "Rectangle grid [ " << xLa << ", " << xLb << " ] x [ " << yLa << ", " << yLb << " ]" << " with Nel_x=" << Nel_x << ", Nel_y=" << Nel_y << ", and total Nel=" << Nel << " elements:" << "\n";
    std::cout << "DG method=" << method_str << ", order k=" << k << ", sigma0=" << sigma0 << "\n";
    std::cout << "Solver method=" <<  "Sparse Direct" << ", package=" << "Eigen intel MKL PardisoLLT+LDLT+LU" << "\n";

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    double seconds = std::chrono::duration<double>(duration).count();
    std::cout << "DG Solver Wall time: " << seconds << " seconds" << std::endl;
	
    /*
    // Check the preliminary vector l2 error
    VectorXd error = u_exact - u_h;
    double l2_error = std::sqrt(error.dot(error))/std::sqrt(u_exact.dot(u_exact));
    std::cout << "relative vector l2 error = " << l2_error << std::endl;
    */
    const int Nelx = Nel_x, Nely = Nel_y;
    DG::DGError err(k, M1_ref, D1_ref);
    err.reset();
    err.add_from_flat(u_h, u_exact, Nelx, Nely, h_n_x, h_n_y);
    // Vector L2 (flat)
    // const double E_vec   = DG::DGError::vector_L2(u_h, u_exact);
    const double E_vec_rel = DG::DGError::vector_L2_rel(u_h, u_exact);

    // Broken norms
    // const double E_L2   = err.broken_L2();
    const double E_L2_rel = err.broken_L2_rel();
    //const double E_H1   = err.broken_H1();
    const double E_H1_rel = err.broken_H1_rel();

    std::cout << "Errors:\n"
            << "  relative ||u-uh||_2        = " << E_vec_rel << "\n"
            << "  relative broken L2         = " << E_L2_rel << "\n"
            << "  relative broken H1 (semi)  = " << E_H1_rel << "\n";

     // Check sparse matrix memory consumption
     std::cout << "Sparse global A approx memory consumption = " << sparseMemoryBytes(A)/1024.0/1024.0 << " MB" << std::endl;

    // Write the solution to a HDF5 file
    if (write_h5 == "w"){
        namespace fs = std::filesystem;

        std::string output_dir  = "data";  // controls path
        fs::create_directories(output_dir);

        std::string case_name = fs::path(case_path).stem().string();

        std::ostringstream fname;
        fname << "DG2DRec"
              << "_"  << case_name
              << "_method=" << method_str
              << "_Nel=" << Nel
              << "_k=" << k
              << "_sigma0=" << sigma0
              << ".h5";

        fs::path filepath = fs::path(output_dir) /fname.str();
        std::cout << "Writing solution to file: " << filepath << std::endl;

        HighFive::File file(filepath.string(), HighFive::File::Overwrite);

        auto grid_group = file.createGroup("Grid");
        grid_group.createAttribute("xLa", xLa);
        grid_group.createAttribute("xLb", xLb);
        grid_group.createAttribute("yLa", yLa);
        grid_group.createAttribute("yLb", yLb);
        grid_group.createAttribute("Nel_x", Nel_x);
        grid_group.createAttribute("Nel_y", Nel_y);

        auto u_exact_group = file.createGroup("ExactSolution");
        u_exact_group.createDataSet("u_exact",  u_exact);

        auto u_h_group = file.createGroup("NumericalSolution");
        u_h_group.createAttribute("method", method_str);
        u_h_group.createAttribute("Nel", Nel);
        u_h_group.createAttribute("k", k);
        u_h_group.createAttribute("sigma0", sigma0);
        u_h_group.createDataSet("u_h",  u_h);
        u_h_group.createDataSet("l2_error", E_vec_rel);
        u_h_group.createDataSet("broken_L2_rel", E_L2_rel);
        u_h_group.createDataSet("broken_H1_rel", E_H1_rel);
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


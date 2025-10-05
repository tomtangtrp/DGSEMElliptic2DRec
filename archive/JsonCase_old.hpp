// JsonCase.hpp (header-only)
// Compile with: just add include paths for json.hpp and exprtk.hpp

#pragma once
#include <string>
#include <fstream>
#include <optional>
#include <unordered_map>
#include <stdexcept>
#include <Eigen/Dense>
#include "json.hpp"      // nlohmann::json
#include "exprtk.hpp"    // exprtk

namespace DG {

class JsonCase {
    using json = nlohmann::json;

    // Small wrapper for an exprtk expression with (x,y) variables + constants
    struct Expr2D {
        exprtk::symbol_table<double> sym;
        exprtk::expression<double>   ex;
        exprtk::parser<double>       parser;
        double x = 0.0, y = 0.0;
        bool compiled = false;  // << track compile status
    
        Expr2D() {
            sym.add_variable("x", x);
            sym.add_variable("y", y);
            ex.register_symbol_table(sym);
        }
        void bind_consts(const std::unordered_map<std::string,double>& P){
            for (const auto& kv : P) sym.add_constant(kv.first, kv.second);
        }
        void compile_or_throw(const std::string& code, const char* tag){
            if (code.empty())
                throw std::runtime_error(std::string("empty expr: ") + tag);
    
            compiled = parser.compile(code, ex);
            if (!compiled) {
                std::ostringstream oss;
                oss << "bad expr: " << tag;
                if (parser.error_count() > 0) {
                    oss << " (";
                    for (std::size_t i = 0; i < parser.error_count(); ++i) {
                        const auto err = parser.get_error(i);
                        if (i) oss << "; ";
                        oss << "#" << i << ": " << err.diagnostic
                            << " [token='" << err.token.value << "' pos=" << err.token.position << "]";
                    }
                    oss << ")";
                }
                throw std::runtime_error(oss.str());
            }
        }
        inline double eval(double xx, double yy){
            // assumes compiled == true
            x = xx; y = yy;
            return ex.value();
        }
        template<class Arr>
        Eigen::ArrayXXd eval2D(const Arr& X, const Arr& Y){
            const int m = static_cast<int>(X.rows());
            const int n = static_cast<int>(X.cols());
            Eigen::ArrayXXd out(m, n);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    out(i,j) = eval(X(i,j), Y(i,j));
            return out;
        }
        inline Eigen::VectorXd evalLineX(const Eigen::VectorXd& xv, double yfix){
            Eigen::VectorXd out(xv.size());
            for (int i = 0; i < xv.size(); ++i) out(i) = eval(xv(i), yfix);
            return out;
        }
        inline Eigen::VectorXd evalLineY(double xfix, const Eigen::VectorXd& yv){
            Eigen::VectorXd out(yv.size());
            for (int i = 0; i < yv.size(); ++i) out(i) = eval(xfix, yv(i));
            return out;
        }
        inline bool ok() const { return compiled; }  // << no internals
    };
    

    std::unordered_map<std::string,double> P_;
    double alpha_ = 1.0;

    // required
    Expr2D f_;

    // optional
    std::optional<Expr2D> u_;
    std::optional<Expr2D> dir_left_, dir_right_, dir_bottom_, dir_top_;
    std::optional<Expr2D> neu_left_, neu_right_, neu_bottom_, neu_top_;

    static std::string slurp(const std::string& path){
        std::ifstream in(path);
        if (!in) throw std::runtime_error("Cannot open JSON case file: "+path);
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    }

    static std::optional<std::string> get_str_opt(const json& j, const char* key){
        if (!j.contains(key)) return std::nullopt;
        if (j.at(key).is_null()) return std::nullopt;
        return j.at(key).get<std::string>();
    }

    static void maybe_compile(std::optional<Expr2D>& slot, const std::optional<std::string>& code,
                              const std::unordered_map<std::string,double>& P, const char* tag){
        if (!code) return;
        slot.emplace();
        slot->bind_consts(P);
        slot->compile_or_throw(*code, tag);
    }

public:
    explicit JsonCase(const std::string& path){
        auto txt = slurp(path);
        json j = json::parse(txt);

        // parameters
        if (j.contains("parameters")) {
            for (auto it = j["parameters"].begin(); it!=j["parameters"].end(); ++it) {
                P_[it.key()] = it.value().get<double>();
            }
        }
        // alpha (constant in your solver)
        if (j.contains("alpha")) alpha_ = j.at("alpha").get<double>();

        // source (required)
        {
            auto code = j.at("source").get<std::string>();
            f_.bind_consts(P_); f_.compile_or_throw(code, "source");
        }

        // exact u (optional)
        if (j.contains("exact") && j["exact"].contains("u")) {
            std::string code = j["exact"]["u"].get<std::string>();
            u_.emplace(); u_->bind_consts(P_); u_->compile_or_throw(code, "exact.u");
        }

        // Dirichlet sides (optional)
        if (j.contains("dirichlet")){
            auto& d = j["dirichlet"];
            maybe_compile(dir_left_  , get_str_opt(d,"left")  , P_, "dirichlet.left");
            maybe_compile(dir_right_ , get_str_opt(d,"right") , P_, "dirichlet.right");
            maybe_compile(dir_bottom_, get_str_opt(d,"bottom"), P_, "dirichlet.bottom");
            maybe_compile(dir_top_   , get_str_opt(d,"top")   , P_, "dirichlet.top");
        }
        // Neumann sides (optional)
        if (j.contains("neumann")){
            auto& n = j["neumann"];
            maybe_compile(neu_left_  , get_str_opt(n,"left")  , P_, "neumann.left");
            maybe_compile(neu_right_ , get_str_opt(n,"right") , P_, "neumann.right");
            maybe_compile(neu_bottom_, get_str_opt(n,"bottom"), P_, "neumann.bottom");
            maybe_compile(neu_top_   , get_str_opt(n,"top")   , P_, "neumann.top");
        }
    }

    // accessors
    inline double alpha() const { return alpha_; }

    // Volume evaluations (shape-preserving)
    template<class Arr> inline Eigen::ArrayXXd f(const Arr& X, const Arr& Y) { return f_.eval2D(X,Y); }
    template<class Arr> inline Eigen::ArrayXXd u (const Arr& X, const Arr& Y) {
        if (u_) return u_->eval2D(X,Y);
        // if exact not provided, return zeros (caller may skip error)
        return Eigen::ArrayXXd::Zero(X.rows(), X.cols());
    }
    inline bool has_exact() const { return u_.has_value(); }

    // Dirichlet values on sides. If side expr missing, fallback to exact.u restricted; else zeros.
    inline Eigen::VectorXd g_left (double xLa, const Eigen::VectorXd& y) {
        if (dir_left_) return dir_left_->evalLineY(xLa, y);
        if (u_)        return u_->evalLineY(xLa, y);
        return Eigen::VectorXd::Zero(y.size());
    }
    inline Eigen::VectorXd g_right(double xLb, const Eigen::VectorXd& y) {
        if (dir_right_) return dir_right_->evalLineY(xLb, y);
        if (u_)         return u_->evalLineY(xLb, y);
        return Eigen::VectorXd::Zero(y.size());
    }
    inline Eigen::VectorXd g_bottom(const Eigen::VectorXd& x, double yLa) {
        if (dir_bottom_) return dir_bottom_->evalLineX(x, yLa);
        if (u_)          return u_->evalLineX(x, yLa);
        return Eigen::VectorXd::Zero(x.size());
    }
    inline Eigen::VectorXd g_top(const Eigen::VectorXd& x, double yLb) {
        if (dir_top_) return dir_top_->evalLineX(x, yLb);
        if (u_)       return u_->evalLineX(x, yLb);
        return Eigen::VectorXd::Zero(x.size());
    }

    // Neumann data on sides. If missing, default to 0.
    inline Eigen::VectorXd gN_left (double xLa, const Eigen::VectorXd& y) {
        if (neu_left_) return neu_left_->evalLineY(xLa, y);
        return Eigen::VectorXd::Zero(y.size());
    }
    inline Eigen::VectorXd gN_right(double xLb, const Eigen::VectorXd& y) {
        if (neu_right_) return neu_right_->evalLineY(xLb, y);
        return Eigen::VectorXd::Zero(y.size());
    }
    inline Eigen::VectorXd gN_bottom(const Eigen::VectorXd& x, double yLa) {
        if (neu_bottom_) return neu_bottom_->evalLineX(x, yLa);
        return Eigen::VectorXd::Zero(x.size());
    }
    inline Eigen::VectorXd gN_top(const Eigen::VectorXd& x, double yLb) {
        if (neu_top_) return neu_top_->evalLineX(x, yLb);
        return Eigen::VectorXd::Zero(x.size());
    }
};

} // namespace DG

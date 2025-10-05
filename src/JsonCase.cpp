#include "JsonCase.hpp"
#include <fstream>
#include <unordered_map>
#include <optional>
#include <stdexcept>
#include <sstream>

// heavy includes live *here*, not in the header:
#include "json.hpp"      // nlohmann::json
#include "exprtk.hpp"    // ExprTk

using json = nlohmann::json;

namespace DG {

struct JsonCase::Impl {
  struct Expr2D {
    exprtk::symbol_table<double> sym;
    exprtk::expression<double>   ex;
    exprtk::parser<double>       parser;
    double x=0.0, y=0.0;
    bool compiled=false;

    Expr2D(){ sym.add_variable("x",x); sym.add_variable("y",y); ex.register_symbol_table(sym); }
    void bind_consts(const std::unordered_map<std::string,double>& P){
      for (auto& kv: P) sym.add_constant(kv.first, kv.second);
    }
    void compile_or_throw(const std::string& code, const char* tag){
      if (code.empty()) throw std::runtime_error(std::string("empty expr: ")+tag);
      compiled = parser.compile(code, ex);
      if (!compiled) {
        std::ostringstream oss; oss << "bad expr: " << tag;
        if (parser.error_count()) {
          oss << " (";
          for (std::size_t i=0;i<parser.error_count();++i){
            auto er = parser.get_error(i);
            if (i) oss << "; ";
            oss << "#" << i << ": " << er.diagnostic
                << " [token='" << er.token.value << "' pos=" << er.token.position << "]";
          }
          oss << ")";
        }
        throw std::runtime_error(oss.str());
      }
    }
    inline double eval(double xx, double yy){ x=xx; y=yy; return ex.value(); }
    Eigen::ArrayXXd eval2D(const Eigen::ArrayXXd& X, const Eigen::ArrayXXd& Y){
      Eigen::ArrayXXd out(X.rows(), X.cols());
      for (int i=0;i<X.rows();++i)
        for (int j=0;j<X.cols();++j)
          out(i,j) = eval(X(i,j),Y(i,j));
      return out;
    }
    Eigen::VectorXd evalLineX(const Eigen::VectorXd& xv, double yfix){
      Eigen::VectorXd out(xv.size());
      for (int i=0;i<xv.size();++i) out(i) = eval(xv(i), yfix);
      return out;
    }
    Eigen::VectorXd evalLineY(double xfix, const Eigen::VectorXd& yv){
      Eigen::VectorXd out(yv.size());
      for (int i=0;i<yv.size();++i) out(i) = eval(xfix, yv(i));
      return out;
    }
  };

  std::unordered_map<std::string,double> P;
  double alpha = 1.0;

  Expr2D f;
  std::optional<Expr2D> u;

  std::optional<Expr2D> dL,dR,dB,dT;
  std::optional<Expr2D> nL,nR,nB,nT;

  static std::string slurp(const std::string& path){
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open JSON case file: "+path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  }
  static std::optional<std::string> get_opt(const json& j, const char* key){
    if (!j.contains(key) || j.at(key).is_null()) return std::nullopt;
    return j.at(key).get<std::string>();
  }
  static void maybe_compile(std::optional<Expr2D>& dst, const std::optional<std::string>& code,
                            const std::unordered_map<std::string,double>& P, const char* tag){
    if (!code) return;
    dst.emplace();
    dst->bind_consts(P);
    dst->compile_or_throw(*code, tag);
  }

  explicit Impl(const std::string& path){
    auto txt = slurp(path);
    json j = json::parse(txt, nullptr, true, /*ignore_comments=*/true);

    if (j.contains("parameters")) {
      for (auto it=j["parameters"].begin(); it!=j["parameters"].end(); ++it)
        P[it.key()] = it.value().get<double>();
    }
    if (j.contains("alpha")) alpha = j.at("alpha").get<double>();

    f.bind_consts(P);
    f.compile_or_throw(j.at("source").get<std::string>(), "source");

    if (j.contains("exact") && j["exact"].contains("u")){
      u.emplace(); u->bind_consts(P);
      u->compile_or_throw(j["exact"]["u"].get<std::string>(), "exact.u");
    }
    if (j.contains("dirichlet")){
      const auto& d=j["dirichlet"];
      maybe_compile(dL, get_opt(d,"left"),   P, "dirichlet.left");
      maybe_compile(dR, get_opt(d,"right"),  P, "dirichlet.right");
      maybe_compile(dB, get_opt(d,"bottom"), P, "dirichlet.bottom");
      maybe_compile(dT, get_opt(d,"top"),    P, "dirichlet.top");
    }
    if (j.contains("neumann")){
      const auto& n=j["neumann"];
      maybe_compile(nL, get_opt(n,"left"),   P, "neumann.left");
      maybe_compile(nR, get_opt(n,"right"),  P, "neumann.right");
      maybe_compile(nB, get_opt(n,"bottom"), P, "neumann.bottom");
      maybe_compile(nT, get_opt(n,"top"),    P, "neumann.top");
    }
  }
};

JsonCase::JsonCase(const std::string& path) : p_(std::make_unique<Impl>(path)) {}
JsonCase::~JsonCase() = default;
JsonCase::JsonCase(JsonCase&&) noexcept = default;
JsonCase& JsonCase::operator=(JsonCase&&) noexcept = default;

double JsonCase::alpha() const { return p_->alpha; }

Eigen::ArrayXXd JsonCase::f(const Eigen::ArrayXXd& X, const Eigen::ArrayXXd& Y) const {
  return p_->f.eval2D(X,Y);
}
Eigen::ArrayXXd JsonCase::u(const Eigen::ArrayXXd& X, const Eigen::ArrayXXd& Y) const {
  return p_->u ? p_->u->eval2D(X,Y) : Eigen::ArrayXXd::Zero(X.rows(),X.cols());
}
bool JsonCase::has_exact() const { return static_cast<bool>(p_->u); }

Eigen::VectorXd JsonCase::g_left (double xLa, const Eigen::VectorXd& y) const {
  if (p_->dL) return p_->dL->evalLineY(xLa,y);
  if (p_->u)  return p_->u->evalLineY(xLa,y);
  return Eigen::VectorXd::Zero(y.size());
}
Eigen::VectorXd JsonCase::g_right(double xLb, const Eigen::VectorXd& y) const {
  if (p_->dR) return p_->dR->evalLineY(xLb,y);
  if (p_->u)  return p_->u->evalLineY(xLb,y);
  return Eigen::VectorXd::Zero(y.size());
}
Eigen::VectorXd JsonCase::g_bottom(const Eigen::VectorXd& x, double yLa) const {
  if (p_->dB) return p_->dB->evalLineX(x,yLa);
  if (p_->u)  return p_->u->evalLineX(x,yLa);
  return Eigen::VectorXd::Zero(x.size());
}
Eigen::VectorXd JsonCase::g_top(const Eigen::VectorXd& x, double yLb) const {
  if (p_->dT) return p_->dT->evalLineX(x,yLb);
  if (p_->u)  return p_->u->evalLineX(x,yLb);
  return Eigen::VectorXd::Zero(x.size());
}

Eigen::VectorXd JsonCase::gN_left (double xLa, const Eigen::VectorXd& y) const {
  if (p_->nL) return p_->nL->evalLineY(xLa,y);
  return Eigen::VectorXd::Zero(y.size());
}
Eigen::VectorXd JsonCase::gN_right(double xLb, const Eigen::VectorXd& y) const {
  if (p_->nR) return p_->nR->evalLineY(xLb,y);
  return Eigen::VectorXd::Zero(y.size());
}
Eigen::VectorXd JsonCase::gN_bottom(const Eigen::VectorXd& x, double yLa) const {
  if (p_->nB) return p_->nB->evalLineX(x,yLa);
  return Eigen::VectorXd::Zero(x.size());
}
Eigen::VectorXd JsonCase::gN_top(const Eigen::VectorXd& x, double yLb) const {
  if (p_->nT) return p_->nT->evalLineX(x,yLb);
  return Eigen::VectorXd::Zero(x.size());
}

} // namespace DG

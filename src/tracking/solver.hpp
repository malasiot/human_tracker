#pragma once

#include <limits>
#include <cstdint>

#include <Eigen/Core>
#include <Eigen/Dense>
//#define HAS_JAC 1
template<typename T, typename F>
class Solver {
public:
    struct Parameters {
        T factor_ = (T)1.0e-3 ;
        T g_tol_ = std::numeric_limits<T>::epsilon() ; // ||J^T e||_inf
        T x_tol_ = std::numeric_limits<T>::epsilon() ; // ||Dp||_2
        T f_tol_ = std::numeric_limits<T>::epsilon() ; // ||e||_2
        size_t max_iter_ = 100 ;
        T delta_ = 1.0e-6 ;  // step used for finite difference approximation of derivatives
    };

    Solver() {}
    Solver(const Parameters &params): params_(params) {}

    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Matrix;

    // minimize objective function with analytic derivatives

    bool minimize(F &obj_func, Vector &p, bool dbg = false) {
        size_t m = p.size() ;
        size_t n = obj_func.terms() ;

        Vector e(n);

        obj_func.errors(p, e) ;

        T eL2 = e.squaredNorm(), pL2, dpL2 = 0, linf = 0;
        T mu, tau = params_.factor_ ;
        T eps2_sq = params_.x_tol_ * params_.x_tol_ ;
        uint nu = 2;
        bool finished = false ;

        for(uint k=0; k<params_.max_iter_ && !finished; ++k){
            /* Note that p and e have been updated at a previous iteration */

            if ( dbg ) {
                obj_func.debug(k, eL2, p) ;
            }

            if( eL2 <= params_.f_tol_) return true ;

            Matrix jtj(m, m) ;
            Vector jte(m) ;
            obj_func.norm(p, jtj, jte) ;

            Vector jdiag = jtj.diagonal() ;
            linf = jte.template lpNorm<Eigen::Infinity>() ;
            pL2 = p.squaredNorm() ;

            /* check for convergence */
            if( linf < params_.g_tol_ ) return true ;

            /* compute initial damping factor */
            if ( k==0 ) {
                T mdiag = -std::numeric_limits<T>::max() ;
                for( uint i=0 ; i<m ; i++ )
                    mdiag = std::max(mdiag, jdiag[i]) ;
                mu=tau*mdiag;
            }

            while (1) {
                /* augment normal equations */

                for(uint i=0; i<m; ++i)
                    jtj(i, i) += mu;

                // compute update
                Vector dp = jtj.ldlt().solve(jte);
                dpL2 = dp.squaredNorm() ;
                Vector pdp = p + dp ;

                if( dpL2 <= eps2_sq*pL2) { /* relative change in p is small, stop */
                    finished = true ;
                    break;
                }

                Vector h(n) ;
                obj_func.errors(pdp, h);

                T pdpL2 = h.squaredNorm(), dL = 0.0 ;

                for(uint i=0 ; i<m; ++i)
                    dL += dp[i]*(mu*dp[i]+jte[i]);

                T dF = eL2-pdpL2;

                if ( dL>0.0 && dF>0.0 ){ /* reduction in error, increment is accepted */
                    T tmp=(2.0*dF/dL-1.0);
                    tmp=1.0-tmp*tmp*tmp;
                    mu=mu*std::max(tmp,  (T)1.0/3);
                    nu=2;

                    p = pdp ;
                    e = h ;
                    eL2 = pdpL2 ;
                    break;
                }


                mu*=nu;
                uint nu2 = nu<<1; // 2*nu;
                if(nu2<=nu) break ;
                nu=nu2;

                for(uint i=0; i<m; ++i)
                    jtj(i, i) = jdiag(i);
            }
        }

        return finished ;

    }


private:


    Parameters params_ ;

};

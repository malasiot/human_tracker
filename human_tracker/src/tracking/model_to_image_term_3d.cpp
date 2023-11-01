#include "model_to_image_term.hpp"

#include <fstream>

using namespace std ;
using namespace cvx ;
using namespace Eigen ;

static const double g_scale_factor = 1.0e3 ;

using PointList3f = std::vector<Eigen::Vector3f> ;

double ModelToImageTerm::energy(const Pose &p) {
    PointList3f mpos, mnorm ;
    mesh_.getTransformedVertices(p, mpos, mnorm) ;

    double total = 0.0 ;
    int count = 0 ;
    for( const Vector3f &p: mpos ) {
        float dist ;
        if ( odf_.distance(p, dist) ) {
            count++ ;
            total += dist * dist ;
        }
    }

    if ( count > 0 )
        return total/count ;
    else
        return 1000.0 ;
}



pair<double, VectorXd> ModelToImageTerm::energyGradient(const Pose &pose) {

    const auto &skeleton = mesh_.skeleton_ ;

    VectorXd diffE(pose.coeffs().size()) ;
    diffE.setZero() ;

    vector<Matrix4f> bder, gder ;
    compute_transform_derivatives(skeleton, pose, bder, gder) ;

    PointList3f mpos, mnorm ;
    mesh_.getTransformedVertices(pose, mpos, mnorm) ;

    std::vector<Matrix4f> transforms ;
    skeleton.computeBoneTransforms(pose, transforms) ;

    uint N = mesh_.positions_.size() ;

    MatrixXd G(pose.coeffs().size(), N) ;
    G.setZero() ;

    const auto &pbv = skeleton.getPoseBones() ;

    size_t n_pose_bones = pbv.size() ;
    size_t n_global_params = Pose::global_rot_params + 3 ;

#define IDX3(i, j, k) (n_pose_bones * 4 * (i) + 4 * (j) + (k))
#define IDX2(i, j) ( n_global_params * (i) + j)

    int count = 0 ;
    double total = 0.0 ;
    for(size_t i=0 ; i<mesh_.positions_.size() ; i++)  {
        const Vector3f &pos = mpos[i] ;
        const Vector3f &orig = mesh_.positions_[i] ;

        Vector3f og ;
        if (! odf_.gradient(pos, og) ) continue ;

        float vd ;
        odf_.distance(pos, vd) ;

        total += vd * vd ;

        ++count ;

        const auto &bdata = mesh_.bones_[i] ;


        for(uint k=0 ; k<pbv.size() ; k++ ) {
            const auto &pb = pbv[k];

            for(uint r=0 ; r<pb.dofs() ; r++) {

                Matrix4f dG = Matrix4f::Zero() ;

                for( int j=0 ; j<MAX_BONES_PER_VERTEX ; j++)  {
                    int idx = bdata.id_[j] ;
                    if ( idx < 0 ) break ;

                    Matrix4f dQ = bder[IDX3(idx, k, r)] * skeleton.getBone(idx).offset_.inverse();

                    dG += dQ * (double)bdata.weight_[j] ;
                }

                Vector3f dp =  (dG * orig.homogeneous()).head(3) ;
                float gd = og.dot(dp) ;

                G(n_global_params + pb.offset() + r, i ) = 2 * vd * gd ;
            }
        }

        for(uint k=0 ; k<3 + Pose::global_rot_params ; k++ ) {
            Matrix4f dG = Matrix4f::Zero() ;

            for( int j=0 ; j<MAX_BONES_PER_VERTEX ; j++)  {
                 int idx = bdata.id_[j] ;
                 if ( idx < 0 ) break ;

                Matrix4f dQ = gder[IDX2(idx, k)] * skeleton.getBone(idx).offset_.inverse();

                dG += dQ * (double)bdata.weight_[j] ;
            }

            Vector3f dp =  (dG * orig.homogeneous()).head(3) ;

            float gd = og.dot(dp) ;

            G(k, i) = 2 * vd * gd ;
        }

    }

    if ( count > 0 ) {
        diffE = G.rowwise().sum()/count ;
        total /= count ;
    } else total = 100 ;

    return make_pair(total, diffE) ;
}


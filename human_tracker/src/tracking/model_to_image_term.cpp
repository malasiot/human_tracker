#include "model_to_image_term.hpp"
#include "distance_transform.hpp"
#include <fstream>
#include <cvx/imgproc/rgbd.hpp>

using namespace std ;
using namespace cvx ;
using namespace Eigen ;

static const double g_scale_factor = 1.0e3 ;

using PointList3f = std::vector<Eigen::Vector3f> ;

const float INF = 1.0e6 ;

void ModelToImageTerm::computeDistanceTransform(const cv::Mat &mask) {
//void distanceTransform2D(float *src, float *dst, unsigned int w, unsigned int h, bool take_sqrt) ;
    uint w = mask.cols, h = mask.rows ;

    cv::Mat_<float> src(h, w), dst(h, w), tmp(h, w) ;
    cv::Mat_<uchar> _mask(mask) ;

    for( uint i=0 ; i<h ; i++ ) {
        for( uint j=0 ; j<w ; j++ ) {
            src[i][j] = ( _mask[i][j] ) ? 0 : INF ;
        }
    }

    distanceTransform2D((float *)(src.data), (float *)dst.data, w, h, true) ;

    dt_ = cv::Mat(h, w, CV_32FC1) ;

    for( uint i=0 ; i<h ; i++ ) {
        for( uint j=0 ; j<w ; j++ ) {
            dt_.at<float>(i, j) = (_mask[i][j] ) ? 0 : dst[i][j] ;

        }
    }

    cv::Mat res(h, w, CV_32FC2, cv::Scalar(0, 0)) ;
    cv::Mat_<cv::Vec2f>  _res(res) ;

    for( uint i=0 ; i<h ; i++ ) {
        for( uint j=0 ; j<w ; j++ ) {
            if ( j < w-1 ) _res[i][j][0] = dt_.at<float>(i,j+1) - dt_.at<float>(i, j) ;
            if ( i < h-1 ) _res[i][j][1] = dt_.at<float>(i+1,j) - dt_.at<float>(i, j) ;
        }
    }

    dt_grad_ =  res ;

/*
    cv::Mat dtc ;
    dt_.convertTo(dtc, CV_8UC1);
    cv::imwrite("/tmp/dt.png",dtc) ;
    */
}

double ModelToImageTerm::energy(const Pose &p) {
    PointList3f mpos, mnorm ;
    mesh_.getTransformedVertices(p, mpos, mnorm) ;

    cv::Mat viz(dt_.size(), CV_8UC3, cv::Scalar(0, 0, 0)) ;

    double total = 0.0 ;
    for( const Vector3f &p: mpos ) {
        cv::Point2d pj1 = cam_.project(cv::Point3d(p.x(), -p.y(), -p.z()));
        float dist = dtValue({pj1.x + 0.5, pj1.y + 0.5});
        total += dist * dist ;
         cv::circle(viz, pj1, 1.0, cv::Scalar(255, 0, 0)) ;
    }

    cv::imwrite("/tmp/viz.png", viz) ;
    return total/mpos.size() ;
}

static Vector2f proj_deriv(float f, const Vector3f &p, const Vector3f &dp) {
    float X = p.x(), Y = p.y(), Z = p.z() ;
    float ZZ = Z * Z / f ;
    return { -( dp.x() * Z -  dp.z() * X )/ZZ,  ( dp.y() * Z -  dp.z() * Y )/ZZ } ;
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

   uint count = mesh_.positions_.size();
    double total = 0.0 ;
    for(size_t i=0 ; i<mesh_.positions_.size() ; i++)  {
        const Vector3f &pos = mpos[i] ;
        const Vector3f &orig = mesh_.positions_[i] ;

        cv::Point2d pj = cam_.project(cv::Point3d(pos.x(), -pos.y(), -pos.z()));
        Vector2f ip{pj.x + 0.5, pj.y + 0.5} ;



        Vector2f og = dtGradient(ip) ;

    //    cout << i << ' ' << pos.adjoint() << ' ' << ip.adjoint() << ' ' << og.adjoint() << endl ;

        float vd = dtValue(ip) ;

        total += vd * vd ;

        const auto &bdata = mesh_.bones_[i] ;

        for(uint k=0 ; k<pbv.size() ; k++ ) {
            const auto &pb = pbv[k];

            for(uint r=0 ; r<pb.dofs() ; r++) {

                Matrix4f dG = Matrix4f::Zero() ;

                for( int j=0 ; j<MAX_WEIGHTS_PER_VERTEX ; j++)  {
                    int idx = bdata.id_[j] ;
                    if ( idx < 0 ) break ;

                    Matrix4f dQ = bder[IDX3(idx, k, r)] * skeleton.getBone(idx).offset_.inverse();

                    dG += dQ * (double)bdata.weight_[j] ;
                }

                Vector3f dp =  (dG * orig.homogeneous()).head(3) ;

                Vector2f dpg = proj_deriv(cam_.fx(), pos, dp) ;
                float gd = og.dot(dpg) ;

                G(n_global_params + pb.offset() + r, i ) = 2 * vd * gd ;
            }
        }

        for(uint k=0 ; k<3 + Pose::global_rot_params ; k++ ) {
            Matrix4f dG = Matrix4f::Zero() ;

            for( int j=0 ; j<MAX_WEIGHTS_PER_VERTEX ; j++)  {
                 int idx = bdata.id_[j] ;
                 if ( idx < 0 ) break ;

                Matrix4f dQ = gder[IDX2(idx, k)] * skeleton.getBone(idx).offset_.inverse();

                dG += dQ * (double)bdata.weight_[j] ;
            }

            Vector3f dp =  (dG * orig.homogeneous()).head(3) ;

            Vector2f dpg = proj_deriv(cam_.fx(), pos, dp) ;
            float gd = og.dot(dpg) ;

            G(k, i) = 2 * vd * gd ;
        }

    }

    if ( count > 0 ) {
        diffE = G.rowwise().sum()/count ;
        total /= count ;
    } else total = 100 ;

    return make_pair(total, diffE) ;
}

float ModelToImageTerm::dtValue(const Eigen::Vector2f &pt)
{
    int x = (int)pt.x();
    int y = (int)pt.y();

    int x0 = cv::borderInterpolate(x,   dt_.cols, cv::BORDER_REFLECT_101);
    int x1 = cv::borderInterpolate(x+1, dt_.cols, cv::BORDER_REFLECT_101);
    int y0 = cv::borderInterpolate(y,   dt_.rows, cv::BORDER_REFLECT_101);
    int y1 = cv::borderInterpolate(y+1, dt_.rows, cv::BORDER_REFLECT_101);

    float a = pt.x() - (float)x;
    float c = pt.y() - (float)y;

    float v = (dt_.at<float>(y0, x0) * (1.f - a) + dt_.at<float>(y0, x1) * a) * (1.f - c)
                              + (dt_.at<float>(y1, x0) * (1.f - a) + dt_.at<float>(y1, x1) * a) * c;

    return v ;
}

Vector2f ModelToImageTerm::dtGradient(const Eigen::Vector2f &pt) {
    int x = (int)pt.x();
    int y = (int)pt.y();

    int x0 = cv::borderInterpolate(x,   dt_.cols, cv::BORDER_REFLECT_101);
    int x1 = cv::borderInterpolate(x+1, dt_.cols, cv::BORDER_REFLECT_101);
    int y0 = cv::borderInterpolate(y,   dt_.rows, cv::BORDER_REFLECT_101);
    int y1 = cv::borderInterpolate(y+1, dt_.rows, cv::BORDER_REFLECT_101);

    float a = pt.x() - (float)x;
    float c = pt.y() - (float)y;

    cv::Vec2f v = (dt_grad_.at<cv::Vec2f>(y0, x0) * (1.f - a) + dt_grad_.at<cv::Vec2f>(y0, x1) * a) * (1.f - c)
                              + (dt_grad_.at<cv::Vec2f>(y1, x0) * (1.f - a) + dt_grad_.at<cv::Vec2f>(y1, x1) * a) * c;

    return { v[0], v[1] };
}


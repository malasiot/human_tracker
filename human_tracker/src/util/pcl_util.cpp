#include <htrac/util/pcl_util.hpp>


#include <Eigen/Eigenvalues>

using namespace std;
using namespace Eigen ;
using namespace cvx ;

PointCloud depthToPointCloud(const cv::Mat &depth, const PinholeCamera &cam, uint cell_size) {

    float center_x = cam.cx();
    float center_y = cam.cy();

    const double unit_scaling = 0.001 ;

    float constant_x = unit_scaling / cam.fx();
    float constant_y = unit_scaling / cam.fy();

    cv::Mat_<ushort> depth_(depth) ;
    vector<Vector3f> coords ;

    for(int i=0 ; i<depth.rows ; i+=cell_size) {
        for(int j=0 ; j<depth.cols ; j+=cell_size)
        {
            ushort val = depth_[i][j] ;

            if ( val == 0 ) continue ;

            coords.push_back(Vector3f((j - center_x) * val * constant_x,
                                      -(i - center_y) * val * constant_y,
                                      -val * unit_scaling )) ;
        }
    }

    return coords ;
}

PointCloud depthToPointCloud(const cv::Mat &depth, const PinholeCamera &cam, const cv::Mat &mask, uint cell_size) {

    float center_x = cam.cx();
    float center_y = cam.cy();

    const double unit_scaling = 0.001 ;

    float constant_x = unit_scaling / cam.fx();
    float constant_y = unit_scaling / cam.fy();

    cv::Mat_<ushort> depth_(depth) ;
    cv::Mat_<uchar> mask_(mask) ;
    vector<Vector3f> coords ;


    for(int i=0 ; i<depth.rows ; i+=cell_size) {
        for(int j=0 ; j<depth.cols ; j+=cell_size)
        {
            if ( !mask_.empty() && mask_[i][j] == 0 ) continue ;

            ushort val = depth_[i][j] ;

            if ( val == 0 ) continue ;

            coords.push_back(Vector3f((j - center_x) * val * constant_x,
                                      -(i - center_y) * val * constant_y,
                                      -val * unit_scaling )) ;
        }
    }

    return coords ;
}

vector<Vector3f> depth_to_point_cloud(const cv::Mat &im, float fy)  {

    int w = im.cols, h = im.rows ;
    cv::Mat_<ushort> dim(im) ;
    vector<Vector3f> coords ;

    for( int i=0 ; i<h ; i++ ) {
        for( int j=0 ; j<w ; j++ ) {
            ushort dval = dim[i][j] ;
            if ( dval == 0 ) continue ;
            float z = -dval * 0.001f ;

            float x = (j - w/2.0)*z/fy ;
            float y = (i - h/2.0)*z/fy ;
            coords.push_back(Vector3f{-x, y, z}) ;
        }
    }

    return coords ;


}


Vector3f centroid(const vector<Vector3f> &pts) {
    Vector3f acc ;
    size_t n = 0 ;

    for( const auto &p: pts ) {
        acc += p ;
        n++ ;
    }
    return acc / n ;
}

Matrix3f covariance(const vector<Vector3f> &pts, const Vector3f &c) {
    Matrix3f acc = Matrix3f::Zero() ;
    size_t n = 0 ;

    for( const auto &p: pts ) {
        auto q = p - c ;
        acc += q * q.transpose() ;
        n++ ;
    }

    return acc / (n - 1) ;
}

void eigenvalues(const Matrix3f &cov, Vector3f &eval, Vector3f evec[3]) {
    Eigen::SelfAdjointEigenSolver<Matrix3f> eigensolver(cov);
    eval = eigensolver.eigenvalues() ;
    Matrix3f U = eigensolver.eigenvectors() ;

    evec[0] = U.col(0) ;
    evec[1] = U.col(1) ;
    evec[2] = U.col(2) ;
}

Vector3f box_dimensions(const vector<Vector3f> &pts, const Vector3f &c, const Vector3f evec[3]) {
    Vector3f vmax ;

    for( size_t i=0 ; i< pts.size() ; i++  ) {
        const auto &p = pts[i] ;
        Vector3f pj ;
        float x  = fabs(( p - c ).dot(evec[0])) ;
        float y  = fabs(( p - c ).dot(evec[1])) ;
        float z  = fabs(( p - c ).dot(evec[2])) ;

        if ( i == 0 )
            vmax = Vector3f{x, y, z} ;
        else {
            vmax.x() = std::max(vmax.x(), x) ;
            vmax.y() = std::max(vmax.y(), y) ;
            vmax.z() = std::max(vmax.z(), z) ;
        }

    }

    return vmax ;
}


VectorXf pcl_features(const std::vector<Vector3f> &pts) {
    VectorXf f ;

    f.resize(18) ;
    auto c = centroid(pts) ;
    auto cov = covariance(pts, c) ;
    Vector3f eval, evec[3] ;
    eigenvalues(cov, eval, evec) ;
    auto hdim = box_dimensions(pts, c, evec) ;

    size_t k = 0  ;
    f[k++] = c[0] ; f[k++] = c[1] ; f[k++] = c[2] ;
    f[k++] = eval[0] ; f[k++] = eval[1] ; f[k++] = eval[2] ;
    for ( size_t j=0 ; j<3 ; j++ ) {
        f[k++] = evec[j][0] ; f[k++] = evec[j][1] ; f[k++] = evec[j][2] ;
    }
    f[k++] = hdim[0] ; f[k++] = hdim[1] ; f[k++] = hdim[2] ;

    return f ;
}

BoundingBox get_bounding_box(const std::vector<Eigen::Vector3f> &pcl) {
    Vector3f minv = pcl[0], maxv = pcl[0] ;

    for( const auto &p: pcl ) {
        minv.x() = std::min(minv.x(), p.x()) ;
        minv.y() = std::min(minv.y(), p.y()) ;
        minv.z() = std::min(minv.z(), p.z()) ;
        maxv.x() = std::max(maxv.x(), p.x()) ;
        maxv.y() = std::max(maxv.y(), p.y()) ;
        maxv.z() = std::max(maxv.z(), p.z()) ;
    }

    return make_pair(minv, maxv) ;
}

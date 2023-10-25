#include "odf.hpp"

#include "distance_transform.hpp"
#include <htrac/util/pcl_util.hpp>

#include <cvx/misc/format.hpp>
#include <cvx/imgproc/rgbd.hpp>

using namespace std ;
using namespace cvx ;
using namespace Eigen ;

const float LARGE = 10000 ;

#define IDX3(x, y, z, o)  data[(z) * ncells_y_ * ncells_x_ + (y) * ncells_x_ + x + o] ;

bool ObservationsDistanceTransform::trilinear(float vx, float vy, float vz, float *data, uint offset, float &val) const
{
   float cx = (vx - bmin_.x())/cell_size_, cy = (vy - bmin_.y())/cell_size_, cz = (vz - bmin_.z())/cell_size_ ;

   uint icx = cx, icy = cy, icz = cz ;

   if ( icx < 0 || icy < 0 || icz < 0 || icx >= ncells_x_-1 || icy >= ncells_y_-1 || icz >= ncells_z_-1 ) return false ;

    float hx = cx - icx, hy = cy - icy, hz = cz - icz ;

    float c000 = IDX3(icx, icy, icz, offset) ;
    float c100 = IDX3(icx+1, icy, icz, offset) ;
    float c010 = IDX3(icx, icy+1, icz, offset) ;
    float c001 = IDX3(icx, icy, icz+1, offset) ;
    float c110 = IDX3(icx+1, icy+1, icz, offset) ;
    float c011 = IDX3(icx, icy+1, icz+1, offset) ;
    float c101 = IDX3(icx+1, icy, icz+1, offset) ;
    float c111 = IDX3(icx+1, icy+1, icz+1, offset) ;

    val = (1-hx)*(1-hy)*(1-hz)*c000 + hx*(1-hy)*(1-hz)*c100 + (1-hx)*hy*(1-hz)*c010 + hx*hy*(1-hz)*c110 +
            (1-hx)*(1-hy)*hz*c001 + hx*(1-hy)*hz*c101 + (1-hx)*hy*hz*c011 + hx*hy*hz*c111;

    return true ;
}

using PointList3f = std::vector<Eigen::Vector3f> ;
//#define DEBUG
void ObservationsDistanceTransform::compute(const SkinnedMesh &mesh, const Pose &p, const cv::Mat &im, const PinholeCamera &cam) {
    PointList3f mpos, mnorm ;
    mesh.getTransformedVertices(p, mpos, mnorm) ;

    std::tie(bmin_, bmax_) = get_bounding_box(mpos) ;

    Vector3f pad(padding_, padding_, padding_) ;
    bmin_ -= pad ;
    bmax_ += pad ;

    ncells_x_ = round((bmax_.x() - bmin_.x())/cell_size_) ;
    ncells_y_ = round((bmax_.y() - bmin_.y())/cell_size_) ;
    ncells_z_ = round((bmax_.z() - bmin_.z())/cell_size_) ;

    size_t sz = ncells_x_ * ncells_y_ * ncells_z_ ;

    std::unique_ptr<float []> vol(new float[sz]) ;
    cv::Mat_<ushort> dim(im) ;

    float minz = -bmax_.z(), maxz = -bmin_.z() ;

    float *vp = vol.get() ;
    for( int z = 0 ; z<ncells_z_ ; z++ ) {

        for( int y=0 ; y<ncells_y_ ; y++ ) {
            for ( int x=0 ; x<ncells_x_ ; x++ ) {
                float X = bmin_.x() + x*cell_size_ ;
                float Y = bmin_.y() + y*cell_size_ ;
                float Z = bmin_.z() + z*cell_size_ ;

                Y = -Y ;
                Z = -Z ;

                cv::Point2d p = cam.project(cv::Point3d(X, -Y, -Z)) ;

                int ix = round(p.x) ;
                int iy = round(p.y) ;

                if ( ix < im.cols && ix >= 0 && iy < im.rows && iy >=0 )  {
                    ushort val = dim[iy][ix] ;
                    float zval = val/1000.0 ;
                    if ( val == 0 || fabs(Z - zval) > cell_size_ || zval < minz || zval > maxz  ) {
                        *vp = LARGE ;
                    }
                    else {
                        *vp = 0 ;
                    }
                } else {
                    *vp = LARGE ;
                }

                ++vp ;
            }
        }

    }

    dist_.reset(new float [sz]) ;

    signedDistanceTransform3D(vol.get(), dist_.get(), ncells_x_, ncells_y_, ncells_z_, true) ;

    std::transform(dist_.get(), dist_.get() + sz, dist_.get(), [this](float v) { return v * cell_size_ ; });
#if 0
    gradient_.reset(new float [sz*3]) ;

    float *gp = gradient_.get() ;

#define IDX(x, y, z) ((z) * ncells_x_ * ncells_y_ + (y) * ncells_x_ + (x))

    for( int iz=0 ; iz < ncells_z_ ; iz++)
        for( int iy=0 ; iy < ncells_y_ ; iy++)
            for( int ix=0 ; ix < ncells_x_ ; ix++)
            {
                int idx = IDX(ix, iy, iz) ;

            //    dist_[idx] *= cell_size_ ;
                int idx1x = IDX(ix-1, iy, iz) ;
                int idx1y = IDX(ix, iy-1, iz) ;
                int idx1z = IDX(ix, iy, iz-1) ;

                float gx, gy , gz ;

                if ( ix == 0 ) gx = 0.0 ;
                else {
                    gx = (dist_[idx] - dist_[idx1x]) ;
                }

                if ( iy == 0 ) gy = 0.0 ;
                else {
                    gy = (dist_[idx] - dist_[idx1y]) ;
                }

                if ( iz == 0 ) gz = 0.0 ;
                else {
                    gz = (dist_[idx] - dist_[idx1z]) ;
                }

                *gp++ = gx ;
                *gp++ = gy ;
                *gp++ = gz ;
            }
#endif
#ifdef DEBUG
    cv::Mat dbim(ncells_y_, ncells_x_, CV_8UC1)  ;

    for( int k=0 ; k<ncells_z_ ; k++ ) {
        for( int i=0 ; i<ncells_y_ ; i++) {
            for(int j = 0 ; j<ncells_x_ ; j++ ) {
                float v = dist_[k * ncells_y_ * ncells_x_ + i * ncells_x_ + j] ;
                 dbim.at<uchar>(i, j) = 200 * fabs(v);
            }
        }
        string fname = cvx::format("/tmp/dt{:02d}.png", k) ;
        cv::imwrite(fname, dbim) ;
    }
#endif

}

bool ObservationsDistanceTransform::distance(const Vector3f &p, float &val) const
{
    float ix = p.x(), iy = p.y(), iz = p.z() ;
    return trilinear(ix, iy, iz, dist_.get(), 0, val ) ;

}

bool ObservationsDistanceTransform::gradient(const Vector3f &p, Vector3f &g) const
{
    float ix = p.x(), iy = p.y(), iz = p.z() ;

    float gx, gy, gz ;

    float delta = 0.001 ;

    float ix0, ix1 ;
    if ( !trilinear(ix-delta, iy, iz, dist_.get(), 0, ix0) ) return false ;
    if ( !trilinear(ix+delta, iy, iz, dist_.get(), 0, ix1) ) return false ;
    gx = (ix1 - ix0)/(2*delta) ;

    float iy0, iy1 ;
    if ( !trilinear(ix, iy-delta, iz, dist_.get(), 0, iy0) ) return false ;
    if ( !trilinear(ix, iy+delta, iz, dist_.get(), 0, iy1) ) return false ;
    gy = (iy1 - iy0)/(2*delta) ;

    float iz0, iz1 ;
    if ( !trilinear(ix, iy, iz-delta, dist_.get(), 0, iz0) ) return false ;
    if ( !trilinear(ix, iy, iz+delta, dist_.get(), 0, iz1) ) return false ;
    gz = (iz1 - iz0)/(2*delta) ;

//    if ( !trilinear(ix, iy, iz, dist_.get(), 0, gx) ) return false ;
//    if ( !trilinear(ix, iy, iz, gradient_.get(), 1, gy) ) return false ;
//    if ( !trilinear(ix, iy, iz, gradient_.get(), 2, gz) ) return false ;

    g << gx, gy, gz ;

    return true ;
}

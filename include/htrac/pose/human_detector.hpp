#ifndef HUMAN_DETECTOR_HPP
#define HUMAN_DETECTOR_HPP

#include <string>
#include <opencv2/opencv.hpp>
#include <cvx/camera/camera.hpp>
#include <opencv2/rgbd/linemod.hpp>
#include <Eigen/Geometry>

using Plane = Eigen::Hyperplane<float, 3> ;

class OcclusionSpace {
public:
    OcclusionSpace() = default ;

    // implement to return true if a point should be disregarded
    virtual bool occluded(const Eigen::Vector3f &pt) const = 0 ;
};

// space defined by a list of planes (all points behind a plane are discarded)
struct Polytope: public OcclusionSpace {
public:
    Polytope() = default ;
    Polytope(const std::vector<Plane> &planes): planes_(planes) {}

    bool occluded(const Eigen::Vector3f &pt) const {
        for( const auto &p: planes_ ) {
            if ( p.signedDistance(pt) < 0 ) return true ;
        }
        return false ;
    }

private:
    std::vector<Plane> planes_ ;
};

using BoundingBox = std::pair<Eigen::Vector3f, Eigen::Vector3f> ;

/* The detector works by:
/* 1. Rejecting points outside occlusion space
 * 2. Projecting remaining points to ground plane coordinate system
 * 3. Create 2d occupancy grid and corresponding heightmap
 * 4. Find connected components and select the one with the heighest value (distance from ground plane)
 * 5. Creating a bounding box around the 3d points associated with the blob.
 */
class HumanDetector {
public:

    struct Parameters {
        float grid_cell_sz_ ; // size of the occupancy grid cell
        float occ_threshold_ ; // threshold of occupancy values above which the cell is consider among candidates
        float occ_min_height_ ; // eleminate pixels with height lower than this
        float occ_min_area_ ;   // eleminate blobs with area less that this
        float occ_max_area_ ;  // range of blob size to consider in occupancy map
        float occ_gaussian_mask_sz_ ; // size (in meters) of Gaussian blur mask

        Parameters(): grid_cell_sz_(0.05), occ_threshold_(20),
            occ_min_area_(0.2 * 0.2), occ_max_area_(0.8 * 0.8), occ_min_height_(0.5), occ_gaussian_mask_sz_(0.25) {}
    };

    HumanDetector() {}
    HumanDetector(const Parameters &params): params_(params) {}

    void setOcclusionSpace(OcclusionSpace *os) { ocs_.reset(os) ; }
    void setGroundPlane(const Plane &gp) { gp_ = gp ; }

    bool detect(const cv::Mat &depth, const cvx::PinholeCamera &cam, cv::Rect &box, cv::Mat &mask) ;

private:

    struct OccCell {
        std::vector<uint> cpts_ ; // cloud points in this cell
    };

    struct OccGrid {

        float cell_sz ;
        uint n_bins_x_, n_bins_y_ ;

        std::map<uint, OccCell> data_ ;
    };

    typedef std::map<uint, OccCell> OccGridType ;
    typedef OccGridType::const_iterator OccGridIterator ;


private:

    void segmentValidPoints(const cv::Mat &depth, const cvx::PinholeCamera &model, std::vector<Eigen::Vector3f> &plane_pts,
                            std::vector<Eigen::Vector3f> &ipts, cv::Mat &mask);

    void makeOccupancyGrid(const std::vector<Eigen::Vector3f> &pts,
                           cv::Mat &occ_mask, OccGrid &occ_grid, cv::Mat &height_map) ;

    bool findCandidate(const cv::Mat &occ_mask, const cv::Mat &height_map,
                                       HumanDetector::OccGrid &cells,
                                       std::vector<size_t> &indices);

    void getCandidateIndices(const cv::Mat &labels, ulong label, const cv::Rect &rect, OccGrid &cells, std::vector<size_t> &indices) ;

    float maxHeight(const cv::Mat &height_map, const cv::Mat &labels, unsigned long label, const cv::Rect &rect) ;
    cv::Rect box2Rect(const BoundingBox &box, const cvx::PinholeCamera &cam);

    Parameters params_ ;

    std::shared_ptr<OcclusionSpace> ocs_ ;
    Plane gp_ ;
};


#endif // HUMAN_DETECTOR_HPP

#include <htrac/pose/dataset.hpp>

#include <cvx/misc/path.hpp>
#include <cvx/math/rng.hpp>
#include <cvx/misc/json_reader.hpp>

#include <fstream>

using namespace cvx ;
using namespace std ;
using namespace Eigen ;

static RNG g_rng ;

CERTHDataset::CERTHDataset(const std::string &root_dir, bool shuffle): root_dir_(root_dir) {
    auto annotations = Path::glob(root_dir, "*/j_*.json", true, true) ;

    for( const auto &a: annotations ) {
        string ref = a.substr(0, 9) ;
        string id = ref.substr(0, 3) + ref.substr(6, 3) ;
        ids_.push_back(id) ;
    }

    if ( shuffle )
        g_rng.shuffle(ids_) ;
    else
        std::sort(ids_.begin(), ids_.end()) ;
}

static string make_path_from_id(const string &id, const char *prefix, const char *suffix) {
    return id.substr(0, 3) + '/' + prefix + '_' + id.substr(3, 3) + suffix ;
}

cv::Mat CERTHDataset::getDepthImage(uint64_t idx) const {
    assert(idx < size()) ;
    auto path = make_path_from_id(ids_[idx], "d", ".png") ;
    return cv::imread(Path::join(root_dir_, path), -1) ;
}

cv::Mat CERTHDataset::getPartLabelImage(uint64_t idx) const {
    assert(idx < size()) ;
    auto path = make_path_from_id(ids_[idx], "l", ".png") ;
    return cv::imread(Path::join(root_dir_, path), -1) ;
}

vector<CERTHDataset::Annotation> CERTHDataset::getAnnotation(uint64_t idx) const {
    assert(idx < size()) ;
    auto jpath = make_path_from_id(ids_[idx], "j", ".json") ;
    auto ppath = make_path_from_id(ids_[idx], "p", ".json") ;

    return parseAnnotation(jpath, ppath) ;
}

PinholeCamera CERTHDataset::getCamera() const {
    return PinholeCamera(275, 275, img_size_.width/2.0f, img_size_.height/2.0f, img_size_) ;
}

std::vector<CERTHDataset::Annotation> CERTHDataset::parseAnnotation(const std::string &jp, const std::string &pp) const {
    vector<Annotation> jan ;
    {
        ifstream strm(pp) ;
        JSONReader json(strm) ;
        json.beginObject() ;
        while ( json.hasNext() ) {
            string jname = json.nextName() ;
            Annotation a ;
            json.beginObject() ;
            while ( json.hasNext() ) {
                string key = json.nextName() ;
                if ( key == "c" ) {
                    json.beginArray() ;
                    a.ipt_.x() = json.nextDouble() ;
                    a.ipt_.y() = json.nextDouble() ;
                    a.ipt_.z() = json.nextDouble() ;
                    json.endArray() ;
                } else if ( key == "v") {
                    a.visible_ = json.nextBoolean() ;
                }
            }

            json.endObject() ;

            jan.emplace_back(std::move(a)) ;
        }

        json.endObject() ;
    }

    {

        int i=0 ;
        ifstream strm(jp) ;
        JSONReader json(strm) ;
        json.beginObject() ;
        while ( json.hasNext() ) {
            string jname = json.nextName() ;
            Vector3f c ;
            json.beginArray() ;
            c.x() = json.nextDouble() ;
            c.y() = json.nextDouble() ;
            c.z() = json.nextDouble() ;
            json.endArray() ;

            jan[i].coords_ = c ;
            ++i ;
        }

        json.endObject() ;
    }

    return jan ;
}

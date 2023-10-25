#include <htrac/pose/dataset.hpp>
#include <cvx/misc/path.hpp>
#include <cvx/misc/strings.hpp>
#include <cvx/math/rng.hpp>
#include <cvx/misc/json_reader.hpp>

#include <fstream>

using namespace std ;
using namespace cvx ;

static RNG g_rng ;

vector<string> ITOP::joint_names_ = {
  "Head", "Torso", "Neck", "R Hip",
  "R Shoulder",  "L Hip",
  "L Shoulder",  "R Knee",
  "R Elbow",     "L Knee",
  "L Elbow",     "R Foot",
  "R Hand",      "L Foot",
  "L Hand",
};


ITOP::ITOP(const std::string &root_dir, const std::string &exclude, bool shuffle): root_dir_(root_dir) {
    vector<string> annotations = Path::glob(root_dir, "*.json") ;
    set<string> ex_set ;

    auto ids = cvx::split(exclude, ",") ;
    std::copy(ids.begin(), ids.end(), std::inserter(ex_set, ex_set.begin()));

    for( const auto &a: annotations ) {
        string ref = a.substr(0, 8) ;
        string id = ref.substr(0, 2) ;
         if ( !ex_set.count(id) )
            ids_.push_back(ref) ;
    }

    if ( shuffle )
        g_rng.shuffle(ids_) ;
    else
        std::sort(ids_.begin(), ids_.end()) ;

}

cv::Mat ITOP::getDepthImage(uint64_t idx) const {
    assert(idx < size()) ;
    return cv::imread(Path::join(root_dir_, ids_[idx] + ".png"), -1) ;
}

std::vector<ITOP::Annotation> ITOP::getAnnotation(uint64_t idx) const {
    assert(idx < size()) ;
    return parseAnnotation(Path::join(root_dir_, ids_[idx] + ".json")) ;
}

PinholeCamera ITOP::getCamera() const {
    return PinholeCamera(286, 286, img_size_.width/2.0f, img_size_.height/2.0f, img_size_) ;
}

vector<ITOP::Annotation> ITOP::parseAnnotation(const std::string &p) const {
    ifstream strm(p) ;
    assert(strm) ;

    vector<Annotation> joints ;

    JSONReader json(strm) ;

    json.beginObject() ;
    while ( json.hasNext() ) {
        string key = json.nextName() ;
        bool is_valid ;
        if ( key == "is_valid" )
            is_valid = json.nextBoolean() ;
        else if ( key == "joints" ) {
           json.beginArray() ;
           while ( json.hasNext() ) {
               json.beginObject() ;
               Annotation a ;
               while ( json.hasNext() ) {
                   string key = json.nextName() ;
                   if ( key == "pt") {
                       json.beginArray() ;
                       a.ipt_.x = json.nextInt() ;
                       a.ipt_.y = json.nextInt() ;
                       json.endArray() ;
                   } else if ( key == "coords" ) {
                       json.beginArray() ;
                       a.coords_.x() = json.nextDouble() ;
                       a.coords_.y() = json.nextDouble() ;
                       a.coords_.z() = json.nextDouble() ;
                       json.endArray() ;
                   } else if ( key == "visible") {
                       a.visible_ = json.nextBoolean() ;
                   }
               }
               joints.emplace_back(std::move(a)) ;
               json.endObject() ;
           }
           json.endArray() ;
        }
    }
    json.endObject() ;

    return joints ;
}

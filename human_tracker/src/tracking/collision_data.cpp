#include "collision_data.hpp"
#include <cvx/misc/json_reader.hpp>
#include <cvx/misc/strings.hpp>
#include <fstream>

using namespace std ;
using namespace cvx ;

void CollisionData::parseJson(const Skeleton &sk, const std::string &fpath) {
    ifstream strm(fpath) ;

    JSONReader json(strm) ;

    json.beginArray() ;
    while ( json.hasNext() ) {
        json.beginObject() ;
        Sphere s ;
        while ( json.hasNext() ) {
            string token = json.nextName() ;
            if ( token == "group" )
                s.group_ = json.nextString() ;
            else if ( token == "bone" ) {
                s.name_ = json.nextString() ;
                string bname = cvx::split(s.name_, ".")[0] ;
                s.bone_ = sk.getBoneIndex(bname) ;
            }
            else if ( token == "center" ) {
                json.beginArray() ;
                s.c_.x() = json.nextDouble() ;
                s.c_.y() = json.nextDouble() ;
                s.c_.z() = json.nextDouble() ;
                json.endArray() ;
            } else if ( token == "radius" ) {
                s.r_ = json.nextDouble() ;
            }
        }

        s.c_ = (sk.getBone(s.bone_).offset_.inverse() * s.c_.homogeneous()).head<3>() ;

        spheres_.emplace_back(std::move(s)) ;

        json.endObject() ;
    }

    json.endArray() ;

}

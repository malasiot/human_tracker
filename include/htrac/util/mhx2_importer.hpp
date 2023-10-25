#ifndef MHX2_IMPORTER
#define MHX2_IMPORTER

#include <vector>
#include <map>
#include <deque>
#include <Eigen/Core>

#include <cvx/misc/json_reader.hpp>

#include "mhx2.hpp"

class Mhx2Importer {
public:
    Mhx2Importer() = default ;

    bool load(const std::string &fname, bool zup=false) ;
    bool loadSkeleton(const std::string &fname, bool zup=false) ;

    const MHX2Model &getModel() const { return model_ ; }

private:

    using JSONReader = cvx::JSONReader ;

    bool parseSkeleton(JSONReader &r, bool zup) ;
    bool parseGeometries(JSONReader &v, bool zuo) ;
    bool parseMesh(MHX2Mesh &geom, JSONReader &v, bool zup) ;
    bool parseVertexGroups(MHX2Mesh &mesh, JSONReader &v) ;
    bool parseMaterials(JSONReader &reader, const std::string &fpath) ;

    MHX2Model model_ ;
};

#endif

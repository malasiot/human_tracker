#include <htrac/util/io_util.hpp>
#include <fstream>

using namespace Eigen ;
using namespace std ;
using namespace cvx ;

MatrixXf parse_json_matrix(JSONReader &json) {
    vector<vector<float>> data ;

    size_t rows, cols ;

    json.beginArray() ;
    while ( json.hasNext() ) {
        vector<float> row ;
        json.beginArray() ;
        while( json.hasNext() ) {
            row.push_back(json.nextDouble());
        }
        json.endArray() ;

        cols = row.size() ;
        data.push_back(std::move(row)) ;

    }
    json.endArray() ;
    rows = data.size() ;

    MatrixXf r(rows, cols) ;
    for( size_t i=0 ; i<rows ; i++ )
        for( size_t j=0 ; j<cols ; j++ ) {
            r(i, j) = data[i][j] ;
        }
    return r ;
}

void save_joint_coords(const map<string, Vector3f> &joints, const string &path) {
    ofstream strm(path) ;
    for( const auto &vp: joints ) {
        strm << "v " << vp.second.adjoint() << endl ;
    }
}

void save_point_cloud(const std::vector<Vector3f> &coords, const std::string &path) {
    ofstream strm(path) ;
    for( const auto &v: coords ) {
        strm << "v " << v.adjoint() << endl ;
    }
}

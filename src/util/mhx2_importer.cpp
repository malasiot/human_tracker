#include <htrac/util/mhx2_importer.hpp>

#include <fstream>
#include <iostream>

#include <Eigen/Geometry>

#include <cvx/misc/path.hpp>
#include <algorithm>

using namespace std ;
using namespace Eigen ;
using namespace cvx ;

bool Mhx2Importer::load(const string &fname, bool zup)
{
    ifstream strm(fname) ;
    if ( !strm ) return false ;

    try {
        JSONReader reader(strm) ;

        reader.beginObject() ;

        while ( reader.hasNext() ) {
            string name = reader.nextName() ;
            if ( name == "geometries" ) {
                parseGeometries(reader, zup) ;
            } else if ( name == "skeleton" ) {
                parseSkeleton(reader, zup) ;
            }
                else if ( name == "materials" ) {
                parseMaterials(reader, fname) ;
            } else {
                reader.skipValue() ;
            }
        }

        reader.endObject() ;

        return true ;
    }
    catch ( JSONParseException &e ) {
        cerr << "Error parsing MHX2 file: " << fname << " (" ;
        cerr << e.what() << ")" ;
        return false ;
    }


}

bool Mhx2Importer::loadSkeleton(const string &fname, bool zup) {
    ifstream strm(fname) ;

    try {
        JSONReader reader(strm) ;

        reader.beginObject() ;

        while ( reader.hasNext() ) {
            string name = reader.nextName() ;
            if ( name == "skeleton" ) {
                parseSkeleton(reader, zup) ;
            } else {
                reader.skipValue() ;
            }
        }

        reader.endObject() ;

        return true ;
    }
    catch ( JSONParseException &e ) {
        cerr << "Error parsing MHX2 file: " << fname << " (" ;
        cerr << e.what() << ")" ;
        return false ;
    }
}

static Vector3f normal_triangle(const Vector3f &v1, const Vector3f &v2, const Vector3f &v3)
{
    Vector3f n1, n2 ;

    n1 = v1 - v2 ;
    n2 = v1 - v3 ;
    return  n1.cross(n2).normalized() ;

}

void MHX2Mesh::computeNormals() {
    size_t n = vertices_.size() ;
    normals_.resize(n) ;

    for( unsigned int i=0 ; i<n ; i++ ) normals_.data()[i] = Vector3f::Zero() ;

    for( unsigned int i=0 ; i<faces_.size() ; i++ )
    {
        const MHX2Face &face = faces_[i] ;

        unsigned int idx0 = face.indices_[0] ;
        unsigned int idx1 = face.indices_[1] ;
        unsigned int idx2 = face.indices_[2] ;
        Vector3f n = normal_triangle(vertices_[idx0], vertices_[idx1], vertices_[idx2]) ;

        // face normal

        for( size_t k=0 ; k<face.num_vertices_ ; k++ ) {
            auto idx = face.indices_[k] ;
            normals_[idx] += n ;
        }

    }

    for( unsigned int i=0 ; i<n ; i++ ) normals_[i].normalize() ;
}

static Vector3f toVector3(JSONReader &r) {
    r.beginArray() ;
    float x = r.nextDouble() ;
    float y = r.nextDouble() ;
    float z = r.nextDouble() ;
    r.endArray() ;
    return Vector3f(x, y, z) ;
}

static Vector2f toVector2(JSONReader &r) {
    r.beginArray() ;
    float x = r.nextDouble() ;
    float y = r.nextDouble() ;
    r.endArray() ;
    return Vector2f(x, y) ;
}


static Matrix4f toMatrix4(JSONReader &r) {

    Matrix4f m ;

    r.beginArray() ;
    for( size_t row = 0 ; row < 4 ; row ++ ) {
        r.beginArray() ;
        for( size_t col = 0 ; col < 4 ; col ++ ) {
            m(row, col) = (float)r.nextDouble() ;
        }
        r.endArray() ;
    }
    r.endArray() ;

    return m ;
}

bool Mhx2Importer::parseMaterials(JSONReader &reader, const std::string &fpath) {

    Path dir = Path(fpath).parentPath() ;

    reader.beginArray() ;

    while ( reader.hasNext() ) {

        string materialName ;
        MHX2Material current ;

        reader.beginObject() ;
        while ( reader.hasNext() ) {
            string name = reader.nextName() ;
            if ( name == "name")
                materialName = reader.nextString() ;
            else if ( name == "diffuse_color" )
                current.diffuse_color_ = toVector3(reader) ;
            else if ( name == "ambient_color" )
                current.ambient_color_ = toVector3(reader) ;
            else if ( name == "specular_color" )
                current.specular_color_ = toVector3(reader) ;
            else if ( name == "shininess" )
                current.shininess_ = (float)reader.nextDouble() ;
            else if ( name == "opacity" )
                current.opacity_ = (float)reader.nextDouble() ;
            else if ( name == "diffuse_texture" )
                current.diffuse_texture_ = Path(dir, reader.nextString()).native() ;
            else reader.skipValue() ;
        }

        if ( !materialName.empty() )
            model_.materials_[materialName] = current ;

        reader.endObject() ;
    }

    reader.endArray() ;

    return true ;
}

extern float angle_normalized_v3v3(const Vector3f &v1, const Vector3f &v2) ;

static Vector3f zUp(const Vector3f &src, bool conv) {
    return (conv) ? Vector3f{src[0], -src[2], src[1]} : src ;
}

bool Mhx2Importer::parseSkeleton(JSONReader &reader, bool zup) {

    string skeletonName ;
    Vector3f offset ;
    float scale = 1.0 ;

    reader.beginObject() ;

    while ( reader.hasNext() ) {
        string name = reader.nextName() ;
        if ( name == "name" )
            skeletonName = reader.nextString() ;
        else if ( name == "offset" ) {
            offset = zUp(toVector3(reader), zup) ;
            model_.offset_ = offset ;
        }
        else if ( name == "scale" )
            scale = (float)reader.nextDouble() ;
        else if ( name == "bones" ) {
            reader.beginArray() ;
            while ( reader.hasNext() ) {
                string boneName ;
                Vector3f head, tail ;
                Matrix4f mat ;
                double roll = 0 ;
                string parent ;

                reader.beginObject() ;
                while ( reader.hasNext() ) {
                    string name = reader.nextName() ;
                    if ( name == "name" ) {
                        boneName = reader.nextString()  ;
                    } else if ( name == "head" ) {
                        head = zUp(toVector3(reader), zup) ;
                    } else if ( name == "tail" ) {
                        tail = zUp(toVector3(reader), zup) ;
                    } else if ( name == "matrix" ) {
                        mat = toMatrix4(reader) ;
                    } else if ( name == "roll" ) {
                        roll = reader.nextDouble() ;
                    } else if ( name == "parent" ) {
                        parent = reader.nextString() ;
                    }
                }
                reader.endObject() ;

                MHX2Bone mhbone ;
                mhbone.head_ = head + offset ;
                mhbone.tail_ = tail + offset ;
                mhbone.roll_ = (float)roll ;



                Matrix4f bmat = mat ;

                if ( zup ) {
                    bmat.row(0) = mat.row(0) ;
                    bmat.row(1) = -mat.row(2) ;
                    bmat.row(2) = mat.row(1) ;
                }

                 bmat.block<3, 1>(0, 3) = mhbone.head_ ;

                mhbone.bmat_ = bmat ;

                mhbone.parent_ = parent ;

                model_.bones_.emplace(boneName, mhbone) ;
            }
            reader.endArray() ;
        } else {
            reader.skipValue() ;
        }
    }

    reader.endObject() ;

    return true ;
}

bool Mhx2Importer::parseGeometries(JSONReader &reader, bool zup)
{
    reader.beginArray() ;

    while ( reader.hasNext() ) {

        string geomName ;
        MHX2Geometry current ;

        reader.beginObject() ;

        while ( reader.hasNext() ) {
            string name = reader.nextName() ;
            if ( name == "name" )
                geomName = reader.nextString() ;
            else if ( name == "offset" )
                current.offset_ = zUp(toVector3(reader), zup) ;
            else if ( name == "scale" )
                current.scale_ = (float)reader.nextDouble() ;
            else if ( name == "mesh" )
                parseMesh(current.mesh_, reader, zup) ;
            else if ( name == "material" )
                current.material_ = reader.nextString() ;
            else
                reader.skipValue() ;
        }

        if ( !geomName.empty() )
            model_.geometries_.emplace(geomName, std::move(current)) ;

        reader.endObject() ;
    }


    reader.endArray() ;

    return true ;
}

bool Mhx2Importer::parseMesh(MHX2Mesh &mesh, JSONReader &reader, bool zup)
{
    reader.beginObject() ;

    while( reader.hasNext() ) {
        string name = reader.nextName() ;
        if ( name == "vertices" ) {
            reader.beginArray() ;
            while ( reader.hasNext() ) {
                Vector3f v = zUp(toVector3(reader), zup) ;
                mesh.vertices_.emplace_back(v) ;
            }
            reader.endArray() ;
        } else if ( name == "uv_coordinates" ) {
            reader.beginArray() ;
            while ( reader.hasNext() ) {
                Vector2f v = toVector2(reader) ;
                mesh.uv_coords_.emplace_back(v) ;
            }
            reader.endArray() ;
        } else if ( name == "faces" ) {
            reader.beginArray() ;
            while ( reader.hasNext() ) {
                std::deque<uint> indices ;
                reader.beginArray() ;
                while ( reader.hasNext() ) {
                    int idx = reader.nextInt() ;
                    indices.push_back(idx) ;
                }

                reader.endArray() ;
                MHX2Face f(indices) ;
                mesh.faces_.emplace_back(f) ;
            }
            reader.endArray() ;
        } else if ( name == "uv_faces" ) {
            reader.beginArray() ;
            while ( reader.hasNext() ) {
                UVFace f;

                reader.beginArray() ;
                while ( reader.hasNext() ) {
                    int idx = reader.nextInt() ;
                    f.indices_.push_back(idx) ;
                }
                reader.endArray() ;

                mesh.uv_faces_.emplace_back(std::move(f)) ;
            }
            reader.endArray() ;
        } else if ( name == "weights" ) {
            parseVertexGroups(mesh, reader) ;
        } else {
            reader.skipValue();
        }

    }

    mesh.computeNormals();
#if 0
    ofstream strm("/tmp/oo.obj") ;

    for( size_t i=0 ; i<mesh.vertices_.size() ; i++ ) {
        strm << "v " << mesh.vertices_[i].adjoint() << endl ;
    }

    for( size_t i=0 ; i<mesh.normals_.size() ; i++ ) {
        strm << "n " << mesh.normals_[i].adjoint() << endl ;
    }

    for( size_t i=0 ; i<mesh.faces_.size() ; i++ ) {
        const auto &f = mesh.faces_[i] ;
        strm << "f " << f.indices_[0] + 1 << ' ' << f.indices_[1] + 1<< ' ' << f.indices_[2] + 1<< ' ' << f.indices_[3] + 1<< endl ;
    }

    strm.close() ;
#endif
    reader.endObject() ;

    return true ;
}

bool Mhx2Importer::parseVertexGroups(MHX2Mesh &mesh, JSONReader &reader)
{
    reader.beginObject() ;
    while ( reader.hasNext() ) {
        string gname = reader.nextName() ;

        MHX2VertexGroup &vg = mesh.groups_[gname] ;

        reader.beginArray() ;
        while ( reader.hasNext() ) {
            reader.beginArray() ;
            uint idx = reader.nextInt() ;
            double weight = reader.nextDouble() ;

            vg.idxs_.push_back(idx) ;
            vg.weights_.push_back(weight) ;
            reader.endArray() ;
        }
        reader.endArray() ;
    }
    reader.endObject() ;

    return true ;
}

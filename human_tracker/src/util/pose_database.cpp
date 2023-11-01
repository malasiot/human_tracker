#include <htrac/util/pose_database.hpp>
#include <cvx/misc/json_reader.hpp>

//using namespace SQLite ;

#include <cassert>
#include <iostream>

using namespace std;
using namespace cvx ;


bool PoseDatabase::connect(const std::string &fpath) {
    try {
        open(fpath, SQLITE_OPEN_READWRITE);
        return true ;
    } catch ( SQLite3::Exception &e ) {
        cerr << e.what() << endl ;
        return false ;
    }
}

PoseDatabase::Cursor PoseDatabase::getReadCursor(bool selected) {
    return PoseDatabase::Cursor(*this, selected) ;
}

void PoseDatabase::getRecord(const string &id, std::map<string, Eigen::Matrix4f> &pose,
                             std::map<string, Eigen::Vector3f> &joints, bool &selected) {
    auto r = query("SELECT pose, joints, selected FROM poses WHERE id = ? LIMIT 1", id) ;
    if ( !r.next() ) {
        cerr << "Cannot found pose record with id: " << id << endl ;
    }
    pose = parsePose(r.get<string>(0)) ;
    joints = parseJoints(r.get<string>(1)) ;
    selected = r.get<bool>(2) ;
}

size_t PoseDatabase::count(bool selected) {
    string sql("SELECT COUNT(*) FROM poses") ;
    if ( selected )
        sql += " WHERE selected = 1" ;

    auto r = query(sql) ;
    if ( r.next() ) return r.get<size_t>(0) ;
    else return 0 ;
}

void PoseDatabase::updateSelected(const std::vector<string> &ids) {
    execute("UPDATE poses SET selected = 0") ;

    SQLite3::Transaction t(*this) ;

    SQLite3::Statement upd(*this, "UPDATE poses SET selected = 1 WHERE id = ?") ;
    for ( const auto &id: ids ) {
        upd.bind(1, id) ;
        upd.exec() ;
        upd.clear();
    }
    t.commit() ;
}

std::vector<string> PoseDatabase::getIds(bool selected) {
    vector<string> ids ;
    string sql("SELECT id FROM poses") ;
    if ( selected )
        sql += " WHERE selected = 1" ;

    auto r = query(sql) ;

    while ( r.next() ) {
        ids.push_back(r.get<string>(0)) ;
    }
    return ids ;
}

std::map<string, Eigen::Matrix4f> PoseDatabase::parsePose(const string &src)
{
    std::map<string, Eigen::Matrix4f> res ;

    stringstream strm(src) ;

    JSONReader r(strm) ;

    r.beginObject() ;

    while ( r.hasNext() ) {
        string name = r.nextName() ;

        Eigen::Matrix4f m ;

        r.beginArray() ;
        for( size_t row = 0 ; row < 4 ; row ++ ) {
            r.beginArray() ;
            for( size_t col = 0 ; col < 4 ; col ++ ) {
                m(row, col) = (float)r.nextDouble() ;
            }
           r.endArray() ;
        }
        r.endArray() ;

        res[name] = m ;
    }

    r.endObject() ;

    return res ;

}

std::map<string, Eigen::Vector3f> PoseDatabase::parseJoints(const string &src) {
    std::map<string, Eigen::Vector3f> res ;

    stringstream strm(src) ;

    JSONReader r(strm) ;

    r.beginObject() ;

    while ( r.hasNext() ) {
        string name = r.nextName() ;

        Eigen::Vector3f m ;

        r.beginArray() ;
        for( size_t c = 0 ; c < 3 ; c ++ )
            m[c] = (float)r.nextDouble() ;
        r.endArray() ;

        res[name] = m ;
    }

    r.endObject() ;

    return res ;
}


bool PoseDatabase::Cursor::next() {
    return res_->next() ;
}

string PoseDatabase::Cursor::id() const {
    return res_->get<std::string>(0) ;
}

std::map<string, Eigen::Matrix4f> PoseDatabase::Cursor::pose() const {
    return parsePose(res_->get<string>(1)) ;
}

std::map<string, Eigen::Vector3f> PoseDatabase::Cursor::joints() const {
    return parseJoints(res_->get<string>(2)) ;
}

bool PoseDatabase::Cursor::selected() const {
    return res_->get<bool>(3) ;
}


PoseDatabase::Cursor::Cursor(SQLite3::Connection &con, bool selected) {
    assert(con) ;
    SQLite3::Query q(con, (selected) ? "SELECT * FROM poses WHERE SELECTED = 1" : "SELECT * FROM poses") ;
    res_.reset(new SQLite3::QueryResult(q.exec())) ;
}

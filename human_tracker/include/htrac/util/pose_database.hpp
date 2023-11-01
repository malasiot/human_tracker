#pragma once

#include <sqlite3.h>
#include <memory>

#include <cvx/misc/database.hpp>

#include <Eigen/Geometry>

class PoseDatabase: protected cvx::SQLite3::Connection {
public:

    bool connect(const std::string &fpath) ;

    class Cursor {
    public:
        bool  next() ;

        std::string id() const ;
        std::map<std::string, Eigen::Matrix4f> pose() const ;
        std::map<std::string, Eigen::Vector3f> joints() const ;
        bool selected() const ;

    private:
        friend class PoseDatabase ;
        Cursor(cvx::SQLite3::Connection &con, bool) ;



        std::unique_ptr<cvx::SQLite3::QueryResult> res_ ;
    };

    Cursor getReadCursor(bool selected = false) ;

    void getRecord(const std::string &id, std::map<std::string, Eigen::Matrix4f> &pose, std::map<std::string, Eigen::Vector3f> &joints,
                   bool &selected);
    void selectRecord(const std::string &id) ;

    size_t count(bool selected = false) ;

    void updateSelected(const std::vector<std::string> &ids) ;

    std::vector<std::string> getIds(bool selected = false) ;

private:

    static std::map<std::string, Eigen::Matrix4f> parsePose(const std::string &) ;
    static std::map<std::string, Eigen::Vector3f> parseJoints(const std::string &) ;


};

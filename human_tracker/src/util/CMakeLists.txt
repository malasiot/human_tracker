add_library(htrac_util SHARED
	pose_database.cpp
	mhx2_importer.cpp
	io_util.cpp
	pcl_util.cpp
        skeleton.cpp
        mesh.cpp
        depth_provider.cpp
)

target_link_libraries(htrac_util assimp)

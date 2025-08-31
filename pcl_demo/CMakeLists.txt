cmake_minimum_required(VERSION 2.8)

project(pcd_write)

# 设置构建类型和 C++ 标准
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# 找到必要的 PCL 和 Boost 库
find_package(PCL REQUIRED COMPONENTS visualization common io filters features keypoints)
find_package(Boost REQUIRED COMPONENTS thread)


# 设置 include 目录
include_directories(
    ${PCL_INCLUDE_DIRS} 
    ${Boost_INCLUDE_DIRS} 
    ${PROJECT_SOURCE_DIR}/ivox3d
    ikd-tree
)

# 设置链接目录
link_directories(${PCL_LIBRARY_DIRS} ${Boost_LIBRARY_DIRS})

# 添加定义
add_definitions(${PCL_DEFINITIONS})

find_package(Threads REQUIRED)

# PCD 读取学习
#add_executable(pcd_write cloud_test/pcd_write.cpp)
#target_link_libraries(pcd_write PRIVATE ${PCL_LIBRARIES})

# KD-Tree 搜索学习
#add_executable(kd_tree_demo cloud_test/kd-tree_demo.cpp)
#target_link_libraries(kd_tree_demo PRIVATE ${PCL_LIBRARIES})

# KD-Tree IKD-Tree IVox 进行 ICP-SVD
#add_executable(s_t_demo s_t_demo.cpp ikd-tree/ikd_Tree.cpp)
#target_link_libraries(s_t_demo PRIVATE ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} glog)

# IKD-Tree 学习
#add_executable(ikd_demo cloud_test/ikd_demo.cpp ikd-tree/ikd_Tree.cpp)
#target_link_libraries(ikd_demo PRIVATE ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

# 测试代码
#add_executable(test test_cpp/test.cpp ikd-tree/ikd_Tree.cpp)
#target_link_libraries(test PRIVATE ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

# IVox 点云搜索学习
#add_executable(ivox_demo cloud_test/ivox_demo.cpp)
#target_link_libraries(ivox_demo PRIVATE ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} glog)

# IVox 点云存储学习
#add_executable(ivoxz_test cloud_test/ivoxz_test.cpp)
#target_link_libraries(ivoxz_test PRIVATE ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} glog)

# 景老师原始代码测试 KD-Tree 时间
#add_executable(raw cloud_test/raw.cpp)
#target_link_libraries(raw PRIVATE ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

# 测试 Rabbit 点云
#add_executable(test_rabbit test_cpp/test_rabbit.cpp)
#target_link_libraries(test_rabbit PRIVATE ${PCL_LIBRARIES})

# 测试 KD-Tree 和 ICP
#add_executable(pcl_kd_icp test_cpp/pcl_kd_icp.cpp)
#target_link_libraries(pcl_kd_icp PRIVATE ${PCL_LIBRARIES})

# 测试 FPFH
#add_executable(FPFH_demo test_cpp/FPFH_demo.cpp)
#target_link_libraries(FPFH_demo PRIVATE ${PCL_LIBRARIES})

# 测试 ISS
#add_executable(iss_demo test_cpp/iss_demo.cpp)
#target_link_libraries(iss_demo PRIVATE ${PCL_LIBRARIES}   Boost::thread)

#查看点云分辨率
#add_executable(cloud_resoulation test_cpp/cloud_resoulation.cpp)
#target_link_libraries(cloud_resoulation PRIVATE ${PCL_LIBRARIES})

#软著 人机交互
#add_executable(ruanzhu ruanzhu.cpp ikd-tree/ikd_Tree.cpp)
#target_link_libraries(ruanzhu PRIVATE ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} glog)

#ivox icp
#add_executable(ivox ivox.cpp ikd-tree/ikd_Tree.cpp)
#target_link_libraries(ivox PRIVATE ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} glog)

#ikd icp
#add_executable(ikd ikd.cpp ikd-tree/ikd_Tree.cpp)
#target_link_libraries(ikd PRIVATE ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} glog)

#kd-tree ikd-tree ivox 三种已经实现
add_executable(finish finish.cpp ikd-tree/ikd_Tree.cpp)
target_link_libraries(finish PRIVATE ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} glog)

# 其他可执行文件可以在此处添加...

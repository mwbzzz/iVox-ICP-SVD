#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/search/kdtree.h>

int main(int argc, char **argv)
{
    // 1. 读取点云数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("../materials/rabbit.pcd", *cloud) == -1)
    {
        PCL_ERROR("Couldn't read input PCD file! \n");
        return (-1);
    }
    std::cout << "Loaded cloud with " << cloud->size() << " points." << std::endl;

    // 2. 计算法线
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    normal_estimation.setInputCloud(cloud);
    normal_estimation.setSearchMethod(tree);
    normal_estimation.setRadiusSearch(0.02); // 设置搜索半径
    normal_estimation.compute(*normals);     // 计算法线
    std::cout << "Calculated " << normals->size() << " normals." << std::endl;

    // 3. 计算 ISS 特征点
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ, pcl::Normal> iss_detector;
    iss_detector.setInputCloud(cloud);
    iss_detector.setSearchMethod(tree);
    iss_detector.setSalientRadius(0.01);
    iss_detector.setNonMaxRadius(0.02);
    iss_detector.setMinNeighbors(5);
    iss_detector.setThreshold21(0.975);
    iss_detector.setThreshold32(0.975);
    iss_detector.compute(*keypoints);
    std::cout << "Detected " << keypoints->size() << " keypoints." << std::endl;

    // 4. 为关键点重新计算法线
    pcl::PointCloud<pcl::Normal>::Ptr keypoint_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> keypoint_normal_estimation;
    keypoint_normal_estimation.setInputCloud(keypoints);
    keypoint_normal_estimation.setSearchMethod(tree);
    keypoint_normal_estimation.setRadiusSearch(0.02); // 设置搜索半径
    keypoint_normal_estimation.compute(*keypoint_normals);
    std::cout << "Recalculated normals for keypoints: " << keypoint_normals->size() << std::endl;

    // 5. 计算 FPFH 特征
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_estimation;
    fpfh_estimation.setInputCloud(keypoints);
    fpfh_estimation.setInputNormals(keypoint_normals); // 使用关键点对应的法线
    fpfh_estimation.setSearchMethod(tree);
    fpfh_estimation.setRadiusSearch(0.05);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_features(new pcl::PointCloud<pcl::FPFHSignature33>);
    fpfh_estimation.compute(*fpfh_features);
    std::cout << "Computed FPFH features for " << fpfh_features->size() << " points." << std::endl;

    // 6. 可视化
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    // 添加点云
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color(cloud, 255, 255, 255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_color, "original cloud");

    // 添加关键点（红色）
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints_color(keypoints, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(keypoints, keypoints_color, "keypoints");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "keypoints");

    // 显示 FPFH 特征值（简单输出前几个特征）
    for (size_t i = 0; i < std::min<size_t>(fpfh_features->size(), 5); ++i)
    {
        std::cout << "FPFH for keypoint " << i << ": ";
        for (int j = 0; j < 33; ++j)
        {
            std::cout << fpfh_features->points[i].histogram[j] << " ";
        }
        std::cout << std::endl;
    }

    // 启动可视化
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
    }

    return 0;
}

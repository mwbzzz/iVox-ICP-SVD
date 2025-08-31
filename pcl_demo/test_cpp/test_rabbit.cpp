
#include <iostream>
#include <ctime>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include "pcl/point_cloud.h"
#include <boost/thread/thread.hpp>
#include <boost/chrono.hpp>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include "../ivox3d/ivox3d.h" // 确保路径正确
#include "../ivox3d/ivox3d_node.hpp"

using namespace std;

int main()
{

  // 读取点云文件
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>("../materials/rabbit.pcd", *cloud_source) == -1)
  {
    PCL_ERROR("Couldn't read pcd file! \n");
    return (-1);
  }
  pcl::PointCloud<pcl::PointXYZ>::Ptr cl_t_filt(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cl_t_st_filt(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> stat_filter;
  pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
  voxel_filter.setInputCloud(cloud_source);
  voxel_filter.setLeafSize(0.05f, 0.05f, 0.05f);
  voxel_filter.filter(*cl_t_filt);
  stat_filter.setInputCloud(cl_t_filt);
  stat_filter.setMeanK(50);
  stat_filter.setStddevMulThresh(1.0);
  stat_filter.setNegative(false);
  stat_filter.filter(*cl_t_st_filt);
  auto data = cl_t_st_filt->size();
  cout << "The size of pcd : " << data << endl;
}

#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <iostream>

double computeCloudResolution(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
  double resolution = 0.0;
  int points = 0;

  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(cloud);

  std::vector<int> indices(2);
  std::vector<float> squared_distances(2);

  for (std::size_t i = 0; i < cloud->size(); ++i)
  {
    if (kdtree.nearestKSearch(cloud->at(i), 2, indices, squared_distances) == 2)
    {
      resolution += std::sqrt(squared_distances[1]); // 跳过自身点
      ++points;
    }
  }

  if (points > 0)
  {
    resolution /= points;
  }

  return resolution;
}

int main()
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile("../materials/xpd_cloud_filtered.pcd", *cloud) == -1)
  {
    PCL_ERROR("Couldn't read the file\n");
    return -1;
  }

  double resolution = computeCloudResolution(cloud);
  std::cout << "Point Cloud Resolution: " << resolution << " meters" << std::endl;

  return 0;
}

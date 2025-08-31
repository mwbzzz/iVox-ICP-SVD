#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

int main() {
  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.width = 50;
  cloud.height = 1;
  cloud.is_dense = false;
  cloud.points.resize(cloud.width * cloud.height);

  for (size_t i = 0; i < cloud.points.size(); i++) {
    cloud.points[i].x = 1024.0f * rand() / RAND_MAX;
    cloud.points[i].y = 1024.0f * rand() / RAND_MAX;
    cloud.points[i].z = 1024.0f * rand() / RAND_MAX;
  }

  pcl::io::savePCDFileASCII("test_pcd.pcd", cloud);
  std::cout << "Saved " << cloud.points.size() << " data points to test_pcd.pcd." << std::endl;

  return 0;
}

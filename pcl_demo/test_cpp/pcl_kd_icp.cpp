#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <Eigen/Dense>
#include <thread>
#include <chrono>

using namespace pcl;
using namespace std;

using PointType = PointXYZ;

void showPointClouds(const PointCloud<PointType>::Ptr &source,
                     const PointCloud<PointType>::Ptr &target,
                     const PointCloud<PointType>::Ptr &aligned)
{
  visualization::PCLVisualizer viewer("3D Viewer");
  viewer.setBackgroundColor(255, 255, 255);

  visualization::PointCloudColorHandlerCustom<PointType> sourceHandler(source, 0, 255, 0);
  visualization::PointCloudColorHandlerCustom<PointType> targetHandler(target, 255, 0, 0);
  visualization::PointCloudColorHandlerCustom<PointType> alignedHandler(aligned, 0, 0, 255);

  viewer.addPointCloud(source, sourceHandler, "source_cloud");
  viewer.addPointCloud(target, targetHandler, "target_cloud");
  viewer.addPointCloud(aligned, alignedHandler, "aligned_cloud");

  viewer.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 3, "source_cloud");
  viewer.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 3, "target_cloud");
  viewer.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 3, "aligned_cloud");

  viewer.addCoordinateSystem(1.0);
  viewer.initCameraParameters();

  while (!viewer.wasStopped())
  {
    viewer.spinOnce(100);
    this_thread::sleep_for(chrono::milliseconds(100));
  }
}

int main()
{
  PointCloud<PointType>::Ptr original_cloud(new PointCloud<PointType>);
  string source_filename = "../materials/xpd_cloud_filtered.pcd";

  if (io::loadPCDFile<PointType>(source_filename, *original_cloud) == -1)
  {
    cerr << "Error: Could not read source point cloud file!" << endl;
    return -1;
  }
  cout << "Loaded original point cloud with " << original_cloud->size() << " points." << endl;

  // Downsample the original cloud to reduce noise and improve ICP
  VoxelGrid<PointType> vox;
  PointCloud<PointType>::Ptr original_cloud_down(new PointCloud<PointType>);
  vox.setInputCloud(original_cloud);
  vox.setLeafSize(0.005f, 0.005f, 0.005f); // Adjust leaf size based on dataset
  vox.filter(*original_cloud_down);
  cout << "Downsampled original point cloud to " << original_cloud_down->size() << " points." << endl;

  // Remove statistical outliers
  StatisticalOutlierRemoval<PointType> sor;
  sor.setInputCloud(original_cloud_down);
  sor.setMeanK(50);
  sor.setStddevMulThresh(1.0);
  sor.filter(*original_cloud_down);

  // Create the target cloud with a transformation
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  transform(0, 3) = 1.0;
  transform(1, 3) = 0.5;
  transform(2, 3) = 0.5;

  PointCloud<PointType>::Ptr target_cloud(new PointCloud<PointType>);
  pcl::transformPointCloud(*original_cloud_down, *target_cloud, transform);

  // Initialize ICP with relaxed parameters
  IterativeClosestPoint<PointType, PointType> icp;
  icp.setInputSource(original_cloud_down);
  icp.setInputTarget(target_cloud);

  // Adjust parameters to allow more correspondences
  icp.setMaxCorrespondenceDistance(1.0); // Increase this threshold
  icp.setMaximumIterations(500);         // Increase iterations
  icp.setTransformationEpsilon(1e-10);
  icp.setEuclideanFitnessEpsilon(1e-8);

  // Use reciprocal correspondences to improve robustness
  icp.setUseReciprocalCorrespondences(true);

  PointCloud<PointType>::Ptr aligned_cloud(new PointCloud<PointType>);

  try
  {
    icp.align(*aligned_cloud);
    if (!icp.hasConverged())
    {
      cerr << "Error: ICP did not converge!" << endl;
      return -1;
    }

    // Save the aligned cloud
    string output_filename = "aligned.pcd";
    if (io::savePCDFile(output_filename, *aligned_cloud) == -1)
    {
      cerr << "Error: Could not save aligned point cloud!" << endl;
      return -1;
    }
    cout << "Aligned point cloud saved to '" << output_filename << "'." << endl;

    // Print transformation matrix and fitness score
    cout << "Final transformation:\n"
         << icp.getFinalTransformation() << endl;
    cout << "Fitness score: " << icp.getFitnessScore() << endl;
  }
  catch (const exception &e)
  {
    cerr << "Error during ICP alignment: " << e.what() << endl;
    return -1;
  }

  // Visualize the results
  showPointClouds(original_cloud_down, target_cloud, aligned_cloud);

  return 0;
}
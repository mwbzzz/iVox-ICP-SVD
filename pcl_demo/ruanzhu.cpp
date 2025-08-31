#include <iostream>
#include <ctime>
#include <chrono>
#include <string>
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
#include "ikd-tree/ikd_Tree.h"
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include "ivox3d/ivox3d.h" // 确保路径正确
#include "ivox3d/ivox3d_node.hpp"

using namespace faster_lio;
using PointType = pcl::PointXYZ;
using IVoxType = IVox<3, IVoxNodeType::DEFAULT, PointType>;

using PointVector = KD_TREE<PointType>::PointVector;

template class KD_TREE<pcl::PointXYZ>;

double measureKdTreeSearchTime(pcl::KdTreeFLANN<pcl::PointXYZ> &kd_tree,
                               pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_S_Trans,
                               float max_correspond_distance,
                               std::vector<int> &match_indexes);

template <typename PointType>
double measureIkdTreeSearchTime(KD_TREE<PointType> &ikd_tree,
                                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_S_Trans,
                                float max_correspond_distance,
                                std::vector<int> &match_indexes);

template <typename PointType>
double measureIVoxSearchTime(IVoxType &ivox,
                             boost::shared_ptr<pcl::PointCloud<PointType>> cloud,
                             float max_correspond_distance,
                             std::vector<int> &match_indexes);

template <int dim, IVoxNodeType node_type, typename PointType>
size_t IVox<dim, node_type, PointType>::NumPoints() const
{
  size_t total_points = 0;
  for (const auto &grid : grids_cache_)
  {
    total_points += grid.second.Size();
  }
  return total_points;
}

enum class PointCloudStructure
{
  KD_TREE,
  IKD_TREE,
  IVOX
};

template <typename PointType>
double searchPointCloud(PointCloudStructure structure,
                        void *tree,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_S_Trans,
                        float max_correspond_distance,
                        std::vector<int> &match_indexes)
{
  auto start = std::chrono::high_resolution_clock::now();

  switch (structure)
  {
  case PointCloudStructure::KD_TREE:
  {
    auto *kd_tree = static_cast<pcl::KdTreeFLANN<pcl::PointXYZ> *>(tree);
    for (size_t i = 0; i < cloud_S_Trans->size(); ++i)
    {
      std::vector<float> k_distances(1);
      std::vector<int> k_indexs(1);
      kd_tree->nearestKSearch(cloud_S_Trans->at(i), 1, k_indexs, k_distances);
      if (k_distances[0] < max_correspond_distance)
      {
        match_indexes.emplace_back(k_indexs[0]);
      }
    }
    break;
  }
  case PointCloudStructure::IKD_TREE:
  {
    auto *ikd_tree = static_cast<KD_TREE<PointType> *>(tree);
    for (size_t i = 0; i < cloud_S_Trans->size(); ++i)
    {
      PointVector nearest_points;
      std::vector<float> point_distances;
      ikd_tree->Nearest_Search(cloud_S_Trans->at(i), 1, nearest_points, point_distances, max_correspond_distance);
      if (!point_distances.empty() && point_distances[0] < max_correspond_distance)
      {
        match_indexes.emplace_back(i);
      }
    }
    break;
  }
  case PointCloudStructure::IVOX:
  {
    auto *ivox = static_cast<IVoxType *>(tree);
    for (size_t i = 0; i < cloud_S_Trans->size(); ++i)
    {
      std::vector<PointType, Eigen::aligned_allocator<PointType>> closest_points;
      bool found = ivox->GetClosestPoint(cloud_S_Trans->at(i), closest_points, 1, max_correspond_distance);
      if (found && !closest_points.empty())
      {
        match_indexes.emplace_back(i);
      }
    }
    break;
  }
  default:
    throw std::invalid_argument("Unsupported point cloud structure");
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  return elapsed.count();
}

void printMatrix(const Eigen::Matrix4f &T)
{
  printf("Transform matrix :\n");
  printf("    | %6.3f %6.3f %6.3f %6.3f | \n", T(0, 0), T(0, 1), T(0, 2), T(0, 3));
  printf("T = | %6.3f %6.3f %6.3f %6.3f | \n", T(1, 0), T(1, 1), T(1, 2), T(1, 3));
  printf("    | %6.3f %6.3f %6.3f %6.3f | \n", T(2, 0), T(2, 1), T(2, 2), T(2, 3));
  printf("    | %6.3f %6.3f %6.3f %6.3f | \n", T(3, 0), T(3, 1), T(3, 2), T(3, 3));
}

double EvaluateCLoudsDist_kd_tree(pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree_cl_s,
                                  pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_s_reg)
{
  double mean_dist = 0;
  for (auto i = 0; i < cl_s_reg->size(); ++i)
  {
    std::vector<float> k_distances(2);
    std::vector<int> k_indexs(2);
    kdtree_cl_s.nearestKSearch(cl_s_reg->at(i), 1, k_indexs, k_distances);
    mean_dist += k_distances[0];
  }
  mean_dist = mean_dist / cl_s_reg->size();

  return mean_dist;
}

double EvaluateCLoudsDist_ikd_tree(KD_TREE<pcl::PointXYZ> &ikdTree_cl_s,
                                   pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_s_reg)
{
  double mean_dist = 0;

  return mean_dist;
}

void CreateTargetGT(pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_s,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_t,
                    Eigen::Matrix4f &T_ts_gt)
{
  Eigen::Matrix4f T_ts = Eigen::Matrix4f::Identity();

  double theta = M_PI / 8;
  T_ts(0, 0) = cos(theta);
  T_ts(0, 1) = -sin(theta);
  T_ts(1, 0) = sin(theta);
  T_ts(1, 1) = cos(theta);

  T_ts(2, 3) = 0.4;

  pcl::transformPointCloud(*cl_s, *cl_t, T_ts);
  T_ts_gt = T_ts;
}

void ICP_PCL(pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_s,
             pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_t,
             pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_final,
             Eigen::Matrix4f &T_ts,
             double &time_cost)
{
  clock_t startT = clock();
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(cl_s);
  icp.setInputTarget(cl_t);
  icp.setMaxCorrespondenceDistance(10);
  icp.setMaximumIterations(50);
  icp.setTransformationEpsilon(1e-10);
  icp.setEuclideanFitnessEpsilon(0.001);
  icp.align(*cl_final);
  T_ts = icp.getFinalTransformation();
  clock_t endT = clock();
  time_cost = (double)(endT - startT) / CLOCKS_PER_SEC;
}

void NDT_PCL(pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_s,
             pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_t,
             pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_final,
             Eigen::Matrix4f &T_ts,
             double &time_cost)
{
  clock_t startT = clock();
  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
  ndt.setInputSource(cl_s);
  ndt.setInputTarget(cl_t);
  ndt.setStepSize(0.1);
  ndt.setResolution(1.0);
  ndt.setMaximumIterations(50);
  ndt.setTransformationEpsilon(1e-10);
  ndt.align(*cl_final);
  T_ts = ndt.getFinalTransformation();
  clock_t endT = clock();
  time_cost = (double)(endT - startT) / CLOCKS_PER_SEC;
}

void ICP_MANUAL(pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_s,
                pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_t,
                pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_final,
                Eigen::Matrix4f &T_ts,
                double &time_cost)
{
  KD_TREE<PointType>::Ptr kdtree_ptr(new KD_TREE<PointType>(0.3, 0.6, 0.2));
  KD_TREE<PointType> &ikd_Tree = *kdtree_ptr;
  ikd_Tree.Build((*cl_t).points);
  std::cout << "The data of ikd_Tree.size : " << ikd_Tree.size() << std::endl;

  clock_t startT = clock();
  int nICP_Step = 50;
  float max_correspond_distance_ = 1;
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();

  double total_ikd_tree_time = 0.0;

  for (int i = 0; i < nICP_Step; ++i)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_S_Trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cl_s, *cloud_S_Trans, transform);

    std::vector<int> match_indexs;
    double ikd_tree_time = searchPointCloud<PointType>(PointCloudStructure::IKD_TREE, kdtree_ptr.get(), cloud_S_Trans, max_correspond_distance_, match_indexs);
    total_ikd_tree_time += ikd_tree_time;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_match_from_S_trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud_S_Trans, match_indexs, *cloud_match_from_S_trans);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_match_from_T(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cl_t, match_indexs, *cloud_match_from_T);

    Eigen::Vector4f mu_S_temp, mu_T_temp;
    pcl::compute3DCentroid(*cloud_match_from_S_trans, mu_S_temp);
    pcl::compute3DCentroid(*cloud_match_from_T, mu_T_temp);
    Eigen::Vector3f mu_S(mu_S_temp[0], mu_S_temp[1], mu_S_temp[2]);
    Eigen::Vector3f mu_T(mu_T_temp[0], mu_T_temp[1], mu_T_temp[2]);

    Eigen::Matrix3f H_icp = Eigen::Matrix3f::Identity();
    for (auto i = 0; i < match_indexs.size(); ++i)
    {
      Eigen::Vector3f s_ci(cloud_match_from_S_trans->at(i).x - mu_S[0],
                           cloud_match_from_S_trans->at(i).y - mu_S[1],
                           cloud_match_from_S_trans->at(i).z - mu_S[2]);
      Eigen::Vector3f t_ci(cloud_match_from_T->at(i).x - mu_T[0],
                           cloud_match_from_T->at(i).y - mu_T[1],
                           cloud_match_from_T->at(i).z - mu_T[2]);
      Eigen::Matrix3f H_temp = s_ci * t_ci.transpose();
      H_icp += H_temp;
    }

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(H_icp, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3f R_ts = svd.matrixU() * (svd.matrixV().transpose());

    Eigen::Vector3f t_ts = mu_T - R_ts * mu_S;
    Eigen::Quaternionf q_ts(R_ts);
    transform.block<3, 3>(0, 0) *= q_ts.normalized().toRotationMatrix().inverse();
    transform.block<3, 1>(0, 3) += t_ts;
  }
  T_ts = transform;
  pcl::transformPointCloud(*cl_s, *cl_final, T_ts);
  clock_t endT = clock();
  time_cost = (double)(endT - startT) / CLOCKS_PER_SEC;
  double average_ikd_tree_time = total_ikd_tree_time / nICP_Step;
  std::cout << "Total ikd-tree search time: " << total_ikd_tree_time << " seconds." << std::endl;
}

double measureKdTreeSearchTime(pcl::KdTreeFLANN<pcl::PointXYZ> &kd_tree,
                               pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_S_Trans,
                               float max_correspond_distance,
                               std::vector<int> &match_indexes)
{
  clock_t start = clock();
  for (auto i = 0; i < cloud_S_Trans->size(); ++i)
  {
    std::vector<float> k_distances(1);
    std::vector<int> k_indexs(1);
    kd_tree.nearestKSearch(cloud_S_Trans->at(i), 1, k_indexs, k_distances);
    if (k_distances[0] < max_correspond_distance)
    {
      match_indexes.emplace_back(k_indexs[0]);
    }
  }
  clock_t end = clock();
  return (double)(end - start) / CLOCKS_PER_SEC;
}

template <typename PointType>
double measureIkdTreeSearchTime(KD_TREE<PointType> &ikd_tree,
                                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_S_Trans,
                                float max_correspond_distance,
                                std::vector<int> &match_indexes)
{
  clock_t start = clock();
  for (size_t i = 0; i < cloud_S_Trans->size(); ++i)
  {
    PointVector nearest_points;
    std::vector<float> point_distances;
    ikd_tree.Nearest_Search(cloud_S_Trans->at(i), 1, nearest_points, point_distances, max_correspond_distance);
    if (!point_distances.empty() && point_distances[0] < max_correspond_distance)
    {
      match_indexes.emplace_back(i);
    }
  }
  clock_t end = clock();
  return static_cast<double>(end - start) / CLOCKS_PER_SEC;
}

template <typename PointType>
double measureIVoxSearchTime(IVoxType &ivox,
                             boost::shared_ptr<pcl::PointCloud<PointType>> cloud,
                             float max_correspond_distance,
                             std::vector<int> &match_indexes)
{
  clock_t start = clock();
  for (size_t i = 0; i < cloud->size(); ++i)
  {
    std::vector<PointType, Eigen::aligned_allocator<PointType>> closest_points;
    bool found = ivox.GetClosestPoint(cloud->at(i), closest_points, 1, max_correspond_distance);
    if (found && !closest_points.empty())
    {
      match_indexes.emplace_back(i);
    }
  }
  clock_t end = clock();
  return static_cast<double>(end - start) / CLOCKS_PER_SEC;
}

void ShowMatchResultsIndividual(pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_s,
                                pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_t,
                                pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_reg)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Individual Match Result"));
  viewer->setBackgroundColor(255, 255, 255);

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cl_s_h(cl_s, 100, 100, 100);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cl_t_h(cl_t, 0, 250, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cl_reg_h(cl_reg, 0, 0, 222);

  viewer->addPointCloud(cl_s, cl_s_h, "source_cloud");
  viewer->addPointCloud(cl_t, cl_t_h, "target_cloud");
  viewer->addPointCloud(cl_reg, cl_reg_h, "registered_cloud");
  // viewer->addCoordinateSystem(1.0);

  while (!viewer->wasStopped())
  {
    viewer->spinOnce(200);
    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  }
}

void ShowMatchResultsCombined(pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_s,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_t,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_reg_pcl_icp,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_reg_pcl_ndt,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_reg_pcl_man)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Combined Match Results"));
  int v1(0), v2(0), v3(0), v4(0);
  viewer->createViewPort(0.0, 0, 0.25, 1, v1);
  viewer->createViewPort(0.25, 0, 0.5, 1, v2);
  viewer->createViewPort(0.5, 0, 0.75, 1, v3);
  viewer->createViewPort(0.75, 0, 1.0, 1, v4);

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cl_s_h(cl_s, 100, 100, 100);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cl_t_h(cl_t, 0, 250, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cl_reg_pcl_icp_h(cl_reg_pcl_icp, 0, 0, 222);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cl_reg_pcl_ndt_h(cl_reg_pcl_ndt, 0, 0, 222);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cl_reg_pcl_man_h(cl_reg_pcl_man, 0, 0, 222);

  viewer->addPointCloud(cl_s, cl_s_h, "s_v1", v1);
  viewer->addPointCloud(cl_t, cl_t_h, "t_v1", v1);
  viewer->addPointCloud(cl_s, cl_s_h, "s_v2", v2);
  viewer->addPointCloud(cl_t, cl_t_h, "t_v2", v2);
  viewer->addPointCloud(cl_reg_pcl_icp, cl_reg_pcl_icp_h, "pcl_icp_v2", v2);
  viewer->addPointCloud(cl_s, cl_s_h, "s_v3", v3);
  viewer->addPointCloud(cl_t, cl_t_h, "t_v3", v3);
  viewer->addPointCloud(cl_reg_pcl_ndt, cl_reg_pcl_ndt_h, "pcl_ndt_v3", v3);
  viewer->addPointCloud(cl_s, cl_s_h, "s_v4", v4);
  viewer->addPointCloud(cl_t, cl_t_h, "t_v4", v4);
  viewer->addPointCloud(cl_reg_pcl_man, cl_reg_pcl_man_h, "icp_man_v4", v4);

  viewer->setBackgroundColor(255, 255, 255, v1);
  viewer->setBackgroundColor(255, 255, 255, v2);
  viewer->setBackgroundColor(255, 255, 255, v3);
  viewer->setBackgroundColor(255, 255, 255, v4);

  viewer->addCoordinateSystem(1.0);

  while (!viewer->wasStopped())
  {
    viewer->spinOnce(200);
    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  }
}

void ShowSourceAndTarget(pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_s,
                         pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_t)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Source and Target Point Clouds"));
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cl_s_h(cl_s, 100, 100, 100);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cl_t_h(cl_t, 0, 250, 0);

  viewer->addPointCloud(cl_s, cl_s_h, "source_cloud");
  viewer->addPointCloud(cl_t, cl_t_h, "target_cloud");
  viewer->setBackgroundColor(255, 255, 255);
  viewer->addCoordinateSystem(1.0);

  while (!viewer->wasStopped())
  {
    viewer->spinOnce(200);
    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  }
}

void ShowPointCloudSizes(pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_s,
                         pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_t)
{
  std::cout << "Source Point Cloud Size: " << cl_s->size() << "\n";
  std::cout << "Target Point Cloud Size: " << cl_t->size() << "\n";
}

void PrintTransformationMatrix(const Eigen::Matrix4f &T)
{
  printMatrix(T);
}

void PrintRMSE(double rmse)
{
  std::cout << "Root Mean Square Error (RMSE): " << rmse << "\n";
}

int main(int argc, char **argv)
{
  KD_TREE<PointType>::Ptr kdtree_ptr(new KD_TREE<PointType>(0.3, 0.6, 0.2));
  KD_TREE<PointType> &ikd_Tree = *kdtree_ptr;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>("../materials/rabbit.pcd", *cloud_source) == -1)
  {
    PCL_ERROR("Couldn't read pcd file! \n");
    return (-1);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target(new pcl::PointCloud<pcl::PointXYZ>);
  Eigen::Matrix4f T_ts_gt = Eigen::Matrix4f::Identity();
  CreateTargetGT(cloud_source, cloud_target, T_ts_gt);

  clock_t ikd_build_time_start, ikd_build_time_end;
  ikd_build_time_start = clock();
  ikd_Tree.Build((*cloud_target).points);
  auto da = ikd_Tree.size();
  std::cout << "---------------" << std::endl;
  std::cout << "ikd data size :" << da << std::endl;
  ikd_build_time_end = clock();
  double build_ikd_tree_times = (double)(ikd_build_time_end - ikd_build_time_start) / CLOCKS_PER_SEC;
  std::cout << "build_ikd_tree_times: " << build_ikd_tree_times << "s" << std::endl;

  pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree_target;
  kd_tree_target.setInputCloud(cloud_target);

  double t_pcl_icp = 0;
  double t_pcl_ndt = 0;
  double t_icp_man = 0;
  Eigen::Matrix4f T_ts_pcl_icp = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f T_ts_pcl_ndt = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f T_ts_man = Eigen::Matrix4f::Identity();
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_reg_pcl_icp(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_reg_pcl_ndt(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_reg_man(new pcl::PointCloud<pcl::PointXYZ>);

  ICP_PCL(cloud_source, cloud_target, cloud_reg_pcl_icp, T_ts_pcl_icp, t_pcl_icp);
  NDT_PCL(cloud_source, cloud_target, cloud_reg_pcl_ndt, T_ts_pcl_ndt, t_pcl_ndt);
  ICP_MANUAL(cloud_source, cloud_target, cloud_reg_man, T_ts_man, t_icp_man);

  clock_t pcl_icp_start, pcl_icp_end, pcl_ndt_start, pcl_ndt_end, icp_man_start, icp_man_end;
  pcl_icp_start = clock();
  double dis_pcl_icp = EvaluateCLoudsDist_kd_tree(kd_tree_target, cloud_reg_pcl_icp);
  pcl_icp_end = clock();
  double pcl_icp_kd_tree_time = (double)(pcl_icp_end - pcl_icp_start) / CLOCKS_PER_SEC;

  pcl_ndt_start = clock();
  double dis_pcl_ndt = EvaluateCLoudsDist_kd_tree(kd_tree_target, cloud_reg_pcl_ndt);
  pcl_ndt_end = clock();
  double pcl_ndt_kd_tree_time = (double)(pcl_ndt_end - pcl_ndt_start) / CLOCKS_PER_SEC;

  icp_man_start = clock();
  double dis_icp_man = EvaluateCLoudsDist_kd_tree(kd_tree_target, cloud_reg_man);
  icp_man_end = clock();
  double icp_man_kd_tree_time = (double)(icp_man_end - icp_man_start) / CLOCKS_PER_SEC;

  PrintTransformationMatrix(T_ts_gt);
  PrintRMSE(dis_pcl_icp);

  ShowMatchResultsCombined(cloud_source, cloud_target, cloud_reg_pcl_icp, cloud_reg_pcl_ndt, cloud_reg_man);

  ShowPointCloudSizes(cloud_source, cloud_target);

  int choice;
  while (true)
  {
    std::cout << "\n=== Main Menu ===" << std::endl;
    std::cout << "0. Display Source and Target Point Clouds" << std::endl;
    std::cout << "1. Show Individual Match Results" << std::endl;
    std::cout << "2. Show Combined Match Results" << std::endl;
    std::cout << "3. Show Point Cloud Sizes" << std::endl;
    std::cout << "4. Show Transformation Matrix and RMSE" << std::endl;
    std::cout << "5. Exit" << std::endl;
    std::cout << "Enter your choice: ";
    std::cin >> choice;

    switch (choice)
    {
    case 0:
      ShowSourceAndTarget(cloud_source, cloud_target);
      break;
    case 1:
      ShowMatchResultsIndividual(cloud_source, cloud_target, cloud_reg_pcl_icp);
      break;
    case 2:
      ShowMatchResultsCombined(cloud_source, cloud_target, cloud_reg_pcl_icp, cloud_reg_pcl_ndt, cloud_reg_man);
      break;
    case 3:
      ShowPointCloudSizes(cloud_source, cloud_target);
      break;
    case 4:
      PrintTransformationMatrix(T_ts_gt);
      PrintRMSE(dis_pcl_icp);
      break;
    case 5:
      std::cout << "Exiting the application." << std::endl;
      return 0;
    default:
      std::cout << "Invalid choice. Please try again." << std::endl;
    }
  }

  return 0;
}
// Time: 2021/3/18
// Author: JQF
// Content: Turtorial of ICP-SVD

#include <iostream>
#include <ctime>
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
#include "ivox3d/ivox3d.h" // 确保路径正确
#include "ivox3d/ivox3d_node.hpp"
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>

using PointType = pcl::PointXYZ;
using namespace faster_lio;
using IVoxType = IVox<3, IVoxNodeType::PHC, PointType>; // 定义 IVox 类型
// using IVoxType = IVox<3, IVoxNodeType::DEFAULT, PointType>;
using PointVector = KD_TREE<PointType>::PointVector;
template class KD_TREE<pcl::PointXYZ>;

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

// Print 4x4 Matrix
void printMatrix(const Eigen::Matrix4f &T)
{
  printf("Transform matrix :\n");
  printf("    | %6.3f %6.3f %6.3f %6.3f | \n", T(0, 0), T(0, 1), T(0, 2), T(0, 3));
  printf("T = | %6.3f %6.3f %6.3f %6.3f | \n", T(1, 0), T(1, 1), T(1, 2), T(1, 3));
  printf("    | %6.3f %6.3f %6.3f %6.3f | \n", T(2, 0), T(2, 1), T(2, 2), T(2, 3));
  printf("    | %6.3f %6.3f %6.3f %6.3f | \n", T(3, 0), T(3, 1), T(3, 2), T(3, 3));
}

// Compute mean distance between [cl_s] and [cl_s_reg] by kd-tree of [cl_s]
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

// Manually rotate [cl_s] to [cl_t] by ground truth T_ts = [R_ts, t_ts]
void CreateTargetGT(pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_s,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_t,
                    Eigen::Matrix4f &T_ts_gt)
{
  Eigen::Matrix4f T_ts = Eigen::Matrix4f::Identity();

  // give some rotation
  double theta = M_PI / 8;
  T_ts(0, 0) = cos(theta);
  T_ts(0, 1) = -sin(theta);
  T_ts(1, 0) = sin(theta);
  T_ts(1, 1) = cos(theta);

  // give some translation
  T_ts(2, 3) = 0.4;

  // target_gt = T_ts * source
  pcl::transformPointCloud(*cl_s, *cl_t, T_ts);
  // return value
  T_ts_gt = T_ts;
}

// Point cloud registration by PCL-ICP
void ICP_PCL(pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_s,
             pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_t,
             pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_final,
             Eigen::Matrix4f &T_ts,
             double &time_cost)
{
  clock_t startT, endT;
  startT = clock();

  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  // trs source to  target = T_ts
  icp.setInputSource(cl_s);
  icp.setInputTarget(cl_t);
  icp.setMaxCorrespondenceDistance(10); // important
  icp.setMaximumIterations(50);
  icp.setTransformationEpsilon(1e-10);
  icp.setEuclideanFitnessEpsilon(0.001);
  icp.align(*cl_final);
  T_ts = icp.getFinalTransformation();

  endT = clock();
  time_cost = (double)(endT - startT) / CLOCKS_PER_SEC;
}

// Point cloud registration by PCL-NDT
void NDT_PCL(pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_s,
             pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_t,
             pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_final,
             Eigen::Matrix4f &T_ts,
             double &time_cost)
{
  clock_t startT, endT;
  startT = clock();

  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
  // pcl::PointCloud<pcl::PointXYZ>::Ptr cl_s_filt(new pcl::PointCloud<pcl::PointXYZ>);
  // pcl::PointCloud<pcl::PointXYZ>::Ptr cl_t_filt(new pcl::PointCloud<pcl::PointXYZ>);
  // pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
  // voxel_filter.setLeafSize(0.1f, 0.1f, 0.1f);
  // voxel_filter.setInputCloud(cl_s);
  // voxel_filter.filter(*cl_s_filt);
  // voxel_filter.setInputCloud(cl_t);
  // voxel_filter.filter(*cl_t_filt);

  ndt.setInputSource(cl_s);
  ndt.setInputTarget(cl_t);
  ndt.setStepSize(0.1);
  ndt.setResolution(1.0);
  ndt.setMaximumIterations(50);
  ndt.setTransformationEpsilon(1e-10);
  ndt.align(*cl_final);
  T_ts = ndt.getFinalTransformation();

  endT = clock();
  time_cost = (double)(endT - startT) / CLOCKS_PER_SEC;
}

// Point cloud registration by manual ICP-SVD
void ICP_MANUAL(pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_s,
                pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_t,
                pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_final,
                Eigen::Matrix4f &T_ts,
                double &time_cost)
{
  clock_t startT, endT, kd_tree_start, kd_tree_end;
  startT = clock();

  // target cloud is fixed, so kd-tree create for target
  pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
  kd_tree.setInputCloud(cl_t);

  IVoxType::Options ivox_options;
  std::shared_ptr<IVoxType> ivox = std::make_shared<IVoxType>(ivox_options);
  ivox->AddPoints(cl_t->points);
  size_t daivo = ivox->NumPoints();
  cout << "The data of ivox.size : " << daivo << endl;

  // IVoxType ivox(ivox_options);
  // ivox.AddPoints(cl_t->points);
  // size_t daivo = ivox.NumPoints();
  // cout << "The data of ivox.size : " << daivo << endl;

  int nICP_Step = 10;                                      // you can change this param for different situations
  float max_correspond_distance_ = 1;                      // you can change this param for different situations
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity(); // init T_st

  // The times of kd-tree is recording

  for (int i = 0; i < nICP_Step; ++i)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_S_Trans(new pcl::PointCloud<pcl::PointXYZ>);
    // let us transform [cl_s] to [cloud_S_Trans] by [T_st](transform)
    // T_st changed for each interation, make [cloud_S_Trans] more and more close to [cl_s]
    pcl::transformPointCloud(*cl_s, *cloud_S_Trans, transform);
    std::vector<int> match_indexs;
    kd_tree_start = clock();
    for (auto i = 0; i < cloud_S_Trans->size(); ++i)
    {
      std::vector<float> k_distances(2);
      std::vector<int> k_indexs(2);

      kd_tree.nearestKSearch(cloud_S_Trans->at(i), 1, k_indexs, k_distances);
      if (k_distances[0] < max_correspond_distance_)
      {
        match_indexs.emplace_back(k_indexs[0]);
      }
    }

    kd_tree_end = clock();
    double kd_times = (double)(kd_tree_end - kd_tree_start) / CLOCKS_PER_SEC;
    cout << "The times of kd-tree in search :" << kd_times << endl;

    // matched clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_match_from_S_trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud_S_Trans, match_indexs, *cloud_match_from_S_trans);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_match_from_T(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cl_t, match_indexs, *cloud_match_from_T);

    // 3d center of two clouds
    Eigen::Vector4f mu_S_temp, mu_T_temp;
    pcl::compute3DCentroid(*cloud_match_from_S_trans, mu_S_temp);
    pcl::compute3DCentroid(*cloud_match_from_T, mu_T_temp);
    Eigen::Vector3f mu_S(mu_S_temp[0], mu_S_temp[1], mu_S_temp[2]);
    Eigen::Vector3f mu_T(mu_T_temp[0], mu_T_temp[1], mu_T_temp[2]);

    // H += (S_i) * (T_i^T)
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

    // H = SVD
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(H_icp, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3f R_ts = svd.matrixU() * (svd.matrixV().transpose());
    Eigen::Vector3f t_ts = mu_T - R_ts * mu_S; // T = R_ts * S + t_ts
    Eigen::Quaternionf q_ts(R_ts);
    transform.block<3, 3>(0, 0) *= q_ts.normalized().toRotationMatrix().inverse(); // we use left form, so we need .inverse()
    transform.block<3, 1>(0, 3) += t_ts;
    // We got a new [transform] here, that is, we are more closer to the [source] than last [transform].
    // This is why we call it Iterative method.
  }
  T_ts = transform; // transform = T_ts
  // Target = T_t * Source
  pcl::transformPointCloud(*cl_s, *cl_final, T_ts);

  endT = clock();
  time_cost = (double)(endT - startT) / CLOCKS_PER_SEC;
}

// pcl viewer for results
void ShowMatchResults(pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_s,
                      pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_t,
                      pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_reg_pcl_icp,
                      pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_reg_pcl_ndt,
                      pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_reg_pcl_man)
{
  // show raw and matched clouds in 4 view
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
      new pcl::visualization::PCLVisualizer("cloud compare view"));

  // multi view port
  int v1(0), v2(0), v3(0), v4(0);
  viewer->createViewPort(0.0, 0, 0.25, 1, v1);
  viewer->createViewPort(0.25, 0, 0.5, 1, v2);
  viewer->createViewPort(0.5, 0, 0.75, 1, v3);
  viewer->createViewPort(0.75, 0, 1.0, 1, v4);
  viewer->setBackgroundColor(255, 255, 255, v1);
  viewer->setBackgroundColor(255, 255, 255, v2);
  viewer->setBackgroundColor(255, 255, 255, v3);
  viewer->setBackgroundColor(255, 255, 255, v4);

  // handle
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cl_s_h(cl_s, 100, 100, 100);                 // source
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cl_t_gt_h(cl_t, 0, 250, 0);                  // target_gt
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cl_reg_pcl_icp_h(cl_reg_pcl_icp, 0, 0, 222); // pcl-icp
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cl_reg_pcl_ndt_h(cl_reg_pcl_ndt, 0, 0, 222); // pcl-ndt
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cl_reg_pcl_man_h(cl_reg_pcl_man, 0, 0, 222); // icp-svd man

  // view 1 : S , T_gt
  viewer->addPointCloud(cl_s, cl_s_h, "s_v1", v1);
  viewer->addPointCloud(cl_t, cl_t_gt_h, "t_v1", v1);
  // view 2 : S , T_gt , T_pcl_icp
  viewer->addPointCloud(cl_s, cl_s_h, "s_v2", v2);
  viewer->addPointCloud(cl_t, cl_t_gt_h, "t_v2", v2);
  viewer->addPointCloud(cl_reg_pcl_icp, cl_reg_pcl_icp_h, "pcl_icp_v2", v2);
  // view 3 : S , T_gt , T_pcl_ndt
  viewer->addPointCloud(cl_s, cl_s_h, "s_v3", v3);
  viewer->addPointCloud(cl_t, cl_t_gt_h, "t_v3", v3);
  viewer->addPointCloud(cl_reg_pcl_ndt, cl_reg_pcl_ndt_h, "pcl_icp_v3", v3);
  // view 4 : S , T_gt , T_icp_man
  viewer->addPointCloud(cl_s, cl_s_h, "s_v4", v4);
  viewer->addPointCloud(cl_t, cl_t_gt_h, "t_v4", v4);
  viewer->addPointCloud(cl_reg_pcl_man, cl_reg_pcl_man_h, "icp_man_v4", v4);

  viewer->addCoordinateSystem(1.0);
  while (!viewer->wasStopped())
  {
    // view->spin();
    viewer->spinOnce(200);
    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  }
}

int main(int argc, char **argv)
{
  // Initialize k-d tree
  KD_TREE<PointType>::Ptr kdtree_ptr(new KD_TREE<PointType>(0.3, 0.6, 0.2));
  KD_TREE<PointType> &ikd_Tree = *kdtree_ptr;

  // read PCD
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>("../materials/rabbit.pcd", *cloud_source) == -1)
  {
    PCL_ERROR("Couldn't read pcd file! \n");
    return (-1);
  }

  // create ground truth Target
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target(new pcl::PointCloud<pcl::PointXYZ>);
  Eigen::Matrix4f T_ts_gt = Eigen::Matrix4f::Identity();
  CreateTargetGT(cloud_source, cloud_target, T_ts_gt);

  // Build ikd-tree
  clock_t ikd_build_time_start, ikd_build_time_end;
  ikd_build_time_start = clock();
  ikd_Tree.Build((*cloud_target).points);
  ikd_build_time_end = clock();
  double build_ikd_tree_times = (double)(ikd_build_time_end - ikd_build_time_start) / CLOCKS_PER_SEC;
  cout << "build_ikd_tree_times: " << build_ikd_tree_times << "s" << endl;

  // Build kd-tree
  clock_t kd_build_time_start, kd_build_time_end;
  kd_build_time_start = clock();
  pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree_target;
  kd_tree_target.setInputCloud(cloud_target);
  kd_build_time_end = clock();
  double build_kd_tree_times = (double)(kd_build_time_end - kd_build_time_start) / CLOCKS_PER_SEC;
  cout << "build_kd_tree_times: " << build_kd_tree_times << "s" << endl;

  // compute dist
  // pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree_target;
  // kd_tree_target.setInputCloud(cloud_target);

  double t_pcl_icp = 0;
  double t_pcl_ndt = 0;
  double t_icp_man = 0;
  Eigen::Matrix4f T_ts_pcl_icp = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f T_ts_pcl_ndt = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f T_ts_man = Eigen::Matrix4f::Identity();
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_reg_pcl_icp(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_reg_pcl_ndt(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_reg_man(new pcl::PointCloud<pcl::PointXYZ>);

  // match two clouds by [PCL-ICP]
  ICP_PCL(cloud_source, cloud_target, cloud_reg_pcl_icp, T_ts_pcl_icp, t_pcl_icp);
  // match two clouds by [PCL-NDT]
  NDT_PCL(cloud_source, cloud_target, cloud_reg_pcl_ndt, T_ts_pcl_ndt, t_pcl_ndt);
  // match two clouds by [ICP-SVD-manual]
  ICP_MANUAL(cloud_source, cloud_target, cloud_reg_man, T_ts_man, t_icp_man);

  // pcl_icp kd-tree spend time
  clock_t pcl_icp_start, pcl_icp_end, pcl_ndt_start, pcl_ndt_end, icp_man_start, icp_man_end;
  pcl_icp_start = clock();
  double dis_pcl_icp = EvaluateCLoudsDist_kd_tree(kd_tree_target, cloud_reg_pcl_icp);
  pcl_icp_end = clock();
  double pcl_icp_kd_tree_time = (double)(pcl_icp_end - pcl_icp_start) / CLOCKS_PER_SEC;

  // pcl_ndt kd-tree spend time
  pcl_ndt_start = clock();
  double dis_pcl_ndt = EvaluateCLoudsDist_kd_tree(kd_tree_target, cloud_reg_pcl_ndt);
  pcl_ndt_end = clock();
  double pcl_ndt_kd_tree_time = (double)(pcl_ndt_end - pcl_ndt_start) / CLOCKS_PER_SEC;

  // icp_man kd-tree spend time
  icp_man_start = clock();
  double dis_icp_man = EvaluateCLoudsDist_kd_tree(kd_tree_target, cloud_reg_man);
  icp_man_end = clock();
  double icp_man_kd_tree_time = (double)(icp_man_end - icp_man_start) / CLOCKS_PER_SEC;

  // print results
  std::cout << "Compare the ground truth T_ts with the estimated T_ts :" << std::endl;
  printMatrix(T_ts_gt);
  printMatrix(T_ts_pcl_icp);
  printMatrix(T_ts_pcl_ndt);
  printMatrix(T_ts_man);
  std::cout << "Compare the mean distances between pointcloud T and T_reg :" << std::endl;
  std::cout << "dist of PCL-ICP = " << dis_pcl_icp << std::endl;
  std::cout << "dist of PCL-NDT = " << dis_pcl_ndt << std::endl;
  std::cout << "dist of ICP-SVD = " << dis_icp_man << std::endl;
  std::cout << "Compare the time cost of each method :" << std::endl;
  cout << "time cost of PCL-ICP = " << t_pcl_icp << "s" << endl;
  cout << "time cost of PCL-NDT = " << t_pcl_ndt << "s" << endl;
  cout << "time cost of ICP-SVD = " << t_icp_man << "s" << endl;
  cout << "Compare the varity of cloud resgister in kd-tree spending times :" << endl;
  cout << "pcl_icp_kd_tree_time :" << pcl_icp_kd_tree_time << "s" << endl;
  cout << "pcl_ndt_kd_tree_time :" << pcl_ndt_kd_tree_time << "s" << endl;
  cout << "icp_man_kd_tree_time :" << icp_man_kd_tree_time << "s" << endl;

  // pcl viewer
  ShowMatchResults(cloud_source, cloud_target, cloud_reg_pcl_icp, cloud_reg_pcl_ndt, cloud_reg_man);
  return (0);
}

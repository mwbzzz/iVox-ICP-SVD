#include <iostream>
#include <ctime>
#include <chrono>
#include <fstream>
#include <sstream>
#include <map>
#include <string>
#include <cmath> // 用于 sqrt 函数
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
using IVoxType = IVox<3, IVoxNodeType::PHC, PointType>; // 定义 IVox 类型
// using IVoxType = IVox<3, IVoxNodeType::DEFAULT, PointType>;

using PointVector = KD_TREE<PointType>::PointVector;

template class KD_TREE<pcl::PointXYZ>;

// measureKdTreeSearchTime info
double measureKdTreeSearchTime(pcl::KdTreeFLANN<pcl::PointXYZ> &kd_tree,
                               pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_S_Trans,
                               float max_correspond_distance,
                               std::vector<int> &match_indexes);

// measureIKdTreeSearchTime info
template <typename PointType>
double measureIkdTreeSearchTime(KD_TREE<PointType> &ikd_tree,
                                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_S_Trans,
                                float max_correspond_distance,
                                std::vector<int> &match_indexes);

// measureIVoxSearchTime info
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

// 定义一个枚举类型来表示不同的点云存储结构
enum class PointCloudStructure
{
  KD_TREE,
  IKD_TREE,
  IVOX
};

template <typename PointType>
double searchPointCloud(PointCloudStructure structure,
                        void *tree, // 指向不同树的指针
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

// Print 4x4 Matrix
void printMatrix(const Eigen::Matrix4f &T)
{
  printf("Transform matrix :\n");
  printf("    | %6.3f %6.3f %6.3f %6.3f | \n", T(0, 0), T(0, 1), T(0, 2), T(0, 3));
  printf("T = | %6.3f %6.3f %6.3f %6.3f | \n", T(1, 0), T(1, 1), T(1, 2), T(1, 3));
  printf("    | %6.3f %6.3f %6.3f %6.3f | \n", T(2, 0), T(2, 1), T(2, 2), T(2, 3));
  printf("    | %6.3f %6.3f %6.3f %6.3f | \n", T(3, 0), T(3, 1), T(3, 2), T(3, 3));
}

// Compute mean distance between [cl_s] and [cl_s_reg] using kd-tree of [cl_s]
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

// RMSE计算函数：利用目标点云构建kd-tree，对每个配准后的点找到最近邻，计算均方根误差
double ComputeRMSE(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_registered)
{
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(cloud_target);
  double sum_squared_error = 0.0;
  int count = 0;
  for (size_t i = 0; i < cloud_registered->size(); i++)
  {
    std::vector<int> indices(1);
    std::vector<float> sqr_distances(1);
    if (kdtree.nearestKSearch(cloud_registered->points[i], 1, indices, sqr_distances) > 0)
    {
      sum_squared_error += sqr_distances[0];
      count++;
    }
  }
  if (count == 0)
    return 0.0;
  double mse = sum_squared_error / count;
  return sqrt(mse);
}

// Manually rotate [cl_s] to [cl_t] by ground truth T_ts = [R_ts, t_ts]
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

// Point cloud registration by manual ICP-SVD
void ICP_MANUAL(pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_s,
                pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_t,
                pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_final,
                Eigen::Matrix4f &T_ts,
                double &time_cost,
                PointCloudStructure structure,
                void *tree)
{

  pcl::PointCloud<pcl::PointXYZ>::Ptr cl_t_filt(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cl_t_st_filt(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> stat_filter;
  pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
  voxel_filter.setInputCloud(cl_t);
  voxel_filter.setLeafSize(0.01f, 0.01f, 0.01f);
  voxel_filter.filter(*cl_t_filt);
  // stat_filter.setInputCloud(cl_t_filt);
  //  stat_filter.setMeanK(50);
  //  stat_filter.setStddevMulThresh(1.0);
  //  stat_filter.setNegative(false);
  //  stat_filter.filter(*cl_t_st_filt);

  // 对目标点云构建kd-tree，用于RMSE计算
  pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
  kd_tree.setInputCloud(cl_t);
  auto kd_size = cl_t->size();
  std::cout << " The data of kd_Tree.size : " << kd_size << std::endl;

  // 构建 IKdTree
  KD_TREE<PointType>::Ptr kdtree_ptr(new KD_TREE<PointType>(0.3, 0.6, 0.2));
  KD_TREE<PointType> &ikd_Tree = *kdtree_ptr;
  ikd_Tree.Build((*cl_t).points);
  auto ikd_size = ikd_Tree.size();
  std::cout << "The data of ikd_Tree.size : " << ikd_size << std::endl;

  // 构建 IVox
  IVoxType::Options ivox_options;

  ivox_options.nearby_type_ = IVoxType::NearbyType::NEARBY6;

  // ivox_options.resolution_ = 2;
  // ivox_options.inv_resolution_ = 0.5;
  IVoxType ivox(ivox_options);
  ivox.AddPoints(cl_t_filt->points);
  size_t daivo = ivox.NumPoints();
  std::cout << "The data of ivox.size : " << daivo << std::endl;

  clock_t startT, endT;
  startT = clock();
  int nICP_Step = 50;
  float max_correspond_distance_ = 5;
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();

  double total_kd_tree_time = 0.0;
  double total_ikd_tree_time = 0.0;
  double total_ivox_time = 0.0;

  for (int i = 0; i < nICP_Step; ++i)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_S_Trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cl_s, *cloud_S_Trans, transform);

    std::vector<int> match_indexs;
    double search_time = searchPointCloud<PointType>(structure, tree, cloud_S_Trans, max_correspond_distance_, match_indexs);

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

  endT = clock();
  time_cost = (double)(endT - startT) / CLOCKS_PER_SEC;

  // ikd-tree time
  double average_ikd_tree_time = total_ikd_tree_time / nICP_Step;
  std::cout << "Total ikd-tree search time: " << total_ikd_tree_time << " seconds." << std::endl;
}

void ShowMatchResults(pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_s,
                      pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_t,
                      pcl::PointCloud<pcl::PointXYZ>::Ptr &cl_reg_pcl_man)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
      new pcl::visualization::PCLVisualizer("cloud compare view"));
  int v1(0);
  viewer->createViewPort(0.0, 0, 1, 1, v1);
  viewer->setBackgroundColor(255, 255, 255, v1);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cl_s_h(cl_s, 100, 100, 100);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cl_t_gt_h(cl_t, 0, 250, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cl_reg_pcl_man_h(cl_reg_pcl_man, 0, 0, 222);
  viewer->addPointCloud(cl_s, cl_s_h, "s_v1", v1);
  viewer->addPointCloud(cl_t, cl_t_gt_h, "t_v1", v1);
  viewer->addPointCloud(cl_reg_pcl_man, cl_reg_pcl_man_h, "icp_man_v1", v1);
  while (!viewer->wasStopped())
  {
    viewer->spinOnce(200);
    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  }
}

struct param
{
  std::string pcd_name;
  float ivox_resolution;
  std::string ivox_nearby_type;
  std::string point_cloud_structure;
};

param readFile(const std::string &file_in)
{
  param params;
  std::ifstream file(file_in);
  if (!file.is_open())
  {
    std::cerr << "无法打开配置文件: " << file_in << std::endl;
    return params;
  }

  std::string line;
  while (std::getline(file, line))
  {
    // 跳过空行和注释行（以#开头）
    if (line.empty() || line[0] == '#')
    {
      continue;
    }

    // 找到等号的位置
    size_t equalPos = line.find('=');
    if (equalPos != std::string::npos)
    {
      std::string key = line.substr(0, equalPos);
      std::string value = line.substr(equalPos + 1);

      if (key == "pcd_name")
      {
        params.pcd_name = value;
      }
      else if (key == "ivox_resolution")
      {
        params.ivox_resolution = std::stof(value);
      }
      else if (key == "ivox_nearby_type")
      {
        params.ivox_nearby_type = value;
      }
      else if (key == "point_cloud_structure")
      {
        params.point_cloud_structure = value;
      }
    }
  }

  file.close();
  return params;
}

int main(int argc, char **argv)
{
  param myparam = readFile("../config.txt");

  // 打印读取的参数
  std::cout << "PCD文件名: " << myparam.pcd_name << std::endl;
  std::cout << "IVox分辨率: " << myparam.ivox_resolution << std::endl;
  std::cout << "IVox附近类型: " << myparam.ivox_nearby_type << std::endl;
  std::cout << "点云存储结构: " << myparam.point_cloud_structure << std::endl;

  // 读取 PCD 文件
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
  std::string pcd_path = "../materials/" + myparam.pcd_name;
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path, *cloud_source) == -1)
  {
    PCL_ERROR("无法读取PCD文件: %s\n", pcd_path.c_str());
    return (-1);
  }

  // 生成目标点云（Ground Truth）
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target(new pcl::PointCloud<pcl::PointXYZ>);
  Eigen::Matrix4f T_ts_gt = Eigen::Matrix4f::Identity();
  CreateTargetGT(cloud_source, cloud_target, T_ts_gt);

  // 对目标点云构建 kd-tree，用于RMSE计算
  pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree_target;
  kd_tree_target.setInputCloud(cloud_target);

  // 初始化点云存储结构
  PointCloudStructure structure;
  void *tree = nullptr;

  if (myparam.point_cloud_structure == "KD_TREE")
  {
    structure = PointCloudStructure::KD_TREE;
    // 初始化 KD_TREE
    pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
    kd_tree.setInputCloud(cloud_target);
    tree = static_cast<void *>(&kd_tree);
  }
  else if (myparam.point_cloud_structure == "IKD_TREE")
  {
    structure = PointCloudStructure::IKD_TREE;
    // 初始化 IKD_TREE
    KD_TREE<PointType>::Ptr kdtree_ptr(new KD_TREE<PointType>(0.3, 0.6, 0.2));
    KD_TREE<PointType> &ikd_Tree = *kdtree_ptr;
    ikd_Tree.Build((*cloud_target).points);
    tree = static_cast<void *>(kdtree_ptr.get());
  }
  else if (myparam.point_cloud_structure == "IVOX")
  {
    structure = PointCloudStructure::IVOX;
    // 初始化 IVOX
    IVoxType::Options ivox_options;
    ivox_options.nearby_type_ = IVoxType::NearbyType::NEARBY6;
    ivox_options.resolution_ = myparam.ivox_resolution;
    ivox_options.inv_resolution_ = 1.0f / myparam.ivox_resolution;
    IVoxType ivox(ivox_options);
    ivox.AddPoints(cloud_target->points);
    tree = static_cast<void *>(&ivox);
  }
  else
  {
    std::cerr << "无效的点云存储结构: " << myparam.point_cloud_structure << std::endl;
    return -1;
  }

  double t_icp_man = 0;
  Eigen::Matrix4f T_ts_man = Eigen::Matrix4f::Identity();
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_reg_man(new pcl::PointCloud<pcl::PointXYZ>);

  // 利用 ICP-SVD 手动配准
  ICP_MANUAL(cloud_source, cloud_target, cloud_reg_man, T_ts_man, t_icp_man, structure, tree);

  // 计算配准后点云与目标点云的均方根误差（RMSE）
  double rmse = ComputeRMSE(cloud_target, cloud_reg_man);
  std::cout << "配准的均方根误差 (RMSE): " << rmse << std::endl;

  // 计算 kd-tree 搜索耗时（评价指标之一）
  clock_t icp_man_start, icp_man_end;
  icp_man_start = clock();
  double dis_icp_man = EvaluateCLoudsDist_kd_tree(kd_tree_target, cloud_reg_man);
  icp_man_end = clock();
  double icp_man_kd_tree_time = (double)(icp_man_end - icp_man_start) / CLOCKS_PER_SEC;

  std::cout << "比较真实变换矩阵 T_ts 和估计的 T_ts :" << std::endl;
  printMatrix(T_ts_gt);
  printMatrix(T_ts_man);
  std::cout << "比较点云 T 和 T_reg 的平均距离 :" << std::endl;
  std::cout << "ICP-SVD 的距离 = " << dis_icp_man << std::endl;
  std::cout << "ICP-SVD 的 kd-tree 时间: " << icp_man_kd_tree_time << "秒" << std::endl;

  // 显示配准结果
  ShowMatchResults(cloud_source, cloud_target, cloud_reg_man);
  return (0);
}
#include <pcl/io/pcd_io.h>                    // 引入点云数据输入输出相关的头文件
#include <pcl/point_types.h>                  // 引入点云类型定义相关的头文件
#include <pcl/common/io.h>                    // 引入常用的点云输入输出函数
#include <pcl/keypoints/iss_3d.h>             // 引入ISS关键点检测算法的头文件
#include <pcl/features/normal_3d.h>           // 引入法线估计相关的头文件
#include <pcl/visualization/pcl_visualizer.h> // 引入PCL可视化工具的头文件
#include <boost/thread.hpp>                   // 引入Boost线程库，用于多线程操作
#include "../ikd-tree/ikd_Tree copy.h"

// 定义点云类型为RGBA格式
typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloud;
using PointType = pcl::PointXYZRGBA;

// 点云可视化函数，接受模型和场景关键点作为参数
void visualize_pcd(PointCloud::Ptr model, pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_keypoints)
{
  // 创建一个PCL可视化对象
  pcl::visualization::PCLVisualizer viewer("registration Viewer");

  // 设置模型点云的颜色为绿色
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> model_color(model, 0, 255, 0);
  // 设置场景关键点的颜色为蓝色
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> scene_keypoint_color(scene_keypoints, 0, 0, 255);

  // 设置背景颜色为白色
  viewer.setBackgroundColor(255, 255, 255);
  // 添加模型点云到视图中
  viewer.addPointCloud(model, model_color, "model");
  // 添加场景关键点到视图中
  viewer.addPointCloud(scene_keypoints, scene_keypoint_color, "scene_keypoints");
  // 设置场景关键点的点大小
  viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "scene_keypoints");

  // 循环直到关闭可视化窗口
  while (!viewer.wasStopped())
  {
    viewer.spinOnce(100);                                               // 更新可视化
    boost::this_thread::sleep(boost::posix_time::microseconds(100000)); // 暂停线程100毫秒
  }
}

// PCD文件的路径
const std::string filename = "../materials/rabbit.pcd";

int main(int, char **argv)
{
  // 创建一个新的点云对象，存储RGBA格式的点云数据
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);

  // 加载PCD文件，如果加载失败则返回错误信息
  if (pcl::io::loadPCDFile<pcl::PointXYZRGBA>(filename, *cloud) == -1)
  {
    pcl::console::print_error("Couldn't read file %s!\n", argv[1]); // 输出错误信息
    return (-1);                                                    // 返回错误状态
  }

  // 输出点云中的点数量
  std::cout << "points: " << cloud->points.size() << std::endl;

  // 创建ISS关键点检测对象
  pcl::ISSKeypoint3D<pcl::PointXYZRGBA, pcl::PointXYZRGB> iss_detector;

  // 创建一个新的点云对象，存储关键点
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZRGB>());
  // 创建Kd树对象用于搜索
  pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBA>());

  // 设置ISS检测器的搜索方法为Kd树
  iss_detector.setSearchMethod(tree);
  // 设置ISS检测器的参数
  iss_detector.setSalientRadius(0.007f); // 设置显著半径
  iss_detector.setNonMaxRadius(0.005f);  // 设置非最大半径
  iss_detector.setThreshold21(0.65);     // 设置阈值21
  iss_detector.setThreshold32(0.1);      // 设置阈值32
  iss_detector.setMinNeighbors(4);       // 设置最小邻居数
  iss_detector.setNumberOfThreads(4);    // 设置使用的线程数
  iss_detector.setInputCloud(cloud);     // 设置输入点云
  iss_detector.compute(*keypoints);      // 计算关键点

  // 输出计算得到的关键点数量
  std::cout << "N of ISS_3D points in the result are " << (*keypoints).points.size() << std::endl;
  // 将计算得到的关键点保存为PCD文件
  pcl::io::savePCDFile("keypoints_iss_3d.pcd", *keypoints, true);

  // 调用可视化函数展示结果
  visualize_pcd(cloud, keypoints);
  return 0; // 程序正常结束
}

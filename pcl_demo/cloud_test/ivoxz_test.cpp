#include <iostream>
#include <vector>
#include <algorithm>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <Eigen/StdVector>
#include "../ivox3d/ivox3d.h" // 确保路径正确
#include "../ivox3d/ivox3d_node.hpp"
// #include "include/options.h"

using namespace faster_lio;

// 定义点类型
using PointType = pcl::PointXYZ;
using IVoxType = IVox<3, IVoxNodeType::PHC, PointType>; // 定义 IVox 类型
using PointCloudType = pcl::PointCloud<PointType>;
using CloudPtr = PointCloudType::Ptr;

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

// 主函数
int main()
{
    // 创建一个点云对象
    CloudPtr cloud(new PointCloudType);

    // 创建 PCDReader 实例
    pcl::PCDReader reader;

    // 读取点云文件
    if (reader.read("../materials/xpd_cloud_filtered.pcd", *cloud) == -1)
    {
        std::cerr << "Couldn't read file xpd_cloud_filtered.pcd" << std::endl;
        return -1;
    }

    // 输出读取到的点云信息
    std::cout << "Loaded point cloud with " << cloud->size() << " points." << std::endl;

    // 设置 IVox 的选项
    IVoxType::Options ivox_options_;
    // ivox_options_.resolution_ = 0.2; // 设置分辨率
    // ivox_options_.capacity_ = 1000000; // 设置体素容量

    // 创建 IVox 实例
    std::shared_ptr<IVoxType> ivox_ = std::make_shared<IVoxType>(ivox_options_);

    // 将点云添加到 IVox 中
    ivox_->AddPoints(cloud->points);
    size_t num_points = ivox_->NumPoints();
    std::cout << "Total number of points in IVox : " << num_points << std::endl;

    std::cout << "Point cloud data has been successfully added to the IVox structure!" << std::endl;

    return 0;
}

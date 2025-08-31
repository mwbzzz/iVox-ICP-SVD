#include <iostream>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include "../ivox3d/ivox3d.h" // 确保路径正确
#include "../ivox3d/ivox3d_node.hpp"
#include <ctime>
#include <cmath>

using namespace faster_lio;

// 定义点类型
using PointType = pcl::PointXYZ;
using IVoxType = IVox<3, IVoxNodeType::DEFAULT, PointType>;

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

int main()
{
    // 1. 创建并初始化 IVox
    IVoxType::Options ivox_options_;
    ivox_options_.resolution_ = 0.3;                             // 设置分辨率
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26; // 设置邻域范围
    IVoxType ivox(ivox_options_);

    // 2. 加载点云数据
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
    if (pcl::io::loadPCDFile<PointType>("../materials/xpd_cloud_filtered.pcd", *cloud) == -1)
    {
        PCL_ERROR("Couldn't read file example.pcd\n");
        return -1;
    }

    std::cout << "Loaded point cloud with " << cloud->size() << " points." << std::endl;

    // 3. 构建 IVox
    ivox.AddPoints(cloud->points);
    std::cout << "Total number of points in IVox: " << ivox.NumPoints() << std::endl;

    // 4. 设置目标点和搜索参数
    PointType target_point;
    target_point.x = 0.5;
    target_point.y = 0.5;
    target_point.z = 0.5;
    int k_nearest = 10;     // 查询 10 个最近邻
    double max_dist = 10.0; // 最大搜索范围

    // 5. 最近邻搜索
    std::vector<PointType, Eigen::aligned_allocator<PointType>> nearest_points;
    clock_t start, end;
    start = clock();
    bool found = ivox.GetClosestPoint(target_point, nearest_points, k_nearest, max_dist);
    end = clock();

    double time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    std::cout << "Search time: " << time << " seconds." << std::endl;

    // 6. 输出结果
    if (found)
    {
        std::cout << "Nearest " << k_nearest << " points to ("
                  << target_point.x << ", " << target_point.y << ", " << target_point.z << "):" << std::endl;
        for (const auto &pt : nearest_points)
        {
            double distance = std::sqrt(std::pow(pt.x - target_point.x, 2) +
                                        std::pow(pt.y - target_point.y, 2) +
                                        std::pow(pt.z - target_point.z, 2));
            std::cout << "Point: (" << pt.x << ", " << pt.y << ", " << pt.z
                      << "), Distance: " << distance << std::endl;
        }
    }
    else
    {
        std::cout << "No neighbors found within the specified range." << std::endl;
    }

    return 0;
}

#include "../ikd-tree/ikd_Tree.h"
#include <pcl/io/pcd_io.h>
#include <iostream>
#include <vector>
#include "pcl/point_types.h"

using PointType = pcl::PointXYZ;

int main() {
    // 1. 创建并初始化 KD 树
    KD_TREE<PointType>::Ptr kdtree_ptr(new KD_TREE<PointType>(0.3, 0.6, 0.2));
    KD_TREE<PointType>& ikd_Tree = *kdtree_ptr;

    // 2. 加载点云数据
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
    if (pcl::io::loadPCDFile<PointType>("../materials/xpd_cloud_filtered.pcd", *cloud) == -1) {
        PCL_ERROR("Couldn't read file example.pcd\n");
        return -1;
    }

    // 3. 构建 iKD 树
    ikd_Tree.Build(cloud->points);
    auto da= ikd_Tree.size();
    cout<<"dadaa:  "<<da<<endl;

    // 4. 设置目标点和搜索参数
    pcl::PointXYZ target_point;
    target_point.x = 0.5;
    target_point.y = 0.5;
    target_point.z = 0.5;
    int k_nearest = 10; // 查询 10 个最近邻
    double max_dist = 10.0; // 最大搜索范围

    clock_t start,end;
    start = clock();
    // 5. 调用 Nearest_Search
    KD_TREE<pcl::PointXYZ>::PointVector nearest_points;
    std::vector<float> point_distances;
    ikd_Tree.Nearest_Search(target_point, k_nearest, nearest_points, point_distances, max_dist);
    end = clock();
    double time = (double)(end-start)/CLOCKS_PER_SEC;
    cout<<"search times :"<<time<<endl;
    // 6. 输出结果
    std::cout << "Nearest " << k_nearest << " points to (" 
              << target_point.x << ", " << target_point.y << ", " << target_point.z << "):" << std::endl;
    for (size_t i = 0; i < nearest_points.size(); ++i) {
        std::cout << "Point: (" << nearest_points[i].x << ", " << nearest_points[i].y << ", " << nearest_points[i].z
                  << "), Distance: " << point_distances[i] << std::endl;
    }
  
    return 0;
}

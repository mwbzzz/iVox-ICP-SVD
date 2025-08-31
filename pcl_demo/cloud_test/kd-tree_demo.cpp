#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <iostream>
#include <vector>
#include <ctime>

using namespace std;

typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointerCloudPtr;
int main()
{
    srand(time(NULL));

    // 创建点云指针
    PointerCloudPtr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PCDReader reader;
    if (reader.read("../materials/xpd_cloud_filtered.pcd", *cloud) == -1)
    {
        std::cerr << "Couldn't read file xpd_cloud.pcd" << std::endl;
        return -1;
    }

    // 生成点云
    //  cloud->width =1000;
    //  cloud->height = 1;
    //  cloud->points.resize(cloud->width*cloud->height);

    // for(size_t i = 0;i<cloud->size();++i)
    // {
    // cloud->points[i].x = 1024*rand()/(RAND_MAX +1.0f);
    // cloud->points[i].y = 1024*rand()/(RAND_MAX +1.0f);
    // cloud->points[i].z = 1024*rand()/(RAND_MAX +1.0f);
    // }

    // 创建KD-tree对象
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

    // 向KD-tree 中传入数据，即将点云数据设置成KD-tree结构
    kdtree.setInputCloud(cloud);

    // 随即生成一个点
    pcl::PointXYZ searchPoint;

    searchPoint.x = 0.5;
    searchPoint.y = 0.5;
    searchPoint.z = 0.5;

    int K = 10;

    // K近邻搜索，即搜索该点周围的10个点
    vector<int> pointIdxKNNSearch(K);

    // 设置搜索距离为10
    vector<float> pointKNNSquaredDistance(K);

    time_t begin1, end1;
    begin1 = clock();
    // cout<<"K nearest neighbor search at(" <<searchPoint.x
    // <<" "<<searchPoint.y
    // <<" "<<searchPoint.z
    // <<") with K = "<<K<<endl;

    // 开始K近邻搜索
    if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance))
    {
        for (size_t i = 0; i < pointIdxKNNSearch.size(); ++i)
        {
            std::cout << " " << cloud->points[pointIdxKNNSearch[i]].x
                      << " " << cloud->points[pointIdxKNNSearch[i]].y
                      << " " << cloud->points[pointIdxKNNSearch[i]].z
                      << " (squared distance: " << pointKNNSquaredDistance[i] << ")" << std::endl;
        }
        end1 = clock();
        double Times = double(end1 - begin1) / CLOCKS_PER_SEC;
        cout << "time: " << Times << " s " << endl;
    }

    // //使用半径搜索条件搜索
    // vector<int> pointIdxRadiusSearch;
    // vector<float> pointRadiusSquaredDistance;
    // //随机生成一个搜索半径
    // float radius = 256.0f *rand()/(RAND_MAX+1.0f);
    // cout<<"Neighbors within radius search at ("<<searchPoint.x
    // <<" "<<searchPoint.y
    // <<" "<<searchPoint.z
    // <<") with radius = "<<radius<<endl;

    // if(kdtree.radiusSearch(searchPoint,radius,pointIdxRadiusSearch,pointRadiusSquaredDistance)>0)
    // {
    // for(size_t i = 0;i<pointIdxRadiusSearch.size();++i)
    // {
    // cout<<" "<<cloud->points[pointIdxRadiusSearch[i]].x
    // <<" "<<cloud->points[pointIdxRadiusSearch[i]].y
    // <<" "<<cloud->points[pointIdxRadiusSearch[i]].z

    // <<"(squared distance : "<<pointRadiusSquaredDistance[i]<<")"<<endl;
    // }
    // }
}

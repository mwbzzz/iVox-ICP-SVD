
#include <vector>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <../ivox3d/ivox3d.h>

namespace faster_lio
{

    // 使用aligned allocator用于内存对齐
     using PointType = pcl::PointXYZ;
    // using AlignedPointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
    //using PointType = pcl::PointXYZINormal;
    using PointCloudType = pcl::PointCloud<PointType>;
    using CloudPtr = PointCloudType::Ptr;
    using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;

    class LaserMapping
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        IVoxType::Options ivox_options_;
        std::shared_ptr<IVoxType> ivox_ = nullptr; // IVox 对象用于本地地图
    };

} // namespace faster_lio

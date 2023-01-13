#define _CRT_SECURE_NO_WARNINGS

#ifndef PHOTO_STITCH_H_
#define PHOTO_STITCH_H_

#include "photo_stitch.h"
#include "read_video.h"

#define DISTANCE(X1, Y1, X2, Y2) (sqrt(pow(X1 - X2, 2) + pow(Y1 - Y2, 2)))
#define DISTANCELINE(X, Y, A, B) (fabs(A * X + B - Y) / sqrt((pow(A, 2) + 1)))

using namespace cv;
using namespace std;
typedef struct cameraIntrix  // 相机内参
{
  float fx;
  float fy;
  float cx;
  float cy;
} cameraIntrix;

//-- Algorithm implementation for robot locating
class PhotoStitch {
 public:
  PhotoStitch();
  PhotoStitch(const PhotoStitch&) = delete;
  PhotoStitch& operator=(const PhotoStitch&) = delete;
  ~PhotoStitch();

  cameraIntrix intrin;   // 内参
  ReadVideo* ThisVideo;  // 驱动

  void Init(ReadVideo& ReadVideo);
  void UpdateImage(void);   // 更新相机数据
  void ShowImage(void);     // 显示图像
  void ImagePrecess(void);  // 图像预处理
  void Zero(void);          // 归零

  void Fusion(void);

 public:
  // camera parameters
  cv::Mat color_intrin;  // 彩色相机内参

  cv::Mat srcimage;  // 源图像
  cv::Mat dstimage;  // 目标图像
  cv::Mat hsvimage;  // HSV

  //-----------------------------------------------------------
 private:
  cv::Mat afterline;  // 标记图像

  // opencv modules

  cv::Mat thresholdimage;  // 二值图像

  std::string filename;
  std::string path;
};

#endif

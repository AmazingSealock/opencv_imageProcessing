#define _CRT_SECURE_NO_WARNINGS

#ifndef READ_VIDEO_H_
#define READ_VIDEO_H_

#include <math.h>
#include <time.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <thread>

#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 480
#define DEPTH_WIDTH 640
#define DEPTH_HEIGHT 480

using namespace std;
class ReadVideo {
 public:
  ReadVideo();
  ReadVideo(const ReadVideo&) = delete;
  ReadVideo operator=(const ReadVideo&) = delete;
  ~ReadVideo();

  void Init(void);                                       // 初始化
  void Update(void);                                     // 更新图像
  void Recording(void);                                  // 录制视频
  inline cv::Mat GetSrcImage(void) { return srcimage; }  // 获得彩色源图像
  void release(void);

  cv::Mat srcimage;  // 源图片，彩色图片source image
  cv::Mat frame;     // 获取帧

  cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat D = cv::Mat::zeros(5, 1, CV_64F);
  cv::Size imageSize;
  const double alpha = 1;
  cv::Mat NewCameraMatrix;

  double fps = 0;  // 帧率
  int fcount = 0;  // 全部帧数
  int width = 0;   // 获取宽度
  int height = 0;  // 获取高度
  int frameNum = 0;
  int frameindex = 0;

 private:
  cv::VideoCapture capture;
  cv::VideoWriter recorder;

  int startminutes = 0;
  int starseconds = 0;
};
#endif

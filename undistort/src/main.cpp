// //
// // Created by jiang on 2020/4/29.
// //
// #include <iostream>
// #include <opencv2/opencv.hpp>

// using namespace std;

// int main() {
//   // 内参矩阵,
//   const cv::Mat K = (cv::Mat_<double>(3, 3) << 1117.68319584794, 0.0, 812.178871540574, 0.0,
//        1117.55177858882, 1302.27285507742, 0.0, 0.0, 1.0);
//   // 畸变参数，
//   const cv::Mat D = (cv::Mat_<double>(5, 1) << -0.236871042539477, 0.0564436203083370, -0.000389358293005375,
//        -0.000454574911062108, -0.00595989323154578);
//   cv::Mat image_1, image_2;
//   cv::Mat RawImage = cv::imread("../image/PS_49.jpg");
//   // cv::imshow("RawImage", RawImage);
//   const int nImage = 1;
//   const int ImgWidth = 2688;
//   const int ImgHeight = 1520;

//   cv::Mat map1, map2;
//   cv::Size imageSize = RawImage.size();
//   const double alpha = 1;
//   cv::Mat NewCameraMatrix =
//       getOptimalNewCameraMatrix(K, D, imageSize, alpha, imageSize, 0);
//   initUndistortRectifyMap(K, D, cv::Mat(), NewCameraMatrix, imageSize, CV_16SC2,
//                           map1, map2);
//   cv::Mat UndistortImage;
//   remap(RawImage, UndistortImage, map1, map2, cv::INTER_LINEAR);

//   cv::resize(UndistortImage, image_1, cv::Size(760, 1344), 0.5, 0.5);
//   cv::resize(RawImage, image_2, cv::Size(760, 1344), 0.5, 0.5);
//   cv::imshow("UndistortImage", image_1);
//   cv::imshow("rawImage", image_2);

//   cv::imwrite("./1.jpg", UndistortImage);

//   cv::waitKey(0);
  

//   return 0;
// }

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int main()
{
  const cv::Mat K = (cv::Mat_<double>(3, 3) << 1169.16548250471, 0.0, 798.719489323269, 0.0,
       1167.61820642849, 1309.37266840546, 0.0, 0.0, 1.0);
  const cv::Mat D = (cv::Mat_<double>(5, 1) << -0.256370950897256, 0.0654348030304680, -0.000821301815103991,
       0.000539601672148005, -0.00729466604004049);

  const int ImgWidth = 2688;
  const int ImgHeight = 1520;

  cv::Mat map1, map2;
  cv::Size imageSize(ImgWidth, ImgHeight);
  const double alpha = 1;
  cv::Mat NewCameraMatrix =
      getOptimalNewCameraMatrix(K, D, imageSize, alpha, imageSize, 0);


  cv::Mat RawImage = cv::imread("../image/PS_49.jpg");
  

  cv::Mat UndistortImage;
  // cv::undistort(RawImage, UndistortImage, K, D, K);
  cv::undistort(RawImage, UndistortImage, K, D, NewCameraMatrix);
  cv::Mat image_1, image_2;
  cv::resize(UndistortImage, image_1, cv::Size(760, 1344), 0.5, 0.5);
  cv::resize(RawImage, image_2, cv::Size(760, 1344), 0.5, 0.5);
  cv::imshow("UndistortImage", image_1);
  cv::imshow("RawImage", image_2);

  cv::imwrite("./1.jpg", UndistortImage);
  cv::waitKey(0);
  

  return 0;
}
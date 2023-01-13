#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
using namespace std;
using namespace cv;
vector<Mat> imgs, images, gray_image, image_desc;
vector<vector<KeyPoint>> Key_point;
Mat image, img1, img2;
Mat first_match;
int minHessian = 200;

Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);

typedef struct {
  Point2f left_top;
  Point2f left_bottom;
  Point2f right_top;
  Point2f right_bottom;
} four_corners_t;

void image_stiching(void);
Mat grayscale_convert(Mat &image);
vector<KeyPoint> extract_keyPoint(Mat &image);
Mat keyPoint_descriptor(Mat &image, vector<KeyPoint> &Key_point);
vector<vector<Point2f>> keyPoint_match(vector<Mat> &image_desc,
                                       vector<Mat> &images,
                                       vector<vector<KeyPoint>> &Key_point);
vector<Mat> image_match(vector<vector<Point2f>> &image_points,
                        vector<Mat> images, four_corners_t &corners);
four_corners_t calc_corners(const Mat &H, const Mat &src);
void optimize_seam(Mat &images, Mat &trans, Mat &dst, four_corners_t &corners);

int main(int argc, char *argv[]) {
  img1 = imread("../image/IMG_1902.JPG");
  // cv::resize(img1, img1, cv::Size(611, 1088));
  img2 = imread("../image/IMG_1901.JPG");
  // cv::resize(img2, img2, cv::Size(611, 1088));
// 
  images.push_back(img1);
  images.push_back(img2);
  
  // imshow("0",images[0]);
  // imshow("1",images[1]);
  // waitKey(0);

  image_stiching();
  return 0;
}

void image_stiching(void) {
  for (size_t i = 0; i < images.size(); i++) {
    gray_image.push_back(grayscale_convert(images[i]));
    Key_point.push_back(extract_keyPoint(images[i]));
    image_desc.push_back(keyPoint_descriptor(gray_image[i], Key_point[i]));
  }
  // Mat aimage, bimage;
  // drawKeypoints(gray_image[0], Key_point[0], aimage);
  // drawKeypoints(gray_image[1], Key_point[1], bimage);

  // imshow("aimage", aimage);
  // imshow("bimage", bimage);
  // waitKey(0);

  vector<vector<Point2f>> image_points = keyPoint_match(image_desc, images, Key_point);
  four_corners_t corners;
  vector<Mat> match_res = image_match(image_points, images, corners);

  imshow("Directly through the perspective matrix transformation", match_res[0]);
  imwrite("image_transform.jpg", match_res[0]);

  imshow("before_opt_dst", match_res[1]);
  imwrite("before_opt_dst.jpg", match_res[1]);

  optimize_seam(images[1], match_res[0], match_res[1], corners);

  imshow("after_opt_dst", match_res[1]);
  imwrite("after_opt_dst.jpg", match_res[1]);

  waitKey(0);
}

Mat grayscale_convert(Mat &image) {
  // 灰度转换
  Mat gray_image;
  cvtColor(image, gray_image, cv::COLOR_RGB2GRAY);
  return gray_image;
}

vector<KeyPoint> extract_keyPoint(Mat &image) {
  // 提取特征点
  
  vector<KeyPoint> keypoints;
  detector->detect(image, keypoints);
  return keypoints;
}

Mat keyPoint_descriptor(Mat &image, vector<KeyPoint> &Key_point) {
  // 特征点描述，为下边的特征点匹配做准备
  Mat image_desc;
  detector->detectAndCompute(image, noArray(), Key_point, image_desc);
  return image_desc;
}

vector<vector<Point2f>> keyPoint_match(vector<Mat> &image_desc,
                                       vector<Mat> &images,
                                       vector<vector<KeyPoint>> &Key_point) {
  FlannBasedMatcher matcher;
  vector<vector<DMatch>> matchePoints;
  vector<DMatch> GoodMatchePoints;

  vector<Mat> train_desc(1, image_desc[0]);
  matcher.add(train_desc);
  matcher.train();

  matcher.knnMatch(image_desc[1], matchePoints, 2);
  std::cout << "total match points: " << matchePoints.size() << std::endl;

  // Lowe's algorithm,获取优秀匹配点
  for (int i = 0; i < matchePoints.size(); i++) {
    if (matchePoints[i][0].distance < 0.4 * matchePoints[i][1].distance) {
      GoodMatchePoints.push_back(matchePoints[i][0]);
    }
  }

  drawMatches(images[1], Key_point[1], images[0], Key_point[0],
              GoodMatchePoints, first_match);
  // Mat afterresize;
  // cv::resize(first_match, afterresize, cv::Size(first_match.cols / 2, first_match.rows / 2));
  imshow("first_match ", first_match);
  // cout << first_match.cols << "  " << first_match.rows << endl;
  // waitKey(0);
  // imwrite("first_match.jpg", first_match);
  // cout << "finish match" << endl;
  vector<vector<Point2f>> image_points(2);
  for (size_t i = 0; i < GoodMatchePoints.size(); i++) {
    image_points[0].push_back(Key_point[0][GoodMatchePoints[i].trainIdx].pt);
    image_points[1].push_back(Key_point[1][GoodMatchePoints[i].queryIdx].pt);
    // cout << image_points[0][i] << "   " << image_points[1][i] << endl;
  }
  return image_points;
}

vector<Mat> image_match(vector<vector<Point2f>> &image_points,
                        vector<Mat> images, four_corners_t &corners) {    
  // 获取图像1到图像2的投影映射矩阵 尺寸为3*3
  // cout << image_points[0][1] << "  " <<image_points[1][1] << endl;
  Mat homo = findHomography(image_points[0], image_points[1], RANSAC);
  // 也可以使用getPerspectiveTransform方法获得透视变换矩阵，不过要求只能有4个点，效果稍差
  // Mat homo=getPerspectiveTransform(imagePoints1,imagePoints2);
  std::cout << "The transformation matrix is: " << std::endl;
  std::cout << homo << std::endl;
  // 输出映射矩阵
  std::cout << std::endl;
  // 计算配准图的四个顶点坐标
  corners = calc_corners(homo, images[0]);
  std::cout << "left_top: " << corners.left_top << std::endl;
  std::cout << "left_bottom: " << corners.left_bottom << std::endl;
  std::cout << "right_top: " << corners.right_top << std::endl;
  std::cout << "right_bottom: " << corners.right_bottom << std::endl;

  // 图像配准
  Mat image_transform;
  warpPerspective(
      images[0], image_transform, homo,
      Size(MAX(corners.right_top.x, corners.right_bottom.x), images[1].rows));
      // Size(2047, 1364));

  imshow("image_transform", image_transform);
  // waitKey(0);
  cout << "warpPerspective" << endl;
  // 创建拼接后的图,需提前计算图的大小
  int dst_width = image_transform.cols;
  //取最右点的长度为拼接图的长度
  int dst_height = images[1].rows;

  cout << "dst_width:" << dst_width << "   dst_height:" << dst_height << endl;
  Mat dst(dst_height, dst_width, CV_8UC3);
  dst.setTo(0);
  cout << "image_transform.cols=" << image_transform.cols << "   image_transform.rows=" << image_transform.rows << endl;
  image_transform.copyTo(dst(Rect(0, 0, image_transform.cols, image_transform.rows)));
  cout << "images[1].cols=" << images[1].cols << "   images[1].rows=" << images[1].rows << endl;
  images[1].copyTo(dst(Rect(0, 0, images[1].cols, images[1].rows)));
  cout << "copy to" << endl;
  // imshow("before_opt_dst", dst);
  // imwrite("before_opt_dst.jpg", dst);
  vector<Mat> match_res = {image_transform, dst};
  cout << "return match_res;" << endl;
  return match_res;
}

four_corners_t calc_corners(const Mat &H, const Mat &src) {
  four_corners_t corners;
  double v2[] = {0, 0, 1};           // 左上角
  double v1[3];                      // 变换后的坐标值
  Mat V2 = Mat(3, 1, CV_64FC1, v2);  // 列向量
  Mat V1 = Mat(3, 1, CV_64FC1, v1);  // 列向量

  V1 = H * V2;
  // 左上角(0,0,1)
  std::cout << "V2: " << V2 << std::endl;
  std::cout << "V1: " << V1 << std::endl;
  cout << endl;
  corners.left_top.x = v1[0] / v1[2];
  corners.left_top.y = v1[1] / v1[2];

  // 左下角(0,src.rows,1)
  v2[0] = 0;
  v2[1] = src.rows;
  v2[2] = 1;
  V2 = Mat(3, 1, CV_64FC1, v2);
  // 列向量
  V1 = Mat(3, 1, CV_64FC1, v1);
  // 列向量
  V1 = H * V2;
  std::cout << "V2: " << V2 << std::endl;
  std::cout << "V1: " << V1 << std::endl;
  cout << endl;
  corners.left_bottom.x = v1[0] / v1[2];
  corners.left_bottom.y = v1[1] / v1[2];

  // 右上角(src.cols,0,1)
  v2[0] = src.cols;
  v2[1] = 0;
  v2[2] = 1;
  V2 = Mat(3, 1, CV_64FC1, v2);
  // 列向量
  V1 = Mat(3, 1, CV_64FC1, v1);
  // 列向量
  V1 = H * V2;
  std::cout << "V2: " << V2 << std::endl;
  std::cout << "V1: " << V1 << std::endl;
  cout << endl;
  corners.right_top.x = v1[0] / v1[2];
  corners.right_top.y = v1[1] / v1[2];

  // 右下角(src.cols,src.rows,1)
  v2[0] = src.cols;
  v2[1] = src.rows;
  v2[2] = 1;
  V2 = Mat(3, 1, CV_64FC1, v2);
  // 列向量
  V1 = Mat(3, 1, CV_64FC1, v1);
  // 列向量
  V1 = H * V2;
  std::cout << "V2: " << V2 << std::endl;
  std::cout << "V1: " << V1 << std::endl;
  cout << endl;
  corners.right_bottom.x = v1[0] / v1[2];
  corners.right_bottom.y = v1[1] / v1[2];

  return corners;
}


// 优化两图的连接处，使得拼接自然
void optimize_seam(Mat &images, Mat &trans, Mat &dst, four_corners_t &corners) {
  int start = MIN(corners.left_top.x, corners.left_bottom.x);
  // 开始位置，即重叠区域的左边界

  double processWidth = images.cols - start;
  // 重叠区域的宽度
  int rows = dst.rows;
  int cols = images.cols;
  // 注意，是列数*通道数
  double alpha = 1;
  // images中像素的权重
  for (int i = 0; i < rows; i++) {
    uchar *p = images.ptr<uchar>(i);
    // 获取第i行的首地址
    uchar *t = trans.ptr<uchar>(i);
    uchar *d = dst.ptr<uchar>(i);
    for (int j = start; j < cols; j++) {
      // 如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
      if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0) {
        alpha = 1;
      } else {
        // img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好
        alpha = (processWidth - (j - start)) / processWidth;
      }

      d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
      d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
      d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);
    }
  }
}

#endif
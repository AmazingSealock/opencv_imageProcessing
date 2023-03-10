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

// #define ONE2ONE
// #define TREE
#define CONNECT

vector<Mat> imgs, images, temp_image, image_desc;
vector<vector<KeyPoint>> Key_point;
Mat image, img1, img2, img3, img4, img5;
Mat first_match;
int minHessian = 200;
string path;
const int translength = 50;
//1=45 2=38 3=25 4=100 5=65 6=35 7=45 8=55 9=37 11= 12=


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
Mat connect_derectly(Mat transimage, Mat addimage);
Mat rotate(Mat src, double angle);
Mat roll(Mat src, double angle);

int main(int argc, char *argv[]) {
  for(int i=323; i<352; i++){
    string new_name;
    stringstream ss;
    ss << i;
    path = "../image/12";
    string filename = new_name.assign(path).append("/") + string("TC_") +
                      string(ss.str() + ".jpg");
    Mat tempimage = imread(filename);
    images.push_back(tempimage);
  }
  std::cout << images.size() << std::endl;
  image_stiching();
  return 0;
}

void image_stiching(void) {

  #ifdef TREE
  temp_image = images;
  while(temp_image.size()>1){
    images.clear();
    images = temp_image;
    temp_image.clear();
    Key_point.clear();
    image_desc.clear();
    // std::cout << "images = : " << images.size() << std::endl;
    for (size_t i = 0; i < images.size(); i++) {
      Key_point.push_back(extract_keyPoint(images[i]));
      image_desc.push_back(keyPoint_descriptor(images[i], Key_point[i]));
    }

    for(int i=0; i<images.size()-1; i=i+2){
      vector<Mat> image_desc_2;
      image_desc_2.push_back(image_desc[i]);
      image_desc_2.push_back(image_desc[i+1]);
      vector<vector<KeyPoint>> Key_point_2;
      Key_point_2.push_back(Key_point[i]);
      Key_point_2.push_back(Key_point[i+1]);
      vector<Mat> images_2;
      images_2.push_back(images[i]);
      images_2.push_back(images[i+1]);
      // imshow("x",images_2[0]);
      // imshow("y",images_2[1]);
      vector<vector<Point2f>> image_points = keyPoint_match(image_desc_2, images_2, Key_point_2);
      four_corners_t corners;
      vector<Mat> match_res = image_match(image_points, images_2, corners);
      // cv::imshow("Directly through the perspective matrix transformation", match_res[0]);
      // cv::imwrite("image_transform.jpg", match_res[0]);

      // cv::imshow("before_opt_dst", match_res[1]);
      // cv::imwrite("before_opt_dst.jpg", match_res[1]);

      optimize_seam(images_2[1], match_res[0], match_res[1], corners);
      temp_image.push_back(match_res[1]);
      // cv::imshow("after_opt_dst", match_res[1]);
      // cv::waitKey(1);
      // cv::imwrite("after_opt_dst.jpg", match_res[1]);
      std::cout << "tempImage = : " << temp_image.size() << std::endl;
      std::cout << "------------------" << std::endl;
    }
  }
  cv::imshow("temp_image", temp_image[0]);
  imwrite("finalImage.jpg", temp_image[0]);
  #endif


  #ifdef ONE2ONE
  for (size_t i = 0; i < images.size(); i++) {
    // gray_image.push_back(grayscale_convert(images[i]));
    Key_point.push_back(extract_keyPoint(images[i]));
    image_desc.push_back(keyPoint_descriptor(images[i], Key_point[i]));
  }

  for(int i=0; i<images.size()-1; i++){
    vector<Mat> image_desc_2;
    image_desc_2.push_back(image_desc[i]);
    image_desc_2.push_back(image_desc[i+1]);
    vector<vector<KeyPoint>> Key_point_2;
    Key_point_2.push_back(Key_point[i]);
    Key_point_2.push_back(Key_point[i+1]);
    vector<Mat> images_2;
    images_2.push_back(images[i]);
    images_2.push_back(images[i+1]);
    // imshow("x",images_2[0]);
    // imshow("y",images_2[1]);
    vector<vector<Point2f>> image_points = keyPoint_match(image_desc_2, images_2, Key_point_2);
    four_corners_t corners;
    vector<Mat> match_res = image_match(image_points, images_2, corners);
    imshow("Directly through the perspective matrix transformation", match_res[0]);
    imwrite("image_transform.jpg", match_res[0]);

    imshow("before_opt_dst", match_res[1]);
    imwrite("before_opt_dst.jpg", match_res[1]);

    optimize_seam(images_2[1], match_res[0], match_res[1], corners);

    imshow("after_opt_dst", match_res[1]);
    imwrite("after_opt_dst.jpg", match_res[1]);

    Key_point[i+1] = extract_keyPoint(match_res[1]);
    image_desc[i+1] = keyPoint_descriptor(match_res[1], Key_point[i+1]);
    images[i+1] = match_res[1];
  
    waitKey(0);

    std::cout << "------------------" << std::endl;
  }
  #endif

  #ifdef CONNECT
  Mat rightimage, leftimage; //????????????????????????????????????????????????
  Mat transferimage;
  int addwidth = 0, addheight = 0;
  rightimage = images[0].clone();
  Point2f AffinePointsSrc[4] = { Point2f(0.f, 0.f), Point2f(rightimage.cols-1.f,0), Point2f(rightimage.cols-1.f,rightimage.rows-1.f),  Point2f(0.f, rightimage.rows-1.f) };
  Point2f AffinePointsDst[4] = { Point2f(0.f, 0.f), Point2f(rightimage.cols-1.f,35),Point2f(rightimage.cols-1.f,rightimage.rows-35.f),  Point2f(0.f, rightimage.rows-1.f) };
		// ????????????????????????
  Mat TransImage = getPerspectiveTransform(AffinePointsSrc, AffinePointsDst);
  Mat warp_dst = Mat::zeros( rightimage.rows, rightimage.cols, rightimage.type() );
  warpPerspective(rightimage, rightimage, TransImage, Size(rightimage.cols, rightimage.rows), INTER_CUBIC);

  // dstTri[0] = Point2f( 0.f, 0.f );
  // dstTri[1] = Point2f( rightimage.cols - 1, 0.f);
  // dstTri[2] = Point2f( 0.f, rightimage.rows - 1.f);
  
  // imshow("input", rightimage);
  // imshow("output", warp_dst);
  // cv::waitKey(0);
  
  for (size_t i = 0; i < images.size()-1; i++) {
    leftimage  = images[i+1].clone();
    warpPerspective(leftimage, warp_dst, TransImage, Size(rightimage.cols, rightimage.rows), INTER_CUBIC);
    // imshow("warp_dst", warp_dst);
    addwidth = rightimage.cols + translength;
    addheight = rightimage.rows;
    transferimage = cv::Mat::zeros(cv::Size(addwidth, addheight), CV_8UC3);
    rightimage.copyTo(transferimage(Rect(translength, 0, rightimage.cols, rightimage.rows))); //??????????????????????????????????????????????????????
    warp_dst(Rect(0, 0, translength, warp_dst.rows)).copyTo(transferimage(Rect(0, 0, translength, rightimage.rows)));
    rightimage = transferimage.clone();
    // waitKey(0);
  }
  cv::imshow("transImage", transferimage);
  imwrite("final_image.jpg", transferimage);


  #endif
  
  cv::waitKey(0);
  
}

Mat grayscale_convert(Mat &image) {
  // ????????????
  Mat gray_image;
  cvtColor(image, gray_image, cv::COLOR_RGB2GRAY);
  return gray_image;
}

vector<KeyPoint> extract_keyPoint(Mat &image) {
  // ???????????????
  
  vector<KeyPoint> keypoints;
  detector->detect(image, keypoints);
  return keypoints;
}

Mat keyPoint_descriptor(Mat &image, vector<KeyPoint> &Key_point) {
  // ??????????????????????????????????????????????????????
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

  // Lowe's algorithm,?????????????????????
  for (int i = 0; i < matchePoints.size(); i++) {
    if (matchePoints[i][0].distance < 0.7 * matchePoints[i][1].distance) {
      GoodMatchePoints.push_back(matchePoints[i][0]);
    }
  }
  cout << " GoodMatchePoints " << GoodMatchePoints.size() << endl;
  drawMatches(images[1], Key_point[1], images[0], Key_point[0],
              GoodMatchePoints, first_match);
  // Mat afterresize;
  // cv::resize(first_match, afterresize, cv::Size(first_match.cols / 2, first_match.rows / 2));
  // imshow("first_match ", first_match);
  // cout << first_match.cols << "  " << first_match.rows << endl;
  // waitKey(0);
  // imwrite("first_match.jpg", first_match);
  if(GoodMatchePoints.size() < 100){
    Mat afterresize;
    cv::resize(first_match, afterresize, cv::Size(first_match.cols / 2, first_match.rows / 2));
    imshow("first_match ", afterresize);
    waitKey(0);
  }
  cout << "finish match" << endl;
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
  // ????????????1?????????2????????????????????? ?????????3*3
  // cout << image_points[0][1] << "  " <<image_points[1][1] << endl;
  Mat homo = findHomography(image_points[0], image_points[1], RANSAC);
  // homo.at<double>(1,0) = 0.f;
  // homo.at<double>(0,1) = 0.f;
  // homo.at<double>(2,0) = 0.f;
  // homo.at<double>(2,1) = 0.f;
  // ???????????????getPerspectiveTransform??????????????????????????????????????????????????????4?????????????????????
  // Mat homo=getPerspectiveTransform(imagePoints1,imagePoints2);
  // std::cout << "The transformation matrix is: " << std::endl;
  // std::cout << homo << std::endl;
  
  // ??????????????????
  // std::cout << std::endl;
  // ????????????????????????????????????k
  corners = calc_corners(homo, images[0]);
  // std::cout << "left_top: " << corners.left_top << std::endl;
  // std::cout << "left_bottom: " << corners.left_bottom << std::endl;
  // std::cout << "right_top: " << corners.right_top << std::endl;
  // std::cout << "right_bottom: " << corners.right_bottom << std::endl;

  // ????????????
  Mat image_transform;
  warpPerspective(
      images[0], image_transform, homo,
      Size(MAX(corners.right_top.x, corners.right_bottom.x), images[1].rows));
      // Size(2047, 1364));

  // imshow("image_transform", image_transform);
  // waitKey(0);
  // cout << "warpPerspective" << endl;
  // ?????????????????????,???????????????????????????
  int dst_width = image_transform.cols;
  //??????????????????????????????????????????
  int dst_height = images[1].rows;

  // cout << "dst_width:" << dst_width << "   dst_height:" << dst_height << endl;
  Mat dst(dst_height, dst_width, CV_8UC3);
  dst.setTo(0);
  // cout << "image_transform.cols=" << image_transform.cols << "   image_transform.rows=" << image_transform.rows << endl;
  image_transform.copyTo(dst(Rect(0, 0, image_transform.cols, image_transform.rows)));
  // cout << "images[1].cols=" << images[1].cols << "   images[1].rows=" << images[1].rows << endl;
  images[1].copyTo(dst(Rect(0, 0, images[1].cols, images[1].rows)));
  // cout << "copy to" << endl;
  // imshow("before_opt_dst", dst);
  // imwrite("before_opt_dst.jpg", dst);
  vector<Mat> match_res = {image_transform, dst};
  // cout << "return match_res;" << endl;
  return match_res;
}

four_corners_t calc_corners(const Mat &H, const Mat &src) {
  four_corners_t corners;
  double v2[] = {0, 0, 1};           // ?????????
  double v1[3];                      // ?????????????????????
  Mat V2 = Mat(3, 1, CV_64FC1, v2);  // ?????????
  Mat V1 = Mat(3, 1, CV_64FC1, v1);  // ?????????

  V1 = H * V2;
  // ?????????(0,0,1)
  // std::cout << "V2: " << V2 << std::endl;
  // std::cout << "V1: " << V1 << std::endl;
  // cout << endl;
  corners.left_top.x = v1[0] / v1[2];
  corners.left_top.y = v1[1] / v1[2];

  // ?????????(0,src.rows,1)
  v2[0] = 0;
  v2[1] = src.rows;
  v2[2] = 1;
  V2 = Mat(3, 1, CV_64FC1, v2);
  // ?????????
  V1 = Mat(3, 1, CV_64FC1, v1);
  // ?????????
  V1 = H * V2;
  // std::cout << "V2: " << V2 << std::endl;
  // std::cout << "V1: " << V1 << std::endl;
  // cout << endl;
  corners.left_bottom.x = v1[0] / v1[2];
  corners.left_bottom.y = v1[1] / v1[2];

  // ?????????(src.cols,0,1)
  v2[0] = src.cols;
  v2[1] = 0;
  v2[2] = 1;
  V2 = Mat(3, 1, CV_64FC1, v2);
  // ?????????
  V1 = Mat(3, 1, CV_64FC1, v1);
  // ?????????
  V1 = H * V2;
  // std::cout << "V2: " << V2 << std::endl;
  // std::cout << "V1: " << V1 << std::endl;
  // cout << endl;
  corners.right_top.x = v1[0] / v1[2];
  corners.right_top.y = v1[1] / v1[2];

  // ?????????(src.cols,src.rows,1)
  v2[0] = src.cols;
  v2[1] = src.rows;
  v2[2] = 1;
  V2 = Mat(3, 1, CV_64FC1, v2);
  // ?????????
  V1 = Mat(3, 1, CV_64FC1, v1);
  // ?????????
  V1 = H * V2;
  // std::cout << "V2: " << V2 << std::endl;
  // std::cout << "V1: " << V1 << std::endl;
  // cout << endl;
  corners.right_bottom.x = v1[0] / v1[2];
  corners.right_bottom.y = v1[1] / v1[2];

  return corners;
}


// ?????????????????????????????????????????????
void optimize_seam(Mat &images, Mat &trans, Mat &dst, four_corners_t &corners) {
  int start = MIN(corners.left_top.x, corners.left_bottom.x);
  // ??????????????????????????????????????????

  double processWidth = images.cols - start;
  // ?????????????????????
  int rows = dst.rows;
  int cols = images.cols;
  // ??????????????????*?????????
  double alpha = 1;
  // images??????????????????
  for (int i = 0; i < rows; i++) {
    uchar *p = images.ptr<uchar>(i);
    // ?????????i???????????????
    uchar *t = trans.ptr<uchar>(i);
    uchar *d = dst.ptr<uchar>(i);
    for (int j = start; j < cols; j++) {
      // ??????????????????trans???????????????????????????????????????img1????????????
      if (t[j * 3] <= 10 && t[j * 3 + 1] <= 10 && t[j * 3 + 2] <= 10) {
        alpha = 1;
      } else {
        // img1????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        alpha = (processWidth - (j - start)) / processWidth;
      }
      // alpha = 1;

      d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
      d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
      d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);
    }
  }
}

Mat connect_derectly(Mat transimage, Mat addimage){
  
}

Mat rotate(Mat src, double angle)   //rotate function returning mat object with parametres imagefile and angle    
{
    Mat dst;      //Mat object for output image file
    Point2f pt(src.cols/2., src.rows/2.);          //point from where to rotate    
    Mat r = getRotationMatrix2D(pt, angle, 1.0);      //Mat object for storing after rotation
    warpAffine(src, dst, r, Size(src.cols, src.rows));  ///applie an affine transforation to image.
    return dst;         //returning Mat object for output image file
}

Mat roll(Mat src, double angle)   //rotate function returning mat object with parametres imagefile and angle    
{
    Mat dst;      //Mat object for output image file
    Point2f pt(src.cols/2., src.rows/2.);          //point from where to rotate    
    Mat r = getRotationMatrix2D(pt, angle, 1.0);      //Mat object for storing after rotation
    warpAffine(src, dst, r, Size(src.cols, src.rows));  ///applie an affine transforation to image.
    return dst;         //returning Mat object for output image file
}

#endif
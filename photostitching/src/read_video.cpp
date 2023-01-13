#include "read_video.h"

// #define RECORDVIDEO
// #define RECORDIMAGE
#define VIDEO
#define IMAGE

ReadVideo::ReadVideo() {}

void ReadVideo::Init(void) {

  string filename = "../video/ch1_20230111145904_20230111150515.mp4";
  // string filename = "../record/test_save.avi";
  capture.open(filename);
  if (!capture.isOpened()) {
    throw "Error when reading steam_MP4";
    getchar();
    return;
  }

  fps = capture.get(cv::CAP_PROP_FPS);              // 帧率
  fcount = capture.get(cv::CAP_PROP_FRAME_COUNT);   // 全部帧数
  width = capture.get(cv::CAP_PROP_FRAME_WIDTH);    // 获取宽度
  height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);  // 获取高度
  cout << "total height is: " << height << endl;
  cout << "total width is: " << width << endl;
  cout << "total frame is: " << fcount << endl;      // 全部帧数
  cout << "total sec is: " << fcount / fps << endl;  // 总时间
  cout << "fps is: " << fps << endl;
  cout << "please press any key to start... " << endl;
  getchar();

  // //内参矩阵,(fx,fy,cx,cy)
  K = ( cv::Mat_<double> ( 3,3 ) << 1169.16548250471, 0.0, 798.719489323269, 0.0, 1167.61820642849, 1309.37266840546, 0.0, 0.0, 1.0 );
  // //畸变参数，(k1,k2,r1,r2,r3)
  D = ( cv::Mat_<double> ( 5,1 ) << -0.256370950897256, 0.0654348030304680, -0.000821301815103991, 0.000539601672148005, -0.00729466604004049);

  // cv::Mat view, rview, map1, map2;
  
  imageSize = cv::Size(height, width);
  NewCameraMatrix = getOptimalNewCameraMatrix(K, D, imageSize, alpha, imageSize, 0);

  // 设置视频开始时间
  startminutes = 1;
  starseconds = 0;
  capture.set(cv::CAP_PROP_POS_MSEC,
              (startminutes * 60) * 1000 + starseconds * 1000);
#ifdef RECORDVIDEO
  // 设置视频的有效帧数，输入的数字是秒，fps是帧率，即计算需要从原视频裁减的帧数是28*fps
  frameNum = 5 * fps;

  // 初始化需保存的视频格式
  int encode_type = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
  bool isColor = true;
  string recordfilename = "../record/test_save.avi";
  recorder = cv::VideoWriter(recordfilename, encode_type, fps,
                             cv::Size(width, height), isColor);
  cout << "recording begin ..." << endl;
#endif

  cout << "Video init done ..\n";
}

void ReadVideo::Update() {
  capture >> frame;
  cv::Mat UndistortImage;
  // cv::undistort(frame, UndistortImage, K, D, K);
  cv::undistort(frame, UndistortImage, K, D, NewCameraMatrix);
#ifdef RECORDVIDEO
  Recording();
#endif
  cv::resize(UndistortImage, srcimage, cv::Size(width / 2, height / 2), 0.5, 0.5);
}

void ReadVideo::Recording() {
  frameindex++;
  if (frameindex >= frameNum) {
    frameindex = frameNum;
    return;
  } else {
    recorder << frame;
  }
}

void ReadVideo::release(void) {}

ReadVideo::~ReadVideo() {}

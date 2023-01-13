#include "photo_stitch.h"

#ifdef VIDEO
VideoWriter writer("../record/src.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'),
                   25.0, Size(640, 480));
#endif

// std::ofstream outfile("../record/test.txt");
PhotoStitch::PhotoStitch() {}

void PhotoStitch::Init(ReadVideo &RVideo) {
  srcimage = cv::Mat::zeros(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3);
  hsvimage = cv::Mat::zeros(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3);

  std::cout << "Initializing stitcher..." << std::endl;

  //-- Set input device
  ThisVideo = &RVideo;
  for (int i = 0; i < 3; i++) {
    ThisVideo->Update();
  }
}

void PhotoStitch::UpdateImage(void) {
  ThisVideo->Update();
  srcimage = ThisVideo->GetSrcImage().clone();
}

void PhotoStitch::Fusion(void) {
  ShowImage();
  Zero();
}

void PhotoStitch::ImagePrecess(void) {
  afterline = srcimage.clone();
  cv::cvtColor(srcimage, hsvimage, cv::COLOR_BGR2HSV);
}

//----------------------------将要发的数归零
void PhotoStitch::Zero(void) {}

void PhotoStitch::ShowImage(void) {
  cv::imshow("src", srcimage);
  char keyinput = cv::waitKey(10);
  if (keyinput == 'r') {
    // if(1){
    static int count = 0;
    string new_name;
    stringstream ss;
    ss << count;
    count++;
    path = "../record/4";
    string filename = new_name.assign(path).append("/") + string("PS2_") +
                      string(ss.str() + ".jpg");
    imwrite(filename, ThisVideo->frame);
  }
}

PhotoStitch::~PhotoStitch() {}

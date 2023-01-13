// 创建时间：2023/1/5
// 创建者：Jie

#include "photo_stitch.h"
#include "read_video.h"

using namespace std;

int main(int argc, char* argv[]) {
  // 初始化
  ReadVideo CVideo;  // 相机驱动模块实例化
  CVideo.Init();
  PhotoStitch Stitcher;  // 机器人定位模块
  Stitcher.Init(CVideo);
  cv::TickMeter tk;
  while (true) {
    tk.start();
    Stitcher.UpdateImage();
    Stitcher.Fusion();
    tk.stop();
    // cout << " 程序总时间: " << tk.getTimeMilli() << "\n" << endl;
    tk.reset();
  }
  return 0;
}

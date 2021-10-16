#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include "cuda_draw.h"

constexpr int width = 640;
constexpr int height = 480;

void printError() {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "Cuda error: " << cudaGetErrorString(err);
  }
}

int main(int argc, char **argv)
{
  FLAGS_alsologtostderr = true;

  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  cv::Mat img(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

  auto d_img = cudraw::uploadImage(width, height, (uint8_t*)img.data);

  // Test - draw string
  cudraw::String string(250, 200, "Hello world!", 12);
  string.color = cudraw::Color(0x00ff00);
  string.font_size = 24;
  cudraw::scanPixel(width, height, d_img, string);
  printError();

  // Test - draw rect
  cudraw::Rect rect(100, 300, 150, 400, 8);
  rect.color = cudraw::Color(0x0000ff);
  cudraw::scanPixel(width, height, d_img, rect);
  printError();

  // Test - fill rect
  cudraw::Rect rect_2(200, 150, 320, 250, 8);
  rect_2.color = cudraw::Color(0xff00ff);
  rect_2.fill_color = cudraw::Color(0x00ffff);
  rect_2.fill = true;
  rect_2.fill_alpha = 0.75;
  cudraw::scanPixel(width, height, d_img, rect_2);
  printError();

  // Test - draw line
  cudraw::Line line(0, 0, 400, 300, 5);
  line.color = cudraw::Color(0xff0000);
  cudraw::scanPixel(width, height, d_img, line);
  printError();

  cudraw::downloadImage(width, height, d_img, img.data);

  cudaFree(d_img);

  cv::imshow("Result", img);
  cv::waitKey(0);

  return 0;
}

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include "cuda_draw.h"

constexpr int width = 640;
constexpr int height = 480;

int main(int argc, char **argv)
{
  FLAGS_alsologtostderr = true;

  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  cv::Mat img(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

  auto d_img = cudraw::uploadImage(width, height, (uint8_t*)img.data);

  cudraw::Line line(0, 0, 200, 200, 5);
  line.color = cudraw::Color(0xff0000);
  cudraw::scanPixel(width, height, d_img, line);

  cudraw::downloadImage(width, height, d_img, img.data);

  cudaFree(d_img);

  cv::imshow("Result", img);
  cv::waitKey(0);

  return 0;
}

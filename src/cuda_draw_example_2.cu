#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>

#include <unistd.h>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include "cuda_draw.h"

#define RW r_width(gen)
#define RH r_height(gen)
#define RT r_line_width(gen)
#define RC r_color(gen)
#define RF r_font(gen)

constexpr int width = 640;
constexpr int height = 480;
constexpr float line_width_max = 32.f;
constexpr float font_min = 4.f;
constexpr float font_max = 72.f;

void testDraw(size_t num) {
  LOG(INFO) << "Testing drawing size: " << num;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> r_width(0, width);
  std::uniform_int_distribution<> r_height(0, height);
  std::uniform_int_distribution<> r_color(0, 256);
  std::uniform_real_distribution<> r_line_width(0, line_width_max);
  std::uniform_real_distribution<> r_font(font_min, font_max);

  std::vector<cudraw::Line>line_pool(num);
  std::vector<cudraw::Rect>rect_pool(num);
  std::vector<cudraw::String>string_pool(num);

  auto start = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::high_resolution_clock::now() - start);

  LOG(INFO) << "Creating shapes...";
  for (size_t i = 0; i < num; i++) {
    line_pool[i] = cudraw::Line(RW, RH, RW, RH, RT);
    line_pool[i].color.r = RC;
    line_pool[i].color.g = RC;
    line_pool[i].color.b = RC;

    rect_pool[i] = cudraw::Rect(RW, RH, RW, RH, RT);
    rect_pool[i].color.r = RC;
    rect_pool[i].color.g = RC;
    rect_pool[i].color.b = RC;

    string_pool[i] = cudraw::String(RW, RH, "Hello world!", 12);
    string_pool[i].color.r = RC;
    string_pool[i].color.g = RC;
    string_pool[i].color.b = RC;
    string_pool[i].font_size = RF;
  }

  cv::Mat img_1(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
  cv::Mat img_2(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

  LOG(INFO) << "Creating shapes done!";

  LOG(INFO) << "Drawing Cuda...";
  start = std::chrono::high_resolution_clock::now();

  auto d_img = cudraw::uploadImage(width, height, (uint8_t*)img_1.data);
  for (size_t i = 0; i < num; i++) {
    cudraw::scanPixel(width, height, d_img, line_pool[i]);
    cudraw::scanPixel(width, height, d_img, rect_pool[i]);
    cudraw::scanPixel(width, height, d_img, string_pool[i]);
  }
  cudraw::downloadImage(width, height, d_img, img_1.data);
  cudaFree(d_img);
  
  duration = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now() - start);
  LOG(INFO) << "Drawing Cuda done! - Time used (including CPU measurement): " 
            << duration.count() << " us";

  LOG(INFO) << "Drawing OpenCV...";
  start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < num; i++) {
    cv::line(img_2, 
             cv::Point(line_pool[i].x0, line_pool[i].y0), 
             cv::Point(line_pool[i].x1, line_pool[i].y1), 
             cv::Scalar(line_pool[i].color.r, 
                        line_pool[i].color.g, 
                        line_pool[i].color.b),
             line_pool[i].width, 
             cv::LINE_8);
    cv::rectangle(img_2, 
             cv::Rect(rect_pool[i].x0, 
                      rect_pool[i].y0,
                      rect_pool[i].x1 - rect_pool[i].x0,
                      rect_pool[i].y1 - rect_pool[i].y0), 
             cv::Scalar(rect_pool[i].color.r, 
                        rect_pool[i].color.g, 
                        rect_pool[i].color.b),
             rect_pool[i].width, 
             cv::LINE_8,
             0);
    cv::putText(img_2,
                "Hello world!",
                cv::Point(string_pool[i].x, string_pool[i].y),
                cv::FONT_HERSHEY_PLAIN,
                string_pool[i].font_size / 16.f,
                cv::Scalar(string_pool[i].color.r, 
                           string_pool[i].color.g, 
                           string_pool[i].color.b),
                string_pool[i].font_size / 8.f);
  }
  
  duration = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now() - start);
  LOG(INFO) << "Drawing Cuda done! - Time used (including CPU measurement): " 
      << duration.count() << " us";

  cv::imshow("Result Cuda", img_1);
  cv::imshow("Result OpenCV", img_2);
  cv::waitKey(0);
}

int main(int argc, char **argv)
{
  FLAGS_alsologtostderr = true;

  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  testDraw(10);
  testDraw(100);
  testDraw(1000);
  testDraw(10000);
  testDraw(100000);

  return 0;
}

#ifndef __CUDA_DRAW__
#define __CUDA_DRAW__

#include <glog/logging.h>
#include <opencv2/opencv.hpp>

namespace cudraw {

// Utilities

// Support U8C3 images only
uint8_t* allocateImage(size_t width, size_t height);

uint8_t* uploadImage(size_t width, size_t height, uint8_t* img);

void uploadImage(size_t width, size_t height, uint8_t* img, uint8_t* d_img);

void downloadImage(size_t width, size_t height, uint8_t* d_img, uint8_t* img);

void copyImage(uint8_t* d_dst, uint8_t* d_src, size_t size);

void freeImage(uint8_t* d_img);

// API

struct Color {
  Color() = default;
  __device__ __host__ Color(
      unsigned char rr, unsigned char gg, unsigned char bb) {
    r = rr;
    g = gg;
    b = bb;
  }
  __device__ __host__ Color(unsigned int hex) {
    b = hex & 0x000000ff;
    hex >>= 16;
    g = hex & 0x000000ff;
    hex >>= 16;
    r = hex & 0x000000ff;
  }

  __device__ __host__ void overlay(const Color& other, float alpha) {
    if (alpha < 0) {
      alpha = 0;
    }
    if (alpha > 1) {
      alpha = 1;
    }
    r = (float)r * (1.f - alpha) + (float)other.r * alpha;
    g = (float)g * (1.f - alpha) + (float)other.g * alpha;
    b = (float)b * (1.f - alpha) + (float)other.b * alpha; 
  }

  unsigned char r = 0;
  unsigned char g = 0;
  unsigned char b = 0;
};

struct AABB {
  AABB() = default;
  __device__ __host__ AABB(int x_0, int y_0, int x_1, int y_1) {
    x0 = x_0;
    y0 = y_0;
    x1 = x_1;
    y1 = y_1;
  }

  __device__ __host__ void inflate(int i) {
    x0 -= i;
    x1 += i;
    y0 -= i;
    y1 += i;
  }

  __device__ __host__ bool inside(int x, int y) {
    return x >= x0 && x <= x1 && y >= y0 && y <= y1;
  }

  int x0 = 0;
  int y0 = 0;
  int x1 = 0;
  int y1 = 0;
};

struct Shape {
  Shape() = default;

  __device__ virtual Color checkPixel(int x, int y, Color color) = 0;
  __device__ __host__ virtual AABB getAABB() const = 0;
};

struct Line : public Shape {
  Line() = default;
  __device__ __host__ Line(int x_0, int y_0, int x_1, int y_1, float w) {
    x0 = x_0;
    y0 = y_0;
    x1 = x_1;
    y1 = y_1;
    width = w;
  }

  __device__ Color checkPixel(int x, int y, Color prev_color) override;
  __device__ __host__ AABB getAABB() const override;

  int x0 = 0;
  int y0 = 0;
  int x1 = 0;
  int y1 = 0;
  float width = 1;

  Color color;
};


void scanPixel(size_t width, size_t height, uint8_t* d_img, const Line& line);

}

#endif

#ifndef CUBEMAPGEN_IMAGEUTILS_H
#define CUBEMAPGEN_IMAGEUTILS_H

#include <vector>
#include <cmath>

#include "types.h"


float32 calcLuminance(Eigen::Ref<const vec3f> c);

float32 calcLuminance(float32 r, float32 g, float32 b);


bool isJpeg(const char* src);

bool isHdr(const char* src);

bool isPng(const char* src);

bool isExr(const char* src);


bool decompressJpeg(const char* src, size_t len, std::vector<uint8>& dst, int& width, int& height);

bool decompressHdr(const char* src, size_t len, std::vector<float32>& dst, int& width, int& height);

bool decompressExr(const char* src, size_t len, std::vector<float16>& dst, int& width, int& height);

bool decompress(const char* src, size_t len, std::vector<float32>& dst, int& width, int& height);


bool compressJpeg(std::vector<float32>& src, std::vector<uint8>& dst, int width, int height, int quality);

bool compressHdr(std::vector<float32>& src, std::vector<uint8>& dst, int width, int height);


float32 computeEV(const std::vector<float32>& src);

float32 computeEV(const std::vector<std::vector<float32>>& src);


void applyExposure(std::vector<float32>& src, float32 ev);

void applyExposure(const std::vector<float32>& src, std::vector<float32>& dst, float32 ev);

void applyTonemap(std::vector<float32>& src);


void encodeRGBM(Eigen::Ref<const vec3f> src, Eigen::Ref<rgba32> dst, float32 exp = 0.0f);

void encodeRGBM(const std::vector<float32>& src, std::vector<uint8>& dst, float32 exp);

void encodeLogLuv(Eigen::Ref<const vec3f> src, Eigen::Ref<rgba32> dst);

void encodeLogLuv(const std::vector<float32>& src, std::vector<uint8>& dst);


void dropAlpha(const std::vector<float32>& src, std::vector<uint8>& dst);


#endif //CUBEMAPGEN_IMAGEUTILS_H

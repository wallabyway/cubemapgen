#include "dds.h"
#include "imageutils.h"
#include <cmath>

void logLuvDDSFromFloats(const std::vector<std::vector<float32>>& src, std::vector<uint8>& dst) {
  // does not have mip levels
  auto faces = (uint32)src.size();
  auto faceSize = (uint32)std::sqrt(src[0].size() / 4);
  DDS_HEADER header;
  header.height = header.width = faceSize;

  uint32 totalDataSize = 0;
  std::vector<std::vector<uint8>> data;
  data.resize(faces);
  Eigen::Map<rgba32> pixel(nullptr);
  std::vector<uint8> logluv;
  for (auto fi = 0; fi < faces; ++fi) {
    encodeLogLuv(src[fi], logluv);
    data[fi].resize(logluv.size());
    for (auto i = 0; i < logluv.size(); i += 4) {
      new (&pixel) Eigen::Map<rgba32>(&data[fi][i]);
      pixel << logluv[i+2], logluv[i+1], logluv[i], logluv[i+3];
    }
    totalDataSize += data[fi].size();
  }
  dst.resize(sizeof(DDS_HEADER) + totalDataSize);
  uint32 off = 0;
  memcpy(&dst[off], &header, sizeof(DDS_HEADER));
  off += sizeof(DDS_HEADER);
  for (auto fi = 0; fi < faces; ++fi) {
    memcpy(&dst[off], &data[fi][0], sizeof(uint8) * data[fi].size());
    off += sizeof(uint8) * data[fi].size();
  }
}

void logLuvDDSFromFloats(const std::vector<std::vector<std::vector<float32>>>& src, std::vector<uint8>& dst) {
  //does have mip levels
  auto levels = (uint32)src.size();
  auto faces = (uint32)src[0].size();
  auto firstFaceSize = (uint32)std::sqrt(src[0][0].size() / 4);
  DDS_HEADER header;
  header.height = header.width = firstFaceSize;
  header.mipMapCount = levels;
  header.flags |= DDS_HEADER_FLAGS::DDSD_MIPMAPCOUNT;
  header.caps |= DDS_CAPS::DDSCAPS_MIPMAP;

  uint32 totalDataSize = 0;
  std::vector<std::vector<std::vector<uint8>>> data;
  data.resize(levels);
  Eigen::Map<rgba32> pixel(nullptr);
  std::vector<uint8> logluv;
  for (auto li = 0; li < levels; ++li) {
    data[li].resize(faces);
    uint32 levelDataSize = 0;
    for (auto fi = 0; fi < faces; ++fi) {
      encodeLogLuv(src[li][fi], logluv);
      data[li][fi].resize(logluv.size());
      for (auto i = 0; i < logluv.size(); i += 4) {
        new (&pixel) Eigen::Map<rgba32>(&data[li][fi][i]);
        pixel << logluv[i+2], logluv[i+1], logluv[i], logluv[i+3];
      }
      levelDataSize += data[li][fi].size();
    }
    totalDataSize += levelDataSize;
  }
  dst.resize(sizeof(DDS_HEADER) + totalDataSize);
  uint32 off = 0;
  memcpy(&dst[off], &header, sizeof(DDS_HEADER));
  off += sizeof(DDS_HEADER);
  for (auto fi = 0; fi < faces; ++fi) {
    for (auto li = 0; li < levels; ++li) {
      memcpy(&dst[off], &data[li][fi][0], sizeof(uint8) * data[li][fi].size());
      off += sizeof(uint8) * data[li][fi].size();
    }
  }
}

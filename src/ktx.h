#ifndef CUBEMAPGEN_KTX_H
#define CUBEMAPGEN_KTX_H

#include "types.h"
#include <vector>

#define KTX_IDENTIFIER_REF  { 0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A }
#define KTX_ENDIAN_REF      (0x04030201)

#define GL_FLOAT                          0x1406
#define GL_RGB                            0x1907
#define GL_RGBA                           0x1908
#define GL_COMPRESSED_RGBA_S3TC_DXT5_EXT  0x83F3
#define GL_RGBA32F                        0x8814
#define GL_RGB32F                         0x8815

enum PIXEL_ENCODING {
  RGBE,
  RGBM,
  LOGLUV
};

typedef struct {
  uint8 identifier[12];
  uint32 endianness;
  uint32 glType;
  uint32 glTypeSize;
  uint32 glFormat;
  uint32 glInternalFormat;
  uint32 glBaseInternalFormat;
  uint32 pixelWidth;
  uint32 pixelHeight;
  uint32 pixelDepth;
  uint32 numberOfArrayElements;
  uint32 numberOfFaces;
  uint32 numberOfMipmapLevels;
  uint32 bytesOfKeyValueData;
} ktxHeader;




void ktxFromFloats(const std::vector<std::vector<std::vector<float32>>>& src, std::vector<uint8>& dst, float32 exp, bool8 compress = false);







#endif //CUBEMAPGEN_KTX_H





#include "imageutils.h"

#include <iostream>
#include <string>
#include <stdexcept>
#include <xmmintrin.h>
#include <algorithm>

#include <turbojpeg.h>
#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfIO.h>
#include <OpenEXR/ImfArray.h>

typedef unsigned char RGBE[4];
typedef OPENEXR_IMF_INTERNAL_NAMESPACE::RgbaInputFile RgbaInputFile;
typedef OPENEXR_IMF_INTERNAL_NAMESPACE::Rgba Rgba;
typedef IMATH_NAMESPACE::Box2i Box2i;

static const mat3f LOGLUV_ENCODING_MATRIX = [] {
  mat3f tmp;
  tmp << 0.2209f, 0.3390f, 0.4184f,
         0.1138f, 0.6780f, 0.7319f,
         0.0102f, 0.1130f, 0.2969f;
  tmp.transposeInPlace();
  return tmp;
}();

static const vec3f LUM_FACTORS(0.2126f, 0.7152f, 0.0722f);

class MemoryFile {
public:
  enum MFSEEK {
    MFSEEK_SET, MFSEEK_CUR
  };

  explicit MemoryFile(const char* src, const size_t length) : len(length), src((unsigned char*)src), start((unsigned char*)src) {}

  explicit MemoryFile(const std::string& string) : len(string.length()), src((unsigned char*)string.c_str()), start((unsigned char*)string.c_str()) {}

  unsigned char getc() {
    if (src < start + len) {
      unsigned char c = *src;
      src++;
      return c;
    } else {
      throw std::range_error("cannot read beyond end of file");
    }
  }

  void seek(long int offset, MFSEEK origin) {
    if (origin == MFSEEK_SET) {
      if (offset < 0) throw std::range_error("cannot seek to before start of file");
      if (offset > len) throw std::range_error("cannot seek beyond end of file");
      src = start + offset;
    } else {
      if (src + offset < start) throw std::range_error("cannot seek to before start of file");
      if (src + offset > start + len) throw std::range_error("cannot seek beyond end of file");
    }
  }

  bool eof() {
    return src == start + len;
  }

private:
  unsigned char* src;
  unsigned char* start;
  const size_t len;
};

class MemoryStream : public OPENEXR_IMF_INTERNAL_NAMESPACE::IStream {
  typedef OPENEXR_IMF_INTERNAL_NAMESPACE::Int64 Int64;
public:
  MemoryStream(const char* src, const size_t length) : OPENEXR_IMF_INTERNAL_NAMESPACE::IStream("temp"), len(length), src((char*)src), start((char*)src) {}

  bool isMemoryMapped() const override { return true; }

  bool read(char c[/*n*/], int n) override {
    if (src + n > start + len) throw std::range_error("attempt to read beyond end of file");
    memcpy(c, src, static_cast<size_t>(n));
    src += n;
    return src < start + len;
  }

  char* readMemoryMapped(int n) override {
    if (src + n > start + len) throw std::range_error("attempt to read beyond end of file");
    char* r = src;
    src += n;
    return r;
  }

  Int64 tellg() override {
    return (Int64)(src - start);
  }

  void seekg(Int64 pos) override {
    if (pos < 0 || pos > len) throw std::range_error("attempt to seek beyond boundaries of file");
    src = start + pos;
  }

  void clear() override {}

  size_t length() { return len; }

private:
  char* src;
  char* start;
  const size_t len;
};


// exported

float32 calcLuminance(Eigen::Ref<const vec3f> c) {
  return c.dot(LUM_FACTORS);
}

float32 calcLuminance(const float32 r, const float32 g, const float32 b) {
  vec3f rgb(r, g, b);
  return calcLuminance(rgb);
}

// internal

// TODO: Eigenize these

uint8 applyGamma(float32 v) {
  if (v <= 0.0031308f) {
    v *= 12.92;
  } else {
    v = (1.0f + 0.055f) * pow(v, 1.0f / 2.4f) - 0.055f;
  }
  if (v > 1.0f) v = 1.0f;
  if (v < 0.0f) v = 0.0f;
  return (uint8)round(255.0f * v);
}

bool oldDecrunch(RGBE* scanline, long len, MemoryFile& srcFile) {
  int i;
  int rshift = 0;
  while (len > 0) {
    scanline[0][0] = srcFile.getc();
    scanline[0][1] = srcFile.getc();
    scanline[0][2] = srcFile.getc();
    scanline[0][3] = srcFile.getc();
    if (srcFile.eof()) return false;
    if (scanline[0][0] == 1 && scanline[0][1] == 1 && scanline[0][2] == 1) {
      for (i = scanline[0][3] << rshift; i > 0; i--) {
        memcpy(&scanline[0][0], &scanline[-1][0], 4);
        scanline++;
        len--;
      }
      rshift += 8;
    } else {
      scanline++;
      len--;
      rshift = 0;
    }
  }
  return true;
}

float convertComponent(int expo, int val) {
  float v = val / 256.0f;
  float d = (float)pow(2, expo);
  return v * d;
}

void workOnRGBE(RGBE* scan, long len, float* cols) {
  while (len-- > 0) {
    int expo = scan[0][3] - 128;
    cols[0] = convertComponent(expo, scan[0][0]);
    cols[1] = convertComponent(expo, scan[0][1]);
    cols[2] = convertComponent(expo, scan[0][2]);
    cols += 3;
    scan++;
  }
}

bool decrunch(RGBE* scanline, long len, MemoryFile& srcFile) {
  int i, j;
  if (len < 0x08 || len > 0x7fff) return oldDecrunch(scanline, len, srcFile);
  i = srcFile.getc();
  if (i != 2) {
    srcFile.seek(-1, MemoryFile::MFSEEK_CUR);
    return oldDecrunch(scanline, len, srcFile);
  }
  scanline[0][1] = srcFile.getc();
  scanline[0][2] = srcFile.getc();
  i = srcFile.getc();
  if (scanline[0][1] != 2 || scanline[0][2] & 128) {
    scanline[0][0] = 2;
    scanline[0][3] = i;
    return oldDecrunch(scanline + 1, len - 1, srcFile);
  }
  for (i = 0; i < 4; i++) {
    for (j = 0; j < len;) {
      unsigned char code = srcFile.getc();
      if (code > 128) {
        code &= 127;
        unsigned char val = srcFile.getc();
        while (code--) scanline[j++][i] = val;
      } else {
        while (code--) {
          scanline[j++][i] = srcFile.getc();
        }
      }
    }
  }
  return !srcFile.eof();
}

float32 smoothStep(const float32 edge0, const float32 edge1, const float32 x) {
  float32 t = std::min(1.0f, std::max(0.0f, (x - edge0) / (edge1 - edge0)));
  return t * t * (3.0f - 2.0f * t);
}

// exported

bool isJpeg(const char* src) {
  const unsigned char magic[2] = {0xff, 0xd8};
  return !memcmp(src, magic, sizeof(magic));
}

bool isHdr(const char* src) {
  const unsigned char magic[10] = {0x23, 0x3f, 0x52, 0x41, 0x44, 0x49, 0x41, 0x4e, 0x43, 0x45};
  return !memcmp(src, magic, sizeof(magic));
}

bool isPng(const char* src) {
  const unsigned char magic[8] = {0x89, 0x59, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
  return !memcmp(src, magic, sizeof(magic));
}

bool isExr(const char* src) {
  const unsigned char magic[4] = {0x76, 0x2f, 0x31, 0x01};
  return !memcmp(src, magic, sizeof(magic));
}

bool decompressJpeg(const char* src, const size_t len, std::vector<uint8>& dst, int& width, int& height) {
  int res, samp, colorSpace;
  tjhandle dec = tjInitDecompress();
  res = tjDecompressHeader3(dec, (uint8*)src, len, &width, &height, &samp, &colorSpace);
  if (res != 0) {
    std::cerr << tjGetErrorStr() << std::endl;
    tjDestroy(dec);
    return false;
  }
  auto outsize = (size_t)width * height * 3;
  unsigned char buffer[outsize];
  res = tjDecompress2(dec, (uint8*)src, len, buffer, width, 0, height, TJPF_RGB, TJFLAG_FASTDCT);
  if (res != 0) {
    std::cerr << tjGetErrorStr() << std::endl;
    tjDestroy(dec);
    return false;
  }
  dst.resize(outsize);
  dst.assign(buffer, buffer + outsize);
  tjDestroy(dec);
  return true;
}

bool decompressHdr(const char* src, const size_t len, std::vector<float32>& dst, int& width, int& height) {
  auto srcFile = MemoryFile(src, len);
  srcFile.seek(11, MemoryFile::MFSEEK_SET);
  char cmd[200];
  int i = 0;
  char c = 0, oldc;
  while (true) {
    oldc = c;
    c = srcFile.getc();
    if (c == 0x0a && oldc == 0x0a) break;
    cmd[i++] = c;
  }
  char reso[200];
  i = 0;
  while (true) {
    c = srcFile.getc();
    reso[i++] = c;
    if (c == 0x0a) break;
  }
  long longh, longw;
  if (!sscanf(reso, "-Y %ld +X %ld", &longh, &longw)) {
    std::cerr << "Could not parse hdr resolution" << std::endl;
    return false;
  }
  height = (int)longh;
  width = (int)longw;
  auto* cols = new float[width * height * 3];
  float* start = cols;
  auto* scanline = new RGBE[width];
  for (long y = height - 1; y >= 0; y--) {
    if (!decrunch(scanline, width, srcFile)) break;
    workOnRGBE(scanline, width, cols);
    cols += width * 3;
  }
  dst.resize((uint64)width * height * 4);
  for (auto i = 0, j = 0; i < (width * height * 3); i += 3, j += 4) {
    dst[j] = start[i];
    dst[j+1] = start[i+1];
    dst[j+2] = start[i+2];
    dst[j+3] = 1.0f;
  }
  delete[] scanline;
  return true;
}

bool decompressExr(const char* src, const size_t len, std::vector<float16>& dst, int& width, int& height) {
  MemoryStream ms(src, len);
  RgbaInputFile file(ms);
  Box2i dw = file.dataWindow();
  width = dw.max.x - dw.min.x + 1;
  height = dw.max.y - dw.min.y + 1;
  uint64 nValues = (uint64)width * height * 4;
  dst.resize(nValues);
  file.setFrameBuffer((Rgba*)(&dst[0] - dw.min.x - dw.min.y * width), 1, (size_t)width);
  file.readPixels(dw.min.y, dw.max.y);
  return true;
}

bool decompress(const char* src, const size_t len, std::vector<float32>& dst, int& width, int& height) {
  if (isJpeg(src)) {
    std::vector<uint8> tmp;
    bool res = decompressJpeg(src, len, tmp, width, height);
    if (res) {
      dst.resize(tmp.size() * 4 / 3);
      for (int i = 0, j = 0; i < tmp.size(); ++i, ++j) {
        dst[j] = (float32)tmp[i] / 255.0f;
        if (i % 3 == 2) {
          dst[++j] = (float32)1.0f;
        }
      }
    }
    return res;
  } else if (isHdr(src)) {
    return decompressHdr(src, len, dst, width, height);
  } else if (isExr(src)) {
    std::vector<float16> tmp;
    bool res = decompressExr(src, len, tmp, width, height);
    if (res) {
      dst.resize(tmp.size());
      for (int i = 0; i < tmp.size(); ++i) {
        dst[i] = (float32)tmp[i];
      }
    }
    return res;
  } else if (isPng(src)) {
    return false;
  } else {
    return false;
  }
}


bool compressJpeg(std::vector<float32>& src, std::vector<uint8>& dst, const int width, const int height, const int quality) {
  std::vector<uint8> srgb;
  srgb.resize(src.size());
  for (size_t i = 0; i < src.size(); i++) {
    if (i % 4 == 3) {
      srgb[i] = 0xff;
    } else {
      srgb[i] = applyGamma(src[i]);
    }
  }
  tjhandle jpegCompressor = tjInitCompress();
  unsigned char* jpeg = nullptr;
  long unsigned int jpegSize;
  int res = tjCompress2(jpegCompressor, &srgb[0], width, 0, height, TJPF_RGBA, &jpeg, &jpegSize, TJSAMP_444, quality, TJFLAG_FASTDCT);
  if (res != 0) std::cerr << tjGetErrorStr() << std::endl;
  if (res != 0) return false;
  tjDestroy(jpegCompressor);
  dst.resize(jpegSize);
  dst.assign(jpeg, jpeg + jpegSize);
  tjFree(jpeg);
  return true;
}

bool compressHdr(std::vector<float32>& src, std::vector<uint8>& dst, const int width, const int height) {
  return false;
}


float32 computeEV(const std::vector<float32>& src) {
  float32 avgLogLum = 0.0f;
  uint32 validPixels = 0;
  float32 lumThresh = 10e-5f;
  for (int i = 0; i < src.size(); i += 4) {
    float32 lum = calcLuminance(src[i], src[i+1], src[i+2]);
    if (lum > lumThresh && lum < INFINITY) {
      avgLogLum += log2(lum);
      validPixels++;
    }
  }
  return (float32)log2(0.18) - (validPixels > 0 ? avgLogLum / validPixels : 0);
}

float32 computeEV(const std::vector<std::vector<float32>>& src) {
  float32 avgLogLum = 0.0f;
  uint32 validPixels = 0;
  float32 lumThresh = 10e-5f;
  for (auto face : src) {
    for (int i = 0; i < face.size(); i += 4) {
      float32 lum = calcLuminance(face[i], face[i+1], face[i+2]);
      if (lum > lumThresh && lum < INFINITY) {
        avgLogLum += log2(lum);
        validPixels++;
      }
    }
  }
  return (float32)log2(0.18) - (validPixels > 0 ? avgLogLum / validPixels : 0);
}


void applyExposure(std::vector<float32>& src, const float32 ev) {
  const auto ef = (float32)pow(2.0, ev);
  const float efv[4] = {ef, ef, ef, 1.0f};
  const __m128 f = _mm_load_ps((float32*)&efv);
  for (int i = 0; i < src.size(); i += 4) {
    __m128 p = _mm_load_ps((float32*)&src[i]) * f;
    _mm_store_ps((float32*)&src[i], p);
  }
}

void applyExposure(const std::vector<float32>& src, std::vector<float32>& dst, const float32 ev) {
  dst = src;
  applyExposure(dst, ev);
}

void applyTonemap(std::vector<float32>& src) {

  // params for 'vivid' tonemap
  // todo: parameterize these

  const float32 burnHighlights  = 0.0f;
  const float32 crushBlacks     = 2.13543794f;
  const float32 midtones        = 0.63596080f;
  const float32 preMultiply     = 5.55161853f;
  const float32 postMultiply    = 1.08926343f;
  const float32 saturation      = 1.0f;
  const bool    colorPreserving = true;

  const float32 MIDDLE_GRAY    = 0.18f;

  float32 cbo = crushBlacks * 2.0f + 1.0f;
  float32 imt = 1.0f / midtones;

  auto curve = [preMultiply, burnHighlights, postMultiply, cbo, imt](float32 x) {
    float32 t = x * preMultiply;
    t = (t * (1.0f + (t * burnHighlights))) / (1.0f + t) * postMultiply;
    if (cbo > 1) {
      float32 intensity = t > 0.0f ? sqrt(t) : 0.0f;
      if (intensity < 1.0f) {
        t = t * intensity + pow(t, cbo) * (1.0f - intensity);
      }
    }
    return imt == 1.0f ? t : pow(t, imt);
  };

  for (int i = 0; i < src.size(); i += 4) {
    vec3f rgbPixel(src[i], src[i+1], src[i+2]);
    if (colorPreserving) {
      float32 lum = calcLuminance(rgbPixel);
      float32 compLum = curve(lum);
      float32 blendAlpha = smoothStep(2.0f * MIDDLE_GRAY, 0.9, compLum);
      if (lum != 0.0f) {
        vec3f compColor(curve(rgbPixel(0)), curve(rgbPixel(1)), curve(rgbPixel(2)));
        rgbPixel = (rgbPixel * compLum / lum) * (1 - blendAlpha) + compColor * blendAlpha;
      }
    } else {
      rgbPixel << curve(rgbPixel(0)), curve(rgbPixel(1)), curve(rgbPixel(2));
    }
    if (saturation != 1.0f) {
      float32 lum = calcLuminance(rgbPixel);
      float32 invSat = 1.0f - saturation;
      rgbPixel = ((rgbPixel * saturation).array() + lum * invSat).array().max(0.0f).matrix();
    }
    rgbPixel = rgbPixel.array().max(0.0f).min(1.0f).matrix();
    src[i] = rgbPixel(0);
    src[i+1] = rgbPixel(1);
    src[i+2] = rgbPixel(2);
  }

}


void encodeRGBM(Eigen::Ref<const vec3f> src, Eigen::Ref<rgba32> dst, const float32 exp) {
  vec3f t = (src * std::pow(2.0f, exp)).cwiseSqrt() * 0.0625f;
  float32 w = std::ceil(std::min(1.0f, std::max(t.maxCoeff(), 1e-6f)) * 255.0f) / 255.0f;
  t = t.cwiseMin(1.0f) / w;
  dst.block<3, 1>(0, 0) = (255.0f * t).array().floor().matrix().cast<uint8>();
  dst(3) = (uint8)std::floor(255.0f * w);
}

void encodeRGBM(const std::vector<float32>& src, std::vector<uint8>& dst, const float32 exp) {
  // src is expected to have four channels; alpha will be ignored;
  Eigen::Map<const vec3f> rgb(nullptr);
  Eigen::Map<rgba32> rgbm(nullptr);
  dst.resize(src.size());
  for (auto i = 0; i < src.size(); i += 4) {
    new (&rgb) Eigen::Map<const vec3f>(&src[i]);
    new (&rgbm) Eigen::Map<rgba32>(&dst[i]);
    encodeRGBM(rgb, rgbm, exp);
  }
}

void encodeLogLuv(Eigen::Ref<const vec3f> src, Eigen::Ref<rgba32> dst) {
  vec3f xyz = (LOGLUV_ENCODING_MATRIX * src);//.cwiseMax(1e-6f);
  float32 le = 2.0f * std::log2(xyz(1)) + 127.0f;
  float32 tmp, a = std::modf(le, &tmp);
  vec4f lle ((xyz(0) / xyz(2)),
             (xyz(1) / xyz(2)),
             (le - (std::floor(a * 255.0f)) / 255.0f) / 255.0f,
             (a));
  dst = (lle * 255.0f).array().floor().matrix().cast<uint8>();
}

void encodeLogLuv(const std::vector<float32>& src, std::vector<uint8>& dst) {
  // src is expected to have four channels; alpha will be ignored;
  Eigen::Map<const vec3f> rgb(nullptr);
  Eigen::Map<rgba32> lluv(nullptr);
  dst.resize(src.size());
  for (auto i = 0; i < src.size(); i += 4) {
    new (&rgb) Eigen::Map<const vec3f>(&src[i]);
    new (&lluv) Eigen::Map<rgba32>(&dst[i]);
    encodeLogLuv(rgb, lluv);
  }
}


void dropAlpha(const std::vector<float32>& src, std::vector<uint8>& dst) {
  dst.resize(3 * src.size());
  for (auto i = 0, j = 0; i < src.size(); i += 4, j += 12) {
    memcpy(&dst[j], &src[i], 12);
  }
}

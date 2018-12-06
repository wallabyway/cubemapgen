#include "ktx.h"

#include <cmath>

#include "imageutils.h"

static uint8 expand5[32];
static uint8 expand6[64];
static uint8 oMatch5[256][2];
static uint8 oMatch6[256][2];
static uint8 quantRBTab[256+16];
static uint8 quantGTab[256+16];

void extractBlock(const std::vector<float32>& src, const int u, const int v, const uint32 faceSize, const float32 exp, std::vector<uint8>& block) {
  int i, j, bx, by;
  int bw = std::min((int)faceSize - u, 4);
  int bh = std::min((int)faceSize - v, 4);
  vec3f rgb;
  rgba32 enc;
  const int rem[] = {
      0, 0, 0, 0,
      0, 1, 0, 1,
      0, 1, 2, 0,
      0, 1, 2, 3
  };
  block.resize(4 * 4 * 4);
  for (i = 0; i < 4; ++i) {
    by = rem[(bh - 1) * 4 + i] + v;
    for (j = 0; j < 4; ++j) {
      bx = rem[(bw - 1) * 4 + j] + u;
      rgb << src[(by * (faceSize * 4)) + (bx * 4) + 0],
             src[(by * (faceSize * 4)) + (bx * 4) + 1],
             src[(by * (faceSize * 4)) + (bx * 4) + 2];
      encodeRGBM(rgb, enc, exp);
      block[(i * 4 * 4) + (j * 4) + 0] = enc(0);
      block[(i * 4 * 4) + (j * 4) + 1] = enc(1);
      block[(i * 4 * 4) + (j * 4) + 2] = enc(2);
      block[(i * 4 * 4) + (j * 4) + 3] = enc(3);
    }
  }
}

static int mul8bit(int a, int b) {
  int t = a * b + 128;
  return (t + (t >> 8)) >> 8;
}

static int lerp13(int a, int b) {
  return ((2 * a + b) * 0xaaab) >> 17;
}

static void prepareTable(uint8* table, const uint8* expand, int size) {
  int i, mn, mx;
  for (i = 0; i < 256; ++i) {
    int bestErr = 256;
    for (mn = 0; mn < size; ++mn) {
      for (mx = 0; mx < size; ++mx) {
        int mine = expand[mn];
        int maxe = expand[mx];
        int err = std::abs(lerp13(maxe, mine) - i);
        err += std::abs(maxe - mine) * 3 / 100;
        if (err < bestErr) {
          table[i * 2 + 0] = (uint8)mx;
          table[i * 2 + 1] = (uint8)mn;
          bestErr = err;
        }
      }
    }
  }
}

static void initDXT() {
  int i, v;
  for (i = 0; i < 32; ++i) {
    expand5[i] = (uint8)((i<<3)|(i>>2));
  }
  for (i = 0; i < 64; ++i) {
    expand6[i] = (uint8)((i<<2)|(i>>4));
  }
  for (i = 0; i < 256 + 16; ++i) {
    v = i - 8 < 0 ? 0 : i - 8 > 255 ? 255 : i - 8;
    quantRBTab[i] = expand5[mul8bit(v, 31)];
    quantGTab[i] = expand6[mul8bit(v, 63)];
  }
  prepareTable(&oMatch5[0][0], expand5, 32);
  prepareTable(&oMatch6[0][0], expand6, 64);
}

static void compressDXT5AlphaBlock(const std::vector<uint8>& block, std::vector<uint8>& dst, uint32& dstOffset) {
  int32 i, dist, bias, dist4, dist2, bits, mask, mn, mx;
  mn = mx = block[3];
  for (i = 1; i < 16; ++i) {
    if (block[i * 4 + 3] < mn) mn = block[i * 4 + 3];
    else if (block[i * 4 + 3] > mx) mx = block[i * 4 + 3];
  }
  dst[dstOffset++] = (uint8)mx;
  dst[dstOffset++] = (uint8)mn;
  dist = mx - mn;
  dist4 = dist * 4;
  dist2 = dist * 2;
  bias = (dist < 8) ? (dist - 1) : (dist / 2 + 2);
  bias -= mn * 7;
  bits = 0, mask = 0;
  for (i = 0; i < 16; ++i) {
    int a = block[i * 4 + 3] * 7 + bias;
    int ind, t;
    t = (a >= dist4) ? -1 : 0; ind  = t & 4; a -= dist4 & t;
    t = (a >= dist2) ? -1 : 0; ind += t & 2; a -= dist2 & t;
    ind += (a >= dist);
    ind = -ind & 7;
    ind ^= (2 > ind);
    mask |= ind << bits;
    if ((bits += 3) >= 8) {
      dst[dstOffset++] = (uint8)mask;
      mask >>= 8;
      bits -= 8;
    }
  }
}

static uint16 as16Bit(int r, int g, int b) {
  return (uint16)((mul8bit(r, 31) << 11) + (mul8bit(g, 63) << 5) + mul8bit(b, 31));
}

static void optimizeColors(const std::vector<uint8>& block, uint16& max16, uint16& min16) {

  int ch, i, vr, vg, vb, minp, maxp, mu[3], min[3], max[3], cov[6];
  float32 vfr, vfg, vfb, covf[6];
  float64 magn;

  for (ch = 0; ch < 3; ++ch) {
    int muv, minv, maxv;
    muv = minv = maxv = block[ch];
    for (i = 4; i < 64; i += 4) {
      muv += block[ch + i];
      if (block[ch + i] < minv) minv = block[ch + i];
      else if (block[ch + i] > maxv) maxv = block[ch + i];
    }
    mu[ch] = (muv + 8) >> 4;
    min[ch] = minv;
    max[ch] = maxv;
  }
  for (i = 0; i < 6; ++i) {
    cov[i] = 0;
  }
  for (i = 0; i < 16; ++i) {
    int r = block[i * 4 + 0] - mu[0];
    int g = block[i * 4 + 1] - mu[1];
    int b = block[i * 4 + 2] - mu[2];
    cov[0] += r * r;
    cov[1] += r * g;
    cov[2] += r * b;
    cov[3] += g * g;
    cov[4] += g * b;
    cov[5] += b * b;
  }
  for (i = 0; i < 6; ++ i) {
    covf[i] = cov[i] / 255.0f;
  }
  vfr = (float32)(max[0] - min[0]);
  vfg = (float32)(max[1] - min[1]);
  vfb = (float32)(max[2] - min[2]);
  for (i = 0; i < 4; ++i) {
    float r = vfr * covf[0] + vfg * covf[1] + vfb * covf[2];
    float g = vfr * covf[1] + vfg * covf[3] + vfb * covf[4];
    float b = vfr * covf[2] + vfg * covf[4] + vfb * covf[5];
    vfr = r;
    vfg = g;
    vfb = b;
  }
  magn = std::fabs(vfr);
  if (std::fabs(vfg) > magn) magn = std::fabs(vfg);
  if (std::fabs(vfb) > magn) magn = std::fabs(vfb);
  if (magn < 4.0f) {
    vr = 299;
    vg = 587;
    vb = 114;
  } else {
    magn = 512.0 / magn;
    vr = (int)(vfr * magn);
    vg = (int)(vfg * magn);
    vb = (int)(vfb * magn);
  }
  int mind = 0x7fffffff, maxd = -0x7fffffff;
  for (i = 0; i < 16; ++i) {
    int dot = block[i * 4 + 0] * vr + block[i * 4 + 1] * vg + block[i * 4 + 2] * vb;
    if (dot < mind) {
      mind = dot;
      minp = i * 4;
    }
    if (dot > maxd) {
      maxd = dot;
      maxp = i * 4;
    }
  }
  max16 = as16Bit(block[maxp + 0], block[maxp + 1], block[maxp + 2]);
  min16 = as16Bit(block[minp + 0], block[minp + 1], block[minp + 2]);
}

static void from16Bit(uint8* out, uint16 v) {
  int rv = (v & 0xf800) >> 11;
  int gv = (v & 0x07e0) >>  5;
  int bv = (v & 0x001f) >>  0;
  out[0] = expand5[rv];
  out[1] = expand6[gv];
  out[2] = expand5[bv];
  out[3] = 0;
}

static void lerp13RGB(uint8* out, uint8* p1, uint8* p2) {
  out[0] = (uint8)lerp13(p1[0], p2[0]);
  out[1] = (uint8)lerp13(p1[1], p2[1]);
  out[2] = (uint8)lerp13(p1[2], p2[2]);
}

static void evalColors(uint8* color, uint16 c0, uint16 c1) {
  from16Bit(color+ 0, c0);
  from16Bit(color+ 4, c1);
  lerp13RGB(color+ 8, color+0, color+4);
  lerp13RGB(color+12, color+4, color+0);
}

static uint32 matchColorsBlock(const std::vector<uint8>& block, uint8* color) {
  uint32 mask = 0;
  int32 dirr = color[0 * 4 + 0] = color[1 * 4 + 0];
  int32 dirg = color[0 * 4 + 1] = color[1 * 4 + 1];
  int32 dirb = color[0 * 4 + 2] = color[1 * 4 + 2];
  int32 dots[16];
  int32 stops[4];
  int32 i;
  int32 c0Point, halfPoint, c3Point;
  for (i = 0; i < 16; ++i) {
    dots[i] = block[i * 4 + 0] * dirr + block[i * 4 + 1] * dirg + block[i * 4 + 2] * dirb;
  }
  for (i = 0; i < 4; ++i) {
    stops[i] = color[i * 4 + 0] * dirr + color[i * 4 + 1] * dirg + color[i * 4 + 2] * dirb;
  }
  c0Point   = (stops[1] + stops[3]) >> 1;
  halfPoint = (stops[3] + stops[2]) >> 1;
  c3Point   = (stops[2] + stops[0]) >> 1;
  const int32 indexMap[8] = { 0 << 30, 2 << 30, 0 << 30, 2 << 30,
                              3 << 30, 3 << 30, 1 << 30, 1 << 30};
  for (i = 0; i < 16; ++i) {
    mask >>= 2;
    int32 bits = ((dots[i] < halfPoint) ? 4 : 0) |
                 ((dots[i] < c0Point)   ? 2 : 0) |
                 ((dots[i] < c3Point)   ? 1 : 0);
    mask |= indexMap[bits];
  }
  return mask;
}

inline static int32 sclamp(float32 y, int32 p0, int32 p1) {
  auto x = (int32)y;
  if (x < p0) return p0;
  if (x > p1) return p1;
  return x;
}

static int32 refineBlock(const std::vector<uint8>& block, uint16& max16, uint16& min16, uint32 mask) {
  static const int32 w1Tab[4] = { 3, 0, 2, 1 };
  static const int32 prods[4] = { 0x090000, 0x000900, 0x040102, 0x010402 };
  float32 frb, fg;
  uint16 oldMin, oldMax;
  int32 i, akku = 0, xx, xy, yy, at1r, at1g, at1b, at2r, at2g, at2b;
  uint32 cm = mask;
  oldMin = min16;
  oldMax = max16;
  if ((mask ^ (mask << 2)) < 4) {
    int r = 8, g = 8, b = 8;
    for (i = 0; i < 16; ++i) {
      r += block[i * 4 + 0];
      g += block[i * 4 + 1];
      b += block[i * 4 + 2];
    }
    r >>= 4; g >>= 4; b >>= 4;
    max16 = (oMatch5[r][0] << 11) | (oMatch6[g][0] << 5) | oMatch5[b][0];
    min16 = (oMatch5[r][1] << 11) | (oMatch6[g][1] << 5) | oMatch5[b][1];
  } else {
    at1r = at1g = at1b = 0;
    at2r = at2g = at2b = 0;
    for (i = 0; i < 16; ++i, cm >>= 2) {
      int32 step = cm & 3;
      int32 w1 = w1Tab[step];
      int32 r = block[i * 4 + 0];
      int32 g = block[i * 4 + 1];
      int32 b = block[i * 4 + 2];
      akku += prods[step];
      at1r += w1 * r;
      at1g += w1 * g;
      at1b += w1 * b;
      at2r += r;
      at2g += g;
      at2b += b;
    }
    at2r = 3 * at2r - at1r;
    at2g = 3 * at2g - at1g;
    at2b = 3 * at2b - at1b;
    xx = akku >> 16;
    yy = (akku >> 8) & 0xff;
    xy = (akku >> 0) & 0xff;
    frb = 3.0f * 31.0f / 255.0f / (xx * yy - xy * xy);
    fg = frb * 63.0f / 31.0f;
    max16  = (uint16)sclamp((at1r * yy - at2r * xy) * frb + 0.5f, 0, 31) << 11;
    max16 |= (uint16)sclamp((at1g * yy - at2g * xy) * fg  + 0.5f, 0, 63) << 5;
    max16 |= (uint16)sclamp((at1b * yy - at2b * xy) * frb + 0.5f, 0, 31) << 0;
    min16  = (uint16)sclamp((at2r * xx - at1r * xy) * frb + 0.5f, 0, 31) << 11;
    min16 |= (uint16)sclamp((at2g * xx - at1g * xy) * fg  + 0.5f, 0, 63) << 5;
    min16 |= (uint16)sclamp((at2b * xx - at1b * xy) * frb + 0.5f, 0, 31) << 0;
  }
  return oldMin != min16 || oldMax != max16;
}

static void compressDXT5ColorBlock(const std::vector<uint8>& block, std::vector<uint8>& dst, uint32& dstOffset) {
  uint32 mask;
  uint8 color[4 * 4];
  int i, refineCount = 2;
  uint16 max16, min16;
  for (i = 1; i < 16; ++i) {
    if (block[i] != block[0]) break;
  }
  if (i == 16) {
    // constant color
    int r = block[0], g = block[1], b = block[2];
    mask = 0xaaaaaaaa;
    max16 = (oMatch5[r][0] << 11) | (oMatch6[g][0] << 5) | oMatch5[b][0];
    min16 = (oMatch5[r][1] << 11) | (oMatch6[g][1] << 5) | oMatch5[b][1];
  } else {
    optimizeColors(block, max16, min16);
    if (max16 != min16) {
      evalColors(color, max16, min16);
      mask = matchColorsBlock(block, color);
    } else {
      mask = 0;
    }
    for (i = 0; i < refineCount; ++i) {
      uint32 lastMask = mask;
      if (refineBlock(block, max16, min16, mask)) {
        if (max16 != min16) {
          evalColors(color, max16, min16);
          mask = matchColorsBlock(block, color);
        } else {
          mask = 0;
          break;
        }
      }
      if (mask == lastMask) {
        break;
      }
    }
  }
  if (max16 < min16) {
    uint16 t = min16;
    min16 = max16;
    max16 = t;
    mask ^= 0x55555555;
  }
  dst[dstOffset++] = (uint8)(max16);
  dst[dstOffset++] = (uint8)(max16 >> 8);
  dst[dstOffset++] = (uint8)(min16);
  dst[dstOffset++] = (uint8)(min16 >> 8);
  dst[dstOffset++] = (uint8)(mask);
  dst[dstOffset++] = (uint8)(mask >> 8);
  dst[dstOffset++] = (uint8)(mask >> 16);
  dst[dstOffset++] = (uint8)(mask >> 24);
}

void compressDXT5Block(const std::vector<uint8>& block, std::vector<uint8>& dst, uint32& dstOffset) {
  static int init = 1;
  if (init) {
    initDXT();
    init = 0;
  }
  compressDXT5AlphaBlock(block, dst, dstOffset);
  compressDXT5ColorBlock(block, dst, dstOffset);
}

void compressDXT5(const std::vector<float32>& src, std::vector<uint8>& dst, const float32 exp) {
  auto faceSize = (uint32)std::sqrt(src.size() / 4);
  auto sizeInBlocks = std::max((uint32)(faceSize / 4), 1u);
  uint32 blockCount = sizeInBlocks * sizeInBlocks;
  dst.resize(blockCount * 16);
  std::vector<uint8> blk(64);
  uint32 dstOffset = 0;
  for (int v = 0; v < faceSize; v += 4) {
    for (int u = 0; u < faceSize; u += 4) {
      extractBlock(src, u, v, faceSize, exp, blk);
      compressDXT5Block(blk, dst, dstOffset);
    }
  }
}

void ktxFromFloats(const std::vector<std::vector<std::vector<float32>>>& src, std::vector<uint8>& dst, const float32 exp, const bool8 compress) {
  auto levels = (uint32)src.size();
  auto faces = (uint32)src[0].size();
  auto firstFaceSize = (uint32)std::sqrt(src[0][0].size() / 4);
  ktxHeader header = KTX_IDENTIFIER_REF;
  header.endianness = KTX_ENDIAN_REF;
  header.glType = compress ? 0 : GL_FLOAT;
  header.glTypeSize = compress ? 1 : 4;
  header.glFormat = compress ? 0 : GL_RGB;
  header.glInternalFormat = compress ? GL_COMPRESSED_RGBA_S3TC_DXT5_EXT : GL_RGB32F;
  header.glBaseInternalFormat = GL_RGB;
  header.pixelWidth = header.pixelHeight = firstFaceSize;
  header.pixelDepth = 0;
  header.numberOfArrayElements = 0;
  header.numberOfFaces = faces;
  header.numberOfMipmapLevels = levels;
  header.bytesOfKeyValueData = 0;

  uint32 totalDataSize = 0;
  std::vector<uint32> imageSize(levels);
  std::vector<std::vector<std::vector<uint8>>> data;
  std::fill(imageSize.begin(), imageSize.end(), 0);
  data.resize(levels);
  for (auto li = 0; li < levels; ++li) {
    data[li].resize(faces);
    uint32 levelDataSize = 0;
    uint32 faceDataSize = 0;
    for (auto fi = 0; fi < faces; ++fi) {
      if (compress) {
        compressDXT5(src[li][fi], data[li][fi], exp);
      } else {
        dropAlpha(src[li][fi], data[li][fi]);
      }
      if (imageSize[li] == 0) {
        imageSize[li] = (uint32)data[li][fi].size();
      } else if (imageSize[li] != (uint32)data[li][fi].size()) {
        std::cerr << "size of face " << fi << " of level " << li << " (" << data[li][fi].size() << ") did not match size of face 0 (" << imageSize[li] << ")" << std::endl;
      }
      faceDataSize = (uint32)data[li][fi].size();
      levelDataSize += data[li][fi].size();
    }
    totalDataSize += levelDataSize;
    imageSize[li] = faceDataSize;
  }
  // write the file!
  dst.resize(sizeof(ktxHeader) + (sizeof(uint32) * imageSize.size()) + totalDataSize);
  uint32 off = 0;
  memcpy(&dst[off], &header, sizeof(ktxHeader));
  off += sizeof(ktxHeader);
  for (auto li = 0; li < levels; ++li) {
    memcpy(&dst[off], &imageSize[li], sizeof(uint32));
    off += sizeof(uint32);
    for (auto fi = 0; fi < faces; ++fi) {
      memcpy(&dst[off], &data[li][fi][0], sizeof(uint8) * data[li][fi].size());
      off += sizeof(uint8) * data[li][fi].size();
    }
  }
}

#include "CubeMap.h"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <thread>
#include <chrono>

#include "imageutils.h"
#include "RNG.h"
#include "dds.h"

#define MAX_INPUT_CUBE_SIZE 512u

using hrc = std::chrono::high_resolution_clock;

typedef hrc::time_point tp;

const std::chrono::duration<float64, std::milli> oneSecond(1000.0);

class Sampler {
public:
  Sampler(vec3d vec, float64 sa, RNG* rng) : center(std::move(vec)), rng(rng) {
    // takes sa in steridians; convert to half conic apex angle
    oneMinusCosMax = sa / (2.0 * M_PI);
    center.normalize();
    if (center(1) == 1.0) {
      xform <<  1,  0,  0,
                0,  0,  1,
                0, -1,  0;
    } else if (center(1) == -1.0) {
      xform <<  1,  0,  0,
                0,  0, -1,
                0,  1,  0;
    } else if (center(2) == 1.0) {
      xform <<  1,  0,  0,
                0,  1,  0,
                0,  0,  1;
    } else if (center(2) == -1.0) {
      xform <<  1,  0,  0,
                0, -1,  0,
                0,  0, -1;
    } else {
      float64 x = center(0), y = center(1), z = center(2);
      xform << (y*y+z+z*z)/(1.0f+z), y*(-1.0f+y*y+z*z)/(x*(1.0f+z)), (x-x*z*z)/(x*x+y*y),
          y*(-1.0f+y*y+z*z)/(x*(1.0f+z)), (1.0f-y*y+z)/(1.0f+z), (y-y*z*z)/(x*x+y*y),
          -x, -y, z;
    }
  };

  vec3d operator()() {
    const float64 r1 = (*rng)();
    const float64 r2 = (*rng)();
    const float64 f = 2.0 * M_PI * r1;
//    const float64 z = 1.0 - (r2 * (1.0 - cosTheta));
    const float64 z = 1.0 - (r2 * oneMinusCosMax);
    const float64 zf = sqrt(1.0 - z * z);
    vec3d v(cos(f) * zf, sin(f) * zf, z);
    v.normalize();
    return xform * v;
  };

  vec3d center;
  mat3d xform;
  float64 oneMinusCosMax;
  RNG* rng;
};

float64 ae(const float64 x, const float64 y) {
  return atan2(x * y, sqrt(x * x + y * y + 1));
}

void texelToVec(const EFACE faceIndex, const uint32 faceSize, const int32 u, const int32 v, Eigen::Ref<vec3d> vec, float64& sa, const bool8 fixup) {
  float64 uc = (2.0 * (u + 0.5) / faceSize) - 1;
  float64 vc = (2.0 * (v + 0.5) / faceSize) - 1;

  // get solid angle before possible fixup adjustment
  float64 invRes = (1.0 / faceSize);
  float64 x0 = uc - invRes;
  float64 x1 = uc + invRes;
  float64 y0 = vc - invRes;
  float64 y1 = vc + invRes;
  // this gives sa in steridians
  sa = ae(x0, y0) - ae(x0, y1) - ae(x1, y0) + ae(x1, y1);

  if (fixup) {
    float64 a = std::pow(float64(faceSize), 2.0) / std::pow(float64(faceSize - 1), 3.0);
    uc = a * std::pow(uc, 3.0) + uc;
    vc = a * std::pow(vc, 3.0) + vc;
  }

  switch(faceIndex) {
////  These appear to give correct results
//    case XPOS: vec <<  uc,  -1, -vc; break;
//    case XNEG: vec << -uc,   1, -vc; break;
//    case YPOS: vec << -vc, -uc,   1; break;
//    case YNEG: vec <<  vc, -uc,  -1; break;
//    case ZPOS: vec <<  -1, -uc, -vc; break;
//    case ZNEG: vec <<   1,  uc, -vc; break;

//  While these are from the original
    case XPOS: vec <<   1, -vc, -uc; break;
    case XNEG: vec <<  -1, -vc,  uc; break;
    case YPOS: vec <<  uc,   1,  vc; break;
    case YNEG: vec <<  uc,  -1, -vc; break;
    case ZPOS: vec <<  uc, -vc,   1; break;
    case ZNEG: vec << -uc, -vc,  -1; break;
  }
  vec.normalize();
}

void vecToTexel(const Eigen::Ref<const vec3d>& vec, const uint32 faceSize, EFACE& face, int32& u, int32& v) {
  vec3d absVec = vec.cwiseAbs();
  float64 maxCoord;
  if ((absVec(0) >= absVec(1)) && (absVec(0) >= absVec(2))) {
    maxCoord = absVec(0);
    if (vec(0) >= 0) {
      face = EFACE::XPOS;
    } else {
      face = EFACE::XNEG;
    }
  } else if ((absVec(1) >= absVec(0)) && (absVec(1) >= absVec(2))) {
    maxCoord = absVec(1);
    if (vec(1) >= 0) {
      face = EFACE::YPOS;
    } else {
      face = EFACE::YNEG;
    }
  } else {
    maxCoord = absVec(2);
    if (vec(2) >= 0) {
      face = EFACE::ZPOS;
    } else {
      face = EFACE::ZNEG;
    }
  }
  vec3d onFaceVec = vec / maxCoord;
  float64 uc = 0, vc = 0;
  switch (face) {

////  These appear to give correct results
//    case EFACE::XPOS: uc =  onFaceVec(0); vc = -onFaceVec(2); break;
//    case EFACE::XNEG: uc = -onFaceVec(0); vc = -onFaceVec(2); break;
//    case EFACE::YPOS: uc = -onFaceVec(1); vc = -onFaceVec(0); break;
//    case EFACE::YNEG: uc = -onFaceVec(1); vc =  onFaceVec(0); break;
//    case EFACE::ZPOS: uc = -onFaceVec(1); vc = -onFaceVec(2); break;
//    case EFACE::ZNEG: uc =  onFaceVec(1); vc = -onFaceVec(2); break;

//  While these are from the original
    case EFACE::XPOS: uc = -onFaceVec(2); vc = -onFaceVec(1); break;
    case EFACE::XNEG: uc =  onFaceVec(2); vc = -onFaceVec(1); break;
    case EFACE::YPOS: uc =  onFaceVec(0); vc =  onFaceVec(2); break;
    case EFACE::YNEG: uc =  onFaceVec(0); vc = -onFaceVec(2); break;
    case EFACE::ZPOS: uc =  onFaceVec(0); vc = -onFaceVec(1); break;
    case EFACE::ZNEG: uc = -onFaceVec(0); vc = -onFaceVec(1); break;

  }
  u = (int32)std::floor((faceSize - 1) * 0.5f * (uc + 1.0f));
  v = (int32)std::floor((faceSize - 1) * 0.5f * (vc + 1.0f));
}

void lookupSphere(const std::vector<float32>& map, const uint32 width, const uint32 height, const Eigen::Ref<const vec3d>& sample, Eigen::Ref<vec4f> pixel) {
  float64 theta = std::atan2(sample(0), sample(1));
  float64 phi = std::acos(sample(2));
  float64 u = (theta * M_1_PI + 1.0) / 2.0;
  float64 v = phi * M_1_PI;
  if ((u != u || v != v) || (u < 0 || u > 1 || v < 0 || v >= 1)) {
    std::cerr << "Sample: " << sample << std::endl;
    throw std::range_error("lookup coordinates out of range");
  }
  u *= width - 1.0;
  v *= height - 1.0;
  auto ui = (uint32)floor(u);
  auto vi = (uint32)floor(v);
  float64 uf = u - ui;
  float64 vf = v - vi;
  vec4f weights;
  weights << (1.0f - vf) * (1.0f - uf),
             (1.0f - vf) * uf,
             vf * (1.0f - uf),
             vf * uf;
  mat4f samples;
  int iSample = 0;
  for (int dv = 0; dv < 2; ++dv) {
    for (int du = 0; du < 2; ++du, ++iSample) {
      int x = (ui + du) % width;
      int y = vi + dv;
      int off = 4 * (y * width + x);
      for (int k = 0; k < 4; ++k) {
        samples(k, iSample) = map[off + k];
      }
    }
  }
  pixel += samples * weights;
}

void lookupCube(const std::vector<std::vector<float32>>& cube, const uint32 faceSize, const Eigen::Ref<const vec3d>& vec, Eigen::Ref<vec4f> pixel) {
  EFACE face;
  int32 u, v;
  vecToTexel(vec, faceSize, face, u, v);
  pixel << cube[face][4 * (v * faceSize + u) + 0],
           cube[face][4 * (v * faceSize + u) + 1],
           cube[face][4 * (v * faceSize + u) + 2],
           cube[face][4 * (v * faceSize + u) + 3];
}

void evalShBasis(const Eigen::Ref<const vec3d>& dir, Eigen::Ref<Eigen::Matrix<float64, 1, 25>> res) {
  float64 xx = dir(0);
  float64 yy = dir(1);
  float64 zz = dir(2);
  float64 x[6], y[6], z[6];
  x[0] = y[0] = z[0] = 1.0;
  for (int i = 1; i < 6; ++i) {
    x[i] = xx * x[i-1];
    y[i] = yy * y[i-1];
    z[i] = zz * z[i-1];
  }
  res(0,  0) =  (1.0 / (2.0 * M_SQRT_PI));
  res(0,  1) = -(sqrt(3.0 / M_PI) * yy) / 2.0;
  res(0,  2) =  (sqrt(3.0 / M_PI) * zz) / 2.0;
  res(0,  3) = -(sqrt(3.0 / M_PI) * xx) / 2.0;
  res(0,  4) =  (sqrt(15.0 / M_PI) * xx * yy)/2.0;
  res(0,  5) = -(sqrt(15.0 / M_PI) * yy * zz) / 2.0;
  res(0,  6) =  (sqrt(5.0 / M_PI) * (-1 + 3 * z[2])) / 4.0;
  res(0,  7) = -(sqrt(15.0 / M_PI) * xx * zz) / 2.0;
  res(0,  8) =   sqrt(15.0 / M_PI) * (x[2] - y[2]) / 4.0;
  res(0,  9) =  (sqrt(35.0 / (2.0 * M_PI)) * (-3.0 * x[2] * yy + y[3])) / 4.0;
  res(0, 10) =  (sqrt(105.0 / M_PI) * xx * yy * zz) / 2.0;
  res(0, 11) = -(sqrt(21.0 / (2.0 * M_PI)) * yy * (-1.0 + 5.0 * z[2])) / 4.0;
  res(0, 12) =  (sqrt(7.0 / M_PI) * zz * (-3.0 + 5.0 * z[2])) / 4.0;
  res(0, 13) = -(sqrt(21.0 / (2.0 * M_PI)) * xx * (-1.0 + 5.0 * z[2])) / 4.0;
  res(0, 14) =  (sqrt(105.0 / M_PI) * (x[2] - y[2]) * zz) / 4.0;
  res(0, 15) = -(sqrt(35.0 / (2.0 * M_PI)) * (x[3] - 3.0 * xx * y[2])) / 4.0;
  res(0, 16) =  ( 3.0 * sqrt(35.0 / M_PI) * xx * yy * (x[2] - y[2])) / 4.0;
  res(0, 17) =  (-3.0 * sqrt(35.0 / (2.0 * M_PI)) * (3.0 * x[2] * yy - y[3]) * zz) / 4.0;
  res(0, 18) =  ( 3.0 * sqrt(5.0 / M_PI) * xx * yy * (-1.0 + 7.0 * z[2])) / 4.0;
  res(0, 19) =  (-3.0 * sqrt(5.0 / (2.0 * M_PI)) * yy * zz * (-3.0 + 7.0 * z[2])) / 4.0;
  res(0, 20) =  ( 3.0 * (3.0 - 30.0 * z[2] + 35.0 * z[4])) / (16.0 * M_SQRT_PI);
  res(0, 21) =  (-3.0 * sqrt(5.0 / (2.0 * M_PI)) * xx * zz * (-3.0 + 7.0 * z[2])) / 4.0;
  res(0, 22) =  ( 3.0 * sqrt(5.0 / M_PI) * (x[2] - y[2]) * (-1.0 + 7.0 * z[2])) / 8.0;
  res(0, 23) =  (-3.0 * sqrt(35.0 / (2.0 * M_PI)) * (x[3] - 3.0 * xx * y[2]) * zz) / 4.0;
  res(0, 24) =  ( 3.0 * sqrt(35.0 /M_PI) * (x[4] - 6.0 * x[2] * y[2] + y[4])) / 16.0;
}

void clearFilterExtents(std::vector<BBox>& extents) {
  for (auto& extent : extents) {
    extent.Clear();
  }
}

void determineFilterExtents(const Eigen::Ref<const vec3d>& centerTapDir, const uint32 size, const int32 bboxSize, std::vector<BBox>& extents) {
  int32 u, v, minU, minV, maxU, maxV, i;
  int32 boAmount[4], boBBoxMin[4], boBBoxMax[4];
  EFACE faceIndex;
  vecToTexel(centerTapDir, size, faceIndex, u, v);
  extents[faceIndex].Augment(u - bboxSize, v - bboxSize, 0);
  extents[faceIndex].Augment(u + bboxSize, v + bboxSize, 0);
  extents[faceIndex].ClampMin(0, 0, 0);
  extents[faceIndex].ClampMax(size - 1, size - 1, 0);
  minU = extents[faceIndex].minCoord(0);
  minV = extents[faceIndex].minCoord(1);
  maxU = extents[faceIndex].maxCoord(0);
  maxV = extents[faceIndex].maxCoord(1);
  boAmount[0] = (bboxSize - u);
  boBBoxMin[0] = minV;
  boBBoxMax[0] = maxV;
  boAmount[1] = (u + bboxSize) - (size - 1);
  boBBoxMin[1] = minV;
  boBBoxMax[1] = maxV;
  boAmount[2] = (bboxSize - v);
  boBBoxMin[2] = minU;
  boBBoxMax[2] = maxU;
  boAmount[3] = (v + bboxSize) - (size - 1);
  boBBoxMin[3] = minU;
  boBBoxMax[3] = maxU;
  for (i = 0; i < 4; ++i) {
    if (boAmount[i] > 0) { const FaceEdge& neigh = FACE_NEIGHBORS[faceIndex][i];
      if ((i == neigh.edge) || ((i + neigh.edge) == 3)) {
        boBBoxMin[i] = (size - 1) - boBBoxMin[i];
        boBBoxMax[i] = (size - 1) - boBBoxMax[i];
      }
      switch(neigh.edge) {
        case EEDGE::LEFT:
          extents[neigh.face].Augment(0, boBBoxMin[i], 0);
          extents[neigh.face].Augment(boAmount[i], boBBoxMax[i], 0);
          break;
        case EEDGE::RIGHT:
          extents[neigh.face].Augment((size - 1), boBBoxMin[i], 0);
          extents[neigh.face].Augment((size - 1) - boAmount[i], boBBoxMax[i], 0);
          break;
        case EEDGE::TOP:
          extents[neigh.face].Augment(boBBoxMin[i], 0, 0);
          extents[neigh.face].Augment(boBBoxMax[i], boAmount[i], 0);
          break;
        case EEDGE::BOTTOM:
          extents[neigh.face].Augment(boBBoxMin[i], (size - 1), 0);
          extents[neigh.face].Augment(boBBoxMax[i], (size - 1) - boAmount[i], 0);
          break;
      }
      extents[neigh.face].ClampMin(0, 0, 0);
      extents[neigh.face].ClampMax(size - 1, size - 1, 0);
    }
    if (boAmount[i] > size) {
      EFACE oppositeFace;
      switch(faceIndex) {
        case EFACE::XPOS: oppositeFace = EFACE::XNEG; break;
        case EFACE::XNEG: oppositeFace = EFACE::XPOS; break;
        case EFACE::YPOS: oppositeFace = EFACE::YNEG; break;
        case EFACE::YNEG: oppositeFace = EFACE::YPOS; break;
        case EFACE::ZPOS: oppositeFace = EFACE::ZNEG; break;
        case EFACE::ZNEG: oppositeFace = EFACE::ZPOS; break;
      }
      extents[oppositeFace].Augment(0, 0, 0);
      extents[oppositeFace].Augment((size - 1), (size - 1), 0);
    }
  }
}

void fixupEdges(std::vector<std::vector<float32>>& faces, const uint32 size) {
  vec4f faceCorners[6][4];
  vec4f cubeCorners[8][3];
  int32 cubeCornerCounters[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  ECORNER corner;
  for (int32 fi = 0; fi < 6; ++fi) {
    faceCorners[fi][0] << faces[fi][0],
                          faces[fi][1],
                          faces[fi][2],
                          faces[fi][3];
    faceCorners[fi][1] << faces[fi][((size - 1) * 4) + 0],
                          faces[fi][((size - 1) * 4) + 1],
                          faces[fi][((size - 1) * 4) + 2],
                          faces[fi][((size - 1) * 4) + 3];
    faceCorners[fi][2] << faces[fi][((size * (size - 1)) * 4) + 0],
                          faces[fi][((size * (size - 1)) * 4) + 1],
                          faces[fi][((size * (size - 1)) * 4) + 2],
                          faces[fi][((size * (size - 1)) * 4) + 3];
    faceCorners[fi][3] << faces[fi][(((size * (size - 1)) + (size - 1)) * 4) + 0],
                          faces[fi][(((size * (size - 1)) + (size - 1)) * 4) + 1],
                          faces[fi][(((size * (size - 1)) + (size - 1)) * 4) + 2],
                          faces[fi][(((size * (size - 1)) + (size - 1)) * 4) + 3];
    for (int32 ci = 0; ci < 4; ++ci) {
      corner = CUBE_CORNERS[fi][ci];
      cubeCorners[corner][cubeCornerCounters[corner]] = faceCorners[fi][ci];
      cubeCornerCounters[corner]++;
    }
  }
  vec4f accum(0, 0, 0, 0);
  for (auto& cubeCorner : cubeCorners) {
    accum << 0, 0, 0, 0;
    for (const auto& i : cubeCorner) {
      accum += i;
    }
    accum /= 3.0f;
    for (auto& i : cubeCorner) {
      i = accum;
    }
  }
  int32 edgeStart, neighborEdgeStart, edgeWalk, neighborEdgeWalk;
  for (int32 ei = 0; ei < 12; ++ei) {
    FaceEdge fe = CUBE_EDGES[ei];
    FaceEdge ne = FACE_NEIGHBORS[fe.face][fe.edge];
    edgeStart = neighborEdgeStart = edgeWalk = neighborEdgeWalk = 0;
    switch (fe.edge) {
      case EEDGE::LEFT:
        edgeWalk = 4 * size;
        break;
      case EEDGE::RIGHT:
        edgeStart += (size - 1) * 4;
        edgeWalk = 4 * size;
        break;
      case EEDGE::TOP:
        edgeWalk = 4;
        break;
      case EEDGE::BOTTOM:
        edgeStart += size * (size - 1) * 4;
        edgeWalk = 4;
        break;
    }
    if ((fe.edge == ne.edge) || ((fe.edge + ne.edge) == 3)) {
      switch(ne.edge) {
        case EEDGE::LEFT:
          neighborEdgeStart += (size - 1) * size * 4;
          neighborEdgeWalk = -(4 * size);
          break;
        case EEDGE::RIGHT:
          neighborEdgeStart += ((size - 1) * size + (size - 1)) * 4;
          neighborEdgeWalk = -(4 * size);
          break;
        case EEDGE::TOP:
          neighborEdgeStart += (size - 1) * 4;
          neighborEdgeWalk = -4;
          break;
        case EEDGE::BOTTOM:
          neighborEdgeStart += ((size - 1) * size + (size - 1)) * 4;
          neighborEdgeWalk = -4;
          break;
      }
    } else {
      switch (ne.edge) {
        case EEDGE::LEFT:
          neighborEdgeWalk = 4 * size;
          break;
        case EEDGE::RIGHT:
          neighborEdgeStart += (size - 1) * 4;
          neighborEdgeWalk = 4 * size;
          break;
        case EEDGE::TOP:
          neighborEdgeWalk = 4;
          break;
        case EEDGE::BOTTOM:
          neighborEdgeStart += size * (size - 1) * 4;
          neighborEdgeWalk = 4;
          break;
      }
    }
    edgeStart += edgeWalk;
    neighborEdgeStart += neighborEdgeWalk;
    for (int32 j = 1; j < (size - 1); ++j) {
      vec4f edgeTap(faces[fe.face][edgeStart + 0],
                    faces[fe.face][edgeStart + 1],
                    faces[fe.face][edgeStart + 2],
                    faces[fe.face][edgeStart + 3]);
      vec4f neighborEdgeTap(faces[ne.face][neighborEdgeStart + 0],
                            faces[ne.face][neighborEdgeStart + 1],
                            faces[ne.face][neighborEdgeStart + 2],
                            faces[ne.face][neighborEdgeStart + 3]);
      vec4f avgTap = (edgeTap + neighborEdgeTap) / 2.0f;
      faces[fe.face][edgeStart + 0] = faces[ne.face][neighborEdgeStart + 0] = avgTap(0);
      faces[fe.face][edgeStart + 1] = faces[ne.face][neighborEdgeStart + 1] = avgTap(1);
      faces[fe.face][edgeStart + 2] = faces[ne.face][neighborEdgeStart + 2] = avgTap(2);
      faces[fe.face][edgeStart + 3] = faces[ne.face][neighborEdgeStart + 3] = avgTap(3);
      edgeStart += edgeWalk;
      neighborEdgeStart += neighborEdgeWalk;
    }
  }
}

float32 evaluateSimpleLobe(const float32 cosTheta, const float32 alpha) {
  float32 aSqr = alpha * alpha;
  float32 b = (float32)M_PI * (1.0f + aSqr + cosTheta * (-1.0f + aSqr));
  return (2.0f * aSqr) / (b * b * (1.0f + std::sqrt(1.0f + (-1.0f + (1.0f / (cosTheta * cosTheta))) * aSqr)));
}

void processFilterExtents(const Eigen::Ref<const vec3d>& centerTapDir, const float32 dotProdThresh, const std::vector<BBox>& extents, const std::vector<std::vector<float32>>& inFaces, const std::vector<std::vector<float64>>& normFaces, const uint32 inFaceSize, Eigen::Ref<vec4f> filteredTexel, const EFILTER_TYPE filterType, const float32 specularPower = 0.0f) {
  int32 fi, u, v, uStart, vStart, uEnd, vEnd, normOff, srcOff, normWalk, srcWalk;
  vec4d dstAccum(0, 0, 0, 0);
  float64 weightAccum = 0.0;
  float32 tapDotProd, weight;
  Eigen::Map<const vec3d> vec(nullptr);
  Eigen::Map<const vec4f> pix(nullptr);

  for (fi = 0; fi < 6; ++fi) {
    if (!(extents[fi].Empty())) {
      uStart = extents[fi].minCoord[0];
      vStart = extents[fi].minCoord[1];
      uEnd = extents[fi].maxCoord[0];
      vEnd = extents[fi].maxCoord[1];
      normOff = 4 * ((vStart * inFaceSize) + uStart);
      srcOff = 4 * ((vStart * inFaceSize) + uStart);
      for (v = vStart; v <= vEnd; ++v) {
        normWalk = srcWalk = 0;
        for (u = uStart; u <= uEnd; ++u) {
          new (&vec) Eigen::Map<const vec3d>(&normFaces[fi][normOff + normWalk]);
          tapDotProd = (float32)vec.dot(centerTapDir);
          if (tapDotProd >= dotProdThresh) {
            weight = (float32)normFaces[fi][normOff + normWalk + 3];
            switch (filterType) {
              case EFILTER_TYPE::COSINE_POWER:
                weight *= std::pow(tapDotProd, specularPower);
                break;
              case EFILTER_TYPE::COSINE:
                if (tapDotProd > 0.0f) {
                  weight *= tapDotProd;
                } else {
                  weight = 0.0f;
                }
                break;
              case EFILTER_TYPE::PRISM_GGX:
                weight *= evaluateSimpleLobe(tapDotProd, std::min(specularPower, 0.9026f));
                break;
            }
            new (&pix) Eigen::Map<const vec4f>(&inFaces[fi][srcOff + srcWalk]);
            dstAccum += weight * pix.cast<float64>();
            srcWalk += 4;
            weightAccum += weight;
          } else {
            srcWalk += 4;
          }
          normWalk += 4;
        }
        normOff += inFaceSize * 4;
        srcOff += inFaceSize * 4;
      }
    }
  }

  if (weightAccum != 0.0) {
    filteredTexel = (dstAccum / weightAccum).cast<float32>();
  } else {
    lookupCube(inFaces, inFaceSize, centerTapDir, filteredTexel);
  }
}

void filterCubeSurface(const std::vector<std::vector<float32>>& inFaces, std::vector<float32>& outFace, const std::vector<std::vector<float64>>& inNormFaces, const std::vector<std::vector<float64>>& outNormFaces, const EFACE faceIndex, const uint32 inFaceSize, const uint32 outFaceSize, const int32 filterSize, const float32 dotProdThresh, const EFILTER_TYPE filterType, const bool8 fixup = false, const float32 specularPower = 0.0f) {
  Eigen::Map<const vec3d> centerTapDir(nullptr);
  std::vector<BBox> extents(6);
  Eigen::Map<vec4f> filteredTexel(nullptr);
  for (auto v = 0; v < outFaceSize; ++v) {
    for (auto u = 0; u < outFaceSize; ++u) {
      new (&centerTapDir) Eigen::Map<const vec3d>(&outNormFaces[faceIndex][4 * (v * outFaceSize + u)]);
      clearFilterExtents(extents);
      determineFilterExtents(centerTapDir, inFaceSize, filterSize, extents);
      new (&filteredTexel) Eigen::Map<vec4f>(&outFace[4 * (v * outFaceSize + u)]);
      processFilterExtents(centerTapDir, dotProdThresh, extents, inFaces, inNormFaces, inFaceSize, filteredTexel, filterType, specularPower);
    }
  }
}

void computeFaceNorm(const uint32 faceSize, const EFACE faceIndex, std::vector<float64>& face, const bool8 fixup = false) {
  Eigen::Map<vec3d> vec(nullptr);
  for (auto v = 0; v < faceSize; ++v) {
    for (auto u = 0; u < faceSize; ++u) {
      new (&vec) Eigen::Map<vec3d>(&face[4 * (v * faceSize + u)]);
      texelToVec(faceIndex, faceSize, u, v, vec, face[4 * (v * faceSize + u) + 3], fixup);
    }
  }
}

float32 computeBaseFilterAngle(const float32 specularPower) {
  return 2.0f * std::acos(std::pow(0.000001f, 1.0f / specularPower));
}



CubeMap::CubeMap(uint32 faceSize) {

  if (log2(faceSize) != floor(log2(faceSize))) {
    faceSize = (uint32)pow(2, floor(log2(faceSize)));
  }

  numThreads = std::thread::hardware_concurrency() - 1;
  if (numThreads > 0) {
    threads.resize(numThreads);
  }

  levels = (uint32)log2(faceSize) + 1;

  data.resize(levels);
  dataNorm.resize(levels);
  dataDot.resize(levels);
  levelSize.resize(levels);

  for (int li = 0; li < levels; ++li) {
    levelSize[li] = (uint32)(faceSize / pow(2, li));
    data[li].resize(6);
    dataNorm[li].resize(6);
    dataDot[li].resize(6);
    for (int fi = 0; fi < 6; ++fi) {
      data[li][fi].resize(levelSize[li] * levelSize[li] * 4);
      dataNorm[li][fi].resize(levelSize[li] * levelSize[li] * 4);
      dataDot[li][fi].resize(levelSize[li] * levelSize[li]);
    }
  }
  for (auto li = 0; li < levels; ++li) {
    for (auto fi = 0; fi < 6; ++fi) {
      computeFaceNorm(levelSize[li], (EFACE)fi, dataNorm[li][fi], true);
    }
  }

  irrFaceSize = faceSize / 2;
  irr.resize(6);
  for (int fi = 0; fi < 6; ++fi) {
    irr[fi].resize(irrFaceSize * irrFaceSize * 4);
  }

}

template <typename F>
void CubeMap::RunInThread(F&& func) {
  if (numThreads > 0) {
    // if there is a free thread, run it there
    for (auto&& thread: threads) {
      if (thread.joinable()) continue;
      thread = std::thread(func);
      return;
    }
    // if not, wait until there is a free thread, then run it there

    while (true) {
      if (std::any_of(threads.begin(), threads.end(), [](const std::thread& thread) { return !thread.joinable(); })) {
        RunInThread(func);
        return;
      }
      std::this_thread::sleep_for(oneSecond);
    }
  } else {
    // there are no threads, so run here.
    func();
  }
}

void CubeMap::AwaitAllThreads() {
  if (numThreads > 0) {
    for (auto&& thread: threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }
}



void CubeMap::LoadSphere(const std::vector<float32>& sphere, const Eigen::Ref<const mat3d>& cubeToSphereMatrix, const uint32 sphereWidth, const uint32 sphereHeight, const uint32 samplingFactor) {

  float32 weight = (float32)samplingFactor * samplingFactor;
  sourceFaceSize = std::min(MAX_INPUT_CUBE_SIZE, (static_cast<uint32>(std::ceil(sphereHeight / M_PI)) + 31) & ~31);
  source.resize(6);
  sourceNorm.resize(6);
  for (auto fi = 0; fi < 6; ++fi) {
    source[fi].resize(sourceFaceSize * sourceFaceSize * 4);
    sourceNorm[fi].resize(sourceFaceSize * sourceFaceSize * 4);
    std::fill(source[fi].begin(), source[fi].end(), 0.0f);
    std::fill(sourceNorm[fi].begin(), sourceNorm[fi].end(), 0.0f);
  }
  for (auto fi = 0; fi < 6; ++fi) {
    computeFaceNorm(sourceFaceSize, (EFACE)fi, sourceNorm[fi]);
  }

  tp start = hrc::now();
  for (auto fi = 0; fi < 6; ++fi) {
    RunInThread([&, fi]() {
      vec3d vec, sample;
      float64 sa;
      Eigen::Map<vec4f> pixel(nullptr);
      RNG rng;
      for (auto v = 0; v < sourceFaceSize; ++v) {
        for (auto u = 0; u < sourceFaceSize; ++u) {
          texelToVec((EFACE)fi, sourceFaceSize, u, v, vec, sa);
          vec = cubeToSphereMatrix * vec;
          Sampler sampler(vec, sa, &rng);
          new (&pixel) Eigen::Map<vec4f>(&source[fi][4 * (v * sourceFaceSize + u)]);
          pixel << 0, 0, 0, 0;
          for (auto si = 0; si < samplingFactor; ++si) {
            for (auto sj = 0; sj < samplingFactor; ++sj) {
              sample = sampler();
              lookupSphere(sphere, sphereWidth, sphereHeight, sample, pixel);
            }
          }
          pixel /= weight;
        }
      }
    });
  }
  AwaitAllThreads();
  tp stop = hrc::now();
  std::cout << "projecting spheremap to cube took " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;
}



void CubeMap::FilterSurfaces(const std::vector<std::vector<float32>>& inFaces, std::vector<std::vector<float32>>& outFaces, const std::vector<std::vector<float64>>& inNormFaces, const std::vector<std::vector<float64>>& outNormFaces, const uint32 inFaceSize, const uint32 outFaceSize, const float32 coneAngle, const EFILTER_TYPE filterType, const bool8 fixup = false, const float32 specularPower = 0.0f) {
  float32 srcTexelAngle = std::atan2(1.0f, (float32)inFaceSize);
  float32 filterAngle = std::min((float32)M_PI_2, std::max(coneAngle / 2.0f, srcTexelAngle));
  int32 filterSize = std::max(1, (int32)std::ceil(filterAngle / srcTexelAngle));
  float32 dotProdThresh = std::cos(filterAngle);
  int32 fi;
  for (fi = 0; fi < 6; ++fi) {
    outFaces[fi].resize(outFaceSize * outFaceSize * 4);
  }
  for (fi = 0; fi < 6; ++fi) {
    RunInThread([&, fi]() {
      filterCubeSurface(inFaces, outFaces[fi], inNormFaces, outNormFaces, (EFACE)fi, inFaceSize, outFaceSize, filterSize, dotProdThresh, filterType, fixup, specularPower);
    });
  }
  AwaitAllThreads();
}

void CubeMap::SHFilterSurfaces(const std::vector<std::vector<float32>>& inFaces, std::vector<std::vector<float32>>& outFaces, const std::vector<std::vector<float64>>& inNormFaces, const std::vector<std::vector<float64>>& outNormFaces, const uint32 inFaceSize, const uint32 outFaceSize) {
  Eigen::Matrix<float64, 3, 25> shRgb = Eigen::Matrix<float64, 3, 25>::Zero();
  Eigen::Matrix<float64, 1, 25> shDir = Eigen::Matrix<float64, 1, 25>::Zero();
  float64 weightAccum = 0.0;
  float64 weight;
  Eigen::Map<const vec3d> vec(nullptr);
  Eigen::Map<const vec3f> rgb(nullptr);
  for (auto fi = 0; fi < 6; ++fi) {
    for (auto v = 0; v < inFaceSize; ++v) {
      for (auto u = 0; u < inFaceSize; ++u) {
        new (&vec) Eigen::Map<const vec3d>(&inNormFaces[fi][4 * (v * inFaceSize + u)]);
        weight = inNormFaces[fi][4 * (v * inFaceSize + u) + 3];
        evalShBasis(vec, shDir);
        new (&rgb) Eigen::Map<const vec3f>(&inFaces[fi][4 * (v * inFaceSize + u)]);
        shRgb.noalias() += rgb.cast<float64>() * shDir * weight;
        weightAccum += weight;
      }
    }
  }
  Eigen::Map<vec3f> out(nullptr);
  shRgb *= 4.0 * M_PI / weightAccum;
  for (auto fi = 0; fi < 6; ++fi) {
    for (auto v = 0; v < outFaceSize; ++v) {
      for (auto u = 0; u < outFaceSize; ++u) {
        new (&vec) Eigen::Map<const vec3d>(&outNormFaces[fi][4 * (v * outFaceSize + u)]);
        evalShBasis(vec, shDir);
        new (&out) Eigen::Map<vec3f>(&outFaces[fi][4 * (v * outFaceSize + u)]);
        shDir = (shDir.array() * SH_BAND_FACTORS.array()).matrix();
        out << (shRgb.row(0).dot(shDir)),
               (shRgb.row(1).dot(shDir)),
               (shRgb.row(2).dot(shDir));
        outFaces[fi][4 * (v * outFaceSize + u) + 3] = 1.0f;
      }
    }
  }
}

void CubeMap::Filter(const float32 initialMipAngle,
                     const float32 mipAnglePerLevelScale,
                     const float32 specularPower,
                     const EFILTER_TYPE filterType) {
  float32 coneAngle = initialMipAngle;
  float32 baseFilterAngle = computeBaseFilterAngle(specularPower);
  int32 fi, li;
  tp start, stop;
  for (li = 0; li < levels; ++li) {
    for (fi = 0; fi < 6; ++fi) {
      data[li][fi].resize(levelSize[li] * levelSize[li] * 4);
    }
    start = hrc::now();
    if (li == 0) {
      // base level filtering
      FilterSurfaces(source, data[li], sourceNorm, dataNorm[li], sourceFaceSize, levelSize[li], baseFilterAngle, filterType, true, specularPower);
    } else {
      FilterSurfaces(data[li - 1], data[li], dataNorm[li - 1], dataNorm[li], levelSize[li - 1], levelSize[li], baseFilterAngle, filterType == EFILTER_TYPE::COSINE_POWER ? EFILTER_TYPE::COSINE : filterType, true);
      coneAngle *= mipAnglePerLevelScale;
    }
    fixupEdges(data[li], levelSize[li]);
    stop = hrc::now();
    std::cout << "filtering level " << li << " took " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;
  }
  start = hrc::now();
  SHFilterSurfaces(source, irr, sourceNorm, dataNorm[1], sourceFaceSize, irrFaceSize);
  stop = hrc::now();
  std::cout << "computing irradiance map took " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;
}

void CubeMap::ToSpecularDDS(std::vector<uint8>& dst) {
  logLuvDDSFromFloats(data, dst);
}

void CubeMap::ToIrradianceDDS(std::vector<uint8>& dst) {
  logLuvDDSFromFloats(irr, dst);
}

void CubeMap::PopulateTestSource(const uint32 faceSize) {
  sourceFaceSize = faceSize;
  source.resize(6);
  for (auto fi = 0; fi < 6; ++fi) {
    source[fi].resize(sourceFaceSize * sourceFaceSize * 4);
    std::fill(source[fi].begin(), source[fi].end(), 0.0f);
  }
  tp start = hrc::now();
  for (auto fi = 0; fi < 6; ++fi) {
    vec4f color;
    switch ((EFACE)fi) {
      case EFACE::XPOS: color << 1.0f, 0.0f, 0.0f, 1.0f; break;
      case EFACE::XNEG: color << 1.0f, 0.0f, 1.0f, 1.0f; break;
      case EFACE::YPOS: color << 0.0f, 1.0f, 0.0f, 1.0f; break;
      case EFACE::YNEG: color << 1.0f, 1.0f, 0.0f, 1.0f; break;
      case EFACE::ZPOS: color << 0.0f, 0.0f, 1.0f, 1.0f; break;
      case EFACE::ZNEG: color << 0.0f, 1.0f, 1.0f, 1.0f; break;
    }
    RunInThread([&, fi, color]() {
      Eigen::Map<vec4f> pixel(nullptr);
      for (auto v = 0; v < sourceFaceSize; ++v) {
        for (auto u = 0; u < sourceFaceSize; ++u) {
          new (&pixel) Eigen::Map<vec4f>(&source[fi][4 * (v * sourceFaceSize + u)]);
          pixel = color;
        }
      }
    });
  }
  AwaitAllThreads();
  tp stop = hrc::now();
  std::cout << "generating test cube took " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;
}

std::string vec2str(const Eigen::Ref<const vec3d>& vec) {
  std::stringstream ss;
  ss << "( " << vec(0) << ", " << vec(1) << ", " << vec(2) << " )";
  return ss.str();
}

std::string mat2str(const Eigen::Ref<const mat3d>& mat) {
  std::stringstream ss;
  ss << "( ( " << mat(0,0) << ", " << mat(0, 1) << ", " << mat(0, 2) << " ), ";
  ss <<   "( " << mat(1,0) << ", " << mat(1, 1) << ", " << mat(1, 2) << " ), ";
  ss <<   "( " << mat(2,0) << ", " << mat(2, 1) << ", " << mat(2, 2) << " ) )";
  return ss.str();
}


void CubeMap::ExportSphere(const Eigen::Ref<const mat3d>& cubeToSphereMatrix, const uint32 sphereWidth, const uint32 sphereHeight, std::vector<float32>& sphere) {
  sphere.resize(sphereWidth * sphereHeight * 4);
  float64 uc, vc, theta, phi;
  vec3d xyz;
  Eigen::Map<vec4f> pixel(nullptr);
  mat3d sphereToCubeMatrix = cubeToSphereMatrix.inverse();
  std::string strxyz, strmat;
  tp start = hrc::now();
  for (auto v = 0; v < sphereHeight; ++v) {
    for (auto u = 0; u < sphereWidth; ++u) {
      uc = (u + 0.5) / sphereWidth;
      vc = (v + 0.5) / sphereHeight;
      theta = -2.0 * M_PI * uc - M_PI_2;
      phi = vc * M_PI;
      xyz << std::sin(phi) * std::cos(theta), std::sin(phi) * std::sin(theta), std::cos(phi);
      xyz.normalize();
      xyz = sphereToCubeMatrix * xyz;
      new (&pixel) Eigen::Map<vec4f>(&sphere[4 * (v * sphereWidth + u)]);
      lookupCube(source, sourceFaceSize, xyz, pixel);
    }
  }
}



#ifndef CUBEMAPGEN_CUBEMAP_H
#define CUBEMAPGEN_CUBEMAP_H

#define M_SQRT_PI 1.77245385090551602729816748334114518
#define M_PI_180  0.01745329251994329576923690768448861


// #define CP_FILTER_IS_PREM(f) ((f & (CP_FILTER_TYPE_COSINE_POWER | CP_FILTER_TYPE_PRISM_GGX | CP_FILTER_TYPE_PRISM_BECKMANN)) != 0 && (f & (f - 1)) == 0)

#include <vector>
#include <stdexcept>
#include <thread>

#include "types.h"
#include "BBox.h"


enum EFACE   : uint32 { XPOS, XNEG, YPOS, YNEG, ZPOS, ZNEG };
enum EEDGE   : uint32 { LEFT, RIGHT, TOP, BOTTOM };
enum ECORNER : uint32 { NNN, NNP, NPN, NPP, PNN, PNP, PPN, PPP };

enum EFILTER_TYPE: uint32 {
//  DISC = 0,
//  CONE = 1,
  COSINE = 2,
//  ANGULAR_GAUSSIAN = 3,
  COSINE_POWER = (1 << 4),
  PRISM_GGX = (1 << 5),
//  PRISM_BECKMANN = (1 << 6)
};

struct FaceEdge {
  EFACE face;
  EEDGE edge;
};

const FaceEdge FACE_NEIGHBORS[6][4] = {{{EFACE::ZPOS, EEDGE::RIGHT },
                                        {EFACE::ZNEG, EEDGE::LEFT  },
                                        {EFACE::YPOS, EEDGE::RIGHT },
                                        {EFACE::YNEG, EEDGE::RIGHT }},
                                       {{EFACE::ZNEG, EEDGE::RIGHT },
                                        {EFACE::ZPOS, EEDGE::LEFT  },
                                        {EFACE::YPOS, EEDGE::LEFT  },
                                        {EFACE::YNEG, EEDGE::LEFT  }},
                                       {{EFACE::XNEG, EEDGE::TOP   },
                                        {EFACE::XPOS, EEDGE::TOP   },
                                        {EFACE::ZNEG, EEDGE::TOP   },
                                        {EFACE::ZPOS, EEDGE::TOP   }},
                                       {{EFACE::XNEG, EEDGE::BOTTOM},
                                        {EFACE::XPOS, EEDGE::BOTTOM},
                                        {EFACE::ZPOS, EEDGE::BOTTOM},
                                        {EFACE::ZNEG, EEDGE::BOTTOM}},
                                       {{EFACE::XNEG, EEDGE::RIGHT },
                                        {EFACE::XPOS, EEDGE::LEFT  },
                                        {EFACE::YPOS, EEDGE::BOTTOM},
                                        {EFACE::YNEG, EEDGE::TOP   }},
                                       {{EFACE::XPOS, EEDGE::RIGHT },
                                        {EFACE::XNEG, EEDGE::LEFT  },
                                        {EFACE::YPOS, EEDGE::TOP   },
                                        {EFACE::YNEG, EEDGE::BOTTOM}}};

const ECORNER CUBE_CORNERS[6][4] = {{ECORNER::PPP, ECORNER::PPN, ECORNER::PNP, ECORNER::PNN},
                                    {ECORNER::NPN, ECORNER::NPP, ECORNER::NNN, ECORNER::NNP},
                                    {ECORNER::NPN, ECORNER::PPN, ECORNER::NPP, ECORNER::PPP},
                                    {ECORNER::NNP, ECORNER::PNP, ECORNER::NNN, ECORNER::PNN},
                                    {ECORNER::NPP, ECORNER::PPP, ECORNER::NNP, ECORNER::PNP},
                                    {ECORNER::PPN, ECORNER::NPN, ECORNER::PNN, ECORNER::NNN}};

const FaceEdge CUBE_EDGES[12] = {{EFACE::XPOS, EEDGE::LEFT  },
                                 {EFACE::XPOS, EEDGE::RIGHT },
                                 {EFACE::XPOS, EEDGE::TOP   },
                                 {EFACE::XPOS, EEDGE::BOTTOM},
                                 {EFACE::XNEG, EEDGE::LEFT  },
                                 {EFACE::XNEG, EEDGE::RIGHT },
                                 {EFACE::XNEG, EEDGE::TOP   },
                                 {EFACE::XNEG, EEDGE::BOTTOM},
                                 {EFACE::ZPOS, EEDGE::TOP   },
                                 {EFACE::ZPOS, EEDGE::BOTTOM},
                                 {EFACE::ZNEG, EEDGE::TOP   },
                                 {EFACE::ZNEG, EEDGE::BOTTOM}};

const Eigen::Matrix<float64, 1, 25> SH_BAND_FACTORS = [] {
  Eigen::Matrix<float64, 1, 25> shBandFactors;
  shBandFactors << 1.0,
      2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0,
      1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      -1.0 / 24.0, -1.0 / 24.0, -1.0 / 24.0, -1.0 / 24.0, -1.0 / 24.0, -1.0 / 24.0, -1.0 / 24.0, -1.0 / 24.0, -1.0 / 24.0;
  return shBandFactors;
}();

class CubeMap {
public:
  explicit CubeMap(uint32 faceSize);

  void LoadSphere(const std::vector<float32>& sphere, const Eigen::Ref<const mat3d>& cubeToSphereMatrix, uint32 sphereWidth, uint32 sphereHeight, uint32 samplingFactor);
  void Filter(float32 initialMipAngle, float32 mipAnglePerLevelScale, float32 specularPower, EFILTER_TYPE filterType);
  void ToSpecularDDS(std::vector<uint8>& dst);
  void ToIrradianceDDS(std::vector<uint8>& dst);

  void PopulateTestSource(uint32 faceSize);
  void ExportSphere(const Eigen::Ref<const mat3d>& cubeToSphereMatrix, uint32 sphereWidth, uint32 sphereHeight, std::vector<float32>& sphere);

  uint32 sourceFaceSize = 0;
  uint32 irrFaceSize = 0;
  uint32 levels = 0;
  uint32 numThreads;

  std::vector<std::vector<float32>> source;
  std::vector<std::vector<float64>> sourceNorm;
  std::vector<std::vector<float32>> irr;
  std::vector<std::vector<std::vector<float32>>> data;
  std::vector<std::vector<std::vector<float64>>> dataNorm;
  std::vector<std::vector<std::vector<float32>>> dataDot;

  std::vector<uint32> levelSize;

private:
  template <typename F>
  void RunInThread(F&& func);
  void AwaitAllThreads();
  void FilterSurfaces(const std::vector<std::vector<float32>>& inFaces, std::vector<std::vector<float32>>& outFaces, const std::vector<std::vector<float64>>& inNormFaces, const std::vector<std::vector<float64>>& outNormFaces, uint32 inFaceSize, uint32 outFaceSize, float32 coneAngle, EFILTER_TYPE filterType,  bool8 fixup, float32 specularPower);
  void SHFilterSurfaces(const std::vector<std::vector<float32>>& inFaces, std::vector<std::vector<float32>>& outFaces, const std::vector<std::vector<float64>>& inNormFaces, const std::vector<std::vector<float64>>& outNormFaces, uint32 inFaceSize, uint32 outFaceSize);


  std::vector<std::thread> threads;
};

void texelToVec(EFACE faceIndex, uint32 faceSize, int32 u, int32 v, Eigen::Ref<vec3d> vec, float64& sa, bool8 fixup = false);
void vecToTexel(const Eigen::Ref<const vec3d>& vec, uint32 faceSize, EFACE& face, int32& u, int32& v);


#endif //CUBEMAPGEN_CUBEMAP_H

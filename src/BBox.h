#ifndef CUBEMAPGEN_BBOX_H
#define CUBEMAPGEN_BBOX_H

#include "types.h"

class BBox {
public:
  vec3i minCoord;
  vec3i maxCoord;
  BBox();
  const bool8 Empty() const;
  void Clear();
  void Augment(vec3i xyz);
  void Augment(int32 x, int32 y, int32 z);
  void ClampMin(vec3i xyz);
  void ClampMin(int32 x, int32 y, int32 z);
  void ClampMax(vec3i xyz);
  void ClampMax(int32 x, int32 y, int32 z);
};

#endif //CUBEMAPGEN_BBOX_H

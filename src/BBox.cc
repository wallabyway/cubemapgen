#include "BBox.h"

BBox::BBox() {
  Clear();
}

const bool8 BBox::Empty() const {
  return (minCoord.array() > maxCoord.array()).any();
}

void BBox::Clear() {
  minCoord << 0x7fffffff, 0x7fffffff, 0x7fffffff;
  maxCoord << 0x80000000, 0x80000000, 0x80000000;
}

void BBox::Augment(vec3i xyz) {
  minCoord = minCoord.cwiseMin(xyz);
  maxCoord = maxCoord.cwiseMax(xyz);
}

void BBox::Augment(int32 x, int32 y, int32 z) {
  vec3i xyz(x, y, z);
  Augment(xyz);
}

void BBox::ClampMin(vec3i xyz) {
  minCoord = minCoord.cwiseMax(xyz);
}

void BBox::ClampMin(int32 x, int32 y, int32 z) {
  vec3i xyz(x, y, z);
  ClampMin(xyz);
}

void BBox::ClampMax(vec3i xyz) {
  maxCoord = maxCoord.cwiseMin(xyz);
}

void BBox::ClampMax(int32 x, int32 y, int32 z) {
  vec3i xyz(x, y, z);
  ClampMax(xyz);
}

#ifndef CUBEMAPGEN_TYPES_H
#define CUBEMAPGEN_TYPES_H

#include <cstddef>
#include <OpenEXR/half.h>
#include "Eigen/Core"
#include "Eigen/Dense"

typedef char bool8;
typedef char char8;
typedef char int8;
typedef unsigned char uint8;

typedef short int16;
typedef unsigned short uint16;

typedef int int32;
typedef unsigned int uint32;

#ifdef _WIN64
typedef long long int64;
typedef unsigned long long uint64;
#else
typedef long int64;
typedef unsigned long uint64;
#endif

typedef half float16;
typedef float float32;
typedef double float64;

typedef Eigen::Vector3i vec3i;
typedef Eigen::Vector4i vec4i;

typedef Eigen::Vector3f vec3f;
typedef Eigen::Vector4f vec4f;

typedef Eigen::Vector3d vec3d;
typedef Eigen::Vector4d vec4d;

typedef Eigen::Matrix3f mat3f;
typedef Eigen::Matrix4f mat4f;

typedef Eigen::Matrix3d mat3d;
typedef Eigen::Matrix4d mat4d;

typedef Eigen::Matrix<uint8, 4, 1> rgba32;



#endif //CUBEMAPGEN_TYPES_H

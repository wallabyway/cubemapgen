#ifndef CUBEMAPGEN_LIBRARY_H
#define CUBEMAPGEN_LIBRARY_H

#include <nan.h>

NAN_METHOD(DecompressExr);
NAN_METHOD(SphereToCubes);
NAN_METHOD(GetTestSphere);

NAN_METHOD(TexelToVec);
NAN_METHOD(VecToTexel);

NAN_MODULE_INIT(Init);

NODE_MODULE(cubemapgen, Init)

#endif

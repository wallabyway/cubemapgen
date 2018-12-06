#include "cubemapgen.h"

#include <chrono>

#include "types.h"
#include "imageutils.h"
#include "CubeMap.h"

using hrc = std::chrono::high_resolution_clock;

typedef hrc::time_point tp;

NAN_METHOD(GetTestSphere) {
  if (!info[0]->IsArray()) Nan::ThrowError("first argument must be an array");
  v8::Local<v8::Array> jsArr = v8::Local<v8::Array>::Cast(info[0]);
  if (jsArr->Length() != 9) Nan::ThrowError("first argument must be an array of length = 9");
  mat3d worldToSphere;
  worldToSphere << jsArr->Get(0)->NumberValue(), jsArr->Get(1)->NumberValue(), jsArr->Get(2)->NumberValue(),
      jsArr->Get(3)->NumberValue(), jsArr->Get(4)->NumberValue(), jsArr->Get(5)->NumberValue(),
      jsArr->Get(6)->NumberValue(), jsArr->Get(7)->NumberValue(), jsArr->Get(8)->NumberValue();
  mat3d cubeToWorld;
  cubeToWorld <<  1,  0,  0,
                  0,  0, -1,
                  0,  1,  0;
  mat3d cubeToSphere = worldToSphere * cubeToWorld;
  CubeMap cube(128);
  cube.PopulateTestSource(512);
  std::vector<float32> sphere;
  cube.ExportSphere(cubeToSphere, 1000, 500, sphere);
  std::vector<uint8> jpeg;
  tp start = hrc::now();
  bool8 res = compressJpeg(sphere, jpeg, 1000, 500, 100);
  tp stop = hrc::now();
  std::cout << "jpeg compression took " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;
  if (!res) Nan::ThrowError("jpeg compression failed");
  else info.GetReturnValue().Set(Nan::CopyBuffer((char*)&jpeg.front(), (uint32)jpeg.size()).ToLocalChecked());
}

NAN_METHOD(DecompressExr) {
  if (info[0]->IsObject() && node::Buffer::HasInstance(info[0])) {
    v8::Local<v8::Object> bufObj = info[0]->ToObject();
    char* buf = node::Buffer::Data(bufObj);
    std::size_t size = node::Buffer::Length(bufObj);
    std::vector<float16> vec;
    int width, height;
    decompressExr(buf, size, vec, width, height);
    v8::Local<v8::Object> ret = Nan::New<v8::Object>();
    Nan::Set(ret, Nan::New("data").ToLocalChecked(), Nan::CopyBuffer((char*)&vec.front(), (uint32)vec.size() * sizeof(float16)).ToLocalChecked());
    Nan::Set(ret, Nan::New("width").ToLocalChecked(), Nan::New(width));
    Nan::Set(ret, Nan::New("height").ToLocalChecked(), Nan::New(height));
    info.GetReturnValue().Set(ret);
  } else {
    Nan::ThrowError(Nan::Error("argument must be a buffer"));
  }
}

NAN_METHOD(SphereToCubes) {
  if (!info[0]->IsObject() || !node::Buffer::HasInstance(info[0])) Nan::ThrowError("first argument must be a buffer");
  if (!info[1]->IsArray()) Nan::ThrowError("second argument must be an array");
  v8::Local<v8::Array> jsArr = v8::Local<v8::Array>::Cast(info[1]);
  if (jsArr->Length() != 9) Nan::ThrowError("second argument must be an array of length = 9");
  mat3d worldToSphere;
  worldToSphere << jsArr->Get(0)->NumberValue(), jsArr->Get(1)->NumberValue(), jsArr->Get(2)->NumberValue(),
                   jsArr->Get(3)->NumberValue(), jsArr->Get(4)->NumberValue(), jsArr->Get(5)->NumberValue(),
                   jsArr->Get(6)->NumberValue(), jsArr->Get(7)->NumberValue(), jsArr->Get(8)->NumberValue();
  mat3d cubeToWorld;
  cubeToWorld <<  1,  0,  0,
                  0,  0, -1,
                  0,  1,  0;
  mat3d cubeToSphere = worldToSphere * cubeToWorld;
  v8::Local<v8::Object> bufObj = info[0]->ToObject();
  char* buf = node::Buffer::Data(bufObj);
  std::size_t size = node::Buffer::Length(bufObj);
  std::vector<float32> map;
  int width, height;
  tp start, stop;
  start = hrc::now();
  bool res = decompress(buf, size, map, width, height);
  stop = hrc::now();
  std::cout << "decompression took " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;

  if (!res) Nan::ThrowError("could not interpret buffer");
  if (width != height * 2) Nan::ThrowError("image does not have 2:1 aspect ratio");

  start = hrc::now();
  float32 ev = computeEV(map);
  stop = hrc::now();
  std::cout << "ev calculation took " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;


  std::cout << "ev: " << ev << std::endl;
  v8::Local<v8::Array> arr = Nan::New<v8::Array>();
  CubeMap cube(128);

  start = hrc::now();
  cube.LoadSphere(map, cubeToSphere, (uint32)width, (uint32)height, 4);
  stop = hrc::now();
  std::cout << "sphere projection took " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;

  start = hrc::now();
  cube.Filter(1.0f * (float32)M_PI_180, 2.0f * (float32)M_PI_180, 2048.0f, EFILTER_TYPE::COSINE_POWER);
  stop = hrc::now();
  std::cout << "filtering took " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;






  std::vector<uint8> specDDS;
  std::vector<uint8> irrDDS;
  start = hrc::now();
  cube.ToSpecularDDS(specDDS);
  stop = hrc::now();
  std::cout << "speuclar dds encoding took " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;
  start = hrc::now();
  cube.ToIrradianceDDS(irrDDS);
  stop = hrc::now();
  std::cout << "irradiance dds encoding took " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;

  Nan::Set(arr, 0, Nan::CopyBuffer((char*)&specDDS.front(), (uint32)specDDS.size()).ToLocalChecked());
  Nan::Set(arr, 1, Nan::CopyBuffer((char*)&irrDDS.front(), (uint32)irrDDS.size()).ToLocalChecked());

  // export check sphere
  std::vector<float32> checkSphere;
  std::vector<uint8> jpeg;
  cube.ExportSphere(cubeToSphere, 1000, 500, checkSphere);
  applyExposure(checkSphere, ev);
  applyTonemap(checkSphere);
  compressJpeg(checkSphere, jpeg, 1000, 500, 100);
  Nan::Set(arr, 2, Nan::CopyBuffer((char*)&jpeg.front(), (uint32)jpeg.size()).ToLocalChecked());


  info.GetReturnValue().Set(arr);
}

NAN_METHOD(TexelToVec) {
  auto face = (EFACE)info[0]->IntegerValue();
  auto faceSize = (uint32)info[1]->IntegerValue();
  auto u = (int32)info[2]->IntegerValue();
  auto v = (int32)info[3]->IntegerValue();
  vec3d vec;
  float64 sa;
  texelToVec(face, faceSize, u, v, vec, sa);
  v8::Local<v8::Object> ret = Nan::New<v8::Object>();
  Nan::Set(ret, Nan::New("x").ToLocalChecked(), Nan::New(vec(0)));
  Nan::Set(ret, Nan::New("y").ToLocalChecked(), Nan::New(vec(1)));
  Nan::Set(ret, Nan::New("z").ToLocalChecked(), Nan::New(vec(2)));
  info.GetReturnValue().Set(ret);
}

NAN_METHOD(VecToTexel) {
  v8::Local<v8::Object> vecObj = Nan::To<v8::Object>(info[0]).ToLocalChecked();
  auto faceSize = (uint32)info[1]->IntegerValue();
  float64 x = Nan::Get(vecObj, Nan::New("x").ToLocalChecked()).ToLocalChecked()->NumberValue();
  float64 y = Nan::Get(vecObj, Nan::New("y").ToLocalChecked()).ToLocalChecked()->NumberValue();
  float64 z = Nan::Get(vecObj, Nan::New("z").ToLocalChecked()).ToLocalChecked()->NumberValue();
  vec3d vec(x, y, z);
  EFACE face;
  int32 u, v;
  vecToTexel(vec, faceSize, face, u, v);
  v8::Local<v8::Object> ret = Nan::New<v8::Object>();
  Nan::Set(ret, Nan::New("face").ToLocalChecked(), Nan::New(face));
  Nan::Set(ret, Nan::New("u").ToLocalChecked(), Nan::New(u));
  Nan::Set(ret, Nan::New("v").ToLocalChecked(), Nan::New(v));
  info.GetReturnValue().Set(ret);
}

NAN_MODULE_INIT(Init) {
  Nan::Export(target, "decompressExr", DecompressExr);
  Nan::Export(target, "sphereToCubes", SphereToCubes);
  Nan::Export(target, "getTestSphere", GetTestSphere);
  Nan::Export(target, "texelToVec", TexelToVec);
  Nan::Export(target, "vecToTexel", VecToTexel);
}

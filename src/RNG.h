#ifndef CUBEMAPGEN_RNG_H
#define CUBEMAPGEN_RNG_H

#include "types.h"
#include <random>

static inline uint64 rotl(const uint64 x, int k) {
  return (x << k) | (x >> (64 - k));
}

class RNG {

public:
  RNG() : state{0,0} {
    std::random_device rd;
    state[0] = uint64{rd()} | (uint64{rd()} << 32);
    state[1] = uint64{rd()} | (uint64{rd()} << 32);
  };

  float64 operator()() {
    const uint64 s0 = state[0];
    uint64 s1 = state[1];
    const uint64 v = s0 + s1;
    s1 ^= s0;
    state[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    state[1] = rotl(s1, 37);
    DI di = { .i = UINT64_C(0x3ff) << 52 | v >> 12 };
    return di.d - 1.0;
  };

private:
  union DI {
    float64 d;
    uint64 i;
  };
  uint64 state[2];

};

#endif //CUBEMAPGEN_RNG_H

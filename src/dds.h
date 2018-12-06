#ifndef CUBEMAPGEN_DDSFILE_H
#define CUBEMAPGEN_DDSFILE_H

#include "types.h"
#include <vector>


static const uint32 DDS_MAGIC = 0x20534444;
enum DDS_HEADER_FLAGS : uint32 {
  DDSD_CAPS            = 0x000001,
  DDSD_HEIGHT          = 0x000002,
  DDSD_WIDTH           = 0x000004,
  DDSD_PITCH           = 0x000008,
  DDSD_BACKBUFFERCOUNT = 0x000020,
  DDSD_ZBUFFERBITDEPTH = 0x000040,
  DDSD_ALPHABITDEPTH   = 0x000080,
  DDSD_LPSURFACE       = 0x000800,
  DDSD_PIXELFORMAT     = 0x001000,
  DDSD_CKDESTOVERLAY   = 0x002000,
  DDSD_CKDESTBLT       = 0x004000,
  DDSD_CKSRCOVERLAY    = 0x008000,
  DDSD_CKSRCBLT        = 0x010000,
  DDSD_MIPMAPCOUNT     = 0x020000,
  DDSD_REFRESHRATE     = 0x040000,
  DDSD_LINEARSIZE      = 0x080000,
  DDSD_TEXTURESTAGE    = 0x100000,
  DDSD_FVF             = 0x200000,
  DDSD_SRCVBHANDLE     = 0x400000,
  DDSD_DEPTH           = 0x800000
};
enum DDS_PIXELFORMAT_FLAGS : uint32 {
  DDPF_ALPHAPIXELS       = 0x00001,
  DDPF_ALPHA             = 0x00002,
  DDPF_FOURCC            = 0x00004,
  DDPF_PALETTEINDEXED4   = 0x00008,
  DDPF_PALETTEINDEXEDTO8 = 0x00010,
  DDPF_PALLENEINDEXED8   = 0x00020,
  DDPF_RGB               = 0x00040,
  DDPF_COMPRESSED        = 0x00080,
  DDPF_RGBTOYUV          = 0x00100,
  DDPF_YUV               = 0x00200,
  DDPF_ZBUFFER           = 0x00400,
  DDPF_PALETTEINDEXED1   = 0x00800,
  DDPF_PALETTEINDEXED2   = 0x01000,
  DDPF_ZPIXELS           = 0x02000,
  DDPF_STENCILBUFFER     = 0x04000,
  DDPF_ALPHAPREMULT      = 0x08000,
  DDPF_LUMINANCE         = 0x20000,
  DDPF_BUMPLUMINANCE     = 0x40000,
  DDPF_BUMPDUDV          = 0x80000
};
enum DDS_CAPS : uint32 {
  DDSCAPS_ALPHA           = 0x00000002,
  DDSCAPS_BACKBUFFER      = 0x00000004,
  DDSCAPS_COMPLEX         = 0x00000008,
  DDSCAPS_FLIP            = 0x00000010,
  DDSCAPS_FRONTBUFFER     = 0x00000020,
  DDSCAPS_OFFSCREENPLAIN  = 0x00000040,
  DDSCAPS_OVERLAY         = 0x00000080,
  DDSCAPS_PALETTE         = 0x00000100,
  DDSCAPS_PRIMARYSURFACE  = 0x00000200,
  DDSCAPS_SYSTEMMEMORY    = 0x00000800,
  DDSCAPS_TEXTURE         = 0x00001000,
  DDSCAPS_3DDEVICE        = 0x00002000,
  DDSCAPS_VIDEOMEMORY     = 0x00004000,
  DDSCAPS_VISIBLE         = 0x00008000,
  DDSCAPS_WRITEONLY       = 0x00010000,
  DDSCAPS_ZBUFFER         = 0x00020000,
  DDSCAPS_OWNDC           = 0x00040000,
  DDSCAPS_LIVEVIDEO       = 0x00080000,
  DDSCAPS_HWCODEC         = 0x00100000,
  DDSCAPS_MODEX           = 0x00200000,
  DDSCAPS_MIPMAP          = 0x00400000,
  DDSCAPS_ALLOCONLOAD     = 0x04000000,
  DDSCAPS_VIDEOPORT       = 0x08000000,
  DDSCAPS_LOCALVIDMEM     = 0x10000000,
  DDSCAPS_NONLOCALVIDMEM  = 0x20000000,
  DDSCAPS_STANDARDVGAMODE = 0x40000000,
  DDSCAPS_OPTIMIZED       = 0x80000000
};
enum DDS_CAPS2 : uint32 {
  DDSCAPS2_HINTDYNAMIC           = 0x00000004,
  DDSCAPS2_HINTSTATIC            = 0x00000008,
  DDSCAPS2_TEXTUREMANAGE         = 0x00000010,
  DDSCAPS2_OPAQUE                = 0x00000080,
  DDSCAPS2_HINTANTIALIASING      = 0x00000100,
  DDSCAPS2_CUBEMAP               = 0x00000200,
  DDSCAPS2_CUBEMAP_POSITIVEX     = 0x00000400,
  DDSCAPS2_CUBEMAP_NEGATIVEX     = 0x00000800,
  DDSCAPS2_CUBEMAP_POSITIVEY     = 0x00001000,
  DDSCAPS2_CUBEMAP_NEGATIVEY     = 0x00002000,
  DDSCAPS2_CUBEMAP_POSITIVEZ     = 0x00004000,
  DDSCAPS2_CUBEMAP_NEGATIVEZ     = 0x00008000,
  DDSCAPS2_MIPMAPSUBLEVEL        = 0x00010000,
  DDSCAPS2_D3DTEXTUREMANAGE      = 0x00020000,
  DDSCAPS2_DONOTPERSIST          = 0x00040000,
  DDSCAPS2_STEREOSURFACELEFT     = 0x00080000,
  DDSCAPS2_VOLUME                = 0x00200000,
  DDSCAPS2_NOTUSERLOCKABLE       = 0x00400000,
  DDSCAPS2_POINTS                = 0x00800000,
  DDSCAPS2_RTPATCHES             = 0x01000000,
  DDSCAPS2_NPATCHES              = 0x02000000,
  DDSCAPS2_DISCARDBACKBUFFER     = 0x10000000,
  DDSCAPS2_ENABLEALPHACHANNEL    = 0x20000000,
  DDSCAPS2_EXTENDEDFORMATPRIMARY = 0x40000000,
  DDSCAPS2_ADDITIONALPRIMARY     = 0x80000000
};
struct DDS_PIXELFORMAT {
  uint32 size = 32;
  uint32 flags = (DDS_PIXELFORMAT_FLAGS::DDPF_RGB | DDS_PIXELFORMAT_FLAGS::DDPF_ALPHAPIXELS);
  uint32 fourCC = 0;
  uint32 rgbBitCount = 32;
  uint32 rBitMask = 0x00ff0000;
  uint32 gBitMask = 0x0000ff00;
  uint32 bBitMask = 0x000000ff;
  uint32 aBitMask = 0xff000000;
};
struct DDS_HEADER {
  uint32 magic = DDS_MAGIC;
  uint32 size = 124;
  uint32 flags = (DDS_HEADER_FLAGS::DDSD_PIXELFORMAT | DDS_HEADER_FLAGS::DDSD_WIDTH | DDS_HEADER_FLAGS::DDSD_HEIGHT | DDS_HEADER_FLAGS::DDSD_CAPS);
  uint32 height = 0;
  uint32 width = 0;
  uint32 pitchOrLinearSize = 0;
  uint32 depth = 0;
  uint32 mipMapCount = 0;
  uint32 reserved[11] {0,0,0,0,0,0,0,0,0,0,0};
  DDS_PIXELFORMAT ddspf;
  uint32 caps = (DDS_CAPS::DDSCAPS_TEXTURE | DDS_CAPS::DDSCAPS_COMPLEX | DDS_CAPS::DDSCAPS_ALPHA);
  uint32 caps2 = (DDS_CAPS2::DDSCAPS2_CUBEMAP | DDS_CAPS2::DDSCAPS2_CUBEMAP_POSITIVEX | DDS_CAPS2::DDSCAPS2_CUBEMAP_NEGATIVEX | DDS_CAPS2::DDSCAPS2_CUBEMAP_POSITIVEY | DDS_CAPS2::DDSCAPS2_CUBEMAP_NEGATIVEY | DDS_CAPS2::DDSCAPS2_CUBEMAP_POSITIVEZ | DDS_CAPS2::DDSCAPS2_CUBEMAP_NEGATIVEZ);
  uint32 caps3 = 0;
  uint32 caps4 = 0;
  uint32 reserved2 = 0;
};

void logLuvDDSFromFloats(const std::vector<std::vector<float32>>& src, std::vector<uint8>& dst);
void logLuvDDSFromFloats(const std::vector<std::vector<std::vector<float32>>>& src, std::vector<uint8>& dst);


/*    SPECULAR
   [ '0x20534444',        magic
     '0x0000007c',        size              = 124
     '0x00021007',        flags             = DDSD_MIPMAPCOUNT | DDSD_PIXELFORMAT | DDSD_WIDTH | DDSD_HEIGHT | DDSD_CAPS
     '0x00000080',        height            = 128
     '0x00000080',        width             = 128
     '0x00000000',        pitchOrLinearSize = 0 (not set)
     '0x00000000',        depth             = 0 (not set)
     '0x00000008',        mipMapCount       = 8
     '0x00000000',        reserved
     '0x00000000',        reserved
     '0x00000000',        reserved
     '0x00000000',        reserved
     '0x00000000',        reserved
     '0x00000000',        reserved
     '0x00000000',        reserved
     '0x00000000',        reserved
     '0x00000000',        reserved
     '0x00000000',        reserved
     '0x00000000',        reserved
     '0x00000020', ddspf  size              = 32
     '0x00000041', ddspf  flags             = DDPF_RGB | DDPF_ALPHAPIXELS
     '0x00000000', ddspf  fourCC            = 0 (not set)
     '0x00000020', ddspf  rgbBitCount       = 32
     '0x00ff0000', ddspf  rBitMask
     '0x0000ff00', ddspf  gBitMask
     '0x000000ff', ddspf  bBitMask
     '0xff000000', ddspf  aBitMask
     '0x0040100a',        caps              = DDSCAPS_MIPMAP | DDSCAPS_TEXTURE | DDSCAPS_COMPLEX | DDSCAPS_ALPHA
     '0x0000fe00',        caps2             = DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_POSITIVEX | DDSCAPS2_CUBEMAP_NEGATIVEX | DDSCAPS2_CUBEMAP_POSITIVEY | DDSCAPS2_CUBEMAP_NEGATIVEY | DDSCAPS2_CUBEMAP_POSITIVEZ | DDSCAPS2_CUBEMAP_NEGATIVEZ
     '0x00000000',        caps3
     '0x00000000',        caps4
     '0x00000000' ]       reserved2
 */

/*    IRRADIANCE
   [ '0x20534444',        magic
     '0x0000007c',        size              = 124
     '0x00001007',        flags             = DDSD_PIXELFORMAT | DDSD_WIDTH | DDSD_HEIGHT | DDSD_CAPS
     '0x00000040',        height            = 64
     '0x00000040',        width             = 64
     '0x00000000',        pitchOrLinearSize = 0 (not set)
     '0x00000000',        depth             = 0 (not set)
     '0x00000000',        mipMapCount       = 0 (not set)
     '0x00000000',        reserved
     '0x00000000',        reserved
     '0x00000000',        reserved
     '0x00000000',        reserved
     '0x00000000',        reserved
     '0x00000000',        reserved
     '0x00000000',        reserved
     '0x00000000',        reserved
     '0x00000000',        reserved
     '0x00000000',        reserved
     '0x00000000',        reserved
     '0x00000020', ddspf  size              = 32
     '0x00000041', ddspf  flags             = DDPF_RGB | DDPF_ALPHAPIXELS
     '0x00000000', ddspf  fourCC            = 0 (not set)
     '0x00000020', ddspf  rgbBitCount       = 32
     '0x00ff0000', ddspf  rBitMask
     '0x0000ff00', ddspf  gBitMask
     '0x000000ff', ddspf  bBitMask
     '0xff000000', ddspf  aBitMask
     '0x0000100a',        caps              = DDSCAPS_TEXTURE | DDSCAPS_COMPLEX | DDSCAPS_ALPHA
     '0x0000fe00',        caps2             = DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_POSITIVEX | DDSCAPS2_CUBEMAP_NEGATIVEX | DDSCAPS2_CUBEMAP_POSITIVEY | DDSCAPS2_CUBEMAP_NEGATIVEY | DDSCAPS2_CUBEMAP_POSITIVEZ | DDSCAPS2_CUBEMAP_NEGATIVEZ
     '0x00000000',        caps3
     '0x00000000',        caps4
     '0x00000000' ]       reserved2
 */




#endif //CUBEMAPGEN_DDSFILE_H

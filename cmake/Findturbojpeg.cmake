find_path(turbojpeg_PREFIX
    NAMES "include/turbojpeg.h"
    PATHS $ENV{turbojpeg_HOME}
          $ENV{EXTERNLIBS}/libjpeg-turbo64
          $ENV{EXTERNLIBS}/libjpeg-turbo
          ~/Library/Frameworks
          /Library/Frameworks
          /usr/local/opt/jpeg-turbo
          /usr/local
          /usr
          /sw
          /opt/local
          /opt/csw
          /opt)

find_path(turbojpeg_INCLUDE_DIR "turbojpeg.h"
    HINTS ${turbojpeg_PREFIX}/include
    PATHS $ENV{turbojpeg_HOME}/include
          $ENV{EXTERNLIBS}/libjpeg-turbo64/include
          $ENV{EXTERNLIBS}/libjpeg-turbo/include
          ~/Library/Frameworks/include
          /Library/Frameworks/include
          /usr/local/opt/jpeg-turbo/include
          /usr/local/include
          /usr/include
          /sw/include
          /opt/local/include
          /opt/csw/include
          /opt/include)

find_library(turbojpeg_LIBRARY
    NAMES turbojpeg
    HINTS ${turbojpeg_PREFIX}/lib ${turbojpeg_PREFIX}/lib64
    PATHS $ENV{turbojpeg_HOME}
          $ENV{EXTERNLIBS}/libjpeg-turbo64
          $ENV{EXTERNLIBS}/libjpeg-turbo
          ~/Library/Frameworks
          /Library/Frameworks
          /usr/local
          /usr/local/opt/jpeg-turbo
          /usr
          /sw
          /opt/local
          /opt/csw
          /opt
          PATH_SUFFIXES lib lib64)

if (turbojpeg_INCLUDE_DIR)
  if (EXISTS "${turbojpeg_INCLUDE_DIR}/jconfig-64.h")
    set(_version_header "${turbojpeg_INCLUDE_DIR}/jconfig-64.h")
  elseif (EXISTS "${turbojpeg_INCLUDE_DIR}/jconfig.h")
    set(_version_header "${turbojpeg_INCLUDE_DIR}/jconfig.h")
  elseif (EXISTS "${turbojpeg_INCLUDE_DIR}/x86_64-linux-gnu/jconfig.h")
    set(_version_header "${turbojpeg_INCLUDE_DIR}/x86_64-linux-gnu/jconfig.h")
  else()
    set(_version_header)
    message(STATUS "Could not find 'jconfig.h' to check turbojpeg version")
  endif()
endif()

if (_version_header)
  file(READ "${_version_header}" _header)
  if (_header)
    string(REGEX REPLACE ".*#define[\t ]+LIBJPEG_TURBO_VERSION[\t ]+([0-9.]+).*" "\\1" turbojpeg_VERSION "${_header}")
  endif()
  unset(_header)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(turbojpeg REQUIRED_VARS turbojpeg_LIBRARY turbojpeg_INCLUDE_DIR VERSION_VAR turbojpeg_VERSION)

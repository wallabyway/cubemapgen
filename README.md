### Requirements

These must be installed on the system. They won't be installed automatically.

1. CMake >= 3.10
1. OpenEXR == 2.2.0
1. TurboJPEG
1. libpng

### Installing

Assuming requirements are met, clone the repo, then run

```
npm install
```

from root of repo. This will fail if there are build errors, e.g., dependencies could not be found.

### Use

The module has a single function on it, `sphereToCubes(buffer, transform)`. It returns an array of buffers, the first is the specular DDS, and the second is the irradiance DDS.

See test.js for example use.

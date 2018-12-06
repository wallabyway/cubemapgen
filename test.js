'use strict';

const cubemapgen = require('./index');

function main() {
  const fs = require('fs');
  const { PerformanceObserver, performance } = require('perf_hooks');
  const obs = new PerformanceObserver((items) => {
    items.getEntries().forEach((e) => {
      console.log(`${e.name}: ${e.duration}`);
    });
  });
  obs.observe({ entryTypes: ['measure'] });

  performance.mark('A');
  let bufIn = fs.readFileSync('./data/riverbank.exr');
  performance.mark('B');
  let out = cubemapgen.sphereToCubes(bufIn, [1,0,0,0,1,0,0,0,1]);
  performance.mark('C');
  fs.writeFileSync('./data/riverbank-specular.dds', out[0]);
  fs.writeFileSync('./data/riverbank-irradiance.dds', out[1]);
  performance.mark('D');

  performance.measure('Reading', 'A', 'B');
  performance.measure('Projecting', 'B', 'C');
  performance.measure('Writing', 'C', 'D');
}

main();

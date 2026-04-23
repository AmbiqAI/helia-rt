# Changelog

## [1.11.0](https://github.com/AmbiqAI/helia-rt/compare/heliaRT-v1.10.1...heliaRT-v1.11.0) (2026-04-23)


### Features

* Add zephyr integration documentation ([c372d63](https://github.com/AmbiqAI/helia-rt/commit/c372d63d23767a1bc38b6761b74cdc8e9482c91d))

## [1.10.1](https://github.com/AmbiqAI/helia-rt/compare/heliaRT-v1.10.0...heliaRT-v1.10.1) (2026-04-11)


### Bug Fixes

* **ci:** remove stale zephyr-release job and replaced scripts ([#90](https://github.com/AmbiqAI/helia-rt/issues/90)) ([22472cd](https://github.com/AmbiqAI/helia-rt/commit/22472cdd4b5cbef333b7c7c94abfeb5919a5aab3))
* **ci:** use literal 'main' in docs.yml push trigger ([#92](https://github.com/AmbiqAI/helia-rt/issues/92)) ([3d71155](https://github.com/AmbiqAI/helia-rt/commit/3d7115521b5a22e37e66c7f6afce0e7a3e8afa9d))
* clear error message for private HELIA backend, document license ([#94](https://github.com/AmbiqAI/helia-rt/issues/94)) ([1bfae7b](https://github.com/AmbiqAI/helia-rt/commit/1bfae7bc1f42f24087d636098cff7e00b87348ff))

## [1.10.0](https://github.com/AmbiqAI/helia-rt/compare/heliaRT-v1.9.0...heliaRT-v1.10.0) (2026-04-10)


### Features

* Zephyr 3-backend default, unified release bundle, ambiq→helia rename ([#88](https://github.com/AmbiqAI/helia-rt/issues/88)) ([0341c50](https://github.com/AmbiqAI/helia-rt/commit/0341c508264f33eba6c127d7960731bacd9fec3e))

## [1.9.0](https://github.com/AmbiqAI/helia-rt/compare/heliaRT-v1.8.0...heliaRT-v1.9.0) (2026-04-10)


### Features

* **zephyr:** default raw module to CMSIS-NN with HELIA opt-in ([#86](https://github.com/AmbiqAI/helia-rt/issues/86)) ([d2abfda](https://github.com/AmbiqAI/helia-rt/commit/d2abfdae1e655eeca7101df215daa54387ab2b2a))

## [1.8.0](https://github.com/AmbiqAI/helia-rt/compare/heliaRT-v1.7.0...heliaRT-v1.8.0) (2026-04-08)


### Features

* add manual neuralSPOT asset packaging workflow ([#83](https://github.com/AmbiqAI/helia-rt/issues/83)) ([c5bad96](https://github.com/AmbiqAI/helia-rt/commit/c5bad96a62236053a5e6c1c56632f5fe2b4ee4b5))
* Allow manually generating builds including reference cmsis-nn implementation ([#80](https://github.com/AmbiqAI/helia-rt/issues/80)) ([720909a](https://github.com/AmbiqAI/helia-rt/commit/720909a277912768703ce83b5762cba3468e1432))

## [1.7.0](https://github.com/AmbiqAI/helia-rt/compare/heliaRT-v1.6.0...heliaRT-v1.7.0) (2025-11-05)


### Features

* Add cmsis optimized comparison operators ([#78](https://github.com/AmbiqAI/helia-rt/issues/78)) ([45f65e8](https://github.com/AmbiqAI/helia-rt/commit/45f65e849823aef1a51822089e241ca57688a5ed))
* Add optimized SUB operator for s8 and s16. ([#79](https://github.com/AmbiqAI/helia-rt/issues/79)) ([dffaa28](https://github.com/AmbiqAI/helia-rt/commit/dffaa285d3f8b5c7e59799a78a69c1ef3f5e0c13))
* Use reusable workflow. ([#74](https://github.com/AmbiqAI/helia-rt/issues/74)) ([366421f](https://github.com/AmbiqAI/helia-rt/commit/366421f6d4d6c9377dceceef2239b72d706ebe71))


### Bug Fixes

* Use latest ns-cmsis-nn w/ corrected arm_fully_connected_per_channel_s16 ([#76](https://github.com/AmbiqAI/helia-rt/issues/76)) ([f9a5ded](https://github.com/AmbiqAI/helia-rt/commit/f9a5ded7851fe63504a75447e960a8a0fe20d902))

## [1.6.0](https://github.com/AmbiqAI/helia-rt/compare/heliaRT-v1.5.0...heliaRT-v1.6.0) (2025-10-22)


### Features

* Push release artifacts. ([#66](https://github.com/AmbiqAI/helia-rt/issues/66)) ([b574659](https://github.com/AmbiqAI/helia-rt/commit/b5746599afe08fc2eeed933633a7fe601893dffa))
* Update Zephyr module  ([#73](https://github.com/AmbiqAI/helia-rt/issues/73)) ([04ac179](https://github.com/AmbiqAI/helia-rt/commit/04ac1794085b5af458a86793a8828764138ea464))
* Update Zephyr module. ([6244ff8](https://github.com/AmbiqAI/helia-rt/commit/6244ff87f7d70bfb52c79638c290cbaf3d1e19c2))


### Bug Fixes

* Add missing dependency for hexdump. ([d632c64](https://github.com/AmbiqAI/helia-rt/commit/d632c649baa707677cf420b87c676aa931935adc))
* Add missing dependency for hexdump. ([#67](https://github.com/AmbiqAI/helia-rt/issues/67)) ([d632c64](https://github.com/AmbiqAI/helia-rt/commit/d632c649baa707677cf420b87c676aa931935adc))

## [1.5.0](https://github.com/AmbiqAI/helia-rt/compare/heliaRT-v1.4.0...heliaRT-v1.5.0) (2025-10-01)


### Features

* Update GHA RP workflow. ([#59](https://github.com/AmbiqAI/helia-rt/issues/59)) ([d14d73e](https://github.com/AmbiqAI/helia-rt/commit/d14d73ed20f401c7d92411244516a4280e5cf22b))


### Bug Fixes

* Fixing release workflow ([#63](https://github.com/AmbiqAI/helia-rt/issues/63)) ([620b66c](https://github.com/AmbiqAI/helia-rt/commit/620b66cc814a566b60d136cc0091e8b0577c70ff))

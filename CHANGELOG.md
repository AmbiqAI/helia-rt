# Changelog

## [1.18.0](https://github.com/AmbiqAI/helia-rt/compare/helia-rt-v1.17.0...helia-rt-v1.18.0) (2026-06-21)


### Features

* **nsx:** Make heliaRT SoC compatibility wildcard ([#180](https://github.com/AmbiqAI/helia-rt/issues/180)) ([6bb086e](https://github.com/AmbiqAI/helia-rt/commit/6bb086e29bbbc90178eb07d532cbb09f8a3f357d))


### Bug Fixes

* Remove python ops resolver from runtime sources ([#176](https://github.com/AmbiqAI/helia-rt/issues/176)) ([595b892](https://github.com/AmbiqAI/helia-rt/commit/595b89276d450b4abc9a2851f74efbe8ce3b338b))

## [1.17.0](https://github.com/AmbiqAI/helia-rt/compare/helia-rt-v1.16.0...helia-rt-v1.17.0) (2026-06-11)


### Features

* add opt-in source profiles for recording/test helpers ([#173](https://github.com/AmbiqAI/helia-rt/issues/173)) ([c52bfdb](https://github.com/AmbiqAI/helia-rt/commit/c52bfdb52e896b6a7468c6df965ec38bca01d724))
* Add optional Ethos-U support to generic CMake builds ([#172](https://github.com/AmbiqAI/helia-rt/issues/172)) ([83c0829](https://github.com/AmbiqAI/helia-rt/commit/83c082901d24db7af456c6a769fe3e338d5c3ed5))
* update HELIA backend version on homepage ([d5220d1](https://github.com/AmbiqAI/helia-rt/commit/d5220d14e9bcf6de1c0f1b63fde2e5dbc3ead31f))


### Bug Fixes

* Remove redundant NS_CMSIS_NN compile define ([#174](https://github.com/AmbiqAI/helia-rt/issues/174)) ([ed596d3](https://github.com/AmbiqAI/helia-rt/commit/ed596d3a6bb9850489ca9542d713f3d1d6860803))
* Revert zephyr.md Source + CMSIS-NN path to correct cmsis-nn module ([797e72c](https://github.com/AmbiqAI/helia-rt/commit/797e72c3acc2411783598e1496fe41e87f9957df))
* Round 0.99 to 1 ([cc67923](https://github.com/AmbiqAI/helia-rt/commit/cc6792343ced1ae75c81ee7b20348e6fb7828414))

## [1.16.0](https://github.com/AmbiqAI/helia-rt/compare/helia-rt-v1.15.0...helia-rt-v1.16.0) (2026-05-17)


### Features

* Add cmsis optimized comparison operators ([#78](https://github.com/AmbiqAI/helia-rt/issues/78)) ([45f65e8](https://github.com/AmbiqAI/helia-rt/commit/45f65e849823aef1a51822089e241ca57688a5ed))
* add manual neuralSPOT asset packaging workflow ([#83](https://github.com/AmbiqAI/helia-rt/issues/83)) ([c5bad96](https://github.com/AmbiqAI/helia-rt/commit/c5bad96a62236053a5e6c1c56632f5fe2b4ee4b5))
* add NSX module type for heliaRT prebuilt libraries ([#99](https://github.com/AmbiqAI/helia-rt/issues/99)) ([967bf06](https://github.com/AmbiqAI/helia-rt/commit/967bf0630d47ffdee0c85a5ae3ef23240b0b706c))
* Add optimized SUB operator for s8 and s16. ([#79](https://github.com/AmbiqAI/helia-rt/issues/79)) ([dffaa28](https://github.com/AmbiqAI/helia-rt/commit/dffaa285d3f8b5c7e59799a78a69c1ef3f5e0c13))
* Add zephyr integration documentation ([c372d63](https://github.com/AmbiqAI/helia-rt/commit/c372d63d23767a1bc38b6761b74cdc8e9482c91d))
* Allow manually generating builds including reference cmsis-nn implementation ([#80](https://github.com/AmbiqAI/helia-rt/issues/80)) ([720909a](https://github.com/AmbiqAI/helia-rt/commit/720909a277912768703ce83b5762cba3468e1432))
* **cmake:** unified source-of-truth for heliaRT ([#147](https://github.com/AmbiqAI/helia-rt/issues/147) Phase 1) ([#148](https://github.com/AmbiqAI/helia-rt/issues/148)) ([985139b](https://github.com/AmbiqAI/helia-rt/commit/985139b7832038cba2891ab92ba9db6381606175))
* **cmsis-pack:** adopt Cclass/Cgroup/Csub/Cvariant convention; gate HELIA on ns-cmsis-nn ([6924641](https://github.com/AmbiqAI/helia-rt/commit/6924641ef3a5274b610df71b1a1f3edad6420637))
* **cmsis-pack:** pin heliaCORE cross-pack require to Cvariant=Source, Cversion=7.24.2 ([7e2bae3](https://github.com/AmbiqAI/helia-rt/commit/7e2bae3e5065486d5bcbbf63237c2a672d249005))
* **cmsis-pack:** pin heliaCORE Cversion=7.24.1 and add pdsc contract guard ([32d471a](https://github.com/AmbiqAI/helia-rt/commit/32d471aa0daab23e659b9e232aa2824ae131b41f))
* **helia:** replant helia-rt on tflm/main (2.3a) ([690a2d7](https://github.com/AmbiqAI/helia-rt/commit/690a2d72517522b8d7988ff7ee9ea01c9715491f))
* land ATfE toolchain + NSX-module-ship on main ([#109](https://github.com/AmbiqAI/helia-rt/issues/109)) ([f7aa52d](https://github.com/AmbiqAI/helia-rt/commit/f7aa52d1d68a78ec8edb6afd062e70961ebe772a))
* Normalize heliaRT packaging identifiers ([#164](https://github.com/AmbiqAI/helia-rt/issues/164)) ([ef2f507](https://github.com/AmbiqAI/helia-rt/commit/ef2f507866a0d49fa4c6f363c1546ae0413f0df7))
* **nsx:** make nsx::heliart backend-selectable via NSX_HELIART_BACKEND ([#158](https://github.com/AmbiqAI/helia-rt/issues/158)) ([993d8b5](https://github.com/AmbiqAI/helia-rt/commit/993d8b5c4b8e3a53207409a000675ace7849c832))
* **nsx:** switch nsx-heliart to source build via SSoT ([#147](https://github.com/AmbiqAI/helia-rt/issues/147) Phase 2) ([4fb6191](https://github.com/AmbiqAI/helia-rt/commit/4fb6191cdae680de52cd9650739285161101c2c4))
* Push release artifacts. ([#66](https://github.com/AmbiqAI/helia-rt/issues/66)) ([b574659](https://github.com/AmbiqAI/helia-rt/commit/b5746599afe08fc2eeed933633a7fe601893dffa))
* **tools:** CMSIS-Pack builder consuming dump_manifest.cmake (Phase 3 of [#147](https://github.com/AmbiqAI/helia-rt/issues/147)) ([2ccb6a5](https://github.com/AmbiqAI/helia-rt/commit/2ccb6a5b5d0d492216af913fbe2757b8aec9ffda))
* Update GHA RP workflow. ([#59](https://github.com/AmbiqAI/helia-rt/issues/59)) ([d14d73e](https://github.com/AmbiqAI/helia-rt/commit/d14d73ed20f401c7d92411244516a4280e5cf22b))
* Update Zephyr module  ([#73](https://github.com/AmbiqAI/helia-rt/issues/73)) ([04ac179](https://github.com/AmbiqAI/helia-rt/commit/04ac1794085b5af458a86793a8828764138ea464))
* Update Zephyr module. ([6244ff8](https://github.com/AmbiqAI/helia-rt/commit/6244ff87f7d70bfb52c79638c290cbaf3d1e19c2))
* Use reusable workflow. ([#74](https://github.com/AmbiqAI/helia-rt/issues/74)) ([366421f](https://github.com/AmbiqAI/helia-rt/commit/366421f6d4d6c9377dceceef2239b72d706ebe71))
* Zephyr 3-backend default, unified release bundle, ambiq→helia rename ([#88](https://github.com/AmbiqAI/helia-rt/issues/88)) ([0341c50](https://github.com/AmbiqAI/helia-rt/commit/0341c508264f33eba6c127d7960731bacd9fec3e))
* **zephyr:** consume cmake/helia_rt_sources.cmake SSoT ([#147](https://github.com/AmbiqAI/helia-rt/issues/147) Phase 4) ([ecc5b37](https://github.com/AmbiqAI/helia-rt/commit/ecc5b37a95b4a7bd00b32be570f1eb7e3c7978e3))
* **zephyr:** default raw module to CMSIS-NN with HELIA opt-in ([#86](https://github.com/AmbiqAI/helia-rt/issues/86)) ([d2abfda](https://github.com/AmbiqAI/helia-rt/commit/d2abfdae1e655eeca7101df215daa54387ab2b2a))
* **zephyr:** west-managed modules, HELIA default backend, auto-configure Kconfig ([#153](https://github.com/AmbiqAI/helia-rt/issues/153)) ([00e2299](https://github.com/AmbiqAI/helia-rt/commit/00e22995bebb7b8f0d279952a322b70553aba7ec))


### Bug Fixes

* Add missing dependency for hexdump. ([d632c64](https://github.com/AmbiqAI/helia-rt/commit/d632c649baa707677cf420b87c676aa931935adc))
* Add missing dependency for hexdump. ([#67](https://github.com/AmbiqAI/helia-rt/issues/67)) ([d632c64](https://github.com/AmbiqAI/helia-rt/commit/d632c649baa707677cf420b87c676aa931935adc))
* allow helia-owned pack helpers in invariant ([3a088a8](https://github.com/AmbiqAI/helia-rt/commit/3a088a8000d48909a306514472af1fd22647a832))
* **armclang:** remove -fshort-wchar from FLAGS_ARMC ([#104](https://github.com/AmbiqAI/helia-rt/issues/104)) ([221d1b3](https://github.com/AmbiqAI/helia-rt/commit/221d1b3dceec73f8ff05458a69ad46a5b75d8bfb))
* **atfe:** make Corstone-300 ATfE matrix green end-to-end ([#115](https://github.com/AmbiqAI/helia-rt/issues/115)) ([698a0b8](https://github.com/AmbiqAI/helia-rt/commit/698a0b8df1e1eadd817fdd5e4263638714a32ac2))
* **ci:** bump CI base image to debian bookworm for ATfE 22.1.0 ([#112](https://github.com/AmbiqAI/helia-rt/issues/112)) ([f79c753](https://github.com/AmbiqAI/helia-rt/commit/f79c753ff99b60c488372025bb3da138fc47e13b))
* **ci:** install newlib headers for arm-none-eabi cross-build ([eb72a3e](https://github.com/AmbiqAI/helia-rt/commit/eb72a3e180b3dad7deb96b59efc0295ab11f3165))
* **ci:** remove stale zephyr-release job and replaced scripts ([#90](https://github.com/AmbiqAI/helia-rt/issues/90)) ([22472cd](https://github.com/AmbiqAI/helia-rt/commit/22472cdd4b5cbef333b7c7c94abfeb5919a5aab3))
* **ci:** restore helia_release.yml workflow (dropped in PR [#90](https://github.com/AmbiqAI/helia-rt/issues/90)) ([#102](https://github.com/AmbiqAI/helia-rt/issues/102)) ([cf4b218](https://github.com/AmbiqAI/helia-rt/commit/cf4b218e4e5003e15484aaefc79ae00085787f32))
* **ci:** use cmake --build -j (not -- -j) for nsx-cross-build ([a651ab2](https://github.com/AmbiqAI/helia-rt/commit/a651ab2ef26067aa429c868ecc8a103fed448dfe))
* **ci:** use literal 'main' in docs.yml push trigger ([#92](https://github.com/AmbiqAI/helia-rt/issues/92)) ([3d71155](https://github.com/AmbiqAI/helia-rt/commit/3d7115521b5a22e37e66c7f6afce0e7a3e8afa9d))
* clear error message for private HELIA backend, document license ([#94](https://github.com/AmbiqAI/helia-rt/issues/94)) ([1bfae7b](https://github.com/AmbiqAI/helia-rt/commit/1bfae7bc1f42f24087d636098cff7e00b87348ff))
* **cmake:** expose helia per-kernel optimization defines (SPEED|SIZE) ([#159](https://github.com/AmbiqAI/helia-rt/issues/159)) ([de0d528](https://github.com/AmbiqAI/helia-rt/commit/de0d5288df95ee6234739267aae121a92959484b))
* **cmsis_pack:** drop attr="config" on preIncludeGlobal stubs ([ca6e82f](https://github.com/AmbiqAI/helia-rt/commit/ca6e82fab89ef6858eabbd5ad44782096403f3a9))
* **corstone-300:** add ATfE-specific linker script and picolibc semihost link ([#114](https://github.com/AmbiqAI/helia-rt/issues/114)) ([36788a2](https://github.com/AmbiqAI/helia-rt/commit/36788a2b4822fb3310135861d3a530819eece284))
* Fixing release workflow ([#63](https://github.com/AmbiqAI/helia-rt/issues/63)) ([620b66c](https://github.com/AmbiqAI/helia-rt/commit/620b66cc814a566b60d136cc0091e8b0577c70ff))
* **nsx:** make nsx-module robust for flattened bundle layouts ([#116](https://github.com/AmbiqAI/helia-rt/issues/116)) ([73a3fe9](https://github.com/AmbiqAI/helia-rt/commit/73a3fe941307dcc13897a9026682b3cec7ac739c))
* Use latest ns-cmsis-nn w/ corrected arm_fully_connected_per_channel_s16 ([#76](https://github.com/AmbiqAI/helia-rt/issues/76)) ([f9a5ded](https://github.com/AmbiqAI/helia-rt/commit/f9a5ded7851fe63504a75447e960a8a0fe20d902))
* **zephyr:** define CMSIS_NN for HELIA backend to prevent kernel redefinition ([#139](https://github.com/AmbiqAI/helia-rt/issues/139)) ([6ca8157](https://github.com/AmbiqAI/helia-rt/commit/6ca8157221c77c3c3c4fd0e83c43e757f3a5f025))


### Refactoring

* **cmsis-pack:** drop dead common_set + surface cmake stderr on failure ([6a40e92](https://github.com/AmbiqAI/helia-rt/commit/6a40e92f0a36e18f91b5617b6d88a8aeb130da68))

## [1.15.0](https://github.com/AmbiqAI/helia-rt/compare/heliaRT-v1.14.0...heliaRT-v1.15.0) (2026-05-16)


### Features

* **nsx:** make nsx::heliart backend-selectable via NSX_HELIART_BACKEND ([#158](https://github.com/AmbiqAI/helia-rt/issues/158)) ([993d8b5](https://github.com/AmbiqAI/helia-rt/commit/993d8b5c4b8e3a53207409a000675ace7849c832))
* **zephyr:** west-managed modules, HELIA default backend, auto-configure Kconfig ([#153](https://github.com/AmbiqAI/helia-rt/issues/153)) ([00e2299](https://github.com/AmbiqAI/helia-rt/commit/00e22995bebb7b8f0d279952a322b70553aba7ec))


### Bug Fixes

* **cmake:** expose helia per-kernel optimization defines (SPEED|SIZE) ([#159](https://github.com/AmbiqAI/helia-rt/issues/159)) ([de0d528](https://github.com/AmbiqAI/helia-rt/commit/de0d5288df95ee6234739267aae121a92959484b))

## [1.14.0](https://github.com/AmbiqAI/helia-rt/compare/heliaRT-v1.13.1...heliaRT-v1.14.0) (2026-05-15)


### Features

* **cmake:** unified source-of-truth for heliaRT ([#147](https://github.com/AmbiqAI/helia-rt/issues/147) Phase 1) ([#148](https://github.com/AmbiqAI/helia-rt/issues/148)) ([985139b](https://github.com/AmbiqAI/helia-rt/commit/985139b7832038cba2891ab92ba9db6381606175))
* **cmsis-pack:** adopt Cclass/Cgroup/Csub/Cvariant convention; gate HELIA on ns-cmsis-nn ([6924641](https://github.com/AmbiqAI/helia-rt/commit/6924641ef3a5274b610df71b1a1f3edad6420637))
* **cmsis-pack:** pin heliaCORE cross-pack require to Cvariant=Source, Cversion=7.25.0 ([7e2bae3](https://github.com/AmbiqAI/helia-rt/commit/7e2bae3e5065486d5bcbbf63237c2a672d249005))
* **cmsis-pack:** pin heliaCORE Cversion=7.24.1 and add pdsc contract guard ([32d471a](https://github.com/AmbiqAI/helia-rt/commit/32d471aa0daab23e659b9e232aa2824ae131b41f))
* **nsx:** switch nsx-heliart to source build via SSoT ([#147](https://github.com/AmbiqAI/helia-rt/issues/147) Phase 2) ([4fb6191](https://github.com/AmbiqAI/helia-rt/commit/4fb6191cdae680de52cd9650739285161101c2c4))
* **tools:** CMSIS-Pack builder consuming dump_manifest.cmake (Phase 3 of [#147](https://github.com/AmbiqAI/helia-rt/issues/147)) ([2ccb6a5](https://github.com/AmbiqAI/helia-rt/commit/2ccb6a5b5d0d492216af913fbe2757b8aec9ffda))
* **zephyr:** consume cmake/helia_rt_sources.cmake SSoT ([#147](https://github.com/AmbiqAI/helia-rt/issues/147) Phase 4) ([ecc5b37](https://github.com/AmbiqAI/helia-rt/commit/ecc5b37a95b4a7bd00b32be570f1eb7e3c7978e3))


### Bug Fixes

* **ci:** install newlib headers for arm-none-eabi cross-build ([eb72a3e](https://github.com/AmbiqAI/helia-rt/commit/eb72a3e180b3dad7deb96b59efc0295ab11f3165))
* **ci:** use cmake --build -j (not -- -j) for nsx-cross-build ([a651ab2](https://github.com/AmbiqAI/helia-rt/commit/a651ab2ef26067aa429c868ecc8a103fed448dfe))
* **cmsis_pack:** drop attr="config" on preIncludeGlobal stubs ([ca6e82f](https://github.com/AmbiqAI/helia-rt/commit/ca6e82fab89ef6858eabbd5ad44782096403f3a9))


### Refactoring

* **cmsis-pack:** drop dead common_set + surface cmake stderr on failure ([6a40e92](https://github.com/AmbiqAI/helia-rt/commit/6a40e92f0a36e18f91b5617b6d88a8aeb130da68))

## [1.13.1](https://github.com/AmbiqAI/helia-rt/compare/heliaRT-v1.13.0...heliaRT-v1.13.1) (2026-05-12)


### Bug Fixes

* **zephyr:** define CMSIS_NN for HELIA backend to prevent kernel redefinition ([#139](https://github.com/AmbiqAI/helia-rt/issues/139)) ([6ca8157](https://github.com/AmbiqAI/helia-rt/commit/6ca8157221c77c3c3c4fd0e83c43e757f3a5f025))

## [1.13.0](https://github.com/AmbiqAI/helia-rt/compare/heliaRT-v1.12.2...heliaRT-v1.13.0) (2026-05-11)


### Features

* **helia:** replant helia-rt on tflm/main (2.3a) ([690a2d7](https://github.com/AmbiqAI/helia-rt/commit/690a2d72517522b8d7988ff7ee9ea01c9715491f))

## [1.12.2](https://github.com/AmbiqAI/helia-rt/compare/heliaRT-v1.12.1...heliaRT-v1.12.2) (2026-04-30)


### Bug Fixes

* **nsx:** make nsx-module robust for flattened bundle layouts ([#116](https://github.com/AmbiqAI/helia-rt/issues/116)) ([73a3fe9](https://github.com/AmbiqAI/helia-rt/commit/73a3fe941307dcc13897a9026682b3cec7ac739c))

## [1.12.1](https://github.com/AmbiqAI/helia-rt/compare/heliaRT-v1.12.0...heliaRT-v1.12.1) (2026-04-29)


### Bug Fixes

* **atfe:** make Corstone-300 ATfE matrix green end-to-end ([#115](https://github.com/AmbiqAI/helia-rt/issues/115)) ([698a0b8](https://github.com/AmbiqAI/helia-rt/commit/698a0b8df1e1eadd817fdd5e4263638714a32ac2))
* **ci:** bump CI base image to debian bookworm for ATfE 22.1.0 ([#112](https://github.com/AmbiqAI/helia-rt/issues/112)) ([f79c753](https://github.com/AmbiqAI/helia-rt/commit/f79c753ff99b60c488372025bb3da138fc47e13b))
* **corstone-300:** add ATfE-specific linker script and picolibc semihost link ([#114](https://github.com/AmbiqAI/helia-rt/issues/114)) ([36788a2](https://github.com/AmbiqAI/helia-rt/commit/36788a2b4822fb3310135861d3a530819eece284))

## [1.12.0](https://github.com/AmbiqAI/helia-rt/compare/heliaRT-v1.11.2...heliaRT-v1.12.0) (2026-04-25)


### Features

* land ATfE toolchain + NSX-module-ship on main ([#109](https://github.com/AmbiqAI/helia-rt/issues/109)) ([f7aa52d](https://github.com/AmbiqAI/helia-rt/commit/f7aa52d1d68a78ec8edb6afd062e70961ebe772a))

## [1.11.2](https://github.com/AmbiqAI/helia-rt/compare/heliaRT-v1.11.1...heliaRT-v1.11.2) (2026-04-24)


### Bug Fixes

* **armclang:** remove -fshort-wchar from FLAGS_ARMC ([#104](https://github.com/AmbiqAI/helia-rt/issues/104)) ([221d1b3](https://github.com/AmbiqAI/helia-rt/commit/221d1b3dceec73f8ff05458a69ad46a5b75d8bfb))

## [1.11.1](https://github.com/AmbiqAI/helia-rt/compare/heliaRT-v1.11.0...heliaRT-v1.11.1) (2026-04-24)


### Bug Fixes

* **ci:** restore helia_release.yml workflow (dropped in PR [#90](https://github.com/AmbiqAI/helia-rt/issues/90)) ([#102](https://github.com/AmbiqAI/helia-rt/issues/102)) ([cf4b218](https://github.com/AmbiqAI/helia-rt/commit/cf4b218e4e5003e15484aaefc79ae00085787f32))

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

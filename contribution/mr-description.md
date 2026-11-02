<!--
DRAFT. Not posted anywhere. Fill in or delete every [bracketed] item before use.
Suggested MR title: Draft: Optional Metal acceleration of spatial convolution in Richardson-Lucy (macOS)
Follows .gitlab/merge_request_templates/default.md in the Siril repository.
-->

### Description

On macOS, this adds an optional Metal implementation of single-precision 2D spatial convolution (`img_t<float>::conv2`). It is used by the non-FFT ("naive") Richardson-Lucy branch, which Siril selects when the PSF size is smaller than the FFT cutoff preference (`fftw_conv_fft_cutoff`, default 15). If Metal is not available, the workload is small, or anything fails, the existing CPU code runs unchanged.

Related issues: [link to the feature issue; #1450 discusses GPU offload in general and is closed].

This MR is two commits on top of upstream master 7765e18: "Add optional Metal spatial convolution for Richardson-Lucy deconvolution on macOS" and "Log Metal convolution dispatch once at debug level". A separate one-commit change (`psf-size-fix`, see "Related separate change") is needed to test the loaded-PSF workflow but is not part of this MR.

The default 15x15 PSF uses the FFT branch (`ks < fft_cutoff` is false for 15), so the default behaviour is not changed by this MR. Only deconvolution with a smaller PSF (7 to 13 pixels with the default cutoff) can reach the new code.

### Changes Made

What is accelerated:

- `img_t<float>::conv2` in `src/algos/img_t/image.hpp` first calls `siril_metal_convolve_float()` when `HAVE_METAL` is defined and all three types are `float`. If it returns 0, the existing CPU loop runs. If it returns 1, the result is mapped into the output.
- The Metal kernel (`src/algos/metal_convolution.mm`, compiled from source at run time) is a direct correlation with zero borders. It has the same orientation and accumulation order as the CPU code. Fast math stays off: `options.mathMode = MTLMathModeSafe` when built with an SDK for macOS 15 or later (and running on macOS 15 or later), otherwise `options.fastMathEnabled = NO`.

When Metal is used (all conditions must hold, otherwise CPU):

- kernel size is odd and between 7 and 31, and not larger than the image width or height
- `width * height * channels * kernel_size^2 >= 32 Mi` (32 * 1024 * 1024) multiply-adds
- the plane size and channel count fit in `INT_MAX` and the buffer fits `maxBufferLength`
- a Metal device exists, the pipeline built, and the command buffer completed

Anything else returns 0 with the output untouched.

Build integration:

- New meson feature option `metal` (default `auto`). On macOS it adds the `objcpp` language, links `Foundation` and `Metal`, defines `HAVE_METAL`, and compiles `src/algos/metal_convolution.mm`. The new files carry the full GPL-3.0-or-later header used by neighbouring files. On other platforms `auto` does nothing and `enabled` is an error. `-Dmetal=disabled` gives the current behaviour on macOS.
- Memory estimates grow slightly when `HAVE_METAL` is defined, for the naive RL branch only. `deconvolution_compute_mem_limits` counts 8 instead of 6 float channels (two extra float buffers) (`src/filters/deconvolution.c`), and `naive_richardson_lucy` uses `num_copies_required = 9` instead of 7 (`src/filters/deconvolution/deconvolve.cpp`). This is a conservative allowance for the Metal buffers and applies to every Metal-enabled build, including runs where Metal is not dispatched. It can cause slightly more image slicing. The values 8 and 9 are estimates; the constrained-memory path (7 versus 9 copies) was not exercised, because it only differs when roughly 30 to 39 MB of memory is free.
- New test `src/tests/metal_convolution_test.cpp`, registered as `metal_convolution` only when `have_metal` is true (see "How to test"). Builds without Metal, including Linux and Windows, get no new test target.
- When Metal first runs a convolution in a process, `siril_log_debug` prints `Metal convolution: device ..., WxH kN`. It appears only in debug-type builds (where `siril_log_debug` is active); there is no runtime switch, and it is printed once per process.

Results (Apple M5 only; see "Limits of the evidence"). The correctness and speed tables below were measured on the earlier, pre-rebase build of the same code; the rebase checks are listed after them.

- Correctness: 13 end-to-end `siril-cli` cases compared CPU and Metal builds of the same source. All outputs were finite and within `|cpu - metal| <= 2e-5 + 2e-5 * |cpu|` per pixel. The largest difference observed was 3.8e-6 (absolute). Cases include `-mul`, default gradient descent, `-tv`, `-fh`, mono and RGB, a real 1024x1024x3 crop, and fallback cases.
- Speed, warm Richardson-Lucy time (the logged `Execution time` of the second RL pass in the same process), 10 iterations, 1024x1024, interleaved CPU/Metal runs, two repeats:

| Case | CPU median (ms) | Metal median (ms) | Median of per-pair speedup (range over 10 pairs) |
|---|---|---|---|
| mono, 7x7 | 379 / 362 | 337 / 326 | 1.13 / 1.14 (1.02 to 1.28) |
| mono, 13x13 | 845 / 703 | 396 / 367 | 2.12 / 1.94 (1.73 to 2.62) |
| RGB, 7x7 | 513 / 426 | 400 / 369 | 1.29 / 1.15 (1.03 to 1.37) |
| RGB, 13x13 | 1120 / 960 | 496 / 427 | 2.15 / 2.28 (1.97 to 2.53) |

(each cell shows run 1 / run 2). In short: about break-even to a modest gain at 7x7, about 2x at 13x13 on this machine. Whole-process time includes start-up and I/O and improves less (13x13 mono: about 1.7x to 1.9x; 7x7: about 1.1x).
- The default 15x15 case (FFT branch, no Metal) measured 882 ms CPU build versus 933 ms Metal build in a single run. That is run-to-run variation on code that is identical in both builds, not an effect of this change.

Load caveat: the machine was shared and not idle during all measurements (a GUI Siril build, a long-running Python process using roughly 50 to 100 percent of one core, desktop processes). The 1-minute load average before each batch was between 3.4 and 8.1. CPU and Metal runs were interleaved (ABBA) to spread the load over both, so the ratios are more reliable than the absolute times, but all timings are indicative only. Full tables and limitations: `BENCHMARK-RESULTS.md` [attach, or paste the relevant sections; it is not part of the diff].

Checks on the final rebased branch (Apple M5):

- `metal=auto`: the `metal_convolution` test (1 of 1) passes on the real GPU, max error 0.
- `metal=disabled`: configure and build succeed; `meson test` reports "No tests defined".
- A debugoptimized build logged the `Metal convolution: device ...` line.
- Slicing forced with `[core] max_slice_size=512`: CPU versus Metal max difference 4.5e-8. The unsliced versus sliced difference (0.134) is existing tile-edge behaviour and is the same on the CPU and Metal paths.
- Rebased warm RL, mono 13x13: median about 1050 ms CPU versus about 461 ms Metal, at a load average of 3.5 to 4.8 (indicative only).

GTK build (gtk=true, metal=enabled, macOS arm64): configures, compiles and passes the unit test; the GUI starts and opens a 1024x1024 FITS image. Metal dispatch was confirmed in the GUI binary running a script (-s): 20 GPU command buffers for a 10-iteration k13 RL, output within 4.5e-8 of the CPU build. The GTK dialog (src/gui-gtk4/newdeconv.c:831,887) sets the same op_desc_deconvolve as the CLI command (src/core/command.c:2640), so it reaches the same deconvolve() code (confirmed by reading the source); it was not driven interactively.

### Limits of the evidence (what is not done or not tested)

- Measured on one machine: Apple M5, 8 GPU cores, 16 GB, macOS 27.2, release build with Homebrew LLVM 22. Other Apple GPUs and older macOS versions are untested. The ChangeLog notes Intel macOS support is dropped (!1054), so discrete GPUs are not a target, but they are also untested.
- Not built or run on Linux or Windows. On those platforms `HAVE_METAL` is undefined and the new test is not registered, so by reasoning the build is unchanged; this was not tried.
- The macOS CI job (`macos` in `.gitlab-ci.yml`) has not been run for this branch. It builds with SDK `MacOSX11.3.sdk` and uses plain `meson setup` without options, so `metal=auto` would enable this code in the CI build. I have not compiled against the 11.3 SDK. The `#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000` branch that selects `fastMathEnabled = NO` on older SDKs is reasoned, not compiled with an old SDK.
- Only the M5 GPU was tried; the real low-memory path (7 versus 9 copies) was not exercised.
- Each call copies input, kernel and output through new shared buffers; there are no persistent device buffers across iterations. The measured Metal times include this cost, and it limits the gain.
- GPU use is logged only at debug level, and only in debug-type builds (see above). In a release build, whether Metal was used is inferred from the conditions above and from timing.
- Only the single-precision spatial convolution is accelerated. The FFT branch, other deconvolution methods, and every other Siril operation are not. Kernels above 31 are not handled by Metal. The GUI and sequence processing were not tested beyond a smoke test (a 1024x1024 mono FITS, 7x7 Gaussian PSF, 10 iterations, in the GUI build); the benchmark runs are single-image CLI runs.
- Constrained-memory behaviour with the larger memory estimate was not tested beyond forced slicing (see above).
- Timings were taken on a loaded machine; no quiet-machine run exists.
- The `-stop` (stopping criterion) option of the naive RL branch never fires, with or without this change. That is an existing defect and is reported separately [link to the issue]. This MR does not change it, and the CPU/Metal comparison only shows that both behave the same.
- Separate processes with identical inputs sometimes produce outputs that differ by about 1 ulp in a few pixels on the CPU path alone (observed with `fftw_multithreaded=true`; cause not isolated).

### How to test

Build (macOS):

```sh
meson setup _build -Dmetal=enabled      # -Dmetal=disabled gives the CPU-only reference build (no tests defined)
ninja -C _build
meson test -C _build metal_convolution -v
```

The test (`src/tests/metal_convolution_test.cpp`, not Criterion based) checks `conv2` against a reference loop on synthetic data (mono and RGB, odd kernels 3 to 13, in-place use, mixed and double types, invalid arguments, four concurrent calls) and, when a Metal device exists, checks the direct Metal call. With no Metal device it exits 77 (skipped). `_build/src/tests/metal_convolution_test --benchmark` prints `conv2` timings.

End to end, compare a Metal-enabled and a Metal-disabled build on the same input. The PSF must be made with `makepsf ... -ks=13`. Do not use `rl -loadpsf=` unless the separate psf-size-fix is applied first (see below): without it, `args->ks` stays at 15, `rl` takes the FFT path, Metal is never used, and the output was intermittently all NaN on the CPU as well.

```
requires 1.4.0
set32bits
load image.fit
makepsf manual -gaussian -fwhm=3 -ks=13
rl -iters=10 -mul
save out
```

Compare the two output files pixel by pixel (tolerance used here: `2e-5 + 2e-5 * |cpu|`). The input must be large enough for the threshold: for 13x13, pixels times channels must be at least about 198,600 (32 Mi divided by 169), for example 512x512 mono or larger. The measurements above used 1024x1024. The earlier benchmark runs used `-loadpsf` with 7x7 and 13x13 Gaussian PSFs on a build that included the psf-size-fix. To see the Metal line, use a debug-type build (`-Dbuildtype=debugoptimized`).

### Related separate change

`psf-size-fix` is a separate one-commit MR ("Set kernel size in deconvolution args when loading a PSF": `args->ks = com.kernelsize;` in `load_kernel`, `src/filters/deconvolution.c`), submitted on its own [link / not yet submitted]. Without it, `rl -loadpsf=` with a PSF whose size differs from 15 leaves `args->ks` at 15, so the FFT branch is chosen and Metal is never used. The intermittent all-NaN output without the fix was inferred from reading the code and from reproduction on a metal-disabled build; I have not proven it on pristine upstream master.

### AI assistance disclosure

[CONFIRM and edit before posting. The Siril contributor documentation and CONTRIBUTING.md have no AI policy. Core maintainers have disclosed AI assistance in MR descriptions in plain prose, for example !1027 and !1070. This text follows that practice.]

This change was developed with AI assistance. The Metal backend, the meson integration, the tests, the benchmark scripts and a first draft of this description were produced by AI coding agents (Claude Code, Anthropic) working under my direction. [Name] reviewed the diff, built and ran the tests and benchmarks listed above on the machine described, and takes responsibility for the contribution. [Only keep this sentence if it is true.] The new files are offered under the project's license (GPL-3.0-or-later).

### Related Issues

[feature issue link]; #1450 (GPU offload, closed, for context); [stopping-criterion issue link]; [loaded-PSF issue/MR link].

### Additional Notes

Files touched: `meson.build`, `meson_options.txt`, `src/meson.build`, `src/algos/img_t/image.hpp`, `src/algos/metal_convolution.{h,mm}` (new), `src/filters/deconvolution.c`, `src/filters/deconvolution/deconvolve.cpp`, `src/tests/meson.build`, `src/tests/metal_convolution_test.cpp` (new). Based on upstream master 7765e18 (two commits).

### Merge Request Checklists

- [ ] Code follows project [coding guidelines](https://siril-contrib-doc.readthedocs.io/en/latest/) and its [checklist](https://siril-contrib-doc.readthedocs.io/en/latest/SubmitingWork.html#checklist)
- [ ] I have submitted a MR in the [documentation repository](https://gitlab.com/free-astro/siril-doc) if required [no user-visible setting or command is added; to be confirmed with maintainers]
- [ ] I have updated Changelog, listing the related issues. [needs the MR number; entry goes under 1.5.0 "Improvements"]
- [ ] If my MR introduces new dependencies, I have run all the CIs, even the optional (manual) jobs. [new optional macOS system frameworks (Foundation, Metal) and the objcpp language; macOS job cannot run from a fork]

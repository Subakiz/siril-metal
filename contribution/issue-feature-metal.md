<!--
DRAFT. Not posted anywhere. Based on .gitlab/issue_templates/feature.md in the Siril repository (label: Feature).
Before posting: the human poster must be able to truthfully say the disclosure sentence below
("I have reviewed the code and run the tests"). If that is not true, change or remove it first.
Also replace <FORK URL> and delete this comment.
Suggested title: Optional macOS Metal path for spatial convolution in naive Richardson-Lucy: is it welcome?
-->

Operating System: macOS (Apple silicon). The feature is macOS-specific.

# Description of the feature

I would like to ask whether an optional, macOS-only Metal implementation of single-precision spatial convolution would be welcome, before I open a merge request. Nothing has been submitted yet.

I know that #1450 (GPU offload frameworks) was closed on the grounds that many algorithms are memory-bandwidth bound, so transfers to and from VRAM cost more than they save, and that the complexity of supporting GPU libraries in the C/C++ code is probably not worth it. I agree that this is a fair concern, and I do not want to reopen the general question. The proposal is deliberately much narrower than what #1450 discussed, and I would like to show the evidence and let you judge.

## What it is

- One function: `img_t<float>::conv2` (single-precision 2D spatial convolution). It tries a Metal compute kernel first and otherwise runs the existing CPU loop.
- It is reached only from the non-FFT ("naive") Richardson-Lucy branch, which Siril uses when the PSF size is below `fftw_conv_fft_cutoff` (default 15).
- The default 15x15 PSF goes to the FFT branch, so default behaviour is unchanged. Only PSFs of 7 to 13 pixels with the default cutoff can reach the new code.
- Metal is used only if the kernel size is odd and between 7 and 31, the work is at least 32 Mi multiply-adds (`width * height * channels * kernel_size^2`), a Metal device exists, and the buffers fit. Otherwise, or if anything fails, the CPU code runs.
- The kernel is a direct correlation with zero borders, with the same orientation and accumulation order as the CPU loop. Fast math stays off (`MTLMathModeSafe` on macOS 15 and later SDKs, `fastMathEnabled = NO` otherwise).

## How it answers the #1450 concerns

- **Transfer cost.** I do not claim it is free. Input, kernel and output buffers are copied on every convolution call; there are no persistent device buffers across iterations. That per-call cost is included in the timings below and is the main thing limiting the gain. It still comes out ahead for the larger kernels; my guess is that direct convolution does enough arithmetic per byte copied at that size, but I have not measured the split.
- **Complexity and dependencies.** There is no new library. It uses the system Metal and Foundation frameworks, plus the `objcpp` language in meson. I know the contributor documentation asks to discuss new dependencies first, and this is why I am opening an issue rather than an MR. It is a new optional build dependency on macOS only, and I would like your view on it.
- **Build option.** A new meson feature option `metal`, default `auto`. On macOS, `auto` enables it if an Objective-C++ compiler and the frameworks are found; elsewhere it does nothing. `-Dmetal=enabled` on a non-macOS platform is an error, and `-Dmetal=disabled` gives the current behaviour (configure and build checked on macOS; the new unit test is registered only when Metal is built). The code compiles to nothing without `HAVE_METAL`.
- **Fallbacks.** Small images or kernels, even or out-of-range kernel sizes, non-float types, no Metal device, pipeline build failure and command buffer failure all return to the existing CPU code with the output untouched.
- **Memory estimate.** When built with Metal, the naive RL branch assumes slightly more buffers (two extra float buffers: 8 instead of 6 float channels in `deconvolution_compute_mem_limits`, and 9 instead of 7 copies in `naive_richardson_lucy`) as a conservative allowance. This applies to every Metal-enabled build, even when Metal is not dispatched, and may cause slightly more image slicing. The values are estimates; the real low-memory path was not exercised (it only differs at roughly 30 to 39 MB free).

## Evidence

All numbers are from one machine: Apple M5, 10 CPU cores, 8 GPU cores, 16 GB, macOS 27.2, release builds of the same source with Metal disabled and enabled, `siril-cli`, 10 RL iterations, 1024x1024 images, `-mul`, PSFs loaded with `-loadpsf` (on a build that includes a separate loaded-PSF size fix, see "Reproducing"). These tables are from the earlier, pre-rebase build of the same code. The full tables are in `BENCHMARK-RESULTS.md` (I can attach it).

Warm RL time (second RL pass in the same process), CPU and Metal runs interleaved in ABBA order over 10 pairs, measured twice. Median of the per-pair CPU/Metal ratio, with the min-max range in brackets:

| Case | Run 1 | Run 2 |
|---|---|---|
| mono 13x13 | 2.12 [1.81-2.62] | 1.94 [1.73-2.21] |
| RGB 13x13 | 2.15 [1.98-2.37] | 2.28 [1.97-2.53] |
| mono 7x7 | 1.13 [1.08-1.28] | 1.14 [1.02-1.19] |
| RGB 7x7 | 1.29 [1.24-1.37] | 1.15 [1.03-1.35] |

In short: about 2x at 13x13 and about break-even to a modest gain (roughly 1.1x) at 7x7. Whole-process wall time improves less because of start-up and I/O (13x13 mono: 1.89 and 1.72 median speedup; 7x7: about 1.07 to 1.16).

**Load caveat.** The machine was shared and not idle during these measurements (a GUI Siril build, a long-running Python process using roughly 50 to 100 percent of a core, and normal desktop processes). The 1-minute load average before each batch was between 3.4 and 8.1. Interleaving CPU and Metal runs makes the ratios more reliable than the absolute times, but all timings are indicative only. I have not yet repeated them on a quiet machine.

**Correctness.** 13 end-to-end `siril-cli` cases compared CPU and Metal builds of the same source (mono and RGB, `-mul`, default gradient descent, `-tv`, `-fh`, a real 1024x1024x3 crop, and fallback cases). All outputs were finite and within `|cpu - metal| <= 2e-5 + 2e-5 * |cpu|` per pixel. The largest observed difference was 3.8e-6 (absolute). There is also a unit test that checks `conv2` against a reference loop; it is skipped when there is no Metal device.

**Final rebased branch (upstream master 7765e18, two commits).** With `metal=auto` the unit test passes on the real GPU (1 of 1, max error 0). With `metal=disabled` configure and build succeed ("No tests defined"). Forcing slicing with `[core] max_slice_size=512` gave a CPU-versus-Metal max difference of 4.5e-8; the unsliced-versus-sliced difference (0.134) is existing tile-edge behaviour on both CPU and Metal. Warm RL, mono 13x13: median about 1050 ms CPU versus about 461 ms Metal, at load average 3.5 to 4.8 (indicative). A debugoptimized build logs `Metal convolution: device ..., WxH kN` once per process; release builds do not.

GTK build (gtk=true, metal=enabled, macOS arm64): configures, compiles and passes the unit test; the GUI starts and opens a 1024x1024 FITS image. Metal dispatch was confirmed in the GUI binary running a script (-s): 20 GPU command buffers for a 10-iteration k13 RL, output within 4.5e-8 of the CPU build. The GTK dialog (src/gui-gtk4/newdeconv.c:831,887) sets the same op_desc_deconvolve as the CLI command (src/core/command.c:2640), so it reaches the same deconvolve() code (confirmed by reading the source); it was not driven interactively.

**Default case.** With the default 15x15 PSF (FFT branch), the CPU and Metal builds measured 882 ms and 933 ms in a single run. The code path is identical in both builds, so this is run-to-run variation, not an effect of the change.

## What is not done

- The FFT path is not accelerated, and neither are other deconvolution methods or any other Siril operation.
- Only one machine was tested (Apple M5). Other Apple GPUs, older macOS versions and discrete GPUs are untested.
- No persistent buffers across iterations. This is the obvious next improvement, but I did not want to add complexity before knowing whether the idea is wanted at all.
- Kernels above 31 are not handled by Metal.
- The GUI and sequence processing were only smoke-tested; the benchmarks are single-image CLI runs.
- Constrained-memory behaviour with the larger estimate was only tested with forced slicing, not under real memory pressure.
- Not built or run on Linux or Windows, and not compiled against the macOS 11.3 SDK used by the project's macOS CI job; both are reasoned only.
- GPU use is logged once per process at debug level, only in debug-type builds (`siril_log_debug`, no runtime switch). In release builds, dispatch has to be inferred from the kernel size, the workload and timing.
- I have not been able to run the macOS CI job (it needs the project's runners), and I have only built on macOS.

## Reproducing

Make the PSF with `makepsf manual -gaussian -fwhm=3 -ks=13`, not with `rl -loadpsf=`:

```
requires 1.4.0
set32bits
load /absolute/path/to/image.fit
makepsf manual -gaussian -fwhm=3 -ks=13
rl -iters=10 -mul
save out
```

Run it with a `-Dmetal=enabled` build and a `-Dmetal=disabled` build and compare the outputs. The input needs at least about 198,600 pixels times channels for 13x13 (for example 512x512 mono).

Without a separate one-line fix (planned as its own MR, `args->ks = com.kernelsize;` in `load_kernel`), `rl -loadpsf=` leaves `args->ks` at 15, so RL takes the FFT path, Metal is never used, and the output was intermittently all NaN on the CPU too. That was inferred from the code and from reproduction on a metal-disabled build; I have not proven it on pristine master.

## Questions

1. Is an optional, macOS-only Metal path of this narrow kind welcome at all? If the answer is no, or "not for 1.5", that is completely fine, and I will not open an MR.
2. If yes: should the `metal` option default to `auto` (enabled automatically when found, which also affects the macOS bundle build since it passes no options) or to `disabled` (opt-in)?
3. Should the 7x7 case be included, given the gain is only about 1.1x there? I can raise the lower bound (for example to 9 or 11) if you prefer, but would need to re-measure.
4. Is the new optional dependency on the macOS system frameworks and `objcpp` acceptable in principle, and is there any other form of review you would want first?
5. Would you like me to open a `Draft:` MR against `master`? If so, is there any particular form of AI-assistance disclosure you want, for example a commit trailer?

## Code

The patches are available as branches on a fork: <FORK URL>. I am not linking an MR because none exists. The code is two commits on upstream master `7765e18` and would be offered under the project's license (GPL-3.0-or-later).

## Disclosure

This work was developed with assistance from Claude (Anthropic's AI model); I have reviewed the code and run the tests.

/label ~"Feature"

<!--
DRAFT. Not posted anywhere. Follows .gitlab/issue_templates/bug.md.
Suggested title: Richardson-Lucy -stop never triggers in the spatial (non-FFT) branch
Fill in or delete every [bracketed] item before use.
-->

Siril version: 1.5.0-e379842 (master at e379842e5b6f; the same lines are in current master 7765e186 and in branch 1.4, see below)

Note: please report against latest git master branch or latest stable version

Operating System: macOS 27.2, arm64 (the code involved is not platform specific)

Package: local build from source (meson, release, Homebrew LLVM 22), CLI binary with Metal disabled. The build also contained a one-line `args->ks` change in `load_kernel`. It does not touch `deconvolve.hpp`, and the reproduction below does not use `-loadpsf`. [Re-run on an unpatched upstream build before posting if possible.]

# Description of the bug

The `-stop=` option of `rl` (the "stopping criterion") can halt Richardson-Lucy early in the FFT branch (PSF size >= `fftw_conv_fft_cutoff`), but it never halts in the spatial branch (PSF size < `fftw_conv_fft_cutoff`, `naive_richardson_lucy`), whatever value is given.

What the code shows (`src/filters/deconvolution/deconvolve.hpp`, master e379842 and 7765e186):

- FFT branch, `rl_deconvolve_fft`: the estimate is copied to `stopcrit` before each update (lines 136 to 140) and the measure `|est - stopcrit| / |stopcrit|` is computed after the update (lines 164 to 168).
- Spatial branch, `rl_deconvolve_naive`: the measure at lines 249 to 252 is `gxy.map(abs(x - gxy) / abs(gxy)); stopping = gxy.sum() / gxy.size; if (stopping < stopcriterion) ...`. `gxy` is declared once before the iteration loop at line 193 (`img_t<T> gxy(f.w, f.h, f.d);`). I could not find any statement that copies the previous estimate into `gxy`.
- The only other write to `gxy` is `gxy.gradientxy(w)` at line 215, which runs only for `regtype == 1 || regtype == 4` (Frobenius-Hessian). In that case the measure compares `x` with a second derivative of the previous estimate, not with the previous estimate.
- `img_t(w, h, d)` constructs `data(w*d*h)` (`src/algos/img_t/image.hpp:108`). By C++ rules a `std::vector` of that size is value-initialised, so I expect `gxy` to start as zeros. I did not instrument this.
- If so, for the non-FH cases the first measure is `x / 0`, which is infinite (or NaN where `x` is 0), and the result is stored back in `gxy`. Later iterations compute infinity minus infinity or infinity divided by infinity. The mean would then never be below the threshold. I did not instrument this either; what I observed is only that the message never appears.

The same lines (`img_t<T> gxy(f.w, f.h, f.d);` and `gxy.map((img::abs(x - gxy)) / img::abs(gxy));`) are in the `1.4` branch (lines 194 and 251 of `deconvolve.hpp` there), so I believe stable is affected too. I did not run a 1.4 build.

# Reproduction

Same image, same PSF, same options. Only the FFT cutoff preference changes, which switches the branch for a 15x15 PSF (the branch test is `ks < fft_cutoff`, `src/filters/deconvolution.c:816`). `-stop=0.999` is used because the command rejects values above 1 (`src/core/command.c`, `-stop=` parsing); with a relative-change criterion, a mean relative change below 0.999 should be met after one iteration.

Input: any positive-valued image of at least a few hundred pixels per side. A synthetic one:

```python
import numpy as np
from astropy.io import fits
y, x = np.mgrid[:1024, :1024]
rng = np.random.default_rng(1)
img = 0.08 + 0.025*np.sin(x/89) + 0.02*np.cos(y/113)
for _ in range(80):
    cx, cy = rng.uniform(0, 1024, 2); s = rng.uniform(1.3, 3.5)
    img += rng.uniform(0.05, 0.6)*np.exp(-((x-cx)**2 + (y-cy)**2)/(2*s*s))
fits.writeto("mono.fit", (img + rng.normal(0, 5e-4, img.shape)).astype(np.float32), overwrite=True)
```

Script `stop.ssf` (use an absolute path for `load`; the CLI changes the working directory):

```
requires 1.4.0
set32bits
load /absolute/path/to/mono.fit
makepsf manual -gaussian -fwhm=3 -ks=15
rl -iters=5 -mul -stop=0.999
close
```

Run twice with an isolated config, offline:

```sh
printf '[core]\nfftw_conv_fft_cutoff=15\n' > cfg15.ini    # FFT branch for a 15x15 PSF
printf '[core]\nfftw_conv_fft_cutoff=16\n' > cfg16.ini    # spatial (naive) branch
siril-cli -o -i cfg15.ini -s /absolute/path/to/stop.ssf
siril-cli -o -i cfg16.ini -s /absolute/path/to/stop.ssf
```

Expected result: both runs stop early (the criterion is a relative change that is far below 0.999 after the first iteration), and the log shows `Richardson-Lucy halted early by the stopping criterion after iteration 1`.

Actual result (macOS arm64 build described above, repeated with `-iters=5` and the other options below):

| Setting | Branch | Log |
|---|---|---|
| `fftw_conv_fft_cutoff=15`, `-ks=15`, `-mul -stop=0.999` | FFT | "halted early ... after iteration 1" |
| `fftw_conv_fft_cutoff=16`, `-ks=15`, `-mul -stop=0.999` | spatial | no "halted early" message (the run completes with "Deconvolution succeeded") |
| `fftw_conv_fft_cutoff=16`, `-ks=15`, `-mul -stop=0.001` | spatial | no "halted early" message |
| default cutoff, `-ks=7`, `-mul -stop=0.999` | spatial | no "halted early" message |

Further runs used `-loadpsf`. They only make sense with the `args->ks` change in `load_kernel` (without it, `rl -loadpsf=` keeps `args->ks` at 15, so the FFT branch is chosen regardless of the loaded PSF size), and were done on a build that had it, so this part is not a reproduction on unmodified upstream: with a loaded 7x7 PSF (spatial branch) and `-iters=5 -stop=0.999`, each of `-mul`, no regulariser flag (gradient descent), `-tv -alpha=3000` and `-fh -alpha=3000` ran every iteration and logged no early stop. With a loaded 15x15 PSF (FFT branch) the same four option sets all logged "halted early ... after iteration 1". An earlier benchmark run with a loaded 7x7 PSF, `-iters=20 -mul -stop=0.01`, also logged no early stop. Note the PSF size differs between the spatial and FFT rows here, unlike in the table above.

# Additional information

- The result does not depend on the Metal change: it was seen with Metal disabled.
- In the FFT runs above the "halted early" line is printed four times for one `rl` call. I did not investigate why (possibly once per processing slice) and it is not part of this report.
- The `rl` help text says only that "The stopping criterion may be activated by specifying a stopping limit with -stop=", and gives no branch restriction.
- I have not proposed or tested a fix. The FFT branch's approach (a copy of the previous estimate taken before the update) looks like the natural model. I can submit an MR if that is wanted. [Delete if not offered.]
- AI disclosure [delete or edit]: the analysis above was done with AI assistance (Claude Code); the runs were done with the CLI and the line references were checked against the source tree.

/label ~"Bug"

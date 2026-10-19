# P3D (de Belsunce / Philcox / Eisenstein HIPSTER-Lya) on Perlmutter

Untarred to `for_Boryana/P3D`. Builds and self-tests clean. Recipe below.

## Build

```bash
cd for_Boryana/P3D
rm -f grid_power.o driver_fits.o power        # tarball ships FASRC-built .o files
make Lya=-DLYA FITS_ENV=$HOME/.conda/envs/jaxeft
```

- Default `g++` on Perlmutter is `gcc-native/14` (14.3.0) — fine, no module needed.
- GSL is system-wide (`/usr/include/gsl`, `-lgsl -lgslcblas`).
- cfitsio: the Makefile's default `FITS_ENV` is Roger's FASRC conda env and does
  not exist here. `~/.conda/envs/jaxeft` has cfitsio 10.4.3.1 + headers, which is
  the same major version, so it is used as the override. The binary carries that
  path as RUNPATH, so **do not delete the `jaxeft` env** or `./power` stops
  running. (Only the FITS readers need it; the text reader does not, but it is
  linked into the same binary.)

## Runtime: `ulimit -s unlimited` is REQUIRED

Without it `./power` **segfaults** right after gridding, before the pair loop —
no error message, just SIGSEGV. `power_counts.h::count_pairs` puts
`Float old_kernel[mbin]` and `Float legendre[...]` on the stack inside an OpenMP
region, and `pair_counter.h` holds larger per-thread buffers. The default 8 MB
stack is not enough.

```bash
ulimit -s unlimited
export OMP_STACKSIZE=512M
```

This is not in the upstream README because FASRC's default stack is unlimited.
Put both lines in every slurm script.

## Login-node memory: a hard 30 GB per-user cgroup

`load_dF_grid` memmaps the 13.9 GB `complex_dF_*.asdf` and reads it strided.
That page cache is charged to the user's cgroup, and on Perlmutter login nodes
that cgroup is capped:

```
/sys/fs/cgroup/user.slice/user-<uid>.slice/memory.max = 32212254720   # 30 GB
                                            memory.events: oom_kill 10
```

So a field load sits right at the limit and gets SIGKILLed (**exit 137**, no
traceback, no message) unpredictably -- phases 001-003 went through, 004 and 005
did not. `free -g` is useless for diagnosing this: it reported 294 GB available
on a node where the user slice was capped at 30.

Check it directly with:

```bash
CG=$(awk -F: '{print $3}' /proc/self/cgroup | head -1)
cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.max
cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.events
```

Anything that calls `load_dF_grid` belongs in a batch job. `--qos=shared` with
`--mem=120G` is enough and queues faster than a whole node.

## Python for the helper scripts / notebook

The login-node default `python` is 2.7 and `python3` is 3.6 without numpy, so
`test/make_synthetic_inputs.py` fails with a SyntaxError on f-strings. Use:

```bash
module load python/3.13-26.1.0     # numpy 2.3.5, scipy 1.17.0, astropy 7.2.0
```

## Self-test

```bash
module load python/3.13-26.1.0
ulimit -s unlimited
cd test && python make_synthetic_inputs.py && bash run_test.sh
```

Result: all three modes (`lyaxlya` text, `lyaxqso`, `qsoxqso`) run, outputs
finite, `SELF-TEST PASSED`. ~1 min on a login node with 4 threads.

## What the estimator actually computes (read from the source, not the README)

`power_mod/power_counts.h::count_pairs`:

```
P_raw_l(k_i) = sum_{i!=j, r_ij<R0, tid_i!=tid_j}
                  (wg_i - w_i)(wg_j - w_j) * W(r_ij;R0) * L_l(mu_ij) * K_l(k_i, r_ij)
RR_l(r)      = sum (4l+1) w_i w_j L_l(mu_ij),   binned in 1000 linear r-bins
```

- `d_i = wg_i - w_i` is the density; `w_i` the weight. Both come straight from
  the input file, so the text reader is a full-freedom interface.
- `W(r;R0)` is a smooth truncation: 1 below R0/2, then a quartic taper to 0 at
  R0. So the measurement is the true P(k) **convolved** with the FT of W — a
  smoothing of width ~2pi/R0 = 0.031 h/Mpc at R0 = 200. This is not optional and
  not deconvolved; theory must be forward-modelled through it.
- Pairs with equal `tid` are dropped, which is what kills the P_1D/n_2D
  aliasing pedestal: the FFT estimator cannot separate same-skewer pixel pairs
  from genuine 3D pairs, this one just never counts them.

## Cost model (the thing that sets what we can run)

This is a pair counter, so the cost is pairs, not particles:

```
N_pairs ~ 1/2 * N * n_3D * (4pi/3) R0^3
```

For our box with the LOS fully sampled (512 pixels along z) and a transverse
sightline fraction `frac`, both N and n_3D scale with `frac`, so the cost goes
as **frac^2**:

```
N = frac * 512^3 = frac * 1.34e8
n_3D = frac * 512^3 / 2000^3 = frac * 0.0168 (Mpc/h)^-3
(4pi/3) R0^3 = 3.35e7 (Mpc/h)^3   at R0 = 200
=>  N_pairs = frac^2 * 3.8e13
```

| frac | QSO/deg^2 | particles | pairs |
|---|---|---|---|
| 1.000 | 323 | 134M | 3.8e13 |
| 0.619 | 200 | 83M | 1.4e13 |
| 0.500 | 162 | 67M | 9.4e12 |
| 0.155 | 50 (DESI) | 21M | 9.0e11 |

Conversion: `qso_per_deg2 = frac * 322.9` at nmesh = 512 (from
`sparse_skewers.frac_to_qso_per_deg2`, chi = 4022 Mpc/h, 1 deg = 70.19 Mpc/h).

Note our box is **much denser in 3D than DESI** at the same QSO/deg^2, because
the DESI forest lives in a thin redshift shell while ours fills a 2000 Mpc/h
cube. So "DESI DR2 = 30M particles in 6-10 min" does NOT transfer: at frac =
0.155 we have fewer particles than DR2 but a far higher 3D density.

### Measured throughput (2026-08-16, Perlmutter login node, 16 threads)

Two calibration runs with `box_to_p3d.py --random` (values do not affect cost),
nmesh = 512, R0 = 200, `-fast 1`:

| frac | particles | accepted pairs | pair-loop wall | pairs/core/s |
|---|---|---|---|---|
| 0.005 | 0.67M | 8.70e8 | 8 s | 6.80e6 |
| 0.015 | 2.01M | 7.60e9 | 37 s | 1.28e7 |

**The frac^2 scaling is confirmed**: 8.70e8 x (0.015/0.005)^2 = 7.83e9 against
7.60e9 measured, 3% high. Calibrated law:

```
N_pairs ~ 3.4e13 * frac^2        (the analytic estimate above is ~8% high,
                                  the difference is the tid exclusion + edges)
```

Throughput *improves* with density (6.8e6 -> 1.28e7 pairs/core/s) as the
per-cell overhead amortizes, so extrapolating at 1.28e7 is conservative:

| frac | QSO/deg^2 | pairs | core-s | 1 node (128c) | 2 nodes |
|---|---|---|---|---|---|
| 0.155 | 50 (DESI) | 8.2e11 | 6.4e4 | 8 min | 4 min |
| 0.500 | 162 | 8.5e12 | 6.6e5 | 1.4 h | 43 min |
| 0.619 | **200** | 1.3e13 | 1.0e6 | **2.2 h** | 1.1 h |
| 1.000 | 323 | 3.4e13 | 2.7e6 | 5.8 h | 2.9 h |

So **everything up to frac = 1 is affordable** — an earlier claim in this
project that the dense case was infeasible came from an assumed pair rate and
was wrong by ~2 orders of magnitude. R0 = 150 instead of 200 would cut all of
these by (150/200)^3 = 0.42 if needed.

Other costs, both minor: `box_to_p3d.py` writes ~670k rows in 4.5 s, so the
83M-row (6 GB ASCII) file for frac = 0.619 takes ~9 min; the C++ text read of
671k particles took ~2 s, so ~4 min at 83M.

### Measured in production (2026-08-17, 1 node, 128 threads)

| frac | QSO/deg^2 | particles | pairs | wall | peak RSS |
|---|---|---|---|---|---|
| 0.619 | 200 | 83,080,712 | 1.45e13 | 78 min | 18.8 GB |
| 0.155 | 50 | 20,803,592 | 9.07e11 | 6.8 min | 6.9 GB |

Pair counts landed within 0.5 % of the predictions above, and throughput reached
2.45e7 pairs/core/s -- about 2x the login-node calibration, because the pair
loop keeps getting more efficient as density rises. So the table above is
conservative by ~2x.

**Memory turned out to be a non-issue**: 18.8 GB at frac = 0.619, not the
~100 GB extrapolated from a single small run (per-particle scaling from 0.67M
particles badly overstates it -- most of that 1.07 GB was fixed overhead).

### mu convention — the reason a compile flag matters for us

`power_counts.h::cleanup_l`:

```cpp
#ifndef PERIODIC
    Float3 los = p1 + p2;                       // midpoint LOS (radial, curved sky)
    mu = fabs(pos.dot(los)/norm/los.norm());
#else
    mu = fabs(pos.z/norm);                      // flat sky, LOS = z
#endif
```

Our boxes are periodic with the LOS on axis 2, and every FFT measurement so far
assumes flat-sky `mu = k_par/k`. So the PERIODIC branch is the matching
convention. Two ways to get it, to be decided next:

- **compile `make Lya=-DLYA Periodic=-DPERIODIC`** — exact flat-sky mu and
  periodic wrapping, so no box-face edge effects at all. **This is what we use**
  (`power_periodic`), but it needed three fixes; see below.
- **place the box at a huge radial distance** (positions are `double`, so
  D ~ 1e7 Mpc/h costs no precision) so the midpoint LOS is z to ~1e-4. Uses the
  validated non-periodic path, but the box faces become real survey edges,
  handled only in the average by the RR normalization. Available as a
  cross-check via `box_to_p3d.py --distant 1e7 --no-ghosts` + `power_survey`.

## PERIODIC + LYA is broken as shipped — four fixes

This combination is not one the upstream README claims to validate, and it does
not work out of the box. All three were confirmed by running before patching.

**1. `-fast 1` silently returns zeros.** `pair_counter.h` dispatches on
`#ifdef PERIODIC / #elif defined LYA`, so with both flags the PERIODIC branch
wins and the entire LYA branch is skipped — including
`transform_fast_counts()`, which is what converts the fast mode's `S_fast`
histograms into `power_counts`. Symptom, on a real run:

```
ell=0            ell=2          ell=4
-1.267884e+07    0.000000e+00   0.000000e+00
-8.462037e+06    0.000000e+00   0.000000e+00
```

ell=2 and ell=4 identically zero, ell=0 just `-RR_analytic`. Nothing warns you.
Fixed by calling `transform_fast_counts()` (a no-op unless `-fast 1`) in the
PERIODIC branch too, guarded by `#ifdef LYA`.

**2. The monopole gets a shot-noise subtraction that does not apply.**
`save_spectrum(Float*)` does `if(j==0) output_pk -= RR_analytic[i]`. That term
removes the uniform-density "1" of `1+xi` that appears when `d_i = 1` per
discrete tracer. In LYA mode `d_i = wg-w = delta_F` is already a zero-mean
fluctuation, so there is no such term. Now guarded by `#ifndef LYA`;
RR_analytic is still computed and written as a diagnostic.

Note the periodic normalization itself, `power_norm = sum_w1*sum_w2/V`, **is**
correct for us: with `w=1` on every pixel it is `N^2/V`, which is exactly the
right normalization for a uniformly-sampled delta field. This is arguably
cleaner than the non-periodic RR route.

**3. The inferred box size is 1996.09, not 2000.** `driver.h` takes the period
from the RANGE of input positions and ignores `-boxsize`. Cell centres span
`0 .. (nmesh-1)*cell`, one cell short of the box. Every wrapped separation
would be off by 3.9 Mpc/h. Fixed in `box_to_p3d.py` by appending eight
zero-weight "ghost corner" particles (`w=0, wg=0`, so `d=0`: they contribute
nothing to either `power_counts` or `RR_counts`, only to `used_pairs`) at
`{0, LBOX-1e-6}^3`. After the fix the code prints
`# Setting periodic box-size to 2000.00`.

**4. Every P_ell comes out exactly 2x low.** The pair loop visits each
unordered pair ONCE, so the counts need doubling. In non-LYA mode that doubling
is a *side effect* of `save_counts()`, which mutates `power_counts` in place
before the periodic `save_spectrum()` reads it. With LYA defined `save_counts`
is compiled out (`#ifndef LYA` at its call site), so the doubling never
happens. The LYA `save_spectrum(int)` does its own doubling, which is why the
non-periodic path is fine. Fixed with a new `apply_one_grid_factor(one_grid)`
called alongside `transform_fast_counts()`.

This one is the most dangerous of the four: a clean factor of 2 on every
multipole at every k. It changes no shape, breaks no plot, and survives every
smoke test.

## End-to-end validation: `validate_p3d.py`

Because three of the four bugs were SILENT, a smoke test is not enough.
`validate_p3d.py` reimplements the estimator directly from the source-level
definition in numpy (minimum-image separations, the exact `kernel_interp.h`
kernel, `pair_weight`, the tid exclusion, `(sum_w)^2/V`) on a small periodic
box, and compares against `power_periodic` on the identical input:

```
$ python validate_p3d.py
  2560 particles, 40 skewers, L = 500.0, R0 = 100.0
  [cpp] # Setting periodic box-size to 500.00
  ...
  worst |ratio-1| = 2.74e-04   PASS
```

Agreement is ~1e-6 relative on every bin except one near-zero value where
cancellation amplifies the residual. This confirms the kernel convention, the
mu convention, the pair weighting, the tid exclusion, the absolute
normalization and the factor-2 fix simultaneously. **Re-run it after any change
to the C++ or to `box_to_p3d.py`.**

Other cosmetics: with LYA defined, `save_counts` is compiled out but the
PERIODIC branch still prints "Printed counts to file as ..._DD_power_counts...".
That file is not written. Harmless, but do not go looking for it.

### Rebuilding

Two binaries, both from the same patched tree:

```bash
rm -f grid_power.o driver_fits.o power
make Lya=-DLYA Periodic=-DPERIODIC FITS_ENV=$HOME/.conda/envs/jaxeft && mv power power_periodic
rm -f grid_power.o driver_fits.o power
make Lya=-DLYA                     FITS_ENV=$HOME/.conda/envs/jaxeft && mv power power_survey
```

`power_periodic` additionally requires the runtime flag `-perbox` (it exits
with a message otherwise), and `power_survey` refuses `-perbox`.

import argparse
import json
import pathlib

import numpy as np

SEED = 3407
N_SAMPLES = 3400
N_TRAIN, N_VAL, N_TEST = 2400, 400, 600
WAVELENGTHS_NM = np.array([325.0, 514.0, 633.0])
LA_BOUNDARY = 20.0
GRID = np.linspace(1000.0, 1800.0, 801)
CORRUPT_FRACTION = 0.061
CORRUPTIONS = ["cosmic_ray", "saturation", "wrong_wavelength", "laser_burn"]
STAGE1_C_514 = 44.0
STAGE2_C_514 = 0.0055
G_SIGMA_COEF = 5.0
SIGMA_MAX_GPA = 8.0


def nm_to_ev(nm):
    return 1239.841984 / nm


def c_stage1(wl_nm):
    return STAGE1_C_514 * (wl_nm / 514.0) ** 2


def c_stage2(wl_nm):
    return STAGE2_C_514 * (wl_nm / 514.0) ** 2


def disorder_to_structure(x, rng):
    if x <= 0.5:
        t = x / 0.5
        la = 300.0 * (1.0 - t) + LA_BOUNDARY * t
        stage_fr = 1
    else:
        t = (x - 0.5) / 0.5
        la = LA_BOUNDARY * (1.0 - t) + 2.0 * t
        stage_fr = 2
    la *= np.exp(rng.normal(0.0, 0.04))
    la = float(np.clip(la, 1.5, 400.0))
    sp3 = float(
        np.clip(
            1.0 / (1.0 + np.exp(-9.0 * (x - 0.62))) + rng.normal(0.0, 0.015), 0.0, 0.95
        )
    )
    h_frac = float(
        np.clip(
            0.55 * (1.0 - x) ** 1.5
            + 0.25 * rng.beta(2, 5) * (1.0 - x)
            + rng.normal(0.0, 0.03),
            0.0,
            0.55,
        )
    )
    sigma = float(
        np.clip(
            SIGMA_MAX_GPA * sp3**1.5 * (1.0 - 0.7 * h_frac) + rng.normal(0.0, 0.25),
            0.0,
            SIGMA_MAX_GPA,
        )
    )
    return la, sp3, h_frac, sigma, stage_fr


def label_stage(la):
    if la >= 43.0:
        return 1
    if la >= 6.0:
        return 2
    return 3


def dg_ratio(la, wl_nm, stage_fr):
    if stage_fr == 1:
        return c_stage1(wl_nm) / la
    return c_stage2(wl_nm) * la * la


def g_position(la, stage_fr, ev, sigma_gpa, h_frac):
    if stage_fr == 1:
        t = np.clip((300.0 - la) / (300.0 - LA_BOUNDARY), 0.0, 1.0)
        base = 1581.0 + 19.0 * t
    else:
        t = np.clip((LA_BOUNDARY - la) / (LA_BOUNDARY - 2.0), 0.0, 1.0)
        base = 1600.0 - 90.0 * t
        base += 25.0 * (ev - nm_to_ev(514.0)) * t
    base += G_SIGMA_COEF * sigma_gpa
    base += 10.0 * h_frac
    return float(np.clip(base, 1480.0, 1640.0))


def g_fwhm(la, stage_fr, sp3, h_frac):
    if stage_fr == 1:
        w = 18.0 + 8.0 * (1.0 - np.clip(la / 300.0, 0.0, 1.0))
    else:
        w = 30.0 + 60.0 * sp3
    return w + 15.0 * h_frac


def d_position(ev):
    return 1350.0 + 50.0 * (ev - nm_to_ev(514.0))


def bwf(x, x0, gamma, amp):
    q = 4.0
    t = (x - x0) / gamma
    return amp * (1.0 + t / q) ** 2 / (1.0 + t * t)


def lorentz(x, x0, gamma, amp):
    return amp * gamma * gamma / ((x - x0) ** 2 + gamma * gamma)


def synthesize(rng):
    return {
        "T_C": float(rng.uniform(20.0, 500.0)),
        "ion_energy_eV": float(rng.uniform(5.0, 120.0)),
        "ch4_frac": float(rng.beta(2.0, 5.0)),
        "pressure_mTorr": float(rng.uniform(1.0, 50.0)),
        "bias_V": float(rng.uniform(0.0, 1200.0)),
    }


def process_to_disorder(p, rng):
    z = (
        1.4 * (p["ion_energy_eV"] - 60.0) / 60.0
        + 1.1 * (p["bias_V"] - 600.0) / 600.0
        + 0.7 * (p["ch4_frac"] - 0.28) / 0.28
        - 0.5 * (p["T_C"] - 260.0) / 260.0
        + rng.normal(0.0, 0.35)
    )
    return float(np.clip(1.0 / (1.0 + np.exp(-1.6 * z)), 0.0, 1.0))


def make_spectrum(la, sp3, h_frac, sigma, stage_fr, wl_nm, rng, noise_scale=1.0):
    ev = nm_to_ev(wl_nm)
    ratio = dg_ratio(la, wl_nm, stage_fr)
    g_amp = 1000.0
    g0 = g_position(la, stage_fr, ev, sigma, h_frac)
    gw = g_fwhm(la, stage_fr, sp3, h_frac)
    spec = bwf(GRID, g0, gw, g_amp)
    if la > 3.0:
        d_amp = g_amp * ratio * 1.15 * (1.0 - 0.6 * h_frac)
        spec += lorentz(GRID, d_position(ev), 22.0 + 0.06 * la, max(d_amp, 0.0))
    spec += 40.0 + 30.0 * (GRID - 1000.0) / 800.0
    pl_slope = 2400.0 * h_frac
    if pl_slope > 1.0:
        spec += pl_slope * (GRID - 1000.0) / 800.0
    spec *= rng.uniform(0.85, 1.15)
    spec += rng.normal(0.0, noise_scale * np.sqrt(np.clip(spec, 1.0, None)) * 0.02)
    return np.clip(spec, 0.0, None)


def corrupt(spec, mode, wl_nm, rng, meta):
    s = spec.copy()
    if mode == "cosmic_ray":
        for _ in range(rng.integers(3, 9)):
            i = int(rng.integers(0, len(s)))
            s[i] += rng.uniform(8.0, 20.0) * np.sqrt(max(s[i], 1.0))
    elif mode == "saturation":
        cap = np.quantile(s, rng.uniform(0.93, 0.985))
        s = np.minimum(s, cap)
    elif mode == "wrong_wavelength":
        others = [w for w in WAVELENGTHS_NM if w != wl_nm]
        meta["true_wavelength_nm"] = wl_nm
        return spec, float(rng.choice(others)), meta
    elif mode == "laser_burn":
        cut = int(rng.integers(len(s) // 3, 2 * len(s) // 3))
        decay = np.linspace(1.0, rng.uniform(0.35, 0.7), len(s) - cut)
        s[cut:] *= decay
    return s, wl_nm, meta


def generate(outdir):
    rng = np.random.default_rng(SEED)
    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    xs = np.concatenate(
        [
            rng.uniform(0.0, 0.5, int(N_SAMPLES * 0.24)),
            rng.uniform(0.38, 0.62, int(N_SAMPLES * 0.21)),
            rng.uniform(
                0.55, 1.0, N_SAMPLES - int(N_SAMPLES * 0.24) - int(N_SAMPLES * 0.21)
            ),
        ]
    )
    rng.shuffle(xs)

    spectra = np.zeros((N_SAMPLES, len(GRID)), dtype=np.float32)
    rows = []
    n_corrupt = int(round(N_SAMPLES * CORRUPT_FRACTION))
    corrupt_idx = set(rng.choice(N_SAMPLES, n_corrupt, replace=False).tolist())

    for i in range(N_SAMPLES):
        p = synthesize(rng)
        x = process_to_disorder(p, rng)
        la, sp3, h_frac, sigma, stage_fr = disorder_to_structure(x, rng)
        wl = float(rng.choice(WAVELENGTHS_NM))
        spec = make_spectrum(la, sp3, h_frac, sigma, stage_fr, wl, rng)
        meta = {"true_wavelength_nm": wl}
        corruption = "none"
        label_wl = wl
        if i in corrupt_idx:
            corruption = str(rng.choice(CORRUPTIONS))
            spec, label_wl, meta = corrupt(spec, corruption, wl, rng, meta)
        spectra[i] = spec.astype(np.float32)
        rows.append(
            {
                "sample_id": i,
                **p,
                "disorder_x": x,
                "la_angstrom": la,
                "sp3_frac": sp3,
                "h_frac": h_frac,
                "stress_gpa": sigma,
                "fr_stage": stage_fr,
                "stage": label_stage(la),
                "wavelength_nm": label_wl,
                "true_wavelength_nm": meta["true_wavelength_nm"],
                "corruption": corruption,
            }
        )

    perm = rng.permutation(N_SAMPLES)
    split = np.empty(N_SAMPLES, dtype="U5")
    split[perm[:N_TRAIN]] = "train"
    split[perm[N_TRAIN : N_TRAIN + N_VAL]] = "val"
    split[perm[N_TRAIN + N_VAL :]] = "test"

    np.savez_compressed(
        out / "mac_synthetic_v2.npz",
        grid=GRID,
        spectra=spectra,
        split=split,
        wavelengths=np.array([r["wavelength_nm"] for r in rows]),
        la=np.array([r["la_angstrom"] for r in rows]),
        sp3=np.array([r["sp3_frac"] for r in rows]),
        stage=np.array([r["stage"] for r in rows]),
        corruption=np.array([r["corruption"] for r in rows]),
    )
    (out / "metadata_v2.json").write_text(json.dumps(rows, indent=1))
    (out / "manifest_v2.json").write_text(
        json.dumps(
            {
                "name": "mac_synthetic_v2",
                "seed": SEED,
                "n_samples": N_SAMPLES,
                "split": [N_TRAIN, N_VAL, N_TEST],
                "grid": [1000.0, 1800.0, 801],
                "wavelengths_nm": WAVELENGTHS_NM.tolist(),
                "physics": {
                    "C_stage1_514nm_A": STAGE1_C_514,
                    "C_stage2_514nm_A-2": STAGE2_C_514,
                    "wavelength_scaling": "(wl/514)^2 both stages; turnover La=20A invariant",
                    "stress_g_shift_cm-1_per_GPa": G_SIGMA_COEF,
                    "sigma_max_GPa": SIGMA_MAX_GPA,
                    "pl_slope": "linear in h_frac (Casiraghi PRB 72 085401)",
                    "d_suppression_with_h": "1 - 0.6*h",
                },
                "provenance": "SYNTHETIC v2 — v1 + wavelength-scaled C, stress G-shift, a-C:H PL/hydrogen (report.pdf next-steps item 2)",
            },
            indent=2,
        )
    )

    stages = np.array([r["stage"] for r in rows])
    print(
        "generated v2:",
        N_SAMPLES,
        "samples; stages:",
        {int(s): int((stages == s).sum()) for s in (1, 2, 3)},
        "corrupted:",
        n_corrupt,
    )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mac")
    args = ap.parse_args()
    generate(args.out)

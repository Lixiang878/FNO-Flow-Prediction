"""Command-line entry for fno-flow-prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import FNO1D, UNet1D, generate_dataset, lowres_solver_error, relative_l2


def cmd_gen(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = generate_dataset(
        n_samples=args.samples, n_grid=args.grid, nu=args.nu, T=args.T,
        seed=args.seed, out_path=out,
    )
    print(f"[gen] wrote {out}  ({data['a'].shape[0]} samples, grid={args.grid})")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    print("[demo] generating a small Burgers dataset (offline, numpy) ...")
    data = generate_dataset(n_samples=args.samples, n_grid=args.grid, seed=args.seed)
    a, u = data["a"], data["u"]

    # Classical under-resolved solver baseline.
    base_err = float(np.mean([lowres_solver_error(u[i:i + 1], a[i:i + 1],
                                                    n_coarse=args.coarse)
                               for i in range(len(u))]))

    # Run the lazy-initialised architectures (random weights) to confirm they
    # execute and preserve the grid. Real trained weights come from `train`.
    fno = FNO1D(n_modes=args.modes, width=args.width)
    unet = UNet1D(width=args.width)
    fno_out = fno(a[:1])
    unet_out = unet(a[:1])

    # Resolution-invariance check for the FNO: same model, different grid.
    a_half = a[:1, ::2]  # N/2 grid
    fno_out_half = fno(a_half)

    print("\n" + "=" * 58)
    print("  FNO vs UNet for 1D Burgers — offline comparison")
    print("=" * 58)
    print(f"  Grid                     : {args.grid}")
    print(f"  Classical low-res solver : rel-L2 = {base_err:.4f}")
    print(f"  FNO (untrained) forward  : shape {tuple(fno_out.shape)}  ok")
    print(f"  UNet (untrained) forward : shape {tuple(unet_out.shape)}  ok")
    print(f"  FNO @ grid {a_half.shape[-1]}     : shape {tuple(fno_out_half.shape)}  "
          f"(resolution-invariant)")
    print("-" * 58)
    print("  To train and get real rel-L2, run:")
    print("    python -m fno_flow.train        # needs `pip install torch`")
    print("=" * 58)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    try:
        from .train import train
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for name in ("fno", "unet"):
        m = train(name, epochs=args.epochs, lr=args.lr,
                  n_samples=args.samples, seed=args.seed, out_json=str(out))
        print(json.dumps(m, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        prog="fno-flow-prediction",
        description="FNO vs UNet surrogate for parametric PDEs (1D Burgers).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate a Burgers dataset (.npz)")
    g.add_argument("--samples", type=int, default=256)
    g.add_argument("--grid", type=int, default=256)
    g.add_argument("--nu", type=float, default=0.01)
    g.add_argument("--T", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=1234)
    g.add_argument("--out", default="data/burgers.npz")
    g.set_defaults(func=cmd_gen)

    d = sub.add_parser("demo", help="offline demo: baseline + architecture smoke")
    d.add_argument("--samples", type=int, default=8)
    d.add_argument("--grid", type=int, default=256)
    d.add_argument("--coarse", type=int, default=64)
    d.add_argument("--modes", type=int, default=16)
    d.add_argument("--width", type=int, default=32)
    d.add_argument("--seed", type=int, default=1234)
    d.set_defaults(func=cmd_demo)

    t = sub.add_parser("train", help="train FNO & UNet (requires torch)")
    t.add_argument("--samples", type=int, default=200)
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=1234)
    t.add_argument("--out", default="results/train_metrics.json")
    t.set_defaults(func=cmd_train)

    args = p.parse_args()
    return args.func(args)


def np_mean(xs):
    import numpy as np
    return float(np.mean(xs))


if __name__ == "__main__":
    raise SystemExit(main())


def np_mean(xs):
    import numpy as np
    return float(np.mean(xs))

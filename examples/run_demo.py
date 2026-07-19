"""End-to-end offline demo for fno-flow-prediction.

Run with:  python examples/run_demo.py
Requires only numpy (the project's zero-dependency core).
"""

from fno_flow import FNO1D, UNet1D, generate_dataset, lowres_solver_error


def main():
    print("[demo] generating a small Burgers dataset ...")
    data = generate_dataset(n_samples=8, n_grid=256, seed=1234)
    a, u = data["a"], data["u"]

    base_err = sum(
        lowres_solver_error(u[i:i + 1], a[i:i + 1], n_coarse=64)
        for i in range(len(u))
    ) / len(u)
    print(f"[demo] classical low-res solver rel-L2 = {base_err:.4f}")

    fno = FNO1D(n_modes=16, width=32)
    unet = UNet1D(width=16)
    fno_out = fno(a[:1])
    unet_out = unet(a[:1])
    print(f"[demo] FNO forward shape {tuple(fno_out.shape)}  (untrained weights)")
    print(f"[demo] UNet forward shape {tuple(unet_out.shape)}  (untrained weights)")
    print("[demo] done. For real trained errors: pip install torch && python -m fno_flow.train")


if __name__ == "__main__":
    main()

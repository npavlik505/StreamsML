"""Utilities for Sparse Sensor Placement Optimization using PySensors."""

from pathlib import Path
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



def run_pysensors(sa_path: Path, output_dir: Path, num_sensors: int = 10) -> Path:
    """Apply PySensors SSPOR on velocity fields in ``sa_path`` and save sensors + overlay plot."""
    if num_sensors <= 0:
        raise ValueError("num_sensors must be positive")

    # Import here so the module can still be imported without pysensors installed
    try:
        from pysensors import SSPOR
    except Exception as exc:
        raise ImportError(
            "The `pysensors` package is required to run Sparse Sensor Placement "
            "Optimization. Install it with `pip install pysensors`."
        ) from exc

    # Grab mesh and span averages for plotting and datasets
    with h5py.File(sa_path.with_name("mesh.h5"), "r") as m:
        xg = np.asarray(m["x_grid"][:]).squeeze()  # (nx,)
        yg = np.asarray(m["y_grid"][:]).squeeze()  # (ny,)

    with h5py.File(sa_path, "r") as f:
        sa = f["span_average"][:]  # span_average shape is (snapshots, measurements, nx, ny)

    # Measurements is rho=0, U=1, V=2
    illustration = sa[-2, 1, :, :]    # The final snapshot is used for background, shape is nx, ny. sa[-1, (0:rho, 1:u, 2:v, 3:w, 4:E), :, :] 
    u = sa[:, 1, :, :]
    v = sa[:, 2, :, :]
    snapshots, nx, ny = u.shape

    # Feature matrix for SSPOR; U and V
    data = np.concatenate([u.reshape(snapshots, -1),
                           v.reshape(snapshots, -1)], axis=1)

    model = SSPOR(n_sensors=num_sensors)
    model.fit(data)
    sensor_indices = np.asarray(model.selected_sensors, dtype=np.int64)[:num_sensors]

    # Plot background and overlay sensor location as dots
    X, Y = np.meshgrid(xg, yg, indexing="ij")

    # Guard against NaNs/const fields
    finite = np.isfinite(illustration)
    if not finite.any():
        print("No finite snapshot values to plot")
        fig, ax = plt.subplots()
        cf = None
    else:
        cmin = float(np.nanmin(illustration))
        cmax = float(np.nanmax(illustration))

        fig, ax = plt.subplots()

        if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax <= cmin:
            # Flat or invalid range -> use pcolormesh without contour levels
            cf = ax.pcolormesh(X, Y, illustration, shading="auto",
                               cmap="viridis")
        else:
            # Proper range -> contourf with strictly increasing levels
            levels = np.linspace(cmin, cmax, 40)
            cf = ax.contourf(X, Y, illustration, levels=levels, cmap="viridis",
                             vmin=cmin, vmax=cmax, extend="max")

    if cf is not None:
        fig.colorbar(cf, ax=ax, shrink=0.4, fraction=0.1)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


    # Map flat index -> (U/V, ix, iy, x, y) and plot markers
    offset = nx * ny
    records = []  # (index, b'U'/b'V', ix, iy, x, y)

    for i in sensor_indices:
        if i < offset:
            ch, k = b"U", int(i)
        else:
            ch, k = b"V", int(i - offset)
        ix = k // ny
        iy = k %  ny
        xp = float(xg[ix])
        yp = float(yg[iy])

        records.append((int(i), ch, int(ix), int(iy), xp, yp))

        ax.scatter(xp, yp,
                   s=30,
                   c=("blue" if ch == b"U" else "red"),
                   edgecolors="white",
                   linewidths=0.6,
                   zorder=5)

    # Single legend (outside the loop)
    ax.scatter([], [], s=30, c="blue", edgecolors="white", linewidths=0.6, label="U sensors")
    ax.scatter([], [], s=30, c="red",  edgecolors="white", linewidths=0.6, label="V sensors")
    ax.legend(loc="upper right", frameon=True, framealpha=0.65)

    # Ensure output dir exists before saving files
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_path = output_dir / "sensors_overlay.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Sensor overlay saved to {fig_path}")

    # Terminal table output
    print(" index  var  ix  iy      x_phys         y_phys")
    for i, ch, ix, iy, x, y in records:
        print(f"{i:6d}  {ch.decode():>1s}  {ix:3d} {iy:3d}  {x:12.6g}  {y:12.6g}")

    # Write sensors.h5 (raw + compound table + metadata)
    out_file = output_dir / "sensors.h5"
    table_dt = np.dtype([
        ("index", "<i4"),
        ("var",   "S1"),   # b"U" or b"V"
        ("ix",    "<i4"),
        ("iy",    "<i4"),
        ("x",     "<f8"),
        ("y",     "<f8"),
    ])
    table = np.array(records, dtype=table_dt)

    with h5py.File(out_file, "w") as f:
        f.create_dataset("indices_raw", data=sensor_indices.astype(np.int32))
        f.create_dataset("sensors_table", data=table)
        f.attrs["nx"] = int(nx)
        f.attrs["ny"] = int(ny)
        f.attrs["offset"] = int(offset)
        f.attrs["flatten_order"] = "C"
        f.attrs["channel_legend"] = "var: b'U'->U, b'V'->V"

    print(f"Sensing results written to {out_file}")
    return out_file


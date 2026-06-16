#!/usr/bin/env python3
"""Create source-free randomized AIFS benchmark Zarr datasets.

The generated dataset is a valid Zarr v2 tree with one randomized compressed
data chunk hardlinked to every expected data chunk coordinate. Use one of the
fixed benchmark presets:

  make_random_hardlink_zarr.py --throughput /path/to/o800-dummy.zarr
  make_random_hardlink_zarr.py --highres /path/to/o2560-dummy.zarr
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numcodecs
import numpy as np
import zarr


COMPRESSOR = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
CONSTANT_FIELDS = [
    "cos_latitude",
    "cos_longitude",
    "lsm",
    "sdor",
    "sin_latitude",
    "sin_longitude",
    "slor",
    "z",
]
PRESSURE_LEVELS = ["100", "1000", "150", "200", "250", "300", "400", "50", "500", "600", "700", "850", "925"]
BASE_STATS = ["count", "has_nans", "maximum", "mean", "minimum", "squares", "stdev", "sums"]
TENDENCY_WINDOWS = ["12h", "1d", "6h"]

THROUGHPUT_VARIABLES = [
    "10u", "10v", "2d", "2t", "cos_julian_day", "cos_latitude", "cos_local_time", "cos_longitude",
    "cp", "insolation", "lsm", "msl", "q_100", "q_1000", "q_150", "q_200", "q_250", "q_300",
    "q_400", "q_50", "q_500", "q_600", "q_700", "q_850", "q_925", "sdor", "sin_julian_day",
    "sin_latitude", "sin_local_time", "sin_longitude", "skt", "slor", "sp", "t_100", "t_1000",
    "t_150", "t_200", "t_250", "t_300", "t_400", "t_50", "t_500", "t_600", "t_700",
    "t_850", "t_925", "tcw", "tp", "u_100", "u_1000", "u_150", "u_200", "u_250", "u_300",
    "u_400", "u_50", "u_500", "u_600", "u_700", "u_850", "u_925", "v_100", "v_1000",
    "v_150", "v_200", "v_250", "v_300", "v_400", "v_50", "v_500", "v_600", "v_700",
    "v_850", "v_925", "w_100", "w_1000", "w_150", "w_200", "w_250", "w_300", "w_400",
    "w_50", "w_500", "w_600", "w_700", "w_850", "w_925", "z", "z_100", "z_1000",
    "z_150", "z_200", "z_250", "z_300", "z_400", "z_50", "z_500", "z_600", "z_700",
    "z_850", "z_925",
]

HIGHRES_VARIABLES = [
    "10u", "10v", "2d", "2t", "cos_julian_day", "cos_latitude", "cos_local_time", "cos_longitude",
    "cos_solar_zenith_angle", "cp", "lsm", "msl", "q_100", "q_1000", "q_150", "q_200", "q_250",
    "q_300", "q_400", "q_50", "q_500", "q_600", "q_700", "q_850", "q_925", "sdor",
    "sin_julian_day", "sin_latitude", "sin_local_time", "sin_longitude", "skt", "slor", "t_100",
    "t_1000", "t_150", "t_200", "t_250", "t_300", "t_400", "t_50", "t_500", "t_600",
    "t_700", "t_850", "t_925", "tcw", "tp", "u_100", "u_1000", "u_150", "u_200", "u_250",
    "u_300", "u_400", "u_50", "u_500", "u_600", "u_700", "u_850", "u_925", "v_100",
    "v_1000", "v_150", "v_200", "v_250", "v_300", "v_400", "v_50", "v_500", "v_600",
    "v_700", "v_850", "v_925", "w_100", "w_1000", "w_150", "w_200", "w_250", "w_300",
    "w_400", "w_50", "w_500", "w_600", "w_700", "w_850", "w_925", "z", "z_100",
    "z_1000", "z_150", "z_200", "z_250", "z_300", "z_400", "z_50", "z_500", "z_600",
    "z_700", "z_850", "z_925",
]


@dataclass(frozen=True)
class Preset:
    name: str
    resolution: str
    variables: list[str]
    data_shape: tuple[int, int, int, int]
    data_chunks: tuple[int, int, int, int]
    coord_chunks: tuple[int]
    start_date: str
    end_date: str
    statistics_end_date: str
    description: str


PRESETS = {
    "throughput": Preset(
        name="throughput",
        resolution="O800",
        variables=THROUGHPUT_VARIABLES,
        data_shape=(14240, 101, 1, 2588800),
        data_chunks=(1, 101, 1, 2588800),
        coord_chunks=(80900,),
        start_date="2023-01-01T00:00:00",
        end_date="2023-12-31T18:00:00",
        statistics_end_date="2023-10-19T18:00:00",
        description="Source-free randomized O800 throughput dummy dataset",
    ),
    "highres": Preset(
        name="highres",
        resolution="O2560",
        variables=HIGHRES_VARIABLES,
        data_shape=(2920, 100, 1, 26306560),
        data_chunks=(1, 100, 1, 1644160),
        coord_chunks=(205520,),
        start_date="2023-09-01T00:00:00",
        end_date="2023-09-07T18:00:00",
        statistics_end_date="2023-09-06T06:00:00",
        description="Source-free randomized O2560 high-resolution dummy dataset",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    preset = parser.add_mutually_exclusive_group(required=True)
    preset.add_argument("--throughput", action="store_true", help="Generate the O800 throughput dataset")
    preset.add_argument("--highres", action="store_true", help="Generate the O2560 high-resolution dataset")
    parser.add_argument("destination", type=Path, help="Destination Zarr dataset")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--force", action="store_true", help="Replace destination if it exists")
    return parser.parse_args()


def chunk_counts(shape: tuple[int, ...], chunks: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(math.ceil(size / chunk) for size, chunk in zip(shape, chunks, strict=True))


def chunk_names(counts: tuple[int, ...]) -> Iterable[str]:
    for coordinate in itertools.product(*(range(count) for count in counts)):
        yield ".".join(str(index) for index in coordinate)


def dimensions_for(name: str) -> list[str]:
    if name == "data":
        return ["time", "variable", "ensemble", "cell"]
    if name == "dates":
        return ["time"]
    if name in ("latitudes", "longitudes"):
        return ["cell"]
    return ["variable"]


def variable_metadata(variables: list[str]) -> dict[str, dict[str, dict[str, object]]]:
    metadata = {}
    for variable in variables:
        param, _, level = variable.partition("_")
        mars: dict[str, object] = {
            "class": "rd",
            "domain": "g",
            "expver": "dummy",
            "param": param,
            "step": 0,
            "stream": "oper",
            "time": 0,
            "type": "an",
        }
        if level.isdigit():
            mars["levtype"] = "pl"
            mars["levelist"] = int(level)
        else:
            mars["levtype"] = "sfc"
        metadata[variable] = {"mars": mars}
    return metadata


def stat_values(name: str, nvars: int, ndates: int) -> np.ndarray:
    if name.endswith("count"):
        return np.full(nvars, ndates, dtype=np.int64)
    if name.endswith("has_nans"):
        return np.zeros(nvars, dtype=np.bool_)
    if name.endswith("maximum"):
        return np.ones(nvars, dtype=np.float64)
    if name.endswith("minimum"):
        return -np.ones(nvars, dtype=np.float64)
    if name.endswith("stdev"):
        return np.ones(nvars, dtype=np.float64)
    if name.endswith("squares"):
        return np.full(nvars, ndates / 3.0, dtype=np.float64)
    return np.zeros(nvars, dtype=np.float64)


def create_array(group: zarr.hierarchy.Group, name: str, data: np.ndarray, chunks: tuple[int, ...]) -> None:
    array = group.create_dataset(
        name,
        data=data,
        chunks=chunks,
        compressor=COMPRESSOR,
        overwrite=True,
    )
    array.attrs["_ARRAY_DIMENSIONS"] = dimensions_for(name)


def octahedral_ring_points(n: int, row: int) -> int:
    k = row if row < n else 2 * n - 1 - row
    return 20 + 4 * k


def gaussian_latitudes(n: int) -> np.ndarray:
    try:
        from scipy.special import roots_legendre

        roots, _ = roots_legendre(2 * n)
    except Exception:
        roots, _ = np.polynomial.legendre.leggauss(2 * n)
    return np.degrees(np.arcsin(roots[::-1]))


def write_coordinates(group: zarr.hierarchy.Group, preset: Preset) -> None:
    ncell = preset.data_shape[-1]
    n = int(preset.resolution[1:])
    if sum(octahedral_ring_points(n, row) for row in range(2 * n)) != ncell:
        raise ValueError(f"{preset.resolution} does not match ncell={ncell}")

    latitudes = np.empty(ncell, dtype=np.float64)
    row_latitudes = gaussian_latitudes(n)
    offset = 0
    for row, latitude in enumerate(row_latitudes):
        points = octahedral_ring_points(n, row)
        latitudes[offset : offset + points] = latitude
        offset += points
    create_array(group, "latitudes", latitudes, preset.coord_chunks)

    longitudes = np.empty(ncell, dtype=np.float64)
    offset = 0
    for row in range(2 * n):
        points = octahedral_ring_points(n, row)
        longitudes[offset : offset + points] = np.arange(points, dtype=np.float64) * (360.0 / points)
        offset += points
    create_array(group, "longitudes", longitudes, preset.coord_chunks)


def write_dates(group: zarr.hierarchy.Group, preset: Preset) -> None:
    # Anemoi cloned datasets use one fewer date entry than the data time axis.
    ndates = preset.data_shape[0] - 1
    start = np.datetime64(preset.start_date, "s")
    offsets = (np.arange(ndates, dtype=np.int64) * 6).astype("timedelta64[h]")
    create_array(group, "dates", start + offsets, (ndates,))


def root_attrs(preset: Preset) -> dict[str, object]:
    nvars = len(preset.variables)
    mean = stat_values("mean", nvars, preset.data_shape[0])
    stdev = stat_values("stdev", nvars, preset.data_shape[0])
    minimum = stat_values("minimum", nvars, preset.data_shape[0])
    maximum = stat_values("maximum", nvars, preset.data_shape[0])
    return {
        "allow_nans": False,
        "attribution": "Generated dummy dataset",
        "chunks": list(preset.data_chunks),
        "constant_fields": CONSTANT_FIELDS,
        "data_request": {"grid": preset.resolution, "source": "random-hardlink-dummy"},
        "description": preset.description,
        "dtype": "float32",
        "end_date": preset.end_date,
        "ensemble_dimension": 1,
        "field_shape": [preset.data_shape[-1]],
        "flatten_grid": True,
        "frequency": "6h",
        "history": ["generated by make_random_hardlink_zarr.py"],
        "latest_write_timestamp": preset.end_date,
        "licence": "generated dummy data",
        "missing_dates": [],
        "order_by": [{"valid_datetime": "ascending"}, {"param_level": "ascending"}, {"number": "ascending"}],
        "proj_string": None,
        "provenance_load": {},
        "recipe": {"preset": preset.name, "kind": "random-hardlink"},
        "remapping": {"param_level": "{param}_{levelist}"},
        "resolution": preset.resolution,
        "shape": list(preset.data_shape),
        "start_date": preset.start_date,
        "statistics": {
            "maximum": maximum.tolist(),
            "mean": mean.tolist(),
            "minimum": minimum.tolist(),
            "stdev": stdev.tolist(),
        },
        "statistics_end_date": preset.statistics_end_date,
        "statistics_start_date": preset.start_date,
        "total_number_of_files": int(preset.data_shape[0]),
        "total_size": 0,
        "uuid": f"random-hardlink-{preset.name}",
        "variables": preset.variables,
        "variables_metadata": variable_metadata(preset.variables),
        "variables_with_nans": [],
        "version": "generated",
    }


def write_statistics(group: zarr.hierarchy.Group, preset: Preset) -> None:
    nvars = len(preset.variables)
    ndates = preset.data_shape[0]
    for stat in BASE_STATS:
        create_array(group, stat, stat_values(stat, nvars, ndates), (nvars,))
    for window in TENDENCY_WINDOWS:
        for stat in BASE_STATS:
            name = f"statistics_tendencies_{window}_{stat}"
            create_array(group, name, stat_values(stat, nvars, ndates), (nvars,))


def write_random_hardlinked_data(group: zarr.hierarchy.Group, destination: Path, preset: Preset, seed: int) -> None:
    data = group.create_dataset(
        "data",
        shape=preset.data_shape,
        chunks=preset.data_chunks,
        dtype="<f4",
        compressor=COMPRESSOR,
        fill_value=np.nan,
        overwrite=True,
    )
    data.attrs["_ARRAY_DIMENSIONS"] = dimensions_for("data")

    rng = np.random.default_rng(seed)
    random_chunk = rng.random(preset.data_chunks, dtype=np.float32)
    random_chunk *= 2.0
    random_chunk -= 1.0
    random_chunk = np.ascontiguousarray(random_chunk)

    data_dir = destination / "data"
    anchor = data_dir / "0.0.0.0"
    anchor.write_bytes(COMPRESSOR.encode(random_chunk))

    total = 0
    for name in chunk_names(chunk_counts(preset.data_shape, preset.data_chunks)):
        target = data_dir / name
        if target == anchor:
            total += 1
            continue
        os.link(anchor, target)
        total += 1

    print(f"data_chunks={total}")
    print(f"anchor_size={anchor.stat().st_size}")
    print(f"anchor_links={anchor.stat().st_nlink}")


def main() -> None:
    args = parse_args()
    preset = PRESETS["throughput" if args.throughput else "highres"]
    destination = args.destination.resolve()

    if destination.exists():
        if not args.force:
            raise SystemExit(f"{destination} already exists; pass --force to replace it")
        shutil.rmtree(destination)

    group = zarr.open_group(str(destination), mode="w")
    group.attrs.update(root_attrs(preset))
    write_dates(group, preset)
    write_coordinates(group, preset)
    write_statistics(group, preset)
    write_random_hardlinked_data(group, destination, preset, args.seed)

    print(f"preset={preset.name}")
    print(f"destination={destination}")
    print(f"shape={preset.data_shape}")
    print(f"chunks={preset.data_chunks}")
    print(f"variables={len(preset.variables)}")


if __name__ == "__main__":
    main()

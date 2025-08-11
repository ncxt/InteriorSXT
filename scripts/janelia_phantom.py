"""
Loader script for the jrc_macrophage-2 phantom
"""

# Standard Libraries
import numpy as np
from pathlib import Path

# Third-Party Libraries
import imageio
import zarr
import dask.array as da
from scipy import ndimage as ndi
from tqdm.auto import tqdm

# Local Modules
from conf import JANELIA_FOLDER
from phantom.read_write_mrc import write_mrc, read_mrc
from fibsem_tools import read_xarray

url_s2 = "s3://janelia-cosem-datasets/jrc_sum159-1/jrc_sum159-1.n5/em/fibsem-uint16/s2"
creds = {"anon": True}  # anonymous credentials for s3
baseurl = "s3://janelia-cosem-datasets/jrc_macrophage-2/jrc_macrophage-2.n5"


def ensure_celldata(scale: int = 2):
    array = read_xarray(baseurl + f"/em/fibsem-uint16/s{scale}", storage_options=creds)
    print(f"Ensuring fibsem-uint16 with resolution s{scale}")
    print(array.pixelResolution["dimensions"])

    filepath = JANELIA_FOLDER / f"stack_s{scale}_em.mrc"
    filepath_tiff = JANELIA_FOLDER / f"stack_s{scale}_em.tiff"
    print(filepath, filepath.exists())
    if not filepath.exists():
        stack_em = array.compute().to_numpy()
        print(stack_em.shape, stack_em.dtype)
        write_mrc(filepath, stack_em)
        imageio.volsave(filepath_tiff, stack_em)


def ensure_cellseg(scale: int = 2):
    segs = ["pm_pred", "mito_seg", "er_seg"]

    group = zarr.open(zarr.N5FSStore(baseurl, anon=True))
    for label in tqdm(segs):
        filepath = JANELIA_FOLDER / f"stack_s{scale}_{label}.tiff"
        if not filepath.exists():
            zdata_em = group[f"labels/{label}/s{scale}"]
            ddata = da.from_array(zdata_em, chunks=zdata_em.chunks)
            stack_seg = ddata.compute()
            imageio.volsave(filepath, stack_seg)


def make_phantom(scale_seg: int = 2, scale_lac: int = 2):
    # These were extracted form the janelia phantom meta data
    imageres = {
        2: np.array(
            [
                13.44,
                16.0,
                16.0,
            ]
        ),
        3: np.array([26.88, 32.0, 32.0]),
        4: np.array(
            [
                53.76,
                64.0,
                64.0,
            ]
        ),
    }
    res_out = imageres[scale_lac]
    # The GV image is flipped in axis 1
    ensure_celldata(scale=scale_seg)
    em_data = read_mrc(JANELIA_FOLDER / f"stack_s{scale_seg}_em.mrc")[:, ::-1, :]
    # The pm_pred was one slice larger in all scales
    ensure_cellseg(scale=scale_seg)
    pm_data = imageio.volread(JANELIA_FOLDER / f"stack_s{scale_seg}_pm_pred.tiff")[:-1]
    er_data = imageio.volread(JANELIA_FOLDER / f"stack_s{scale_seg}_er_seg.tiff")
    mito_data = imageio.volread(JANELIA_FOLDER / f"stack_s{scale_seg}_mito_seg.tiff")

    gb = 100 / imageres[scale_seg]
    pm_smooth = ndi.gaussian_filter(1.0 * (pm_data > 0), gb)

    crop = [int(2500 / imageres[scale_seg][0]), int(6400 / imageres[scale_seg][2])]
    pm_fill = ndi.binary_fill_holes(pm_smooth > 0.2)
    pm_fill[: crop[0], :, :] = 0
    pm_fill[-crop[0] :, :, :] = 0
    pm_fill[:, :, : crop[1]] = 0
    pm_fill[:, :, -crop[1] :] = 0

    cellmask = pm_fill > 0
    er_mask = er_data > 0
    mito_mask = mito_data > 0

    lac_table = [
        0,
        0.2,
        0.29,
        0.33,
    ]  # approx LAC for VOID, mean unlabeled cell, mito an ER
    em_signal = [
        np.mean(em_data[cellmask == 0]),
        np.mean(em_data[cellmask > 0]),
        np.mean(em_data[er_mask > 0]),
        np.mean(em_data[mito_mask > 0]),
    ]

    p = np.polyfit(em_signal, lac_table, 1)

    # scale_mask
    ensure_celldata(scale=scale_lac)
    em_data_out = read_mrc(JANELIA_FOLDER / f"stack_s{scale_lac}_em.mrc")[:, ::-1, :]
    zoom = np.array(em_data_out.shape) / np.array(em_data.shape)
    cellmask_out = ndi.zoom(cellmask, zoom, order=0)
    phantom_scaled = cellmask_out * (em_data_out * p[0] + p[1])
    phantom_scaled[phantom_scaled < 0] = 0

    crop0 = [int(3216 / imageres[scale_lac][0]), int(2680 / imageres[scale_lac][0])]
    crop2 = [int(6400 / imageres[scale_lac][2]), int(6400 / imageres[scale_lac][2])]
    phantom_crop = phantom_scaled[crop0[0] : -crop0[1], :, crop2[0] : -crop2[1]]

    # label_res =
    target_res = np.array(3 * [20 * 2 ** (scale_lac - 2)])
    zoom = res_out / target_res
    phantom_out = ndi.zoom(phantom_crop, zoom, order=1)

    write_mrc(
        JANELIA_FOLDER / f"phantom_{int(target_res[0])}nm.mrc",
        phantom_out.astype("float32"),
    )


def load_janelia_phantom(res: int):
    filepath = JANELIA_FOLDER / f"phantom_{res}nm.mrc"
    kwargs = {
        20: {"scale_seg": 3, "scale_lac": 2},
        40: {"scale_seg": 3, "scale_lac": 3},
        80: {"scale_seg": 4, "scale_lac": 4},
    }
    if not filepath.exists():
        make_phantom(**kwargs[res])
    return read_mrc(filepath), res / 1000


if __name__ == "__main__":
    # ensure_celldata(scale = 4)
    # ensure_cellseg(scale = 4)
    # make_phantom(4,4)
    make_phantom(3, 3)

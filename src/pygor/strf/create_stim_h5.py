"""
Create jitter-noise H5 stimulus files for QDSpy.

Change the parameters below, then run the script. It will generate the
basis noise, subsample, jitter-upscale, and save the result as an H5 file
in the same directory as this script.

CRITICAL REPRODUCIBILITY NOTE:
----------------------------------------
For identical output, you MUST use the EXACT same values for:
- NOISE_SEED: The random seed
- NOISE_MAX_DIMS: Maximum noise dimensions (y, x)
- BOX_SIZE_AU: Determines final grid_shape
- FRAME_MAX: Number of frames to use (None means all available)
- SHIFT_RATIO: Determines possible shift positions
- SHIFT_N_FRAMES: How many consecutive frames share the same shift

!!NB!!: If you change ANY of these parameters, you will get a different noise pattern, even with the same seed.
This is expected behavior. Record all three parameters for reproducible experiments.

Simen's core parameters for reproducibility:
- NOISE_SEED = 1312
- NOISE_MAX_DIMS = (100, 100)
- BOX_SIZE_AU = 200
- FRAME_MAX = 2500, 10000, 30000 (check against IGOR pxp anaylsis)
- SHIFT_RATIO = 1/4
- SHIFT_N_FRAMES = 1

[46. 73.]
[23. 37.]
[12. 19.]
[ 6. 10.]
[3. 5.]
[2. 3.]
"""

import itertools
import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np

# # %%
# from igor2 import binarywave

# target_arr = np.array(
#     binarywave.load(
#         "/home/simen/Documents/Git_repos/2p_analysis/pygor/src/pygor/strf/NoiseArray3D.ibw"
#     )["wave"]["wData"]
# ).T


# %%
# ============================================================================
# PARAMETERS — edit these
# ============================================================================

# Screen / optics (usually constant)
SCREEN_WIDTH_PIX_AU = 1820
SCREEN_HEIGHT_PIX_AU = 1140

# Box size in QDSpy arbitrary units — determines grid_shape automatically
BOX_SIZE_AU = 200  # e.g. 5, 10, 25, 50, 100, 200, 400, 800

# Basis noise
NOISE_MAX_DIMS = (100, 100)  # (y, x) — must be >= largest grid shape you'll use
NOISE_MAX_FRAMES = 30000  # max temporal extent of basis noise
NOISE_SEED = 1312

# Subsampling / jitter
FRAME_MAX = None  # number of frames to use (None = all available)
SHIFT_RATIO = 1 / 4  # enter as ratio, e.g. 1/4 (1 means no shift)
SHIFT_N_FRAMES = 1  # how many consecutive frames share the same shift

# What to save
SAVE_SINGLE_COLOUR = True  # True  -> save single-colour variant (for ColourSwitcher)
# False -> save full 6-colour variant
SAVE_DIR = pathlib.Path(__file__).parent  # saves next to this script

# ============================================================================
# END OF PARAMETERS
# ============================================================================


def calculate_boxes_on_screen(
    box_width_pix,
    screen_width_pix_au=SCREEN_WIDTH_PIX_AU,
    screen_height_pix_au=SCREEN_HEIGHT_PIX_AU,
):
    blocks_horz = screen_width_pix_au / box_width_pix
    blocks_vert = screen_height_pix_au / box_width_pix
    return np.array([blocks_vert, blocks_horz])


def jitter_upscale(noise_array, shift_list, num_frames):
    x_unique = np.unique(shift_list[0])
    y_unique = np.unique(shift_list[1])
    num_shifts = len(x_unique)
    target = np.ones((num_shifts, num_shifts), dtype=np.uint8)
    noise_upscaled = np.kron(noise_array.astype(np.uint8), target)
    for frame in range(num_frames):
        frame_shift = shift_list[:, frame]
        y_pix = round(frame_shift[0] * num_shifts)
        x_pix = round(frame_shift[1] * num_shifts)
        for color in range(noise_upscaled.shape[0]):
            noise_upscaled[color, frame] = np.pad(
                noise_upscaled[color, frame],
                ((int(x_pix), 0), (int(y_pix), 0)),
            )[: noise_upscaled.shape[2], : noise_upscaled.shape[3]]
    return noise_upscaled


def main():
    # --- Derive grid shape from box size ---
    grid_shape = np.ceil(calculate_boxes_on_screen(BOX_SIZE_AU)).astype(int)
    print(f"Box size {BOX_SIZE_AU} AU  ->  grid shape (y, x) = {tuple(grid_shape)}")

    # --- Build basis noise (6 colour channels: R, G, B, UV1, UV2, UV3) ---
    np.random.seed(NOISE_SEED)
    basis_noise = np.random.randint(
        0, 2, (4, NOISE_MAX_FRAMES, NOISE_MAX_DIMS[0], NOISE_MAX_DIMS[1])
    ).astype(np.uint8)
    uv_part = np.expand_dims(basis_noise[3], axis=0)
    basis_noise = np.concatenate((basis_noise, uv_part, uv_part), axis=0)

    # --- Subsample ---
    if grid_shape[0] > NOISE_MAX_DIMS[0] or grid_shape[1] > NOISE_MAX_DIMS[1]:
        raise ValueError(
            f"grid_shape {tuple(grid_shape)} exceeds NOISE_MAX_DIMS {NOISE_MAX_DIMS}. "
            f"Increase NOISE_MAX_DIMS or use a larger BOX_SIZE_AU."
        )
    frame_max = FRAME_MAX  # None means take all
    noise = basis_noise[:, :frame_max, : grid_shape[0], : grid_shape[1]].astype("uint8")

    num_frames = noise.shape[1]
    y_boxes = noise.shape[2]
    x_boxes = noise.shape[3]
    print(f"Noise shape: {noise.shape}  (colours, frames, y, x)")

    subsample_params = {
        "grid_shape": tuple(grid_shape),
        "frame_max": num_frames,
        "shift_ratio": SHIFT_RATIO,
        "shift_n_frames": SHIFT_N_FRAMES,
        "seed": NOISE_SEED,
    }

    # --- Compute jitter shift list ---
    shift_possibilities_base = np.arange(0, 1, SHIFT_RATIO)
    assert num_frames % SHIFT_N_FRAMES == 0, (
        f"shift_n_frames ({SHIFT_N_FRAMES}) must divide evenly into num_frames ({num_frames})"
    )
    np.random.seed(NOISE_SEED)
    all_possible_positions = np.array(
        list(itertools.product(shift_possibilities_base, shift_possibilities_base))
    )
    shift_list = np.empty((2, num_frames))
    choice_history = []
    for i in range(num_frames)[::SHIFT_N_FRAMES]:
        chose_from = np.arange(0, len(all_possible_positions))
        chose_from_historical_bias = np.delete(chose_from, choice_history)
        choice = np.random.choice(chose_from_historical_bias)
        choice_history.append(choice)
        shift_list[:, i : i + SHIFT_N_FRAMES] = np.repeat(
            all_possible_positions[choice], SHIFT_N_FRAMES, axis=0
        ).reshape(2, SHIFT_N_FRAMES)
        if len(choice_history) == len(all_possible_positions):
            choice_history = []

    # --- Jitter upscale ---
    # For single-colour mode, only upscale channel 0 to save memory
    if SAVE_SINGLE_COLOUR:
        print("Jitter upscaling (single colour channel only)...")
        noise_1ch = np.expand_dims(noise[0], 0)  # (1, frames, y, x)
        jitter_noise = jitter_upscale(noise_1ch, shift_list, num_frames).astype("uint8")
    else:
        print("Jitter upscaling (all 6 channels)...")
        jitter_noise = jitter_upscale(noise, shift_list, num_frames).astype("uint8")
    print(f"Jitter noise shape: {jitter_noise.shape}")

    # --- Determine filename and save ---
    is_white = np.all(basis_noise[0] == basis_noise[1])

    if SAVE_SINGLE_COLOUR:
        colour_tag = "SINGLEcolour"
        subsample_params["frame_max"] = jitter_noise.shape[1]
    elif is_white:
        colour_tag = "WHITE"
    else:
        colour_tag = ""

    save_str = (
        f"jitternoise_{colour_tag + '_' if colour_tag else ''}"
        f"{subsample_params['frame_max']}x{grid_shape[0]}x{grid_shape[1]}_"
        f"{np.round(SHIFT_RATIO, 3)}_{SHIFT_N_FRAMES}"
    )
    out_path = SAVE_DIR / f"{save_str}.h5"

    with h5py.File(out_path, "w") as f:
        if SAVE_SINGLE_COLOUR:
            f["noise"] = noise[0]
            f["noise_jitter"] = np.swapaxes(jitter_noise[0], 0, 2)
        else:
            f["noise"] = noise
            f["noise_jitter"] = jitter_noise
        f["shift"] = shift_list
        f.attrs.update(subsample_params)

    print(f"Saved: {out_path}")

    # Sanity check
    with h5py.File(out_path, "r") as f:
        for key in f.keys():
            print(f"  {f[key]}")
        for attr in f.attrs:
            print(f"  {attr}: {f.attrs[attr]}")
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(
            f["noise_jitter"][:, :, 0].T, cmap="Greys_r", origin="lower"
        )  # First frame
        ax[0].set_title("First jittered noise frame")
        ax[0].axis("off")
        ax[1].imshow(
            f["noise_jitter"][:, :, -1].T, cmap="Greys_r", origin="lower"
        )  # Last frame
        ax[1].set_title("Last jittered noise frame")
        ax[1].axis("off")
        plt.show()

        # print("testing target array equality...")
        # print("- target array shape:", target_arr.T.shape)
        # print("- file array shape:", f["noise_jitter"].shape)
        # assert np.array_equal(target_arr.T, f["noise_jitter"])


if __name__ == "__main__":
    main()

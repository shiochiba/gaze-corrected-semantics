#!/usr/bin/env python3
"""Plot eyetracking points on movies and save overlay videos.

This script combines:
- Gaze loading and alignment logic from the semantic-segmentation notebook.
- Video overlay/export logic from /projects/eyetracking/plot-et-on-movie-gif.py.
"""

from __future__ import annotations

import argparse
import re
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import sys
sys.path.append('/hehe/projects/gaze-corrected-semantics/jen_reference')
from jen_reference.eyetracking_helper import *

DEFAULT_MOVIES = [f"test2_run-{i:02d}" for i in range(1, 7)] + [f"train{i:02d}" for i in range(1, 11)]
CONDITION_SUFFIX_RE = re.compile(r"(impmatch|match|ecprod|rest)$")
DEFAULT_EYETRACKING_OVERLAY_SAVEPATH = Path("/hehe/projects/gaze-corrected-semantics/jen_reference/data/eyetracking-overlays")


def align_gaze_to_frames_mean(
    gaze_df: pd.DataFrame,
    n_frames: int,
    eyetracking_sampling_rate: float = 250.0,
    movie_fps: float = 15.0,
) -> np.ndarray:
    """Average gaze samples per movie frame, matching the original overlay script."""
    if n_frames <= 0:
        return np.zeros((0, 2), dtype=float)

    xy = gaze_df[["x", "y"]].to_numpy(dtype=float)
    if xy.shape[0] == 0:
        return np.zeros((n_frames, 2), dtype=float)
    if xy.shape[0] == 1:
        out = np.repeat(xy, n_frames, axis=0)
        return out

    samples_per_frame = eyetracking_sampling_rate / movie_fps
    frame_idx = np.floor(np.arange(xy.shape[0]) / samples_per_frame).astype(int)

    df = pd.DataFrame({"frame": frame_idx, "x": xy[:, 0], "y": xy[:, 1]})
    averaged = df.groupby("frame")[["x", "y"]].mean()
    averaged = averaged.reindex(range(n_frames)).interpolate(method="linear", limit_direction="both")
    averaged = averaged.fillna(method="ffill").fillna(method="bfill")
    return averaged[["x", "y"]].to_numpy(dtype=float)


def infer_coordinate_space(gaze_df: pd.DataFrame, frame_width: int, frame_height: int) -> str:
    """Infer whether gaze is already in movie pixels or in 1024x768 screen pixels."""
    x = gaze_df["x"].to_numpy(dtype=float)
    y = gaze_df["y"].to_numpy(dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return "screen1024x768"

    x = x[finite]
    y = y[finite]

    in_movie = (x >= -5) & (x <= frame_width + 5) & (y >= -5) & (y <= frame_height + 5)
    frac_in_movie = float(np.mean(in_movie))
    if frac_in_movie >= 0.95:
        return "movie"

    return "screen1024x768"


def map_point_to_video_pixels(
    raw_x: float,
    raw_y: float,
    frame_width: int,
    frame_height: int,
    coord_space: str,
) -> tuple[int, int, float, float]:
    """Map gaze point to video pixel coordinates and return both mapped float and clipped int."""
    if coord_space == "movie":
        mapped_x = raw_x
        mapped_y = raw_y
    elif coord_space == "screen1024x768":
        x_offset = (1024 - 768) / 2
        y_offset = 0
        mapped_x = (raw_x - x_offset) * (frame_width / 768.0)
        mapped_y = (raw_y - y_offset) * (frame_height / 768.0)
    else:
        raise ValueError(f"Unsupported coord_space: {coord_space}")

    x_i = int(np.clip(np.rint(mapped_x), 0, frame_width - 1))
    y_i = int(np.clip(np.rint(mapped_y), 0, frame_height - 1))
    return x_i, y_i, mapped_x, mapped_y


def parse_task_key_from_filename(file_path: str) -> str | None:
    patterns = [
        r"task-(.*?)_et_corrected\.npy$",
        r"task-(.*?)_corrected\.npy$",
        r"task-(.*?) corrected\.npy$",
    ]

    for pattern in patterns:
        m = re.search(pattern, file_path)
        if m:
            return m.group(1)

    return None


def normalize_movie_key(task_key: str) -> str:
    # already-renamed test2 keys e.g. test2match_run-03 -> test2_run-03
    m = re.match(r"(test\d+)(impmatch|match|ecprod|rest)_run-(\d+)", task_key)
    if m:
        return f"{m.group(1)}_run-{m.group(3)}"

    # original BIDS test2 keys e.g. test2match03_run-01 -> test2_run-05
    m = re.match(r"(test\d+)(impmatch|match|ecprod|rest)(\d+)_run-(\d+)", task_key)
    if m:
        session_num = int(m.group(3))
        run_num = int(m.group(4))
        global_run = (session_num - 1) * 2 + run_num
        return f"{m.group(1)}_run-{global_run:02d}"

    key = CONDITION_SUFFIX_RE.sub("", task_key)

    m = re.match(r"train(\d+)", key)
    if m:
        return f"train{int(m.group(1)):02d}"

    m = re.match(r"trn(\d+)", key)
    if m:
        return f"train{int(m.group(1)):02d}"

    m = re.match(r"(test\d+)", key)
    if m:
        return m.group(1)

    return key


def load_psychedelic_eyetracking(
    subject: str,
    session: str,
    eyetracking_root: Path,
    runtype: str | None = None,
    makefivetwelve: bool = True,
) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    path = eyetracking_root / "psychedelic" / f"sub-{subject}" / f"ses-{session}" / "eyetracking"
    files = sorted(glob(str(path / "*corrected.npy")))

    for file in files:
        key = parse_task_key_from_filename(file)
        if key is None:
            continue

        if "trn" in key:
            key = key.replace("trn0", "train")

        if key in data:
            raise ValueError(f"Duplicate key found: {key}")

        df = pd.DataFrame(np.load(file, allow_pickle=True)).rename(columns={0: "x", 1: "y"})
        df["x"] = df["x"].interpolate(method="linear", limit_direction="both")
        df["y"] = df["y"].interpolate(method="linear", limit_direction="both")
        data[key] = df

    if runtype is not None:
        data = {k: v for k, v in data.items() if runtype in k}

    if makefivetwelve:
        for key in list(data.keys()):
            data[key] = convert_to_movie_pixels(data[key], clip=True)

    return data


def dedupe_and_normalize_gaze_keys(gaze_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    for raw_key, df in gaze_data.items():
        norm_key = normalize_movie_key(raw_key)
        print(f"  normalizing: {raw_key!r} -> {norm_key!r}")
        if norm_key in out:
            if len(df) > len(out[norm_key]):
                print(f"  Replacing duplicate gaze key {norm_key} with longer sample set from {raw_key}.")
                out[norm_key] = df
            else:
                print(f"  Keeping existing gaze key {norm_key}; skipping duplicate from {raw_key}.")
        else:
            out[norm_key] = df

    return out


def movie_key_to_video_stem(movie_key: str) -> str:
    # train01 -> trn001
    m = re.match(r"train(\d+)$", movie_key)
    if m:
        return f"trn{int(m.group(1)):03d}"
    # test2_run-03 -> test2 (all runs use the same video)
    m = re.match(r"(test\d+)_run-\d+$", movie_key)
    if m:
        return m.group(1)
    return movie_key


def resolve_video_path(movie_key: str, movie_dir: Path) -> Path | None:
    stem = movie_key_to_video_stem(movie_key)
    candidates = [
        movie_dir / f"{stem}_redcross.mp4",
        movie_dir / f"{stem}.mp4",
        movie_dir / f"{movie_key}_redcross.mp4",
        movie_dir / f"{movie_key}.mp4",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def parse_bgr_color(color_text: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in color_text.split(",")]
    if len(parts) != 3:
        raise ValueError("--dot-color must contain exactly 3 comma-separated integers.")

    color = tuple(int(part) for part in parts)
    if any(c < 0 or c > 255 for c in color):
        raise ValueError("--dot-color values must be in [0, 255].")

    return color  # type: ignore[return-value]


def overlay_gaze_on_video(
    input_video_path: Path,
    output_video_path: Path,
    gaze_df: pd.DataFrame,
    dot_radius: int,
    dot_color_bgr: tuple[int, int, int],
    alignment: str,
    coord_space: str,
    eyetracking_sampling_rate: float,
    movie_fps: float,
    print_mapping: bool,
) -> None:
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 15.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_frames_for_alignment = frame_count if frame_count > 0 else len(gaze_df)
    if alignment == "frame-mean":
        aligned_xy = align_gaze_to_frames_mean(
            gaze_df,
            n_frames_for_alignment,
            eyetracking_sampling_rate=eyetracking_sampling_rate,
            movie_fps=movie_fps,
        )
    elif alignment == "lanczos":
        aligned_xy = align_gaze_to_frames(gaze_df, n_frames_for_alignment)
    else:
        raise ValueError(f"Unsupported alignment: {alignment}")

    chosen_space = coord_space
    if coord_space == "auto":
        chosen_space = infer_coordinate_space(pd.DataFrame(aligned_xy, columns=["x", "y"]), width, height)
    print(f"Using coord space: {chosen_space}")
    print(f"Using alignment: {alignment}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(aligned_xy):
            raw_x = float(aligned_xy[frame_idx, 0])
            raw_y = float(aligned_xy[frame_idx, 1])

            if np.isfinite(raw_x) and np.isfinite(raw_y):
                x_i, y_i, mapped_x, mapped_y = map_point_to_video_pixels(
                    raw_x,
                    raw_y,
                    frame_width=width,
                    frame_height=height,
                    coord_space=chosen_space,
                )
                if print_mapping:
                    print(
                        f"Frame {frame_idx}: raw=({raw_x:.1f},{raw_y:.1f}) "
                        f"-> mapped=({mapped_x:.1f},{mapped_y:.1f})"
                    )
                cv2.circle(frame, (x_i, y_i), dot_radius, dot_color_bgr, -1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    print(f"Saved gaze overlay: {output_video_path}")
    print(f"  Frames written: {frame_idx}")


def parse_movies_csv(text: str) -> list[str]:
    movies = [item.strip() for item in text.split(",") if item.strip()]
    if not movies:
        raise ValueError("At least one movie must be provided in --movies.")
    return movies


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overlay eyetracking gaze points on movie files and save MP4 outputs."
    )

    parser.add_argument("experiment", choices=["psychedelic", "report-attn"])
    parser.add_argument("subject", help="Subject ID, e.g. BCSP1065")
    parser.add_argument(
        "session",
        help=(
            "Session key. For psychedelic: V4/V8/etc. "
            "For report-attn: condition key (match/rest/impmatch/ecprod)."
        ),
    )

    parser.add_argument(
        "--movies",
        default=",".join(DEFAULT_MOVIES),
        help="Comma-separated movie keys, e.g. test2_run-01,train01,train02",
    )
    parser.add_argument(
        "--runtype",
        default=None,
        help="Optional substring filter for eyetracking task keys, e.g. train",
    )

    parser.add_argument(
        "--eyetracking-root",
        type=Path,
        default=Path("/hehe/data/preproc/01_mri-bids/GOOD_DATA"),
        help="Root folder containing psychedelic and report-attn-eyetracking directories",
    )
    parser.add_argument(
        "--movie-dir",
        type=Path,
        default=Path(
            "/projects/semantic-psychedelics/00_stimuli/"
            "psychedelicvm_exp7-greyscreen_report-attn/stimuli"
        ),
        help="Directory containing movie mp4 files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_EYETRACKING_OVERLAY_SAVEPATH),
        help="Output root directory for gaze overlay videos",
    )

    parser.add_argument("--dot-radius", type=int, default=10, help="Radius of gaze point")
    parser.add_argument(
        "--dot-color",
        default="0,255,0",
        help="Gaze point color in B,G,R format, e.g. 0,255,0",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output videos")
    parser.add_argument(
        "--alignment",
        choices=["frame-mean", "lanczos"],
        default="frame-mean",
        help="Frame alignment method (default matches original script behavior).",
    )
    parser.add_argument(
        "--coord-space",
        choices=["auto", "screen1024x768", "movie"],
        default="auto",
        help="Input coordinate space. Use auto to infer per run.",
    )
    parser.add_argument(
        "--eyetracking-sampling-rate",
        type=float,
        default=250.0,
        help="Eyetracking sampling rate in Hz used for frame-mean alignment.",
    )
    parser.add_argument(
        "--movie-fps",
        type=float,
        default=15.0,
        help="Movie frame rate used for frame-mean alignment.",
    )
    parser.add_argument(
        "--quiet-mapping",
        action="store_true",
        help="Disable per-frame raw->mapped coordinate printouts.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    movies = parse_movies_csv(args.movies)
    print(f"Movies to process: {movies}")
    dot_color_bgr = parse_bgr_color(args.dot_color)

    if args.experiment == "psychedelic":
        raw_gaze_data = load_psychedelic_eyetracking(
            subject=args.subject,
            session=args.session,
            eyetracking_root=args.eyetracking_root,
            runtype=args.runtype,
            makefivetwelve=False,
        )
    else:
        raw_gaze_data = load_reportattn_eyetracking(
            subject=args.subject,
            session=args.session,
            runtype=args.runtype,
            makefivetwelve=False,
        )

    if not raw_gaze_data:
        raise SystemExit("No eyetracking files found for the provided arguments.")

    print(f"Raw gaze keys from loader: {list(raw_gaze_data.keys())}")

    gaze_data = dedupe_and_normalize_gaze_keys(raw_gaze_data)
    print(f"Normalized gaze keys: {list(gaze_data.keys())}")

    output_dir = args.output_dir / args.experiment / f"sub-{args.subject}" / f"ses-{args.session}"
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for movie in movies:
        if movie not in gaze_data:
            print(f"Skipping {movie}: missing eyetracking data after normalization.")
            skipped += 1
            continue

        video_path = resolve_video_path(movie, args.movie_dir)
        if video_path is None:
            print(f"Skipping {movie}: no matching video found in {args.movie_dir}.")
            skipped += 1
            continue

        out_path = output_dir / f"{movie}_eyetrackoverlay_20260316.mp4"
        if out_path.exists() and not args.overwrite:
            print(f"Skipping {movie}: output exists (use --overwrite to replace): {out_path}")
            skipped += 1
            continue

        print(f"Processing {movie}")
        print(f"  video: {video_path}")
        print(f"  output: {out_path}")

        overlay_gaze_on_video(
            input_video_path=video_path,
            output_video_path=out_path,
            gaze_df=gaze_data[movie],
            dot_radius=args.dot_radius,
            dot_color_bgr=dot_color_bgr,
            alignment=args.alignment,
            coord_space=args.coord_space,
            eyetracking_sampling_rate=args.eyetracking_sampling_rate,
            movie_fps=args.movie_fps,
            print_mapping=not args.quiet_mapping,
        )
        processed += 1

    print("Done.")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()

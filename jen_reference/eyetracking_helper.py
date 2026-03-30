"""Eyetracking utility helpers used by notebook QC workflows.
"""

from __future__ import annotations

import re
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONDITION_SUFFIX_RE = re.compile(r"(impmatch|match|ecprod|rest)$")
DEFAULT_REPORTATTN_ROOT = Path("/hehe/data/preproc/01_mri-bids/GOOD_DATA/report-attn-eyetracking")


def convert_to_movie_pixels(df: pd.DataFrame, clip: bool = True) -> pd.DataFrame:
	"""Map eyetracking coordinates from 1024x768 display space to 512x512 movie space."""
	x_offset = (1024 - 768) / 2
	y_offset = 0
	scale = 512 / 768

	out = df.copy()
	out["x"] = (out["x"] - x_offset) * scale
	out["y"] = (out["y"] - y_offset) * scale

	if clip:
		out["x"] = out["x"].clip(0, 511)
		out["y"] = out["y"].clip(0, 511)
	return out


def extract_condition(task_key: str) -> str | None:
	"""Extract report-attn condition token from a task key."""
	m = re.search(r"test\d+([a-z]+)\d+", task_key)
	if m:
		return m.group(1)

	m = re.search(r"train\d+([a-z]+)", task_key)
	if m:
		return m.group(1)

	return None


def parse_task_key_from_filename(file_path: str) -> str | None:
	"""Extract BIDS task key from corrected eyetracking filename."""
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
	"""Normalize heterogeneous task keys into canonical movie labels."""
	# test2match03_run-01 -> test2_run-05
	m = re.match(r"(test\d+)(impmatch|match|ecprod|rest)(\d+)_run-(\d+)", task_key)
	if m:
		session_num = int(m.group(3))
		run_num = int(m.group(4))
		global_run = (session_num - 1) * 2 + run_num
		return f"{m.group(1)}_run-{global_run:02d}"

	# test2match_run-03 -> test2_run-03
	m = re.match(r"(test\d+)(impmatch|match|ecprod|rest)_run-(\d+)", task_key)
	if m:
		return f"{m.group(1)}_run-{int(m.group(3)):02d}"

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


def dedupe_and_normalize_gaze_keys(gaze_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
	"""Normalize keys and keep the longest timeseries when collisions occur."""
	out: dict[str, pd.DataFrame] = {}

	for raw_key, df in gaze_data.items():
		norm_key = normalize_movie_key(raw_key)
		if norm_key in out:
			if len(df) > len(out[norm_key]):
				out[norm_key] = df
		else:
			out[norm_key] = df

	return out


def _validate_xy_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	if "x" in df.columns and "y" in df.columns:
		out = df[["x", "y"]].copy()
	else:
		out = df.rename(columns={0: "x", 1: "y"})[["x", "y"]].copy()

	out["x"] = pd.to_numeric(out["x"], errors="coerce").interpolate(method="linear", limit_direction="both")
	out["y"] = pd.to_numeric(out["y"], errors="coerce").interpolate(method="linear", limit_direction="both")
	return out


def load_reportattn_eyetracking(
	subject: str,
	condition: str | None,
	runtype: str | None = None,
	makefivetwelve: bool = True,
	eyetracking_root: str | Path = DEFAULT_REPORTATTN_ROOT,
	normalize_keys: bool = False,
) -> dict[str, pd.DataFrame]:
	"""Load report-attn eyetracking runs for a subject.

	Args:
		subject: Subject code, e.g. "KLR".
		condition: Task condition token, e.g. "match". Use None for all conditions.
		runtype: Optional substring filter (e.g. "train" or "test").
		makefivetwelve: Convert from 1024x768 to 512x512 movie coordinates.
		eyetracking_root: Root containing sub-<SUBJECT>/ses-*/... corrected .npy files.
		normalize_keys: Convert key names to canonical movie labels.
	"""
	root = Path(eyetracking_root)
	base = root / f"sub-{subject}"
	if not base.exists():
		raise FileNotFoundError(f"Eyetracking directory not found: {base}")

	data: dict[str, pd.DataFrame] = {}
	session_dirs = sorted(glob(str(base / "ses-*/")))

	for session_dir in session_dirs:
		files = sorted(glob(str(Path(session_dir) / "*corrected.npy")))
		for file in files:
			key = parse_task_key_from_filename(file)
			if key is None:
				continue

			if condition is not None:
				extracted = extract_condition(key)
				if extracted != condition:
					continue

			if runtype is not None and runtype not in key:
				continue
            
			try:
				df = _validate_xy_dataframe(pd.DataFrame(np.load(file, allow_pickle=True)))
				
			except Exception as e:
				print(f"Warning: failed to load {file}: {e}")
				continue
	
			if makefivetwelve:
				df = convert_to_movie_pixels(df, clip=True)

			if key in data:
				if len(df) > len(data[key]):
					data[key] = df
			else:
				data[key] = df

	if normalize_keys:
		data = dedupe_and_normalize_gaze_keys(data)

	if len(data) == 0:
		condition_msg = "all conditions" if condition is None else f"condition '{condition}'"
		raise RuntimeError(
			f"No eyetracking runs found for sub-{subject} ({condition_msg}) in {base}."
		)

	return data


def _resolve_movie_key(
	eyetracking_data: dict[str, pd.DataFrame],
	test_movie_key: str | None,
) -> str:
	if test_movie_key is not None:
		if test_movie_key not in eyetracking_data:
			available = ", ".join(sorted(eyetracking_data.keys()))
			raise KeyError(f"{test_movie_key} not found. Available keys: {available}")
		return test_movie_key

	test_keys = sorted([k for k in eyetracking_data.keys() if "test" in k])
	if len(test_keys) > 0:
		return test_keys[0]

	return sorted(eyetracking_data.keys())[0]


def plot_xy_timeseries(
	eyetracking_data: dict[str, pd.DataFrame],
	subject: str,
	condition: str,
	test_movie_key: str | None = None,
	skip_samples: int = 2500,
) -> None:
	"""Plot x/y gaze traces over time for one movie."""
	movie_key = _resolve_movie_key(eyetracking_data, test_movie_key)
	df = eyetracking_data[movie_key]

	x = df["x"].to_numpy()
	y = df["y"].to_numpy()
	t = np.arange(len(df))

	fig = plt.figure(figsize=(12, 6))

	plt.plot(t, x, linewidth=1, label="X")
	plt.axhline(256, linestyle="--", linewidth=1)
	plt.ylabel("X (pixels)")
	plt.ylim(0, 512)
	plt.title(f"sub-{subject} report-{condition} {movie_key}: X over time")

	plt.plot(t, y, linewidth=1, label="Y")
	plt.axhline(256, linestyle="--", linewidth=1)
	plt.ylabel("Y (pixels)")
	plt.xlabel("Sample index")
	plt.ylim(0, 512)

	plt.legend()
	plt.tight_layout()
	plt.show()


def plot_scatter_xy(
	eyetracking_data: dict[str, pd.DataFrame],
	subject: str,
	condition: str,
	test_movie_key: str | None = None,
	skip_samples: int = 2500,
	alpha: float = 0.2,
	s: float = 3,
) -> None:
	"""Plot gaze scatter in movie pixel space for one movie."""
	movie_key = _resolve_movie_key(eyetracking_data, test_movie_key)
	df = eyetracking_data[movie_key]

	start = min(max(skip_samples, 0), len(df))
	plot_df = df.iloc[start:]

	plt.figure(figsize=(6, 6))
	plt.scatter(plot_df["x"], plot_df["y"], alpha=alpha, s=s)
	plt.scatter([256], [256], marker="x", s=80)
	plt.xlim(0, 512)
	plt.ylim(512, 0)
	plt.xlabel("X (pixels)")
	plt.ylabel("Y (pixels)")
	plt.title(f"sub-{subject} report-{condition} {movie_key}: gaze scatter")
	plt.tight_layout()
	plt.show()


def plot_scatter_and_timeseries(
	eyetracking_data: dict[str, pd.DataFrame],
	subject: str,
	condition: str,
	test_movie_key: str | None = None,
	skip_samples: int = 2500,
) -> None:
	"""Convenience wrapper to display both QC plots for one movie."""
	plot_xy_timeseries(
		eyetracking_data,
		subject,
		condition,
		test_movie_key=test_movie_key,
		skip_samples=skip_samples,
	)
	plot_scatter_xy(
		eyetracking_data,
		subject,
		condition,
		test_movie_key=test_movie_key,
		skip_samples=skip_samples,
	)


__all__ = [
	"convert_to_movie_pixels",
	"dedupe_and_normalize_gaze_keys",
	"extract_condition",
	"load_reportattn_eyetracking",
	"normalize_movie_key",
	"parse_task_key_from_filename",
	"plot_scatter_and_timeseries",
	"plot_scatter_xy",
	"plot_xy_timeseries",
]

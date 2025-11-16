from __future__ import annotations

import argparse
import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from ultralytics import YOLO

try:
	from tqdm import tqdm
	_HAS_TQDM = True
except ModuleNotFoundError:
	_HAS_TQDM = False

	def tqdm(iterable: Iterable, **_: object) -> Iterable:
		return iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT_DIR / "XJTLU_2023_Detection_ALL"
CLASSES_FILE = DATA_ROOT / "classes.txt"
PREPARED_ROOT = ROOT_DIR / "prepared_dataset"
DEFAULT_SPLITS = (0.8, 0.1, 0.1)


@dataclass(frozen=True)
class DatasetConfig:
	train_dir: Path
	val_dir: Path
	test_dir: Path
	yaml_path: Path
	class_names: Sequence[str]


def read_classes(path: Path) -> List[str]:
	if not path.is_file():
		raise FileNotFoundError(f"classes file not found: {path}")
	classes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
	if not classes:
		raise ValueError(f"no class names found in {path}")
	return classes


def collect_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
	if not images_dir.exists() or not labels_dir.exists():
		raise FileNotFoundError("images or labels directory is missing")
	image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
	image_map = {p.stem: p for p in images_dir.rglob("*") if p.suffix.lower() in image_exts}
	label_map = {p.stem: p for p in labels_dir.rglob("*.txt")}
	common = sorted(image_map.keys() & label_map.keys())
	missing_images = sorted(label_map.keys() - image_map.keys())
	missing_labels = sorted(image_map.keys() - label_map.keys())
	if missing_images:
		sample = ", ".join(missing_images[:5])
		logging.warning("%d labels do not have images, e.g. %s", len(missing_images), sample)
	if missing_labels:
		sample = ", ".join(missing_labels[:5])
		logging.warning("%d images do not have labels, e.g. %s", len(missing_labels), sample)
	pairs = [(image_map[name], label_map[name]) for name in common]
	if not pairs:
		raise ValueError("no matching image/label pairs found")
	return pairs


def split_indices(count: int, splits: Sequence[float]) -> Tuple[range, range, range]:
	if len(splits) != 3:
		raise ValueError("splits must contain three ratios: train, val, test")
	if not 0.99 <= sum(splits) <= 1.01:
		raise ValueError("splits must sum to 1.0")
	train_count = int(count * splits[0])
	val_count = int(count * splits[1])
	train = range(0, train_count)
	val = range(train_count, train_count + val_count)
	test = range(train_count + val_count, count)
	return train, val, test


def ensure_dir_structure(root: Path) -> DatasetConfig:
	images_root = root / "images"
	labels_root = root / "labels"
	for sub in ("train", "val", "test"):
		(images_root / sub).mkdir(parents=True, exist_ok=True)
		(labels_root / sub).mkdir(parents=True, exist_ok=True)
	yaml_path = root / "rm_dataset.yaml"
	return DatasetConfig(
		train_dir=images_root / "train",
		val_dir=images_root / "val",
		test_dir=images_root / "test",
		yaml_path=yaml_path,
		class_names=read_classes(CLASSES_FILE),
	)


def convert_label(src: Path, dst: Path) -> None:
	lines: List[str] = []
	for raw_line in src.read_text(encoding="utf-8").splitlines():
		parts = raw_line.strip().split()
		if not parts:
			continue
		if len(parts) < 5:
			raise ValueError(f"label {src} has fewer than five values: '{raw_line}'")
		lines.append(" ".join(parts[:5]))  # trim to class + bbox for detection
	dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def copy_pair(image: Path, label: Path, dst_images: Path, dst_labels: Path) -> None:
	dst_img = dst_images / image.name
	dst_lbl = dst_labels / label.name
	shutil.copy2(image, dst_img)
	convert_label(label, dst_lbl)


def prepare_dataset(force: bool, splits: Sequence[float], seed: int) -> DatasetConfig:
	config = ensure_dir_structure(PREPARED_ROOT)
	dataset_ready = config.yaml_path.is_file() and any(config.train_dir.glob("*.jpg"))
	if dataset_ready and not force:
		logging.info("using existing prepared dataset at %s", PREPARED_ROOT)
		write_yaml(config)
		return config

	for images_sub in (config.train_dir, config.val_dir, config.test_dir):
		for file in images_sub.glob("*"):
			if file.is_file():
				file.unlink()
	for labels_sub in (PREPARED_ROOT / "labels" / split for split in ("train", "val", "test")):
		for file in labels_sub.glob("*"):
			if file.is_file():
				file.unlink()

	pairs = collect_pairs(DATA_ROOT / "images", DATA_ROOT / "labels")
	random.Random(seed).shuffle(pairs)
	train_idx, val_idx, test_idx = split_indices(len(pairs), splits)
	if not _HAS_TQDM:
		logging.info("Install 'tqdm' for progress bars during dataset preparation")

	split_info = {
		"train": (train_idx, config.train_dir, PREPARED_ROOT / "labels" / "train"),
		"val": (val_idx, config.val_dir, PREPARED_ROOT / "labels" / "val"),
		"test": (test_idx, config.test_dir, PREPARED_ROOT / "labels" / "test"),
	}

	total = len(pairs)
	logging.info(
		"Preparing dataset with %d samples (train=%d, val=%d, test=%d)",
		total,
		len(train_idx),
		len(val_idx),
		len(test_idx),
	)

	for split_name, (index_range, img_dst, lbl_dst) in split_info.items():
		count = len(index_range)
		logging.info("copying %d samples to %s", count, split_name)
		for idx in tqdm(index_range, desc=f"{split_name} split", leave=False):
			image_path, label_path = pairs[idx]
			copy_pair(image_path, label_path, img_dst, lbl_dst)

	write_yaml(config)
	return config


def write_yaml(config: DatasetConfig) -> None:
	yaml_lines = [
		f"path: {PREPARED_ROOT.as_posix()}",
		"train: images/train",
		"val: images/val",
		"test: images/test",
		f"nc: {len(config.class_names)}",
		"names:",
	]
	for idx, name in enumerate(config.class_names):
		yaml_lines.append(f"  {idx}: {name}")
	config.yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


def train_model(
	config: DatasetConfig,
	epochs: int,
	imgsz: int,
	batch: int,
	weights: str,
) -> Path:
	model = YOLO(weights)
	logging.info("Starting training for %d epochs (batch=%d, imgsz=%d)", epochs, batch, imgsz)
	results = model.train(
		data=str(config.yaml_path),
		epochs=epochs,
		imgsz=imgsz,
		batch=batch,
		project=str(ROOT_DIR / "runs"),
		name="rm_yolov8n",
		verbose=True,
	)
	save_dir = Path(results.save_dir)
	best_weights = save_dir / "weights" / "best.pt"
	if not best_weights.exists():
		raise FileNotFoundError(f"best weights not found at {best_weights}")
	metrics_summary = {}
	metrics_obj = getattr(results, "metrics", None)
	if metrics_obj is not None:
		metrics_summary = getattr(metrics_obj, "results_dict", None) or getattr(metrics_obj, "__dict__", {})
	if not metrics_summary and hasattr(results, "results_dict"):
		metrics_summary = results.results_dict  # type: ignore[attr-defined]
	if metrics_summary:
		picked = []
		for key in ("loss", "box", "cls", "dfl", "fitness", "mAP50", "mAP50-95"):
			value = metrics_summary.get(key)
			if isinstance(value, (int, float)):
				picked.append(f"{key}={value:.4f}")
		if picked:
			logging.info("Final metrics: %s", ", ".join(picked))
	logging.info("training finished, best weights saved to %s", best_weights)
	return best_weights


def export_to_onnx(weights_path: Path, opset: int, dynamic: bool, simplify: bool) -> Path:
	logging.info(
		"exporting %s to ONNX (opset=%d, dynamic=%s, simplify=%s)",
		weights_path,
		opset,
		dynamic,
		simplify,
	)
	model = YOLO(str(weights_path))
	try:
		exported_path = model.export(
			format="onnx",
			opset=opset,
			dynamic=dynamic,
			simplify=simplify,
		)
	except ModuleNotFoundError as exc:
		logging.error("missing dependency for ONNX export: %s", exc)
		raise
	onnx_path = Path(exported_path)
	if not onnx_path.exists():
		raise FileNotFoundError("ONNX export did not produce a file")
	logging.info("ONNX model saved to %s", onnx_path)
	return onnx_path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train YOLOv8n on the RM dataset")
	parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
	parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
	parser.add_argument("--batch", type=int, default=16, help="Batch size")
	parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Weights to start from")
	parser.add_argument("--val-split", type=float, default=DEFAULT_SPLITS[1], help="Validation split ratio")
	parser.add_argument("--test-split", type=float, default=DEFAULT_SPLITS[2], help="Test split ratio")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
	parser.add_argument("--force-rebuild", action="store_true", help="Rebuild the prepared dataset")
	parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
	parser.add_argument("--dynamic", action="store_true", help="Enable dynamic axes during ONNX export")
	parser.add_argument("--no-dynamic", dest="dynamic", action="store_false")
	parser.add_argument("--simplify", dest="simplify", action="store_true", help="Simplify ONNX graph during export")
	parser.add_argument("--no-simplify", dest="simplify", action="store_false")
	parser.add_argument("--export", dest="export", action="store_true", help="Export ONNX after training")
	parser.add_argument("--no-export", dest="export", action="store_false", help="Skip ONNX export")
	parser.set_defaults(dynamic=False, export=True, simplify=False)
	return parser.parse_args()


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
	args = parse_args()
	train_ratio = 1.0 - args.val_split - args.test_split
	if train_ratio <= 0:
		raise ValueError("validation and test splits leave no data for training")
	splits = (train_ratio, args.val_split, args.test_split)
	config = prepare_dataset(force=args.force_rebuild, splits=splits, seed=args.seed)
	best_weights = train_model(
		config,
		epochs=args.epochs,
		imgsz=args.imgsz,
		batch=args.batch,
		weights=args.weights,
	)
	if args.export:
		export_to_onnx(best_weights, opset=args.opset, dynamic=args.dynamic, simplify=args.simplify)


if __name__ == "__main__":
	main()
#!/usr/bin/env python3
"""Read-only data inspection and five plots for YOLO datasets."""
import argparse
import hashlib
import json
import math
import os
import textwrap
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.colors import LogNorm
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
COLORS = ("#007a78", "#7754a0", "#4079ad", "#b46b36", "#52934c")
BIN_CMAP = plt.get_cmap("viridis").copy()
BIN_CMAP.set_bad("#eef2f5")
OVERLAY_CMAP = plt.get_cmap("magma").copy()
OVERLAY_CMAP.set_bad("#eef2f5")
PLOT_ZH = False


@dataclass
class Group:
    name: str
    path: Path
    images: list = field(default_factory=list)
    pairs: dict = field(default_factory=dict)
    labels: list = field(default_factory=list)
    orphans: list = field(default_factory=list)
    label_candidates: dict = field(default_factory=dict)
    ambiguous_stems: list = field(default_factory=list)
    mask_directory_present: bool = False
    masks: list = field(default_factory=list)
    mask_candidates: dict = field(default_factory=dict)
    orphan_masks: list = field(default_factory=list)


@dataclass
class Label:
    raw: int = 0
    classes: Counter = field(default_factory=Counter)
    invalid: Counter = field(default_factory=Counter)
    rows: list = field(default_factory=list)
    read_error: str = ""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--groups", nargs="+", metavar="GROUP",
                        help="Group directories relative to root; NAME=PATH sets a custom name.")
    parser.add_argument("--classes", type=Path, help="One class per line (ID = zero-based line index). "
                        "Default: root/classes.txt if present, otherwise observed IDs.")
    parser.add_argument("--task", choices=("auto", "segmentation", "detection"), default="auto")
    parser.add_argument("--name", help="Dataset name for figure titles.")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--hash-cross-groups", action="store_true",
                        help="Compare exact image bytes across groups using SHA-256.")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("--root must be an existing directory")
    if args.workers < 1:
        parser.error("--workers must be at least one")
    return args


def files_with_suffixes(path, suffixes):
    return sorted(Path(parent) / name for parent, _, names in os.walk(path)
                  for name in names if Path(name).suffix.lower() in suffixes)


def corresponding_root(path, target):
    # For layouts like images/train/ and labels/train/.
    parts = path.parts
    indices = [i for i, part in enumerate(parts) if part == "images"]
    if indices:
        i = indices[-1]
        return Path(*parts[:i], target, *parts[i + 1:])
    return None


def discover(group, dataset_root):
    labels_from_layout = set()
    masks_from_layout = set()
    roots = []
    relative_parts = group.path.relative_to(dataset_root).parts if dataset_root in group.path.parents else ()
    direct_image_tree = ("images" in relative_parts or group.path.name == "images"
                         or group.path.parent.name == "images")
    matching_root = corresponding_root(group.path, "labels") if direct_image_tree else None
    if matching_root is not None:
        roots.append((group.path, matching_root))
    else:
        for parent, directories, _ in os.walk(group.path):
            directory = Path(parent)
            if directory.name == "images":
                roots.append((directory, directory.parent / "labels"))
                directories[:] = []
            elif directory.name == "labels":
                labels_from_layout.update(files_with_suffixes(directory, {".txt"}))
                directories[:] = []
            elif directory.name.lower() in ("mask", "masks"):
                group.mask_directory_present = True
                masks_from_layout.update(files_with_suffixes(directory, IMAGE_SUFFIXES))
                directories[:] = []
    for images_root, labels_root in roots:
        labels = files_with_suffixes(labels_root, {".txt"})
        labels_from_layout.update(labels)
        by_relative_path = defaultdict(list)
        for path in labels:
            by_relative_path[path.relative_to(labels_root).with_suffix(".txt")].append(path)
        masks_by_stem = defaultdict(list)
        for directory_name in ("Mask", "mask", "masks", "Masks"):
            mask_root = corresponding_root(images_root, directory_name)
            if mask_root is None or not mask_root.is_dir():
                continue
            group.mask_directory_present = True
            for path in files_with_suffixes(mask_root, IMAGE_SUFFIXES):
                masks_from_layout.add(path)
                masks_by_stem[path.relative_to(mask_root).with_suffix("")].append(path)
        stems = defaultdict(list)
        for image in files_with_suffixes(images_root, IMAGE_SUFFIXES):
            group.images.append(image)
            relative_stem = image.relative_to(images_root).with_suffix("")
            label_key = image.relative_to(images_root).with_suffix(".txt")
            stems[label_key].append(image)
            alternatives = by_relative_path[label_key]
            if alternatives:
                group.label_candidates[image] = alternatives
                group.pairs[image] = alternatives[0]
            mask_candidates = set(masks_by_stem[relative_stem])
            mask_candidates.update(masks_by_stem[relative_stem.with_name(relative_stem.name + "_mask")])
            if mask_candidates:
                group.mask_candidates[image] = sorted(mask_candidates)
        for key, images in stems.items():
            if len(images) > 1:
                group.ambiguous_stems.append({
                    "expected_label": str(labels_root / key),
                    "image_paths": [str(image) for image in images],
                    "label_candidates": [str(p) for p in by_relative_path[key]],
                })
    group.images.sort()
    group.labels = sorted(labels_from_layout)
    group.orphans = sorted(labels_from_layout - {p for paths in group.label_candidates.values() for p in paths})
    group.masks = sorted(masks_from_layout)
    group.orphan_masks = sorted(masks_from_layout - {p for paths in group.mask_candidates.values() for p in paths})
    return group


def find_groups(root, specs):
    if specs is None:
        specs = [name for name in ("train", "val", "test")
                 if (root / name).is_dir() or (root / "images" / name).is_dir()]
        if not specs:
            specs = ["all=."]
    groups = []
    for spec in specs:
        name, subpath = spec.split("=", 1) if "=" in spec else (spec, spec)
        path = root / subpath
        if not path.is_dir() and "=" not in spec:
            path = root / "images" / subpath
        if not path.is_dir():
            raise ValueError(f"Group directory does not exist: {path}")
        if not name or any(group.name == name for group in groups):
            raise ValueError(f"Invalid or duplicate group name: {name}")
        groups.append(discover(Group(name, path.resolve()), root))
    return groups


def load_classes(args):
    if args.classes is not None:
        path = args.classes
        if not path.is_file():
            path = args.root / path
        if not path.is_file():
            raise ValueError(f"Class file does not exist: {path}")
    else:
        path = args.root / "classes.txt"
        if not path.is_file():
            return None, None
    names = path.read_text(encoding="utf-8-sig").splitlines()
    if any(not name.strip() for name in names) or not names:
        raise ValueError(f"Class file must have one nonempty name per line: {path}")
    return {i: name.strip() for i, name in enumerate(names)}, str(path.resolve())


def read_rows(path):
    return [row.split() for row in path.read_bytes().splitlines() if row.strip()]


def candidate_formats(path):
    counts = Counter()
    try:
        for row in read_rows(path):
            n = len(row)
            if n == 5:
                counts["detection"] += 1
            elif n >= 7 and n % 2 == 1:
                counts["segmentation"] += 1
    except OSError:
        pass
    return counts


def infer_task(groups, pool):
    # Prefer labels associated with actual images. Fall back to unpaired labels for empty data.
    labels = set(label for group in groups for label in group.pairs.values())
    if not labels:
        labels = set(label for group in groups for label in group.labels)
    counts = Counter()
    for result in pool.map(candidate_formats, sorted(labels)):
        counts.update(result)
    task = ("detection" if counts["detection"] > counts["segmentation"] else "segmentation")
    return task, dict(counts)


def parse_label(path, task, classes):
    data = Label()
    try:
        rows = read_rows(path)
    except OSError as exc:
        data.read_error = str(exc)
        return data
    for fields in rows:
        data.raw += 1
        try:
            class_id = int(fields[0])
        except ValueError:
            data.invalid["parse_class"] += 1
            continue
        data.classes[class_id] += 1
        if class_id < 0 or (classes is not None and class_id not in classes):
            data.invalid["unknown_class"] += 1
            continue
        values = fields[1:]
        if (task == "detection" and len(values) != 4
                or task == "segmentation" and (len(values) < 6 or len(values) % 2)):
            data.invalid["invalid_coordinate_count"] += 1
            continue
        try:
            values = np.array(values, dtype=np.float64)
        except ValueError:
            data.invalid["parse_coordinate"] += 1
            continue
        if not np.all(np.isfinite(values) & (values >= 0) & (values <= 1)):
            data.invalid["out_of_range_or_nonfinite"] += 1
            continue
        if task == "detection":
            cx, cy, width, height = values
            left, right = cx - width / 2, cx + width / 2
            top, bottom = cy - height / 2, cy + height / 2
            if min(left, top) < -1e-9 or max(right, bottom) > 1 + 1e-9:
                data.invalid["bbox_outside_image"] += 1
                continue
            xy = np.clip(np.array([[left, top], [right, top], [right, bottom], [left, bottom]]), 0, 1)
            area = width * height
        else:
            xy = values.reshape(-1, 2)
            x, y = xy.T
            area = 0.5 * abs(x @ np.roll(y, -1) - y @ np.roll(x, -1))
        if area <= 1e-12 or np.any(np.ptp(xy, axis=0) <= 0):
            data.invalid["degenerate_geometry"] += 1
            continue
        box = np.stack([xy.min(axis=0), xy.max(axis=0)])
        data.rows.append((class_id, xy, box, float(area)))
    return data


def image_info(path):
    try:
        with Image.open(path) as image:
            return image.size, image.mode
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        return None, str(exc)


def dimensions_array(values):
    return np.array(values, dtype=np.float64).reshape(-1, 2)


def scan_group(group, task, classes, pool):
    labels = dict(zip(group.labels, pool.map(lambda p: parse_label(p, task, classes), group.labels)))
    status = Counter()
    raw_classes, valid_classes, class_images, invalid = Counter(), Counter(), Counter(), Counter()
    all_label_classes, all_label_invalid = Counter(), Counter()
    orphan_classes, orphan_invalid = Counter(), Counter()
    all_raw = orphan_raw = 0
    label_errors = {}
    for path, label in labels.items():
        all_raw += label.raw
        all_label_classes.update(label.classes)
        all_label_invalid.update(label.invalid)
        if label.read_error:
            label_errors[str(path)] = label.read_error
    for path in group.orphans:
        label = labels[path]
        orphan_raw += label.raw
        orphan_classes.update(label.classes)
        orphan_invalid.update(label.invalid)
    dimensions, geometries, centers, relative, pixels, areas = [], [], [], [], [], []
    modes = Counter()
    image_errors = {}
    for image, info in zip(group.images, pool.map(image_info, group.images)):
        size, mode = info
        if size is None:
            image_errors[str(image)] = mode
        else:
            dimensions.append(size)
            modes[mode] += 1
        label_path = group.pairs.get(image)
        if label_path is None:
            status["missing_label"] += 1
            continue
        label = labels[label_path]
        if label.read_error:
            status["unreadable_label"] += 1
            continue
        status["empty_label" if label.raw == 0 else "nonempty_label"] += 1
        status["raw_rows"] += label.raw
        raw_classes.update(label.classes)
        invalid.update(label.invalid)
        if size is None:
            invalid["unreadable_image"] += len(label.rows)
            continue
        if label.rows:
            status["images_with_valid_instances"] += 1
            class_images.update(set(row[0] for row in label.rows))
        for class_id, xy, box, area in label.rows:
            valid_classes[class_id] += 1
            geometries.append((size, xy, box))
            centers.append(box.mean(axis=0))
            wh = box[1] - box[0]
            relative.append(wh)
            pixels.append(wh * size)
            areas.append(area)
    assert status["raw_rows"] == len(geometries) + sum(invalid.values())
    print(f"{group.name}: {len(group.images):,} images, {len(geometries):,} valid instances, "
          f"{sum(invalid.values()):,} invalid paired rows, {len(group.orphans):,} orphan labels",
          flush=True)
    return {
        "group": group, "status": status, "raw_classes": raw_classes, "valid_classes": valid_classes,
        "class_images": class_images, "invalid": invalid, "res": dimensions_array(dimensions),
        "geometries": geometries, "centers": dimensions_array(centers),
        "rel_wh": dimensions_array(relative), "px_wh": dimensions_array(pixels),
        "areas": np.array(areas), "modes": modes, "image_errors": image_errors,
        "label_errors": label_errors, "all_raw": all_raw, "all_label_classes": all_label_classes,
        "all_label_invalid": all_label_invalid, "orphan_raw": orphan_raw,
        "orphan_classes": orphan_classes, "orphan_invalid": orphan_invalid,
    }


def scalar_summary(values):
    values = np.asarray(values)
    if not values.size:
        return None
    q = np.percentile(values, [0, 10, 50, 90, 99, 100])
    return dict(zip(("min", "p10", "median", "p90", "p99", "max"), (float(x) for x in q)))


def xy_summary(values):
    return {"width": scalar_summary(values[:, 0]), "height": scalar_summary(values[:, 1])}


def top_dimensions(values, count=10):
    counts = Counter(map(tuple, values.astype(int)))
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:count]


def group_summary(stat, classes):
    group = stat["group"]
    status = stat["status"]
    valid = len(stat["geometries"])
    missing_labels = sorted(str(image) for image in group.images if image not in group.pairs)
    centers = stat["centers"]
    records = [{"id": class_id, "name": name,
                "raw_paired_instances": stat["raw_classes"][class_id],
                "valid_instances": stat["valid_classes"][class_id],
                "images_with_valid_class": stat["class_images"][class_id],
                "all_label_file_instances": stat["all_label_classes"][class_id],
                "orphan_label_instances": stat["orphan_classes"][class_id]}
               for class_id, name in sorted(classes.items())]
    return {
        "name": group.name, "path": str(group.path),
        "image_files": len(group.images),
        "image_extensions": dict(Counter(image.suffix.lower() for image in group.images)),
        "image_modes": dict(stat["modes"]),
        "image_header_readable": len(stat["res"]), "image_header_errors": stat["image_errors"],
        "label_files": len(group.labels),
        "label_files_with_matched_images": len(set(group.pairs.values())),
        "label_files_with_image_candidates": len({p for paths in group.label_candidates.values() for p in paths}),
        "same_stem_image_groups": len(group.ambiguous_stems),
        "same_stem_image_details": group.ambiguous_stems,
        "ambiguous_label_candidate_images": [
            {"image": str(image), "labels": [str(p) for p in paths]}
            for image, paths in group.label_candidates.items() if len(paths) > 1
        ],
        "orphan_label_files": len(group.orphans), "orphan_label_paths": [str(p) for p in group.orphans],
        "mask_directory_present": group.mask_directory_present,
        "mask_files": len(group.masks),
        "mask_files_with_matched_images": len({p for paths in group.mask_candidates.values() for p in paths}),
        "images_with_mask": len(group.mask_candidates),
        "images_missing_mask": len(group.images) - len(group.mask_candidates) if group.mask_directory_present else None,
        "image_paths_missing_mask": [str(p) for p in group.images if p not in group.mask_candidates]
        if group.mask_directory_present else [],
        "orphan_mask_files": len(group.orphan_masks),
        "orphan_mask_paths": [str(p) for p in group.orphan_masks],
        "ambiguous_mask_candidate_images": [
            {"image": str(image), "masks": [str(p) for p in paths]}
            for image, paths in group.mask_candidates.items() if len(paths) > 1
        ],
        "image_label_status": {k: status[k] for k in ("missing_label", "unreadable_label", "empty_label",
                                                     "nonempty_label", "images_with_valid_instances")},
        "images_without_valid_instances": len(group.images) - status["images_with_valid_instances"],
        "image_paths_missing_label": missing_labels,
        "label_read_errors": stat["label_errors"],
        "raw_instance_rows_from_paired_images": status["raw_rows"], "valid_instances": valid,
        "invalid_paired_rows": sum(stat["invalid"].values()), "invalid_paired_rows_by_reason": dict(stat["invalid"]),
        "raw_paired_classes": dict(stat["raw_classes"]),
        "raw_instance_rows_all_label_files": stat["all_raw"],
        "raw_classes_all_label_files": dict(stat["all_label_classes"]),
        "invalid_rows_all_label_files_by_reason": dict(stat["all_label_invalid"]),
        "raw_instance_rows_orphan_labels": stat["orphan_raw"],
        "raw_classes_orphan_labels": dict(stat["orphan_classes"]),
        "invalid_rows_orphan_labels_by_reason": dict(stat["orphan_invalid"]),
        "classes": records,
        "resolution_px": xy_summary(stat["res"]),
        "distinct_resolutions": len(set(map(tuple, stat["res"]))),
        "top_resolutions": [{"width": int(w), "height": int(h), "images": n}
                            for (w, h), n in top_dimensions(stat["res"])],
        "target_bbox_relative": xy_summary(stat["rel_wh"]),
        "target_bbox_px": xy_summary(stat["px_wh"]),
        "target_area_fraction": scalar_summary(stat["areas"]),
        "center_in_middle_40_percent": float(np.mean(np.all((centers >= .3) & (centers <= .7), axis=1)))
        if len(centers) else None,
    }


def file_size(path):
    try:
        return path, path.stat().st_size
    except OSError:
        return path, None


def image_hash(path):
    try:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1 << 20), b""):
                digest.update(chunk)
        return path, digest.hexdigest()
    except OSError:
        return path, None


def cross_group_hashes(groups, pool):
    sizes = defaultdict(list)
    errors = []
    for i, group in enumerate(groups):
        for path, size in pool.map(file_size, group.images):
            if size is None:
                errors.append(str(path))
            else:
                sizes[size].append((i, path))
    candidates = [pair for files in sizes.values() if len({i for i, _ in files}) > 1 for pair in files]
    hashes = defaultdict(Counter)
    for (i, _), (path, digest) in zip(candidates, pool.map(image_hash, (p for _, p in candidates))):
        if digest is None:
            errors.append(str(path))
        else:
            hashes[digest][i] += 1
    results = []
    for i, j in combinations(range(len(groups)), 2):
        common = [counts for counts in hashes.values() if counts[i] and counts[j]]
        left, right = sum(row[i] for row in common), sum(row[j] for row in common)
        results.append({
            "groups": [groups[i].name, groups[j].name],
            "distinct_shared_sha256_contents": len(common),
            "files_with_shared_content": {groups[i].name: left, groups[j].name: right},
            "fraction_with_shared_content": {
                groups[i].name: left / len(groups[i].images) if groups[i].images else None,
                groups[j].name: right / len(groups[j].images) if groups[j].images else None,
            },
        })
    return {"method": "cross-group size prefilter and SHA-256 of original file bytes",
            "hash_read_errors": sorted(set(errors)), "pairwise": results}


def configure_plots():
    global PLOT_ZH
    cv2.setNumThreads(1)
    families = ["DejaVu Sans"]
    font_files = [Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
                  Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
    for path in font_files:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            families.insert(0, font_manager.FontProperties(fname=str(path)).get_name())
            PLOT_ZH = True
            break
    if not PLOT_ZH:
        available = {font.name for font in font_manager.fontManager.ttflist}
        for name in ("Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "SimHei", "WenQuanYi Micro Hei"):
            if name in available:
                families.insert(0, name)
                PLOT_ZH = True
                break
    plt.rcParams.update({
        "font.family": families, "axes.unicode_minus": False, "axes.labelcolor": "#24364b",
        "axes.titlesize": 13, "axes.titleweight": "bold", "text.color": "#15283c",
        "xtick.color": "#405468", "ytick.color": "#405468", "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#a4b2c0", "savefig.facecolor": "white",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def tr(chinese, english):
    return chinese if PLOT_ZH else english


def figure(rows, cols, width, height, title, subtitle):
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False,
                             constrained_layout=True)
    max_title_chars = int(width * (5 if PLOT_ZH else 8))
    max_subtitle_chars = int(width * (6 if PLOT_ZH else 11))
    title = "\n".join(textwrap.wrap(title, width=max_title_chars))
    subtitle = "\n".join(textwrap.wrap(subtitle, width=max_subtitle_chars))
    fig.suptitle(f"{title}\n{subtitle}", fontsize=14, fontweight="bold", linespacing=1.45)
    return fig, axes


def save(fig, out, filename):
    fig.savefig(out / filename, dpi=160, bbox_inches="tight", pad_inches=.2, facecolor="white")
    plt.close(fig)
    print(f"Wrote {filename}", flush=True)


def empty_panel(ax, message=None):
    message = message or tr("无有效实例", "No valid instances")
    ax.text(.5, .5, message, transform=ax.transAxes, va="center", ha="center",
            fontsize=11, color="#547086")


def annotate_bars(ax, bars):
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f"{int(width):,}", (width, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=9)


def chart_classes(data, classes, name, out):
    ids = sorted(classes, key=lambda class_id: (-data[0]["valid_classes"][class_id], class_id))
    cols = min(3, len(data))
    rows = math.ceil(len(data) / cols)
    fig, axes = figure(rows, cols, 6.5 * cols, max(4.5, len(ids) * .34 + 2.1) * rows,
                       f"{name} | " + tr("类别实例数", "Class instance counts"),
                       tr("汇总有效配图标注，类别顺序按第一组实例数排列。",
                          "Valid matched-image annotations, with class order based on the first group."))
    for i, stat in enumerate(data):
        ax = axes.flat[i]
        group = stat["group"].name
        values = [stat["valid_classes"][c] for c in ids]
        if ids:
            bars = ax.barh(np.arange(len(ids)), values, color=COLORS[i % len(COLORS)], height=.67)
            annotate_bars(ax, bars)
            ax.set(yticks=np.arange(len(ids)), yticklabels=[f"{classes[c]} [{c}]" for c in ids],
                   xlim=(0, max(1, max(values) * 1.25)), ylim=(-.65, len(ids) - .35))
            ax.invert_yaxis()
            ax.grid(axis="x", color="#e1e8ef", linewidth=.7)
        else:
            empty_panel(ax, tr("无类别 ID", "No class IDs"))
        ax.set_title(f"{group} | " + tr(f"有效 {sum(values):,}，异常 {sum(stat['invalid'].values()):,}",
                                        f"{sum(values):,} valid, {sum(stat['invalid'].values()):,} invalid"))
        ax.set_xlabel(tr("实例数量", "Instances"))
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    for ax in axes.flat[len(data):]:
        ax.set_visible(False)
    save(fig, out, "01_class_instance_counts.png")


def add_mask(hist, points, limit):
    n = hist.shape[0]
    xy = (points + limit) * ((n - 1) / (2 * limit))
    low = np.maximum(np.floor(xy.min(axis=0)).astype(int) - 1, 0)
    high = np.minimum(np.ceil(xy.max(axis=0)).astype(int) + 2, n)
    x0, y0 = low
    x1, y1 = high
    if x0 >= x1 or y0 >= y1:
        return
    tile = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    points_fp = np.rint((xy - low) * 16).astype(np.int32)
    cv2.fillPoly(tile, [points_fp], 1, shift=4)
    hist[y0:y1, x0:x1] += tile


def coverages(stat, pixel_limit):
    relative = np.zeros((385, 385), dtype=np.uint32)
    pixels = np.zeros_like(relative)
    count = len(stat["geometries"])
    for size, xy, box in stat["geometries"]:
        centered = xy - box.mean(axis=0)
        add_mask(relative, centered, .5)
        add_mask(pixels, centered * size, pixel_limit)
    return relative / max(count, 1), pixels / max(count, 1)


def density_norm(matrices):
    positive = [m[m > 0] for m in matrices if np.any(m > 0)]
    if not positive:
        return LogNorm(vmin=.001, vmax=1)
    low = min(p.min() for p in positive)
    high = max(p.max() for p in positive)
    if low == high:
        low /= 10
    return LogNorm(vmin=low, vmax=high)


def percentage_colorbar(fig, item, axes, label):
    colorbar = fig.colorbar(item, ax=axes, shrink=.82, pad=.035)
    colorbar.set_label(label)
    colorbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda p, _: f"{p * 100:.3g}%"))
    colorbar.ax.yaxis.set_minor_formatter(ticker.NullFormatter())


def chart_overlay(data, task, name, out):
    side_p99 = [float(np.percentile(np.max(stat["px_wh"], axis=1), 99))
                for stat in data if len(stat["px_wh"])]
    half_extent = max(side_p99, default=2) / 2
    step = 10 ** math.floor(math.log10(max(1, half_extent))) / 5
    limit = max(1, math.ceil(half_extent / step) * step)
    maps = [coverages(stat, limit) for stat in data]
    norm = density_norm([panel for pair in maps for panel in pair])
    kind = "Mask" if task == "segmentation" else "BBox"
    geometry = (tr("多边形掩膜", "Polygon mask") if task == "segmentation"
                else tr("矩形检测框（BBox 标注）", "Bounding rectangle (bbox annotations)"))
    subtitle = tr(f"将{geometry}的外接框中心对齐，颜色表示覆盖该位置的实例比例。",
                  f"{geometry} coverage with bounding-box centers aligned; color = fraction of instances.")
    fig, axes = figure(len(data), 2, 11.8, 4.7 * len(data) + .95,
                       f"{name} | {kind} " + tr("尺寸叠加", "size overlay"), subtitle)
    for i, stat in enumerate(data):
        for j, (extent, values) in enumerate(zip((.5, limit), maps[i])):
            ax = axes[i, j]
            item = ax.imshow(np.ma.masked_less_equal(values, 0), cmap=OVERLAY_CMAP,
                             norm=norm, extent=(-extent, extent, extent, -extent),
                             interpolation="nearest", origin="upper")
            for draw in (ax.axhline, ax.axvline):
                draw(0, color="#81acbb", alpha=.6, linewidth=.7)
            if j == 0:
                ax.set(xlabel=tr("相对中心 X / 图宽", "Centered X / image width"),
                       ylabel=tr("相对中心 Y / 图高", "Centered Y / image height"))
                ax.xaxis.set_major_formatter(ticker.PercentFormatter(1))
                ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
                title = tr("相对原图尺寸", "Relative image dimensions")
            else:
                ax.set(xlabel=tr("相对中心 X (px)", "Centered X (px)"),
                       ylabel=tr("相对中心 Y (px)", "Centered Y (px)"))
                ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
                outside = np.mean(np.max(stat["px_wh"], axis=1) > 2 * limit) if len(stat["px_wh"]) else 0
                title = tr(f"原图像素 | {outside:.2%} 超出视窗", f"Original pixels | {outside:.2%} exceed view")
            ax.set_title(f"{stat['group'].name} | {title}")
            if not stat["geometries"]:
                empty_panel(ax)
    percentage_colorbar(fig, item, list(axes.flat), tr("实例覆盖率", "Instance coverage"))
    filename = "02_mask_size_overlay.png" if task == "segmentation" else "02_bbox_size_overlay.png"
    save(fig, out, filename)


def density_grid(values, edges_x, edges_y):
    grid, _, _ = np.histogram2d(values[:, 0], values[:, 1], bins=(edges_x, edges_y))
    return grid.T / max(1, len(values))


def chart_centers(data, name, out):
    bins = np.linspace(0, 1, 61)
    maps = [density_grid(d["centers"], bins, bins) for d in data]
    norm = density_norm(maps)
    cols = min(3, len(data))
    fig, axes = figure(math.ceil(len(data) / cols), cols, 5.7 * cols + .7,
                       5.5 * math.ceil(len(data) / cols) + .8,
                       f"{name} | " + tr("目标中心位置分布", "Target center distribution"),
                       tr("外接框中心按原图宽高归一化，左上角为原点，颜色表示每格实例占比。",
                          "Bounding-box centers normalized by original image dimensions. Origin: top-left; color = share per bin."))
    for i, stat in enumerate(data):
        ax = axes.flat[i]
        item = ax.pcolormesh(bins, bins, np.ma.masked_less_equal(maps[i], 0),
                             cmap=BIN_CMAP, norm=norm, shading="flat")
        ax.set(xlim=(0, 1), ylim=(1, 0), xlabel=tr("中心 X / 图宽", "Center X / width"),
               ylabel=tr("中心 Y / 图高", "Center Y / height"),
               aspect="equal")
        ax.axhline(.5, color="#9bb7c1", linewidth=.7, linestyle="--")
        ax.axvline(.5, color="#9bb7c1", linewidth=.7, linestyle="--")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(ticker.PercentFormatter(1))
            axis.set_major_locator(ticker.MultipleLocator(.2))
        centers = stat["centers"]
        middle = np.mean(np.all((centers >= .3) & (centers <= .7), axis=1)) if len(centers) else None
        title = stat["group"].name
        if middle is not None:
            title += " | " + tr(f"{middle:.1%} 位于中央 40%×40%", f"{middle:.1%} in middle 40% x 40%")
        ax.set_title(title)
        if not len(centers):
            empty_panel(ax)
    for ax in axes.flat[len(data):]:
        ax.set_visible(False)
    percentage_colorbar(fig, item, list(axes.flat[:len(data)]), tr("每格有效实例占比", "Share of valid instances per bin"))
    save(fig, out, "03_object_center_distribution.png")


def log_edges(values, relative=False):
    if not len(values):
        return np.geomspace(.001, 1, 57) if relative else np.geomspace(1, 1000, 57)
    low = 10 ** (math.floor(math.log10(float(np.min(values))) * 2) / 2)
    high = 1 if relative else 10 ** (math.ceil(math.log10(float(np.max(values))) * 2) / 2)
    if not low < high:
        low /= math.sqrt(10)
    # Widen by one representable value to include the exact boundary.
    low = np.nextafter(low, 0)
    high = np.nextafter(high, np.inf)
    return np.geomspace(low, high, 57)


def format_log_axis(axis, relative):
    formatter = (lambda x, _: f"{x * 100:.4g}%") if relative else (lambda x, _: f"{x:,.6g}")
    axis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 3), numticks=9))
    axis.set_major_formatter(ticker.FuncFormatter(formatter))
    axis.set_minor_locator(ticker.NullLocator())


def size_heatmap(ax, values, edges, matrix, norm, relative, xlabel, ylabel):
    x, y = edges
    item = ax.pcolormesh(x, y, np.ma.masked_less_equal(matrix, 0), cmap=BIN_CMAP, norm=norm)
    ax.set(xscale="log", yscale="log", xlim=(x[0], x[-1]), ylim=(y[0], y[-1]),
           xlabel=xlabel, ylabel=ylabel)
    format_log_axis(ax.xaxis, relative)
    format_log_axis(ax.yaxis, relative)
    if len(values):
        ax.scatter(*np.median(values, axis=0), s=30, color="white", edgecolor="#16294a", linewidth=.8)
    return item


def chart_wh(data, name, out):
    keys = ("rel_wh", "px_wh")
    bounds, maps = [], []
    for key, relative in zip(keys, (True, False)):
        values = np.concatenate([d[key] for d in data], axis=0)
        edges = [log_edges(values[:, k], relative) for k in range(2)]
        bounds.append(edges)
        maps.append([density_grid(d[key], *edges) for d in data])
    norm = density_norm([matrix for family in maps for matrix in family])
    fig, axes = figure(len(data), 2, 12.4, 4.65 * len(data) + 1.05,
                       f"{name} | " + tr("目标宽高联合热力图", "Target width / height density"),
                       tr("左侧为外接框的相对宽高，右侧为原始像素。对数坐标，白点为宽、高各自的中位数。",
                          "Bounding-box sizes, relative and original pixels. Log axes; white marker = coordinate-wise median."))
    for i, stat in enumerate(data):
        for j, key in enumerate(keys):
            ax = axes[i, j]
            xlab, ylab = (tr("目标宽 / 图宽", "Target width / image width"),
                          tr("目标高 / 图高", "Target height / image height")) if j == 0 else (
                tr("目标宽 (px)", "Target width (px)"), tr("目标高 (px)", "Target height (px)"))
            item = size_heatmap(ax, stat[key], bounds[j], maps[j][i], norm, j == 0, xlab, ylab)
            ax.set_title(f"{stat['group'].name} | " + (tr("宽高比例", "Relative") if j == 0 else tr("像素宽高", "Original pixels")))
            if not len(stat[key]):
                empty_panel(ax)
    percentage_colorbar(fig, item, list(axes.flat), tr("每格有效实例占比", "Share of valid instances per bin"))
    save(fig, out, "04_width_height_joint_heatmap.png")


def chart_resolution(data, name, out):
    values = np.concatenate([d["res"] for d in data], axis=0)
    edges = [log_edges(values[:, k]) for k in range(2)]
    maps = [density_grid(d["res"], *edges) for d in data]
    norm = density_norm(maps)
    fig, axes = figure(len(data), 2, 12.8, 4.65 * len(data) + 1.05,
                       f"{name} | " + tr("原始分辨率分布", "Original image resolution"),
                       tr("统计可读原图尺寸（包含无有效标注图片）。左侧为对数宽高分布，右侧为最常见 8 种分辨率。",
                          "All readable image headers, including images without valid labels. Left: log density; right: top eight exact sizes."))
    for i, stat in enumerate(data):
        group = stat["group"].name
        item = size_heatmap(axes[i, 0], stat["res"], edges, maps[i], norm, False,
                            tr("原图宽 (px)", "Image width (px)"), tr("原图高 (px)", "Image height (px)"))
        axes[i, 0].set_title(f"{group} | " + tr(f"{len(stat['res']):,} 张可读图片", f"{len(stat['res']):,} readable images"))
        ax = axes[i, 1]
        ranked = top_dimensions(stat["res"], 8)
        if ranked:
            values = [count for _, count in ranked]
            bars = ax.barh(np.arange(len(ranked)), values, height=.67, color=COLORS[i % len(COLORS)])
            annotate_bars(ax, bars)
            ax.set(yticks=np.arange(len(ranked)),
                   yticklabels=[f"{w:,} × {h:,}" for (w, h), _ in ranked],
                   ylim=(-.65, len(ranked) - .35), xlim=(0, max(1, max(values) * 1.23)))
            ax.invert_yaxis()
        else:
            empty_panel(axes[i, 0], tr("无可读图片", "No readable images"))
            empty_panel(ax, tr("无可读图片", "No readable images"))
        distinct = len(set(map(tuple, stat["res"])))
        ax.set_title(f"{group} | " + tr(f"{distinct:,} 种分辨率", f"{distinct:,} distinct resolutions"))
        ax.set_xlabel(tr("图片数", "Images"))
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    percentage_colorbar(fig, item, list(axes[:, 0]), tr("每格可读图片占比", "Share of readable images per bin"))
    save(fig, out, "05_original_resolution_distribution.png")


def main():
    args = parse_args()
    args.root = args.root.resolve()
    try:
        groups = find_groups(args.root, args.groups)
        classes, class_file = load_classes(args)
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc))
    args.out.mkdir(parents=True, exist_ok=True)
    notes = [
        "Image dimensions come from Pillow image headers; full-pixel decoding is not performed.",
        "Valid instance rows are matched-image rows with known (if configured) nonnegative class IDs, "
        "proper geometry, finite normalized coordinates and usable image dimensions.",
        "Raw paired counts are per image-label association; all-label-file counts use unique paths per group.",
        "Image extensions sharing the same relative stem can reference the same label; these stem collisions are reported. "
        "If multiple labels match one image, the first sorted path is used for geometry statistics.",
        "Masks are inventoried in sibling Mask/mask/Masks/masks directories and paired by stem or stem_mask. "
        "Mask presence is optional; missing-mask count is null when no mask directory is found.",
        "Orphan label rows are excluded from valid instance counts and all geometry plots.",
        "Detection overlays depict bounding rectangles; segmentation overlays rasterize the annotated polygons.",
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        formats = None
        task = args.task
        if task == "auto":
            task, formats = infer_task(groups, pool)
            if formats.get("detection") and formats.get("segmentation"):
                warning = ("Auto task inference found both 5-field and polygon-shaped rows; the majority "
                           "shape determines the task. Supply --task explicitly to override.")
                notes.append(warning)
                print(f"WARNING: {warning}", flush=True)
            elif not formats:
                notes.append("No recognizable annotation rows; auto task defaults to segmentation.")
        print(f"Task: {task}", flush=True)
        data = [scan_group(group, task, classes, pool) for group in groups]
        duplicates = cross_group_hashes(groups, pool) if args.hash_cross_groups else None
    if classes is None:
        ids = sorted({c for stat in data for c in stat["all_label_classes"] if c >= 0})
        classes = {c: f"Class {c}" for c in ids}
        notes.append("Class names inferred from observed nonnegative IDs; semantic names are unknown.")
    summary = {
        "root": str(args.root), "dataset": args.name or args.root.name, "task": task,
        "task_requested": args.task, "auto_format_candidates": formats,
        "class_source": class_file or "observed_ids",
        "classes": [{"id": c, "name": n} for c, n in sorted(classes.items())],
        "groups": [group_summary(stat, classes) for stat in data],
        "cross_group_image_hashes": duplicates,
        "notes": notes,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    configure_plots()
    name = summary["dataset"]
    chart_classes(data, classes, name, args.out)
    chart_overlay(data, task, name, args.out)
    chart_centers(data, name, args.out)
    chart_wh(data, name, args.out)
    chart_resolution(data, name, args.out)


if __name__ == "__main__":
    main()

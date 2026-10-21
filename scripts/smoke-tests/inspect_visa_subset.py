"""
Step 1: Prepare a small MMR-AD/VisA sample for inspecting images and supervision.

The script selects cashew queries with text annotations, obtains healthy images
from a separate reference bank, and copies the corresponding RGB images and real
pixel masks out of the official ZIP. HTTP byte ranges avoid storing the full ZIP.
Query/reference filenames are checked for overlap, and anomaly mask sizes and
nonempty contents are checked against the query images.

Outputs under data/mmrad_visa_sample/:
    - Original selected files, with their archive-relative paths preserved.
    - manifest.json: query identity, image label, mask path, healthy reference
      candidates, raw narrative, raw final answer, and anomaly-mask area fraction.
The preview is saved in codex_project_handoff/docs/visa_sample_preview.png.

The full text_annos.json is retained: its <think> field contains descriptive
narratives, while <answer> generally contains a decision plus boxes and labels.
The manifest stores both fields explicitly for inspection. The narrative still
needs curation into a concise defect-and-location target. Neither field is a
model input.

This deterministic inspection sample is not a training/evaluation split.
Usage (Conda environment cs7643-project):
    python scripts/inspect_visa_subset.py
    python scripts/inspect_visa_subset.py --anomalies 4 --normals 4 --references 4
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from remote_zip import RemoteZip


DATASET_REVISION = "99f89fd9308c9c67c7a46c62e1873d9438e1cad0"
# Pin the archive revision so a later upstream update cannot change this sample.
DATASET_URL = f"https://huggingface.co/datasets/xcyao00/MMR-AD/resolve/{DATASET_REVISION}/visa.zip"
CATEGORY = "cashew"
ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = ROOT / "data" / "mmrad_visa_sample"
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
NARRATIVE_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def extract_member(archive: zipfile.ZipFile, name: str, destination: Path) -> None:
    """Copy one ZIP member to disk, reusing an already extracted destination.

    Arguments
    ----------
    - archive: zipfile.ZipFile = Open archive backed by the HTTP range reader.
    - name: str = Exact member path as listed in the archive's directory.
    - destination: Path = Local output file, including its filename.

    Returns
    -------
    - None. Missing parent directories are created. Completed downloads replace
      their .part file atomically; an interrupted copy is never a completed file.

    Existing files are skipped rather than redownloaded. Image and mask checks
    later in main() validate the selected local query/mask pairs.
    """
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with archive.open(name) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target)
    temporary.replace(destination)


def query_path(name: str) -> str:
    """Translate a cashew text-annotation identifier to its RGB archive path.

    Arguments
    ----------
    - name: str = Annotation name, e.g. 'anomaly-000.png' or 'test_good-006.png'.

    Returns
    -------
    - str = Relative RGB path ending in .JPG. Annotation identifiers use .png
      even though their query images in this archive are JPEGs. Unsupported
      annotation groups raise ValueError; main() checks the resulting path exists.
    """
    stem = Path(name).stem
    if stem.startswith("anomaly-"):
        return f"visa/{CATEGORY}/test/anomaly/{stem.removeprefix('anomaly-')}.JPG"
    if stem.startswith("test_good-"):
        return f"visa/{CATEGORY}/test_good/{stem.removeprefix('test_good-')}.JPG"
    raise ValueError(f"Unexpected query name: {name}")


def plot_preview(rows: list[dict], destination: Path) -> None:
    """Show a normal and anomalous query alongside a healthy reference and mask.

    Arguments
    ----------
    - rows: list[dict] = Manifest rows containing both labels. Image and mask
      paths are resolved relative to SAMPLE_ROOT, which main() sets from --output.
    - destination: Path = Output PNG path; its parent is created if necessary.

    Returns
    -------
    - None. Writes a 2-by-3 preview. Only nonzero anomaly pixels receive a red
      overlay; a normal query is shown unchanged because its mask is empty.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [next(row for row in rows if row["label"] == label) for label in (1, 0)]
    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    for row_index, row in enumerate(selected):
        query = np.asarray(Image.open(SAMPLE_ROOT / row["query_path"]).convert("RGB"))
        reference = np.asarray(Image.open(SAMPLE_ROOT / row["reference_paths"][0]).convert("RGB"))
        axes[row_index, 0].imshow(query)
        axes[row_index, 0].set_title(f"Query: {'anomaly' if row['label'] else 'normal'}")
        axes[row_index, 1].imshow(reference)
        axes[row_index, 1].set_title("Healthy reference")
        axes[row_index, 2].imshow(query)
        if row["mask_path"]:
            mask = np.asarray(Image.open(SAMPLE_ROOT / row["mask_path"]).convert("L")) > 0
            axes[row_index, 2].imshow(np.ma.masked_where(~mask, mask), alpha=0.7, cmap="Reds", vmin=0, vmax=1)
        axes[row_index, 2].set_title("Query with mask overlay")
    for axis in axes.flat:
        axis.axis("off")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=130)
    plt.close(figure)


def main() -> None:
    """Select, download, validate, and visualize an annotated inspection sample.

    Arguments
    ----------
    - None directly. Reads --anomalies, --normals, --references, and --output
      from the command line; default counts are 8, 8, and 4 respectively.

    Returns
    -------
    - None. Writes downloaded files, a manifest, and the preview, then prints
      observed counts and raw final answers. Missing files or invalid masks raise
      an exception before an apparently valid manifest is written.
    """
    global SAMPLE_ROOT

    # Parse the script arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anomalies", type=int, default=8)
    parser.add_argument("--normals", type=int, default=8)
    parser.add_argument("--references", type=int, default=4)
    parser.add_argument("--output", type=Path, default=SAMPLE_ROOT)
    args = parser.parse_args()
    SAMPLE_ROOT = args.output.resolve()
    if min(args.anomalies, args.normals, args.references) < 1:
        parser.error("All sample counts must be positive")

    # Read selected portions of a remote ZIP archive dataset without downloading the whole ZIP.
    remote = RemoteZip(DATASET_URL)
    with zipfile.ZipFile(remote) as archive:
        available = set(archive.namelist())
        prefix = f"visa/{CATEGORY}/"
        annotation_path = prefix + "text_annos.json"
        annotations = json.loads(archive.read(annotation_path))
        
        # Pick (8: anomalies) identifiers only covered by the text release. The image archive
        # contains additional images without retained text annotations.
        anomalies = sorted(
            (item for item in annotations if item["name"].startswith("anomaly-")),
            key=lambda item: item["name"],
        )[: args.anomalies]

        # Normal (8: normals) queries come from `test_good`.
        normals = sorted(
            (item for item in annotations if item["name"].startswith("test_good-")),
            key=lambda item: item["name"],
        )[: args.normals]
        if len(anomalies) != args.anomalies or len(normals) != args.normals:
            raise RuntimeError("Requested more annotated examples than available")

        # Healthy reference candidates (4: references) come from `train_good`. A query therefore cannot retrieve its own file here.
        references = sorted(
            name for name in available if name.startswith(prefix + "train_good/") and name.lower().endswith(".jpg")
        )[: args.references]
        if len(references) != args.references:
            raise RuntimeError("Requested more healthy references than available")

        selected = anomalies + normals
        required = {annotation_path, prefix + "reference.json", prefix + "bbox_annos.json", *references}
        rows = []
        for item in selected:
            image = query_path(item["name"])
            label = int(item["name"].startswith("anomaly-")) # 1 = True: names starts with "anomaly-"; 0: False
            mask = prefix + "ground_truth/anomaly/" + Path(image).stem + ".png" if label else None

            # Keep narrative and structured answer separate and visible in the manifest. Curation into a short training 
            # target is a later step; these annotations must not be supplied as inspection model inputs.
            answer_match = ANSWER_RE.search(item["text"])
            narrative_match = NARRATIVE_RE.search(item["text"])
            if not answer_match:
                raise RuntimeError(f"Missing final answer for {item['name']}")
            if not narrative_match or not narrative_match.group(1).strip():
                raise RuntimeError(f"Missing narrative for {item['name']}")

            # After the `answer` and `narrative` check, add the image and its mask.
            required.add(image)
            if mask:
                required.add(mask)
            if image in references:
                raise RuntimeError("Query/reference identity overlap")

            # Record rows w.r.t. only the picked data samples (8: anomalies, 8: normals, 4: references) from the 
            # selected parts ("cashew" class) of the dataset.
            rows.append(
                {
                    "source_dataset"                    : "VisA via MMR-AD",
                    "category"                          : CATEGORY,
                    "source_id"                         : item["name"],
                    "query_path"                        : image,
                    "label"                             : label,
                    "mask_path"                         : mask,
                    "reference_paths"                   : references,
                    "raw_answer_for_inspection_only"    : answer_match.group(1).strip(),
                    "raw_narrative_for_inspection_only" : narrative_match.group(1).strip(),
                }
            )

        # Check all the `required` (picked) data samples are available in the `archive` (selected dataset).
        missing = required - available
        if missing:
            raise FileNotFoundError(f"Archive members missing: {sorted(missing)}")

        # Download only the picked data samples (8: anomalies, 8: normals, 4: references). 
        for name in sorted(required):
            extract_member(archive, name, SAMPLE_ROOT / name)

    # A bounding box is not a mask. Validate actual pixel-mask files, retaining each image's original geometry; 
    # normal masks are implicitly all zero.
    for row in rows:
        # Checks if the image file is corrupted by without decoding the full pixel data. Returns `None` if image is fine, else raise an error.
        with Image.open(SAMPLE_ROOT / row["query_path"]) as query:
            query.verify()
            
        if row["mask_path"]:
            with Image.open(SAMPLE_ROOT / row["query_path"]) as query, Image.open(SAMPLE_ROOT / row["mask_path"]) as mask:
                # Ensure that the query image and its mask have the same size.
                query_size, mask_size = query.size, mask.size
                if query_size != mask_size:
                    raise ValueError(f"Misaligned mask: {row['source_id']} {query_size} != {mask_size}")
                # Ensure that there is at least one pixel with non-zero value representing the mask.
                mask_values = np.unique(np.asarray(mask))
                if not np.any(mask_values):
                    raise ValueError(f"Empty anomaly mask: {row['source_id']}")
                row["mask_nonzero_fraction"] = float(np.mean(np.asarray(mask) > 0))
        else:
            row["mask_nonzero_fraction"] = 0.0

    SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = SAMPLE_ROOT / "manifest.json"
    manifest.write_text(json.dumps({"archive": DATASET_URL, "revision": DATASET_REVISION, "purpose": "inspection_only", "rows": rows}, indent=2) + "\n")
    preview = ROOT / "codex_project_handoff" / "docs" / "visa_sample_preview.png"
    plot_preview(rows, preview)
    print(f"Prepared {len(rows)} queries, {len(references)} references, and {len(anomalies)} masks")
    print(f"Manifest: {manifest}")
    print(f"Preview: {preview}")
    for row in rows:
        print(row["source_id"], "anomaly" if row["label"] else "normal", row["mask_nonzero_fraction"], row["raw_answer_for_inspection_only"][:100])


if __name__ == "__main__":
    main()

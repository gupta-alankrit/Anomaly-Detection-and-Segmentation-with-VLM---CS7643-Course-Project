"""
Step 2: Audit which MMR-AD/VisA cashew records have all required supervision sources.

The retained local text_annos.json contains both <think> narratives and <answer>
structured responses. This audit counts both fields separately, maps annotation
identifiers to original RGB images, and checks real mask-member availability in
the pinned remote ZIP. Presence of a narrative is not a semantic quality score.

Only the remote ZIP directory is fetched; no additional RGB images, masks, or
model weights are downloaded. The directory is cached under data/ and reused on
later runs. The previously downloaded sample's real images and masks are checked
locally, including dimensions, mask values, and query/reference file exclusion.

Outputs
-------
- data/mmrad_visa_sample/supervision_audit.json: counts, per-record availability,
  and checks on locally downloaded masks. Paths are archive-relative.
- data/mmrad_visa_archive_index.json: reproducible, revision-tagged ZIP directory.

Usage (Conda environment cs7643-project)
--------------------------------------
    python scripts/audit_visa_supervision.py

This script does not produce final caption targets, approve category splits, or
claim that every generated sentence is visually correct.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from inspect_visa_subset import DATASET_REVISION, DATASET_URL, SAMPLE_ROOT
from remote_zip import RemoteZip


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "data" / "mmrad_visa_archive_index.json"
TAG_PATTERNS = {
    tag: re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
    for tag in ("think", "answer")
}


def load_archive_index() -> set[str]:
    """Load or fetch the directory of the pinned VisA ZIP archive.

    Arguments
    ----------
    - None. Uses DATASET_URL, DATASET_REVISION, and INDEX_PATH.

    Returns
    -------
    - set[str] = Paths of all non-directory ZIP members. A matching local index
      avoids network access. A missing index is fetched and saved under data/.
    """
    if INDEX_PATH.exists():
        cached = json.loads(INDEX_PATH.read_text())
        if cached["revision"] == DATASET_REVISION:
            return set(cached["members"])

    remote = RemoteZip(DATASET_URL)
    try:
        with zipfile.ZipFile(remote) as archive:
            members = sorted(item.filename for item in archive.infolist() if not item.is_dir())
    finally:
        remote.session.close()
        remote.close()

    # Create the index JSON that includes the dataset revision # and all image filenames, bounding box JSON,
    # reference JSON, and text annotation JSON filenames of each class in the data subset.
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps({"revision": DATASET_REVISION, "members": members}, indent=2) + "\n")

    return set(members)


def annotation_paths(name: str) -> tuple[str, str | None, bool]:
    """Map one cashew annotation identifier to an RGB path, mask, and label.

    Arguments
    ----------
    - name: str = Identifier such as anomaly-000.png, test_good-006.png, or
      train_good-246.png. Label comes from the image group, not generated text.

    Returns
    -------
    - image_path  : str()  = Path of the original image corresponding to `name` identifier.
    - mask_path   : str()  = Path of the pixel-wise mask image corresponding to `name` idenitifier. Normal examples
                                have no mask file, their desired pixel targer is all zeros and hence return None.
    - is_anomalous: bool() = True: if there is an anomaly; False: if the normal example.
    Unsupported identifier groups raise ValueError.
    """
    group, image_id = Path(name).stem.split("-", 1)
    prefix = "visa/cashew/"
    if group == "anomaly":
        return prefix + f"test/anomaly/{image_id}.JPG", prefix + f"ground_truth/anomaly/{image_id}.png", True
    if group in ("test_good", "train_good"):
        return prefix + f"{group}/{image_id}.JPG", None, False
    raise ValueError(f"Unsupported annotation identifier: {name}")


def extract_tag(text: str, tag: str) -> str:
    """Return a trimmed tag body (of text), or an empty string for a missing tag pair.

    Arguments
    ----------
    - text: str() = Full generated annotation text.
    - tag : str() = 'think' for narrative or 'answer' for decision/box information.

    Returns
    -------
    - str = Verbatim tag contents after stripping outer whitespace. No attempt
      is made to summarize, validate, or rewrite the narrative.
    """
    match = TAG_PATTERNS[tag].search(text)
    return match.group(1).strip() if match else ""


def check_local_masks(manifest: dict) -> dict:
    """Check actual downloaded query/mask geometry and healthy-reference exclusion.

    Arguments
    ----------
    - manifest: dict = Loaded sample manifest containing a 'rows' list. Paths
      are relative to SAMPLE_ROOT; label 1 means anomaly and label 0 means normal.

    Returns
    -------
    - dict = Number of local pairs checked and observed anomaly mask values.
      Invalid geometry, empty anomalous masks, or query/reference overlap raise
      ValueError. Filename exclusion does not establish content deduplication.
    """
    values = set()
    anomaly_count = 0
    for row in manifest["rows"]:
        if row["query_path"] in row["reference_paths"]:
            raise ValueError(f"Query is its own reference: {row['source_id']}")
        with Image.open(SAMPLE_ROOT / row["query_path"]) as query:
            query_size = query.size
        if row["label"]:
            with Image.open(SAMPLE_ROOT / row["mask_path"]) as mask:
                mask_array = np.asarray(mask)
                if mask.size != query_size or not np.any(mask_array > 0):
                    raise ValueError(f"Invalid mask for {row['source_id']}")
                values.update(int(value) for value in np.unique(mask_array))
            anomaly_count += 1
    return {"queries": len(manifest["rows"]), "anomalous_masks_checked": anomaly_count,
            "mask_values": sorted(values), "query_reference_filename_overlap": False}


def main() -> None:
    """Audit retained category text against archive members and the local sample.

    Arguments
    ----------
    - None. Uses the sample and pinned archive prepared by inspect_visa_subset.py.

    Returns
    -------
    - None. Saves a detailed JSON report and prints aggregate counts. Archive
      availability and checks on downloaded pixels are reported separately.
    """

    # Load the names (strings) of all the images, bbox_annos.json, reference.json, and text_annos.json 
    # files for all the classes in the selected dataset ("cashew" class of VisA) as index directory.
    members = load_archive_index()

    # Read the annotations (bounding box, label, text) of all the images in the selected dataset ("cashew" class of VisA).
    annotations = json.loads((SAMPLE_ROOT / "visa/cashew/text_annos.json").read_text())

    counts = Counter()
    records = []
    for item in annotations:
        # Get the paths to the original image, pixel-wise mask image, and a boolean of anomaly (True if exists).
        image, mask, anomalous = annotation_paths(item["name"])

        # Extract narrative ("think") and label info ("answer") text from the completed provided text.
        narrative = extract_tag(item["text"], "think")
        answer = extract_tag(item["text"], "answer")

        # Find if the original image and its mask is present in the `members` (index directory).
        image_present = image in members
        mask_present = mask in members if anomalous else None

        # File-level supervision availability includes actual mask members for anomalies and an implicit empty target 
        # for normal examples. `counts` carries the important audit numbers throughout the selected dataset ("cashew" 
        # class of VisA) and `records` carries per-image audit results.
        complete = image_present and bool(narrative) and bool(answer) and (not anomalous or mask_present)
        consistent_decision = bool(re.match(r"^Yes\b" if anomalous else r"^No\b", answer, re.IGNORECASE))
        counts["text_records"] += 1
        counts["anomalous_records" if anomalous else "normal_records"] += 1
        counts["nonempty_narrative"] += bool(narrative)
        counts["nonempty_answer"] += bool(answer)
        counts["images_present_in_archive"] += image_present
        counts["anomalous_masks_present_in_archive"] += anomalous and bool(mask_present)
        counts["records_with_all_supervision_sources"] += bool(complete)
        counts["decisions_consistent_with_image_group"] += consistent_decision
        records.append({
            "name"               : item["name"], 
            "image_path"         : image, 
            "mask_path"          : mask,
            "is_anomalous"       : anomalous, 
            "narrative_present"  : bool(narrative),
            "answer_present"     : bool(answer), 
            "image_present"      : image_present,
            "mask_member_present": mask_present, 
            "all_sources_present": bool(complete)
        })

    # Add some additional numbers in `counts` w.r.t. the selected dataset ("cashew" class of VisA).
    id_counts = Counter(item["name"] for item in annotations)
    counts["unique_text_ids"] = len(id_counts)
    counts["unique_anomalous_images_with_text"] = sum(name.startswith("anomaly-") for name in id_counts)
    counts["unique_normal_images_with_text"] = sum(not name.startswith("anomaly-") for name in id_counts)
    counts["archive_cashew_anomaly_images"] = sum(name.startswith("visa/cashew/test/anomaly/") for name in members)
    counts["archive_cashew_normal_images"] = sum(name.startswith(("visa/cashew/test_good/", "visa/cashew/train_good/")) for name in members)

    # Check all the masks for the downloaded data samples of the selected dataset ("cashew" class of VisA).
    local = check_local_masks(json.loads((SAMPLE_ROOT / "manifest.json").read_text()))

    # Compile all the audit information and write in a JSON file.
    report = {
        "dataset_revision": DATASET_REVISION, 
        "category": "cashew", 
        "counts": dict(counts),
        "local_pixel_checks": local, 
        "records": records,
        "duplicate_annotation_ids": {name: n for name, n in id_counts.items() if n > 1},
        "limitations": ["Nonempty narrative does not certify factual accuracy or explicit spatial wording.",
                        "Remote records were checked against ZIP member names; only local sample masks were decoded.",
                        "Concise description targets and category splits have not been finalized."]
    }
    destination = SAMPLE_ROOT / "supervision_audit.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    
    print(json.dumps({"counts": dict(counts), "local_pixel_checks": local}, indent=2))
    print(f"Report: {destination}")


if __name__ == "__main__":
    main()

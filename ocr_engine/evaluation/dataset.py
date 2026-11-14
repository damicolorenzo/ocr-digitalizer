
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkSample:
    id: str
    image: Path
    language: str
    ground_truth: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_dataset(
    dataset_path: str | Path,
    project_root: str | Path,
) -> list[BenchmarkSample]:
    dataset_path = Path(dataset_path)
    project_root = Path(project_root).resolve()

    records = json.loads(dataset_path.read_text(encoding="utf-8"))

    if not isinstance(records, list) or not records:
        raise ValueError("Dataset must be a non-empty list")

    samples = []
    seen_ids = set()

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each entry must be a JSON object")

        for key in ("id", "image", "language", "ground_truth"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f"Missing or invalid field: {key}")

        if record["id"] in seen_ids:
            raise ValueError(f"Duplicate sample ID: {record['id']}")
        seen_ids.add(record["id"])

        relative_path = Path(record["image"])
        if relative_path.is_absolute():
            raise ValueError("Image paths must be relative")

        image_path = (project_root / relative_path).resolve()
        if not image_path.is_relative_to(project_root):
            raise ValueError("Image path escapes project root")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a JSON object")

        samples.append(
            BenchmarkSample(
                id=record["id"],
                image=image_path,
                language=record["language"],
                ground_truth=record["ground_truth"],
                metadata=metadata,
            )
        )

    return samples


import re
import unicodedata

from jiwer import cer, wer

from collections.abc import Sequence


def normalize_text(text: str) -> str:
    """Normalize OCR text for a secondary evaluation."""
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()

    # Remove punctuation while preserving Unicode letters.
    text = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in text
    )

    return re.sub(r"\s+", " ", text).strip()


def evaluate_recognition(
    ground_truth: str,
    prediction: str,
) -> dict:
    """Calculate raw and normalized OCR error rates."""
    if not ground_truth.strip():
        raise ValueError("Ground truth must not be empty")

    normalized_gt = normalize_text(ground_truth)
    normalized_prediction = normalize_text(prediction)

    return {
        "cer": cer(ground_truth, prediction),
        "wer": wer(ground_truth, prediction),
        "normalized_cer": (
            cer(normalized_gt, normalized_prediction)
            if normalized_gt else None
        ),
        "normalized_wer": (
            wer(normalized_gt, normalized_prediction)
            if normalized_gt else None
        ),
    }

def evaluate_corpus(
    references: Sequence[str],
    predictions: Sequence[str],
) -> dict:
    if not references:
        raise ValueError("Corpus cannot be empty")

    if len(references) != len(predictions):
        raise ValueError("Corpus lengths do not match")

    if any(not text.strip() for text in references):
        raise ValueError("Ground truth must not be empty")

    normalized_refs = [normalize_text(x) for x in references]
    normalized_preds = [normalize_text(x) for x in predictions]

    if any(not text for text in normalized_refs):
        raise ValueError("Normalized ground truth must not be empty")

    return {
        "cer": cer(list(references), list(predictions)),
        "wer": wer(list(references), list(predictions)),
        "normalized_cer": cer(normalized_refs, normalized_preds),
        "normalized_wer": wer(normalized_refs, normalized_preds),
    }

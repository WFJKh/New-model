"""SMIDS cross-validation entry point leveraging the shared sperm_cv pipeline."""

from sperm_cv import SMIDS_CONFIG, main
from sperm_cv.pipeline import predict_folder_with_best_pipeline, reproduce_best

__all__ = [
    "SMIDS_CONFIG",
    "main",
    "predict_folder_with_best_pipeline",
    "reproduce_best",
]


if __name__ == "__main__":
    main(SMIDS_CONFIG)

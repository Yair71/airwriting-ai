"""Synthetic unistroke data and training pipeline (Python / GPU)."""

from ml_pipeline.augmentor import AugmentConfig, Augmentor
from ml_pipeline.dataset import UnistrokeDataset, generate_synthetic_dataset, kinematic_preprocess
from ml_pipeline.font_sampler import CHARSETS, build_charset, discover_fonts, extract_glyph_polyline, resample_arc_length
from ml_pipeline.model import ModelConfig, UnistrokeNet, count_parameters
from ml_pipeline.stroke_generator import generate_stroke_dataset

__all__ = [
    "AugmentConfig",
    "Augmentor",
    "CHARSETS",
    "ModelConfig",
    "UnistrokeDataset",
    "UnistrokeNet",
    "build_charset",
    "count_parameters",
    "discover_fonts",
    "extract_glyph_polyline",
    "generate_stroke_dataset",
    "generate_synthetic_dataset",
    "kinematic_preprocess",
    "resample_arc_length",
]

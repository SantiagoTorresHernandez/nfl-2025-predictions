from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    RANDOM_STATE: int = 42
    TRAIN_START_SEASON: int = 2010
    TRAIN_END_SEASON:   int = 2024  # last season to include for training/validation
    VAL_YEARS: int = 2              # last N seasons used for validation (e.g., 2023–2024)

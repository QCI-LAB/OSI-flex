from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class OSIFlexConfig:
    # Input images
    input_images: List[str]                 # List of file paths

    # Shifts (per image)
    shears: List[Tuple[float, float]]      # list of (dx, dy) tuples

    # Algorithm options
    align_images: bool = False
    max_iter: int = 400
    tv_coeff: float = 1e-8
    sign_reg_coeff: float = 0.0      # positive → nonnegativity, negative → nonpositivity
    shear_lr: float = 1e-5                  # 0 → disable fine-tuning

    # Optimizer
    opti_method: str = "Adam"
    learning_rate: float = 0.1

    def __post_init__(self):
        # Ensure floats where needed
        self.tv_coeff = float(self.tv_coeff)
        self.sign_reg_coeff = float(self.sign_reg_coeff)
        self.shear_lr = float(self.shear_lr)
        self.learning_rate = float(self.learning_rate)

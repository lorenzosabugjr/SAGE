from enum import Enum, unique
import numpy as np

@unique
class NoiseType(Enum):
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"

def add_noise(value: float, noise_type: NoiseType, param: float) -> float:
    """
    Adds noise to a value based on the noise type and parameter.
    
    Args:
        value: The original value.
        noise_type: The type of noise (UNIFORM or GAUSSIAN).
        param: The noise parameter. 
               For UNIFORM, this is the total width of the interval [-param/2, param/2].
               For GAUSSIAN, this is the standard deviation (1-sigma).
    """
    if noise_type == NoiseType.UNIFORM:
        return value + param * (np.random.rand() - 0.5)
    elif noise_type == NoiseType.GAUSSIAN:
        return value + np.random.normal(0, param)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

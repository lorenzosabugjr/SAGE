import numpy as np
from utils.noise import NoiseType, add_noise

class BaseProblem:
    def __init__(self, randseed:int=1):
        pass

    def eval(self, x:np.ndarray, noise_type: NoiseType = NoiseType.UNIFORM, noise_param: float = 0.0) -> float:
        """
        Evaluate the objective function at x.
        
        Args:
            x: Input vector.
            noise_type: Type of noise to add.
            noise_param: Parameter for the noise.
            
        Returns:
            float: Function value + noise.
        """
        val = self._eval_deterministic(x)
        return add_noise(val, noise_type, noise_param)

    def _eval_deterministic(self, x: np.ndarray) -> float:
        raise NotImplementedError("Subclasses must implement _eval_deterministic")

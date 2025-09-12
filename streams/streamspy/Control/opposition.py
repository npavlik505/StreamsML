import numpy as np
from streamspy.base_Classical import BaseController

class controller(BaseController):
    """Opposition control: act against measured quantity."""

    def __init__(
        self,
        gain: float = 1.0,
        sensor_index: int = 0,
        min_action: float | None = None,
        max_action: float | None = None,
        **kwargs,
    ) -> None:
        self.gain = gain
        self.sensor_index = sensor_index
        self.min_action = min_action
        self.max_action = max_action

    def reset(self) -> None:
        pass

    def compute_action(self, observation):
        output = -self.gain * observation[self.sensor_index]
        if self.min_action is not None or self.max_action is not None:
            low = self.min_action if self.min_action is not None else output
            high = self.max_action if self.max_action is not None else output
            output = np.clip(output, low, high)
        return float(output)

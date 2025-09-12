import numpy as np
from streamspy.base_Classical import BaseController


class controller(BaseController):
    """Simple PID controller."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        setpoint: float = 0.0,
        sensor_index: int = 0,
        dt: float = 1.0,
        min_action: float | None = None,
        max_action: float | None = None,
        **kwargs,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.sensor_index = sensor_index
        self.dt = dt
        self.min_action = min_action
        self.max_action = max_action
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0

    def compute_action(self, observation):
        value = observation[self.sensor_index]
        error = self.setpoint - value
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        if self.min_action is not None or self.max_action is not None:
            low = self.min_action if self.min_action is not None else output
            high = self.max_action if self.max_action is not None else output
            output = np.clip(output, low, high)
        return float(output)

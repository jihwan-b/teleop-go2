"""Glove controller for velocity command input via serial communication."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class GloveConfig:
    """Configuration for glove controller."""

    port: str = "/dev/ttyACM0"  # Linux default, use COM3 for Windows or /dev/cu.usbmodem* for Mac
    baud_rate: int = 115200
    max_displacement: float = 80.0  # Maximum displacement from glove sensor
    magnet_threshold: float = 90.0  # Z_corr threshold to enable interface
    activation_time: float = 0.5  # Time (seconds) magnet must be detected to activate
    deadzone_ratio: float = 0.35  # Center deadzone as ratio of max_displacement

    # Velocity mapping parameters
    max_lin_vel: float = 1.0  # Maximum linear velocity (m/s)
    max_ang_vel: float = 1.0  # Maximum angular velocity (rad/s)

    # Control mode: "holonomic" or "differential"
    # - holonomic: Uses both lin_x and lin_y (strafing enabled)
    # - differential: Only lin_x and ang_z (like a wheeled robot)
    control_mode: str = "holonomic"

    # Finger bend sensing (MPR121)
    finger_names: List[str] = field(default_factory=lambda: ["Middle", "Ring", "Pinky"])


class GloveController:
    """Real-time glove input controller via serial communication.

    Reads Heading, Displacement, and Z_corr from serial port and converts
    them to velocity commands (lin_x, lin_y, ang_z).
    """

    def __init__(self, config: GloveConfig | None = None):
        """Initialize glove controller.

        Args:
            config: Glove configuration. If None, uses default config.
        """
        self.cfg = config or GloveConfig()

        # Serial connection
        self._serial = None
        self._serial_thread = None
        self._running = False

        # Current sensor readings
        self._lock = threading.Lock()
        self._heading = 0.0  # degrees (0-360)
        self._displacement = 0.0  # units
        self._z_corr = 0.0  # magnetic field strength

        # State machine: "OFF", "WAITING", "ON"
        self._ui_state = "OFF"
        self._detection_start_time = 0.0

        # Output velocity commands
        self._lin_x = 0.0  # m/s
        self._lin_y = 0.0  # m/s
        self._ang_z = 0.0  # rad/s

        # Finger bend data (from MPR121): O=OPEN, X=BENT
        n_fingers = len(self.cfg.finger_names)
        self._finger_bent = [False] * n_fingers

    def start(self) -> bool:
        """Start serial communication thread.

        Returns:
            True if successfully connected, False otherwise.
        """
        try:
            import serial

            self._serial = serial.Serial(self.cfg.port, self.cfg.baud_rate, timeout=1)
            print(f"✅ Glove connected: {self.cfg.port}")
        except Exception as e:
            print(f"❌ Failed to connect to glove: {e}")
            return False

        self._running = True
        self._serial_thread = threading.Thread(target=self._serial_read_loop, daemon=True)
        self._serial_thread.start()
        return True

    def stop(self) -> None:
        """Stop serial communication and close connection."""
        self._running = False
        if self._serial_thread is not None:
            self._serial_thread.join(timeout=1.0)
        if self._serial is not None:
            self._serial.close()
            print("🔌 Glove disconnected")

    def get_velocity_command(self) -> Tuple[float, float, float]:
        """Get current velocity command from glove input.

        Returns:
            Tuple of (lin_x, lin_y, ang_z) in m/s and rad/s.
        """
        with self._lock:
            return self._lin_x, self._lin_y, self._ang_z

    def is_active(self) -> bool:
        """Check if glove interface is currently active (ON state).

        Returns:
            True if interface is ON, False otherwise.
        """
        with self._lock:
            return self._ui_state == "ON"

    def get_state(self) -> str:
        """Get current state machine state.

        Returns:
            One of "OFF", "WAITING", or "ON".
        """
        with self._lock:
            return self._ui_state

    def get_finger_data(self) -> List[Tuple[str, bool]]:
        """Get current finger bend data.

        Returns:
            List of (name, is_bent) for each finger.
        """
        with self._lock:
            return [
                (self.cfg.finger_names[i], self._finger_bent[i])
                for i in range(len(self.cfg.finger_names))
            ]

    def _serial_read_loop(self) -> None:
        """Background thread that reads serial data and updates state."""
        while self._running:
            try:
                if self._serial and self._serial.in_waiting:
                    line = self._serial.readline().decode("utf-8", errors="ignore").strip()

                    # Parse data using regex
                    match_head = re.search(r"Heading:([-0-9.]+)", line)
                    match_disp = re.search(r"Displacement:([-0-9.]+)", line)
                    match_z = re.search(r"Z_corr:([-0-9.]+)", line)

                    # Parse finger bend data: Bending:OXO (O=BENT, X=OPEN)
                    match_bending = re.search(r"Bending:([OX]+)", line)

                    with self._lock:
                        if match_head:
                            self._heading = float(match_head.group(1))
                        if match_disp:
                            self._displacement = float(match_disp.group(1))
                        if match_z:
                            self._z_corr = float(match_z.group(1))

                        if match_bending:
                            bending_str = match_bending.group(1)
                            for i in range(min(len(bending_str), len(self._finger_bent))):
                                self._finger_bent[i] = (bending_str[i] == "X")

                # Update state machine and velocity commands
                self._update_state()

            except Exception as e:
                print(f"⚠️ Serial read error: {e}")
                time.sleep(0.01)

            time.sleep(0.001)  # 1ms loop

    def _update_state(self) -> None:
        """Update state machine and compute velocity commands.

        State transitions:
        - OFF -> WAITING: Magnet detected
        - WAITING -> ON: Magnet held for activation_time
        - WAITING/ON -> OFF: Magnet removed
        """
        with self._lock:
            is_magnet_detected = abs(self._z_corr) >= self.cfg.magnet_threshold
            current_time = time.time()

            # State machine transitions
            if is_magnet_detected:
                if self._ui_state == "OFF":
                    self._ui_state = "WAITING"
                    self._detection_start_time = current_time
                elif self._ui_state == "WAITING":
                    if current_time - self._detection_start_time >= self.cfg.activation_time:
                        self._ui_state = "ON"
                # ON state: keep ON
            else:
                # Magnet removed -> immediate reset
                self._ui_state = "OFF"
                self._detection_start_time = 0.0

            # Compute velocity commands based on state
            if self._ui_state == "ON":
                self._compute_velocity_command()
            else:
                # Not active -> zero velocity
                self._lin_x = 0.0
                self._lin_y = 0.0
                self._ang_z = 0.0

    def _compute_velocity_command(self) -> None:
        """Convert heading and displacement to velocity commands."""
        # Apply deadzone
        deadzone_threshold = self.cfg.max_displacement * self.cfg.deadzone_ratio
        if self._displacement < deadzone_threshold:
            self._lin_x = 0.0
            self._lin_y = 0.0
            self._ang_z = 0.0
            return

        # Normalize displacement to [0, 1]
        normalized_disp = min(self._displacement, self.cfg.max_displacement) / self.cfg.max_displacement

        # Convert heading to radians (0° = North = Forward)
        heading_rad = np.deg2rad(self._heading)

        if self.cfg.control_mode == "holonomic":
            # Holonomic mode: Use both linear x and y
            # Matplotlib polar: theta_direction=1 (CCW), theta_zero='N'
            # heading 0° = forward, 90° (CCW) = left, 180° = backward, 270° (CCW) = right
            self._lin_x = normalized_disp * self.cfg.max_lin_vel * np.cos(heading_rad)
            self._lin_y = normalized_disp * self.cfg.max_lin_vel * np.sin(heading_rad)
            self._ang_z = 0.0  # No rotation in pure translation

        elif self.cfg.control_mode == "differential":
            # Differential mode: Forward/backward + rotation
            # Forward component (cos of heading)
            forward_component = np.cos(heading_rad)
            # Rotation component (sin of heading) - CCW is positive
            rotation_component = np.sin(heading_rad)

            self._lin_x = normalized_disp * self.cfg.max_lin_vel * forward_component
            self._lin_y = 0.0  # No lateral movement in differential mode
            self._ang_z = normalized_disp * self.cfg.max_ang_vel * rotation_component

        else:
            raise ValueError(f"Unknown control mode: {self.cfg.control_mode}")


class FingerBendDisplay:
    """Small matplotlib window for debugging finger bend status."""

    def __init__(self, controller: GloveController, update_interval: int = 50):
        """Initialize display.

        Args:
            controller: GloveController instance to read finger data from.
            update_interval: Matplotlib update interval in ms.
        """
        self._controller = controller
        self._update_interval = update_interval

    def start(self) -> None:
        """Launch the matplotlib display window (blocks the calling thread)."""
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        finger_names = self._controller.cfg.finger_names
        n = len(finger_names)

        fig, ax = plt.subplots(figsize=(3.5, 2))
        fig.canvas.manager.set_window_title("Finger Bend Debug")

        # Full-width bars as status indicators
        bars = ax.barh(range(n), [1.0] * n, color=["#4CAF50"] * n, height=0.6)
        ax.set_yticks(range(n))
        ax.set_yticklabels(finger_names)
        ax.set_xlim(0, 1)
        ax.set_xticks([])

        status_texts = []
        for i in range(n):
            txt = ax.text(0.5, i, "OPEN", ha="center", va="center",
                          fontsize=12, fontweight="bold", color="white")
            status_texts.append(txt)

        def update(_frame):
            data = self._controller.get_finger_data()
            for i, (_name, bent) in enumerate(data):
                if bent:
                    bars[i].set_color("#F44336")
                    status_texts[i].set_text("BENT")
                else:
                    bars[i].set_color("#4CAF50")
                    status_texts[i].set_text("OPEN")
            return list(bars) + status_texts

        _anim = FuncAnimation(fig, update, interval=self._update_interval, blit=False)
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    import sys

    # Create glove controller
    config = GloveConfig(
        port="/dev/ttyACM0",  # Adjust for your system
        control_mode="holonomic",  # or "differential"
        max_lin_vel=1.0,
        max_ang_vel=1.0,
    )

    controller = GloveController(config)

    if not controller.start():
        print("Failed to start glove controller")
        sys.exit(1)

    print("Glove controller started. Press Ctrl+C to exit.")
    print(f"Control mode: {config.control_mode}")

    # Launch finger bend debug display if --finger-debug flag is given
    # 고쳐야함: random.py 충돌로 matplotlib 사용 불가능
    show_finger = "--finger-debug" in sys.argv

    if show_finger:
        print("Finger bend display enabled (close window to exit)")
        display = FingerBendDisplay(controller)
        try:
            display.start()  # blocks until window is closed
        except KeyboardInterrupt:
            pass
        finally:
            controller.stop()
    else:
        try:
            while True:
                # 속도 및 상태 데이터 가져오기
                lin_x, lin_y, ang_z = controller.get_velocity_command()
                state = controller.get_state()

                # [추가됨] 핑거 벤딩 데이터 가져오기 및 포맷팅 (O: 굽힘, X: 펴짐)
                finger_data = controller.get_finger_data()
                # finger_data는 (이름, bool) 튜플의 리스트입니다. bool 값만 추출하여 변환합니다.
                bending_str = "".join(["O" if bent else "X" for _, bent in finger_data])

                print(
                    f"\rState: {state:8s} | "
                    f"lin_x: {lin_x:+.3f} m/s | "
                    f"lin_y: {lin_y:+.3f} m/s | "
                    f"ang_z: {ang_z:+.3f} rad/s | "
                    f"Bending:{bending_str}",  # 요청하신 출력 형식 추가
                    end="",
                )
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n\nStopping glove controller...")
            controller.stop()

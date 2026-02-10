"""Glove-controlled viewer that integrates glove input with MuJoCo viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from mjlab.utils.glove_controller import GloveConfig, GloveController
from mjlab.viewer.base import EnvProtocol, PolicyProtocol, VerbosityLevel
from mjlab.viewer.native.viewer import NativeMujocoViewer, PlotCfg

if TYPE_CHECKING:
    from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand


class GlovePlayViewer(NativeMujocoViewer):
    """MuJoCo viewer with glove input control for velocity commands.

    This viewer extends the native MuJoCo viewer to accept real-time
    velocity commands from a glove controller via serial communication.
    """

    def __init__(
        self,
        env: EnvProtocol,
        policy: PolicyProtocol,
        glove_config: GloveConfig | None = None,
        frame_rate: float = 60.0,
        key_callback: Optional[Callable[[int], None]] = None,
        plot_cfg: PlotCfg | None = None,
        enable_perturbations: bool = True,
        verbosity: VerbosityLevel = VerbosityLevel.SILENT,
    ):
        """Initialize glove-controlled viewer.

        Args:
            env: RL environment.
            policy: Policy network for inference.
            glove_config: Glove controller configuration. If None, uses default.
            frame_rate: Viewer frame rate (Hz).
            key_callback: Optional keyboard callback.
            plot_cfg: Reward plot configuration.
            enable_perturbations: Enable interactive perturbations.
            verbosity: Logging verbosity level.
        """
        super().__init__(
            env,
            policy,
            frame_rate=frame_rate,
            key_callback=key_callback,
            plot_cfg=plot_cfg,
            enable_perturbations=enable_perturbations,
            verbosity=verbosity,
        )

        # Create glove controller
        self.glove_config = glove_config or GloveConfig()
        self.glove_controller = GloveController(self.glove_config)

        # Find velocity command term
        self._velocity_command_term: Optional[UniformVelocityCommand] = None

    def setup(self) -> None:
        """Setup viewer and glove controller."""
        super().setup()

        # Start glove controller
        if self.glove_controller.start():
            self.log("✅ Glove controller started", VerbosityLevel.INFO)
            self.log(
                f"   Port: {self.glove_config.port} | "
                f"Mode: {self.glove_config.control_mode}",
                VerbosityLevel.INFO,
            )
        else:
            self.log("❌ Failed to start glove controller", VerbosityLevel.INFO)
            self.log("   Falling back to default velocity commands", VerbosityLevel.INFO)

        # Find velocity command term in the environment
        self._velocity_command_term = self._find_velocity_command_term()
        if self._velocity_command_term is None:
            self.log(
                "⚠️  Warning: No velocity command term found in environment",
                VerbosityLevel.INFO,
            )
            self.log("   Glove input will have no effect", VerbosityLevel.INFO)
        else:
            self.log("✅ Velocity command term found", VerbosityLevel.INFO)

    def step_simulation(self) -> None:
        """Step simulation with glove input integration."""
        if self._is_paused:
            return

        # Update velocity command from glove input BEFORE stepping environment
        self._update_velocity_from_glove()

        # Normal simulation step (calls policy and env.step)
        super().step_simulation()

    def _update_velocity_from_glove(self) -> None:
        """Update environment velocity command from glove input."""
        if self._velocity_command_term is None:
            return

        # Get current velocity command from glove
        # Note: GloveController returns (0, 0, 0) when magnet is not detected (OFF/WAITING state)
        lin_x, lin_y, ang_z = self.glove_controller.get_velocity_command()


        # [added 0210] 핑거 벤딩 데이터 가져오기 (O: 굽힘, X: 펴짐)
        finger_data = self.glove_controller.get_finger_data()
        bending_str = "".join(["O" if bent else "X" for _, bent in finger_data])


        # Always enable external control - glove controller handles stopping (velocity=0) automatically
        # This ensures robot stops when magnet is removed, instead of reverting to random commands
        self._velocity_command_term.set_external_command(
            lin_x=lin_x,
            lin_y=lin_y,
            ang_z=ang_z,
            enabled=True,  # Always enabled - glove controller outputs 0 when inactive
        )

        # Log state changes (only when transitioning)
        state = self.glove_controller.get_state()

        # [added 0210] 터미널에 실시간 상태 출력 (\r 사용)   
        # 매 스텝 출력하면 너무 빠를 수 있으므로 필요하다면 카운터를 두어 조절 가능
        print(
            f"\rState: {state:8s} | "
            f"lin_x: {lin_x:+.3f} m/s | "
            f"lin_y: {lin_y:+.3f} m/s | "
            f"ang_z: {ang_z:+.3f} rad/s | "
            f"Bending:{bending_str}   ", # 끝에 공백을 조금 두어 잔상 제거
            end="",
            flush=True
        )

        if not hasattr(self, "_last_glove_state"):
            self._last_glove_state = "OFF"

        if state != self._last_glove_state:
            if state == "ON":
                self.log("🟢 Glove control ACTIVE", VerbosityLevel.INFO)
            elif state == "OFF":
                self.log("🔴 Glove control INACTIVE", VerbosityLevel.INFO)
            elif state == "WAITING":
                self.log("🟡 Glove control WAITING (hold magnet)...", VerbosityLevel.INFO)
            self._last_glove_state = state

    def _find_velocity_command_term(self) -> Optional[UniformVelocityCommand]:
        """Find the velocity command term in the environment.

        Returns:
            UniformVelocityCommand instance if found, None otherwise.
        """
        try:
            # Access command manager from unwrapped environment
            env = self.env.unwrapped
            if not hasattr(env, "command_manager"):
                return None

            # Look for velocity command terms (common names: "twist", "velocity", "vel_command")
            possible_names = ["twist", "velocity", "vel_command", "base_velocity"]
            for name in possible_names:
                term = env.command_manager.get_term(name)
                if term is not None:
                    # Check if it has the set_external_command method (UniformVelocityCommand)
                    if hasattr(term, "set_external_command"):
                        self.log(f"   Found velocity command: '{name}'", VerbosityLevel.DEBUG)
                        return term

            # Fallback: iterate all terms to find one with set_external_command
            for name, term in env.command_manager._terms.items():
                if hasattr(term, "set_external_command"):
                    self.log(f"   Found velocity command: '{name}'", VerbosityLevel.DEBUG)
                    return term

        except Exception as e:
            self.log(f"   Error finding velocity command: {e}", VerbosityLevel.DEBUG)

        return None

    def close(self) -> None:
        """Close viewer and glove controller."""
        self.glove_controller.stop()
        super().close()
        self.log("✅ Glove controller stopped", VerbosityLevel.INFO)


# Convenience function for quick testing
def run_glove_viewer(
    env: EnvProtocol,
    policy: PolicyProtocol,
    port: str = "/dev/ttyACM0",
    control_mode: str = "holonomic",
    max_lin_vel: float = 1.0,
    max_ang_vel: float = 1.0,
) -> None:
    """Run glove-controlled viewer with default settings.

    Args:
        env: RL environment.
        policy: Policy network.
        port: Serial port for glove (e.g., "/dev/ttyACM0", "COM3").
        control_mode: "holonomic" or "differential".
        max_lin_vel: Maximum linear velocity (m/s).
        max_ang_vel: Maximum angular velocity (rad/s).
    """
    glove_config = GloveConfig(
        port=port,
        control_mode=control_mode,
        max_lin_vel=max_lin_vel,
        max_ang_vel=max_ang_vel,
    )

    viewer = GlovePlayViewer(
        env=env,
        policy=policy,
        glove_config=glove_config,
        verbosity=VerbosityLevel.INFO,
    )

    viewer.run()

# 🧤 Glove Control Integration Guide

This guide shows how to control the trained RL robot (Unitree Go2) using glove input via serial communication.

## 📋 Overview

The glove control system allows you to control the robot's velocity commands in real-time using:
- **Heading** (0-360°): Direction of movement
- **Displacement**: Speed magnitude
- **Z_corr** (magnetic field): Interface activation trigger

The system converts these inputs into velocity commands (`lin_x`, `lin_y`, `ang_z`) that the trained RL policy follows.

---

## 🔧 Hardware Setup

1. **Connect glove to computer via USB serial**
   - Linux: `/dev/ttyACM0` (or `/dev/ttyUSB0`)
   - macOS: `/dev/cu.usbmodem*`
   - Windows: `COM3` (or similar)

2. **Verify serial connection**
   ```bash
   # Linux/macOS
   ls /dev/tty* | grep -E 'ACM|USB|usbmodem'

   # Test connection (optional)
   python mjlab/utils/glove_controller.py
   ```

---

## 🚀 Quick Start

### Basic Usage (Holonomic Mode)

Run the trained Go2 robot with glove control:

```bash
python scripts/play.py Mjlab-Velocity-Flat-Unitree-Go2 \
  --checkpoint_file=logs/rsl_rl/go2_velocity/2026-02-06_23-07-57/model_20000.pt \
  --viewer=glove \
  --glove_port=/dev/ttyACM0
```

### Differential Drive Mode

For tank-like controls (forward/backward + rotation only):

```bash
python scripts/play.py Mjlab-Velocity-Flat-Unitree-Go2 \
  --checkpoint_file=logs/rsl_rl/go2_velocity/2026-02-06_23-07-57/model_20000.pt \
  --viewer=glove \
  --glove_port=/dev/ttyACM0 \
  --glove_control_mode=differential
```

### Custom Velocity Limits

```bash
python scripts/play.py Mjlab-Velocity-Flat-Unitree-Go2 \
  --checkpoint_file=logs/rsl_rl/go2_velocity/2026-02-06_23-07-57/model_20000.pt \
  --viewer=glove \
  --glove_max_lin_vel=0.5 \
  --glove_max_ang_vel=0.8
```

---

## ⚙️ Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--viewer` | `auto` | Set to `glove` to enable glove control |
| `--glove_port` | `/dev/ttyACM0` | Serial port for glove connection |
| `--glove_control_mode` | `holonomic` | `holonomic` (strafe) or `differential` (tank) |
| `--glove_max_lin_vel` | `1.0` | Maximum linear velocity (m/s) |
| `--glove_max_ang_vel` | `1.0` | Maximum angular velocity (rad/s) |

---

## 🎮 Control Modes

### 1. Holonomic Mode (Default)
- **Best for**: Omnidirectional movement, strafing
- **Mapping**:
  - `heading = 0°` → Move forward (`lin_x > 0`)
  - `heading = 90°` → Move right (`lin_y < 0`)
  - `heading = 180°` → Move backward (`lin_x < 0`)
  - `heading = 270°` → Move left (`lin_y > 0`)
  - `displacement` → Speed magnitude

### 2. Differential Mode
- **Best for**: Tank-like controls, wheeled robots
- **Mapping**:
  - Forward component: `lin_x = displacement × cos(heading)`
  - Rotation component: `ang_z = -displacement × sin(heading)`
  - `lin_y` is always 0 (no lateral movement)

---

## 🔍 Glove Input Protocol

The system expects serial data in this format:

```
Heading:45.2 Displacement:65.3 Z_corr:120.5
```

### Activation Logic
1. **OFF** → **WAITING**: Magnet detected (`|Z_corr| >= 90.0`)
2. **WAITING** → **ON**: Magnet held for 0.2 seconds
3. **WAITING/ON** → **OFF**: Magnet removed (instant reset)

### Deadzone
- Center deadzone: 35% of max displacement
- Below deadzone → velocity = 0 (neutral stance)

---

## 📊 Data Flow Diagram

```
┌─────────────────┐
│  Glove Sensor   │
│  (Serial USB)   │
└────────┬────────┘
         │ Heading, Displacement, Z_corr
         ▼
┌─────────────────────────────────────┐
│  GloveController (Background Thread)│
│  - Parse serial data                │
│  - State machine (OFF/WAITING/ON)   │
│  - Convert to velocity              │
└────────┬────────────────────────────┘
         │ (lin_x, lin_y, ang_z)
         ▼
┌─────────────────────────────────────┐
│  UniformVelocityCommand             │
│  - set_external_command()           │
│  - Override default commands        │
└────────┬────────────────────────────┘
         │ vel_command_b [num_envs, 3]
         ▼
┌─────────────────────────────────────┐
│  ObservationManager                 │
│  - Include command in obs           │
└────────┬────────────────────────────┘
         │ observation tensor
         ▼
┌─────────────────────────────────────┐
│  RL Policy (Neural Network)         │
│  - Trained to follow commands       │
└────────┬────────────────────────────┘
         │ joint position targets
         ▼
┌─────────────────────────────────────┐
│  MuJoCo Simulation                  │
│  - Go2 robot executes commands      │
└─────────────────────────────────────┘
```

---

## 🐛 Troubleshooting

### Issue: "Failed to connect to glove"
**Solution:**
- Check USB connection
- Verify correct port: `ls /dev/tty*`
- Check permissions: `sudo chmod 666 /dev/ttyACM0`
- Install pyserial: `pip install pyserial`

### Issue: Robot doesn't move when glove is active
**Solution:**
- Check if `|Z_corr| >= 90.0` (magnet threshold)
- Hold magnet for 0.2 seconds to activate
- Increase displacement beyond deadzone (35%)
- Check serial data format matches expected format

### Issue: Jerky or unstable movement
**Solution:**
- Reduce `--glove_max_lin_vel` and `--glove_max_ang_vel`
- Ensure serial baud rate is correct (default: 115200)
- Check for USB connection stability

---

## 🧪 Testing Standalone Controller

Test the glove controller without running the full simulation:

```bash
# Test holonomic mode
python -c "
from mjlab.utils.glove_controller import GloveConfig, GloveController
import time

config = GloveConfig(
    port='/dev/ttyACM0',
    control_mode='holonomic',
    max_lin_vel=1.0,
    max_ang_vel=1.0,
)

controller = GloveController(config)
if controller.start():
    print('Glove controller started. Press Ctrl+C to exit.')
    try:
        while True:
            lin_x, lin_y, ang_z = controller.get_velocity_command()
            state = controller.get_state()
            print(f'\rState: {state:8s} | lin_x: {lin_x:+.3f} | lin_y: {lin_y:+.3f} | ang_z: {ang_z:+.3f}', end='')
            time.sleep(0.05)
    except KeyboardInterrupt:
        controller.stop()
"
```

---

## 📝 Example: Full Command

```bash
python scripts/play.py Mjlab-Velocity-Flat-Unitree-Go2 \
  --checkpoint_file=logs/rsl_rl/go2_velocity/2026-02-06_23-07-57/model_20000.pt \
  --viewer=glove \
  --glove_port=/dev/ttyACM0 \
  --glove_control_mode=holonomic \
  --glove_max_lin_vel=0.8 \
  --glove_max_ang_vel=1.2 \
  --num_envs=1
```

---

## 🎯 Tips for Best Performance

1. **Single environment**: Use `--num_envs=1` for glove control (commands broadcast to all envs)
2. **Smooth movements**: Start with low max velocities and gradually increase
3. **Control mode**:
   - Use `holonomic` for quadrupeds with omnidirectional capability
   - Use `differential` for simpler forward/turn behaviors
4. **Deadzone**: Keep hands steady near center to avoid unintended small movements

---

## 🔗 Related Files

- **Glove Controller**: [mjlab/utils/glove_controller.py](mjlab/utils/glove_controller.py)
- **Glove Viewer**: [mjlab/viewer/glove_viewer.py](mjlab/viewer/glove_viewer.py)
- **Velocity Command**: [mjlab/tasks/velocity/mdp/velocity_command.py](mjlab/tasks/velocity/mdp/velocity_command.py)
- **Play Script**: [scripts/play.py](scripts/play.py)

---

## 📄 License

This integration is part of the Unitree RL MJLab project.

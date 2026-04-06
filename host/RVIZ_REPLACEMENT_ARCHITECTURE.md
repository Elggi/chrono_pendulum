# RViz-based IMU Viewer Replacement Architecture

## Nodes
1. `rviz_pendulum_calibration_node.py`
   - Subscribes: `/imu/data`, `/hw/enc`, `/hw/pwm_applied`, `/ina219/current_ma`
   - Publishes: `/pendulum/rviz_markers` (`visualization_msgs/MarkerArray`)
   - Visuals:
     - IMU orientation arm marker
     - encoder-derived arm marker
     - current magnitude bar marker

2. (Optional) Interactive marker node
   - `interactive_markers` package can add handles for:
     - pivot
     - radius
     - reference axis

3. Calibration writer service (host-side)
   - Collects CPR/radius/g/current offset
   - Writes JSON format compatible with `calibration.py`

## ROS message contracts
- `sensor_msgs/Imu`: orientation + angular velocity + linear acceleration
- `std_msgs/Float32`: encoder count, PWM, INA current
- `visualization_msgs/MarkerArray`: geometry overlays
- `visualization_msgs/InteractiveMarker` (optional): interactive calibration controls

## Data flow
Raw hardware (`bridge_node`) -> topics -> host calibration node -> derived estimates -> JSON -> loaded by `chrono_pendulum.py`/replay/stage fitting.

## Calibration integration targets
- CPR estimation from encoder delta vs orientation cycles
- radius estimation from orientation geometry
- gravity estimation from low-dynamics windows
- current offset from motor-off windows (PWM≈0)

## Output JSON fields
```json
{
  "summary": {
    "mean_cpr": 0,
    "mean_radius_m": 0,
    "g_eff_mps2": 0,
    "ina_current_offset_mA": 0
  }
}
```

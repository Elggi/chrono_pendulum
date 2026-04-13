#!/usr/bin/env bash
set -Eeuo pipefail

# ==========================================================
# Chrono Pendulum master stack launcher
# ==========================================================

resolve_script_dir() {
  local src="${BASH_SOURCE[0]}"
  while [ -h "$src" ]; do
    local dir
    dir="$(cd -P "$(dirname "$src")" >/dev/null 2>&1 && pwd)"
    src="$(readlink "$src")"
    [[ "$src" != /* ]] && src="$dir/$src"
  done
  cd -P "$(dirname "$src")" >/dev/null 2>&1 && pwd
}

SCRIPT_DIR="$(resolve_script_dir)"
REPO_ROOT=""

if git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
  REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
  REPO_ROOT="$SCRIPT_DIR"
fi

SRC_DIR="$REPO_ROOT/src"
ROS_WS_DIR="$REPO_ROOT/ros2_ws"
DATA_DIR="$REPO_ROOT/data"
MODELS_DIR="$REPO_ROOT/models"
CONFIGS_DIR="$REPO_ROOT/configs"

PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

detect_python_with_module() {
  local module="$1"
  shift
  local candidates=("$@")
  for candidate in "${candidates[@]}"; do
    if [ -z "$candidate" ]; then
      continue
    fi
    if ! command_exists "$candidate"; then
      continue
    fi
    if "$candidate" - <<PY >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("$module") else 1)
PY
    then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

ROS_PYTHON_BIN="$(detect_python_with_module "rclpy" python3 /usr/bin/python3 python || true)"
SIM_PYTHON_BIN="$PYTHON_BIN"

print_header() {
  local title="$1"
  echo
  echo "=========================================================="
  echo "$title"
  echo "=========================================================="
}

press_enter() {
  read -r -p "Press Enter to continue..." _
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

run_in_repo() {
  local cmd="$1"
  echo
  echo ">>> $cmd"
  (
    cd "$REPO_ROOT"
    eval "$cmd"
  )
}

run_in_repo_bg() {
  local cmd="$1"
  echo
  echo ">>> $cmd"
  (
    cd "$REPO_ROOT"
    nohup bash -lc "$cmd" >/tmp/pendulum_stack_$(date +%s).log 2>&1 &
    echo "Launched in background. Check /tmp/pendulum_stack_*.log"
  )
}

run_ros_module() {
  local cmd="$1"
  if [ -z "$ROS_PYTHON_BIN" ]; then
    echo "ROS Python (rclpy) not found. Install/source ROS Python env (e.g. /usr/bin/python3 with rclpy)."
    return 1
  fi
  run_in_repo "$ROS_PYTHON_BIN -m $cmd"
}

prompt_yes_no() {
  local prompt="$1"
  while true; do
    read -r -p "$prompt [y/n]: " yn
    case "$yn" in
      [Yy]*) return 0 ;;
      [Nn]*) return 1 ;;
      *) echo "Please enter y or n." ;;
    esac
  done
}

collect_files() {
  local root="$1"
  local pattern="$2"
  if [ ! -d "$root" ]; then
    return 0
  fi
  find "$root" -type f -name "$pattern" 2>/dev/null | sort
}

choose_file() {
  local prompt="$1"
  shift
  local files=("$@")
  local count="${#files[@]}"

  if [ "$count" -eq 0 ]; then
    echo ""
    return 1
  fi

  echo "$prompt"
  local i=1
  for f in "${files[@]}"; do
    echo "  $i) ${f#$REPO_ROOT/}"
    i=$((i + 1))
  done
  echo "  b) Back"

  while true; do
    read -r -p "Select file: " choice
    if [ "$choice" = "b" ]; then
      return 1
    fi
    if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "$count" ]; then
      local idx=$((choice - 1))
      echo "${files[$idx]}"
      return 0
    fi
    echo "Invalid selection."
  done
}

project_info() {
  print_header "Project info / quick help"
  cat <<'EOF'
Chrono Pendulum digital twin principles:

- Chrono handles rigid-body dynamics: geometry, mass, inertia, gravity, contact, joints.
- Actuation enters through ChLinkMotorRotationTorque.
- FREE-DECAY data (u≈0): train nominal passive model -> models/nominal/model.pt
- EXCITATION data (u!=0): train actuator/residual models -> models/actuator + models/sparse
- RL calibration (PPO): tune simulator calibration parameters for sim-vs-real agreement.
EOF
  press_enter
}

environment_checks() {
  print_header "Environment / dependency checks"
  echo "Repo root: $REPO_ROOT"
  echo "Python: ${PYTHON_BIN:-NOT FOUND}"
  echo "ROS Python (rclpy): ${ROS_PYTHON_BIN:-NOT FOUND}"
  echo "ros2: $(command -v ros2 || echo 'NOT FOUND')"
  echo "colcon: $(command -v colcon || echo 'NOT FOUND')"

  echo
  echo "Directory checks:"
  for d in "$SRC_DIR" "$DATA_DIR" "$MODELS_DIR" "$CONFIGS_DIR" "$ROS_WS_DIR"; do
    if [ -d "$d" ]; then
      echo "  [OK] ${d#$REPO_ROOT/}"
    else
      echo "  [MISSING] ${d#$REPO_ROOT/}"
    fi
  done

  echo
  echo "Discovered artifacts:"
  local pt_count json_count csv_count
  pt_count=$(collect_files "$MODELS_DIR" "*.pt" | wc -l | tr -d ' ')
  json_count=$(collect_files "$MODELS_DIR" "*.json" | wc -l | tr -d ' ')
  csv_count=$(collect_files "$DATA_DIR" "*.csv" | wc -l | tr -d ' ')
  echo "  model .pt files:  $pt_count"
  echo "  model .json files: $json_count"
  echo "  data .csv files:   $csv_count"

  if [ -n "$PYTHON_BIN" ]; then
    echo
    echo "Python package probe:"
    "$PYTHON_BIN" - <<'PY'
import importlib
pkgs = ["numpy", "pandas", "torch", "gymnasium", "stable_baselines3", "pychrono", "matplotlib"]
for p in pkgs:
    ok = importlib.util.find_spec(p) is not None
    print(f"  {'[OK]' if ok else '[MISSING]'} {p}")
PY
  fi
  press_enter
}

launch_imu_viewer() {
  print_header "IMU viewer"
  if [ -z "$PYTHON_BIN" ]; then
    echo "Python is not available."
    press_enter
    return
  fi
  if [ -z "$ROS_PYTHON_BIN" ]; then
    echo "ROS Python with rclpy is not available."
    press_enter
    return
  fi

  local imu_topic
  local window_s
  imu_topic="/imu/data"
  window_s="10.0"
  read -r -p "IMU topic [$imu_topic]: " custom_topic
  read -r -p "Rolling window seconds [$window_s]: " custom_window
  if [ -n "${custom_topic:-}" ]; then
    imu_topic="$custom_topic"
  fi
  if [ -n "${custom_window:-}" ]; then
    window_s="$custom_window"
  fi

  run_in_repo "$ROS_PYTHON_BIN -m src.visualization.imu_viewer --topic '$imu_topic' --window '$window_s'"
  press_enter
}

chrono_runtime_menu() {
  while true; do
    print_header "Chrono simulation / data collection"
    echo "1) Build sanity check only"
    echo "2) Run short open-loop simulation"
    echo "3) Start data collection flow"
    echo "b) Back"
    read -r -p "Select: " choice
    case "$choice" in
      1)
        run_in_repo "$SIM_PYTHON_BIN -m src.chrono_core.runtime --mode sanity --config '$CONFIGS_DIR/default_pendulum.json'"
        press_enter
        ;;
      2)
        run_in_repo "$SIM_PYTHON_BIN -m src.chrono_core.runtime --mode simulate --seconds 3.0 --config '$CONFIGS_DIR/default_pendulum.json'"
        press_enter
        ;;
      3)
        local data_mode
        echo "Select data collection target:"
        echo "  1) Free-decay"
        echo "  2) Excitation"
        read -r -p "Mode: " data_mode
        if [ "$data_mode" = "1" ]; then
          echo "Launching IMU viewer only (free-decay trajectory monitoring)."
          launch_imu_viewer
          continue
        fi
        if [ "$data_mode" != "2" ]; then
          echo "Invalid selection."
          press_enter
          continue
        fi

        local control_mode
        echo "Excitation control mode:"
        echo "  1) Host mode (/cmd/u published by host CLI)"
        echo "  2) Jetson mode (/cmd/u published onboard Jetson controller)"
        read -r -p "Mode: " control_mode
        if [ "$control_mode" = "1" ]; then
          if [ -z "$ROS_PYTHON_BIN" ]; then
            echo "ROS Python with rclpy is not available."
            press_enter
            continue
          fi
          local host_view
          read -r -p "Run realtime IMU viewer together? [y/N]: " host_view
          if [[ "$host_view" =~ ^[Yy]$ ]]; then
            run_in_repo_bg "$ROS_PYTHON_BIN -m src.visualization.imu_viewer --topic /imu/data --window 10.0"
          fi
          run_in_repo "$ROS_PYTHON_BIN -m src.ros_io.keyboard_controller --topic-cmd-u /cmd/u --loop-hz 20.0 --pwm-step 10.0 --pwm-max 255.0"
          press_enter
        elif [ "$control_mode" = "2" ]; then
          if [ -z "$ROS_PYTHON_BIN" ]; then
            echo "ROS Python with rclpy is not available."
            press_enter
            continue
          fi
          echo "Jetson mode selected: expecting onboard publisher on /cmd/u."
          echo "Launching INA219 bridge on ROS Python and running Chrono collection on simulation Python."
          local current_file
          local bridge_log
          local runtime_log
          current_file="/tmp/chrono_current_ma.txt"
          bridge_log="$DATA_DIR/raw/current_bridge_$(date +%Y%m%d_%H%M%S).csv"
          runtime_log="$DATA_DIR/raw/chrono_collect_$(date +%Y%m%d_%H%M%S).csv"
          run_in_repo_bg "$ROS_PYTHON_BIN -m src.ros_io.current_topic_bridge --topic /ina219/current_ma --imu-topic /imu/data --pwm-topic /hw/pwm_applied --out '$current_file' --log '$bridge_log' --warmup-sec 1.0"
          run_in_repo "$SIM_PYTHON_BIN -m src.chrono_core.runtime --mode collect-excitation --config '$CONFIGS_DIR/default_pendulum.json' --seconds 20.0 --current-file '$current_file' --log-csv '$runtime_log'"
          press_enter
        else
          echo "Invalid selection."
          press_enter
        fi
        ;;
      b) return ;;
      *) echo "Invalid option" ;;
    esac
  done
}

system_identification_menu() {
  while true; do
    print_header "System identification"
    echo "1) Train nominal passive model (free-decay only)"
    echo "2) Train actuator A-2 neural model (excitation only)"
    echo "3) Fit interpretable actuator regression"
    echo "4) Fit sparse residual equation (PySINDy / SINDy-PI)"
    echo "b) Back"
    read -r -p "Select: " choice

    mapfile -t csv_candidates < <(collect_files "$DATA_DIR" "*.csv")

    case "$choice" in
      1)
        local data_file
        data_file="$(choose_file 'Choose FREE-DECAY CSV (should include segment_type=free_decay):' "${csv_candidates[@]}")" || continue
        run_in_repo "$PYTHON_BIN -m src.cli.pipeline train-nominal --data '$data_file'"
        press_enter
        ;;
      2)
        local data_file
        data_file="$(choose_file 'Choose EXCITATION CSV (should include segment_type=excitation):' "${csv_candidates[@]}")" || continue
        run_in_repo "$PYTHON_BIN -m src.cli.pipeline train-actuator-a2 --data '$data_file'"
        press_enter
        ;;
      3)
        local data_file
        data_file="$(choose_file 'Choose excitation CSV with tau_eff:' "${csv_candidates[@]}")" || continue
        run_in_repo "$PYTHON_BIN -m src.cli.pipeline fit-regression --data '$data_file'"
        press_enter
        ;;
      4)
        local data_file
        data_file="$(choose_file 'Choose residual CSV with tau_residual:' "${csv_candidates[@]}")" || continue
        run_in_repo "$PYTHON_BIN -m src.cli.pipeline fit-sparse --data '$data_file'"
        press_enter
        ;;
      b) return ;;
      *) echo "Invalid option" ;;
    esac
  done
}

rl_menu() {
  print_header "RL calibration / fine-tuning"
  if [ -z "$PYTHON_BIN" ]; then
    echo "Python is not available."
    press_enter
    return
  fi

  mapfile -t csv_candidates < <(collect_files "$DATA_DIR" "*.csv")
  local data_file
  data_file="$(choose_file 'Choose rollout CSV with theta, omega, u:' "${csv_candidates[@]}")" || return

  local timesteps
  timesteps="20000"
  read -r -p "Training timesteps [$timesteps]: " user_steps
  if [[ -n "${user_steps:-}" ]]; then
    timesteps="$user_steps"
  fi

  run_in_repo "$PYTHON_BIN - <<'PY'
from pathlib import Path
from src.calibration_rl.train_ppo import RLTrainConfig, train_rl_calibrator
cfg = RLTrainConfig(data_csv=Path('$data_file'), timesteps=int('$timesteps'))
train_rl_calibrator(cfg)
print('Saved RL policy to models/rl/ppo_calibrator.zip')
PY"
  press_enter
}

analysis_menu() {
  while true; do
    print_header "Plot / replay / analysis"
    echo "1) Run realtime IMU trajectory viewer"
    echo "2) List available model artifacts"
    echo "3) List available data logs"
    echo "b) Back"
    read -r -p "Select: " choice
    case "$choice" in
      1) launch_imu_viewer ;;
      2)
        find "$MODELS_DIR" -type f \( -name '*.pt' -o -name '*.json' -o -name '*.zip' \) | sed "s|$REPO_ROOT/||"
        press_enter
        ;;
      3)
        find "$DATA_DIR" -type f -name '*.csv' | sed "s|$REPO_ROOT/||"
        press_enter
        ;;
      b) return ;;
      *) echo "Invalid option" ;;
    esac
  done
}

main_menu() {
  while true; do
    print_header "Chrono Pendulum Stack Console"
    echo "Repo: $REPO_ROOT"
    echo "1) Project info / quick help"
    echo "2) Environment / dependency checks"
    echo "3) IMU viewer"
    echo "4) Chrono simulation / data collection"
    echo "5) System identification"
    echo "6) RL calibration / fine-tuning"
    echo "7) Plot / replay / analysis"
    echo "8) Open interactive shell at repo root"
    echo "0) Exit"

    read -r -p "Select option: " choice
    case "$choice" in
      1) project_info ;;
      2) environment_checks ;;
      3) launch_imu_viewer ;;
      4) chrono_runtime_menu ;;
      5) system_identification_menu ;;
      6) rl_menu ;;
      7) analysis_menu ;;
      8)
        print_header "Interactive shell"
        echo "Starting subshell in $REPO_ROOT (type 'exit' to return)."
        (cd "$REPO_ROOT" && bash)
        ;;
      0) echo "Goodbye."; exit 0 ;;
      *) echo "Invalid option. Please try again." ;;
    esac
  done
}

if [ -z "$PYTHON_BIN" ]; then
  echo "Warning: no python interpreter found (python3/python). Some features unavailable."
fi

main_menu

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

  mapfile -t csv_candidates < <(collect_files "$DATA_DIR" "*.csv")
  if [ "${#csv_candidates[@]}" -lt 2 ]; then
    echo "Need at least two CSV files (real + sim) in data/ for overlay plotting."
    echo "You can still place files in data/raw or data/processed and retry."
    press_enter
    return
  fi

  local real_file sim_file out_file
  real_file="$(choose_file 'Select REAL trajectory CSV:' "${csv_candidates[@]}")" || return
  sim_file="$(choose_file 'Select SIM trajectory CSV:' "${csv_candidates[@]}")" || return

  mapfile -t png_candidates < <(collect_files "$DATA_DIR" "*.png")
  out_file="$DATA_DIR/processed/imu_overlay_$(date +%Y%m%d_%H%M%S).png"

  read -r -p "Output PNG path [$out_file]: " custom
  if [ -n "${custom:-}" ]; then
    out_file="$custom"
  fi

  run_in_repo "$PYTHON_BIN -m src.visualization.imu_viewer --real '$real_file' --sim '$sim_file' --out '$out_file'"
  press_enter
}

chrono_runtime_menu() {
  while true; do
    print_header "Chrono simulation / runtime"
    echo "1) Run Chrono plant build sanity check"
    echo "2) Run short open-loop simulation (if pychrono available)"
    echo "b) Back"
    read -r -p "Select: " choice
    case "$choice" in
      1)
        run_in_repo "$PYTHON_BIN -m src.chrono_core.runtime --mode sanity --config '$CONFIGS_DIR/default_pendulum.json'"
        press_enter
        ;;
      2)
        run_in_repo "$PYTHON_BIN -m src.chrono_core.runtime --mode simulate --seconds 3.0 --config '$CONFIGS_DIR/default_pendulum.json'"
        press_enter
        ;;
      b) return ;;
      *) echo "Invalid option" ;;
    esac
  done
}

ros_data_menu() {
  while true; do
    print_header "Data collection / ROS"
    echo "1) Show ROS workspace packages"
    echo "2) Build ROS workspace (colcon build)"
    echo "3) Launch IMU node (ros2 run hw_yahboom_imu imu_node)"
    echo "4) Launch bridge node (ros2 run hw_arduino_bridge bridge_node)"
    echo "b) Back"
    read -r -p "Select: " choice

    case "$choice" in
      1)
        if [ -d "$ROS_WS_DIR/src" ]; then
          find "$ROS_WS_DIR/src" -maxdepth 2 -name package.xml -print | sed "s|$REPO_ROOT/||"
        else
          echo "ROS workspace not found: $ROS_WS_DIR"
        fi
        press_enter
        ;;
      2)
        if ! command_exists colcon; then
          echo "colcon not installed."
          press_enter
        else
          run_in_repo "cd '$ROS_WS_DIR' && colcon build"
          press_enter
        fi
        ;;
      3)
        if ! command_exists ros2; then
          echo "ros2 command missing."
        else
          echo "Tip: source your ROS env first if needed (e.g. /opt/ros/<distro>/setup.bash)."
          run_in_repo "cd '$ROS_WS_DIR' && ros2 run hw_yahboom_imu imu_node"
        fi
        press_enter
        ;;
      4)
        if ! command_exists ros2; then
          echo "ros2 command missing."
        else
          echo "Tip: source your ROS env first if needed (e.g. /opt/ros/<distro>/setup.bash)."
          run_in_repo "cd '$ROS_WS_DIR' && ros2 run hw_arduino_bridge bridge_node"
        fi
        press_enter
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
    echo "4) Fit sparse residual equation (SINDy-style)"
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
    echo "1) Run IMU overlay plotting utility"
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
    echo "4) Chrono simulation / runtime"
    echo "5) Data collection / ROS"
    echo "6) System identification"
    echo "7) RL calibration / fine-tuning"
    echo "8) Plot / replay / analysis"
    echo "9) Open interactive shell at repo root"
    echo "0) Exit"

    read -r -p "Select option: " choice
    case "$choice" in
      1) project_info ;;
      2) environment_checks ;;
      3) launch_imu_viewer ;;
      4) chrono_runtime_menu ;;
      5) ros_data_menu ;;
      6) system_identification_menu ;;
      7) rl_menu ;;
      8) analysis_menu ;;
      9)
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

#!/bin/bash

# ======================================================
# 환경 세팅
# ======================================================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WS_SETUP="$SCRIPT_DIR/../ros2_ws/install/setup.bash"
WS_SETUP="${WS_SETUP:-$DEFAULT_WS_SETUP}"

source /opt/ros/humble/setup.bash
if [ ! -f "$WS_SETUP" ]; then
    echo "[ERROR] Workspace setup not found: $WS_SETUP"
    echo "[HINT] Build the ROS workspace first, then source it:"
    echo "       cd \"$SCRIPT_DIR/../ros2_ws\" && colcon build"
    echo "       source \"$SCRIPT_DIR/../ros2_ws/install/setup.bash\""
    exit 1
fi
source "$WS_SETUP"
export ROS_DOMAIN_ID=7
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

BASE_DIR="$SCRIPT_DIR"
CSV_DIR="$BASE_DIR/run_logs"

# ======================================================
# 유틸
# ======================================================

pause() {
    read -p "Press Enter to continue..."
}

select_csv_file() {
    echo "--------------------------------" >&2
    echo "[INFO] CSV 파일 선택" >&2

    if [ ! -d "$CSV_DIR" ]; then
        echo "[ERROR] run_logs 폴더 없음" >&2
        return 1
    fi

    select file in "$CSV_DIR"/*.csv; do
        if [ -n "$file" ]; then
            echo "[INFO] Selected: $file" >&2
            echo "$file"
            return 0
        else
            echo "[ERROR] 잘못된 선택" >&2
        fi
    done
}

select_json_file() {
    echo "--------------------------------" >&2
    echo "[INFO] Parameter JSON 파일 선택" >&2

    local files=("$BASE_DIR"/run_logs/*.json "$BASE_DIR"/rl_results/*.json)
    local valid=()
    for f in "${files[@]}"; do
        [ -f "$f" ] && valid+=("$f")
    done

    if [ "${#valid[@]}" -eq 0 ]; then
        local default_json="$BASE_DIR/run_logs/calibration_latest.json"
        echo "[WARN] 선택 가능한 json 파일이 없습니다. 기본값 사용: $default_json" >&2
        echo "$default_json"
        return 0
    fi

    select file in "${valid[@]}"; do
        if [ -n "$file" ]; then
            echo "[INFO] Selected JSON: $file" >&2
            echo "$file"
            return 0
        else
            echo "[ERROR] 잘못된 선택" >&2
        fi
    done
}

# ======================================================
# 실행 함수들
# ======================================================

run_imu_viewer() {
    echo "[INFO] imu_viewer 실행"
    python3 $BASE_DIR/imu_viewer.py
}

run_chrono_pendulum() {
    echo "--------------------------------"
    echo "Select mode:"
    echo "1) Host mode (keyboard control)"
    echo "2) Jetson mode (ROS input)"
    read -p "Enter number: " mode

    json_file=$(select_json_file)
    if [ -z "$json_file" ]; then return; fi

    if [ "$mode" == "1" ]; then
        echo "[INFO] chrono_pendulum (HOST mode)"
        python3 $BASE_DIR/chrono_pendulum.py --mode host --calibration-json "$json_file"
    elif [ "$mode" == "2" ]; then
        echo "[INFO] chrono_pendulum (JETSON mode)"
        python3 $BASE_DIR/chrono_pendulum.py --mode jetson --calibration-json "$json_file"
    else
        echo "[ERROR] Invalid selection"
    fi
}

run_system_identification() {
    echo "--------------------------------"
    echo "[INFO] System Identification 시작"
    echo "[INFO] host role로 adaptive calibration 실행"
    read -p "enter max-pwm (default 80): " input_max_pwm
    read -p "enter sweep-pwm-step (default 5): " input_sweep_step
    read -p "enter sweep-hold-sec (default 0.4): " input_sweep_hold

    max_pwm="${input_max_pwm:-80}"
    sweep_step="${input_sweep_step:-5}"
    sweep_hold="${input_sweep_hold:-0.4}"

    common_args=(
        --max-calib-pwm "$max_pwm"
        --sweep-pwm-step "$sweep_step"
        --sweep-hold-sec "$sweep_hold"
    )

    python3 $BASE_DIR/system_identification.py --role host "${common_args[@]}"
}

run_plot() {
    file=$(select_csv_file)
    if [ -z "$file" ]; then return; fi

    python3 $BASE_DIR/plot_pendulum.py --csv "$file"
}

run_rl_fitting() {
    file=$(select_csv_file)
    if [ -z "$file" ]; then return; fi

    echo "--------------------------------"
    echo "Select RL Algorithm:"
    echo "1) PPO (stable)"
    echo "2) SAC (more exploration)"
    read -p "Enter number: " algo_choice

    if [ "$algo_choice" == "1" ]; then
        algo="ppo"
    elif [ "$algo_choice" == "2" ]; then
        algo="sac"
    else
        echo "[ERROR] Invalid selection"
        return
    fi

    echo "[INFO] RL fitting 실행 ($algo)"
    python3 $BASE_DIR/RL_fitting.py --csv "$file" --algo $algo
}

run_full_pipeline() {
    echo "[INFO] Full pipeline 메뉴는 System Identification → Optimization → Chrono 순 workflow로 대체 예정"
}

# ======================================================
# 메인 메뉴
# ======================================================

while true; do
    clear
    echo "=========================================="
    echo "      Pendulum Digital Twin Stack"
    echo "=========================================="
    echo "1) IMU Viewer (Standalone Viewer)"
    echo "2) System Identification (Parameter Calibration)"
    echo "3) Parameter Optimization (Reinforcement Learning)"
    echo "4) Chrono Pendulum (Select Host/Jetson mode)"
    echo "5) Plot Data"
    echo "6) Exit"
    echo "=========================================="

    read -p "Select option: " choice

    case $choice in
        1)
            run_imu_viewer
            pause
            ;;
        2)
            run_system_identification
            pause
            ;;
        3)
            run_rl_fitting
            pause
            ;;
        4)
            run_chrono_pendulum
            pause
            ;;
        5)
            run_plot
            pause
            ;;
        6)
            echo "Bye!"
            exit 0
            ;;
        *)
            echo "[ERROR] Invalid input"
            pause
            ;;
    esac
done
select_json_file() {
    echo "--------------------------------" >&2
    echo "[INFO] Parameter JSON 파일 선택" >&2
    local files=("$BASE_DIR"/run_logs/*.json "$BASE_DIR"/rl_results/*.json)
    local valid=()
    for f in "${files[@]}"; do
        [ -f "$f" ] && valid+=("$f")
    done
    if [ "${#valid[@]}" -eq 0 ]; then
        echo "[ERROR] 선택 가능한 json 파일이 없습니다." >&2
        return 1
    fi
    select file in "${valid[@]}"; do
        if [ -n "$file" ]; then
            echo "[INFO] Selected JSON: $file" >&2
            echo "$file"
            return 0
        else
            echo "[ERROR] 잘못된 선택" >&2
        fi
    done
}

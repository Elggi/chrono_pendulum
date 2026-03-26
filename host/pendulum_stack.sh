#!/bin/bash

# ======================================================
# 환경 세팅
# ======================================================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/humble/setup.bash
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

    if [ "$mode" == "1" ]; then
        echo "[INFO] chrono_pendulum (HOST mode)"
        python3 $BASE_DIR/chrono_pendulum.py --mode host
    elif [ "$mode" == "2" ]; then
        echo "[INFO] chrono_pendulum (JETSON mode)"
        python3 $BASE_DIR/chrono_pendulum.py --mode jetson
    else
        echo "[ERROR] Invalid selection"
    fi
}

run_system_identification() {
    echo "--------------------------------"
    echo "[INFO] System Identification 시작"
    echo "[INFO] Calibration 실행"
    read -p "enter max-pwm (default 80): " input_max_pwm
    read -p "enter sweep-pwm-step (default 5): " input_sweep_step
    read -p "enter sweep-hold-sec (default 0.4): " input_sweep_hold
    read -p "enter max-turn (default 1.2): " input_max_turn

    max_pwm="${input_max_pwm:-80}"
    sweep_step="${input_sweep_step:-5}"
    sweep_hold="${input_sweep_hold:-0.4}"
    max_turn="${input_max_turn:-1.2}"

    common_args=(
        --max-calib-pwm "$max_pwm"
        --sweep-pwm-step "$sweep_step"
        --sweep-hold-sec "$sweep_hold"
        --max-turns-one-side "$max_turn"
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

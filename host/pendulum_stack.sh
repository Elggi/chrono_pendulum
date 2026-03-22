#!/bin/bash

# ======================================================
# 환경 세팅
# ======================================================
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=7
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR"
CSV_DIR="$BASE_DIR/run_logs"

# ======================================================
# 유틸
# ======================================================

pause() {
    read -p "Press Enter to continue..."
}

select_csv_file() {
    echo "--------------------------------"
    echo "[INFO] CSV 파일 선택"

    if [ ! -d "$CSV_DIR" ]; then
        echo "[ERROR] run_logs 폴더 없음"
        return 1
    fi

    select file in "$CSV_DIR"/*.csv; do
        if [ -n "$file" ]; then
            echo "[INFO] Selected: $file"
            echo "$file"
            return 0
        else
            echo "[ERROR] 잘못된 선택"
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
    echo "================================"
    echo "[PIPELINE] Chrono → Plot → RL"
    echo "================================"

    echo "[STEP 1] Chrono 실행"
    run_chrono_pendulum

    echo "[STEP 2] Plot 실행"
    run_plot

    echo "[STEP 3] RL fitting 실행"
    run_rl_fitting

    echo "[DONE] Full pipeline complete"
}

# ======================================================
# 메인 메뉴
# ======================================================

while true; do
    clear
    echo "=========================================="
    echo "      Pendulum Digital Twin Stack"
    echo "=========================================="
    echo "1) IMU Viewer (standalone)"
    echo "2) Chrono Pendulum (simulation)"
    echo "3) Plot CSV (analysis)"
    echo "4) RL Fitting (offline optimization)"
    echo "5) Full Pipeline (Chrono → Plot → RL)"
    echo "6) Exit"
    echo "=========================================="

    read -p "Select option: " choice

    case $choice in
        1)
            run_imu_viewer
            pause
            ;;
        2)
            run_chrono_pendulum
            pause
            ;;
        3)
            run_plot
            pause
            ;;
        4)
            run_rl_fitting
            pause
            ;;
        5)
            run_full_pipeline
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

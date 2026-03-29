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

select_json_file() {
    local label="${1:-JSON 파일}"
    echo "--------------------------------" >&2
    echo "[INFO] ${label} 선택" >&2

    local files=("$BASE_DIR"/run_logs/*.json "$BASE_DIR"/rl_results/*.json)
    local valid=()
    for f in "${files[@]}"; do
        if [ ! -f "$f" ]; then
            continue
        fi
        if [[ "$f" == *.meta.json ]]; then
            continue
        fi
        if [ -f "$f" ]; then
            valid+=("$f")
        fi
    done

    if [ "${#valid[@]}" -eq 0 ]; then
        echo "[WARN] 선택 가능한 json 파일이 없습니다. 없음으로 진행합니다." >&2
        echo ""
        return 0
    fi

    local options=("없음 (JSON 미적용)")
    options+=("${valid[@]}")
    select file in "${options[@]}"; do
        if [ "$REPLY" == "1" ]; then
            echo "[INFO] Selected JSON: 없음" >&2
            echo ""
            return 0
        elif [ -n "$file" ]; then
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
    python3 "$BASE_DIR/imu_viewer.py"
}

run_chrono_pendulum() {
    echo "--------------------------------"
    echo "Select mode:"
    echo "1) Host mode (keyboard control)"
    echo "2) Jetson mode (ROS input)"
    read -p "Enter number: " mode

    param_json=$(select_json_file "Model Parameter JSON")

    calib_json=$(select_json_file "Calibration JSON")

    echo "--------------------------------"
    echo "Self-fitting mode:"
    echo "1) ON  (online parameter fitting)"
    echo "2) OFF (pure simulation)"
    read -p "Enter number: " fit_mode_choice
    if [ "$fit_mode_choice" == "1" ]; then
        self_fit_mode="on"
    elif [ "$fit_mode_choice" == "2" ]; then
        self_fit_mode="off"
    else
        echo "[ERROR] Invalid self-fitting selection"
        return
    fi

    if [ "$mode" == "1" ]; then
        echo "[INFO] chrono_pendulum (HOST mode)"
        cmd=(python3 "$BASE_DIR/chrono_pendulum.py" --mode host --self-fit "$self_fit_mode")
        [ -n "$param_json" ] && cmd+=(--calibration-json "$param_json")
        [ -n "$calib_json" ] && cmd+=(--radius-json "$calib_json")
        "${cmd[@]}"
    elif [ "$mode" == "2" ]; then
        echo "[INFO] chrono_pendulum (JETSON mode)"
        cmd=(python3 "$BASE_DIR/chrono_pendulum.py" --mode jetson --self-fit "$self_fit_mode")
        [ -n "$param_json" ] && cmd+=(--calibration-json "$param_json")
        [ -n "$calib_json" ] && cmd+=(--radius-json "$calib_json")
        "${cmd[@]}"
    else
        echo "[ERROR] Invalid selection"
    fi
}

run_system_identification() {
    echo "--------------------------------"
    echo "[INFO] Calibration 시작"
    python3 $BASE_DIR/calibration.py
}

run_plot() {
    file=$(select_csv_file)
    if [ -z "$file" ]; then return; fi

    python3 "$BASE_DIR/plot_pendulum.py" --csv "$file"
}

run_rl_fitting() {
    file=$(select_csv_file)
    if [ -z "$file" ]; then return; fi

    calib_json=$(select_json_file "Calibration JSON (required)")
    if [ -z "$calib_json" ]; then
        if [ -f "$BASE_DIR/run_logs/calibration_latest.json" ]; then
            calib_json="$BASE_DIR/run_logs/calibration_latest.json"
            echo "[INFO] calibration json fallback: $calib_json"
        else
            echo "[ERROR] calibration json이 필요합니다."
            return
        fi
    fi
    param_json=$(select_json_file "Parameter JSON (optional)")

    echo "--------------------------------"
    echo "[INFO] Offline RL replay calibration 실행 (train_pendulum_rl.py)"
    cmd=(python3 "$BASE_DIR/train_pendulum_rl.py" --calibration_json "$calib_json" --csv "$file" --outdir "$BASE_DIR/rl_results" --renderOFF)
    [ -n "$param_json" ] && cmd+=(--parameter_json "$param_json")
    "${cmd[@]}"
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

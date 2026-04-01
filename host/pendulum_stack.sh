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

    local search_dirs=("$BASE_DIR/run_logs" "$BASE_DIR/rl_results")
    local files=()
    local d
    for d in "${search_dirs[@]}"; do
        if [ -d "$d" ]; then
            while IFS= read -r -d '' f; do
                files+=("$f")
            done < <(find "$d" -type f -name "*.csv" -print0 2>/dev/null)
        fi
    done

    if [ "${#files[@]}" -eq 0 ]; then
        echo "[ERROR] 선택 가능한 CSV가 없습니다 (run_logs, rl_results 확인)." >&2
        return 1
    fi

    IFS=$'\n' files=($(printf "%s\n" "${files[@]}" | sort))
    unset IFS

    select file in "${files[@]}"; do
        if [ -n "$file" ]; then
            echo "[INFO] Selected: $file" >&2
            echo "$file"
            return 0
        else
            echo "[ERROR] 잘못된 선택" >&2
        fi
    done
}

select_plot_csv_file() {
    echo "--------------------------------" >&2
    echo "[INFO] Plot 가능한 CSV 선택 (chrono/replay 형식만 표시)" >&2

    local search_dirs=("$BASE_DIR/run_logs" "$BASE_DIR/rl_results")
    local files=()
    local d
    for d in "${search_dirs[@]}"; do
        if [ -d "$d" ]; then
            while IFS= read -r -d '' f; do
                local bn
                bn=$(basename "$f")
                # Exclude SB3/metrics CSV that do not match chrono_run-style schema.
                if [[ "$bn" == "sb3_monitor.csv" || "$bn" == "history.csv" ]]; then
                    continue
                fi
                files+=("$f")
            done < <(find "$d" -type f -name "*.csv" -print0 2>/dev/null)
        fi
    done

    if [ "${#files[@]}" -eq 0 ]; then
        echo "[ERROR] plot 가능한 CSV가 없습니다." >&2
        return 1
    fi

    IFS=$'\n' files=($(printf "%s\n" "${files[@]}" | sort))
    unset IFS

    select file in "${files[@]}"; do
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

    local valid=()
    local search_dirs=("$BASE_DIR/run_logs" "$BASE_DIR/rl_results")
    local d
    for d in "${search_dirs[@]}"; do
        if [ ! -d "$d" ]; then
            continue
        fi
        while IFS= read -r -d '' f; do
            if [[ "$f" == *.meta.json ]]; then
                continue
            fi
            valid+=("$f")
        done < <(find "$d" -type f -name "*.json" -print0 2>/dev/null)
    done
    if [ "${#valid[@]}" -gt 0 ]; then
        IFS=$'\n' valid=($(printf "%s\n" "${valid[@]}" | sort))
        unset IFS
    fi

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
    echo "--------------------------------"
    echo "Real alpha source is fixed: filtered derivative (omega_diff)"
    alpha_arg="omega_diff"

    param_json=$(select_json_file "Model Parameter JSON")

    calib_json=$(select_json_file "Calibration JSON")

    if [ "$mode" == "1" ]; then
        echo "[INFO] chrono_pendulum (HOST mode)"
        cmd=(python3 "$BASE_DIR/chrono_pendulum.py" --mode host --real-alpha-source "$alpha_arg")
        [ -n "$param_json" ] && cmd+=(--parameter-json "$param_json")
        [ -n "$calib_json" ] && cmd+=(--calibration-json "$calib_json" --radius-json "$calib_json")
        "${cmd[@]}"
    elif [ "$mode" == "2" ]; then
        echo "[INFO] chrono_pendulum (JETSON mode)"
        cmd=(python3 "$BASE_DIR/chrono_pendulum.py" --mode jetson --real-alpha-source "$alpha_arg")
        [ -n "$param_json" ] && cmd+=(--parameter-json "$param_json")
        [ -n "$calib_json" ] && cmd+=(--calibration-json "$calib_json" --radius-json "$calib_json")
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
    file=$(select_plot_csv_file)
    if [ -z "$file" ]; then return; fi

    python3 "$BASE_DIR/plot_pendulum.py" --csv "$file"
}

run_rl_fitting() {
    echo "--------------------------------"
    echo "Replay dataset:"
    echo "1) Single CSV"
    echo "2) All CSVs in run_logs (csv_dir)"
    read -p "Enter number [1]: " data_mode
    data_mode=${data_mode:-1}

    file=""
    use_csv_dir=0
    if [ "$data_mode" == "2" ]; then
        use_csv_dir=1
        echo "[INFO] using --csv_dir $CSV_DIR"
    else
        file=$(select_csv_file)
        if [ -z "$file" ]; then return; fi
    fi

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
    echo "[INFO] Parameter Finetuning (Reinforcement Learning) 실행 (train_pendulum_rl.py)"
    read -p "num_episodes [1000]: " num_episodes
    num_episodes=${num_episodes:-1000}
    read -p "batch_size [20]: " batch_size
    batch_size=${batch_size:-20}
    read -p "seed [auto]: " seed
    if [ -z "$seed" ]; then
        seed=$(date +%s)
        echo "[INFO] auto seed: $seed"
    fi

    read -p "domain_randomization ON? (y/n) [y]: " dr_yn
    dr_yn=${dr_yn:-y}
    read -p "aggressive_search ON? (y/n) [n]: " aggr_yn
    aggr_yn=${aggr_yn:-n}

    run_id=$(date +"run_%Y%m%d_%H%M%S")
    run_outdir="$BASE_DIR/rl_results/runs/$run_id"
    latest_link="$BASE_DIR/rl_results/latest"

    cmd=(python3 "$BASE_DIR/train_pendulum_rl.py" --calibration_json "$calib_json" --outdir "$run_outdir" --num_episodes "$num_episodes" --batch_size "$batch_size" --seed "$seed")
    if [ "$use_csv_dir" == "1" ]; then
        cmd+=(--csv_dir "$CSV_DIR")
    else
        cmd+=(--csv "$file")
    fi
    [ -n "$param_json" ] && cmd+=(--parameter_json "$param_json")
    if [[ "$dr_yn" =~ ^[Yy]$ ]]; then
        cmd+=(--domain_randomizationON)
    else
        cmd+=(--domain_randomizationOFF)
    fi
    if [[ "$aggr_yn" =~ ^[Yy]$ ]]; then
        cmd+=(--aggressive_search)
    fi
    echo "[INFO] command: ${cmd[*]}"
    "${cmd[@]}"

    mkdir -p "$BASE_DIR/rl_results"
    ln -sfn "$run_outdir" "$latest_link"
    echo "[INFO] latest -> $run_outdir"

    if [ -f "$run_outdir/rl_dashboard.png" ]; then
        echo "[INFO] RL dashboard: $run_outdir/rl_dashboard.png"
    fi
    if [ -f "$run_outdir/replay_best.csv" ]; then
        echo "[INFO] replay export CSV: $run_outdir/replay_best.csv"
        echo "[INFO] replay plot: python3 $BASE_DIR/plot_pendulum.py --csv $run_outdir/replay_best.csv"
    fi
    if [ -f "$run_outdir/chrono_run_best.csv" ]; then
        echo "[INFO] chrono_run-style export CSV: $run_outdir/chrono_run_best.csv"
        echo "[INFO] replay plot: python3 $BASE_DIR/plot_pendulum.py --csv $run_outdir/chrono_run_best.csv"
    fi
    if [ -f "$run_outdir/history.csv" ]; then
        echo "[INFO] RL+replay one-window plot: python3 $BASE_DIR/plot_pendulum.py --rl-dir $run_outdir"
    fi
    if [ -f "$run_outdir/sb3_monitor.csv" ]; then
        echo "[INFO] SB3 monitor plot (rl_zoo3): python -m rl_zoo3.plots.plot_train --algo ppo --env PendulumReplayEnv-v0 -f $run_outdir"
    fi
    echo "[INFO] TensorBoard: tensorboard --logdir $run_outdir/tensorboard"
}

run_staged_calibration() {
    echo "--------------------------------"
    echo "[INFO] Trajectory-level Physical System Identification 실행"
    echo "[INFO] (K_u, l_com, b_eq, tau_eq stage-wise calibration)"
    echo "Select staged calibration mode:"
    echo "1) stage1  (sin only: optimize K_u, l_com)"
    echo "2) stage12 (sin+square: + optimize b_eq)"
    echo "3) full    (sin+square+burst: + optimize tau_eq)"
    read -p "Enter number [3]: " mode_sel
    mode_sel=${mode_sel:-3}

    local mode_arg="full"
    case "$mode_sel" in
        1) mode_arg="stage1" ;;
        2) mode_arg="stage12" ;;
        3) mode_arg="full" ;;
        *)
            echo "[WARN] 잘못된 입력이어서 full 모드로 진행합니다."
            mode_arg="full"
            ;;
    esac

    cmd=(python3 "$BASE_DIR/staged_pendulum_calibration.py" --interactive --run-logs "$CSV_DIR" --mode "$mode_arg")
    echo "[INFO] command: ${cmd[*]}"
    "${cmd[@]}"
    echo "[INFO] staged calibration 완료. 결과 확인:"
    echo "       - $CSV_DIR/trajectory_model_params.json"
    echo "       - $CSV_DIR/trajectory_fit_summary.json"
}

run_replay_validation() {
    echo "--------------------------------"
    echo "[INFO] Replay Runs"
    file=$(select_csv_file)
    if [ -z "$file" ]; then return; fi

    calib_json=""
    if [ -f "$BASE_DIR/run_logs/calibration_latest.json" ]; then
        calib_json="$BASE_DIR/run_logs/calibration_latest.json"
        echo "[INFO] Auto calibration JSON: $calib_json"
    fi

    param_json=""
    csv_base="${file%.csv}"
    if [ -f "${csv_base}_best_param.json" ]; then
        param_json="${csv_base}_best_param.json"
    elif [ -f "$BASE_DIR/rl_results/latest/final_params_rl.json" ]; then
        param_json="$BASE_DIR/rl_results/latest/final_params_rl.json"
    elif [ -f "$BASE_DIR/rl_results/latest/best_params.json" ]; then
        param_json="$BASE_DIR/rl_results/latest/best_params.json"
    fi
    if [ -n "$param_json" ]; then
        echo "[INFO] Auto parameter JSON: $param_json"
    else
        echo "[INFO] Auto parameter JSON: 없음"
    fi

    read -p "Replay speed [1.0]: " replay_speed
    replay_speed=${replay_speed:-1.0}

    cmd=(python3 "$BASE_DIR/replay_pendulum_cli.py" --csv "$file" --speed "$replay_speed")
    [ -n "$calib_json" ] && cmd+=(--calibration_json "$calib_json")
    [ -n "$param_json" ] && cmd+=(--parameter_json "$param_json")

    echo "[INFO] command: ${cmd[*]}"
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
    echo "2) Model Calibration (radius, IMU gravity)"
    echo "3) Parameter Finetuning (SB3 PPO)"
    echo "4) Chrono Pendulum (Select Host/Jetson mode)"
    echo "5) Plot Data (Sim vs Real)"
    echo "6) Replay Runs"
    echo "7) System Identification (Physical Params, Stage-wise)"
    echo "8) Exit"
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
            run_replay_validation
            pause
            ;;
        7)
            run_staged_calibration
            pause
            ;;
        8)
            echo "Bye!"
            exit 0
            ;;
        *)
            echo "[ERROR] Invalid input"
            pause
            ;;
    esac
done

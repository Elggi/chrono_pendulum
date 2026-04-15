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
REPORTS_STAGE1_DIR="$BASE_DIR/../reports/Stage1_CMAES"

# ======================================================
# 유틸
# ======================================================

pause() {
    read -p "Press Enter to continue..."
}

select_csv_file() {
    echo "--------------------------------" >&2
    echo "[INFO] CSV 파일 선택" >&2

    local search_dirs=("$BASE_DIR" "$BASE_DIR/run_logs" "$BASE_DIR/rl_results" "$REPORTS_STAGE1_DIR")
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

    IFS=$'\n' files=($(printf "%s\n" "${files[@]}" | sort -u))
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

select_csv_files_multi() {
    echo "--------------------------------" >&2
    echo "[INFO] 학습용 CSV 파일들 선택 (복수 선택 가능)" >&2

    local search_dirs=("$BASE_DIR" "$BASE_DIR/run_logs" "$BASE_DIR/rl_results" "$REPORTS_STAGE1_DIR")
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
        echo "[ERROR] 선택 가능한 CSV가 없습니다." >&2
        return 1
    fi

    IFS=$'\n' files=($(printf "%s\n" "${files[@]}" | sort -u))
    unset IFS

    local i=1
    for f in "${files[@]}"; do
        echo "  [$i] $f" >&2
        i=$((i + 1))
    done

    local default_sel="1"
    read -p "선택 번호들 (공백구분) [${default_sel}]: " sel
    sel=${sel:-$default_sel}

    local out=()
    local idx
    for idx in $sel; do
        if [[ "$idx" =~ ^[0-9]+$ ]] && [ "$idx" -ge 1 ] && [ "$idx" -le "${#files[@]}" ]; then
            out+=("${files[$((idx - 1))]}")
        fi
    done

    if [ "${#out[@]}" -eq 0 ]; then
        echo "[ERROR] 유효한 CSV 선택이 없습니다." >&2
        return 1
    fi

    printf "%s\n" "${out[@]}"
}

select_plot_csv_file() {
    echo "--------------------------------" >&2
    echo "[INFO] Plot 가능한 CSV 선택 (chrono/replay 형식만 표시)" >&2

    local search_dirs=("$BASE_DIR" "$BASE_DIR/run_logs" "$BASE_DIR/rl_results" "$REPORTS_STAGE1_DIR")
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

    IFS=$'\n' files=($(printf "%s\n" "${files[@]}" | sort -u))
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
    local search_dirs=("$BASE_DIR" "$BASE_DIR/run_logs" "$BASE_DIR/rl_results" "$REPORTS_STAGE1_DIR")
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
        IFS=$'\n' valid=($(printf "%s\n" "${valid[@]}" | sort -u))
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
    read -p "Initial angle theta0 [deg] (+CCW, -CW) [0]: " theta0_deg
    theta0_deg=${theta0_deg:-0}
    if ! [[ "$theta0_deg" =~ ^[+-]?[0-9]+([.][0-9]+)?$ ]]; then
        echo "[WARN] Invalid theta0 input '$theta0_deg'. Fallback to 0 deg."
        theta0_deg=0
    fi
    echo "theta0_deg = $theta0_deg (+:CCW, -:CW)"
    read -p "Enable Chrono free-decay startup mode? (y/n) [n]: " free_decay_mode_yn
    free_decay_mode_yn=${free_decay_mode_yn:-n}
    free_decay_args=()
    if [[ "$free_decay_mode_yn" =~ ^[Yy]$ ]]; then
        read -p "free-decay arm minimum angle [deg] (default: 5.0): " arm_min_angle_deg
        arm_min_angle_deg=${arm_min_angle_deg:-5.0}
        if ! [[ "$arm_min_angle_deg" =~ ^[+-]?[0-9]+([.][0-9]+)?$ ]]; then
            echo "[WARN] Invalid arm minimum angle '$arm_min_angle_deg'. Fallback to 5.0 deg."
            arm_min_angle_deg=5.0
        fi
        free_decay_args+=(--enable-free-decay-mode --free-decay-arm-min-angle-deg "$arm_min_angle_deg")
    fi
    echo "--------------------------------"
    echo "Select mode:"
    echo "1) Host mode (keyboard control)"
    echo "2) Jetson mode (ROS input)"
    read -p "Enter number: " mode
    echo "--------------------------------"
    echo "Finalized alpha source for ID: tangential accel / radius (gravity compensated)"

    param_json=$(select_json_file "Model Parameter JSON")

    calib_json=$(select_json_file "Calibration JSON")

    if [ "$mode" == "1" ]; then
        echo "[INFO] chrono_pendulum (HOST mode)"
        cmd=(python3 "$BASE_DIR/chrono_pendulum.py" --mode host --theta0-deg "$theta0_deg")
        cmd+=("${free_decay_args[@]}")
        [ -n "$param_json" ] && cmd+=(--parameter-json "$param_json")
        [ -n "$calib_json" ] && cmd+=(--calibration-json "$calib_json" --radius-json "$calib_json")
        "${cmd[@]}"
    elif [ "$mode" == "2" ]; then
        echo "[INFO] chrono_pendulum (JETSON mode)"
        cmd=(python3 "$BASE_DIR/chrono_pendulum.py" --mode jetson --theta0-deg "$theta0_deg")
        cmd+=("${free_decay_args[@]}")
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
    read -p "collect free-decay data[y]/n? : " free_decay_yn
    free_decay_yn=${free_decay_yn:-y}
    if [[ "$free_decay_yn" =~ ^[Yy]$ ]]; then
        read -p "arm minimum angle [deg] (default: 5.0): " arm_min_angle_deg
        arm_min_angle_deg=${arm_min_angle_deg:-5.0}
        if ! [[ "$arm_min_angle_deg" =~ ^[+-]?[0-9]+([.][0-9]+)?$ ]]; then
            echo "[WARN] Invalid arm minimum angle '$arm_min_angle_deg'. Fallback to 5.0 deg."
            arm_min_angle_deg=5.0
        fi
        echo "[INFO] free-decay mode 실행 (arm_min_angle_deg=${arm_min_angle_deg})"
        python3 "$BASE_DIR/calibration.py" --mode free_decay --free-decay-arm-min-angle-deg "$arm_min_angle_deg"
    else
        echo "[INFO] manual CPR/r calibration 실행"
        python3 "$BASE_DIR/calibration.py"
    fi
}

run_stage1_pem_identification() {
    echo "--------------------------------"
    echo "[INFO] Stage1 Identification 실행 (CMA-ES + Headless Chrono)"
    mapfile -t selected_csvs < <(select_csv_files_multi)
    if [ "${#selected_csvs[@]}" -eq 0 ]; then
        echo "[ERROR] 선택된 CSV가 없습니다." >&2
        return
    fi
    default_calib="$BASE_DIR/run_logs/calibration_latest.json"
    default_model_param="$BASE_DIR/model_parameter.json"
    if [ ! -f "$default_model_param" ]; then
        default_model_param="$BASE_DIR/model_parameter.template.json"
    fi
    read -p "Calibration JSON [${default_calib}]: " calib_json
    calib_json=${calib_json:-$default_calib}
    read -p "Model Parameter JSON [${default_model_param}]: " model_param_json
    model_param_json=${model_param_json:-$default_model_param}
    read -p "출력 폴더 [${BASE_DIR}/../reports/Stage1_CMAES]: " outdir
    outdir=${outdir:-$BASE_DIR/../reports/Stage1_CMAES}
    read -p "세대 수 [30]: " gens
    gens=${gens:-30}
    read -p "병렬 worker 수 [8]: " workers
    workers=${workers:-8}
    read -p "theta RMSE weight (w_theta) [default: 1.0]: " w_theta
    w_theta=${w_theta:-1.0}
    read -p "omega RMSE weight (w_omega) [default: 0.1]: " w_omega
    w_omega=${w_omega:-0.1}
    echo "[INFO] Stage1 weighted RMSE loss: w_theta=${w_theta}, w_omega=${w_omega}"
    local csv_args=("${selected_csvs[@]}")
    cmd=(python3 "$BASE_DIR/stage1_cmaes_chrono.py" --csv "${csv_args[@]}" --calibration-json "$calib_json" --model-parameter-json "$model_param_json" --outdir "$outdir" --max-generations "$gens" --workers "$workers" --w-theta "$w_theta" --w-omega "$w_omega")
    echo "[INFO] command: ${cmd[*]}"
    "${cmd[@]}"
}

run_stage2_sindy_identification() {
    echo "--------------------------------"
    echo "[INFO] Stage2 Greybox Residual-Torque SINDy 실행"
    mapfile -t selected_csvs < <(select_csv_files_multi)
    if [ "${#selected_csvs[@]}" -eq 0 ]; then
        echo "[ERROR] 선택된 CSV가 없습니다." >&2
        return
    fi
    default_model_param="$BASE_DIR/model_parameter.json"
    if [ ! -f "$default_model_param" ]; then
        default_model_param="$BASE_DIR/model_parameter.template.json"
    fi
    read -p "Model Parameter JSON [${default_model_param}]: " model_param_json
    model_param_json=${model_param_json:-$default_model_param}
    read -p "출력 폴더 [${BASE_DIR}/../reports/SINDy_stage2]: " outdir
    outdir=${outdir:-$BASE_DIR/../reports/SINDy_stage2}
    read -p "sparsity threshold [1e-4]: " threshold
    threshold=${threshold:-1e-4}
    read -p "feature set (comma-separated) [1,theta,omega,sin_theta,cos_theta,theta2,omega2,motor_input]: " feature_set
    feature_set=${feature_set:-1,theta,omega,sin_theta,cos_theta,theta2,omega2,motor_input}
    local csv_args=("${selected_csvs[@]}")
    cmd=(python3 "$BASE_DIR/stage2_sindy_entry.py" --csv "${csv_args[@]}" --model-parameter-json "$model_param_json" --outdir "$outdir" --threshold "$threshold" --features "$feature_set")
    echo "[INFO] command: ${cmd[*]}"
    "${cmd[@]}"
}

run_stage3_ppo_identification() {
    echo "--------------------------------"
    echo "[INFO] Stage3 PPO Identification 실행"
    default_csv="$BASE_DIR/run_logs/chrono_run_1.finalized.csv"
    default_meta="$BASE_DIR/run_logs/chrono_run_1.meta.json"
    read -p "CSV 경로 [${default_csv}]: " csv_path
    csv_path=${csv_path:-$default_csv}
    read -p "META JSON 경로 [${default_meta}]: " meta_path
    meta_path=${meta_path:-$default_meta}
    read -p "출력 폴더 [${BASE_DIR}/../reports/PPO_stage3]: " outdir
    outdir=${outdir:-$BASE_DIR/../reports/PPO_stage3}
    read -p "PPO timesteps [12000]: " ppo_steps
    ppo_steps=${ppo_steps:-12000}
    cmd=(python3 "$BASE_DIR/stage3_ppo_entry.py" --csv "$csv_path" --meta "$meta_path" --outdir "$outdir" --ppo-steps "$ppo_steps")
    echo "[INFO] command: ${cmd[*]}"
    "${cmd[@]}"
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
    read -p "학습 후 chrono_pendulum.py 프로세스 실행? (y/n) [n]: " launch_chrono_yn
    launch_chrono_yn=${launch_chrono_yn:-n}
    if [[ "$launch_chrono_yn" =~ ^[Yy]$ ]]; then
        read -p "chrono duration sec [8.0]: " chrono_dur
        chrono_dur=${chrono_dur:-8.0}
        cmd+=(--launch-chrono-after --chrono-duration "$chrono_dur")
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

    read -p "학습 결과로 Replay 3D + IMU Viewer 실행? (y/n) [y]: " replay_after_yn
    replay_after_yn=${replay_after_yn:-y}
    if [[ "$replay_after_yn" =~ ^[Yy]$ ]]; then
        replay_csv="$run_outdir/chrono_run_best.csv"
        if [ ! -f "$replay_csv" ]; then
            replay_csv="$run_outdir/replay_best.csv"
        fi
        if [ ! -f "$replay_csv" ]; then
            echo "[WARN] replay csv를 찾지 못해 수동 선택으로 진행합니다."
            replay_csv=$(select_csv_file)
        fi
        if [ -n "$replay_csv" ]; then
            replay_cmd=(python3 "$BASE_DIR/replay_pendulum_cli.py" --csv "$replay_csv" --speed 1.0)
            [ -f "$BASE_DIR/run_logs/calibration_latest.json" ] && replay_cmd+=(--calibration_json "$BASE_DIR/run_logs/calibration_latest.json")
            [ -f "$run_outdir/final_params_rl.json" ] && replay_cmd+=(--parameter_json "$run_outdir/final_params_rl.json")
            echo "[INFO] replay command: ${replay_cmd[*]}"
            "${replay_cmd[@]}"
        fi
    fi
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

    read -p "보정 결과로 Replay 3D + IMU Viewer 실행? (y/n) [y]: " replay_after_yn
    replay_after_yn=${replay_after_yn:-y}
    if [[ "$replay_after_yn" =~ ^[Yy]$ ]]; then
        replay_csv=$(select_csv_file)
        if [ -n "$replay_csv" ]; then
            replay_cmd=(python3 "$BASE_DIR/replay_pendulum_cli.py" --csv "$replay_csv" --speed 1.0)
            [ -f "$BASE_DIR/run_logs/calibration_latest.json" ] && replay_cmd+=(--calibration_json "$BASE_DIR/run_logs/calibration_latest.json")
            [ -f "$CSV_DIR/trajectory_model_params.json" ] && replay_cmd+=(--parameter_json "$CSV_DIR/trajectory_model_params.json")
            echo "[INFO] replay command: ${replay_cmd[*]}"
            "${replay_cmd[@]}"
        fi
    fi
}

run_offline_benchmark_pem_sindy_ppo() {
    echo "--------------------------------"
    echo "[INFO] Offline Identification Benchmark (PEM + SINDy-PI + PPO)"
    echo "[INFO] 기본 데이터셋: host/run_logs/chrono_run_1.finalized.csv + chrono_run_1.meta.json"

    default_csv="$BASE_DIR/run_logs/chrono_run_1.finalized.csv"
    default_meta="$BASE_DIR/run_logs/chrono_run_1.meta.json"

    if [ ! -f "$default_csv" ]; then
        echo "[ERROR] default csv 없음: $default_csv"
        return
    fi
    if [ ! -f "$default_meta" ]; then
        echo "[ERROR] default meta 없음: $default_meta"
        return
    fi

    read -p "CSV 경로 [${default_csv}]: " csv_path
    csv_path=${csv_path:-$default_csv}
    read -p "META JSON 경로 [${default_meta}]: " meta_path
    meta_path=${meta_path:-$default_meta}
    read -p "출력 폴더 [${BASE_DIR}/../reports/PEM_SINDy_PPO]: " outdir
    outdir=${outdir:-$BASE_DIR/../reports/PEM_SINDy_PPO}
    read -p "train_ratio [0.75]: " train_ratio
    train_ratio=${train_ratio:-0.75}
    read -p "fit l_com도 함께 식별? (y/n) [n]: " fit_lcom_yn
    fit_lcom_yn=${fit_lcom_yn:-n}
    read -p "Stage2(SINDy-PI) skip? (y/n) [n]: " skip_s2_yn
    skip_s2_yn=${skip_s2_yn:-n}
    read -p "Stage3(PPO) skip? (y/n) [n]: " skip_s3_yn
    skip_s3_yn=${skip_s3_yn:-n}
    read -p "PPO timesteps [12000]: " ppo_steps
    ppo_steps=${ppo_steps:-12000}

    cmd=(python3 "$BASE_DIR/offline_id_pem_sindy_ppo.py" --csv "$csv_path" --meta "$meta_path" --outdir "$outdir" --train-ratio "$train_ratio" --ppo-steps "$ppo_steps")
    if [[ "$fit_lcom_yn" =~ ^[Yy]$ ]]; then
        cmd+=(--fit-lcom)
    fi
    if [[ "$skip_s2_yn" =~ ^[Yy]$ ]]; then
        cmd+=(--skip-stage2)
    fi
    if [[ "$skip_s3_yn" =~ ^[Yy]$ ]]; then
        cmd+=(--skip-stage3)
    fi

    echo "[INFO] command: ${cmd[*]}"
    "${cmd[@]}"

    echo "[INFO] 완료. 주요 아티팩트:"
    echo "       - $outdir/config_used.json"
    echo "       - $outdir/stage1_result.json"
    echo "       - $outdir/stage2_result.json (if enabled)"
    echo "       - $outdir/stage3_result.json (if enabled)"
    echo "       - $outdir/benchmark_report.md"
    echo "       - $outdir/stage1_trajectories.png"
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
    elif [ -f "$BASE_DIR/model_parameter.json" ]; then
        param_json="$BASE_DIR/model_parameter.json"
    elif [ -f "$BASE_DIR/model_parameter.template.json" ]; then
        param_json="$BASE_DIR/model_parameter.template.json"
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
    echo "3) Stage1 Identification (CMA-ES Headless)"
    echo "4) Stage2 Identification (SINDy)"
    echo "5) Stage3 Identification (PPO)"
    echo "6) Chrono Pendulum (Select Host/Jetson mode)"
    echo "7) Plot Data (Sim vs Real)"
    echo "8) Replay Runs"
    echo "9) Offline Benchmark (PEM+SINDy-PI+PPO)"
    echo "10) Exit"
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
            run_stage1_pem_identification
            pause
            ;;
        4)
            run_stage2_sindy_identification
            pause
            ;;
        5)
            run_stage3_ppo_identification
            pause
            ;;
        6)
            run_chrono_pendulum
            pause
            ;;
        7)
            run_plot
            pause
            ;;
        8)
            run_replay_validation
            pause
            ;;
        9)
            run_offline_benchmark_pem_sindy_ppo
            pause
            ;;
        10)
            echo "Bye!"
            exit 0
            ;;
        *)
            echo "[ERROR] Invalid input"
            pause
            ;;
    esac
done

#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR"
RUN_DIR="$BASE_DIR/run_logs"
MOTOR_JSON_DEFAULT="$BASE_DIR/motor_torque.json"
CALIB_JSON_DEFAULT="$RUN_DIR/calibration_latest.json"

pause(){ read -p "Press Enter to continue..."; }
choose_csv(){
  mapfile -t files < <(find "$RUN_DIR" -type f -name "*.csv" | sort)
  [ ${#files[@]} -eq 0 ] && echo "" && return
  select f in "${files[@]}"; do echo "$f"; return; done
}
choose_multi_csv(){
  echo "Enter CSV paths separated by spaces (leave empty to pick one interactively):"
  read -r line
  if [ -n "$line" ]; then echo "$line"; else one=$(choose_csv); echo "$one"; fi
}

while true; do
  clear
  cat <<EOF
Pendulum Digital Twin Stack

1. IMU Viewer (RViz)
2. Model Calibration (radius, encoder CPR, IMU gravity)
3. COM calculator (geometry only)
4. Chrono Pendulum (Select Host/Jetson mode)
5. Stage 1: Regression
6. Stage 2: SINDy
7. Stage 3: SB3 PPO
8. Blackbox Identification: SINDy
9. Plot Data (Sim vs Real)
10. Replay Runs
11. NN Dynamics Test (MBD-NODE style)
12. Exit
==========================================
EOF
  read -p "Select option: " opt
  case "$opt" in
    1) python3 "$BASE_DIR/rviz_calibration_node.py"; pause ;;
    2) python3 "$BASE_DIR/calibration.py"; pause ;;
    3)
      read -p "motor_torque.json path [$MOTOR_JSON_DEFAULT]: " mjson
      mjson=${mjson:-$MOTOR_JSON_DEFAULT}
      python3 "$BASE_DIR/inertia_com_calculator.py" --motor_torque_json "$mjson"
      pause ;;
    4)
      read -p "Mode host/jetson/offline/live [host]: " mode
      mode=${mode:-host}
      replay_arg=(); host_arg=(); extra=()
      if [ "$mode" = "host" ]; then
        read -p "Host input source ros/synthetic/replay [ros]: " host_input
        host_input=${host_input:-ros}
        host_arg=(--host_input "$host_input")
        if [ "$host_input" = "replay" ]; then csv=$(choose_csv); [ -n "$csv" ] && replay_arg=(--replay_csv "$csv"); fi
        read -p "Free decay capture with 1s warmup sync? y/n [n]: " fd
        if [[ "$fd" =~ ^[Yy]$ ]]; then extra+=(--free_decay_capture --warmup_sec 1.0); fi
      fi
      if [ "$mode" = "offline" ]; then csv=$(choose_csv); [ -n "$csv" ] && replay_arg=(--replay_csv "$csv"); fi
      python3 "$BASE_DIR/chrono_pendulum.py" --mode "$mode" --calibration_json "$CALIB_JSON_DEFAULT" --motor_torque_json "$MOTOR_JSON_DEFAULT" "${host_arg[@]}" "${replay_arg[@]}" "${extra[@]}"
      pause ;;
    5)
      read -p "calibration json [$CALIB_JSON_DEFAULT]: " calib; calib=${calib:-$CALIB_JSON_DEFAULT}
      read -p "motor_torque json [$MOTOR_JSON_DEFAULT]: " mjson; mjson=${mjson:-$MOTOR_JSON_DEFAULT}
      csvs=$(choose_multi_csv)
      read -p "Mode free_decay/driven_current/staged_auto [staged_auto]: " rid_mode; rid_mode=${rid_mode:-staged_auto}
      # shellcheck disable=SC2086
      python3 "$BASE_DIR/stage1_regression.py" --calibration_json "$calib" --motor_torque_json "$mjson" --mode "$rid_mode" --csvs $csvs
      pause ;;
    6)
      csv=$(choose_csv)
      [ -n "$csv" ] && python3 "$BASE_DIR/stage2_sindy.py" --csvs "$csv" --motor_torque_json "$MOTOR_JSON_DEFAULT" --llm_config "$BASE_DIR/llm_config.example.json"
      pause ;;
    7)
      npz_path="$RUN_DIR/pysindy_input_arrays.npz"
      if [ -f "$npz_path" ]; then
        python3 "$BASE_DIR/rl_train/train_ppo.py" --npz "$npz_path" --motor_torque_json "$MOTOR_JSON_DEFAULT" --calibration_json "$CALIB_JSON_DEFAULT"
      else
        csv=$(choose_csv)
        [ -n "$csv" ] && python3 "$BASE_DIR/rl_train/train_ppo.py" --csvs "$csv" --motor_torque_json "$MOTOR_JSON_DEFAULT" --calibration_json "$CALIB_JSON_DEFAULT"
      fi
      pause ;;
    8)
      csv=$(choose_csv)
      [ -n "$csv" ] && python3 "$BASE_DIR/blackbox_sindy.py" --csvs "$csv" --motor_torque_json "$MOTOR_JSON_DEFAULT" --llm_config "$BASE_DIR/llm_config.example.json"
      pause ;;
    9)
      csv=$(choose_csv)
      [ -n "$csv" ] && python3 "$BASE_DIR/plot_pendulum.py" --csv "$csv"
      pause ;;
    10)
      csv=$(choose_csv)
      [ -n "$csv" ] && python3 "$BASE_DIR/replay_pendulum_cli.py" --csv "$csv"
      pause ;;
    11)
      read -p "Train or eval mnode? [train/eval]: " mmode
      mmode=${mmode:-train}
      npz_path="$RUN_DIR/pysindy_input_arrays.npz"
      if [ "$mmode" = "eval" ]; then
        if [ -f "$npz_path" ]; then
          python3 "$BASE_DIR/mnode/eval_mnode.py" --npz "$npz_path" --motor_torque_json "$MOTOR_JSON_DEFAULT"
        else
          csv=$(choose_csv); [ -n "$csv" ] && python3 "$BASE_DIR/mnode/eval_mnode.py" --csvs "$csv" --motor_torque_json "$MOTOR_JSON_DEFAULT"
        fi
      else
        if [ -f "$npz_path" ]; then
          python3 "$BASE_DIR/mnode/train_mnode.py" --npz "$npz_path" --motor_torque_json "$MOTOR_JSON_DEFAULT"
        else
          csv=$(choose_csv); [ -n "$csv" ] && python3 "$BASE_DIR/mnode/train_mnode.py" --csvs "$csv" --motor_torque_json "$MOTOR_JSON_DEFAULT"
        fi
      fi
      pause ;;
    12) echo "Bye"; exit 0 ;;
    *) echo "Invalid"; pause ;;
  esac
done

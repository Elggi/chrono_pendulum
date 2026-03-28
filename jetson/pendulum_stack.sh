#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# =========================
# Jetson ROS stack launcher
# =========================

# ---- user-configurable ----
ROS_SETUP="/opt/ros/humble/setup.bash"
DEFAULT_WS_SETUP="/home/jetson/ros2_ws/install/setup.bash"
WS_SETUP="${WS_SETUP:-$DEFAULT_WS_SETUP}"

export ROS_DOMAIN_ID=7
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

ARDUINO_PORT_PATTERN="${ARDUINO_PORT_PATTERN:-/dev/serial/by-id/*Arduino*}"
ARDUINO_PORT_FALLBACK="${ARDUINO_PORT_FALLBACK:-/dev/ttyACM0}"
ARDUINO_BAUD="115200"

IMU_PORT_PATTERN="${IMU_PORT_PATTERN:-/dev/serial/by-id/*IMU*}"
IMU_PORT_FALLBACK="${IMU_PORT_FALLBACK:-/dev/ttyUSB0}"
IMU_BAUD="921600"

LOG_DIR="$HOME/jetson_ros_logs"
PID_DIR="$LOG_DIR/pids"

mkdir -p "$LOG_DIR" "$PID_DIR"

timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

load_env() {
    if [ ! -f "$ROS_SETUP" ]; then
        echo "[ERROR] ROS setup not found: $ROS_SETUP"
        exit 1
    fi
    source "$ROS_SETUP"

    if [ ! -f "$WS_SETUP" ]; then
        echo "[ERROR] Workspace setup not found: $WS_SETUP"
        echo "[HINT] Build the ROS workspace first, then source it:"
        echo "       cd \"$(dirname "$(dirname "$WS_SETUP")")\" && colcon build"
        echo "       source \"$WS_SETUP\""
        exit 1
    fi
    source "$WS_SETUP"
}

resolve_serial_port() {
    local label="$1"
    local pattern="$2"
    local fallback="$3"
    local resolved=""

    shopt -s nullglob
    local matches=( $pattern )
    shopt -u nullglob

    if [ "${#matches[@]}" -gt 0 ]; then
        resolved="${matches[0]}"
    else
        resolved="$fallback"
    fi

    if [ ! -e "$resolved" ]; then
        echo "[ERROR] $label serial device not found. pattern=$pattern fallback=$fallback resolved=$resolved"
        exit 1
    fi

    printf '%s' "$resolved"
}

start_process() {
    local name="$1"
    local cmd="$2"
    local log_file="$LOG_DIR/${name}.log"
    local pid_file="$PID_DIR/${name}.pid"

    if [ -f "$pid_file" ]; then
        local old_pid
        old_pid="$(cat "$pid_file" 2>/dev/null || true)"
        if [ -n "${old_pid:-}" ] && kill -0 "$old_pid" 2>/dev/null; then
            echo "[$(timestamp)] $name already running (PID=$old_pid)"
            return 0
        else
            rm -f "$pid_file"
        fi
    fi

    echo "[$(timestamp)] Starting $name ..."
    bash -lc "source '$ROS_SETUP'; source '$WS_SETUP'; export ROS_DOMAIN_ID=$ROS_DOMAIN_ID; export RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION; $cmd" \
        >"$log_file" 2>&1 &
    local pid=$!
    echo "$pid" > "$pid_file"
    sleep 1

    if kill -0 "$pid" 2>/dev/null; then
        echo "[$(timestamp)] $name started (PID=$pid)"
        echo "[$(timestamp)] log: $log_file"
    else
        echo "[$(timestamp)] Failed to start $name"
        echo "---- last log ----"
        tail -n 50 "$log_file" || true
        exit 1
    fi
}

stop_process() {
    local name="$1"
    local pid_file="$PID_DIR/${name}.pid"

    if [ ! -f "$pid_file" ]; then
        echo "[$(timestamp)] $name is not running (no pid file)"
        return 0
    fi

    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"

    if [ -z "${pid:-}" ]; then
        rm -f "$pid_file"
        echo "[$(timestamp)] $name pid file was empty, cleaned"
        return 0
    fi

    if kill -0 "$pid" 2>/dev/null; then
        echo "[$(timestamp)] Stopping $name (PID=$pid) ..."
        kill "$pid" || true
        sleep 2

        if kill -0 "$pid" 2>/dev/null; then
            echo "[$(timestamp)] Force killing $name (PID=$pid) ..."
            kill -9 "$pid" || true
        fi
    else
        echo "[$(timestamp)] $name already stopped"
    fi

    rm -f "$pid_file"
}

status_process() {
    local name="$1"
    local pid_file="$PID_DIR/${name}.pid"

    if [ ! -f "$pid_file" ]; then
        echo "$name: STOPPED"
        return 0
    fi

    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"

    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
        echo "$name: RUNNING (PID=$pid)"
    else
        echo "$name: STOPPED (stale pid file)"
    fi
}

start_nodes() {
    load_env

    local arduino_port
    local imu_port
    arduino_port="$(resolve_serial_port "Arduino" "$ARDUINO_PORT_PATTERN" "$ARDUINO_PORT_FALLBACK")"
    imu_port="$(resolve_serial_port "IMU" "$IMU_PORT_PATTERN" "$IMU_PORT_FALLBACK")"

    start_process "arduino_bridge" \
        "ros2 run hw_arduino_bridge bridge_node --ros-args -p port:=$arduino_port -p baud:=$ARDUINO_BAUD -p pwm_limit:=255.0"

    start_process "imu_node" \
        "ros2 run wheeltec_n100_imu imu_node --ros-args -p serial_port:=$imu_port -p serial_baud:=$IMU_BAUD -r imu:=/imu/data"

    echo
    echo "[$(timestamp)] All nodes started."
    echo "Use:"
    echo "  $0 monitor    # topic echo들 확인"
    echo "  $0 status     # 프로세스 상태"
    echo "  $0 logs       # 로그 tail"
    echo "  $0 stop       # 전체 종료"
    echo "  $0 controller # 컨트롤러  실행"

}

stop_nodes() {
    stop_process "imu_node"
    stop_process "arduino_bridge"
    echo "[$(timestamp)] All nodes stopped."
}

status_nodes() {
    status_process "arduino_bridge"
    status_process "imu_node"

}

logs_nodes() {
    echo "===== arduino_bridge.log ====="
    tail -n 50 "$LOG_DIR/arduino_bridge.log" 2>/dev/null || true
    echo
    echo "===== imu_node.log ====="
    tail -n 50 "$LOG_DIR/imu_node.log" 2>/dev/null || true
}

monitor_live() {
    load_env
    echo "[$(timestamp)] Starting one-line live monitor (Ctrl+C to exit)..."
    python3 "$SCRIPT_DIR/live_monitor.py"
}

diag_topic_once() {
    local topic="$1"
    local timeout_sec="${2:-2}"
    local out
    out="$(timeout "${timeout_sec}s" ros2 topic echo --once "$topic" 2>&1 || true)"
    if echo "$out" | rg -q "data:|header:|orientation:|angular_velocity:|linear_acceleration:"; then
        echo "[OK] topic has data: $topic"
        return 0
    fi
    echo "[WARN] no sample received from $topic (timeout=${timeout_sec}s)"
    return 1
}

diag_topic_pub() {
    local topic="$1"
    local info
    info="$(ros2 topic info "$topic" 2>&1 || true)"
    local pubs
    pubs="$(echo "$info" | awk -F': ' '/Publisher count/ {print $2}' | tr -d '\r' | tail -n 1)"
    if [ -n "${pubs:-}" ] && [ "$pubs" -ge 1 ] 2>/dev/null; then
        echo "[OK] publisher count for $topic = $pubs"
        return 0
    fi
    echo "[WARN] publisher missing for $topic"
    return 1
}

diagnose_nodes_and_topics() {
    load_env
    echo "===== diagnose: node process status ====="
    status_nodes
    echo

    echo "===== diagnose: ROS node graph ====="
    ros2 node list || true
    echo

    echo "===== diagnose: key topic publishers ====="
    diag_topic_pub "/cmd/u" || true
    diag_topic_pub "/hw/pwm_applied" || true
    diag_topic_pub "/hw/enc" || true
    diag_topic_pub "/ina219/bus_voltage_v" || true
    diag_topic_pub "/ina219/current_ma" || true
    diag_topic_pub "/ina219/power_mw" || true
    diag_topic_pub "/imu/data" || true
    echo

    echo "===== diagnose: sample data check ====="
    diag_topic_once "/hw/pwm_applied" 2 || true
    diag_topic_once "/hw/enc" 2 || true
    diag_topic_once "/ina219/bus_voltage_v" 2 || true
    diag_topic_once "/ina219/current_ma" 2 || true
    diag_topic_once "/ina219/power_mw" 2 || true
    diag_topic_once "/imu/data" 2 || true
    echo

    echo "===== diagnose: recent logs ====="
    logs_nodes
}


check_once() {
    load_env
    echo "===== environment ====="
    echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
    echo "RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
    echo

    echo "===== device check ====="
    local arduino_port
    local imu_port
    arduino_port="$(resolve_serial_port "Arduino" "$ARDUINO_PORT_PATTERN" "$ARDUINO_PORT_FALLBACK")"
    imu_port="$(resolve_serial_port "IMU" "$IMU_PORT_PATTERN" "$IMU_PORT_FALLBACK")"
    echo "Arduino port: $arduino_port"
    ls -l "$arduino_port" 2>/dev/null || true
    echo "IMU port: $imu_port"
    ls -l "$imu_port" 2>/dev/null || true
    echo

    echo "===== topic list ====="
    ros2 topic list || true
}

usage() {
    cat <<EOF
Usage: $0 {start|stop|restart|status|logs|monitor|check|diagnose|controller}

Commands:
  start    : 아두이노 브리지 + IMU 노드 실행
  stop     : 실행 중인 노드 종료
  restart  : 재시작
  status   : 현재 프로세스 상태 확인
  logs     : 최근 로그 확인
  monitor  : 한줄 live monitor
  check    : 장치와 ROS 환경 간단 점검
  diagnose : 어떤 노드/토픽이 비었는지 상세 점검
  controller : pendulum_controller.py 직접 실행
EOF
}
main() {
    local action="${1:-}"

    case "$action" in
        start)
            start_nodes
            ;;
        stop)
            stop_nodes
            ;;
        restart)
            stop_nodes
            sleep 1
            start_nodes
            ;;
        status)
            status_nodes
            ;;
        logs)
            logs_nodes
            ;;
        monitor)
            monitor_live
            ;;
        check)
            check_once
            ;;
        diagnose)
            diagnose_nodes_and_topics
            ;;
        controller)
            load_env
            echo "[$(timestamp)] Starting pendulum_controller (foreground)..."
            python3 "$SCRIPT_DIR/pendulum_controller.py"
            ;;
        *)
            usage
            exit 1
            ;;
    esac
}

main "$@"

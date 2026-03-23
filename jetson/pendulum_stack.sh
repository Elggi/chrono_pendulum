#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# =========================
# Jetson ROS stack launcher
# =========================

# ---- user-configurable ----
ROS_SETUP="/opt/ros/humble/setup.bash"
WS_SETUP="$HOME/ros2_ws/install/setup.bash"

export ROS_DOMAIN_ID=7
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

ARDUINO_PORT="/dev/ttyACM0"
ARDUINO_BAUD="115200"

IMU_PORT="/dev/ttyUSB0"
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
        exit 1
    fi
    source "$WS_SETUP"
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

    start_process "arduino_bridge" \
        "ros2 run hw_arduino_bridge bridge_node --ros-args -p port:=$ARDUINO_PORT -p baud:=$ARDUINO_BAUD -p pwm_limit:=255.0"

    start_process "imu_node" \
        "ros2 run wheeltec_n100_imu imu_node --ros-args -p serial_port:=$IMU_PORT -p serial_baud:=$IMU_BAUD -r imu:=/imu/data"

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

monitor_menu() {
    load_env

    echo "======================================="
    echo " Jetson ROS topic monitor"
    echo "======================================="
    echo "1) /imu/data"
    echo "2) /hw/enc"
    echo "3) /cmd/u"
    echo "4) /hw/pwm_applied"
    echo "5) /hw/arduino_ms"
    echo "6) /ina219/bus_voltage_v"
    echo "7) /ina219/current_ma"
    echo "8) /ina219/power_mw"
    echo "9) topic list"
    echo "10) topic hz /imu/data"
    echo "11) topic hz /hw/enc"
    echo "12) topic hz /ina219/bus_voltage_v"
    echo "13) topic hz /ina219/current_ma"
    echo "14) topic hz /ina219/power_mw"
    echo "q) quit"
    echo "======================================="
    read -rp "Select: " choice

    case "$choice" in
        1) ros2 topic echo /imu/data ;;
        2) ros2 topic echo /hw/enc ;;
        3) ros2 topic echo /cmd/u ;;
        4) ros2 topic echo /hw/pwm_applied ;;
        5) ros2 topic echo /hw/arduino_ms ;;
        6) ros2 topic echo /ina219/bus_voltage_v ;;
        7) ros2 topic echo /ina219/current_ma ;;
        8) ros2 topic echo /ina219/power_mw ;;
        9) ros2 topic list ;;
        10) ros2 topic hz /imu/data ;;
        11) ros2 topic hz /hw/enc ;;
        12) ros2 topic hz /ina219/bus_voltage_v ;;
        13) ros2 topic hz /ina219/current_ma ;;
        14) ros2 topic hz /ina219/power_mw ;;
        q|Q) exit 0 ;;
        *) echo "Invalid choice"; exit 1 ;;
    esac
}


check_once() {
    load_env
    echo "===== environment ====="
    echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
    echo "RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
    echo

    echo "===== device check ====="
    ls -l "$ARDUINO_PORT" 2>/dev/null || echo "Missing: $ARDUINO_PORT"
    ls -l "$IMU_PORT" 2>/dev/null || echo "Missing: $IMU_PORT"
    echo

    echo "===== topic list ====="
    ros2 topic list || true
}

usage() {
    cat <<EOF
Usage: $0 {start|stop|restart|status|logs|monitor|check|controller}

Commands:
  start    : 아두이노 브리지 + IMU 노드 실행
  stop     : 실행 중인 노드 종료
  restart  : 재시작
  status   : 현재 프로세스 상태 확인
  logs     : 최근 로그 확인
  monitor  : topic echo / hz 확인 메뉴
  check    : 장치와 ROS 환경 간단 점검
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
            monitor_menu
            ;;
        check)
            check_once
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

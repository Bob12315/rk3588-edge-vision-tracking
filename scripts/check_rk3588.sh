#!/usr/bin/env bash
set -eu

show_command() {
  label="$1"
  command_name="$2"
  if command -v "$command_name" >/dev/null 2>&1; then
    "$command_name" --version 2>&1 | head -n 1 | sed "s/^/${label}: /"
  else
    echo "${label}: not found"
  fi
}

echo "OS: $(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown}")"
echo "Kernel: $(uname -sr)"
echo "Architecture: $(uname -m)"
show_command "Python" python3
show_command "Git" git

if [ -r /proc/device-tree/compatible ]; then
  echo "Device tree: $(tr '\0' ',' </proc/device-tree/compatible | sed 's/,$//')"
else
  echo "Device tree: unavailable (run this script on the target board)"
fi

if compgen -G '/dev/rknpu*' >/dev/null; then
  echo "RKNPU devices: $(find /dev -maxdepth 1 -name 'rknpu*' -printf '%f ' 2>/dev/null)"
else
  echo "RKNPU devices: none found"
fi

if command -v rknn_server >/dev/null 2>&1; then
  echo "RKNN server: $(command -v rknn_server)"
else
  echo "RKNN server: not found"
fi

echo "CPU count: $(getconf _NPROCESSORS_ONLN)"
echo "Memory: $(awk '/MemTotal/ {printf "%.1f GiB", $2 / 1024 / 1024}' /proc/meminfo)"
echo "Disk: $(df -h . | awk 'NR == 2 {print $4 " available on " $1}')"

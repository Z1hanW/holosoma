#!/usr/bin/env bash
# Is the laptop GPU actually able to run FFS at frame rate right now?
#
# After a reboot the RTX PRO 5000 has come up capped at 20 W (SW Power Cap),
# pinned at P8 / 180 MHz, PCIe Gen1 x8 - Fast-FoundationStereo then takes ~240 ms
# a frame instead of ~25 ms and the depth path is unusable. This prints the
# numbers that tell the two states apart and exits non-zero in the bad one.
set -uo pipefail
PY="${HOLOSOMA_INFERENCE_PYTHON:-$HOME/.holosoma_deps/miniconda3/envs/hsinference/bin/python}"
D=/sys/bus/pci/devices/0000:02:00.0

echo "platform profile : $(cat /sys/firmware/acpi/platform_profile 2>/dev/null)   ($(powerprofilesctl get 2>/dev/null))"
echo "AC power         : $(cat /sys/class/power_supply/AC/online 2>/dev/null)"
# Dell caps the dGPU (20 W) when the adapter cannot supply the platform's rated
# power. A USB-C PD source shows its contract here; the RTX PRO 5000 alone wants
# 115-175 W, so anything under ~180 W means a capped GPU regardless of drivers.
for p in /sys/class/power_supply/ucsi-source-psy-*; do
  [[ "$(cat $p/online 2>/dev/null)" == "1" ]] || continue
  vmax=$(cat $p/voltage_max); imax=$(cat $p/current_max); vnow=$(cat $p/voltage_now); inow=$(cat $p/current_now 2>/dev/null || echo 0)
  echo "USB-C PD source  : contract $((vmax/1000000))V x $(awk -v i=$imax 'BEGIN{printf "%.1f",i/1e6}')A = $(awk -v v=$vmax -v i=$imax 'BEGIN{printf "%.0f",v/1e6*i/1e6}') W, now $((vnow/1000000))V x $(awk -v i=$inow 'BEGIN{printf "%.1f",i/1e6}')A = $(awk -v v=$vnow -v i=$inow 'BEGIN{printf "%.0f",v/1e6*i/1e6}') W   (GPU alone needs 115-175 W)"
done
lim=$(nvidia-smi -q -d POWER 2>/dev/null | awk -F: '/Current Power Limit/{gsub(/ /,"",$2); print $2; exit}')
echo "GPU power limit  : ${lim:-?}   (expected 115.00W; 20.00W = stuck)"
echo "GPU persistence  : $(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null)"
echo "PCIe link        : $(cat $D/current_link_speed 2>/dev/null) x$(cat $D/current_link_width 2>/dev/null)   (max $(cat $D/max_link_speed 2>/dev/null) x$(cat $D/max_link_width 2>/dev/null))"
echo "RTD3             : $(grep -m1 'Runtime D3 status' /proc/driver/nvidia/gpus/*/power 2>/dev/null | awk -F: '{print $2}' | xargs)  suspended $(( $(cat $D/power/runtime_suspended_time)/1000 ))s of $(cut -d. -f1 /proc/uptime)s uptime"

echo -n "sustained load   : "
"$PY" - <<'PYEOF' 2>/dev/null &
import torch, time
x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
end = time.time() + 8
while time.time() < end: (x @ x)
torch.cuda.synchronize()
PYEOF
sleep 5
read -r util clk pw ps < <(nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw,pstate --format=csv,noheader,nounits | tr -d ',')
wait
echo "${util}% util, ${clk} MHz, ${pw} W, ${ps}   (expected ~100%, ~1800 MHz, ~115 W, P1)"

ok=1
[[ "${lim%W}" == "115.00" ]] || ok=0
[[ "${clk:-0}" -ge 1200 ]] || ok=0
if [[ $ok == 1 ]]; then echo "RESULT: GPU is at full performance"; else
  echo "RESULT: GPU is POWER-CAPPED / stuck low. Fix (needs sudo):"
  echo "    sudo nvidia-smi -pm 1 && sudo nvidia-smi -rgc     # then re-run this script"
  echo "  if still capped, disable runtime D3 and reboot:"
  echo "    echo 'options nvidia NVreg_DynamicPowerManagement=0x00' | sudo tee /etc/modprobe.d/nvidia-no-rtd3.conf"
  echo "    sudo update-initramfs -u && sudo reboot"
  exit 1
fi

# Runbook — Switch L4T GPU node operations

Operations guide for running a Switchroot L4T Ubuntu Nintendo Switch as a headless GPU-capable
ARM64 lab node. Assumes the stock Switchroot Noble (unity 5.1.2) image line.

## Access

The L4T images expose three independent management paths over the USB-C port, plus Wi-Fi:

| Path | How | Notes |
|---|---|---|
| USB network (primary) | Switch is 192.168.55.1; set your PC's "Remote NDIS"/UsbNcm adapter to static 192.168.55.100/24 | works with no Wi-Fi |
| Wi-Fi | `nmcli device wifi connect SSID password ...` (run via sudo over USB/serial; WPA2 only — WPA3 is broken on this image) | profile persists |
| USB serial console | 115200 8N1 on the serial port the gadget exposes (e.g. COMn / /dev/ttyACM0) | login prompt; last-resort recovery |

Enable SSH access: the image ships no ssh server. Either complete the first-boot OEM wizard once
on the touchscreen, or pre-inject an sshd before first boot (see repo history/README approach —
rootfs edits: install openssh-server files, create the `sshd` privilege-separation user, enable
`ssh.service`, copy `/usr/share/openssh/sshd_config` to `/etc/ssh/sshd_config`, generate host keys,
and skip `nv-oem-config` by pointing `default.target` at `graphical.target`).

**GPU device access requires your user to be in the `video` group:**
`sudo usermod -aG video $USER` (re-login to apply).

## Headless conversion (optional, saves ~0.6 GiB RAM and power)

```bash
sudo systemctl stop lightdm
sudo systemctl set-default multi-user.target
echo 0 | sudo tee /sys/class/backlight/backlight/brightness
```

## Using the GPU (this repo)

```bash
make native && make bench-native    # native compile + verified benchmark suite
./ncc kernel.cu && ./gpu_workload kernel.ptx
```
Cross (nvcc) path: `tools/setup-wsl-nvcc.sh` once, then `tools/build-gpu.sh kernel.cu` from the
PC, `make bench` on the Switch. Details in the README; kernel/harness reference in
[switch_gpu.cu](../switch_gpu.cu) and [gpu_workload.c](../gpu_workload.c).

## Boot modes & recovery

- Boot menu: inject payload → Hekate → Launch (Atmosphere entries) or **More Configs → L4T
  Ubuntu Noble**.
- From Linux back to the Hekate menu: `echo bootloader > /sys/devices/r2p/action` (root) + reboot.
- Expose the SD card to a PC from Linux: `echo ums > /sys/devices/r2p/action && echo 0 > /sys/devices/r2p/param1` + reboot.
- Shutdown cleanly: `sudo poweroff`. Never cut power mid-boot.

## Power

Verified on a Raspberry Pi USB port (5 V): charges at idle/normal load; sustained all-core load
draws ~0.5–0.7 W more than the port supplies (battery eases down, recovers after). For 24/7
sustained GEMM/inference use a PD charger or powered hub. Battery telemetry:
`/sys/class/power_supply/battery/{status,capacity,current_now}`.

## Hardware facts (verified on-device)

- CPU: 4× Cortex-A57, 102–2091 MHz (schedutil); AES/SHA/CRC/atomics/NEON present.
- GPU compute: CUDA **driver API works** (`libcuda` 10.0) with PTX ≤ 6.3 / sm_53; see README.
- GPU frequency control via `/sys/class/devfreq/57000000.gpu/` (governor locked to
  `nvhost_podgov`; ramps under load, first-run scores read low).
- No CUDA compiler from NVIDIA for arm64 anymore (archives now sbsa-cross only) — clang-18 is the
  native path; archived x86_64 nvcc 10.0 in WSL is the maximum-performance cross path.
- Vulkan: CPU-only (llvmpipe). OpenGL/EGL: NVIDIA tegra libs (the graphics path).
- Media: NVDEC/VIC device nodes exist; stock distro ffmpeg has no NVDEC integration.
- Thermals: 55 °C max zone at sustained full GPU load (dockless, headless).

## Safety net

Before doing anything adventurous to the SD card, take a sector image of it and verify by hash.
With a verified image, any Linux-side disaster is one re-flash from recovery — including the
emuMMC, games, and the Atmosphere setup sharing the card with Linux.

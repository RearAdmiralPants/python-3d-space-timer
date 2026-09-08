# Shared GPU selection, sourced by launch.sh and tools/bench-gpu.sh.
#
# On a hybrid-graphics machine (an Intel iGPU wired to the display plus a
# discrete NVIDIA), OpenGL goes to the *integrated* GPU unless a program asks
# otherwise. Asking otherwise means setting a couple of environment variables
# that redirect GLX to NVIDIA's vendor library; the discrete GPU renders and
# the result is copied back to the Intel-driven display each frame.
#
# This has nothing to do with CUDA. It is a GLX vendor switch, and an AMD card
# would be selected the same way (via DRI_PRIME) despite having no CUDA at all.
#
# `switcherooctl list` prints the correct environment for every GPU it finds,
# which is where these variable names come from.
#
# Usage:  gpu_select <intel|nvidia|default>
#         gpu_args_consumed  -- how many argv entries `--gpu X` ate

GPU_CHOICE=default

# Parse a leading `--gpu <name>` / `--gpu=<name>`. Sets GPU_CHOICE and
# GPU_SHIFT (the number of arguments to shift away).
gpu_parse_args() {
    GPU_SHIFT=0
    case "${1:-}" in
        --gpu)   GPU_CHOICE="${2:-}"; GPU_SHIFT=2 ;;
        --gpu=*) GPU_CHOICE="${1#--gpu=}"; GPU_SHIFT=1 ;;
    esac

    case "$GPU_CHOICE" in
        intel|nvidia|default) ;;
        list)
            if command -v switcherooctl >/dev/null 2>&1; then
                switcherooctl list
            else
                echo "switcherooctl not installed; try: lspci | grep -Ei 'vga|3d'" >&2
            fi
            exit 0
            ;;
        *)
            echo "error: --gpu takes intel, nvidia, default, or list (got '$GPU_CHOICE')" >&2
            exit 2
            ;;
    esac
}

gpu_select() {
    case "$1" in
        nvidia)
            if ! command -v nvidia-smi >/dev/null 2>&1; then
                echo "error: --gpu nvidia, but nvidia-smi is not installed." >&2
                echo "Either the driver is missing or this machine has no NVIDIA GPU." >&2
                echo "Run './launch.sh --gpu list' to see what is actually here." >&2
                exit 1
            fi
            export __NV_PRIME_RENDER_OFFLOAD=1
            export __GLX_VENDOR_LIBRARY_NAME=nvidia
            export __VK_LAYER_NV_optimus=NVIDIA_only
            echo ">> GPU: NVIDIA (PRIME offload)"
            ;;
        intel)
            # The integrated GPU already drives the display, so selecting it
            # means *not* setting the offload variables. Unset them in case
            # they leaked in from the caller's environment.
            unset __NV_PRIME_RENDER_OFFLOAD __GLX_VENDOR_LIBRARY_NAME __VK_LAYER_NV_optimus
            echo ">> GPU: integrated (no offload)"
            ;;
        default)
            # Whatever the system picks. On hybrid machines this is the iGPU.
            ;;
    esac
}

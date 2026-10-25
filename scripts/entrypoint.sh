#!/bin/sh
# Hardware guard + llama-server launcher.
#
# The binary is compiled with -march=native and for a specific compute
# capability. If the image runs on a machine other than the one it was built
# on, at best it is slow and at worst it crashes with SIGILL (Illegal
# instruction) from deep inside ggml, an error that is hard to diagnose.
# So we check before starting.
set -eu

LLAMA_DIR=/opt/llama
BUILD_INFO="${LLAMA_DIR}/build-info.env"

red()  { printf '\033[1;31m%s\033[0m\n' "$*"; }
dim()  { printf '\033[2m%s\033[0m\n' "$*"; }
bold() { printf '\033[1m%s\033[0m\n' "$*"; }

[ -r "$BUILD_INFO" ] || { red "ERROR: $BUILD_INFO is missing (corrupt image). Rebuild with: docker compose build"; exit 1; }

# The values are not sourced: they contain parentheses and spaces ("Intel(R) Core(TM)
# Ultra 7 265K") that would break the shell parser. They are read with sed, so
# both sides of the comparison use exactly the same parsing.
BUILT="$(cat "$BUILD_INFO")"
CURRENT="$(sh "${LLAMA_DIR}/hw-fingerprint.sh")"

get() { printf '%s\n' "$2" | sed -n "s/^$1=//p" | head -n1; }

# ---------------------------------------------------------------------------
# The GPU must be visible
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    red "ERROR: no GPU visible inside the container (nvidia-smi not found)."
    echo
    echo "This binary was compiled for CUDA $(get CUDA_ARCH "$BUILT") and cannot run without a GPU."
    echo "Check that the service has 'gpus: all' and that the host has the toolkit:"
    echo
    echo "    sudo apt install -y nvidia-container-toolkit"
    echo "    sudo nvidia-ctk runtime configure --runtime=docker"
    echo "    sudo systemctl restart docker"
    exit 1
fi

# ---------------------------------------------------------------------------
# The driver must support the CUDA major version used for the build
# ---------------------------------------------------------------------------
# The image has no compat libs (they are removed in the Dockerfile), so the host
# driver is used. Within the same major the minor does not matter (a 13.3 binary
# runs on a 13.0 driver), but a different major does require a newer driver.
BUILT_MAJOR="$(get CUDA_VERSION "$BUILT" | cut -d. -f1)"
DRV_CUDA="$(get DRIVER_CUDA "$CURRENT")"
DRV_MAJOR="$(printf '%s' "$DRV_CUDA" | cut -d. -f1)"

if [ -n "$BUILT_MAJOR" ] && [ -n "$DRV_MAJOR" ] \
   && [ "$BUILT_MAJOR" -eq "$BUILT_MAJOR" ] 2>/dev/null \
   && [ "$DRV_MAJOR" -eq "$DRV_MAJOR" ] 2>/dev/null \
   && [ "$DRV_MAJOR" -lt "$BUILT_MAJOR" ]; then
    red "ERROR: the NVIDIA driver on this machine is too old."
    echo
    echo "  built with CUDA    : $(get CUDA_VERSION "$BUILT")"
    echo "  driver             : $(get DRIVER_VERSION "$CURRENT")  (supports up to CUDA ${DRV_CUDA})"
    echo
    echo "Update the NVIDIA driver, or set a CUDA_VERSION from the ${DRV_MAJOR}.x"
    echo "branch in .env and rebuild with 'docker compose build'."
    exit 1
fi

# ---------------------------------------------------------------------------
# Field-by-field comparison (not a global hash) so we can report WHAT changed
# ---------------------------------------------------------------------------
MISMATCH=""
check() {
    _label="$1"
    _built="$(get "$2" "$BUILT")"
    _now="$(get "$2" "$CURRENT")"
    [ "$_built" = "$_now" ] && return 0
    MISMATCH="yes"
    red "  ${_label}:"
    echo "      built for    : ${_built}"
    echo "      this machine : ${_now}"
}

check "CPU"           CPU_MODEL
check "CPU ISA"       CPU_ISA_HASH

# The GPU is compared by compute capability, not by model name: what decides
# whether the binary works is the SASS kernels it contains. Two different
# cards with the same capability (e.g. 5070 Ti and 5080, both sm_120) run
# the same binary just fine.
#   CUDA_ARCH "120a-real" -> 120        compute_cap "12.0" -> 120
BUILT_CC="$(get CUDA_ARCH "$BUILT" | tr ';' '\n' | sed 's/[^0-9]//g' | sort -u | tr '\n' ' ')"
NOW_CC="$(get GPU_CC "$CURRENT" | tr ',' '\n' | tr -d '.' | sort -u | tr '\n' ' ')"

for cc in $NOW_CC; do
    case " $BUILT_CC " in
        *" $cc "*) ;;
        *)
            MISMATCH="yes"
            red "  GPU compute capability:"
            echo "      built for    : $(get CUDA_ARCH "$BUILT")"
            echo "      this machine : $(get GPU_NAME "$CURRENT") (sm_$(get GPU_CC "$CURRENT"))"
            ;;
    esac
done

if [ -n "$MISMATCH" ]; then
    echo
    if [ "${ALLOW_HW_MISMATCH:-0}" = "1" ]; then
        red "WARNING: hardware mismatch, but ALLOW_HW_MISMATCH=1. Continuing."
        red "         If it crashes with 'Illegal instruction', this is why."
        echo
    else
        red "======================================================================"
        red " This machine's hardware does not match the one this image was built for."
        red "======================================================================"
        echo
        echo "This image was compiled natively (-march=$(get CPU_MARCH "$BUILT")"
        echo "+ CUDA $(get CUDA_ARCH "$BUILT")) and is not portable. Rebuild it with:"
        echo
        bold "    docker compose build && docker compose up -d"
        echo
        dim  "To skip the check once (at your own risk):"
        dim  "    ALLOW_HW_MISMATCH=1 docker compose up -d"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# All good
# ---------------------------------------------------------------------------
bold "llama-server $(get LLAMA_CPP_REF "$BUILT") ($(get LLAMA_COMMIT "$BUILT"))"
dim  "  CPU  : $(get CPU_MODEL "$BUILT")  [-march=$(get CPU_MARCH "$BUILT")]"
dim  "  GPU  : $(get GPU_NAME "$CURRENT")  [sm_$(get GPU_CC "$CURRENT") -> $(get CUDA_ARCH "$BUILT")]"
dim  "  CUDA : $(get CUDA_VERSION "$BUILT")   built on $(get BUILD_DATE "$BUILT")"
echo

# LLAMA_EXTRA_ARGS is unquoted on purpose: we want word splitting so that
# several flags can be passed from a single .env variable
# shellcheck disable=SC2086
exec "${LLAMA_DIR}/llama-server" ${LLAMA_EXTRA_ARGS:-} "$@"

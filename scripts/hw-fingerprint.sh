#!/bin/sh
# Fingerprint of the hardware being built for / running on.
#
# Used in TWO phases, which is why it lives in a single file:
#   - build   : the Dockerfile runs it and saves the output to build-info.env
#   - runtime : the entrypoint runs it again and compares with build-info.env
#
# Prints KEY=value pairs to stdout. The GPU part is only emitted if nvidia-smi
# is present (it is not during the build: those values come as build-args).
set -eu

# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------
# Allowlist of flags that affect -march=native. We deliberately do NOT hash
# every flag in /proc/cpuinfo: entries such as ibrs, stibp, flush_l1d, hwp or
# arat change with microcode updates or kernel mitigations, and would cause
# false "different hardware" detections after a routine update even though
# the ISA has not changed at all.
ISA_ALLOW="
mmx sse sse2 pni sse3 ssse3 sse4_1 sse4_2 sse4a
avx avx2 fma f16c bmi1 bmi2 popcnt lzcnt abm movbe adx
aes pclmulqdq sha_ni sha512 vaes vpclmulqdq gfni sm3 sm4
rdrand rdseed rdpid
avx_vnni avx_ne_convert avx_vnni_int8 avx_vnni_int16 avx_ifma
avx512f avx512dq avx512bw avx512vl avx512cd avx512er avx512pf
avx512vnni avx512ifma avx512vbmi avx512vbmi2 avx512bitalg
avx512vpopcntdq avx512_bf16 avx512_fp16 avx512_vp2intersect avx512_vnni
amx_tile amx_int8 amx_bf16 amx_fp16 amx_complex
xsave xsaveopt xsavec xsaves xgetbv1
movdiri movdir64b serialize waitpkg cldemote clwb clflushopt
fsrm erms prefetchi cmpccxadd wrmsrns raoint uintr apx_f
avx10 avx10_1 avx10_2
"

cpu_model=$(sed -n 's/^model name[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo | head -n1)
[ -n "$cpu_model" ] || cpu_model="unknown"

cpu_flags=$(sed -n 's/^flags[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo | head -n1)

# Intersection of CPU flags x allowlist, sorted so the hash is stable
# regardless of the order in which the kernel lists them.
cpu_isa=$(
    for f in $cpu_flags; do
        case " $(echo $ISA_ALLOW) " in
            *" $f "*) echo "$f" ;;
        esac
    done | sort -u | tr '\n' ' ' | sed 's/[[:space:]]*$//'
)

cpu_isa_hash=$(printf '%s' "$cpu_isa" | sha256sum | cut -c1-16)

echo "CPU_MODEL=${cpu_model}"
echo "CPU_ISA=${cpu_isa}"
echo "CPU_ISA_HASH=${cpu_isa_hash}"

# ---------------------------------------------------------------------------
# GPU (runtime only: nvidia-smi is injected by the container toolkit)
# ---------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | sed 's/[[:space:]]*$//')
    gpu_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ',' | sed 's/,$//' | tr -d ' ')
    echo "GPU_NAME=${gpu_name:-unknown}"
    echo "GPU_CC=${gpu_cc:-unknown}"

    # CUDA version supported by the driver (the "ceiling"). Not part of the
    # fingerprint comparison (updating the driver does not require a rebuild), but
    # the entrypoint uses it to check that it covers the CUDA major used for the build.
    driver_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')
    driver_cuda=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -n1)
    echo "DRIVER_VERSION=${driver_ver:-unknown}"
    echo "DRIVER_CUDA=${driver_cuda:-unknown}"
fi

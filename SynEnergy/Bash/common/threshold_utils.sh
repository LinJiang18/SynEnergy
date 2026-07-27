#!/usr/bin/env bash
# Shared helpers for threshold-sweep pipeline scripts.
# Suffix mapping: 0.01 -> 001, 0.05 -> 005, 0.10 -> 010

if [[ -z "${THRESHOLDS+x}" ]]; then
  THRESHOLDS=(0.279 0.156 0.073) # 0.01 0.05 0.2
fi

threshold_suffix() {
  local thr="$1"
  python - <<PY
thr = float("${thr}")
print(f"{int(round(thr * 100)):03d}")
PY
}

patch_controlnet_threshold() {
  local src_config="$1"
  local dst_config="$2"
  local threshold="$3"
  python - <<PY
import yaml
from pathlib import Path

src = Path("${src_config}")
dst = Path("${dst_config}")
threshold = float("${threshold}")

cfg = yaml.safe_load(src.read_text())
cn = cfg.setdefault("controlnet", {})
cn["filter_threshold"] = threshold
cn["occurrence_threshold"] = threshold
dst.write_text(yaml.dump(cfg, sort_keys=False, default_flow_style=False))
PY
}

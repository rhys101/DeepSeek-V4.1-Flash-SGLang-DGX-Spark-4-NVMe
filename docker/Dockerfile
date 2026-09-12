# Pinned Linux ARM64 image. Source overlays are also checked at each staged launch.
FROM lmsysorg/sglang@sha256:3475d88ec3124867d9d6f7b3bd49afdcf8ef4d6a9f8454fab331d5adb5204de7
COPY patches /tmp/tp4-patches
COPY scripts/apply-overlays.py /tmp/tp4-apply-overlays.py
RUN python3 /tmp/tp4-apply-overlays.py --source /tmp/tp4-patches --target /sgl-workspace/sglang
LABEL org.opencontainers.image.description="Pinned base and eleven verified TP4 SGLang source overlays; use scripts/cluster.py for the runtime kit and NVMe row-store"
HEALTHCHECK NONE

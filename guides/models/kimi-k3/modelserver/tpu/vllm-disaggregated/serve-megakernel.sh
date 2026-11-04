#!/usr/bin/env bash
# Kimi-K3 on the vllm-torchtpu megakernel backend: one P/D role (ROLE=prefill|decode) on one
# tpu7x 2x2x4 slice. Every pod of the LeaderWorkerSet runs this script; pod LWS_WORKER_INDEX is
# node rank N of one TP=32 x DCP=32 engine launched with `vllm serve --nnodes 4`. Rank 0 serves
# the OpenAI API (prefill on :8000; decode on :8200 behind the routing sidecar on :8000), ranks
# 1-3 run headless.
#
# The megakernel backend runs the role's program from kv_role: the KV producer runs the
# sequence-parallel prefill stack, the KV consumer the fused 93-layer decoder. Both roles keep a
# host-DRAM KV store (TPURaidenOffloadingConnector, cpu_bytes_to_use split over the 32 ranks,
# served from rank 0's host: store_server_ip) next to the Raiden P->D transfer connector.
set -euo pipefail

: "${ROLE:?ROLE must be prefill or decode}"
RANK="${LWS_WORKER_INDEX:?}"
until HEAD_IP=$(getent ahostsv4 "${LWS_LEADER_ADDRESS}" | awk 'NR==1{print $1}') && [ -n "${HEAD_IP}" ]; do
  echo "Waiting for ${LWS_LEADER_ADDRESS} to resolve..."; sleep 5
done

# The engine forms its TPU world itself (--nnodes): clear the slice-wide multi-host environment
# the GKE TPU webhook injects and describe this host's place in the 2x2x4 slice instead.
export MEGASCALE_COORDINATOR_ADDRESS= MEGASCALE_NUM_SLICES= MEGASCALE_PORT= MEGASCALE_SLICE_ID= \
  TPU_PROCESS_ADDRESSES= TPU_PROCESS_PORT= TPU_WORKER_HOSTNAMES=
export TPU_MULTIHOST_BACKEND=mp
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_HOST_BOUNDS=1,1,4 TPU_NODE_ID="${RANK}"
export VLLM_HOST_IP="${POD_IP}"
# Raiden advertises the role's head; the per-role ids and ports keep prefill rank k and decode
# rank k from sharing a Raiden work unit (transfers fail with "No transfer chunks found").
export TPU_RAIDEN_ADVERTISE_HOST="${HEAD_IP}"
export TPU_RAIDEN_JOB_NAME="kimi-k3-${ROLE}" TPU_RAIDEN_ENGINE_ID="kimi-k3-${ROLE}-engine"
export TPU_KV_TRANSFER_NAMESPACE="kimi-k3"

RAIDEN='"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector"'
OFFLOAD="{\"kv_connector\":\"TPURaidenOffloadingConnector\",\"kv_connector_module_path\":\"vllm_torchtpu.offload.raiden_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":${HOST_KV_BYTES},\"raiden_controller_port\":${OFFLOAD_CONTROLLER_PORT},\"raiden_job_name\":\"kimi-k3-${ROLE}-offload\",\"store_server_ip\":\"${HEAD_IP}\"}}"
case "${ROLE}" in
  prefill)
    export TPU_RAIDEN_CONTROLLER_ADDRESS="${HEAD_IP}:27340" TPU_RAIDEN_RESHARD_PORT_BASE=27340 \
      TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=27440 TPU_KV_TRANSFER_PORT=29340 TPU_SIDE_CHANNEL_PORT=29740
    KV_ROLE=kv_producer PORT=8000
    ROLE_ARGS=(--max-num-seqs="${MAX_NUM_SEQS}" --max-num-batched-tokens=8192)
    ;;
  decode)
    export TPU_RAIDEN_CONTROLLER_ADDRESS="${HEAD_IP}:28340" TPU_RAIDEN_RESHARD_PORT_BASE=28340 \
      TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=28440 TPU_KV_TRANSFER_PORT=30340 TPU_SIDE_CHANNEL_PORT=30740
    KV_ROLE=kv_consumer PORT=8200
    ROLE_ARGS=(--max-num-seqs="${MAX_NUM_SEQS}" --max-num-batched-tokens=512)
    ;;
  *) echo "ROLE must be prefill or decode, got '${ROLE}'" >&2; exit 2 ;;
esac
KV_TRANSFER="{\"kv_connector\":\"TPUMultiConnector\",\"kv_connector_module_path\":\"vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector\",\"kv_role\":\"${KV_ROLE}\",\"kv_connector_extra_config\":{\"connectors\":[{${RAIDEN},\"kv_role\":\"${KV_ROLE}\"},${OFFLOAD}]}}"

if [ "${RANK}" = 0 ]; then
  NODE_ARGS=(--host=0.0.0.0 --port="${PORT}")
else
  NODE_ARGS=(--headless)
fi

exec vllm serve /models/Kimi-K3 \
  --served-model-name=moonshotai/Kimi-K3 \
  --trust-remote-code \
  --language-model-only \
  --nnodes=4 --node-rank="${RANK}" --master-addr="${HEAD_IP}" \
  --distributed-timeout-seconds=1800 \
  "${NODE_ARGS[@]}" \
  --tensor-parallel-size=32 \
  --enable-expert-parallel \
  --decode-context-parallel-size=32 \
  --load-format=megakernel \
  --model-loader-extra-config='{"presharded":"/models/kimi-k3-tp32"}' \
  --max-model-len=1048576 \
  --kv-cache-dtype=fp8 \
  --block-size=512 \
  --cp-kv-cache-interleave-size=512 \
  --kv-cache-memory-bytes=19327352832 \
  --mamba-cache-mode=align \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  "${ROLE_ARGS[@]}" \
  --kv-transfer-config="${KV_TRANSFER}" \
  --enable-auto-tool-choice \
  --tool-call-parser=kimi_k3 \
  --reasoning-parser=kimi_k3 \
  --disable-access-log-for-endpoints=/health,/metrics,/v1/models \
  --shutdown-timeout=45

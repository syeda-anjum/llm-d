# gpt-oss-120b on TPU 7x (SGLang)

This recipe deploys `openai/gpt-oss-120b` on Google TPU v7x (`tpu7x`) with SGLang on Torch TPU (`sglang-torchtpu`), composing prefix-aware routing, host-DRAM hierarchical KV caching (HiCache), and prefill/decode disaggregation over TPU Sync (`raiden`); see the
[Models overview](../../../docs/well-lit-paths/models/README.md) for how the model guides fit together.

## Overview

This guide deploys a tuned `llm-d` configuration for multi-turn agentic and reasoning workloads on `openai/gpt-oss-120b`:

- **Prefix-aware routing** (`approx-prefix-cache-producer` + `prefix-cache-affinity-filter` at 64-token page granularity) to maximize prefix cache reuse across replicas
- **Tiered prefix cache (HiCache KV offloading)** to host DRAM (`--enable-hierarchical-cache --hicache-io-backend=raiden --hicache-mem-layout=page_first_fused --hicache-write-policy=write_through`) on both `unified` and `disaggregated` (prefill and decode roles), backing each pod's HBM KV pool with 256 GiB of pinned host DRAM (32 GiB × 8 TP ranks)
- **Prefill/Decode disaggregation (`TOPOLOGY=disaggregated`)** via a `DisaggregatedSet` (3 prefill + 1 decode for this prefill-heavy workload, so the router chooses among prefill pods for prefix-cache affinity) transferring KV blocks over SGLang's `raiden` (`tpu-sync`) backend with `--disaggregation-decode-enable-radix-cache` so decode retains transferred and generated KV across turns
- **Load balancing** (`token-load-scorer` / `active-request-scorer`) to prevent hot-spotting under bursty arrivals

## Default Configuration

| Parameter | Value |
| --- | --- |
| Model | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| Replicas | 2 (`unified`) · 3 prefill + 1 decode (`disaggregated`) |
| Accelerator | Google TPU 7x (`tpu7x`) |
| Topology | `2x2x1` (4 chips / 8 TensorCores per node) |
| Parallelism | `TP=8`, `EP=8` (MXFP4 checkpoint weights, `fp8_e4m3` KV cache) |
| Context length | 131,072 tokens, the model's full context (`--page-size=64`, `--chunked-prefill-size=4096`) |
| KV cache offload | SGLang HiCache (`--hicache-io-backend=raiden`, `page_first_fused`, `write_through`), 32 GiB per TP rank (256 GiB pinned host DRAM per pod) |
| Model server | SGLang on Torch TPU (`docker.io/lmsysorg/sglang:v0.1.0-tpu`) |

## Supported Accelerators and Model Servers

This guide includes configurations for the following accelerator and model server combinations:

<!-- guide:support start -->
| Accelerator | `ACCELERATOR_TYPE` | Served model | SGLang | Notes |
| --- | --- | --- | --- | --- |
| Google TPU v7x | `tpu` | `openai/gpt-oss-120b` | 🟡 community | GKE only · SGLang on Torch TPU · `unified`: 2 replicas × 4 chips (`tpu7x` `2x2x1`, TP=8, EP=8, MXFP4 weights, fp8 KV cache, 256 GiB host-DRAM HiCache over TPU Sync, 131k context) · `TOPOLOGY=disaggregated`: a DisaggregatedSet with 3 prefill + 1 decode (TP=8, EP=8 each), TPU Sync (`raiden`) P/D KV transfer and 256 GiB host-DRAM HiCache on both roles; scale to xPyD by raising the role replicas |

✅ validated: covered by a nightly E2E workflow · 🟡 community: maintained by the hardware vendor or community, not covered by nightly E2E · ❌ not supported: tracked in the linked issue · — no configuration.
<!-- guide:support end -->

Set `TOPOLOGY` to pick the serving topology: `unified` (default) runs one pool of 2 replicas with a 256 GiB HiCache host-DRAM tier per pod;
`disaggregated` runs the SGLang Torch TPU (`raiden`) Prefill/Decode configuration as a `DisaggregatedSet`
(3 prefill + 1 decode, 256 GiB HiCache host-DRAM tier on every pod). The `sglang-disaggregated` overlay reuses the SGLang TPU7x
overlay of the [P/D disaggregation guide](../../pd-disaggregation/README.md) and only overrides the pod labels, the context length and the role replicas. Raise a role's `replicas` for xPyD.

## Prerequisites

- Have the [proper client tools installed on your local system](../../../helpers/client-setup/README.md) to use this guide.

- A GKE cluster with a TPU v7x (`tpu7x`, `2x2x1`) node pool: 2 nodes for `unified`, 4 nodes
  (three prefill, one decode) for `disaggregated`, 4 chips each, with at least 512 GiB of host memory per node to hold the 256 GiB pinned HiCache tier plus weights and runtime.
  Weights are downloaded once per node into `/var/cache/huggingface` (a `hostPath`); the checkpoint is about 65 GB, so give the nodes a
  boot disk of at least 200 GB to keep it, the image, and the node's own usage above the kubelet eviction threshold.

- Create a [HuggingFace token](../../../helpers/hf-token.md) and export it as `HF_TOKEN` in your shell.

- (Optional) Install the [monitoring stack](../../../docs/operations/observability/setup.md) if you plan to enable Prometheus monitoring.

### Get the guide

Every command below runs from a local clone of the [llm-d repository](https://github.com/llm-d/llm-d): the manifests, Helm values, and Kustomize overlays it applies live next to this guide. Set the branch and clone the repo (if you already have a checkout, skip this and run the remaining commands from inside it):

<!-- guide:prerequisites.clone start -->
<!-- llm-d-cicd:skip start -->
```bash
export BRANCH=main
git clone https://github.com/llm-d/llm-d.git && cd llm-d && git checkout ${BRANCH}
```
<!-- llm-d-cicd:skip end -->
<!-- guide:prerequisites.clone end -->

### Configure the environment

**Set the guide-specific environment variables:**

<!-- guide:env.static start -->
```bash
export REPO_ROOT=$(realpath $(git rev-parse --show-toplevel))
export GUIDE_NAME=gpt-oss-120b
export NAMESPACE=llm-d-gpt-oss-120b
export MONITORING=false # options: false, true
export MONITORING_VALUES=
export ACCELERATOR_TYPE=tpu # options: tpu
export MODEL_SERVER=sglang # options: sglang
export TOPOLOGY=unified # options: unified, disaggregated
export MODEL=openai/gpt-oss-120b # the model both topologies serve
source ${REPO_ROOT}/guides/env.sh # defines GAIE_VERSION, ROUTER_CHART_VERSION, router chart URLs, and CURL_TEST_IMAGE
```
<!-- guide:env.static end -->

**Install the Gateway API Inference Extension CRDs:**

<!-- guide:prerequisites.gaie start -->
```bash
# GAIE_URL is automatically calculated from GAIE_VERSION at ${REPO_ROOT}/guides/env.sh
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/${GAIE_URL}/v1-manifests.yaml
```
<!-- guide:prerequisites.gaie end -->

**(`TOPOLOGY=disaggregated` only) Install the LeaderWorkerSet controller with the `DisaggregatedSet` API** (skip if your cluster already runs LWS `v0.11.1` or newer with `enableDisaggregatedSet=true`):

<!-- guide:prerequisites.lws[0] start -->
<!-- llm-d-cicd:skip start -->
```bash
# only when TOPOLOGY=disaggregated:
# helm upgrade never updates CRDs, so apply the chart's CRDs first (see LWS
# "Upgrade by Helm"); this upgrades them on clusters that run an older LWS.
helm pull oci://registry.k8s.io/lws/charts/lws --version=${LWS_VERSION} --untar
kubectl apply --server-side --force-conflicts -f lws/crds
helm upgrade --install lws oci://registry.k8s.io/lws/charts/lws \
  --version=${LWS_VERSION} \
  --namespace lws-system --create-namespace \
  --set enableDisaggregatedSet=true \
  --wait --timeout 300s
```
<!-- llm-d-cicd:skip end -->
<!-- guide:prerequisites.lws[0] end -->

**Create a target namespace for the installation:**

<!-- guide:prerequisites.namespace start -->
```bash
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
```
<!-- guide:prerequisites.namespace end -->

**Create the `llm-d-hf-token` secret** in your target namespace with the key [`HF_TOKEN`](../../../helpers/hf-token.md) matching a valid HuggingFace token to pull models:

<!-- guide:prerequisites.secrets start -->
<!-- llm-d-cicd:skip start -->
```bash
kubectl create secret generic llm-d-hf-token \
  --from-literal="HF_TOKEN=${HF_TOKEN}" \
  --namespace "${NAMESPACE}" \
  --dry-run=client -o yaml | kubectl apply -f -
```
<!-- llm-d-cicd:skip end -->
<!-- guide:prerequisites.secrets end -->

## Installation Instructions

### 1. Deploy the llm-d Router

**Prepare the paths to the `helm` values files** for the `llm-d` router. The P/D disaggregated
topology uses its own router values
([`router/gpt-oss-120b-disagg.values.yaml`](router/gpt-oss-120b-disagg.values.yaml)):

<!-- tabs:start group=topology -->
<details open>
<summary><b>Unified (Default)</b></summary>

<!-- guide:deploy.router_values.unified start -->
```bash
# only when TOPOLOGY=unified:
export ROUTER_BASE_VALUES="${REPO_ROOT}/guides/recipes/router/base.values.yaml"
export ROUTER_VALUES="${REPO_ROOT}/guides/models/${GUIDE_NAME}/router/${GUIDE_NAME}.values.yaml"
```
<!-- guide:deploy.router_values.unified end -->

</details>
<details>
<summary><b>Disaggregated (Experimental)</b></summary>

<!-- guide:deploy.router_values.disaggregated start -->
<!-- llm-d-cicd:skip start -->
```bash
# only when TOPOLOGY=disaggregated:
export ROUTER_BASE_VALUES="${REPO_ROOT}/guides/recipes/router/base.values.yaml"
export ROUTER_VALUES="${REPO_ROOT}/guides/models/${GUIDE_NAME}/router/${GUIDE_NAME}-disagg.values.yaml"
```
<!-- llm-d-cicd:skip end -->
<!-- guide:deploy.router_values.disaggregated end -->

</details>
<!-- tabs:end -->

**(Optional) Enable Prometheus monitoring on the `llm-d` router** by defining the `helm` values file (requires installing the monitoring stack mentioned in [Prerequisites](#prerequisites)):

<!-- guide:deploy.monitoring_values start -->
```bash
# only when MONITORING=true:
export MONITORING_VALUES="-f ${REPO_ROOT}/guides/recipes/router/features/monitoring.values.yaml"
```
<!-- guide:deploy.monitoring_values end -->

> [!NOTE]
> **`peakPrefillThroughput` is hardware/model-specific.** The router values set the values measured
> with the [router calibration tool](../../recipes/router/calibration/README.md) on this deployment
> (`CHUNK_SIZE=4096`, the engine's `--chunked-prefill-size`): `23649` for `unified` and `17878` for
> `disaggregated`, which is measured through the full P/D path and so includes the raiden KV transfer.
> Re-measure if you change the hardware, model or engine settings.

**Deploy the router** in [Standalone Mode](../../../docs/architecture/router/proxy.md), with an Envoy sidecar in front of the router. Keep the release name `${GUIDE_NAME}`: the verification steps look up the `${GUIDE_NAME}-epp` Service by that name. To front the router with a Kubernetes Gateway instead, see Gateway Mode in the [Optimized Baseline](../../optimized-baseline/README.md#1-deploy-the-llm-d-router).

<!-- guide:deploy.standalone start -->
```bash
helm install ${GUIDE_NAME} \
  ${ROUTER_STANDALONE_CHART} \
  -f ${ROUTER_BASE_VALUES} \
  ${MONITORING_VALUES} \
  -f ${ROUTER_VALUES} \
  -n ${NAMESPACE} --version ${ROUTER_CHART_VERSION}
```
<!-- guide:deploy.standalone end -->

### 2. Deploy the Model Server

**Apply the Kustomize overlay** for your topology (both overlays enable SGLang HiCache host-DRAM KV offloading over `raiden`):

<!-- tabs:start group=topology -->
<details open>
<summary><b>Unified (Default)</b></summary>

<!-- guide:deploy.modelserver.unified start -->
```bash
# only when TOPOLOGY=unified:
kubectl apply -n ${NAMESPACE} \
  -k ${REPO_ROOT}/guides/models/${GUIDE_NAME}/modelserver/${ACCELERATOR_TYPE}/${MODEL_SERVER}/
```
<!-- guide:deploy.modelserver.unified end -->

</details>
<details>
<summary><b>Disaggregated (Experimental)</b></summary>

<!-- guide:deploy.modelserver.disaggregated start -->
<!-- llm-d-cicd:skip start -->
```bash
# only when TOPOLOGY=disaggregated:
kubectl apply -n ${NAMESPACE} \
  -k ${REPO_ROOT}/guides/models/${GUIDE_NAME}/modelserver/${ACCELERATOR_TYPE}/${MODEL_SERVER}-disaggregated/
```
<!-- llm-d-cicd:skip end -->
<!-- guide:deploy.modelserver.disaggregated end -->

</details>
<!-- tabs:end -->

Wait for the model server to become ready:

```bash
# TOPOLOGY=unified
kubectl rollout status deployment/gpt-oss-120b-tpu-sglang-decode -n ${NAMESPACE}
# TOPOLOGY=disaggregated
kubectl wait --for=condition=Available disaggregatedset/pd-disagg-tpu-sglang -n ${NAMESPACE} --timeout=1800s
```

On a node without the checkpoint, the first start downloads about 65 GB into the node's HuggingFace cache and then compiles
the TPU graphs (about 20 minutes in total). While the download is still running, a model server can restart once or twice
with `model.safetensors.index.json references ... shard file(s) missing`; it recovers by itself when the download completes.

**(Optional) Deploy the monitoring resources for model servers** (requires installing the monitoring stack mentioned in [Prerequisites](#prerequisites)):

<!-- guide:deploy.monitoring start -->
```bash
# only when MONITORING=true:
kubectl apply -n ${NAMESPACE} -k ${REPO_ROOT}/guides/recipes/modelserver/components/monitoring-pd
```
<!-- guide:deploy.monitoring end -->

## Verification

### 1. Get the IP of the Proxy

<!-- guide:verify.endpoint.standalone start -->
```bash
export IP=$(kubectl get service ${GUIDE_NAME}-epp -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')
```
<!-- guide:verify.endpoint.standalone end -->

### 2. Send Test Requests

**Send a completion request from a temporary pod inside the cluster:**

<!-- guide:verify.tests.request start -->
```bash
kubectl run curl-test --rm -i --restart=Never \
  --image=${CURL_TEST_IMAGE} \
  --namespace="${NAMESPACE}" \
  --env="IP=${IP}" \
  --env="MODEL=${MODEL}" \
  -- /bin/sh -c 'curl -sS -X POST "http://${IP}/v1/completions" -H "Content-Type: application/json" -d "{\"model\": \"${MODEL}\", \"prompt\": \"Explain how a simple agent loop works in 3 sentences.\"}"'
```
<!-- guide:verify.tests.request end -->

**Check that the router reaches a model server that serves `MODEL`:** the response must name the model and carry generated tokens:

<!-- guide:verify.tests.served_model start -->
```bash
# The router must reach a model server that serves MODEL and returns tokens
kubectl run serve-check --rm -i --restart=Never \
  --image=${CURL_TEST_IMAGE} \
  --namespace="${NAMESPACE}" \
  --env="IP=${IP}" \
  --env="MODEL=${MODEL}" \
  -- /bin/sh -c 'R=$(curl -sS -X POST "http://${IP}/v1/completions" -H "Content-Type: application/json" -d "{\"model\": \"${MODEL}\", \"prompt\": \"Say hello.\", \"max_tokens\": 16}")
    echo "${R}" | grep -q "\"model\":\"${MODEL}\"" && echo "${R}" | grep -q "\"choices\"" \
      && echo "OK: ${MODEL} served through the router" \
      || { echo "FAIL: ${R}"; exit 1; }'
```
<!-- guide:verify.tests.served_model end -->

## Benchmarking

This guide includes an `inference-perf` benchmark preset (defined in [inference-perf.yaml](inference-perf.yaml)) for multi-turn agentic workloads with shared and per-session system prompts, sized to the model's 131,072-token context:

| Workload Characteristic | Metric / Distribution Type | Min | Max | Mean / Constant | Std Dev | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Shared System Prompt** | Constant | - | - | 3,000 tokens | - | Instructions and tool schemas shared across all sessions. Highly cacheable. |
| **Dynamic System Prompt** | Lognormal | 2,000 | 64,000 | 16,000 tokens | 16,000 | Per-session context (repository files, retrieved documents). |
| **Turns per Conversation** | Lognormal | 1 | 200 | 30 turns | 30 | Depth of the agent loop; each turn re-sends the conversation so far, up to the 126,976-token prompt cap (context minus the 4,096-token output maximum). |
| **Input Tokens per Turn** | Lognormal | 100 | 10,000 | 1,500 tokens | 1,200 | New user or tool input appended per turn (test logs, diffs, follow-ups). |
| **Output Tokens per Turn** | Lognormal | 50 | 4,096 | 425 tokens | 825 | Model generation per turn. |
| **Tool Call Latency** | Lognormal | 1s | 100s | 15 seconds | 55 | Client-side idle time between turns. |

### 1. Run the Benchmark

[`inference-perf.yaml`](inference-perf.yaml) runs an
[`inference-perf`](https://github.com/kubernetes-sigs/inference-perf) job against the `${GUIDE_NAME}-epp` Service. Set the per-run parameters, render it, and apply it:

```bash
export CONCURRENCY_LEVEL=32
export NUM_REQUESTS=$((12 * CONCURRENCY_LEVEL))
export SEED=$((7 + CONCURRENCY_LEVEL))

kubectl delete job inference-perf -n ${NAMESPACE} --ignore-not-found
envsubst < ${REPO_ROOT}/guides/models/${GUIDE_NAME}/inference-perf.yaml | kubectl apply -n ${NAMESPACE} -f -
```

### 2. Collect the Results

Follow the job's logs until it completes:

```bash
kubectl logs -n ${NAMESPACE} job/inference-perf -f
```

> [!TIP]
> **Tuning Prefill-to-Decode Ratios:**
> The `sglang-disaggregated` overlay (`TOPOLOGY=disaggregated`) deploys 3 prefill + 1 decode. Scale roles independently by patching the `DisaggregatedSet` role replicas (role 0 = decode, role 1 = prefill):
>
> ```bash
> kubectl patch disaggregatedset/pd-disagg-tpu-sglang -n ${NAMESPACE} --type=json \
>   -p '[{"op": "replace", "path": "/spec/roles/0/spec/replicas", "value": 4},
>        {"op": "replace", "path": "/spec/roles/1/spec/replicas", "value": 2}]'
> ```

## Benchmark Results

Results for `openai/gpt-oss-120b` with SGLang on TPU 7x are being collected with the workload above for both topologies and will be
published here with the slice, image digest and benchmark parameters of each run.

## Cleanup

To remove the deployed components:

<!-- tabs:start group=topology -->
<details open>
<summary><b>Unified (Default)</b></summary>

<!-- guide:cleanup.modelserver.unified start -->
```bash
# only when TOPOLOGY=unified:
kubectl delete -n ${NAMESPACE} -k ${REPO_ROOT}/guides/models/${GUIDE_NAME}/modelserver/${ACCELERATOR_TYPE}/${MODEL_SERVER}/
```
<!-- guide:cleanup.modelserver.unified end -->

</details>
<details>
<summary><b>Disaggregated (Experimental)</b></summary>

<!-- guide:cleanup.modelserver.disaggregated start -->
<!-- llm-d-cicd:skip start -->
```bash
# only when TOPOLOGY=disaggregated:
kubectl delete -n ${NAMESPACE} -k ${REPO_ROOT}/guides/models/${GUIDE_NAME}/modelserver/${ACCELERATOR_TYPE}/${MODEL_SERVER}-disaggregated/
```
<!-- llm-d-cicd:skip end -->
<!-- guide:cleanup.modelserver.disaggregated end -->

</details>
<!-- tabs:end -->

<!-- guide:cleanup.rest start -->
```bash
helm uninstall ${GUIDE_NAME} -n ${NAMESPACE}

kubectl delete -n ${NAMESPACE} -k ${REPO_ROOT}/guides/recipes/modelserver/components/monitoring-pd --ignore-not-found=true
```
<!-- llm-d-cicd:skip start -->
```bash
kubectl delete namespace ${NAMESPACE}
```
<!-- llm-d-cicd:skip end -->
<!-- guide:cleanup.rest end -->

# Kimi-K3 on TPU 7x

This recipe is optimized for agentic code generation; see the
[agentic workloads section of the Models overview](../../../docs/well-lit-paths/models/README.md#agentic-workloads)
for the workload framing, and the [Qwen3.5-397B](../qwen3-5-397b/README.md) guide for the other
model served on TPU 7x against the same workload.

## Overview

This guide deploys [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) on Google
TPU 7x with vLLM on Torch TPU and the llm-d router. It composes the
[tiered prefix cache](../../tiered-prefix-cache/README.md) foundation with prefix-aware routing:

- **Tiered prefix cache (KV cache offloading)** to host DRAM: each data-parallel rank offloads
  evicted KV blocks (and Kimi-K3's KDA recurrent-state blocks) to a host-memory tier on its own
  host through `TPURaidenOffloadingConnector` (TPU Sync), and loads them back when a multi-turn
  or agentic conversation reuses the prefix instead of recomputing its prefill.
- **Tiered prefix-aware routing**: the router keeps one prefix index for what each endpoint holds
  in HBM and one for its host tier, and scores both together with queue depth and KV-cache
  utilization.
- **One endpoint per data-parallel rank**: Kimi-K3 runs as TP=8 × DP=4 with experts sharded over
  all 32 TPU devices of a `2x2x4` slice. vLLM's external data-parallel load balancing gives each
  rank its own API server, so the router, not vLLM's prefix-unaware internal DP balancer, decides
  which rank (and therefore which HBM cache and host tier) serves a request.

## Default Configuration

| Parameter | Value |
| --- | --- |
| Model | [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) (MXFP4 experts, FP8 attention, 1.56 TB) |
| Replicas | 1 LeaderWorkerSet group (4 pods) |
| Accelerator | Google TPU 7x (`tpu7x`) |
| Topology | `2x2x4` (4 hosts × 4 chips, 32 TPU devices) per replica |
| Parallelism | TP=8 × DP=4 × EP=32; one DP rank per host, each serving its own endpoint (`--data-parallel-rank`, external DP load balancing) |
| Context length | 262,144 tokens (fp8 KV cache) |
| KV block | `--block-size 256`, raised by vLLM to 1,360 tokens so a block holds one KDA state (TP=8, fp8 KV) |
| KV cache offload | `TPURaidenOffloadingConnector`, 256 GiB host tier per DP rank (1 TiB per replica) |
| Model weights | `ReadOnlyMany` Hyperdisk ML volume hydrated once from a GCS copy of the Hugging Face repo; each rank reads only its own experts |
| Model server | vLLM on Torch TPU (`docker.io/vllm/vllm-torchtpu:v0.32.0`) |

## Supported Accelerators and Model Servers

This guide includes configurations for the following accelerator and model server combinations:

<!-- guide:support start -->
| Accelerator | `ACCELERATOR_TYPE` | Served model | vLLM | Notes |
| --- | --- | --- | --- | --- |
| Google TPU v7x | `tpu` | `moonshotai/Kimi-K3` | 🟡 community | GKE only · Torch TPU vLLM · 1 LeaderWorkerSet replica on a `tpu7x` `2x2x4` slice (4 hosts × 4 chips): TP=8 × DP=4 × EP=32, one data-parallel rank per host, each serving its own endpoint (vLLM external DP load balancing) · fp8 KV cache, 262k context, 256 GiB host-DRAM KV tier per rank (`TPURaidenOffloadingConnector`) · weights on a ReadOnlyMany Hyperdisk ML volume hydrated from GCS |

✅ validated: covered by a nightly E2E workflow · 🟡 community: maintained by the hardware vendor or community, not covered by nightly E2E · ❌ not supported: tracked in the linked issue · — no configuration.
<!-- guide:support end -->

## Prerequisites

- Have the [proper client tools installed on your local system](../../../helpers/client-setup/README.md) to use this guide.

- A GKE cluster with a TPU 7x (`tpu7x`) `2x2x4` node pool (4 nodes of 4 chips) per replica, with
  [Workload Identity Federation for GKE](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity)
  enabled. Node boot disks do not need to hold the checkpoint: it lives on a Hyperdisk ML volume.
  Each node needs about 850 GiB of allocatable host memory (the 256 GiB host KV tier, the
  weight-loading buffers and `/dev/shm`).

- The [LeaderWorkerSet (LWS) controller](../../../docs/infrastructure/model-server-orchestration.md).

- A Google Cloud Storage bucket in the cluster's region, exported as `MODEL_BUCKET`, and a
  Hyperdisk ML quota of at least 3.4 TB in the TPU zone.

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
export GUIDE_NAME=kimi-k3
export NAMESPACE=llm-d-kimi-k3
export MONITORING=false # options: false, true
export MONITORING_VALUES=
export ACCELERATOR_TYPE=tpu # options: tpu
export MODEL_SERVER=vllm # options: vllm
export MODEL=moonshotai/Kimi-K3 # the served model name
export MODEL_BUCKET=my-model-bucket # GCS bucket that holds the moonshotai/Kimi-K3 copy
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

**Install the LeaderWorkerSet controller** (skip if your cluster already runs LWS):

<!-- guide:prerequisites.lws start -->
<!-- llm-d-cicd:skip start -->
```bash
helm upgrade --install lws oci://registry.k8s.io/lws/charts/lws \
  --version=${LWS_VERSION} \
  --namespace lws-system --create-namespace \
  --wait --timeout 300s
```
<!-- llm-d-cicd:skip end -->
<!-- guide:prerequisites.lws end -->

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

### Stage the weights

At 1.56 TB the checkpoint is larger than a TPU node's boot disk, and downloading it on every boot
would dominate start-up. Copy it once into `MODEL_BUCKET` (in the cluster's region):

<!-- guide:prerequisites.model_bucket start -->
<!-- llm-d-cicd:skip start -->
```bash
# One-time copy of the 1.56 TB checkpoint into a bucket in the cluster's zone's region.
hf download moonshotai/Kimi-K3 --local-dir ./Kimi-K3
gcloud storage cp -r ./Kimi-K3 gs://${MODEL_BUCKET}/moonshotai/
```
<!-- llm-d-cicd:skip end -->
<!-- guide:prerequisites.model_bucket end -->

Then create the model volume: a `ReadOnlyMany`
[Hyperdisk ML](https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/hyperdisk-ml)
PVC that the [GKE Volume Populator](https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/volume-populator)
hydrates from the bucket once. Every model-server pod mounts it read-only at `/models`, so a
restart reads the weights from local block storage instead of from GCS. Give the populator's
transfer job read access to the bucket:

<!-- guide:prerequisites.model_storage_access start -->
<!-- llm-d-cicd:skip start -->
```bash
# The GKE Volume Populator's transfer job reads the bucket as this Kubernetes ServiceAccount.
kubectl create serviceaccount kimi-k3-model-populator -n ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
export PROJECT_ID=$(gcloud config get-value project)
export PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)')
gcloud storage buckets add-iam-policy-binding gs://${MODEL_BUCKET} \
  --member="principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/${NAMESPACE}/sa/kimi-k3-model-populator" \
  --role=roles/storage.objectViewer
```
<!-- llm-d-cicd:skip end -->
<!-- guide:prerequisites.model_storage_access end -->

**Create the StorageClass, data source and PVC.** The PVC is sized for the checkpoint plus the
megakernel pre-shard that `TOPOLOGY=disaggregated` reads (see below), and hydrates when the first
pod that mounts it is scheduled:

<!-- guide:prerequisites.model_storage start -->
<!-- llm-d-cicd:skip start -->
```bash
# A ReadOnlyMany Hyperdisk ML volume that every model-server pod mounts at /models,
# hydrated once from gs://${MODEL_BUCKET}/moonshotai by the GKE Volume Populator.
kubectl apply -n ${NAMESPACE} -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: kimi-k3-hyperdisk-ml
provisioner: pd.csi.storage.gke.io
parameters:
  type: hyperdisk-ml
  provisioned-throughput-on-create: "6400Mi"
allowVolumeExpansion: false
volumeBindingMode: WaitForFirstConsumer
---
apiVersion: datalayer.gke.io/v1
kind: GCPDataSource
metadata:
  name: kimi-k3-models
spec:
  cloudStorage:
    serviceAccountName: kimi-k3-model-populator
    uri: gs://${MODEL_BUCKET}/moonshotai/
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: kimi-k3-models
spec:
  accessModes: [ReadOnlyMany]
  storageClassName: kimi-k3-hyperdisk-ml
  resources:
    requests:
      storage: 3400Gi
  dataSourceRef:
    apiGroup: datalayer.gke.io
    kind: GCPDataSource
    name: kimi-k3-models
EOF
```
<!-- llm-d-cicd:skip end -->
<!-- guide:prerequisites.model_storage end -->

## Installation Instructions

### 1. Deploy the llm-d Router

**Prepare the paths to the `helm` values files** for the `llm-d` router:

<!-- guide:deploy.router_values start -->
```bash
export ROUTER_BASE_VALUES="${REPO_ROOT}/guides/recipes/router/base.values.yaml"
export ROUTER_VALUES="${REPO_ROOT}/guides/models/${GUIDE_NAME}/router/${GUIDE_NAME}.values.yaml"
```
<!-- guide:deploy.router_values end -->

**(Optional) Enable Prometheus monitoring on the `llm-d` router** by defining the `helm` values file (requires installing the monitoring stack mentioned in [Prerequisites](#prerequisites)):

<!-- guide:deploy.monitoring_values start -->
```bash
# only when MONITORING=true:
export MONITORING_VALUES="-f ${REPO_ROOT}/guides/recipes/router/features/monitoring.values.yaml"
```
<!-- guide:deploy.monitoring_values end -->

> [!NOTE]
> The router's block size (`blockSizeTokens: 1360`) must equal the KV block the engine settles on,
> which depends on the TP size and KV dtype. The engine logs it at start-up; change both together.
> The host-tier index (`cpu-prefix-cache-producer`) is sized for the 256 GiB host tier per rank; if
> you change `cpu_bytes_to_use`, scale `lruCapacityPerServer` with it.

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

**Apply the Kustomize overlay:**

<!-- guide:deploy.modelserver start -->
```bash
kubectl apply -n ${NAMESPACE} \
  -k ${REPO_ROOT}/guides/models/${GUIDE_NAME}/modelserver/${ACCELERATOR_TYPE}/${MODEL_SERVER}/
```
<!-- guide:deploy.modelserver end -->

Wait for the model server to become ready. The first start also hydrates the model volume; a
cold start then reads about 400 GB per host and compiles; budget about 30 minutes (weights ~3 min, compilation ~15 min, engine and host-tier start-up):

```bash
kubectl wait --for=condition=Available leaderworkerset/kimi-k3-tpu-vllm-decode -n ${NAMESPACE} --timeout=7200s
```

**(Optional) Deploy the monitoring resources for model servers** (requires installing the monitoring stack mentioned in [Prerequisites](#prerequisites)):

<!-- guide:deploy.monitoring start -->
```bash
# only when MONITORING=true:
kubectl apply -n ${NAMESPACE} -k ${REPO_ROOT}/guides/recipes/modelserver/components/monitoring
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

**Check the host KV tier.** Each rank logs its host tier as it starts
(`TPURaidenOffloadingConnector: rank N registered with offload controller ... (1644 host kernel blocks)`),
and exports `vllm:prefix_cache_hits_total` (HBM) and `vllm:external_prefix_cache_hits_total`
(host tier). Send a long prompt twice, push it out of HBM with other long prompts, and send it
again: the third request's prompt tokens are counted in `external_prefix_cache_hits_total`.

```bash
kubectl logs -n ${NAMESPACE} kimi-k3-tpu-vllm-decode-0 | grep "registered with offload controller" | head -1
```

## Benchmarking

This guide comes with an `inference-perf` benchmark preset (defined in [inference-perf.yaml](inference-perf.yaml)),
the same agentic code-generation workload as the [Qwen3.5-397B](../qwen3-5-397b/README.md#benchmarking)
guide: multi-turn sessions over a shared system prompt, a large per-session repository context,
and tool-call pauses between turns.

```bash
# CONCURRENCY_LEVEL is the number of concurrent coding sessions; NUM_REQUESTS is 20 per session;
# SEED is varied per concurrency level so prompts don't overlap across runs.
export CONCURRENCY_LEVEL=16
export NUM_REQUESTS=$((20 * CONCURRENCY_LEVEL))
export SEED=$((7 + CONCURRENCY_LEVEL))

kubectl delete job inference-perf -n ${NAMESPACE} --ignore-not-found
envsubst < ${REPO_ROOT}/guides/models/${GUIDE_NAME}/inference-perf.yaml | kubectl apply -n ${NAMESPACE} -f -
kubectl logs -n ${NAMESPACE} job/inference-perf -f
```

## Benchmark Results

Results for `moonshotai/Kimi-K3` are being collected with the workload above on one `tpu7x`
`2x2x4` replica (`docker.io/vllm/vllm-torchtpu:v0.32.0`, the overlay in
[`modelserver/tpu/vllm`](modelserver/tpu/vllm/)) and will be published here with the slice, image
digest and benchmark parameters of each run.

## Cleanup

To remove the deployed components:

<!-- guide:cleanup.modelserver start -->
```bash
kubectl delete -n ${NAMESPACE} -k ${REPO_ROOT}/guides/models/${GUIDE_NAME}/modelserver/${ACCELERATOR_TYPE}/${MODEL_SERVER}/
```
<!-- guide:cleanup.modelserver end -->

<!-- guide:cleanup.rest start -->
```bash
helm uninstall ${GUIDE_NAME} -n ${NAMESPACE}

kubectl delete -n ${NAMESPACE} -k ${REPO_ROOT}/guides/recipes/modelserver/components/monitoring --ignore-not-found=true
```
<!-- llm-d-cicd:skip start -->
```bash
# The Hyperdisk ML volume holds a copy of the weights; delete it when you no longer need it.
kubectl delete pvc kimi-k3-models -n ${NAMESPACE} --ignore-not-found=true
kubectl delete gcpdatasource kimi-k3-models -n ${NAMESPACE} --ignore-not-found=true

kubectl delete namespace ${NAMESPACE}
```
<!-- llm-d-cicd:skip end -->
<!-- guide:cleanup.rest end -->

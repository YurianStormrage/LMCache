source ../.venv/bin/activate

USE_VLLM_V1=1 \
LMCACHE_USE_EXPERIMENTAL=Ture \
LMCACHE_CONFIG_FILE="./pfs.yaml" \
LMCACHE_LOG_LEVEL=debug \
vllm serve \
  "/data1/pqa/models/Qwen3-32B" \
  --dtype "bfloat16" \
  --port 8011 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  --disable-log-requests \
  --max-model-len 33000 \
  --gpu_memory_utilization 0.8 \
  > /tmp/lmcache_log 2>&1 &

# Other models
  #"/data1/pqa/models/Meta-Llama-3.1-8B-Instruct/" \


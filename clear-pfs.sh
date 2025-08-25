for i in {0..9}; do
    rm /data3/stage/vllm\@_data1_pqa_models_Qwen3-32B\@1\@0\@-${i}*
    rm /data3/stage/vllm\@_data1_pqa_models_Qwen3-32B\@1\@0\@${i}*
    rm /data3/stage/vllm\@-data1-pqa-models-Qwen3-32B\@1\@0\@-${i}*
    rm /data3/stage/vllm\@-data1-pqa-models-Qwen3-32B\@1\@0\@${i}*
done


c2s \
  --cuda-include-path=/home/baodi/workspace/cuda_12.0 \
  --extra-arg="-I /home/baodi/workspace/DeepSpeed/csrc/transformer/inference/includes" \
  --extra-arg="-I /home/baodi/workspace/DeepSpeed/csrc/includes" \
  --extra-arg="-DBF16_AVAILABLE" \
  --in-root=/home/baodi/workspace/DeepSpeed \
  --out-root=/home/baodi/workspace/DeepSpeed/dpct_test \
  /home/baodi/workspace/DeepSpeed/csrc/transformer/inference/includes/inference_context.h

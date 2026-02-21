import torch

print("CUDA Available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("Memory (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)

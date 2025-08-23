import torch

if torch.cuda.is_available():
    print("CUDA is available ✅")
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("Total Memory (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
    print("CUDA Capability:", torch.cuda.get_device_capability(0))
else:
    print("CUDA is NOT available ❌")

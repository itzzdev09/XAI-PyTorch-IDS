import torch
import time

# Size of matrix (increase for heavier test)
N = 10000  

# CPU Test
a_cpu = torch.randn(N, N)
b_cpu = torch.randn(N, N)

start = time.time()
c_cpu = torch.mm(a_cpu, b_cpu)  # Matrix multiplication
end = time.time()

print(f"CPU Time: {end - start:.4f} seconds")

# GPU Test (if available)
if torch.cuda.is_available():
    device = torch.device("cuda")
    a_gpu = a_cpu.to(device)
    b_gpu = b_cpu.to(device)

    torch.cuda.synchronize()  # Ensure GPU is ready
    start = time.time()
    c_gpu = torch.mm(a_gpu, b_gpu)
    torch.cuda.synchronize()  # Wait for GPU to finish
    end = time.time()

    print(f"GPU Time: {end - start:.4f} seconds")
    print("Speedup:", ( (end - start) / (end - start) ) )
else:
    print("CUDA is NOT available ❌")

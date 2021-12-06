import minitorch
import time

backend = minitorch.make_tensor_backend(minitorch.CudaOps, is_cuda=True)
# backend = minitorch.make_tensor_backend(minitorch.FastOps, is_cuda=False)

total = 0
for j in range(50):
    a = minitorch.rand((1000, 1000), backend=backend)
    start = time.time()
    a.sum(1)
    end = time.time()
    if j != 0:
        total += end - start
print(total / 50)

import minitorch

FastTensorBackend = minitorch.make_tensor_backend(minitorch.FastOps)
CUDA = minitorch.make_tensor_backend(minitorch.CudaOps, is_cuda=True)
# Backend = FastTensorBackend
Backend = CUDA
A = minitorch.rand((1, 5), requires_grad=True, backend=Backend)
B = minitorch.rand((5, 1), requires_grad=True, backend=Backend)

(A + B).sum().backward()

print(A.grad)
print(B.grad)

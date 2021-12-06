# from .tensor import rand
# from .functions import matmul, conv2d
# from .module import Module, Parameter


# class tLinear(Module):
#     def __init__(self, in_size, out_size):
#         super().__init__()
#         self.weights = Parameter(rand((in_size, out_size)))
#         self.bias = Parameter(rand((out_size,)))
#         self.out_size = out_size

#     def forward(self, x):
#         batch, in_size = x.shape
#         return (
#             self.weights.value.view(1, in_size, self.out_size)
#             * x.view(batch, in_size, 1)
#         ).sum(1).view(batch, self.out_size) + self.bias.value.view(1, self.out_size)


# class tLinear2(Module):
#     def __init__(self, in_size, out_size):
#         super().__init__()
#         self.weights = Parameter(rand((in_size, out_size)))
#         self.bias = Parameter(rand((out_size,)))
#         self.out_size = out_size

#     def forward(self, x):
#         batch, in_size = x.shape
#         return matmul(x, self.weights.value) + self.bias.value.view(1, self.out_size)


# class Dropout(Module):
#     def __init__(self, rate):
#         super().__init__()
#         self.rate = rate

#     def forward(self, x):
#         return (rand(x.shape) / 2 + 0.5 < self.rate) * x


# class Conv2d(Module):
#     def __init__(self, in_features, out_features, size):
#         super().__init__()
#         size1 = [size[0], size[1], in_features, out_features]
#         size2 = [size[0], size[1], out_features]
#         self.weights = Parameter(rand(size1))
#         self.bias = Parameter(rand(size2))

#     def forward(self, x):
#         return conv2d(x, self.weights.value, self.bias.value)


# # class MaxPool2d(Module):
# #     def __init__(self, in_features, out_features, size):
# #         super().__init__()


# #     def forward(self, x):
# #         return conv2d(x, self.weights.value, self.bias.value)

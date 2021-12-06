from hypothesis import given
from hypothesis.strategies import integers, lists
from .strategies import scalars
import random
import minitorch


class Network(minitorch.Module):
    def __init__(self):
        super().__init__()
        self.layer = ScalarLinear(2, 1)


class Network2(minitorch.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ScalarLinear(2, 2)
        self.layer2 = ScalarLinear(2, 1)


class ScalarLinear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = []
        self.bias = []
        for i in range(in_size):
            self.weights.append([])
            for j in range(out_size):
                self.weights[i].append(
                    self.add_parameter(
                        f"weight_{i}_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                    )
                )
        for j in range(out_size):
            self.bias.append(
                self.add_parameter(
                    f"bias_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                )
            )

    def forward(self, inputs):
        y = [b.value for b in self.bias]
        for i, x in enumerate(inputs):
            for j in range(len(y)):
                y[j] = y[j] + x * self.weights[i][j].value
        return y


@given(lists(scalars(), max_size=10), integers(min_value=5, max_value=20))
def test_linear(inputs, out_size):
    lin = ScalarLinear(len(inputs), out_size)
    mid = lin.forward(inputs)
    lin2 = ScalarLinear(out_size, 1)
    lin2.forward(mid)


# @given(
#     lists(scalars(min_value=-10, max_value=10), min_size=2, max_size=2),
#     lists(scalars(min_value=-10, max_value=10), min_size=2, max_size=2),
# )
# def test_nn2(inputs, bias):
#     model = Network2()

#     def check(x1, x2, b1, b2):
#         model.layer1.bias[0].update(b1)
#         model.layer1.bias[1].update(b2)
#         return model.forward([x1, x2])

#     minitorch.derivative_check(check, *(inputs + bias))


def test_nn_size():
    model = Network2()
    assert len(model.parameters()) == (
        len(model.layer1.parameters()) + len(model.layer2.parameters())
    )

    assert model.layer2.bias[0].value.data != 0
    assert model.layer1.bias[0].value.data != 0
    assert model.layer1.weights[0][0].value.data != 0

    for p in model.parameters():
        p.update(minitorch.Scalar(0))

    assert model.layer2.bias[0].value.data == 0
    assert model.layer1.bias[0].value.data == 0
    assert model.layer1.weights[0][0].value.data == 0

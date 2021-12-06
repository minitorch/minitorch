import minitorch
import pytest
from minitorch import History

# ## Task 1.3 - Tests for the autodifferentiation machinery.

# Simple sanity check and debugging tests.


class Function1(minitorch.ScalarFunction):
    @staticmethod
    def forward(ctx, x, y):
        ":math:`f(x, y) = x + y + 10`"
        return x + y + 10

    @staticmethod
    def backward(ctx, d_output):
        "Derivatives are :math:`f'_x(x, y) = 1` and :math:`f'_y(x, y) = 1`"
        return d_output, d_output


class Function2(minitorch.ScalarFunction):
    @staticmethod
    def forward(ctx, x, y):
        ":math:`f(x, y) = x \timex y + x`"
        ctx.save_for_backward(x, y)
        return x * y + x

    @staticmethod
    def backward(ctx, d_output):
        "Derivatives are :math:`f'_x(x, y) = y + 1` and :math:`f'_y(x, y) = x`"
        x, y = ctx.saved_values
        return d_output * (y + 1), d_output * x


# Checks for the chain rule function.


@pytest.mark.task1_3
def test_chain_rule1():
    "Check that constants are ignored."
    constant = minitorch.Variable(None)
    back = Function1.chain_rule(ctx=None, inputs=[constant, constant], d_output=5)
    assert len(list(back)) == 0


@pytest.mark.task1_3
def test_chain_rule2():
    "Check that constrants are ignored and variables get derivatives."
    var = minitorch.Variable(History())
    constant = minitorch.Variable(None)
    back = Function1.chain_rule(ctx=None, inputs=[var, constant], d_output=5)
    back = list(back)
    assert len(back) == 1
    variable, deriv = back[0]
    assert variable.name == var.name
    assert deriv == 5


@pytest.mark.task1_3
def test_chain_rule3():
    "Check that constrants are ignored and variables get derivatives."
    constant = 10
    var = minitorch.Scalar(5)

    ctx = minitorch.Context()
    Function2.forward(ctx, constant, var.data)

    back = Function2.chain_rule(ctx=ctx, inputs=[constant, var], d_output=5)
    back = list(back)
    assert len(back) == 1
    variable, deriv = back[0]
    assert variable.name == var.name
    assert deriv == 5 * 10


@pytest.mark.task1_3
def test_chain_rule4():
    var1 = minitorch.Scalar(5)
    var2 = minitorch.Scalar(10)

    ctx = minitorch.Context()
    Function2.forward(ctx, var1.data, var2.data)

    back = Function2.chain_rule(ctx=ctx, inputs=[var1, var2], d_output=5)
    back = list(back)
    assert len(back) == 2
    variable, deriv = back[0]
    assert variable.name == var1.name
    assert deriv == 5 * (10 + 1)
    variable, deriv = back[1]
    assert variable.name == var2.name
    assert deriv == 5 * 5


# ## Task 1.4 - Run some simple backprop tests

# Main tests are in test_scalar.py


@pytest.mark.task1_4
def test_backprop1():
    # Example 1: F1(0, v)
    var = minitorch.Scalar(0)
    var2 = Function1.apply(0, var)
    var2.backward(d_output=5)
    assert var.derivative == 5


@pytest.mark.task1_4
def test_backprop2():
    # Example 2: F1(0, 0)
    var = minitorch.Scalar(0)
    var2 = Function1.apply(0, var)
    var3 = Function1.apply(0, var2)
    var3.backward(d_output=5)
    assert var.derivative == 5


@pytest.mark.task1_4
def test_backprop3():
    # Example 3: F1(F1(0, v1), F1(0, v1))
    var1 = minitorch.Scalar(0)
    var2 = Function1.apply(0, var1)
    var3 = Function1.apply(0, var1)
    var4 = Function1.apply(var2, var3)
    var4.backward(d_output=5)
    assert var1.derivative == 10


@pytest.mark.task1_4
def test_backprop4():
    # Example 4: F1(F1(0, v1), F1(0, v1))
    var0 = minitorch.Scalar(0)
    var1 = Function1.apply(0, var0)
    var2 = Function1.apply(0, var1)
    var3 = Function1.apply(0, var1)
    var4 = Function1.apply(var2, var3)
    var4.backward(d_output=5)
    assert var0.derivative == 10

from minitorch import grad_check, tensor
import pytest
from hypothesis import given
from hypothesis.strategies import lists, data, permutations
from .strategies import tensors, shaped_tensors, assert_close, small_floats
from minitorch import MathTestVariable

one_arg, two_arg, red_arg = MathTestVariable._tests()


@given(lists(small_floats, min_size=1))
def test_create(t1):
    t2 = tensor(t1)
    for i in range(len(t1)):
        assert t1[i] == t2[i]


@given(tensors())
@pytest.mark.task2_3
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(fn, t1):
    name, base_fn, tensor_fn = fn
    t2 = tensor_fn(t1)
    for ind in t2._tensor.indices():
        assert_close(t2[ind], base_fn(t1[ind]))


@given(shaped_tensors(2))
@pytest.mark.task2_3
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(fn, ts):
    name, base_fn, tensor_fn = fn
    t1, t2 = ts
    t3 = tensor_fn(t1, t2)
    for ind in t3._tensor.indices():
        assert_close(t3[ind], base_fn(t1[ind], t2[ind]))


@given(tensors())
@pytest.mark.task2_4
@pytest.mark.parametrize("fn", one_arg)
def test_one_derivative(fn, t1):
    name, _, tensor_fn = fn
    grad_check(tensor_fn, t1)


@given(data(), tensors())
@pytest.mark.task2_4
def test_permute(data, t1):
    permutation = data.draw(permutations(range(len(t1.shape))))

    def permute(a):
        return a.permute(*permutation)

    grad_check(permute, t1)


def test_grad_size():
    "Check that extra grad dim is removed (from @WannaFy)"
    a = tensor([1], requires_grad=True)
    b = tensor([[1, 1]], requires_grad=True)

    c = (a * b).sum()

    c.backward()
    assert c.shape == (1,)
    assert a.shape == a.grad.shape
    assert b.shape == b.grad.shape


@given(tensors())
@pytest.mark.task2_4
@pytest.mark.parametrize("fn", red_arg)
def test_reduce(fn, t1):
    name, _, tensor_fn = fn
    grad_check(tensor_fn, t1)


@given(shaped_tensors(2))
@pytest.mark.task2_4
@pytest.mark.parametrize("fn", two_arg)
def test_two_grad(fn, ts):
    name, _, tensor_fn = fn
    t1, t2 = ts
    grad_check(tensor_fn, t1, t2)


@given(shaped_tensors(2))
@pytest.mark.task2_4
@pytest.mark.parametrize("fn", two_arg)
def test_two_grad_broadcast(fn, ts):
    name, base_fn, tensor_fn = fn
    t1, t2 = ts
    grad_check(tensor_fn, t1, t2)

    # broadcast check
    grad_check(tensor_fn, t1.sum(0), t2)
    grad_check(tensor_fn, t1, t2.sum(0))


def test_fromlist():
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)
    t = tensor([[[2, 3, 4], [4, 5, 7]]])
    assert t.shape == (1, 2, 3)


def test_view():
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)
    t2 = t.view(6)
    assert t2.shape == (6,)
    t2 = t2.view(1, 6)
    assert t2.shape == (1, 6)
    t2 = t2.view(6, 1)
    assert t2.shape == (6, 1)
    t2 = t2.view(2, 3)
    assert t.is_close(t2).all().item() == 1.0


@given(tensors())
def test_back_view(t1):
    def view(a):
        a = a.contiguous()
        return a.view(a.size)

    grad_check(view, t1)


@pytest.mark.xfail
def test_permute_view():
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)
    t2 = t.permute(1, 0)
    t2.view(6)


@pytest.mark.xfail
def test_index():
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)
    t[50, 2]


def test_fromnumpy():
    t = tensor([[2, 3, 4], [4, 5, 7]])
    print(t)
    assert t.shape == (2, 3)
    n = t.to_numpy()
    t2 = tensor(n.tolist())
    for ind in t._tensor.indices():
        assert t[ind] == t2[ind]


## Student Submitted Tests


@pytest.mark.task2_3
def test_reduce_forward_one_dim():
    # shape (3, 2)
    t = tensor([[2, 3], [4, 6], [5, 7]])

    # here 0 means to reduce the 0th dim, 3 -> nothing
    t_summed = t.sum(0)

    # shape (2)
    t_sum_expected = tensor([[11, 16]])
    assert t_summed.is_close(t_sum_expected).all().item()


@pytest.mark.task2_3
def test_reduce_forward_one_dim_2():
    # shape (3, 2)
    t = tensor([[2, 3], [4, 6], [5, 7]])

    # here 1 means reduce the 1st dim, 2 -> nothing
    t_summed_2 = t.sum(1)

    # shape (3)
    t_sum_2_expected = tensor([[5], [10], [12]])
    assert t_summed_2.is_close(t_sum_2_expected).all().item()


@pytest.mark.task2_3
def test_reduce_forward_all_dims():
    # shape (3, 2)
    t = tensor([[2, 3], [4, 6], [5, 7]])

    # reduce all dims, (3 -> 1, 2 -> 1)
    t_summed_all = t.sum()

    # shape (1, 1)
    t_summed_all_expected = tensor([27])

    assert_close(t_summed_all[0], t_summed_all_expected[0])

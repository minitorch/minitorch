import minitorch
from hypothesis import settings
from hypothesis.strategies import composite, floats, integers, lists, permutations

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


small_ints = integers(min_value=1, max_value=3)
small_floats = floats(min_value=-100, max_value=100, allow_nan=False)
med_ints = integers(min_value=1, max_value=20)


@composite
def vals(draw, size, number):
    pts = draw(lists(number, min_size=size, max_size=size,))
    return minitorch.tensor(pts)


@composite
def scalars(draw, min_value=-100000, max_value=100000):
    val = draw(floats(min_value=min_value, max_value=max_value))
    return minitorch.Scalar(val)


small_scalars = scalars(min_value=-100, max_value=100)


@composite
def shapes(draw):
    lsize = draw(lists(small_ints, min_size=1, max_size=4))
    return tuple(lsize)


@composite
def tensor_data(draw, numbers=floats(), shape=None):
    if shape is None:
        shape = draw(shapes())
    size = int(minitorch.prod(shape))
    data = draw(lists(numbers, min_size=size, max_size=size))
    permute = draw(permutations(range(len(shape))))
    permute_shape = tuple([shape[i] for i in permute])
    reverse_permute = [a[0] for a in sorted(enumerate(permute), key=lambda a: a[1])]
    td = minitorch.TensorData(data, permute_shape)
    ret = td.permute(*reverse_permute)
    assert ret.shape[0] == shape[0]
    return ret


@composite
def indices(draw, layout):
    return tuple((draw(integers(min_value=0, max_value=s - 1)) for s in layout.shape))


@composite
def tensors(
    draw,
    numbers=floats(allow_nan=False, min_value=-100, max_value=100),
    backend=None,
    shape=None,
):
    backend = minitorch.TensorFunctions if backend is None else backend
    td = draw(tensor_data(numbers, shape=shape))
    return minitorch.Tensor(td, backend=backend)


@composite
def shaped_tensors(
    draw,
    n,
    numbers=floats(allow_nan=False, min_value=-100, max_value=100),
    backend=None,
):
    backend = minitorch.TensorFunctions if backend is None else backend
    td = draw(tensor_data(numbers))
    values = []
    for i in range(n):
        data = draw(lists(numbers, min_size=td.size, max_size=td.size))
        values.append(
            minitorch.Tensor(
                minitorch.TensorData(data, td.shape, td.strides), backend=backend
            )
        )
    return values


@composite
def matmul_tensors(
    draw, numbers=floats(allow_nan=False, min_value=-100, max_value=100)
):

    i, j, k = [draw(integers(min_value=1, max_value=10)) for _ in range(3)]

    l1 = (i, j)
    l2 = (j, k)
    values = []
    for shape in [l1, l2]:
        size = int(minitorch.prod(shape))
        data = draw(lists(numbers, min_size=size, max_size=size))
        values.append(minitorch.Tensor(minitorch.TensorData(data, shape)))
    return values


def assert_close(a, b):
    assert minitorch.operators.is_close(a, b), "Failure x=%f y=%f" % (a, b)


def assert_close_tensor(a, b):
    if a.is_close(b).all().item() != 1.0:
        assert False, (
            "Tensors are not close \n x.shape=%s \n x=%s \n y.shape=%s \n y=%s \n Diff=%s %s"
            % (a.shape, a, b.shape, b, a - b, a.is_close(b))
        )

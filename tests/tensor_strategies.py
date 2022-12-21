from typing import List, Optional

from hypothesis import settings
from hypothesis.strategies import (
    DrawFn,
    SearchStrategy,
    composite,
    floats,
    integers,
    lists,
    permutations,
)

import minitorch
from minitorch import Tensor, TensorBackend, TensorData, UserIndex, UserShape

from .strategies import small_ints

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


@composite
def vals(draw: DrawFn, size: int, number: SearchStrategy[float]) -> Tensor:
    pts = draw(
        lists(
            number,
            min_size=size,
            max_size=size,
        )
    )
    return minitorch.tensor(pts)


@composite
def shapes(draw: DrawFn) -> minitorch.UserShape:
    lsize = draw(lists(small_ints, min_size=1, max_size=4))
    return tuple(lsize)


@composite
def tensor_data(
    draw: DrawFn,
    numbers: SearchStrategy[float] = floats(),
    shape: Optional[UserShape] = None,
) -> TensorData:
    if shape is None:
        shape = draw(shapes())
    size = int(minitorch.prod(shape))
    data = draw(lists(numbers, min_size=size, max_size=size))
    permute: List[int] = draw(permutations(range(len(shape))))
    permute_shape = tuple([shape[i] for i in permute])
    z = sorted(enumerate(permute), key=lambda a: a[1])
    reverse_permute = [a[0] for a in z]
    td = minitorch.TensorData(data, permute_shape)
    ret = td.permute(*reverse_permute)
    assert ret.shape[0] == shape[0]
    return ret


@composite
def indices(draw: DrawFn, layout: Tensor) -> UserIndex:
    return tuple((draw(integers(min_value=0, max_value=s - 1)) for s in layout.shape))


@composite
def tensors(
    draw: DrawFn,
    numbers: SearchStrategy[float] = floats(
        allow_nan=False, min_value=-100, max_value=100
    ),
    backend: Optional[TensorBackend] = None,
    shape: Optional[UserShape] = None,
) -> Tensor:
    backend = minitorch.SimpleBackend if backend is None else backend
    td = draw(tensor_data(numbers, shape=shape))
    return minitorch.Tensor(td, backend=backend)


@composite
def shaped_tensors(
    draw: DrawFn,
    n: int,
    numbers: SearchStrategy[float] = floats(
        allow_nan=False, min_value=-100, max_value=100
    ),
    backend: Optional[TensorBackend] = None,
) -> List[Tensor]:
    backend = minitorch.SimpleBackend if backend is None else backend
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
    draw: DrawFn,
    numbers: SearchStrategy[float] = floats(
        allow_nan=False, min_value=-100, max_value=100
    ),
) -> List[Tensor]:

    i, j, k = [draw(integers(min_value=1, max_value=10)) for _ in range(3)]

    l1 = (i, j)
    l2 = (j, k)
    values = []
    for shape in [l1, l2]:
        size = int(minitorch.prod(shape))
        data = draw(lists(numbers, min_size=size, max_size=size))
        values.append(minitorch.Tensor(minitorch.TensorData(data, shape)))
    return values


def assert_close_tensor(a: Tensor, b: Tensor) -> None:
    if a.is_close(b).all().item() != 1.0:
        assert False, (
            "Tensors are not close \n x.shape=%s \n x=%s \n y.shape=%s \n y=%s \n Diff=%s %s"
            % (a.shape, a, b.shape, b, a - b, a.is_close(b))
        )

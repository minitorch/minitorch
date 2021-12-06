import random
from .operators import prod
from numpy import array, float64, ndarray
import numba

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


def index_to_position(index, strides):
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index (array-like): index tuple of ints
        strides (array-like): tensor strides

    Returns:
        int : position in storage
    """

    # TODO: Implement for Task 2.1.
    raise NotImplementedError('Need to implement for Task 2.1')


def to_index(ordinal, shape, out_index):
    """
    Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        ordinal (int): ordinal position to convert.
        shape (tuple): tensor shape.
        out_index (array): the index corresponding to position.

    Returns:
      None : Fills in `out_index`.

    """
    # TODO: Implement for Task 2.1.
    raise NotImplementedError('Need to implement for Task 2.1')


def broadcast_index(big_index, big_shape, shape, out_index):
    """
    Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
        big_index (array-like): multidimensional index of bigger tensor
        big_shape (array-like): tensor shape of bigger tensor
        shape (array-like): tensor shape of smaller tensor
        out_index (array-like): multidimensional index of smaller tensor

    Returns:
        None : Fills in `out_index`.
    """
    # TODO: Implement for Task 2.2.
    raise NotImplementedError('Need to implement for Task 2.2')


def shape_broadcast(shape1, shape2):
    """
    Broadcast two shapes to create a new union shape.

    Args:
        shape1 (tuple) : first shape
        shape2 (tuple) : second shape

    Returns:
        tuple : broadcasted shape

    Raises:
        IndexingError : if cannot broadcast
    """
    # TODO: Implement for Task 2.2.
    raise NotImplementedError('Need to implement for Task 2.2')


def strides_from_shape(shape):
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    def __init__(self, storage, shape, strides=None):
        if isinstance(storage, ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self):  # pragma: no cover
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self):
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a, shape_b):
        return shape_broadcast(shape_a, shape_b)

    def index(self, index):
        if isinstance(index, int):
            index = array([index])
        if isinstance(index, tuple):
            index = array(index)

        # Check for errors
        if index.shape[0] != len(self.shape):
            raise IndexingError(f"Index {index} must be size of {self.shape}.")
        for i, ind in enumerate(index):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {index} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {index} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self):
        lshape = array(self.shape)
        out_index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self):
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key):
        return self._storage[self.index(key)]

    def set(self, key, val):
        self._storage[self.index(key)] = val

    def tuple(self):
        return (self._storage, self._shape, self._strides)

    def permute(self, *order):
        """
        Permute the dimensions of the tensor.

        Args:
            order (list): a permutation of the dimensions

        Returns:
            :class:`TensorData`: a new TensorData with the same storage and a new dimension order.
        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        # TODO: Implement for Task 2.1.
        raise NotImplementedError('Need to implement for Task 2.1')

    def to_string(self):
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s

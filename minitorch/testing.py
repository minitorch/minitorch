# type: ignore

from typing import Callable, Generic, Iterable, Tuple, TypeVar

import minitorch.operators as operators

A = TypeVar("A")


class MathTest(Generic[A]):
    @staticmethod
    def neg(a: A) -> A:
        "Negate the argument"
        return -a

    @staticmethod
    def addConstant(a: A) -> A:
        "Add contant to the argument"
        return 5 + a

    @staticmethod
    def square(a: A) -> A:
        "Manual square"
        return a * a

    @staticmethod
    def cube(a: A) -> A:
        "Manual cube"
        return a * a * a

    @staticmethod
    def subConstant(a: A) -> A:
        "Subtract a constant from the argument"
        return a - 5

    @staticmethod
    def multConstant(a: A) -> A:
        "Multiply a constant to the argument"
        return 5 * a

    @staticmethod
    def div(a: A) -> A:
        "Divide by a constant"
        return a / 5

    @staticmethod
    def inv(a: A) -> A:
        "Invert after adding"
        return operators.inv(a + 3.5)

    @staticmethod
    def sig(a: A) -> A:
        "Apply sigmoid"
        return operators.sigmoid(a)

    @staticmethod
    def log(a: A) -> A:
        "Apply log to a large value"
        return operators.log(a + 100000)

    @staticmethod
    def relu(a: A) -> A:
        "Apply relu"
        return operators.relu(a + 5.5)

    @staticmethod
    def exp(a: A) -> A:
        "Apply exp to a smaller value"
        return operators.exp(a - 200)

    @staticmethod
    def explog(a: A) -> A:
        return operators.log(a + 100000) + operators.exp(a - 200)

    @staticmethod
    def add2(a: A, b: A) -> A:
        "Add two arguments"
        return a + b

    @staticmethod
    def mul2(a: A, b: A) -> A:
        "Mul two arguments"
        return a * b

    @staticmethod
    def div2(a: A, b: A) -> A:
        "Divide two arguments"
        return a / (b + 5.5)

    @staticmethod
    def gt2(a: A, b: A) -> A:
        return operators.lt(b, a + 1.2)

    @staticmethod
    def lt2(a: A, b: A) -> A:
        return operators.lt(a + 1.2, b)

    @staticmethod
    def eq2(a: A, b: A) -> A:
        return operators.eq(a, (b + 5.5))

    @staticmethod
    def sum_red(a: Iterable[A]) -> A:
        return operators.sum(a)

    @staticmethod
    def mean_red(a: Iterable[A]) -> A:
        return operators.sum(a) / float(len(a))

    @staticmethod
    def mean_full_red(a: Iterable[A]) -> A:
        return operators.sum(a) / float(len(a))

    @staticmethod
    def complex(a: A) -> A:
        return (
            operators.log(
                operators.sigmoid(
                    operators.relu(operators.relu(a * 10 + 7) * 6 + 5) * 10
                )
            )
            / 50
        )

    @classmethod
    def _tests(
        cls,
    ) -> Tuple[
        Tuple[str, Callable[[A], A]],
        Tuple[str, Callable[[A, A], A]],
        Tuple[str, Callable[[Iterable[A]], A]],
    ]:
        """
        Returns a list of all the math tests.
        """
        one_arg = []
        two_arg = []
        red_arg = []
        for k in dir(MathTest):
            if callable(getattr(MathTest, k)) and not k.startswith("_"):
                base_fn = getattr(cls, k)
                # scalar_fn = getattr(cls, k)
                tup = (k, base_fn)
                if k.endswith("2"):
                    two_arg.append(tup)
                elif k.endswith("red"):
                    red_arg.append(tup)
                else:
                    one_arg.append(tup)
        return one_arg, two_arg, red_arg

    @classmethod
    def _comp_testing(cls):
        one_arg, two_arg, red_arg = cls._tests()
        one_argv, two_argv, red_argv = MathTest._tests()
        one_arg = [(n1, f2, f1) for (n1, f1), (n2, f2) in zip(one_arg, one_argv)]
        two_arg = [(n1, f2, f1) for (n1, f1), (n2, f2) in zip(two_arg, two_argv)]
        red_arg = [(n1, f2, f1) for (n1, f1), (n2, f2) in zip(red_arg, red_argv)]
        return one_arg, two_arg, red_arg


class MathTestVariable(MathTest):
    @staticmethod
    def inv(a):
        return 1.0 / (a + 3.5)

    @staticmethod
    def sig(x):
        return x.sigmoid()

    @staticmethod
    def log(x):
        return (x + 100000).log()

    @staticmethod
    def relu(x):
        return (x + 5.5).relu()

    @staticmethod
    def exp(a):
        return (a - 200).exp()

    @staticmethod
    def explog(a):
        return (a + 100000).log() + (a - 200).exp()

    @staticmethod
    def sum_red(a):
        return a.sum(0)

    @staticmethod
    def mean_red(a):
        return a.mean(0)

    @staticmethod
    def mean_full_red(a):
        return a.mean()

    @staticmethod
    def eq2(a, b):
        return a == (b + 5.5)

    @staticmethod
    def gt2(a, b):
        return a + 1.2 > b

    @staticmethod
    def lt2(a, b):
        return a + 1.2 < b

    @staticmethod
    def complex(a):
        return (((a * 10 + 7).relu() * 6 + 5).relu() * 10).sigmoid().log() / 50

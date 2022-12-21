import pytest
from hypothesis import given

import minitorch

from .strategies import med_ints, small_floats

# # Tests for module.py


# ## Website example

# This example builds a module
# as shown at https://minitorch.github.io/modules.html
# and checks that its properties work.


class ModuleA1(minitorch.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p1 = minitorch.Parameter(5)
        self.non_param = 10
        self.a = ModuleA2()
        self.b = ModuleA3()


class ModuleA2(minitorch.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p2 = minitorch.Parameter(10)


class ModuleA3(minitorch.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c = ModuleA4()


class ModuleA4(minitorch.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p3 = minitorch.Parameter(15)


@pytest.mark.task0_4
def test_stacked_demo() -> None:
    "Check that each of the properties match"
    mod = ModuleA1()
    np = dict(mod.named_parameters())

    x = str(mod)
    print(x)
    assert mod.p1.value == 5
    assert mod.non_param == 10

    assert np["p1"].value == 5
    assert np["a.p2"].value == 10
    assert np["b.c.p3"].value == 15


# ## Advanced Tests

# These tests generate a stack of modules of varying sizes to check
# properties.

VAL_A = 50.0
VAL_B = 100.0


class Module1(minitorch.Module):
    def __init__(self, size_a: int, size_b: int, val: float) -> None:
        super().__init__()
        self.module_a = Module2(size_a)
        self.module_b = Module2(size_b)
        self.parameter_a = minitorch.Parameter(val)


class Module2(minitorch.Module):
    def __init__(self, extra: int = 0) -> None:
        super().__init__()
        self.parameter_a = minitorch.Parameter(VAL_A)
        self.parameter_b = minitorch.Parameter(VAL_B)
        self.non_parameter = 10
        self.module_c = Module3()
        for i in range(extra):
            self.add_parameter(f"extra_parameter_{i}", 0)


class Module3(minitorch.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter_a = minitorch.Parameter(VAL_A)


@pytest.mark.task0_4
@given(med_ints, med_ints)
def test_module(size_a: int, size_b: int) -> None:
    "Check the properties of a single module"
    module = Module2()
    module.eval()
    assert not module.training
    module.train()
    assert module.training
    assert len(module.parameters()) == 3

    module = Module2(size_b)
    assert len(module.parameters()) == size_b + 3

    module = Module2(size_a)
    named_parameters = dict(module.named_parameters())
    assert named_parameters["parameter_a"].value == VAL_A
    assert named_parameters["parameter_b"].value == VAL_B
    assert named_parameters["extra_parameter_0"].value == 0


@pytest.mark.task0_4
@given(med_ints, med_ints, small_floats)
def test_stacked_module(size_a: int, size_b: int, val: float) -> None:
    "Check the properties of a stacked module"
    module = Module1(size_a, size_b, val)
    module.eval()
    assert not module.training
    assert not module.module_a.training
    assert not module.module_b.training
    module.train()
    assert module.training
    assert module.module_a.training
    assert module.module_b.training

    assert len(module.parameters()) == 1 + (size_a + 3) + (size_b + 3)

    named_parameters = dict(module.named_parameters())
    assert named_parameters["parameter_a"].value == val
    assert named_parameters["module_a.parameter_a"].value == VAL_A
    assert named_parameters["module_a.parameter_b"].value == VAL_B
    assert named_parameters["module_b.parameter_a"].value == VAL_A
    assert named_parameters["module_b.parameter_b"].value == VAL_B


# ## Misc Tests

# Check that the module runs forward correctly.


class ModuleRun(minitorch.Module):
    def forward(self) -> int:
        return 10


@pytest.mark.task0_4
@pytest.mark.xfail
def test_module_fail_forward() -> None:
    mod = minitorch.Module()
    mod()


@pytest.mark.task0_4
def test_module_forward() -> None:
    mod = ModuleRun()
    assert mod.forward() == 10

    # Calling directly should call forward
    assert mod() == 10


# Internal check for the system.


class MockParam:
    def __init__(self) -> None:
        self.x = False

    def requires_grad_(self, x: bool) -> None:
        self.x = x


def test_parameter() -> None:
    t = MockParam()
    q = minitorch.Parameter(t)
    print(q)
    assert t.x
    t2 = MockParam()
    q.update(t2)
    assert t2.x

import streamlit as st
from interface.streamlit_utils import get_img_tag
from interface.train import render_train_interface
from argparse import ArgumentParser
from run_torch import TorchTrain
from math_interface import render_math_sandbox

parser = ArgumentParser()
parser.add_argument("module_num", type=int)
parser.add_argument(
    "--hide_function_defs", action="store_true", dest="hide_function_defs"
)
args = parser.parse_args()
module_num = args.module_num
hide_function_defs = args.hide_function_defs

st.set_page_config(page_title="interactive minitorch")
st.sidebar.markdown(
    """
<h1 style="font-size:30pt; float: left; margin-right: 20px; margin-top: 1px;">MiniTorch</h1>{}
""".format(
        get_img_tag("https://minitorch.github.io/_images/match.png", width="40")
    ),
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    [Documentation](https://minitorch.github.io/)
"""
)

module_selection = st.sidebar.radio(
    "Module",
    ["Module 0", "Module 1", "Module 2", "Module 3", "Module 4"][: module_num + 1],
    index=module_num,
)


PAGES = {}

if module_selection == "Module 0":
    from run_manual import ManualTrain
    from module_interface import render_module_sandbox

    def render_run_manual_interface():
        st.header("Module 0 - Manual")
        render_train_interface(ManualTrain, False, False, True)

    def render_m0_sandbox():
        return render_math_sandbox(False)

    PAGES["Math Sandbox"] = render_m0_sandbox
    PAGES["Module Sandbox"] = render_module_sandbox

    def render_run_torch_interface():
        st.header("Demo - Torch")
        render_train_interface(TorchTrain, False)

    PAGES["Torch Example"] = render_run_torch_interface
    PAGES["Module 0: Manual"] = render_run_manual_interface

if module_selection == "Module 1":
    from run_scalar import ScalarTrain
    from show_expression_interface import render_show_expression

    def render_m1_sandbox():
        return render_math_sandbox(True)

    def render_run_scalar_interface():
        st.header("Module 1 - Scalars")
        render_train_interface(ScalarTrain)

    PAGES["Scalar Sandbox"] = render_m1_sandbox
    PAGES["Autodiff Sandbox"] = render_show_expression
    PAGES["Module 1: Scalar"] = render_run_scalar_interface

if module_selection == "Module 2":
    from run_tensor import TensorTrain
    from tensor_interface import render_tensor_sandbox
    from show_expression_interface import render_show_expression

    def render_run_tensor_interface():
        st.header("Module 2 - Tensors")
        render_train_interface(TensorTrain)

    def render_m2_sandbox():
        return render_math_sandbox(True, True)

    PAGES["Tensor Sandbox"] = lambda: render_tensor_sandbox(hide_function_defs)
    PAGES["Tensor Math Sandbox"] = render_m2_sandbox
    PAGES["Autograd Sandbox"] = lambda: render_show_expression(True)
    PAGES["Module 2: Tensor"] = render_run_tensor_interface


if module_selection == "Module 3":
    from run_fast_tensor import FastTrain

    def render_run_fast_interface():
        st.header("Module 3 - Efficient")
        render_train_interface(FastTrain, False)

    PAGES["Module 3: Efficient"] = render_run_fast_interface

if module_selection == "Module 4":
    from run_mnist_interface import render_run_image_interface
    from sentiment_interface import render_run_sentiment_interface

    PAGES["Module 4: Images"] = render_run_image_interface
    PAGES["Module 4: Sentiment"] = render_run_sentiment_interface


PAGE_OPTIONS = list(PAGES.keys())

page_selection = st.sidebar.radio("Pages", PAGE_OPTIONS)
page = PAGES[page_selection]
page()

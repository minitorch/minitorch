from minitorch.tensor_data import TensorData
from project.interface.streamlit_utils import render_function
import streamlit as st
from minitorch import index_to_position, operators, to_index, Tensor, TensorFunctions
from show_tensor import tensor_figure
import numpy as np
import plotly.graph_objects as go


def st_select_index(tensor_shape, n_cols=3):
    out_index = [0] * len(tensor_shape)
    cols = st.beta_columns(n_cols)
    for idx, dim in enumerate(tensor_shape):
        out_index[idx] = cols[idx % n_cols].number_input(
            f"Dimension {idx} index:", value=0, min_value=0, max_value=dim - 1
        )
    return out_index


def st_visualize_storage(tensor: Tensor, selected_position: int, max_size=10):
    tensor_size = len(tensor._tensor._storage)
    if tensor_size > max_size:
        st.warning(f"Showing first {max_size} elements from the tensor storage.")
    x = list(range(min(tensor_size, max_size)))
    y = [0] * len(x)
    data = [
        go.Scatter(
            hoverinfo="skip",
            mode="markers+text",
            x=x,
            y=y,
            marker=dict(
                size=50,
                symbol="square",
                color=[
                    "#69BAC9" if x_ == selected_position else "lightgray" for x_ in x
                ],
            ),
            text=tensor._tensor._storage[:max_size],
            textposition="middle center",
            textfont_size=20,
        )
    ]

    lr_margin = 25 if len(x) >= 9 else 75 if len(x) >= 6 else 175

    layout = go.Layout(
        title={"text": "Tensor Storage", "x": 0.5, "y": 1.0, "xanchor": "center"},
        font={"family": "Raleway", "size": 20, "color": "black"},
        xaxis={"showgrid": False, "showticklabels": False},
        yaxis={"showgrid": False, "showticklabels": False},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        autosize=True,
        width=650,
        height=125,
        showlegend=False,
        margin=dict(l=lr_margin, r=lr_margin, t=0, b=0),
    )

    fig = go.Figure(data=data, layout=layout)
    st.write(fig)


def st_visualize_tensor(
    tensor: Tensor, highlighted_index, strides=None, show_value=True
):
    depth = tensor.shape[0]
    rows = tensor.shape[1] if len(tensor.shape) > 1 else 1
    columns = tensor.shape[2] if len(tensor.shape) > 2 else 1

    if strides is None:
        strides = tensor._tensor.strides

    if len(tensor.shape) != 3:
        # TODO: Fix visualization instead of showing warning
        st.error("Can only visualize a tensor which has 3 dimensions")
        return

    position_in_storage = index_to_position(highlighted_index, strides)

    if position_in_storage >= 0 and show_value:
        st.write(
            f"**Value at position {position_in_storage}:** {tensor._tensor._storage[position_in_storage]}"
        )

    # Map index to highlight since tensor_figure doesn't know about strides
    st.write("highlighted", highlighted_index)
    if len(highlighted_index) > 2:
        highlighted_position = highlighted_index[0]
        highlighted_position += highlighted_index[1] * depth * columns
        highlighted_position += highlighted_index[2] * depth
    elif len(highlighted_index) > 1:
        highlighted_position = highlighted_index[0]
        highlighted_position += highlighted_index[1] * depth * columns
    else:
        highlighted_position = highlighted_index[0]

    fig = tensor_figure(
        depth,
        columns,
        rows,
        highlighted_position,
        f"Storage position: {position_in_storage}, Index: {highlighted_index}",
        # Fix for weirndess in tensor_figure axis
        axisTitles=["depth (i)", "columns (k)", "rows (j)"],
        show_fig=False,
        slider=False,
    )
    fig.update_layout(
        width=650,
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=50, t=0, b=0),
    )
    st.write(fig)


def interface_visualize_tensor(tensor: Tensor, hide_function_defs: bool):
    st.write(f"**Tensor strides:** {tensor._tensor.strides}")
    selected_position = st.slider(
        "Selected position in storage", 0, len(tensor._tensor._storage) - 1, value=0
    )
    out_index = [0] * len(tensor.shape)
    to_index(selected_position, tensor.shape, out_index)
    st.write(f"**Corresponding index:** {out_index}")
    st_visualize_tensor(tensor, out_index, show_value=False)
    st_visualize_storage(tensor, selected_position)


def interface_index_to_position(tensor: Tensor, hide_function_defs: bool):
    if not hide_function_defs:
        with st.beta_expander("Show function definition"):
            render_function(index_to_position)
    col1, col2 = st.beta_columns(2)
    idx = eval(
        col1.text_input(
            "Multi-dimensional index", value=str([0] * len(tensor._tensor.strides))
        )
    )
    tensor_strides = eval(
        col2.text_input("Tensor strides", value=str(tensor._tensor.strides))
    )
    st_visualize_tensor(tensor, idx, tensor_strides)


def interface_to_index(tensor: Tensor, hide_function_defs: bool):
    if not hide_function_defs:
        with st.beta_expander("Show function definition"):
            render_function(to_index)
    tensor_shape = tensor.shape
    st.write(f"**Tensor strides:** {tensor._tensor.strides}")
    selected_position = st.number_input(
        "Position in storage",
        value=0,
        min_value=0,
        max_value=len(tensor._tensor._storage) - 1,
    )
    out_index = [0] * len(tensor_shape)
    to_index(selected_position, tensor_shape, out_index)
    st.write(
        f"**Value at position {selected_position}:** {tensor._tensor._storage[selected_position]}"
    )
    st.write("**Out index:**", out_index)

    st_visualize_tensor(tensor, out_index, show_value=False)


def interface_strides(tensor: Tensor, hide_function_defs: bool):
    strides = eval(st.text_input("Tensor strides", value=str(tensor._tensor.strides)))

    st.write("**Try it out:**")
    out_index = st_select_index(tensor.shape)
    cols = st.beta_columns(len(strides))
    for dim, stride in enumerate(strides):
        cols[dim].write(f"*moves {stride} positions in storage*")
    st_visualize_tensor(tensor, out_index, strides, show_value=True)


def interface_permute(tensor: Tensor, hide_function_defs: bool):
    if not hide_function_defs:
        with st.beta_expander("Show function definition"):
            render_function(TensorData.permute)

    st.write(f"**Tensor strides:** {tensor._tensor.strides}")
    default_permutation = list(range(len(tensor.shape)))
    default_permutation.reverse()
    permutation = eval(st.text_input("Tensor permutation", value=default_permutation))
    p_tensor = tensor.permute(*permutation)
    p_tensor_strides = p_tensor._tensor.strides
    st.write(f"**Permuted tensor strides:** {p_tensor_strides}")

    st.write("**Try selecting a tensor value by index:**")
    out_index = st_select_index(tensor.shape)
    viz_type = st.selectbox(
        "Choose tensor visualization", options=["Original tensor", "Permuted tensor"]
    )
    if viz_type == "Original tensor":
        viz_tensor = tensor
    else:
        viz_tensor = p_tensor
    st_visualize_tensor(viz_tensor, out_index, show_value=False)
    st_visualize_storage(
        tensor, index_to_position(out_index, viz_tensor._tensor.strides)
    )


def st_eval_error_message(expression: str, error_msg: str):
    try:
        return eval(expression)
    except Exception as e:
        st.error(error_msg)
        raise e


def render_tensor_sandbox(hide_function_defs: bool):
    st.write("## Sandbox for Tensors")
    st.write("**Define your tensor**")
    # Consistent random number generator
    rng = np.random.RandomState(42)
    # col1, col2 = st.beta_columns(2)
    tensor_shape = st_eval_error_message(
        st.text_input("Tensor shape", value="(2, 2, 2)"),
        "Tensor shape must be defined as an in-line tuple, i.e. (2, 2, 2)",
    )
    tensor_size = int(operators.prod(tensor_shape))
    random_tensor = st.checkbox("Fill tensor with random numbers", value=True)
    if random_tensor:
        tensor_data = np.round(rng.rand(tensor_size), 2)
        st.write("**Tensor data storage:**")
        # Visualize horizontally
        st.write(tensor_data.reshape(1, -1))
    else:
        tensor_data = st_eval_error_message(
            st.text_input("Tensor data storage", value=str(list(range(tensor_size)))),
            "Tensor data storage must be defined as an in-line list, i.e. [1, 2, 3, 4]",
        )

    try:
        test_tensor = Tensor.make(tensor_data, tensor_shape, backend=TensorFunctions)
    except AssertionError as e:
        storage_size = len(tensor_data)
        if tensor_size != storage_size:
            st.error(
                f"Tensor data storage must define all values in shape ({tensor_size} != {storage_size    })"
            )
        else:
            st.error(e)
        return

    select_fn = {
        "Visualize Tensor Definition": interface_visualize_tensor,
        "Visualize Tensor Strides": interface_strides,
        "function: index_to_position": interface_index_to_position,
        "function: to_index": interface_to_index,
        "function: TensorData.permute": interface_permute,
    }

    selected_fn = st.selectbox("Select an interface", options=list(select_fn.keys()))

    select_fn[selected_fn](test_tensor, hide_function_defs)

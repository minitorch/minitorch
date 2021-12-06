import streamlit as st
import graph_builder
import networkx as nx


def render_show_expression(tensor=False):

    if tensor:
        st.text("Build an expression of tensors x, y, and z. (All the same shape)")
        code = st.text_area(
            label="Expression of x,y,z", value="(x * x) * y + 10.0 * x.sum()"
        )
        out = graph_builder.build_tensor_expression(code)
    else:
        code = st.text_area(label="Expression of x,y,z", value="(x * x) * y + 10.0 * x")
        out = graph_builder.build_expression(code)

    G = graph_builder.GraphBuilder().run(out)
    G.graph["graph"] = {"rankdir": "LR"}
    st.graphviz_chart(nx.nx_pydot.to_pydot(G).to_string())

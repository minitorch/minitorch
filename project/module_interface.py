import streamlit as st
import networkx as nx
import minitorch

MyModule = None
minitorch


def render_module_sandbox():
    st.write("## Sandbox for Module Trees")

    st.write(
        "Visual debugging checks showing the module tree that your code constructs."
    )

    code = st.text_area(
        label="Module code",
        height=600,
        value="""
class MyModule(minitorch.Module):
    def __init__(self):
        super().__init__()
        self.parameter1 = minitorch.Parameter(15)
""",
    )
    out = exec(code, globals())
    out = MyModule()
    st.write(dict(out.named_parameters()))
    G = nx.MultiDiGraph()
    G.add_node("base")
    stack = [(out, "base")]

    while stack:
        n, name = stack[0]
        stack = stack[1:]
        for pname, p in n.__dict__["_parameters"].items():
            G.add_node(name + "." + pname, shape="rect", penwidth=0.5)
            G.add_edge(name, name + "." + pname)

        for cname, m in n.__dict__["_modules"].items():
            G.add_edge(name, name + "." + cname)
            stack.append((m, name + "." + cname))

    G.graph["graph"] = {"rankdir": "TB"}
    st.graphviz_chart(nx.nx_pydot.to_pydot(G).to_string())

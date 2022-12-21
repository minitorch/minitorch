"""
Be sure you have the extra requirements installed.

>>> pip install -r requirements.extra.txt
"""

import networkx as nx

import minitorch


## Create an autodiff expression here.
def expression():
    x = minitorch.Scalar(1.0, name="x")
    y = minitorch.Scalar(1.0, name="y")
    z = (x * x) * y + 10.0 * x
    z.name = "z"
    return z


class GraphBuilder:
    def __init__(self):
        self.op_id = 0
        self.hid = 0
        self.intermediates = {}

    def get_name(self, x):
        if not isinstance(x, minitorch.Scalar):
            return "constant %s" % (x,)
        elif len(x.name) > 15:
            if x.name in self.intermediates:
                return "v%d" % (self.intermediates[x.name],)
            else:
                self.hid = self.hid + 1
                self.intermediates[x.name] = self.hid
                return "v%d" % (self.hid,)
        else:
            return x.name

    def run(self, final):
        queue = [[final]]

        G = nx.MultiDiGraph()
        G.add_node(self.get_name(final))

        while queue:
            (cur,) = queue[0]
            queue = queue[1:]

            if cur.history is None:
                continue
            elif cur.is_leaf():
                continue
            else:
                op = "%s (Op %d)" % (cur.history.last_fn.__name__, self.op_id)
                G.add_node(op, shape="square", penwidth=3)
                G.add_edge(op, self.get_name(cur))
                self.op_id += 1
                for i, input in enumerate(cur.history.inputs):
                    G.add_edge(self.get_name(input), op, f"{i}")

                for input in cur.history.inputs:
                    if not isinstance(input, minitorch.Scalar):
                        continue

                    seen = False
                    for s in queue:
                        if s[0] == input:
                            seen = True
                    if not seen:
                        queue.append([input])
        return G


def make_graph(y, lr=False):
    G = GraphBuilder().run(y)
    if lr:
        G.graph["graph"] = {"rankdir": "LR"}
    output_graphviz_svg = nx.nx_pydot.to_pydot(G).create_svg()
    return output_graphviz_svg

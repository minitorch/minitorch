import minitorch
import networkx as nx


def build_expression(code):
    out = eval(
        code,
        {
            "x": minitorch.Scalar(1.0, name="x"),
            "y": minitorch.Scalar(1.0, name="y"),
            "z": minitorch.Scalar(1.0, name="z"),
        },
    )
    out.name = "out"
    return out


def build_tensor_expression(code):
    out = eval(
        code,
        {
            "x": minitorch.tensor([[1.0, 2.0, 3.0]], requires_grad=True),
            "y": minitorch.tensor([[1.0, 2.0, 3.0]], requires_grad=True),
            "z": minitorch.tensor([[1.0, 2.0, 3.0]], requires_grad=True),
        },
    )
    out.name = "out"
    return out


class GraphBuilder:
    def __init__(self):
        self.op_id = 0
        self.hid = 0
        self.intermediates = {}

    def get_name(self, x):
        if not isinstance(x, minitorch.Variable):
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

            if minitorch.is_constant(cur) or cur.is_leaf():
                continue
            else:
                op = "%s (Op %d)" % (cur.history.last_fn.__name__, self.op_id)
                G.add_node(op, shape="square", penwidth=3)
                G.add_edge(op, self.get_name(cur))
                self.op_id += 1
                for i, input in enumerate(cur.history.inputs):
                    G.add_edge(self.get_name(input), op, f"{i}")

                for input in cur.history.inputs:
                    if not isinstance(input, minitorch.Variable):
                        continue
                    queue.append([input])
        return G

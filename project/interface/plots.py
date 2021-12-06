import plotly.graph_objects as go


def make_scatters(graph, model=None, size=50):
    color_map = ["#69bac9", "#ea8484"]
    symbol_map = ["circle-dot", "x"]
    colors = [color_map[y] for y in graph.y]
    symbols = [symbol_map[y] for y in graph.y]
    scatters = []

    if model is not None:
        colorscale = [[0, "#69bac9"], [1.0, "#ea8484"]]
        z = [
            model([[j / (size + 1.0), k / (size + 1.0)] for j in range(size + 1)])
            for k in range(size + 1)
        ]
        scatters.append(
            go.Contour(
                z=z,
                dx=1 / size,
                x0=0,
                dy=1 / size,
                y0=0,
                zmin=0.2,
                zmax=0.8,
                line_smoothing=0.5,
                colorscale=colorscale,
                opacity=0.6,
                showscale=False,
            )
        )
    scatters.append(
        go.Scatter(
            mode="markers",
            x=[p[0] for p in graph.X],
            y=[p[1] for p in graph.X],
            marker_symbol=symbols,
            marker_color=colors,
            marker=dict(size=15, line=dict(width=3, color="Black")),
        )
    )
    return scatters


def animate(self, models, names):
    import plotly.graph_objects as go

    scatters = [make_scatters(self, m) for m in models]
    background = [s[0] for s in scatters]
    for i, b in enumerate(background):
        b["visible"] = i == 0
    points = scatters[0][1]
    steps = []
    for i in range(len(background)):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(background) + [True]},
                {},
            ],  # layout attribute
            label="%1.3f" % names[i],
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [
        dict(active=0, currentvalue={"prefix": "b="}, pad={"t": 50}, steps=steps)
    ]

    fig = go.Figure(data=background + [points],)
    fig.update_layout(sliders=sliders)

    fig.update_layout(
        template="simple_white",
        xaxis={
            "showgrid": False,  # thin lines in the background
            "zeroline": False,  # thick line at x=0
            "visible": False,  # numbers below
        },
        yaxis={
            "showgrid": False,  # thin lines in the background
            "zeroline": False,  # thick line at x=0
            "visible": False,  # numbers below
        },
    )
    fig.show()


def make_oned(graph, model=None, size=50):
    scatters = []
    color_map = ["#69bac9", "#ea8484"]
    symbol_map = ["circle-dot", "x"]
    colors = [color_map[y] for y in graph.y]
    symbols = [symbol_map[y] for y in graph.y]

    if model is not None:
        # colorscale = [[0, "#69bac9"], [1.0, "#ea8484"]]
        y = model([[j / (size + 1.0), 0.0] for j in range(size + 1)])

        x = [j / (size + 1.0) for j in range(size + 1)]
        scatters.append(
            go.Scatter(
                mode="lines",
                x=[j / (size + 1.0) for j in range(size + 1)],
                y=y,
                marker=dict(size=15, line=dict(width=3, color="Black")),
            )
        )
        print(x, y)
    scatters.append(
        go.Scatter(
            mode="markers",
            x=[p[0] for p in graph.X],
            y=graph.y,
            marker_symbol=symbols,
            marker_color=colors,
            marker=dict(size=15, line=dict(width=3, color="Black")),
        )
    )
    return scatters


def plot_out(graph, model=None, name="", size=50, oned=False):
    if oned:
        scatters = make_oned(graph, model, size=size)
    else:
        scatters = make_scatters(graph, model, size=size)

    fig = go.Figure(scatters)
    fig.update_layout(
        xaxis={
            "showgrid": False,  # thin lines in the background
            "visible": False,  # numbers below
            "range": [0, 1],
        },
        yaxis={
            "showgrid": False,  # thin lines in the background
            "visible": False,  # numbers below
            "range": [0, 1],
        },
    )
    return fig


def plot(graph, model=None, name=""):
    plot_out(graph, model, name).show()


def plot_function(title, fn, arange=[(i / 5.0) - 4.0 for i in range(0, 40)]):
    ys = [fn(x) for x in arange]
    scatter = go.Scatter(x=arange, y=ys)

    fig = go.Figure(scatter)
    fig.update_layout(template="simple_white", title=title)

    fig.show()

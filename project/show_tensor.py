import numpy as np
import plotly.graph_objects as go

x = [1, 2, 1, 2, 1, 2]
y = [1, 1, 2, 2, 3, 3]
x1 = np.array(list(map(lambda x: x + 3, x))).ravel()
y1 = y.copy()
x2 = np.array(list(map(lambda x: x1 + 3, x))).ravel()
y2 = y.copy()
initial_matrix = np.vstack(
    [
        np.hstack([[1, 1, 1, 1, 1], np.ones(5) * 2]),
        np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]),
    ]
)
axis_default = ["i", "k", "j"]


def permute(mat, x, y):
    return mat.transpose(x, y)


def plot_matrix(x, y, title, w=300, h=500, bg="white"):
    data = [
        go.Scatter(
            hoverinfo="skip",
            mode="markers",
            x=x,
            y=y,
            marker=dict(color="black", size=50, symbol="square-open"),
        ),
    ]

    layout = go.Layout(
        title={"text": title, "x": 0.5, "y": 0.9, "xanchor": "center"},
        font={"family": "Raleway", "size": 40, "color": "black"},
        xaxis={"showgrid": False, "showticklabels": False},
        yaxis={"showgrid": False, "showticklabels": False},
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        autosize=False,
        width=w,
        height=h,
        showlegend=False,
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()


def plot_map():

    data = [
        go.Scatter(
            hoverinfo="skip",
            mode="markers",
            x=x,
            y=y,
            marker=dict(color="black", size=50, symbol="square-open"),
        ),
        go.Scatter(
            hoverinfo="skip",
            mode="markers",
            x=x1,
            y=y1,
            marker=dict(color="#69BAC9", size=50, symbol="square"),
        ),
    ]

    layout = go.Layout(
        title={"text": "map", "x": 0.5, "y": 0.9, "xanchor": "center"},
        font={"family": "Raleway", "size": 40, "color": "black"},
        xaxis={"showgrid": False, "showticklabels": False},
        yaxis={"showgrid": False, "showticklabels": False},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        autosize=False,
        width=500,
        height=400,
        showlegend=False,
    )

    fig = go.Figure(data=data, layout=layout)

    fig.add_annotation(
        x=5,
        y=2,
        ax=2,
        ay=2,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        font=dict(
            size=15,
            color="black",
        ),
    )
    fig.add_annotation(
        x=4,
        y=3,
        ax=1,
        ay=3,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        font=dict(
            size=15,
            color="black",
        ),
    )
    fig.show()


def plot_zip():

    data = [
        go.Scatter(
            hoverinfo="skip",
            mode="markers",
            x=x,
            y=y,
            marker=dict(color="black", size=50, symbol="square-open"),
        ),
        go.Scatter(
            hoverinfo="skip",
            mode="markers",
            x=x1,
            y=y1,
            marker=dict(color="black", size=50, symbol="square-open"),
        ),
        go.Scatter(
            hoverinfo="skip",
            mode="markers",
            x=x2,
            y=y2,
            marker=dict(color="#69BAC9", size=50, symbol="square"),
        ),
    ]

    layout = go.Layout(
        title={"text": "zip", "x": 0.5, "y": 0.9, "xanchor": "center"},
        font={"family": "Raleway", "size": 40, "color": "black"},
        xaxis={"showgrid": False, "showticklabels": False},
        yaxis={"showgrid": False, "showticklabels": False},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        autosize=False,
        width=700,
        height=400,
        showlegend=False,
    )

    fig = go.Figure(data=data, layout=layout)
    fig.add_annotation(
        x=8,
        y=3,
        ax=2,
        ay=3,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        font=dict(
            size=15,
            color="black",
        ),
    )
    fig.add_annotation(
        x=8,
        y=3,
        ax=5,
        ay=2.8,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        font=dict(
            size=15,
            color="black",
        ),
    )
    fig.show()


def plot_reduce():

    data = [
        go.Scatter(
            hoverinfo="skip",
            mode="markers",
            x=x,
            y=y,
            marker=dict(color="black", size=50, symbol="square-open"),
        ),
        go.Scatter(
            hoverinfo="skip",
            mode="markers",
            x=x1[:2],
            y=y1[:2],
            marker=dict(color="#69BAC9", size=50, symbol="square"),
        ),
    ]

    layout = go.Layout(
        title={"text": "reduce", "x": 0.5, "y": 0.9, "xanchor": "center"},
        font={"family": "Raleway", "size": 40, "color": "black"},
        xaxis={"showgrid": False, "showticklabels": False},
        yaxis={"showgrid": False, "showticklabels": False},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        autosize=False,
        width=700,
        height=400,
        showlegend=False,
    )

    fig = go.Figure(data=data, layout=layout)

    fig.add_annotation(
        x=2,
        y=1,
        ax=2,
        ay=3,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        font=dict(
            size=15,
            color="black",
        ),
    )
    fig.add_annotation(
        x=1,
        y=1,
        ax=1,
        ay=3,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        font=dict(
            size=15,
            color="black",
        ),
    )
    fig.add_annotation(
        x=3.5,
        y=1,
        ax=2.3,
        ay=1,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        font=dict(
            size=15,
            color="black",
        ),
    )
    fig.show()


def plot_tensor(x, y, z, active=5):
    fig = go.Figure()

    # Construct tensor coordinates
    def construct_tensor(shape=[x, y, z]):
        coords = []
        for z in list(range(shape[2])):
            for y in list(range(shape[1])):
                for x in list(range(shape[0])):

                    coords.append([x, y, z])
        return np.array(coords) * 1.1

    tensor_coords = construct_tensor(shape=[x, y, z])

    # Construct one 3d mesh box
    def add_one_box(ind, xs, ys, zs, name, alpha=1.0):

        # Build triangles from tensor coordinates
        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
        triangles = np.vstack((i, j, k)).T

        vertices = np.vstack([xs, ys, zs]).T
        tri_points = vertices[triangles]
        Xe = []
        Ye = []
        Ze = []

        for T in tri_points:
            Xe.extend([T[k % 3][0] for k in range(4)] + [None])
            Ye.extend([T[k % 3][1] for k in range(4)] + [None])
            Ze.extend([T[k % 3][2] for k in range(4)] + [None])

        # Build mesh box from triangles
        if ind == active:
            c = "#69bac9"
        else:
            c = "white"
        fig.add_trace(
            go.Mesh3d(
                hoverinfo="skip",
                opacity=alpha,
                x=xs,
                y=ys,
                z=zs,
                color=c,
                flatshading=True,
                alphahull=0,
                name=name,
                showscale=False,
                visible=True,
                lighting=dict(ambient=0.5, diffuse=0.6),
                lightposition=dict(x=0, y=0, z=0),
            )
        )

    # Add boxes to fig
    def box_adder(boxes):

        # Construct boxes from tensor coordinates
        def construct_whole_box(initXYZ):
            for ind, i in enumerate(initXYZ):
                if ind == 0:
                    wholeBoxXs = [
                        i + 0,
                        i + 1,
                        i + 0,
                        i + 1,
                        i + 0,
                        i + 1,
                        i + 0,
                        i + 1,
                    ]
                if ind == 1:
                    wholeBoxYs = [
                        i + 0,
                        i + 0,
                        i + 1,
                        i + 1,
                        i + 0,
                        i + 0,
                        i + 1,
                        i + 1,
                    ]
                if ind == 2:
                    wholeBoxZs = [
                        i + 0,
                        i + 0,
                        i + 0,
                        i + 0,
                        i + 1,
                        i + 1,
                        i + 1,
                        i + 1,
                    ]
            return wholeBoxXs, wholeBoxYs, wholeBoxZs

        for ind, i in enumerate(boxes):
            add_one_box(
                ind,
                *construct_whole_box(i),
                str((np.array([i[0], i[2], i[1]]) / (1.1)).astype(int))
                .replace(" ", ",")
                .replace("[", "(")
                .replace("]", ")")
            )

    box_adder(tensor_coords)
    return fig


def tensor_figure(
    x,
    y,
    z,
    active,
    title,
    xr=None,
    yr=None,
    zr=None,
    axisTitles=axis_default,
    slider=True,
    eye=dict(x=2.8, y=1.6, z=1.6),
    show_fig=True,
):
    fig = plot_tensor(x, y, z, active=active)
    if xr is None:
        xr = [x + 0.2, 0]
        yr = [0, y + 0.2]
        zr = [z + 0.2, 0]
    # Create and add slider
    if slider:
        steps = []
        for i, val in enumerate(fig.data):
            step = dict(
                method="update",
                args=[
                    {
                        "opacity": [1.0] * len(fig.data),
                        "color": ["white"] * len(fig.data),
                    },
                    {"title": "Tensor Index: " + val["name"]},
                ],
            )
            # Toggle i'th trace
            step["args"][0]["opacity"][i] = 1.0
            step["args"][0]["color"][i] = "#69bac9"
            steps.append(step)

        fig.update_layout(
            sliders=[
                dict(
                    active=active,
                    steps=steps,
                    currentvalue=dict(visible=False),
                    tickcolor="#fcfcfc",
                    font=dict(color="#fcfcfc"),
                ),
            ],
        )

    camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=eye)

    fig.update_layout(
        title={"text": title, "x": 0.5, "y": 0.9, "xanchor": "center"},
        font={"family": "Raleway", "size": 40, "color": "black"},
        scene_camera=camera,
        paper_bgcolor="#fcfcfc",
        font_size=20,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#fcfcfc", font_size=30, font_family="Times New Roman"),
        scene=dict(
            xaxis=dict(
                showbackground=False,
                zerolinecolor="#fcfcfc",
                showticklabels=False,
                range=xr,
                title=axisTitles[0],
            ),
            yaxis=dict(
                showbackground=False,
                zerolinecolor="#fcfcfc",
                showticklabels=False,
                range=yr,
                title=axisTitles[1],
            ),
            zaxis=dict(
                showbackground=False,
                zerolinecolor="#fcfcfc",
                showticklabels=False,
                range=zr,
                title=axisTitles[2],
            ),
        ),
        width=500,
        height=500,
        margin=dict(r=10, l=10, b=10, t=50),
    )
    if show_fig:
        fig.show()
    else:
        return fig

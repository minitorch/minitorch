import random

import chalk as ch
from chalk import (
    Trail,
    empty,
    make_path,
    path,
    place_on_path,
    rectangle,
    unit_x,
    unit_y,
)
from colour import Color
from drawing import aqua, black, lightblue, lightred

import minitorch

random.seed(10)

s = minitorch.datasets["Simple"](10)
spl = minitorch.datasets["Split"](10)


def base_model(x1, x2):
    return 1 if x1 + x2 > 1.0 else -1


s1 = [s.X[i] for i in range(len(s.y)) if base_model(*s.X[i]) < 0]
s2 = [s.X[i] for i in range(len(s.y)) if base_model(*s.X[i]) > 0]
s.y = [1 if base_model(*s.X[i]) > 0 else -1 for i in range(len(s.y))]
d = spl
s1_hard = [d.X[i] for i in range(len(d.y)) if d.y[i] == 0]
s2_hard = [d.X[i] for i in range(len(d.y)) if d.y[i] == 1]


def show(model):
    "Plot over model"
    return draw_graph(model) + split_graph(s1, s2, show_origin=False)


def circle_mark():
    d = (
        ch.Trail.circle(2).centered().to_path()
        + ch.Trail.circle(1.0, False).centered().to_path()
    )
    return d.stroke().fill_color(aqua).line_width(0.2).scale_uniform_to_x(0.1)


def origin():
    return ch.rectangle(1, 1).translate(0.5, -0.5).fill_color(white).line_color(white)


def axes():
    return (ch.vrule(1).translate(0, -0.5) + ch.hrule(1).translate(0.5, 0)).line_width(
        0.2
    )


def d_mark():
    # big = Primitive.from_shape(Path.polygon(4, 2)).fill_color(papaya).line_width(0.1)
    # small = Primitive.from_shape(Path.polygon(4, 1).reverse()).line_width(0.1)
    t = Trail.from_offsets(
        [
            unit_x,
            unit_y,
            unit_x,
            unit_y,
            -unit_x,
            unit_y,
            -unit_x,
            -unit_y,
            -unit_x,
            -unit_y,
            unit_x,
            -unit_y,
        ],
        True,
    )
    return (
        t.rotate_by(0.25 / 2).line_width(0.2).scale_uniform_to_x(0.1).fill_color(blue)
    )


def x_mark():
    t = Trail.from_offsets(
        [
            unit_x,
            unit_y,
            unit_x,
            unit_y,
            -unit_x,
            unit_y,
            -unit_x,
            -unit_y,
            -unit_x,
            -unit_y,
            unit_x,
            -unit_y,
        ],
        True,
    )
    return (
        t.stroke()
        .rotate_by(0.25 / 2)
        .line_width(0.2)
        .center_xy()
        .scale_uniform_to_x(0.1)
        .fill_color(Color("red"))
    )

    # big = rectangle(2, 1).fill_color(black).line_width(0)
    # small = rectangle(1.5, 0.5).fill_color(Color("red")).line_width(0)
    # def make_x(d):
    #     return d.rotate_by(0.25 / 2)  + d.rotate_by(-0.25 / 2)
    # return (make_x(big) + make_x(small)).scale_uniform_to_x(0.1)


def points(m, pts):
    return place_on_path([m] * len(pts), Path.from_list_of_tuples(pts).reflect_y())


def draw_below(fn):
    return make_path([(0, 0), (0, fn(0)), (1, fn(1)), (1, 0), (0, 0)]).reflect_y()


def split_graph(circles, crosses, show_origin=True):
    dia = empty() if not show_origin else origin() + axes()
    return dia + points(circle_mark(), circles) + points(x_mark(), crosses)


def quad(fn, c1, c2):
    """
    Quad draws in an arbitrary contour surface.
    """

    def q(tl, s):
        # Evaluate at 4 corners.
        v = [fn(tl.x + d1 * s, tl.y + d2 * s) for d1 in range(2) for d2 in range(2)]

        if s < 1 and ((v[0] == v[1] and v[1] == v[2] and v[2] == v[3]) or s < 0.05):
            # If all the same draw a box.
            c = c1 if v[0] == 1 else c2
            r = rectangle(s, s).translate(s / 2, s / 2).line_color(c).fill_color(c)
            return r
        else:
            # Otherwise draw each separately.
            s /= 2
            return (q(tl, s) | q(tl + s * unit_x, s)) / (
                q(tl + s * unit_y, s) | q(tl + s * (unit_y + unit_x), s)
            )

    return q(P2(0, 0), 1)


def draw_graph(f, c1=lightred, c2=lightblue):
    return quad(lambda x1, x2: f.forward(x1, x2) > 0, c1, c2).reflect_y() + axes()


def compare(m1, m2):
    return (
        draw_graph(m1).center_xy()
        | hstrut(0.5)
        | text("→", 0.5).fill_color(black)
        | hstrut(0.5)
        | draw_graph(m2).center_xy()
    )


def with_points(pts1, pts2, b):
    "Draw a picture showing line to boundary"
    w1, w2 = 1, 1
    model = Linear(w1, w2, b)
    line = make_path([(0, b), (1, b + 1)])
    dia = draw_graph(model) + split_graph(pts1, pts2, False)

    for pt in pts1:
        pt2 = line.get_trace().trace_p(P2(pt[0], -pt[1]), V2(-1, 1))
        if pt2:
            dia += make_path([(pt[0], -pt[1]), pt2]).dashing([5, 5], 0)
    return dia


def graph(fn, xs=[], os=[], width=4, offset=0, c=Color("red")):
    "Draw a graph with points on it"
    path = []
    m = 0
    for a in range(100):
        a = width * ((a / 100) - 0.5) - offset
        path.append((a, fn(a)))
        m = max(m, fn(a))
    dia = (
        make_path([(0, 0), (0, m)])
        + make_path([(-width / 2, 0), (width / 2, 0)])
        + make_path(path).line_color(c).line_width(0.2)
    )

    for pt in xs:
        dia += x_mark().scale(width / 2).translate(pt, fn(pt))
    for pt in os:
        dia += circle_mark().scale(width / 2).translate(pt, fn(pt))
    return dia.reflect_y()


def show_loss(full_loss):
    d = empty()
    scores = []
    path = []
    i = 0
    for j, b in enumerate(range(20)):
        b = -1.7 + b / 20
        m = Linear(1, 1, b)
        pt = (b, full_loss(m))
        path.append(pt)
        if j % 5 == 0:
            d = d | hstrut(0.5) | show(m).named(("graph", i))
            p = circle(0.01).translate(pt[0], pt[1]).fill_color(black)
            p = p.named(("x", i))
            i += 1
            scores.append(p)
    d = (
        (concat(scores) + make_path(path)).center_xy().scale(3)
        / vstrut(0.5)
        / d.scale(2).center_xy()
    )
    for i in range(i):
        d = d.connect(("graph", i), ("x", i), ArrowOpts(head_pad=0.1))
    return d


def draw_with_hard_points(model, c1=None, c2=None):
    if c1 is None:
        d = draw_graph(model)
    else:
        d = draw_graph(model, c1=c1, c2=c2)
    return d + split_graph(s1_hard, s2_hard, show_origin=False)

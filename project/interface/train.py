import time

import graph_builder
import interface.plots as plots
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import minitorch


def render_train_interface(
    TrainCls, graph=True, hidden_layer=True, parameter_control=False
):
    datasets_map = minitorch.datasets
    st.write("## Sandbox for Model Training")

    st.markdown("### Dataset")
    col1, col2 = st.columns(2)
    points = col2.slider("Number of points", min_value=1, max_value=150, value=50)
    selected_dataset = col1.selectbox("Select dataset", list(datasets_map.keys()))

    @st.cache
    def get_dataset(selected_dataset, points):
        return datasets_map[selected_dataset](points)

    dataset = get_dataset(selected_dataset, points)

    fig = plots.plot_out(dataset)
    fig.update_layout(width=600, height=600)
    st.plotly_chart(fig)

    st.markdown("### Model")
    if hidden_layer:
        hidden_layers = st.number_input(
            "Size of hidden layer", min_value=1, max_value=200, step=1, value=2
        )
    else:
        hidden_layers = 0

    @st.cache
    def get_train(hidden_layers):
        train = TrainCls(hidden_layers)
        one_output = train.run_one(dataset.X[0])
        G = graph_builder.GraphBuilder().run(one_output)
        return nx.nx_pydot.to_pydot(G).to_string()

    train = TrainCls(hidden_layers)
    if graph:
        graph = get_train(hidden_layers)
        if st.checkbox("Show Graph"):
            st.graphviz_chart(graph)

    if parameter_control:
        st.markdown("### Parameters")
        for n, p in train.model.named_parameters():
            value = st.slider(
                f"Parameter: {n}", min_value=-10.0, max_value=10.0, value=p.value
            )
            p.update(value)

    oned = st.checkbox("Show X-Axis Only (For Simple)", False)

    def plot():
        if hasattr(train, "run_many"):

            def contour(ls):
                t = train.run_many(ls)
                return [t[i, 0] for i in range(len(ls))]

        else:

            def contour(ls):
                out = [train.run_one(x) for x in ls]
                out = [(x.data if hasattr(x, "data") else x) for x in out]
                return out

        fig = plots.plot_out(dataset, contour, size=15, oned=oned)
        fig.update_layout(width=600, height=600)
        return fig

    st.markdown("### Initial setting")
    st.write(plot())

    if hasattr(train, "train"):
        st.markdown("### Hyperparameters")
        col1, col2 = st.columns(2)
        learning_rate = col1.selectbox(
            "Learning rate", [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0], index=2
        )

        max_epochs = col2.number_input(
            "Number of epochs", min_value=1, step=25, value=500
        )

        col1, col2 = st.columns(2)
        st_train_button = col1.empty()
        col2.button("Stop Model")

    st_progress = st.empty()
    st_epoch_timer = st.empty()
    st_epoch_image = st.empty()
    st_epoch_plot = st.empty()
    st_epoch_stats = st.empty()

    start_time = time.time()

    df = []

    def log_fn(epoch, total_loss, correct, losses):
        time_elapsed = time.time() - start_time
        if hasattr(train, "train"):
            st_progress.progress(epoch / max_epochs)
            time_per_epoch = time_elapsed / (epoch + 1)
            st_epoch_timer.markdown(
                "Epoch {}/{}. Time per epoch: {:,.3f}s. Time left: {:,.2f}s.".format(
                    epoch,
                    max_epochs,
                    time_per_epoch,
                    (max_epochs - epoch) * time_per_epoch,
                )
            )
        df.append({"epoch": epoch, "loss": total_loss, "correct": correct})
        st_epoch_stats.write(pd.DataFrame(reversed(df)))

        st_epoch_image.plotly_chart(plot())
        if hasattr(train, "train"):
            loss_graph = go.Scatter(mode="lines", x=list(range(len(losses))), y=losses)
            fig = go.Figure(loss_graph)
            fig.update_layout(
                title="Loss Graph",
                xaxis=dict(range=[0, max_epochs]),
                yaxis=dict(range=[0, max(losses)]),
            )
            st_epoch_plot.plotly_chart(fig)

            print(
                f"Epoch: {epoch}/{max_epochs}, loss: {total_loss}, correct: {correct}"
            )

    if hasattr(train, "train") and st_train_button.button("Train Model"):
        train.train(dataset, learning_rate, max_epochs, log_fn)
    else:
        log_fn(0, 0, 0, [0])

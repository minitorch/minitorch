import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from run_mnist_multiclass import ImageTrain, make_mnist


def render_run_image_interface():

    st.markdown("### Dataset")
    n_training_samples = st.number_input(
        "Number of training samples",
        min_value=100,
        max_value=10000,
        step=10,
        value=1000,
    )
    (X_train, y_train) = make_mnist(0, n_training_samples)

    show = st.number_input("Image", min_value=0, max_value=100, step=1, value=1)
    st.write(
        px.imshow(X_train[show], title="y =" + str([int(i) for i in y_train[show]]))
    )

    st.markdown("### Hyperparameters")
    col1, col2 = st.columns(2)
    learning_rate = col1.selectbox(
        "Learning rate", [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0], index=2
    )
    max_epochs = col2.number_input("Number of epochs", min_value=1, step=25, value=500)

    col1, col2 = st.columns(2)
    st_train_button = col1.empty()
    col2.button("Stop Model")

    st_progress = st.empty()
    st_epoch_timer = st.empty()
    st_epoch_image = st.empty()
    st_epoch_plot = st.empty()
    st_epoch_stats = st.empty()

    df = []

    if st_train_button.button("Train Model"):
        start_time = time.time()
        train = ImageTrain()

        def log_fn(epoch, total_loss, correct, losses, model):
            time_elapsed = time.time() - start_time
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

            # Visualize test batch
            fig = px.imshow(-1 * model.mid.to_numpy()[0], facet_col=0)
            st_epoch_image.plotly_chart(fig)

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

        train.train(
            (X_train, y_train),
            make_mnist(10000, 10500),
            learning_rate,
            max_epochs,
            log_fn,
        )

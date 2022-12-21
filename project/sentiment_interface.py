import time

import embeddings
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from project.run_sentiment import CNNSentimentKim, encode_sentiment_data
from run_sentiment import SentenceSentimentTrain

from datasets import load_dataset

EMBEDDING_SIZE = 50


def predictions_dataframe(predictions, sentences):
    df_predictions = pd.DataFrame(
        predictions, columns=["true_label", "predicted_label", "logit"]
    )
    df_predictions["sentence"] = sentences[: len(predictions)]
    df_predictions["error"] = df_predictions.apply(
        lambda row: abs(row.true_label - row.logit), axis=1
    )
    # reorder
    return df_predictions[
        ["sentence", "true_label", "predicted_label", "logit", "error"]
    ]


@st.cache(allow_output_mutation=True)
def load_glue_dataset():
    print("Loading dataset")
    return load_dataset("glue", "sst2")


#  st.subheader("Encode training data as nxk Glove embeddings tensor representations")
@st.cache(allow_output_mutation=True)
def load_data(dataset, n_train, n_val):
    print("Loading embeddings... This can take a while the first time.")
    return encode_sentiment_data(
        dataset,
        embeddings.GloveEmbedding(
            "wikipedia_gigaword", d_emb=EMBEDDING_SIZE, show_progress=True
        ),
        n_train,
        n_val,
    )


def render_run_sentiment_interface():

    st.header("Sentiment Classification")
    st.write(
        "[Glue SS2 Dataset documentation](https://huggingface.co/datasets/glue/viewer/sst2/train)"
    )

    dataset_classes = ["negative", "positive"]
    class_id_to_label = {idx: label for (idx, label) in enumerate(dataset_classes)}
    st.write("Each sentence is classified according to it's sentiment:")
    st.write(class_id_to_label)
    dataset = load_glue_dataset()
    train_sentences = dataset["train"]["sentence"]
    # validation_sentences = dataset["validation"]["sentence"]
    st.subheader("First 3 sentences in train")
    st.table(
        pd.DataFrame(
            list(zip(dataset["train"]["sentence"][:3], dataset["train"]["label"][:3])),
            columns=["Sentence", "Label"],
        )
    )

    st.subheader("CNN Hyperparameters")
    col1, col2 = st.columns(2)
    feature_map_size = col1.number_input("Feature map size", value=100)
    dropout = col2.number_input("Dropout", value=0.25)
    st.write("**Conv layer filter sizes**")
    col1, col2, col3 = st.columns(3)
    filter_size_1 = col1.number_input("Filter 1", value=3)
    filter_size_2 = col2.number_input("Filter 2", value=4)
    filter_size_3 = col3.number_input("Filter 3", value=5)

    st.subheader("Training config")
    col1, col2 = st.columns(2)
    max_epochs = col1.number_input("Max epochs", value=250)
    learning_rate = col2.number_input(
        "Learning rate", value=0.01, step=0.001, format="%.3f"
    )
    n_training_data = col1.number_input("N datapoints from training data", value=450)
    n_validation_data = col2.number_input(
        "N datapoints from validation data", value=100
    )
    batch_size = st.number_input("Batch size", value=10)

    if st.button("Train model"):
        df = []
        st_progress = st.progress(0 / max_epochs)
        st_epoch_timer = st.markdown("Epoch {}/{}".format(0, max_epochs))
        st_epoch_accuracy = st.empty()
        st_epoch_plot = st.empty()
        with st.expander("Predictions for training data"):
            st_train_predictions = st.empty()
        with st.expander("Predictions for validation data"):
            st_validation_predictions = st.empty()
        with st.expander("Epoch stats"):
            st_epoch_stats = st.empty()

        (X_train, y_train), (X_val, y_val) = load_data(
            dataset, n_training_data, n_validation_data
        )
        print("Initializing model...")
        sentiment_model_trainer = SentenceSentimentTrain(
            CNNSentimentKim(
                feature_map_size=feature_map_size,
                filter_sizes=[filter_size_1, filter_size_2, filter_size_3],
                dropout=dropout,
            )
        )
        start_time = time.time()

        def log_fn(
            epoch,
            total_loss,
            losses,
            train_predictions,
            train_accuracy,
            validation_predictions,
            validation_accuracy,
        ):
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
            df.append(
                {
                    "epoch": epoch,
                    "loss": total_loss,
                    "train_accuracy": train_accuracy[-1],
                    "validation_accuracy": validation_accuracy[-1],
                }
            )
            st_epoch_stats.write(pd.DataFrame(reversed(df)))

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    mode="lines",
                    x=list(range(len(train_accuracy))),
                    y=train_accuracy,
                    name="train accuracy",
                )
            )
            fig.add_trace(
                go.Scatter(
                    mode="lines",
                    x=list(range(len(validation_accuracy))),
                    y=validation_accuracy,
                    name="validation accuracy",
                )
            )
            fig.update_layout(
                title="Accuracy Graph",
                xaxis=dict(range=[0, max_epochs]),
                yaxis=dict(range=[0, 1]),
            )
            st_epoch_accuracy.plotly_chart(fig)

            loss_graph = go.Scatter(mode="lines", x=list(range(len(losses))), y=losses)
            fig = go.Figure(loss_graph)
            fig.update_layout(
                title="Loss Graph",
                xaxis=dict(range=[0, max_epochs]),
                yaxis=dict(range=[0, max(losses)]),
            )
            st_epoch_plot.plotly_chart(fig)

            # Visualize predictions
            st_train_predictions.dataframe(
                predictions_dataframe(train_predictions, train_sentences)
            )
            st_validation_predictions.dataframe(
                predictions_dataframe(validation_predictions, train_sentences)
            )

            print(
                f"Epoch: {epoch}/{max_epochs}, loss: {total_loss}, train accuracy: {train_accuracy[-1]}"
            )

        sentiment_model_trainer.train(
            (X_train, y_train),
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            log_fn=log_fn,
            data_val=(X_val, y_val),
        )

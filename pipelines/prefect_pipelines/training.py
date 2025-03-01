from prefect import flow, task


@flow(name="training-pipeline", log_prints=True)
def train_model():
    pass


if __name__ == "__main__":
    train_model()

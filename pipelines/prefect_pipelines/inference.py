from prefect import flow, task


@flow(name="inference-pipeline", log_prints=True)
def run_inference():
    print("Running inference pipeline")
    pass


if __name__ == "__main__":
    run_inference()

from prefect_pipelines.inference import run_inference


if __name__ == '__main__':
    run_inference.serve(name="inference-pipeline")

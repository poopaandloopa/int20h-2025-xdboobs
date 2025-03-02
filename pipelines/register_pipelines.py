import multiprocessing
from prefect_pipelines.training import data_processing_pipeline, train_model_pipeline
from prefect_pipelines.inference import inference_pipeline


def serve_pipeline(pipeline, name):
    print(f"Serving {name}...")
    pipeline.serve(name=name)


if __name__ == '__main__':
    pipelines = [
        (data_processing_pipeline, "data-processing-pipeline"),
        (train_model_pipeline, "model-training-pipeline"),
        (inference_pipeline, "inference-pipeline"),
    ]

    processes = []
    for pipeline, name in pipelines:
        p = multiprocessing.Process(target=serve_pipeline, args=(pipeline, name))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
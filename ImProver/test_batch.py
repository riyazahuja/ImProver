import ray
from packaging.version import Version
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
from ray.data import DataContext
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
ray.init(num_cpus=12, num_gpus=2)
DataContext.get_current().wait_for_min_actors_s = 1800 

assert Version(ray.__version__) >= Version(
    "2.44.1"), "Ray version must be at least 2.44.1"



# Uncomment to reduce clutter in stdout
# ray.init(log_to_driver=False)
# ray.data.DataContext.get_current().enable_progress_bars = False

# Read one text file from S3. Ray Data supports reading multiple files
# from cloud storage (such as JSONL, Parquet, CSV, binary format).
ds = ray.data.read_text("s3://anonymous@air-example-data/prompts.txt")
print(ds.schema())

size = ds.count()
print(f"Size of dataset: {size} prompts")

# ctx.execution_options = ExecutionOptions(task_extra_resources={"CPU": 0.25})

# Configure vLLM engine.
config = vLLMEngineProcessorConfig(
    model_source="deepseek-ai/DeepSeek-Prover-V2-7B",
    engine_resources={"CPU": 6, "GPU": 1},
    concurrency=2,                            # spawn 2 replicas
    engine_kwargs={
        "tensor_parallel_size": 1,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 4096,
        "max_model_len": 16384,
    },
    max_concurrent_batches=16,
    batch_size=128,
)

# Create a Processor object, which will be used to
# do batch inference on the dataset
vllm_processor = build_llm_processor(
    config,
    preprocess=lambda row: dict(
        messages=[{
            "role": "system",
            "content": "You are a bot that responds with haikus."
        }, {
            "role": "user",
            "content": row["text"]
        }],
        sampling_params=dict(
            temperature=0.3,
            max_tokens=250,
        )),
    postprocess=lambda row: dict(
        answer=row["generated_text"],
        **row  # This will return all the original columns in the dataset.
    ),
)

# ds = vllm_processor(ds)

# Peek first 10 results.
# NOTE: This is for local testing and debugging. For production use case,
# one should write full result out as shown below.
# outputs = ds.take(limit=10)

# for output in outputs:
#     prompt = output["prompt"]
#     generated_text = output["generated_text"]
#     print(f"Prompt: {prompt!r}")
#     print(f"Generated text: {generated_text!r}")
print("HERE")
ds = vllm_processor(ds).materialize()   # run everything, keep distributed
ds.write_json("local:///home/riyaza/data_out.json")

# Write inference output data out as Parquet files to S3.
# Multiple files would be written to the output destination,
# and each task would write one or more files separately.
#
# ds.write_parquet("s3://<your-output-bucket>")

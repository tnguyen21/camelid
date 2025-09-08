import os
import modal

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .run_commands(
        "pip install --upgrade pip",
        "pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade",
    )
    .pip_install(
        "numpy",
        "tqdm",
        "huggingface-hub",
    )
    .add_local_file("train_llama.py", remote_path="/train_llama.py")
)

app = modal.App(name="modded-llama2", image=image)


def _download_fineweb10b(num_train_shards: int) -> None:
    from huggingface_hub import hf_hub_download

    local_dir = os.path.join("data", "fineweb10B")
    os.makedirs(local_dir, exist_ok=True)

    def get(fname: str):
        path = os.path.join(local_dir, fname)
        if not os.path.exists(path):
            hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=fname, repo_type="dataset", local_dir=local_dir)

    get("fineweb_val_%06d.bin" % 0)
    for i in range(1, int(num_train_shards) + 1):
        get("fineweb_train_%06d.bin" % i)


@app.function(
    gpu="H100:8",  # 8x H100 to match world_size == 8
    timeout=24 * 60 * 60,  # 24 hours
    cpu=32,  # More CPU for 7B model
    memory=256,  # More memory for 7B model
)
def run_training(num_train_shards: int = 8):
    """
    Download FineWeb shards, then spawn 8 ranks to run the 6.6B Llama2 training code.
    """
    import subprocess
    import sys
    import os

    # Download data shards
    _download_fineweb10b(int(num_train_shards))

    # Launch training across 8 GPUs (one process per GPU)
    result = subprocess.run(["torchrun", "--nproc_per_node=8", "/train_llama.py"], capture_output=True, text=True)

    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
    print(f"Return code: {result.returncode}")

    return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}


# Convenience function to run locally
@app.local_entrypoint()
def main(num_shards: int = 8):
    """
    Run the training with specified parameters
    """
    result = run_training.remote(num_train_shards=num_shards)
    print(f"Training completed with return code: {result['returncode']}")
    if result["returncode"] != 0:
        print("STDERR output:")
        print(result["stderr"])

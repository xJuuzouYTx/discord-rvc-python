from huggingface_hub import snapshot_download
repo_id = "juuxn/RVCModels"

snapshot_download(repo_id=repo_id, local_dir="./", max_workers=10)
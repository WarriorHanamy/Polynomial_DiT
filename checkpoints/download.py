from huggingface_hub import snapshot_download

REPO_ID = "dwl1437/Polynomial-DiT"

snapshot_download(
    repo_id=REPO_ID,
    repo_type="model",
    local_dir="Polynomial-DiT-local",
    resume_download=True,
    local_dir_use_symlinks=False,
    ignore_patterns=[".git/*"],
)

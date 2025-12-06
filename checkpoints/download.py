
from huggingface_hub import snapshot_download

# 设置 Token 和仓库 ID
TOKEN = "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"
REPO_ID = "dwl1437/Polynomial-DiT"

# 下载权重到本地目录
snapshot_download(
    repo_id=REPO_ID,
    repo_type="model",
    local_dir="Polynomial-DiT-local",
    token=TOKEN,
    resume_download=True,
    local_dir_use_symlinks=False,
    ignore_patterns=[".git/*"]
)

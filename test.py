from huggingface_hub import snapshot_download
import os

os.environ['REQUESTS_CA_BUNDLE'] = ''
snapshot_download(
    repo_id="vidore/colpali-v1.3",
    local_dir="colpali-v1.3",
    local_dir_use_symlinks=False,
    resume_download=True
)

print("✅ Model downloaded to ./colpali-v1.3")

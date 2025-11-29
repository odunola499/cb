from huggingface_hub import hf_hub_download
from safetensors.torch import safe_open


def load_weights(repo_id, filename="model.safetensors"):
    local_filepath = hf_hub_download(repo_id=repo_id, filename=filename)
    tensors = {}
    with safe_open(local_filepath, framework="pt") as fp:
        for key in fp.keys():
            tensors[key[6:]] = fp.get_tensor(key)
    tensors["lm_head.weight"] = tensors["embed_tokens.weight"]
    return tensors

from .llama_adapter   import LLaMAAdapter
from .qwen_adapter    import QwenAdapter
from .gemma_adapter   import GemmaAdapter
from .phi_adapter     import PhiAdapter
from .mistral_adapter import MistralAdapter

_ADAPTERS = {
    "qwen"      : QwenAdapter,
    "qwen_base" : QwenAdapter,    # same architecture, different model_id
    "qwen_heldout" : QwenAdapter,
    "llama"     : LLaMAAdapter,
    "llama_base": LLaMAAdapter,   # same architecture, different model_id
    # New (offline prep, unverified against real checkpoints; included for
    # forthcoming releases — untested against the current Qwen det_4x4 data
    # drop). Registered here so run.py --model {gemma,phi,mistral} resolves
    # once corresponding MODEL_CONFIGS entries are added to config.py.
    "gemma"     : GemmaAdapter,
    "phi"       : PhiAdapter,
    "mistral"   : MistralAdapter,
}


def get_adapter(model_name: str, config: dict):
    if model_name not in _ADAPTERS:
        raise ValueError(f"No adapter for '{model_name}'. Available: {list(_ADAPTERS)}")
    return _ADAPTERS[model_name](model_name, config)

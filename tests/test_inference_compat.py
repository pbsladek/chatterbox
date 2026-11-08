"""Small real backbone passes catch Transformers/Torch API regressions without weights."""

import numpy as np
import pytest
import torch

from chatterbox.models.t3 import T3
from chatterbox.models.t3.llama_configs import LLAMA_CONFIGS
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.t3.modules.t3_config import T3Config


@pytest.mark.parametrize("backbone,cfg_weight", [("llama", 0.0), ("llama", 0.5), ("gpt2", 0.0)])
def test_sampling_with_kv_cache_on_tiny_backbones(monkeypatch, backbone, cfg_weight):
    config = dict(model_type=backbone, hidden_size=32, vocab_size=8,
                  num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                  intermediate_size=64, max_position_embeddings=128,
                  n_embd=32, n_layer=1, n_head=2, n_positions=128,
                  attn_implementation="eager")
    monkeypatch.setitem(LLAMA_CONFIGS, "test", config)
    hp = T3Config()
    hp.llama_config_name = "test"
    hp.use_perceiver_resampler = False
    hp.speech_cond_prompt_len = 0
    if backbone == "gpt2":
        hp.input_pos_emb = None
    model = T3(hp).eval()
    cond = T3Cond(speaker_emb=torch.zeros(1, 256), emotion_adv=torch.ones(1, 1, 1) * 0.5)
    text = torch.tensor([[hp.start_text_token, 10, hp.stop_text_token]])
    torch.manual_seed(42)
    if backbone == "llama":
        batch_size = 2 if cfg_weight > 0 else 1
        batches = []
        def observe_batch(module, args, kwargs):
            batches.append(kwargs["inputs_embeds"].size(0))
        handle = model.tfmr.register_forward_pre_hook(observe_batch, with_kwargs=True)
        tokens = model.inference(t3_cond=cond, text_tokens=text.repeat(batch_size, 1),
                                 max_new_tokens=3, cfg_weight=cfg_weight)
        handle.remove()
        assert batches and all(batch == batch_size for batch in batches)
        assert 1 <= tokens.shape[1] <= 3
    else:
        tokens = model.inference_turbo(cond, text, max_gen_len=2)
        assert 1 <= tokens.shape[1] <= 3
    assert tokens.shape[0] == 1
    assert tokens.dtype == torch.int64
    assert torch.all((tokens >= 0) & (tokens < hp.speech_tokens_dict_size))


def test_bundled_perth_weights_can_watermark_audio_offline():
    import perth
    watermarker = perth.PerthImplicitWatermarker()
    samples = np.sin(np.arange(24000, dtype=np.float32) * 2 * np.pi * 220 / 24000) * 0.1
    result = watermarker.apply_watermark(samples, sample_rate=24000)
    assert result.size > 0
    assert np.isfinite(result).all()
    assert float(np.sqrt(np.mean(result ** 2))) > 1e-5

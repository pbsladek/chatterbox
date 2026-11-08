"""Regressions reproduced while reviewing the retired upstream PR backlog."""

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch

from chatterbox.models.utils import load_map_location, resolve_device


@pytest.mark.parametrize("device", ["cpu", "cpu:0", torch.device("cpu"), "mps", "mps:0",
                                     torch.device("mps"), "cuda", "cuda:3", "xpu:0"])
def test_checkpoint_staging_never_uses_the_saved_gpu_index(device):
    assert load_map_location(device) == torch.device("cpu")


@pytest.mark.parametrize("device", ["mps", "mps:0", torch.device("mps")])
def test_unavailable_mps_fallback_handles_all_device_representations(monkeypatch, device):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device(device) == "cpu"


@pytest.mark.parametrize("module_name,class_name", [
    ("tts", "ChatterboxTTS"), ("mtl_tts", "ChatterboxMultilingualTTS"),
    ("tts_turbo", "ChatterboxTurboTTS"), ("vc", "ChatterboxVC"),
])
@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("mps"), "mps:0"])
def test_all_hub_loaders_normalize_devices(monkeypatch, tmp_path, module_name, class_name, device):
    module = importlib.import_module(f"chatterbox.{module_name}")
    cls = getattr(module, class_name)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(module, "download_snapshot", Mock(return_value=str(tmp_path)))
    load = Mock(return_value="loaded")
    monkeypatch.setattr(cls, "from_local", load)
    assert cls.from_pretrained(device) == "loaded"
    assert load.call_args.args[1] == "cpu"


@pytest.mark.parametrize("module_name,class_name", [
    ("tts", "ChatterboxTTS"), ("mtl_tts", "ChatterboxMultilingualTTS"),
    ("tts_turbo", "ChatterboxTurboTTS"), ("vc", "ChatterboxVC"),
])
def test_local_loaders_stage_conditioning_and_pt_weights_on_cpu(
        monkeypatch, tmp_path, module_name, class_name):
    module = importlib.import_module(f"chatterbox.{module_name}")
    cls = getattr(module, class_name)
    (tmp_path / "conds.pt").touch()
    monkeypatch.setattr(cls, "__init__", lambda self, *args, **kwargs: None)
    for name in ["VoiceEncoder", "T3", "S3Gen", "EnTokenizer", "MTLTokenizer"]:
        if hasattr(module, name):
            monkeypatch.setattr(module, name, MagicMock())
    if hasattr(module, "AutoTokenizer"):
        tokenizer = MagicMock()
        tokenizer.__len__.return_value = 50276
        monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", Mock(return_value=tokenizer))
    for name in ["load_file", "load_safetensors"]:
        if hasattr(module, name):
            monkeypatch.setattr(module, name, Mock(return_value={}))
    pt_load = Mock(return_value={"gen": {}})
    monkeypatch.setattr(torch, "load", pt_load)
    conditional_load = None
    if hasattr(module, "Conditionals"):
        conditional_load = Mock()
        monkeypatch.setattr(module.Conditionals, "load", conditional_load)
    cls.from_local(tmp_path, torch.device("cpu"))
    calls = pt_load.call_args_list + (conditional_load.call_args_list if conditional_load else [])
    assert calls
    assert all(call.kwargs["map_location"] == torch.device("cpu") for call in calls)


def test_chinese_converter_initializes_once_and_only_for_chinese(monkeypatch, tmp_path):
    from chatterbox.models.tokenizers import tokenizer as module
    native = Mock()
    native.get_vocab.return_value = {module.SOT: 0, module.EOT: 1}
    native.encode.return_value = SimpleNamespace(ids=[1, 2])
    monkeypatch.setattr(module.Tokenizer, "from_file", Mock(return_value=native))
    converter = Mock(return_value="converted")
    factory = Mock(return_value=converter)
    monkeypatch.setattr(module, "ChineseCangjieConverter", factory)
    tokenizer = module.MTLTokenizer(str(tmp_path / "tokenizer.json"))
    tokenizer.encode("Bonjour", language_id="fr")
    tokenizer.encode("Olá", language_id="pt")
    factory.assert_not_called()
    assert tokenizer.encode("你好", language_id="zh") == [1, 2]
    tokenizer.encode("再见", language_id="zh")
    factory.assert_called_once_with(tmp_path)
    assert converter.call_count == 2
    native.encode.assert_called_with("[zh]converted")


@pytest.mark.parametrize("text", ["मेरा नाम हसन है।", "श्लोक ॥", "كيف حالك؟", "مرحبا،",
                                    "مرحبا۔", "你好。", "こんにちは！", "Hello!", "A question?"])
def test_multilingual_sentence_punctuation_is_preserved(text):
    from chatterbox.mtl_tts import punc_norm
    assert punc_norm(text) == text


def test_multilingual_unpunctuated_text_still_gets_a_period():
    from chatterbox.mtl_tts import punc_norm
    assert punc_norm("नमस्ते") == "नमस्ते."


def test_turbo_loudness_normalization_preserves_float32_with_numpy_1_and_2():
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    waveform = np.sin(np.arange(24000, dtype=np.float32) * np.float32(0.05)) * np.float32(0.1)
    normalized = ChatterboxTurboTTS.__new__(ChatterboxTurboTTS).norm_loudness(waveform, 24000)
    assert normalized.dtype == np.float32
    assert np.isfinite(normalized).all()
    assert normalized.shape == waveform.shape
    assert not np.allclose(normalized, waveform)


@pytest.mark.parametrize("waveform", [np.ones(160, dtype=np.float64), torch.ones(160, dtype=torch.float64)])
def test_speech_tokenizer_casts_double_audio_before_device_transfer(waveform):
    from chatterbox.models.s3tokenizer import S3Tokenizer
    prepared = S3Tokenizer._prepare_audio(None, [waveform])
    assert prepared[0].dtype == torch.float32
    assert prepared[0].shape == (1, 160)


def test_voice_encoder_casts_double_mels_before_device_transfer():
    from chatterbox.models.voice_encoder import VoiceEncoder
    inference = Mock(return_value=torch.zeros(1, 256))
    encoder = SimpleNamespace(device=torch.device("cpu"), inference=inference)
    result = VoiceEncoder.embeds_from_mels(encoder, torch.ones(1, 20, 80, dtype=torch.float64))
    assert result.shape == (1, 256)
    assert inference.call_args.args[0].dtype == torch.float32


@pytest.mark.parametrize("module_name,class_name", [
    ("tts", "ChatterboxTTS"), ("mtl_tts", "ChatterboxMultilingualTTS"),
])
@pytest.mark.parametrize("cfg_weight,batch_size", [(0.0, 1), (0.5, 2)])
def test_generate_uses_one_text_batch_when_guidance_is_disabled(module_name, class_name, cfg_weight, batch_size):
    module = importlib.import_module(f"chatterbox.{module_name}")
    cls = getattr(module, class_name)
    model = cls.__new__(cls)
    model.device = "cpu"
    model.sr = 24000
    model.conds = SimpleNamespace(t3=SimpleNamespace(emotion_adv=torch.tensor([[[0.5]]])), gen={})
    model.tokenizer = Mock()
    model.tokenizer.text_to_tokens.return_value = torch.tensor([[1, 2]])
    model.t3 = Mock()
    model.t3.hp.start_text_token = 255
    model.t3.hp.stop_text_token = 0
    model.t3.inference.return_value = torch.tensor([[10, 11]])
    model.s3gen = Mock()
    model.s3gen.inference.return_value = (torch.ones(1, 2400) * 0.01, None)
    model.watermarker = Mock()
    model.watermarker.apply_watermark.side_effect = lambda waveform, sample_rate: waveform
    options = {"language_id": "fr"} if module_name == "mtl_tts" else {}
    waveform = model.generate("Bonjour", cfg_weight=cfg_weight, **options)
    assert model.t3.inference.call_args.kwargs["text_tokens"].shape[0] == batch_size
    assert waveform.shape[0] == 1 and torch.isfinite(waveform).all()

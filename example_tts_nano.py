import soundfile as sf
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Automatically detect the best available device.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Load the Nano model (same class as Turbo, selected with nano=True).
model = ChatterboxTurboTTS.from_pretrained(device=device, nano=True)

# Generate with Paralinguistic Tags
text = "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. It's the SkyNet T-800 series and it's got basically everything. Including AI integration with ChatGPT and all that jazz. Would you like me to get some prices for you?"

# Generate audio (requires a reference clip for voice cloning)
# wav = model.generate(text, audio_prompt_path="your_10s_ref_clip.wav")
wav = model.generate(text)
sf.write("test-nano.wav", wav.squeeze(0).detach().cpu().numpy(), model.sr, subtype="FLOAT")

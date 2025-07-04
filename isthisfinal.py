import streamlit as st
import torch
import torchaudio
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import noisereduce as nr
import tempfile
import os

# Try importing microphone libraries
try:
    import sounddevice as sd
    import wavio
    HAS_MIC = True
except:
    HAS_MIC = False

from asteroid.models import ConvTasNet
from asteroid.utils import tensors_to_device

# Streamlit config
st.set_page_config(page_title="Real-Time BSS", layout="wide")
st.title("🎙️ Blind Source Separation (ConvTasNet + Noise Reduction)")

# Load model
@st.cache_resource
def load_model():
    model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
    model.eval()
    return model

model = load_model()

# Plot audio waveform and spectrogram
def plot_audio_features(audio, sr, title="Audio"):
    fig, axs = plt.subplots(2, 1, figsize=(8, 4))
    axs[0].plot(audio)
    axs[0].set_title(f"{title} - Waveform")
    axs[1].specgram(audio, Fs=sr, NFFT=1024, noverlap=512)
    axs[1].set_title(f"{title} - Spectrogram")
    st.pyplot(fig)

# Input method
options = ["Upload Audio File"]
if HAS_MIC:
    options.append("Record via Microphone")

input_method = st.radio("Select Input Source", options)

waveform = None
sr = None

# Upload
if input_method == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload a 2-speaker mixed WAV file", type=["wav"])
    if uploaded_file is not None:
        waveform, sr = torchaudio.load(uploaded_file)

# Record (only if sounddevice available)
elif input_method == "Record via Microphone" and HAS_MIC:
    duration = st.slider("Recording duration (seconds)", 1, 20, 5)
    if st.button("🎙️ Record Now"):
        st.info("Recording...")
        fs = 16000
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        temp_path = tempfile.mktemp(suffix=".wav")
        wavio.write(temp_path, recording, fs, sampwidth=2)
        waveform, sr = torchaudio.load(temp_path)
        st.success("Recording complete.")
        st.audio(temp_path, format="audio/wav")

# Main processing
if waveform is not None and sr is not None:
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 16000:
        st.warning(f"Resampling from {sr} Hz to 16kHz...")
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        sr = 16000

    st.subheader("🎧 Input Mixture")
    temp_mix = tempfile.mktemp(suffix=".wav")
    sf.write(temp_mix, waveform.squeeze().numpy(), sr)
    st.audio(temp_mix, format="audio/wav")
    plot_audio_features(waveform[0].numpy(), sr, "Mixture")

    # Separate sources
    with st.spinner("Separating sources..."):
        input_tensor = waveform.unsqueeze(0)
        input_tensor = tensors_to_device(input_tensor, device="cpu")
        model.to("cpu")
        with torch.no_grad():
            separated = model.separate(input_tensor)

    src1 = separated[0, 0].cpu().numpy()
    src2 = separated[0, 1].cpu().numpy()

    st.subheader("🛠 Noise Reduction")
    reduced_src1 = nr.reduce_noise(y=src1, sr=sr)
    reduced_src2 = nr.reduce_noise(y=src2, sr=sr)

    # Save outputs
    temp_src1 = tempfile.mktemp(suffix=".wav")
    temp_src2 = tempfile.mktemp(suffix=".wav")
    sf.write(temp_src1, reduced_src1, sr)
    sf.write(temp_src2, reduced_src2, sr)

    # Output results
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🗣️ Source 1**")
        st.audio(temp_src1, format="audio/wav")
        plot_audio_features(reduced_src1, sr, "Source 1")

    with col2:
        st.markdown("**🗣️ Source 2**")
        st.audio(temp_src2, format="audio/wav")
        plot_audio_features(reduced_src2, sr, "Source 2")

    st.success("✅ Source separation and denoising complete!")

else:
    st.info("Please upload or record an audio file to begin.")

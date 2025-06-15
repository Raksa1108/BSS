import streamlit as st
import torch
import torchaudio
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import noisereduce as nr
import sounddevice as sd
import wavio
from asteroid.models import ConvTasNet
from asteroid.utils import tensors_to_device
import tempfile
import os

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

# Plotting
def plot_audio_features(audio, sr, title="Audio"):
    fig, axs = plt.subplots(2, 1, figsize=(8, 4))
    axs[0].plot(audio)
    axs[0].set_title(f"{title} - Waveform")
    axs[1].specgram(audio, Fs=sr, NFFT=1024, noverlap=512)
    axs[1].set_title(f"{title} - Spectrogram")
    st.pyplot(fig)

# Choose input method
input_method = st.radio("Select Input Source", ["Upload Audio File", "Record via Microphone"])

# File uploader or recorder
if input_method == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload a 2-speaker mixed WAV file", type=["wav"])
    if uploaded_file is not None:
        waveform, sr = torchaudio.load(uploaded_file)
elif input_method == "Record via Microphone":
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

# Proceed only if waveform is loaded
if 'waveform' in locals():
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 16000:
        st.warning(f"Resampling from {sr} Hz to 16kHz...")
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        sr = 16000

    st.subheader("🎧 Input Mixture")
    st.audio(sf.write("mixture.wav", waveform.squeeze().numpy(), sr), format="audio/wav")
    plot_audio_features(waveform[0].numpy(), sr, "Mixture")

    # Separation
    with st.spinner("Separating sources..."):
        input_tensor = waveform.unsqueeze(0)
        input_tensor = tensors_to_device(input_tensor, device="cpu")
        model.to("cpu")
        with torch.no_grad():
            separated = model.separate(input_tensor)

    # Extract and noise reduce
    src1 = separated[0, 0].cpu().numpy()
    src2 = separated[0, 1].cpu().numpy()

    st.subheader("🛠 Noise Reduction")
    reduced_src1 = nr.reduce_noise(y=src1, sr=sr)
    reduced_src2 = nr.reduce_noise(y=src2, sr=sr)

    # Save outputs
    sf.write("source1_clean.wav", reduced_src1, sr)
    sf.write("source2_clean.wav", reduced_src2, sr)

    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🗣️ Source 1**")
        st.audio("source1_clean.wav", format="audio/wav")
        plot_audio_features(reduced_src1, sr, "Source 1")

    with col2:
        st.markdown("**🗣️ Source 2**")
        st.audio("source2_clean.wav", format="audio/wav")
        plot_audio_features(reduced_src2, sr, "Source 2")

    st.success("✅ Source separation and denoising complete!")





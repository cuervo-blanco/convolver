import numpy as np
import tkinter as tk
from tkinter import filedialog
import soundfile as sf
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt


def get_audio_file_directory():
    """Open a file dialog to select an audio file and return the directory."""
    # Create a hidden root window
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(
            title="Select an audio file",
            filetypes=[("Audio Files", "*.mp3 *.wav *.aac *.ogg *.flac")],
            )
    if file_path:
        # Get the directory name
        return file_path
    else:
        print("No file was selected.")
        return None


def load_audio(filename):
    """Load and audio file."""
    signal, sample_rate = sf.read(filename)
    return signal, sample_rate


def normalize_audio(signal):
    """Normalize the signal to range [-1, 1]."""
    return signal / np.max(np.abs(signal))


def clip_signal(signal):
    """Clip the signal to the range [-1, 1]."""
    return np.clip(signal, -1.0, 1.0)


def convolve_signals(signal, impulse_response):
    """Convolve the audio signal with the impulse_response."""
    return fftconvolve(signal, impulse_response, mode='full')


def save_audio(filename, signal, sample_rate):
    """Save the convolved signal to a file."""
    sf.write(filename, signal, sample_rate)


def plot_signals(original, convolved):
    """Plot the original and convolved signals."""
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.title("Original Signal")
    plt.plot(original)
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.title("Convolved Signal")
    plt.plot(convolved)
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


def main():
    # Load the audio signal and impulse response
    audio_file = get_audio_file_directory()
    print(f"Selected audio file: {audio_file}")
    impulse_file = get_audio_file_directory()
    print(f"Selected impulse file: {impulse_file}")

    signal, sample_rate = load_audio(audio_file)
    impulse_response, _ = load_audio(impulse_file)
    impulse_response = normalize_audio(impulse_response)

    impulse_response = impulse_response.flatten()

    print(f"Signal shape: {signal.shape}")
    print(f"Impulse response shape: {impulse_response.shape}")

    # Convolve the audio signal with the impulse response
    convolved_signal = convolve_signals(signal, impulse_response)

    # Clip the convolved signal
    # convolved_signal = clip_signal(convolved_signal)

    # Trim convolved signal to the length of the original signal
    convolved_signal = convolved_signal[:len(signal)]
    convolved_signal = normalize_audio(convolved_signal)

    # Save the output
    output_file = 'convolved_output.wav'
    save_audio(output_file, convolved_signal, sample_rate)

    # Plot the original and convolved_signal
    plot_signals(signal, convolved_signal)


if __name__ == "__main__":
    main()

# Create spectrograms and resize
def wav_to_spectrogram_scipy(wav_path, n_fft=2048, fmin=4000, fmax=20000):
    sample_rate, samples = wavfile.read(wav_path)

    if samples.ndim > 1:
        samples = samples[:, 0]  # use one channel

    if len(samples) < n_fft:
        print(f"Skipping {wav_path}: too short for FFT size {n_fft}")
        return None

    frequencies, times, Sxx = signal.spectrogram(samples, sample_rate, nperseg=n_fft)

    if len(frequencies) != Sxx.shape[0]:
        print(f"Shape mismatch in {wav_path}")
        return None

    freq_mask = (frequencies >= fmin) & (frequencies <= fmax)

    if not np.any(freq_mask):
        print(f"No frequencies in desired range for {wav_path}")
        return None

    Sxx = Sxx[freq_mask, :]
    Sxx_log = np.log(Sxx + 1e-10)
    Sxx_norm = (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())

    return Sxx_norm



spectrograms = []
labels = []

for wav_path, label in tqdm(wav_label_sub):
    spect = wav_to_spectrogram_scipy(wav_path, n_fft= 2048)
    spectrograms.append(spect)
    labels.append(label)



target_shape = (128, 128)
spectrograms_resized = []
valid_labels = []

for i, (s, label) in enumerate(zip(spectrograms, labels)):
    try:
        resized = resize(s, target_shape, mode='constant', anti_aliasing=True)

        # Fix extra singleton dimension if it appears
        if resized.ndim == 3 and resized.shape[-1] == 1:
            resized = np.squeeze(resized, axis=-1)

        # Final shape check
        if resized.shape != target_shape:
            print(f"Warning: Resized shape mismatch at index {i}, got {resized.shape}")
            continue

        spectrograms_resized.append(resized)
        valid_labels.append(label)  # keep label only if resize succeeded

    except Exception as e:
        print(f"Error resizing spectrogram at index {i}: {e}")


        
# Encode labels to integers
label_to_idx = {'Insular': 0, 'Pelagic': 1}
y = np.array([label_to_idx[l] for l in valid_labels])
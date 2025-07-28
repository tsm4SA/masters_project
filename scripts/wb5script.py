import os
import wave
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm  
from skimage.transform import resize
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import time
from PIL import Image
from tempfile import TemporaryDirectory

####
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Walk through all subfolders in 'insular' and 'pelagic' looking for .wav files, append to 'data' 
# if over 1 second
def get_wav_files_and_labels(base_dir, min_duration=1.0):
    data = []
    label_counts = {'Insular': 0, 'Pelagic': 0}

    for label in ['Insular', 'Pelagic']:
        label_path = os.path.join(base_dir, label)
        for root, dirs, files in os.walk(label_path):
            if os.path.basename(root) == "Raven wave file clips":
                for f in files:
                    if f.lower().endswith('.wav'):
                        full_path = os.path.join(root, f)
                        try:
                            with wave.open(full_path, 'rb') as wf:
                                frames = wf.getnframes()
                                rate = wf.getframerate()
                                duration = frames / float(rate)
                                if duration >= min_duration:
                                    data.append((full_path, label))
                                    label_counts[label] += 1
                        except wave.Error as e:
                            print(f"Could not process {full_path}: {e}")

    # print no. of spectrograms >1 second
    print(f"Found {label_counts['Insular']} Insular files")
    print(f"Found {label_counts['Pelagic']} Pelagic files")
    print(f"Total: {label_counts['Insular'] + label_counts['Pelagic']} files at least {min_duration} second(s) long")

    return data

# create wave and label pairs
base_dir = '../data'
wav_label_pairs = get_wav_files_and_labels(base_dir)

# Function for creating a subset of wav_label_pairs
# Note n here is number of samples in each not overall sample number
def create_balanced_subset(wav_label_pairs, n=100, seed=None):
    if seed is not None:
        random.seed(seed)  # Set random seed for reproducibility

    # Separate by label
    insular = [pair for pair in wav_label_pairs if pair[1] == 'Insular']
    pelagic = [pair for pair in wav_label_pairs if pair[1] == 'Pelagic']

    # Check that there are enough samples
    if len(insular) < n or len(pelagic) < n:
        raise ValueError(f"Not enough samples: Found {len(insular)} Insular and {len(pelagic)} Pelagic.")

    # Random sampling
    subset = random.sample(insular, n) + random.sample(pelagic, n)
    random.shuffle(subset)

    return subset

# Example usage
wav_label_sub = create_balanced_subset(wav_label_pairs, n=400, seed=42)
print(f"Subset contains {len(wav_label_sub)} samples.")

import librosa
import librosa.feature

def wav_to_spectrogram_scipy(wav_path, n_fft=2048, fmin=4000, fmax=20000):
    sample_rate, samples = wavfile.read(wav_path)
    if samples.ndim > 1:
        samples = samples[:, 0]
    if len(samples) < n_fft:
        return None

    frequencies, times, Sxx = signal.spectrogram(samples, sample_rate, nperseg=n_fft)
    if len(frequencies) != Sxx.shape[0]:
        return None
    freq_mask = (frequencies >= fmin) & (frequencies <= fmax)
    if not np.any(freq_mask):
        return None

    Sxx = Sxx[freq_mask, :]
    Sxx_log = np.log(Sxx + 1e-10)
    Sxx_norm = (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())

    # Compute deltas
    delta = librosa.feature.delta(Sxx_norm)
    delta_delta = librosa.feature.delta(Sxx_norm, order=2)

    # Stack along "channel" axis: shape becomes (3, freq, time)
    stacked = np.stack([Sxx_norm, delta, delta_delta], axis=0)
    return stacked




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
        if s is None:
            continue  # Skip failed spectrograms

        # Resize each channel independently: (3, freq, time) → 3 x (128,128)
        resized_channels = [resize(channel, target_shape, mode='constant', anti_aliasing=True) for channel in s]
        resized = np.stack(resized_channels, axis=0)  # shape: (3, 128, 128)

        # Check shape
        if resized.shape != (3, 128, 128):
            print(f"Warning: Resized shape mismatch at index {i}, got {resized.shape}")
            continue

        spectrograms_resized.append(resized)
        valid_labels.append(label)

    except Exception as e:
        print(f"Error resizing spectrogram at index {i}: {e}")



        
# Encode labels to integers
label_to_idx = {'Insular': 0, 'Pelagic': 1}
y = np.array([label_to_idx[l] for l in valid_labels])

# turn into numpy array and check shape
spectrograms_resized = np.array(spectrograms_resized)
print(f"Spectrogram dataset shape: {spectrograms_resized.shape}")
print(f"Labels shape: {y.shape}")

def plot_spectrogram_triplet(spectrogram_3ch, label=None):
    # spectrogram_3ch shape: (3, freq, time)
    titles = ['Original', 'Delta', 'Delta-Delta']

    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(spectrogram_3ch[i], aspect='auto', origin='lower', cmap='viridis')
        plt.title(f"{titles[i]} Spectrogram")
        plt.colorbar(format='%+2.0f dB')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.tight_layout()

    if label is not None:
        plt.suptitle(f"Label: {label}", fontsize=16)
    plt.show()

# Example usage with first spectrogram
plot_spectrogram_triplet(spectrograms_resized[0], valid_labels[0])

class SpectrogramDataset(Dataset):
    def __init__(self, spectrograms, labels, transform=None):
        self.spectrograms = spectrograms
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        image = self.spectrograms[idx].astype(np.float32)

        if idx == 0:  # Print just once
            print(f"Input shape at index {idx}: {image.shape}")

        image = torch.tensor(image, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

# Step 1: Split into train+val and test
X_temp, X_test, y_temp, y_test = train_test_split(
    spectrograms_resized,
    y,
    test_size=0.2,  # 20% test set
    random_state=42,
    stratify=y
)

# Step 2: Split train+val into train and val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,  # 25% of 80% = 20% → 60% train, 20% val, 20% test total
    random_state=42,
    stratify=y_temp
)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalize 
])

train_dataset = SpectrogramDataset(X_train, y_train, transform=transform)
val_dataset = SpectrogramDataset(X_val, y_val, transform=transform)
test_dataset = SpectrogramDataset(X_test, y_test, transform=transform)

g = torch.Generator()
g.manual_seed(42)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True, generator=g),
    'val': DataLoader(val_dataset, batch_size=32, shuffle=False),
    'test': DataLoader(test_dataset, batch_size=32, shuffle=False)
}

dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset),
    'test': len(test_dataset)
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        # To store metrics
        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Save metrics
                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                    train_acc_history.append(epoch_acc.item())
                else:
                    val_loss_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc.item())

                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')

        # Load best model
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))

    # Plotting
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Acc')
    plt.plot(val_acc_history, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model

model_ft = models.resnet18(weights='IMAGENET1K_V1')

# Freeze earlier layers if needed
for param in model_ft.parameters():
    param.requires_grad = False  # True to unfreeze all or False to freeze earlier layers

# Unfreeze layer4 and fc (last block and classifier head)
for name, param in model_ft.named_parameters():
   if name.startswith('layer4') or name.startswith('fc'):
       param.requires_grad = True

num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.

# below can be uncommented when not including dropout
# model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft.fc = nn.Sequential(
    nn.Dropout(p=0.8),           # 50% dropout
    nn.Linear(num_ftrs, 2)
)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized (SGD)
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Observe that all parameters are being optimized below (ADAMW)
# optimizer_ft = optim.AdamW(model_ft.parameters(), lr=0.001, weight_decay=1e-4)

# Here we update only the unfrozen parameters (ADAMW)
optimizer_ft = optim.AdamW(
    filter(lambda p: p.requires_grad, model_ft.parameters()),  # only update unfrozen params
    lr=0.001,
    weight_decay=1e-4
)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=20)

class_names = ['Insular', 'Pelagic']

def evaluate_model(model, dataloader, class_names):
    model.eval()
    running_corrects = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = running_corrects.double() / total
    print(f"\nTest Accuracy: {accuracy:.4f}")

    return all_preds, all_labels

preds, labels = evaluate_model(model_ft, dataloaders['test'], class_names)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def show_confusion_matrix(model, dataloader, class_names):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


show_confusion_matrix(model_ft, dataloaders['test'], class_names)

def print_predictions_vs_actuals(model, dataloader, class_names, max_print=100):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            predictions.extend(preds.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    print(f"\n{'Index':<5} {'Predicted':<15} {'Actual':<15}")
    print("-" * 40)
    for i in range(min(max_print, len(predictions))):
        pred_label = class_names[predictions[i]]
        true_label = class_names[actuals[i]]
        print(f"{i:<5} {pred_label:<15} {true_label:<15}")


print_predictions_vs_actuals(model_ft, dataloaders['test'], class_names)

def show_selected_spectrograms_with_deltas(model, dataset, indices, class_names):
    model.eval()
    n_images = len(indices)

    fig, axes = plt.subplots(n_images, 3, figsize=(12, 4 * n_images))

    if n_images == 1:
        axes = np.expand_dims(axes, axis=0)  # Ensure 2D array even with one image

    channel_titles = ['Original', 'Delta', 'Delta-Delta']

    with torch.no_grad():
        for i, idx in enumerate(indices):
            spectrogram, actual_label = dataset[idx]
            input_tensor = spectrogram.unsqueeze(0).to(device)

            output = model(input_tensor)
            _, predicted_label = torch.max(output, 1)

            spectrogram_np = spectrogram.numpy()  # (3, 128, 128)

            for ch in range(3):
                ax = axes[i][ch]
                ax.imshow(spectrogram_np[ch], cmap='magma', origin='lower', aspect='auto')
                if ch == 1:
                    ax.set_title(f"Pred: {class_names[predicted_label.item()]}, Actual: {class_names[actual_label]}\n{channel_titles[ch]}")
                else:
                    ax.set_title(channel_titles[ch])
                ax.axis('off')

    plt.tight_layout()
    plt.show()


selected_indices = [0,9,12,18,10,13,4,14,19]  # choose whatever you want
show_selected_spectrograms_with_deltas(model_ft, test_dataset, selected_indices, class_names)
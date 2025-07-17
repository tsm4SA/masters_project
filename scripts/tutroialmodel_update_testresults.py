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


# Walk through all subfolders in 'insular' and 'pelagic' looking for .wav files, append to 'data' 
# if over 1 second
def get_wav_files_and_labels(base_dir, min_duration=1.0):
    data = []
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
                        except wave.Error as e:
                            print(f"Could not process {full_path}: {e}")
    return data

# Return no. of spectrograms >1 second
base_dir = '../data'
wav_label_pairs = get_wav_files_and_labels(base_dir)
print(f"Found {len(wav_label_pairs)} wav files at least 1 second long.")

# Function for creating a subset of wav_label_pairs

def create_balanced_subset(wav_label_pairs, n=300, seed=None):
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
wav_label_sub = create_balanced_subset(wav_label_pairs, n=300, seed=42)
print(f"Subset contains {len(wav_label_sub)} samples.")

# Create spectrograms and resize
def wav_to_spectrogram_scipy(wav_path, n_fft=8192):
    sample_rate, samples = wavfile.read(wav_path)
    frequencies, times, Sxx = signal.spectrogram(samples, sample_rate, nperseg=n_fft)
    # Sxx is power spectral density, take log to get dB scale
    Sxx_log = np.log(Sxx + 1e-10)  # add small number to avoid log(0)
    # Normalize to 0-1 range
    Sxx_norm = (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())
    return Sxx_norm


spectrograms = []
labels = []

for wav_path, label in tqdm(wav_label_sub):
    spect = wav_to_spectrogram_scipy(wav_path, n_fft=8192)
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

# turn into numpy array and check shape
spectrograms_resized = np.array(spectrograms_resized)
print(f"Spectrogram dataset shape: {spectrograms_resized.shape}")
print(f"Labels shape: {y.shape}")

# To visualize one spectrogram
plt.imshow(spectrograms_resized[0], aspect='auto', origin='lower')
plt.title(f"Label: {valid_labels[0]}")
plt.colorbar()
plt.show()


class SpectrogramDataset(Dataset):
    def __init__(self, spectrograms, labels, transform=None):
        self.spectrograms = spectrograms
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        image = self.spectrograms[idx].astype(np.float32)

        # Convert 1-channel to 3-channel by duplicating
        image = np.stack([image]*3, axis=0)  # shape: (3, H, W)

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

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=32),
    'test': DataLoader(test_dataset, batch_size=32)
}

dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset),
    'test': len(test_dataset)
}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))  # CHW -> HWC
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean  # unnormalize
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {y_train[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)

visualize_model(model_ft)

class_names = ['Insular', 'Pelagic']

def print_predictions_vs_actuals(model, dataloader, class_names, max_print=30):
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

print_predictions_vs_actuals(model_ft, dataloaders['val'], class_names)

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

show_confusion_matrix(model_ft, dataloaders['val'], class_names)

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

print_predictions_vs_actuals(model_ft, dataloaders['test'], class_names)

show_confusion_matrix(model_ft, dataloaders['test'], class_names)


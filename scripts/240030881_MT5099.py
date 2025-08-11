# Description: Script for MT5099 machine learning project
# Author: Tristan McKenzie
# Date: 11/08/25

# import all used packages
import os
import wave
import numpy as np
import matplotlib.pyplot as plt
import random
import librosa
import librosa.feature
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm  
from skimage.transform import resize
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models import mobilenet_v2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import time
from PIL import Image
from tempfile import TemporaryDirectory

# Function for setting various seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Setting the initial seed
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

    # print no. of spectrograms >1 second prints number in each class and total number
    print(f"Found {label_counts['Insular']} Insular files")
    print(f"Found {label_counts['Pelagic']} Pelagic files")
    print(f"Total: {label_counts['Insular'] + label_counts['Pelagic']} files at least {min_duration} second(s) long")
    
    # Return 'data' object after call
    return data

# create .wav and label pairs
base_dir = '../data'
wav_label_pairs = get_wav_files_and_labels(base_dir)

# Function for creating a subset of wav_label_pairs
# Note 'n' here is number of samples in each class not overall sample number
def create_balanced_subset(wav_label_pairs, n=100, seed=None):
    if seed is not None:
        random.seed(seed)  # Set random seed for reproducibility

    # Separate by label
    insular = [pair for pair in wav_label_pairs if pair[1] == 'Insular']
    pelagic = [pair for pair in wav_label_pairs if pair[1] == 'Pelagic']

    # Check that there are enough samples specified by 'n' and print error if not
    if len(insular) < n or len(pelagic) < n:
        raise ValueError(f"Not enough samples: Found {len(insular)} Insular and {len(pelagic)} Pelagic.")

    # Random sampling and shuffle the samples
    subset = random.sample(insular, n) + random.sample(pelagic, n)
    random.shuffle(subset)

    return subset

# Calling the function to create subset
wav_label_sub = create_balanced_subset(wav_label_pairs, n=400, seed=42)
print(f"Subset contains {len(wav_label_sub)} samples.")

# Function to create spectrograms, fmin and fmax can be used to create a frequency filter

def wav_to_spectrogram_scipy(wav_path, n_fft=2048, fmin=2500, fmax=20000):
    sample_rate, samples = wavfile.read(wav_path)
    if samples.ndim > 1:
        samples = samples[:, 0]
    if len(samples) < n_fft:
        return None

    # Generate spectrograms
    frequencies, times, Sxx = signal.spectrogram(samples, sample_rate, nperseg=n_fft)
    if len(frequencies) != Sxx.shape[0]:
        return None
    
    # Implement frequency filter "mask"
    freq_mask = (frequencies >= fmin) & (frequencies <= fmax)
    if not np.any(freq_mask):
        return None

    # Implement the mask, log transform, and normalise
    Sxx = Sxx[freq_mask, :]
    Sxx_log = np.log(Sxx + 1e-10) # note a small value is added here to avoid log(0)
    Sxx_norm = (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())

    # Compute deltas
    delta = librosa.feature.delta(Sxx_norm)
    delta_delta = librosa.feature.delta(Sxx_norm, order=2)

    # Stack along "channel" axis: shape becomes (3, freq, time)
    stacked = np.stack([Sxx_norm, delta, delta_delta], axis=0)
    return stacked



# Create empty objects to store spectrograms and labels
spectrograms = []
labels = []

# Calling the spectrogram creation function and appending to spectrograms and labels objects
for wav_path, label in tqdm(wav_label_sub):
    spect = wav_to_spectrogram_scipy(wav_path, n_fft= 2048)
    spectrograms.append(spect)
    labels.append(label)


# Setting the target shape for resizing and creating new empty objects to append successful resized
    # spectrograms and respective labels 
target_shape = (128, 128)
spectrograms_resized = []
valid_labels = []

for i, (s, label) in enumerate(zip(spectrograms, labels)):
    try:
        if s is None:
            continue  # Skip failed spectrograms

        # Resize each channel independently: (3, freq, time) 3 x (128,128)
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

# function to plot spectrograms with deltas
def plot_spectrogram_triplet(spectrogram_3ch, label=None):
    # spectrogram_3ch shape: (3, freq, time)
    titles = ['Original', 'Delta', 'Delta-Delta']

    # Set plotting parameters
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

# Call the function to plot a triplet here
plot_spectrogram_triplet(spectrograms_resized[0], valid_labels[0])

# Create custom dataset for use in PyTorch
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

# Create the train/val/test split, currently set for a 60/20/20 split
# Split into train+val and test
X_temp, X_test, y_temp, y_test = train_test_split(
    spectrograms_resized,
    y,
    test_size=0.2,  # 20% test set
    random_state=42,
    stratify=y
)

# Split train+val into train and val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,  # 25% of 80% = 20% 
    random_state=42,
    stratify=y_temp
)

# Calling transform, dataset, and dataloader

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

# Device setting allows for use of gpu if avaialable and routes to cpu if unavailable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Function for calling the training loop
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create temporary directory for saving the 'best' model settings
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        # To store metrics
        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []

        # Loop through each epoch
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

                    # Here the model with a validation accuracy higher than the current best is saved
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')

        # Load best model
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))

    # Plotting the training and validation loss plots and training and validation accuracies
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

# Here is where model selection and configurations can be set
# resets the seed for reproducability when editing model choice
set_seed(24)

# Base model selections ----------------------------------------------------------------------

# # ResNet-18
# model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# # ResNet-50
# model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
# # AlexNet
# model_ft = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
# # SqueezeNet 1.1
# model_ft = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1) 
# MobileNet_V2
model_ft = mobilenet_v2(weights='IMAGENET1K_V1')
# # GoogLeNet
# model_ft = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)

######### Layer manipulations for base models -----------------------------------------------
##### RESNET LAYER MANIPULATIONS ------------------------------
# # Freeze earlier layers if needed
# for param in model_ft.parameters():
#     param.requires_grad = False  # True to unfreeze all or False to freeze earlier layers

# # Unfreeze layer4 and fc (fully connected layer (classifier))
# for name, param in model_ft.named_parameters():
#    if name.startswith('layer4') or name.startswith('fc'):
#        param.requires_grad = True

# # setting 'num_features' to the number of inputs 
# num_ftrs = model_ft.fc.in_features


# # below can be uncommented when not including dropout takes expected 
# # input set by num_ftrs and transforms into 2 outputs
# model_ft.fc = nn.Linear(num_ftrs, 2)

# # below can be uncommented when including dropout
# model_ft.fc = nn.Sequential(
#     nn.Dropout(p=0.8),           # set droput rate here (e.g 0.5 = 50% dropout)
#     nn.Linear(num_ftrs, 2)
# )

####### ALEXNET LAYER MANIPULATIONS --------------------------------------
# # Freeze all layers if needed
# for param in model_ft.parameters():
#     param.requires_grad = False # change to false to freeze, true to unfreeze

# # Unfreeze classifier only
# for param in model_ft.classifier.parameters():
#     param.requires_grad = True

# # Replace final classification layer (last Linear in classifier)
# num_ftrs = model_ft.classifier[6].in_features

# # # For no dropout 
# # model_ft.classifier[6] = nn.Linear(num_ftrs, 2)  # for 2 output classes

# # For dropout
# model_ft.classifier[6] = nn.Sequential(
#     nn.Dropout(0.2),
#     nn.Linear(num_ftrs, 2)
# )

# ##### SQUEEZENET LAYER MANIPULATIONS ----------------------------------
# # Freeze all layers
# for param in model_ft.parameters():
#     param.requires_grad = False

# # Unfreeze classifier only
# for param in model_ft.classifier.parameters():
#     param.requires_grad = True

# # Modify final classifier
# model_ft.classifier = nn.Sequential(
#     nn.Dropout(p=0.2),
#     nn.Conv2d(512, 2, kernel_size=1),
#     nn.ReLU(inplace=True),
#     nn.AdaptiveAvgPool2d((1, 1))
# )

##### MOBILENET_V2 LAYER MANIPULATIONS -----------------------------------
# Freeze all layers
for param in model_ft.parameters():
    param.requires_grad = False

# Unfreeze classifier only
for param in model_ft.classifier.parameters():
    param.requires_grad = True

# Modify final classifier
model_ft.classifier[1] = nn.Linear(model_ft.classifier[1].in_features, 2)

# # Replace the classifier with dropout + custom linear layer
# model_ft.classifier = nn.Sequential(
#     nn.Dropout(p=0.5),  # You can adjust this rate
#     nn.Linear(model_ft.classifier[1].in_features, 2)
# )

##### GOOGLENET LAYER MANIPULATIONS -----------------------------
# # Disable auxiliary classifiers
# model_ft.aux_logits = False

# # Freeze all layers
# for param in model_ft.parameters():
#     param.requires_grad = False

# # Unfreeze the final fully connected layer
# for param in model_ft.fc.parameters():
#     param.requires_grad = True

# # Modify the final classifier with dropout + linear layer
# in_features = model_ft.fc.in_features
# model_ft.fc = nn.Sequential(
#     nn.Dropout(p=0.2),  # Adjust dropout rate as needed
#     nn.Linear(in_features, 2)  # Two output classes
# )

########## Global settings for all models ---------------------------------------

## send model to device
model_ft = model_ft.to(device)

### setting the loss 
criterion = nn.CrossEntropyLoss()

####### Optimiser settings ---------------------------------------------------
## Observe that all parameters below are being optimized (SGD)
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

## Here we update only the unfrozen parameters (SGD)
# optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()),  # only update unfrozen params
#     lr=0.001,
#     weight_decay=1e-4
# )

## Observe that all parameters are being optimized below (ADAMW)
# optimizer_ft = optim.AdamW(model_ft.parameters(), lr=0.001, weight_decay=1e-4)

# Here we update only the unfrozen parameters (ADAMW)
optimizer_ft = optim.AdamW(
    filter(lambda p: p.requires_grad, model_ft.parameters()),  # only update unfrozen params
    lr=0.001,
    weight_decay=1e-4
)

######## Setting the learning rate decay (All models) ----------------------------------
# Decay LR by a factor of 0.1 every 5 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

# Calling the training loop

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=20)

# Setting a class names object for the next evaluation step
class_names = ['Insular', 'Pelagic']

# Function for evaluating a model on a test set
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

# Calling the function for testing
preds, labels = evaluate_model(model_ft, dataloaders['test'], class_names)

# function for displaying confusion matrices
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


# Calling the function to display confusion matrices on chosen data set
show_confusion_matrix(model_ft, dataloaders['test'], class_names)

# This function can be used to create a list that shows predictions vs. actuals
def print_predictions_vs_actuals(model, dataloader, class_names, max_print=200):
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

    # Print the table
    print(f"\n{'Index':<5} {'Predicted':<15} {'Actual':<15}")
    print("-" * 40)
    for i in range(min(max_print, len(predictions))):
        pred_label = class_names[predictions[i]]
        true_label = class_names[actuals[i]]
        print(f"{i:<5} {pred_label:<15} {true_label:<15}")


# Calling the function to print predicted vs. actuals 
print_predictions_vs_actuals(model_ft, dataloaders['test'], class_names)

# This function is used to display the triplets
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


# indices can be selected here to show selected triplets
selected_indices = [17,19,23,26,37]  

# Calling the function to display triplets at selected indices
show_selected_spectrograms_with_deltas(model_ft, test_dataset, selected_indices, class_names)

# Resetting the seed for the ensemble
set_seed(42)

# MobileNet_V2
model_mv2 = mobilenet_v2(weights='IMAGENET1K_V1')

# Freeze all layers
for param in model_mv2.parameters():
    param.requires_grad = False

# Unfreeze classifier only
for param in model_mv2.classifier.parameters():
    param.requires_grad = True

# Modify final classifier
model_mv2.classifier[1] = nn.Linear(model_mv2.classifier[1].in_features, 2)

# send model to device
model_mv2 = model_mv2.to(device)

# setting the loss 
criterion = nn.CrossEntropyLoss()

# setting optimiser for mobile net
optimizer_mv2 = optim.AdamW(
    filter(lambda p: p.requires_grad, model_mv2.parameters()),  # only update unfrozen params
    lr=0.001,
    weight_decay=1e-4
)

# Decay LR by a factor of 0.1 every 5 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_mv2, step_size=5, gamma=0.1)

# training the mobile net model for the ensemble
model_mv2 = train_model(model_mv2, criterion, optimizer_mv2, exp_lr_scheduler,
                       num_epochs=20)

# ResNet-18
model_res18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze earlier layers if needed
for param in model_res18.parameters():
    param.requires_grad = False  # True to unfreeze all or False to freeze earlier layers

# Unfreeze layer4 and fc (last block and classifier head)
for name, param in model_res18.named_parameters():
   if name.startswith('layer4') or name.startswith('fc'):
       param.requires_grad = True

num_ftrs = model_res18.fc.in_features

model_res18.fc = nn.Sequential(
    nn.Dropout(p=0.8),           # set droput rate here (e.g 0.5 = 50% dropout)
    nn.Linear(num_ftrs, 2)
)

## send model to device
model_res18 = model_res18.to(device)

### setting the loss 
criterion = nn.CrossEntropyLoss()

# Here we update only the unfrozen parameters (ADAMW)
optimizer_res18 = optim.AdamW(
    filter(lambda p: p.requires_grad, model_res18.parameters()),  # only update unfrozen params
    lr=0.001,
    weight_decay=1e-4
)

# Decay LR by a factor of 0.1 every 5 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_res18, step_size=5, gamma=0.1)

# train the resnet 18 for the ensemble
model_res18 = train_model(model_res18, criterion, optimizer_res18, exp_lr_scheduler,
                       num_epochs=20)

# Send models to the available device 
model_mv2.to(device)
model_res18.to(device)

# Sets models to evaluation mode
model_mv2.eval()
model_res18.eval()

# Function for evaulating the ensemble on a test set note that 
# voting can only be currenty set to hard or soft in this function
def evaluate_ensemble(esmodels, dataloader, class_names, voting='soft'):
    for model in esmodels:
        model.eval()

    running_corrects = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            if voting == 'soft':
                probs_list = []
                for model in esmodels:
                    outputs = model(inputs)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    probs_list.append(probs)

                avg_probs = torch.stack(probs_list).mean(dim=0)
                _, preds = torch.max(avg_probs, 1)

            elif voting == 'hard':
                votes = []
                for model in esmodels:
                    outputs = model(inputs)
                    _, preds_model = torch.max(outputs, 1)
                    votes.append(preds_model)

                votes = torch.stack(votes)
                preds, _ = torch.mode(votes, dim=0)

            else:
                raise ValueError("voting must be 'soft' or 'hard'")

            running_corrects += torch.sum(preds == labels)
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = running_corrects.double() / total
    print(f"\nEnsemble Test Accuracy ({voting} voting): {accuracy:.4f}")

    return all_preds, all_labels

# set the models contributing to the ensemble
esmodels = [model_mv2, model_res18]

# Call the function to evaluate the ensemble
preds, labels = evaluate_ensemble(esmodels, dataloaders['test'], class_names, voting='soft')


# Function for creating a confusion matrix specifically for the ensemble
# note that it currently only supports soft voting so if hard voting is specified earlier
# it can not currently be displayed by a matrix
def show_confusion_matrix_ensemble(model1, model2, dataloader, class_names):
    model1.eval()
    model2.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs1 = model1(inputs)
            outputs2 = model2(inputs)

            # Apply softmax to get probabilities
            probs1 = F.softmax(outputs1, dim=1)
            probs2 = F.softmax(outputs2, dim=1)

            # Average the probabilities (soft voting)
            avg_probs = (probs1 + probs2) / 2

            # Final prediction is the argmax of averaged probabilities
            _, combined_preds = torch.max(avg_probs, 1)

            all_preds.extend(combined_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    disp.plot(cmap="Blues", values_format='d')
    plt.title("Ensemble Confusion Matrix (Soft Voting)")
    plt.show()

    return cm


# Calling the function to display ensemble confusion matrix
cm = show_confusion_matrix_ensemble(model_mv2, model_res18, dataloaders['test'], class_names)



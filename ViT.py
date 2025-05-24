import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
import pandas as pd
import matplotlib.cm as cm
import random
import seaborn as sns

# Configuration
config = {
    'data_dir': './CASIA_B123',
    'patch_size': 16,
    'emb_size': 128,
    'image_size': 144,
    'batch_size': 32,
    'num_workers': 0,
    'num_classes': 90,
    'epochs': 200,
    'learning_rate': 1e-4,
    'model_save_path': './vit_casia_b.pth',
    'in_channels': 1,
    'n_clusters': 34,  # Number of clusters for test set (34 classes from 091-124)
}

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Check library versions
import torch, sklearn, matplotlib

print(f"PyTorch: {torch.__version__}, Scikit-learn: {sklearn.__version__}, Matplotlib: {matplotlib.__version__}")


# Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, img_size, patch_size, emb_size):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(emb_size)
        self.linear = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.linear(x)
        return x


# Class Token
class ClassToken(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        return x


# Position Embedding
class PositionEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim), requires_grad=True)

    def forward(self, x):
        x = x + self.pos_embed
        return x


# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.msa = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.ln1(x)
        attn_output, _ = self.msa(x_norm, x_norm, x_norm)
        x = x + attn_output
        x_norm = self.ln2(x)
        x = x + self.mlp(x_norm)
        return x


# Classifier
class Classifier(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# ViT Model
class ViT(nn.Module):
    def __init__(self, in_channels, img_size, patch_size, embed_dim, num_encoders, num_heads, hidden_dim, dropout,
                 num_classes):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = PatchEmbedding(in_channels, img_size, patch_size, embed_dim)
        self.cls_token = ClassToken(embed_dim)
        self.pos_embed = PositionEmbedding(num_patches, embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerEncoder(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_encoders)])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = Classifier(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.cls_token(x)
        x = self.pos_embed(x)
        x = self.transformer(x)
        x = x[:, 0, :].squeeze(1)
        x = self.norm(x)
        x = self.classifier(x)
        return x


# ViT Extractor for feature extraction
class ViT_Extractor(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.patch_embed = vit_model.patch_embed
        self.cls_token = vit_model.cls_token
        self.pos_embed = vit_model.pos_embed
        self.transformer = vit_model.transformer
        self.norm = vit_model.norm

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.cls_token(x)
        x = self.pos_embed(x)
        x = self.transformer(x)
        x = x[:, 0, :].squeeze(1)
        x = self.norm(x)
        return x


# Dataset class
class CASIABDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Dataset directory {self.data_dir} does not exist.")
        samples = []
        all_subjects = sorted(os.listdir(self.data_dir))
        subject_to_idx = {sid: idx for idx, sid in enumerate(all_subjects)}  # 0 to 124
        for subject_id in all_subjects:
            subject_path = os.path.join(self.data_dir, subject_id)
            if os.path.isdir(subject_path):
                for condition in os.listdir(subject_path):
                    condition_path = os.path.join(subject_path, condition)
                    if os.path.isdir(condition_path):
                        for view in os.listdir(condition_path):
                            view_path = os.path.join(condition_path, view)
                            if os.path.isdir(view_path):
                                for image_file in os.listdir(view_path):
                                    if image_file.endswith('.png'):
                                        image_path = os.path.join(view_path, image_file)
                                        label = subject_to_idx[subject_id]
                                        # Adjust label for test set (091-124) to fit within 0-33
                                        if int(subject_id) >= 91:
                                            label = label - 90  # Map 91->0, 92->1, ..., 124->33
                                        samples.append({
                                            'image_path': image_path,
                                            'subject_id': subject_id,
                                            'condition': condition,
                                            'view': view,
                                            'label': label
                                        })
        if not samples:
            raise ValueError(f"No valid images found in {self.data_dir}")
        print(f"Loaded {len(samples)} samples")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image = Image.open(sample['image_path']).convert('L')
            image = np.array(image) / 255.0
            image = image.astype(np.float32)
            image = image[..., np.newaxis]
            label = sample['label']
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            image = np.zeros((config['image_size'], config['image_size'], 1), dtype=np.float32)
            label = -1

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'subject_id': sample['subject_id'],
            'label': torch.tensor(label, dtype=torch.long),
            'condition': sample['condition'],
            'view': sample['view']
        }


# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((config['image_size'], config['image_size'])),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


# Training function with angle accuracy tracking
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    train_correct_angles = {}  # {angle: count}
    val_correct_angles = {}  # {angle: count}

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            views = batch['view']
            if (labels == -1).any():
                continue
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Track correct predictions by angle
            correct_mask = (predicted == labels)
            for i in range(len(views)):
                if correct_mask[i]:
                    angle = views[i]
                    train_correct_angles[angle] = train_correct_angles.get(angle, 0) + 1

        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                views = batch['view']
                if (labels == -1).any():
                    continue
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Track correct predictions by angle
                correct_mask = (predicted == labels)
                for i in range(len(views)):
                    if correct_mask[i]:
                        angle = views[i]
                        val_correct_angles[angle] = val_correct_angles.get(angle, 0) + 1

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config['model_save_path'])
            print(f"  Saved best model with Val Acc: {val_acc:.2f}%")

    return history, train_correct_angles, val_correct_angles


# Extract features
def extract_features(model, data_loader, device):
    model.eval()
    features = []
    labels = []
    subject_ids = []

    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            lbls = batch['label'].to(device)
            if (lbls == -1).any():
                continue
            feat = model(images)
            features.append(feat.cpu().numpy())
            labels.append(lbls.cpu().numpy())
            subject_ids.extend(batch['subject_id'])

    features = np.vstack(features)
    labels = np.hstack(labels)
    return features, labels, subject_ids


# Cluster and map labels for test set
def cluster_and_map_labels(features, n_clusters):
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_pca)

    mapped_labels = cluster_labels
    print(f"Silhouette Score: {silhouette_score(features_pca, cluster_labels):.3f}")

    return features_pca, mapped_labels, kmeans.cluster_centers_


# Visualize clusters for test set with subject_id colors, random 10 subjects
def visualize_clusters(features_pca, cluster_labels, subject_ids, output_file="test_clusters.png"):
    # Convert subject_ids to numpy array with string type for consistency
    subject_ids = np.array(subject_ids, dtype=str)

    # Get unique subject IDs and select 10 random subjects
    unique_subjects = np.unique(subject_ids)
    if len(unique_subjects) > 10:
        random.seed(42)  # For reproducibility
        selected_subjects = np.array(random.sample(list(unique_subjects), 10), dtype=str)
    else:
        selected_subjects = unique_subjects

    # Filter data for the selected subjects
    indices = np.isin(subject_ids, selected_subjects)
    features_pca_filtered = features_pca[indices]
    cluster_labels_filtered = cluster_labels[indices]
    subject_ids_filtered = subject_ids[indices]

    # Assign a color to each selected subject
    num_subjects = len(selected_subjects)
    colors = cm.rainbow(np.linspace(0, 1, num_subjects))
    subject_to_color = {subj: colors[i] for i, subj in enumerate(selected_subjects)}

    # Plot
    plt.figure(figsize=(10, 8))
    for subj in selected_subjects:
        indices = np.where(subject_ids_filtered == subj)[0]
        if len(indices) > 0:  # Check if there are any indices to avoid empty scatter
            plt.scatter(
                features_pca_filtered[indices, 0],
                features_pca_filtered[indices, 1],
                c=[subject_to_color[subj]] * len(indices),
                label=subj,
                alpha=0.6
            )

    plt.title('Test Set Clusters (PCA Reduced) with 10 Random Subjects')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    for i, txt in enumerate(subject_ids_filtered):
        plt.annotate(txt, (features_pca_filtered[i, 0], features_pca_filtered[i, 1]), fontsize=6)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


# Visualize PCA for train and validation sets with random 10 subjects
def visualize_pca_train_val(model, train_loader, val_loader, device, output_file_train="train_pca.png",
                            output_file_val="val_pca.png"):
    extractor = ViT_Extractor(model).to(device)

    for loader, dataset_name, output_file in [(train_loader, "Train", output_file_train),
                                              (val_loader, "Validation", output_file_val)]:
        features, labels, subject_ids = extract_features(extractor, loader, device)

        # Convert subject_ids to numpy array with string type for consistency
        subject_ids = np.array(subject_ids, dtype=str)

        # Get unique subject IDs and select 10 random subjects
        unique_subjects = np.unique(subject_ids)
        if len(unique_subjects) > 10:
            random.seed(42)  # For reproducibility
            selected_subjects = np.array(random.sample(list(unique_subjects), 10), dtype=str)
        else:
            selected_subjects = unique_subjects

        # Filter data for the selected subjects
        indices = np.isin(subject_ids, selected_subjects)
        features_filtered = features[indices]
        subject_ids_filtered = subject_ids[indices]

        # Apply PCA
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_filtered)
        print(f"{dataset_name} PCA Explained variance ratio: {pca.explained_variance_ratio_}")

        # Assign a color to each selected subject
        num_subjects = len(selected_subjects)
        colors = cm.rainbow(np.linspace(0, 1, num_subjects))
        subject_to_color = {subj: colors[i] for i, subj in enumerate(selected_subjects)}

        # Plot
        plt.figure(figsize=(10, 8))
        for subj in selected_subjects:
            indices = np.where(subject_ids_filtered == subj)[0]
            if len(indices) > 0:  # Check if there are any indices to avoid empty scatter
                plt.scatter(
                    features_pca[indices, 0],
                    features_pca[indices, 1],
                    c=[subject_to_color[subj]] * len(indices),
                    label=subj,
                    alpha=0.6
                )

        plt.title(f'{dataset_name} Set PCA of Final Features with 10 Random Subjects')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        for i, txt in enumerate(subject_ids_filtered):
            plt.annotate(txt, (features_pca[i, 0], features_pca[i, 1]), fontsize=6)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()


# Plot confusion matrix for test and validation sets
def plot_confusion_matrix(model, test_loader, val_loader, device, output_file_test="confusion_matrix_test.png",
                          output_file_val="confusion_matrix_val.png"):
    model.eval()
    for loader, dataset_name, output_file, num_classes in [(test_loader, "Test", output_file_test, 34),
                                                           (val_loader, "Validation", output_file_val, 90)]:
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                if (labels == -1).any():
                    continue
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {dataset_name} Set')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"{dataset_name} Set Confusion Matrix saved to {output_file}")


# Plot training history
def plot_training_history(history, epochs, output_file="training_history.png"):
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc')
    plt.plot(epochs_range, history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


# Plot correct angles for train and validation
def plot_correct_angles(train_correct_angles, val_correct_angles, output_file="correct_angles.png"):
    # Get all unique angles
    all_angles = sorted(set(list(train_correct_angles.keys()) + list(val_correct_angles.keys())))

    train_counts = [train_correct_angles.get(angle, 0) for angle in all_angles]
    val_counts = [val_correct_angles.get(angle, 0) for angle in all_angles]

    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(all_angles))

    plt.bar(index, train_counts, bar_width, label='Train', color='skyblue')
    plt.bar(index + bar_width, val_counts, bar_width, label='Validation', color='lightcoral')

    plt.xlabel('Angle')
    plt.ylabel('Number of Correct Predictions')
    plt.title('Number of Correct Predictions by Angle')
    plt.xticks(index + bar_width / 2, all_angles, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


# Main function
def main():
    # Dataset and DataLoader
    dataset = CASIABDataset(config['data_dir'], transform=transform)

    # Split dataset based on classes and conditions
    train_conditions = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'cl-01', 'bg-01']
    val_conditions = ['nm-06', 'bg-02', 'cl-02']
    train_val_subjects = [str(i).zfill(3) for i in range(1, 91)]
    test_subjects = [str(i).zfill(3) for i in range(91, 125)]

    train_indices = []
    val_indices = []
    test_indices = []

    for idx, sample in enumerate(dataset.samples):
        subject_id = sample['subject_id']
        condition = sample['condition']
        if subject_id in train_val_subjects:
            if condition in train_conditions:
                train_indices.append(idx)
            elif condition in val_conditions:
                val_indices.append(idx)
        elif subject_id in test_subjects:
            test_indices.append(idx)

    if not train_indices:
        raise ValueError("No samples found for training set")
    if not val_indices:
        raise ValueError("No samples found for validation set")
    if not test_indices:
        raise ValueError("No samples found for test set")

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False,
                             num_workers=config['num_workers'])

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    # ViT model
    model = ViT(
        in_channels=config['in_channels'],
        img_size=config['image_size'],
        patch_size=config['patch_size'],
        embed_dim=config['emb_size'],
        num_encoders=8,
        num_heads=4,
        hidden_dim=config['emb_size'] * 4,
        dropout=0.3,
        num_classes=config['num_classes']
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Train and validate
    history, train_correct_angles, val_correct_angles = train_model(model, train_loader, val_loader, criterion,
                                                                    optimizer, config['epochs'], device)

    # Plot training history
    plot_training_history(history, config['epochs'])

    # Plot correct angles for train and validation
    print("\nPlotting correct angles for train and validation...")
    plot_correct_angles(train_correct_angles, val_correct_angles)

    # Load best model for evaluation
    print("\nLoading best model for evaluation...")
    if not os.path.exists(config['model_save_path']):
        raise FileNotFoundError(f"Model file {config['model_save_path']} not found")
    model.load_state_dict(torch.load(config['model_save_path'], map_location=device))

    # Visualize PCA for train and validation sets
    print("\nGenerating PCA plots for train and validation sets...")
    visualize_pca_train_val(model, train_loader, val_loader, device)

    # Plot confusion matrices for test and validation sets
    print("\nGenerating confusion matrices for test and validation sets...")
    plot_confusion_matrix(model, test_loader, val_loader, device)

    # Extract features from test set
    extractor = ViT_Extractor(model).to(device)
    test_features, test_labels, test_subject_ids = extract_features(extractor, test_loader, device)

    # Cluster and map labels for test set
    features_pca, cluster_labels, cluster_centers = cluster_and_map_labels(test_features, config['n_clusters'])

    # Visualize clusters for test set with subject_id colors
    visualize_clusters(features_pca, cluster_labels, test_subject_ids)


if __name__ == "__main__":
    main()
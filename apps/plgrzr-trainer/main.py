import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import logging
from typing import Dict
import wandb  # for experiment tracking


class HandwritingDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, split="train"):
        """
        Dataset for handwriting analysis.

        Args:
            data_dir: Directory containing the dataset
            transform: Optional transforms to apply
            split: 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.split = split

        # Expected directory structure:
        # data_dir/
        #   - author1/
        #     - sample1.png
        #     - sample2.png
        #   - author2/
        #     - sample1.png
        #     ...

        self.samples = []
        self.labels = []
        self.authors = {}

        # Load data
        for author_idx, author in enumerate(os.listdir(data_dir)):
            author_dir = os.path.join(data_dir, author)
            if os.path.isdir(author_dir):
                self.authors[author_idx] = author
                for sample in os.listdir(author_dir):
                    if sample.endswith((".png", ".jpg")):
                        self.samples.append(os.path.join(author_dir, sample))
                        self.labels.append(author_idx)

        # Split data
        indices = list(range(len(self.samples)))
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

        # Select appropriate indices based on split
        if split == "train":
            self.indices = train_idx
        elif split == "val":
            self.indices = val_idx
        else:
            self.indices = test_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        true_idx = self.indices[idx]
        image_path = self.samples[true_idx]
        label = self.labels[true_idx]

        # Load and preprocess image
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        if self.transform:
            image = self.transform(image)

        return image, label


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, 10),  # input is 1 channel (grayscale)
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
        )

    def forward_one(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + label
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive


class ModelTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize wandb
        wandb.init(project="handwriting-analysis", config=config)

        # Setup data
        self.setup_data()

        # Initialize model
        self.model = SiameseNetwork().to(self.device)
        self.criterion = ContrastiveLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def setup_data(self):
        """Setup data loaders"""
        transform = transforms.Compose(
            [transforms.Resize((100, 100)), transforms.ToTensor()]
        )

        # Create datasets
        train_dataset = HandwritingDataset(
            self.config["data_dir"], transform=transform, split="train"
        )

        val_dataset = HandwritingDataset(
            self.config["data_dir"], transform=transform, split="val"
        )

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=4,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=4,
        )

    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for batch_idx, (data, labels) in enumerate(self.train_loader):
            # Create positive and negative pairs
            anchor = data[0::2].to(self.device)
            paired = data[1::2].to(self.device)

            # 1 means different class, 0 means same class
            target = (labels[0::2] != labels[1::2]).float().to(self.device)

            # Forward pass
            output1, output2 = self.model(anchor, paired)
            loss = self.criterion(output1, output2, target)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % self.config["log_interval"] == 0:
                self.logger.info(
                    f"Training Batch: {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.6f}"
                )
                wandb.log({"batch_loss": loss.item()})

        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for data, labels in self.val_loader:
                anchor = data[0::2].to(self.device)
                paired = data[1::2].to(self.device)
                target = (labels[0::2] != labels[1::2]).float().to(self.device)

                output1, output2 = self.model(anchor, paired)
                loss = self.criterion(output1, output2, target)
                val_loss += loss.item()

        return val_loss / len(self.val_loader)

    def train(self):
        """Main training loop"""
        best_val_loss = float("inf")

        for epoch in range(self.config["epochs"]):
            self.logger.info(f'Epoch {epoch+1}/{self.config["epochs"]}')

            # Train and validate
            train_loss = self.train_epoch()
            val_loss = self.validate()

            # Log metrics
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            self.logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    self.config["model_save_path"],
                )
                self.logger.info("Saved best model checkpoint")


# Example usage
if __name__ == "__main__":
    config = {
        "data_dir": "path/to/handwriting/dataset",
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 100,
        "log_interval": 10,
        "model_save_path": "handwriting_model.pth",
    }

    trainer = ModelTrainer(config)
    trainer.train()

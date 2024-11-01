import torch
from tqdm import tqdm

class EfficientNetTrainer:
    def __init__(self, model, device, criterion, optimizer, checkpoint_file='logs/checkpoint.pth', use_checkpoint=False, patience=5):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_file = checkpoint_file
        self.use_checkpoint = use_checkpoint
        self.result_record = {'Train Loss': [], 'Train Acc': [], 'Validation Loss': [], 'Validation Acc': []}
        self.best_accuracy = 0.0
        self.patience = patience  # Early stopping patience value
        self.early_stop_counter = 0  # Early stopping counter
        self.best_val_loss = float('inf')  # Best validation loss initialized to infinity

    def train_one_epoch(self, train_loader, epoch, num_epochs):
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero gradients, forward pass, backward pass, and optimization
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # Calculate running loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        self.result_record['Train Loss'].append(epoch_loss)
        self.result_record['Train Acc'].append(epoch_accuracy)
        print(f'Epoch [{epoch+1}/{num_epochs}] finished with Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

        return epoch_loss, epoch_accuracy

    def validate_model(self, val_loader):
        """Validate the model."""
        self.model.eval()
        cost = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                cost += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        validation_loss = cost / len(val_loader)
        validation_accuracy = 100 * correct / total

        self.result_record['Validation Loss'].append(validation_loss)
        self.result_record['Validation Acc'].append(validation_accuracy)

        print(f'Validation Accuracy: {validation_accuracy:.2f}%\tValidation Loss: {validation_loss:.4f}')

        # Early stopping logic
        if validation_loss < self.best_val_loss:
            self.best_val_loss = validation_loss
            self.early_stop_counter = 0  # Reset early stopping counter
            print("Validation loss improved. Reset early stopping counter.")
            torch.save(self.model.state_dict(), "save_model/best_efficientnet_model.pth")  # Save the best model
        else:
            self.early_stop_counter += 1  # Increment counter
            print(f"No improvement in validation loss. Early stopping counter: {self.early_stop_counter}/{self.patience}")

        if self.early_stop_counter >= self.patience:
            print("Early stopping triggered. Stopping training.")
            return "early_stop"

        return validation_loss, validation_accuracy

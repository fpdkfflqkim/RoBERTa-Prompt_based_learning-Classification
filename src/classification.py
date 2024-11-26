import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
from tqdm import tqdm
from sklearn.model_selection import train_test_split

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
config = RobertaConfig.from_pretrained('roberta-large')
config.num_labels = 2
model = RobertaForSequenceClassification(config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

class YelpPolarityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = int(self.labels.iloc[idx])

        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    all_train_losses = []
    
    for epoch in range(num_epochs+1):
        train_losses = []
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        average_train_loss = sum(train_losses) / len(train_losses)
        all_train_losses.append(average_train_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} - Average Train Loss: {average_train_loss:.4f}")
    
        train_loss[epoch] = average_train_loss
        
        
    return all_train_losses

def validate_model(model, val_loader, criterion):
    model.eval()
    all_val_losses = []
    all_val_accuracy = []
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        epoch_losses = []
        
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            epoch_losses.append(loss.item())

            logits = outputs.logits
            _, predicted_labels = torch.max(logits, 1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    average_loss = sum(epoch_losses) / len(epoch_losses)
    all_val_losses.append(average_loss)
    accuracy = total_correct / total_samples
    all_val_accuracy.append(accuracy)
    
    val_loss[epoch] = average_loss
    val_accuracy[epoch] = accuracy
    
    print(f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {average_loss:.4f} - Accuracy: {accuracy}")

    return all_val_losses, all_val_accuracy

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy
import torch
from sklearn.metrics import f1_score, accuracy_score

def train_one_epoch(model, dataloader, criterion, optimizer, device, use_hrv=False):
    """
    Runs training over a single epoch.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for signals, hrvs, labels in dataloader:
        signals, hrvs, labels = signals.to(device), hrvs.to(device), labels.to(device)
        optimizer.zero_grad()
        if use_hrv:
            outputs = model(signals, hrvs)
        else:
            outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_f1, epoch_acc

def evaluate(model, dataloader, criterion, device, use_hrv=False):
    """
    Evaluates the model over the validation set.
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for signals, hrvs, labels in dataloader:
            signals, hrvs, labels = signals.to(device), hrvs.to(device), labels.to(device)
            if use_hrv:
                outputs = model(signals, hrvs)
            else:
                outputs = model(signals)
            batch_loss = criterion(outputs, labels)
            val_loss += batch_loss.item()
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = val_loss / len(dataloader)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_f1, epoch_acc, all_preds, all_probs, all_labels

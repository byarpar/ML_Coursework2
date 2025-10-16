# REMOVED

This file was removed from the repository at the request of the user on October 16, 2025.

## 🔬 Experiment Ideas

### 1. Hyperparameter Tuning

#### Learning Rate
Try different learning rates to see how they affect training:

```python
# In main.py, modify line in train_and_evaluate()
learning_rate=0.01   # Higher learning rate (default: 0.001)
learning_rate=0.0001 # Lower learning rate
```

**Expected outcome**: Higher rates train faster but may be unstable. Lower rates are more stable but slower.

#### Batch Size
Modify batch size to affect training speed and memory usage:

```python
# In main.py, load_and_prepare_data()
batch_size = 64   # Smaller batches (default: 128)
batch_size = 256  # Larger batches
```

**Expected outcome**: Larger batches train faster but need more memory. Smaller batches may generalize better.

#### Number of Epochs
Change training duration:

```python
# In main.py, main()
num_epochs=50   # More training (default: 30)
num_epochs=10   # Less training
```

**Expected outcome**: More epochs improve accuracy but risk overfitting. Fewer epochs train faster.

### 2. Architecture Modifications

#### Add More Convolutional Layers
In `main.py`, add another convolutional block:

```python
self.conv4 = nn.Sequential(
    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2)
)
```

**Expected outcome**: Deeper networks can learn more complex features but may overfit.

#### Change Filter Sizes
Modify the number of filters:

```python
# Instead of 32, 64, 128, try:
# 64, 128, 256 (more filters = more capacity)
# 16, 32, 64 (fewer filters = faster, less memory)
```

#### Adjust Dropout Rates
Change dropout to control overfitting:

```python
nn.Dropout(0.7)  # More dropout (more regularization)
nn.Dropout(0.2)  # Less dropout (less regularization)
```

**Expected outcome**: Higher dropout prevents overfitting but may reduce training accuracy.

### 3. Data Augmentation

Add data augmentation to improve generalization:

```python
# In main.py, load_and_prepare_data()
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Flip images randomly
    transforms.RandomCrop(32, padding=4),  # Random crops
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                        (0.2470, 0.2435, 0.2616))
])
```

**Expected outcome**: Better generalization and higher test accuracy.

### 4. Different Optimizers

Try different optimizers:

```python
# Instead of Adam, try:

# SGD with momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# AdamW (Adam with weight decay)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

**Expected outcome**: Different optimizers converge differently. Adam usually works well.

### 5. Learning Rate Schedulers

Try different learning rate schedules:

```python
# Cosine annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Step decay
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Exponential decay
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
```

### 6. Advanced Architectures

#### Residual Connections (ResNet-style)
Add skip connections to help with deep networks:

```python
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out
```

#### Global Average Pooling
Replace flatten + dense with global average pooling:

```python
self.gap = nn.AdaptiveAvgPool2d(1)  # Global average pooling
self.fc = nn.Linear(128, num_classes)

def forward(self, x):
    x = self.conv3(x)
    x = self.gap(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x
```

## 📊 Tracking Experiments

Create a simple experiment log:

```python
experiments = {
    'experiment_1': {
        'description': 'Baseline model',
        'hyperparameters': {
            'learning_rate': 0.001,
            'batch_size': 128,
            'epochs': 30
        },
        'results': {
            'train_acc': 0.89,
            'test_acc': 0.78
        }
    },
    'experiment_2': {
        'description': 'Higher learning rate',
        'hyperparameters': {
            'learning_rate': 0.01,
            'batch_size': 128,
            'epochs': 30
        },
        'results': {
            'train_acc': 0.91,
            'test_acc': 0.76
        }
    }
}
```

## 🎯 Performance Goals

Try to achieve these milestones:

- **Baseline**: 75-80% test accuracy
- **Good**: 80-85% test accuracy
- **Excellent**: 85-90% test accuracy
- **State-of-the-art**: 90%+ test accuracy

## 🔍 Debugging Tips

### Model is not learning (accuracy stays low)
- Check learning rate (might be too low or too high)
- Verify data preprocessing (images should be normalized)
- Ensure loss is decreasing
- Check for bugs in forward pass

### Overfitting (high train acc, low test acc)
- Add more dropout
- Use data augmentation
- Reduce model complexity
- Add weight decay
- Train for fewer epochs

### Underfitting (both accuracies are low)
- Increase model capacity (more filters/layers)
- Train for more epochs
- Reduce dropout
- Increase learning rate

### Training is too slow
- Reduce batch size
- Use GPU if available
- Reduce model size
- Train for fewer epochs

## 📈 Monitoring Training

Add these prints to monitor training:

```python
# Check gradient magnitudes
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.abs().mean()}")

# Check learning rate
print(f"Current LR: {optimizer.param_groups[0]['lr']}")

# Save checkpoints
if epoch % 5 == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'checkpoint_epoch_{epoch}.pth')
```

## 🧮 Advanced Techniques

### Mixed Precision Training (Faster on modern GPUs)
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Gradient Clipping (Prevents exploding gradients)
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Weight Initialization
```python
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)

model.apply(init_weights)
```

## 📚 Further Learning

### Concepts to Explore
1. Transfer Learning (using pre-trained models)
2. Ensemble Methods (combining multiple models)
3. Adversarial Training
4. Self-supervised Learning
5. Neural Architecture Search

### Resources
- PyTorch Tutorials: https://pytorch.org/tutorials/
- Papers with Code: https://paperswithcode.com/
- Deep Learning Book: https://www.deeplearningbook.org/

## 💡 Tips for Coursework

1. **Start Simple**: Begin with the baseline model, then gradually add improvements
2. **Document Everything**: Keep track of what you tried and the results
3. **Visualize**: Use plots to understand what's happening during training
4. **Be Patient**: Deep learning takes time to train and tune
5. **Test Incrementally**: Make small changes and test each one

## ⚠️ Common Mistakes to Avoid

1. Not normalizing input data
2. Using too high a learning rate
3. Not shuffling training data
4. Forgetting to set model to eval mode during testing
5. Not using dropout during training
6. Training on test set (data leakage)
7. Not saving the best model

---

**Happy Experimenting! 🚀**

Remember: Understanding why something works is more important than just getting high accuracy.

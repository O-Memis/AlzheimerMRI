
"""
May 2025 - Oguzhan Memis


OASIS 1 DATASET : Alzheimer's Disease (AD) Brain MRI Images

2007

Dataset description:
    
    T1-W MRI images in Transverse plane
    
    Cross-sectional data collection of 461 subjects, 18-96 yrs
    
    80k images (2D slices) converted to .JPG from NIFTI
    
    8-bit and [248, 496] dimensions of gray-scale images, but saved in 3 channels
    
    There are 4 classes to classify AD: 
                                        non-demented
                                        mild-demented
                                        modereate demented
                                        very demented
                                        
 

Obtained from here for educational purpose: 
https://www.kaggle.com/datasets/ninadaithal/imagesoasis

                                       
"""







#%% 1) Inspect the image attributes


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt



image_dir = 'Data/Mild Dementia'  #--------------------change here for the other files



# List to store image attributes
image_info = []

# Iterate through each image in the directory
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg'):  # Add other image extensions if needed
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)

        if img is not None:
            # Get the size and channels
            height, width = img.shape[:2]
            channels = img.shape[2] if len(img.shape) == 3 else 1

            # Check if the image is grayscale with redundant channels
            is_grayscale = False
            if channels == 3:
                # Compare all three channels to see if they are identical
                b, g, r = cv2.split(img)
                if np.array_equal(b, g) and np.array_equal(b, r):
                    is_grayscale = True


            # Get additional attributes
            file_size = os.path.getsize(img_path)  # File size in bytes
            dtype = img.dtype  # Data type of the image
            bit_depth = img.itemsize * 8  # itemsize gives the size of one pixel element in bytes

            # Calculate aspect ratio
            aspect_ratio = round(width / height, 2)


           

            # Append the information to the list
            image_info.append({
                "filename": filename,
                "size": (height, width),
                "channels": channels,
                "is_grayscale": is_grayscale,
                "file_size": file_size,
                "dtype": dtype,
                "bit_depth": bit_depth,
                "aspect_ratio": aspect_ratio,

            })
        else:
            print(f"Could not read image: {img_path}")


# Print the collected image information
for info in image_info:
    print(f"Image: {info['filename']}, Size: {info['size']}, Channels: {info['channels']}, "
          f"Grayscale: {info['is_grayscale']}, File Size: {info['file_size']} bytes, "
          f"Data Type: {info['dtype']}, Bit Depth: {info['bit_depth']}, Aspect Ratio: {info['aspect_ratio']}")




# ------Inspection of basic statistical features from the first 10 images



detailed_info = []


for idx, info in enumerate(image_info[:10]):  
    img_path = os.path.join(image_dir, info['filename'])
    img = cv2.imread(img_path)

    if img is not None:
        if info['is_grayscale'] or info['channels'] == 1:  # compatible with grayscale
            
            # Calculate mean and standard deviation
            mean, stddev = cv2.meanStdDev(img)[:2]
            mean = mean[0][0]
            stddev = stddev[0][0]

            # Calculate histogram
            histogram = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        else:
            mean, stddev, histogram = None, None, None    # Not calculated for non-grayscale images!

        
        detailed_info.append({
            "filename": info['filename'],
            "mean_pixel_value": mean,
            "stddev_pixel_value": stddev,
            "histogram": histogram,
        })


# Plot histograms for the first 10 images in subplots
fig, axes = plt.subplots(5, 2, figsize=(20, 30))  


axes = axes.ravel()  # Flatten the 2D grid of axes into a 1D array, to access with easier loop

for idx, detail in enumerate(detailed_info):
    if detail['histogram'] is not None:
        axes[idx].plot(detail['histogram'], color='blue')
        axes[idx].set_title(f"Image {idx+1}: {detail['filename']}", fontsize=10)
        axes[idx].set_xlabel("Pixel Intensity", fontsize=8)
        axes[idx].set_ylabel("Frequency", fontsize=8)
        axes[idx].grid(True)



plt.tight_layout()
plt.show()


# Print the mean and standard deviation for the first 10 images
print("\nStatistical Features for First 10 Images:")

for detail in detailed_info:
    print(f"Image: {detail['filename']}")
    if detail['mean_pixel_value'] is not None and detail['stddev_pixel_value'] is not None:
        print(f"  Mean Pixel Value: {detail['mean_pixel_value']:.2f}, StdDev Pixel Value: {detail['stddev_pixel_value']:.2f}")




#%% 2) Image processing and further analysis







from torchvision import datasets, transforms
from torch.utils.data import DataLoader



image_dir = "Data"  


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale (1 channel)
    transforms.ToTensor(),  
])


# Load the dataset using ImageFolder (assumes images are organized in subfolders by class)
dataset = datasets.ImageFolder(root=image_dir, transform=transform)









#%% 3) CNN model training




import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
import seaborn as sns






# 1. Processing Unit Selection
print("CUDA Availability:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computing Device: {device}")





# ---------------------------------------------------------Hyperparameters here
lr = 0.001 # learning rate
batch = 64
epochs = 30 
patience = 10  # Early stopping patience





# 5. Data split ratios
dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)  # accepts only integer numbers, which corresponds the real size
test_size = dataset_size - train_size  # %30 test size



# 6. Train-test splitting. It allows preventing mixture and leakage between train and test data. It splits in the same way in each run.
train_data , test_data = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))



# 7. Data loader functions, to mix in each epoch, and load them in batches
train_loader = DataLoader(train_data, batch_size= batch, shuffle=True)
test_loader = DataLoader(test_data, batch_size= batch, shuffle=True)   # It prevents the dependency on data order




# 7. Model definition__________________an AlexNet variant______________________
class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2))  # Input: 1 channel, Output: 32 channels
        self.relu1 = nn.SiLU()  
        self.lrn1 = nn.BatchNorm2d(32)  
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(32, 96, kernel_size=3, padding=1)  # Input: 32 channels, Output: 96 channels
        self.relu2 = nn.SiLU()  
        self.lrn2 = nn.BatchNorm2d(96)  
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  

        # Third Convolutional Layer
        self.conv3 = nn.Conv2d(96, 288, kernel_size=3, padding=1)  
        self.relu3 = nn.SiLU()  

        # Fourth Convolutional Layer
        self.conv4 = nn.Conv2d(288, 288, kernel_size=3, padding=1)  
        self.relu4 = nn.SiLU()  

        # Fifth Convolutional Layer
        self.conv5 = nn.Conv2d(288, 288, kernel_size=3, padding=1)  
        self.relu5 = nn.SiLU()  
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  

        # Flatten Layer
        self.flatten = nn.Flatten()  # Flattens the feature maps into a vector 

        # Fully Connected Layers
        self.fc1 = nn.Linear(288*15*30, 4096)
        self.relu6 = nn.SiLU()  
        self.dropout1 = nn.Dropout(p=0.2)  

        self.fc2 = nn.Linear(4096, 512)  
        self.relu7 = nn.SiLU()  
        self.dropout2 = nn.Dropout(p=0.1)  

        self.fc3 = nn.Linear(512, 4)  

    def forward(self, x):

        # 1. Convolutional Block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.lrn1(x)
        x = self.pool1(x)

        # 2. Convolutional Block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.lrn2(x)
        x = self.pool2(x)

        # 3. Convolutional Block
        x = self.conv3(x)
        x = self.relu3(x)

        # 4. Convolutional Block
        x = self.conv4(x)
        x = self.relu4(x)

        # 5. Convolutional Block
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)

        # Flatten and Fully Connected Layers
        x = self.flatten(x)  
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu7(x)
        x = self.dropout2(x)

        x = self.fc3(x)  # Final output layer

        return x



model = AlexNet().to(device)



# 8. Loss and Optimizer________________________________________________________
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)



# 9. Train and Evaluation loops with Early Stopping 
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

best_test_accuracy = 0.0  # variables to control and save the best weights
best_model_state = None  
early_stopping_counter = 0


for epoch in range(epochs):
    # Training
    model.train()
    running_loss = 0.0
    y_true_train = []
    y_pred_train = []
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        y_true_train.extend(labels.cpu().numpy())
        y_pred_train.extend(predicted.cpu().numpy())
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = accuracy_score(y_true_train, y_pred_train) * 100
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # Test
    model.eval()
    test_loss = 0.0
    y_true_test = []
    y_pred_test = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            y_true_test.extend(labels.cpu().numpy())
            y_pred_test.extend(predicted.cpu().numpy())
    
    test_loss = test_loss / len(test_loader)
    test_accuracy = accuracy_score(y_true_test, y_pred_test) * 100
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    # Check for Early Stopping
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        # copy of the model state
        best_model_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
    
    if early_stopping_counter >= patience:
        print("Early stopping triggered.")
        break
    
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")


#%% 4) Testing and metrics


print("\nRestoring best model weights...")
if best_model_state is not None:
    best_model_state = {key: value.to(device) for key, value in best_model_state.items()}
    model.load_state_dict(best_model_state) #-------------------best model loaded-----------
    print(f"Best model restored with test accuracy: {best_test_accuracy:.2f}%")


# 11. Final evaluation
model.eval()
test_loss = 0.0
y_true_test = []
y_pred_test = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        y_true_test.extend(labels.cpu().numpy())
        y_pred_test.extend(predicted.cpu().numpy())


# 12. Calculate the  metrics
accuracy = accuracy_score(y_true_test, y_pred_test)
precision = precision_score(y_true_test, y_pred_test, average='weighted')
f1 = f1_score(y_true_test, y_pred_test, average='weighted')

print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test Precision: {precision:.4f}')
print(f'Test F1 Score: {f1:.4f}')



plt.figure(figsize=(14, 6))

# Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.show()

# Confusion matrix
cm = confusion_matrix(y_true_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


#%% debug if needed


for _, labels in train_loader:
    print(labels.dtype)  # Should be torch.int64
    break

print(f"Classes: {dataset.classes}")
print(f"Class-to-Index Mapping: {dataset.class_to_idx}")

print(f"Labels range: {min(dataset.targets)} to {max(dataset.targets)}")

for inputs, labels in train_loader:
    print(inputs.shape)  # Should be [batch_size, 1, 248, 496]
    break

torch.cuda.is_available()  # Should return True
torch.cuda.current_device()  # Should return the current GPU ID
torch.cuda.get_device_name(torch.cuda.current_device())  # Should return the GPU name



#%% 5) Cross-validation






#%% 6) Final test and save the model



image_path = "Data/Moderate Dementia/OAS1_0308_MR1_mpr-2_100.jpg"  

image = cv2.imread(image_path)


plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(f"Testing: {image_path}")
plt.show()


# Apply transforms for consistency
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

image_tensor = transform(image).unsqueeze(0).to(device)  




model.eval()
with torch.no_grad():
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    predicted_idx = predicted.item()

# Get class the names 
class_names = dataset.classes
predicted_class = class_names[predicted_idx]


print(f"Predicted class: {predicted_class}")
print(f"Predicted class ID: {predicted_idx}")


torch.save(model.state_dict(), 'alzheimer_mri_alexnet1.pth')

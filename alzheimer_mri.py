
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


Comments are added for documentation                                       
"""




#%% 1) Importing general libraries for modular usage


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score





#%% 2) Inspect the image attributes




image_dir = 'Data/Mild Dementia'  #-----------------------change here to look for the other files in the dataset




image_info = []

# 2.1)  Iterate through each image in the directory
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg'):                     # adding image extensions 
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)

        if img is not None:
            
            # 2.2) getting the size and channels
            height, width = img.shape[:2]
            channels = img.shape[2] if len(img.shape) == 3 else 1


            # 2.3) check if the image is grayscale 
            is_grayscale = False
            if channels == 3:
                
                b, g, r = cv2.split(img)
                if np.array_equal(b, g) and np.array_equal(b, r):
                    is_grayscale = True



            # 2.3) look for the other attributes of the images----------------- 
            
            file_size = os.path.getsize(img_path)        # File size in bytes
            dtype = img.dtype                            # Data type of the image
            bit_depth = img.itemsize * 8                 # itemsize gives the size of one pixel element in bytes
           
            aspect_ratio = round(width / height, 2)      # aspect ratio


           

            # 2.4) append the information to the list
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




# 2.5) print the collected image information
for info in image_info:
    print(f"Image: {info['filename']}, Size: {info['size']}, Channels: {info['channels']}, "
          f"Grayscale: {info['is_grayscale']}, File Size: {info['file_size']} bytes, "
          f"Data Type: {info['dtype']}, Bit Depth: {info['bit_depth']}, Aspect Ratio: {info['aspect_ratio']}")




# 2.6) Inspection of basic statistical features from the first 10 images



detailed_info = []


for idx, info in enumerate(image_info[:10]):  
    img_path = os.path.join(image_dir, info['filename'])
    img = cv2.imread(img_path)

    if img is not None:
        if info['is_grayscale'] or info['channels'] == 1:  # compatible with grayscale
            
        
            # 2.7) calculate the mean and standard deviation
            mean, stddev = cv2.meanStdDev(img)[:2]
            mean = mean[0][0]
            stddev = stddev[0][0]


            # 2.8) compute the histogram
            histogram = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        else:
            mean, stddev, histogram = None, None, None    # not calculated for non-grayscale

        
        detailed_info.append({
            "filename": info['filename'],
            "mean_pixel_value": mean,
            "stddev_pixel_value": stddev,
            "histogram": histogram,
        })



# 2.9) Plot histograms for the first 10 images in subplots
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



# 2.10) Print the mean and standard deviation for the first 10 images
print("\nStatistical Features for First 10 Images:")

for detail in detailed_info:
    print(f"Image: {detail['filename']}")
    if detail['mean_pixel_value'] is not None and detail['stddev_pixel_value'] is not None:
        print(f"  Mean Pixel Value: {detail['mean_pixel_value']:.2f}, StdDev Pixel Value: {detail['stddev_pixel_value']:.2f}")




#%% 3) Dataset preparation





image_dir = "Data"  # the root directory for the image folders



# 3.1) apply the necessary transformations to format the data

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),    # convert to grayscale (1 channel)
    transforms.ToTensor(),                          # replace them from numpy array to tensors
])




# 3.2) load the dataset using ImageFolder 
dataset = datasets.ImageFolder(root=image_dir, transform=transform)



"""
It automatically generates class information for our model,
by assuming that the images are organized in subfolders by classes.


          Data
        ___|________________________
       |               |           |
       |               |           |
      class 0      class 1       class 2

"""



#%% 4) CNN model training




# 4.1) Processing unit selection
print("CUDA Availability:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computing Device: {device}")





# ----  4.2) define hyperparameters here---------------------------------------

lr = 0.001    # learning rate
batch = 128
epochs = 10 
patience = 5  # Early stopping patience





# 4.3) dataset split ratios
dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)   # accepts only integer numbers, which corresponds the real size
test_size = dataset_size - train_size  # %30 test size obtained




# 4.4) train-test data assignment
train_data , test_data = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))




# 4.5) Data loader functions to load them in batches
train_loader = DataLoader(train_data, batch_size= batch, shuffle=True)
test_loader = DataLoader(test_data, batch_size= batch, shuffle=True)    
# It prevents the dependency on data order in batches





# 4.6) Model definitions__________________an AlexNet variant______________________________________
class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        # definitions of the functions that will be used
        

        # 1. Convolutional Layer-----------------------------------------------
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2))  
        # Input: 1 channel image, Output: 32 channels of filtered images = feature maps
        # kernel size = the size of filters,  stride = downsampling of feature maps
        
        
        self.relu1 = nn.SiLU()                                 # activation function
        self.lrn1 = nn.BatchNorm2d(32)                         # normalization of the values 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)     # taking the maximum value, also downsamples it


        # 2. Convolutional Layer-----------------------------------------------
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Input: 32 channels, Output: 64 channels
        self.relu2 = nn.SiLU()  
        self.lrn2 = nn.BatchNorm2d(64)  
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  


        # 3. Conv. Layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  
        self.relu3 = nn.SiLU()  

        # 4. Conv. Layer
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  
        self.relu4 = nn.SiLU()  

        # 5. Conv. Layer
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  
        self.relu5 = nn.SiLU()  
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  


        # Flattening
        self.flatten = nn.Flatten()   # Flattens the feature maps into a vector 


        # Fully Connected Layers (MLP)
        self.fc1 = nn.Linear(64*15*30, 4096)   # calculation of the size is important
        self.relu6 = nn.SiLU()  
        self.dropout1 = nn.Dropout(p=0.2)  

        self.fc2 = nn.Linear(4096, 512)  
        self.relu7 = nn.SiLU()  
        self.dropout2 = nn.Dropout(p=0.1)  

        self.fc3 = nn.Linear(512, 4)  


    def forward(self, x):
        # arrangement of workflow of the functions
        

        # 1. Convolutional Layer
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.lrn1(x)
        x = self.pool1(x)

        # 2. Convolutional Layer
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.lrn2(x)
        x = self.pool2(x)

        # 3. Conv. Layer
        x = self.conv3(x)
        x = self.relu3(x)

        # 4. Conv. Layer
        x = self.conv4(x)
        x = self.relu4(x)

        # 5. Conv. Layer
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




# 4.7) assigning the model to a variable, and moving it to GPU
model = AlexNet().to(device)  





# ---- 4.8) Loss Function and Optimizer are defined here

criterion = nn.CrossEntropyLoss()  # appropriate for classification tasks, includes Softmax

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005) 
# weight decay as a regularization parameter





# ---- 4.9)  Train-test loops (Epoch loops) with Early Stopping --------------- 


train_losses = []           # to store the metric scores
test_losses = []
train_accuracies = []
test_accuracies = []



best_test_accuracy = 0.0    # variables to check and save the best weights
best_model_state = None  
early_stopping_counter = 0



#--------- 2 nested loops for training procedure, and to obtain test accuracies
for epoch in range(epochs):
    
    
    model.train()        # training mode
    running_loss = 0.0
    y_true_train = []
    y_pred_train = []
    
    for inputs, labels in train_loader:      # train data-loader functions
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()                # deletes previous gradients
        outputs = model(inputs)              # forward processing of data
        loss = criterion(outputs, labels)    # usage of loss function
        

        loss.backward()                      # calculation of gradients
        optimizer.step()                     # updating the parameters via optimizer 
        
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)           # probabilities to labels
        y_true_train.extend(labels.cpu().numpy())      # transferring to CPU
        y_pred_train.extend(predicted.cpu().numpy())
    
    
    train_loss = running_loss / len(train_loader)      # average loss info
    train_accuracy = accuracy_score(y_true_train, y_pred_train) * 100
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)            # train accuracies are saved
    
    
    
    # Testing mode to obtain test accuracies 
    model.eval()
    test_loss = 0.0
    y_true_test = []
    y_pred_test = []
    
    with torch.no_grad():                   # disables the gradient calculation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)   # test data into GPU
            
            outputs = model(inputs)             # forward processing
            loss = criterion(outputs, labels)   # loss function
            
            test_loss += loss.item()            
            _, predicted = torch.max(outputs, 1)   # probabilities to labels
            y_true_test.extend(labels.cpu().numpy())
            y_pred_test.extend(predicted.cpu().numpy())
   
    
    test_loss = test_loss / len(test_loader)
    test_accuracy = accuracy_score(y_true_test, y_pred_test) * 100
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)      # test accuracies are saved
    
    
    
    # Checking the Early Stopping
    
    if test_accuracy > best_test_accuracy:     # if the previous best score is surpassed
        best_test_accuracy = test_accuracy
        
        # copy of the model components
        best_model_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        early_stopping_counter = 0
        
    else:
        early_stopping_counter += 1
    
    if early_stopping_counter >= patience:
        print("Early stopping triggered.")
        break
    
    
    # print the metrics while epochs are ongoing
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")



# --- end of epoch loops 




#%% 5) Final evaluation and the metrics


# 5.1) Loading the saved best model during epochs

print("\nRestoring best model weights...")

if best_model_state is not None:
    
    best_model_state = {key: value.to(device) for key, value in best_model_state.items()}
    model.load_state_dict(best_model_state) 
    print(f"Best model restored with test accuracy: {best_test_accuracy:.2f}%")




# 5.2) Model into evaluation state
model.eval()
test_loss = 0.0
y_true_test = []
y_pred_test = []

with torch.no_grad():
    for inputs, labels in test_loader:     # loading the test data by the loader function
        inputs, labels = inputs.to(device), labels.to(device)     # data to GPU
        
        # forward processing steps
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        y_true_test.extend(labels.cpu().numpy())     # label information to CPU
        y_pred_test.extend(predicted.cpu().numpy())




# 5.3)  Calculate the classification metrics

accuracy = accuracy_score(y_true_test, y_pred_test)
precision = precision_score(y_true_test, y_pred_test, average='weighted')
f1 = f1_score(y_true_test, y_pred_test, average='weighted')
recal = recall_score(y_true_test, y_pred_test, average='weighted')




""" 
---- Metrics for Class Imbalance

Averaging options for classification metrics are crucial for multi-class classification problems
They help to determine how these scores will be calculated -across different classes-


average='weighted': Can be used when you want to give appropriate weight to each class 
                    based on its frequency.


average='macro' : Calculates the metric for each class individually, 
                  then takes the simple average of these values.
   
    
   
average='micro' : Counting total TP, FP, FN across all classes, 
                  then calculates individual metrics by using these counts.


average=None :  Simply returns a list of the metric for each class.

"""



print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test Precision: {precision:.4f}')
print(f'Test F1 Score: {f1:.4f}')
print(f'Test Recall: {recal:.4f}')




# 5.4)  Plotting the metrics that tracked in epochs

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



# 5.4)  Plotting the Confusion Matrix that shows how the classes are predicted

cm = confusion_matrix(y_true_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()



#%% 6) Cross-validation


"""
The CV is used for more effective utilization of training set. 

K-fold CV relies on the principle of splitting the training data into smaller parts, 
and iteratively use one part for testing and remaining for the training.
To demonstrate the model performance on all of the dataset, without leakage.


To adress the bias between groups, and the class imbalance;
different methods of k-fold CV should be used. 
"""



# 6.1) Stratified k-fold to distribute the data in accurate class proportions
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import SubsetRandomSampler



# ---- Hyperparameters for model are here -------------------------------------
lr = 0.001 
batch = 128
epochs = 10 
patience = 5  # Early stopping counter


# folds are defined by the function
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)




# 6.2) access the labels of training data
train_labels = []

for _, label in train_data:
    if isinstance(label, torch.Tensor):      # to handle different label format
        train_labels.append(label.item())
        
    else:
        train_labels.append(label)

train_labels = np.array(train_labels)





# ---- 6.3) Loop through each fold

fold_accuracies = []    # variables to store metrics for each fold
fold_precisions = []
fold_recalls = []
fold_f1_scores = []


# folds arranged according to labels
fold_splits = skf.split(np.arange(len(train_labels)), train_labels) 



for fold_number, (train_indices, validation_indices) in enumerate(fold_splits):
    current_fold = fold_number + 1
    print(f"Processing fold {current_fold} of {k}")
    
    
    # sampler to generate the fold data from existing one
    train_sampler = SubsetRandomSampler(train_indices)   
    val_sampler = SubsetRandomSampler(validation_indices)
    
    fold_train_loader = DataLoader(train_data, batch_size=batch, sampler=train_sampler)
    fold_val_loader = DataLoader(train_data, batch_size=batch, sampler=val_sampler)
    
    
    # Initialize a fresh model for each fold
    model = AlexNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    
    
    
    # Epoch loops 
    early_stopping_counter = 0  # early stopping can either be excluded or hold
    best_val_accuracy = 0
    
    for epoch in range(epochs):
        
        model.train()
        for inputs, labels in fold_train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()   
            
            # train accuracies for individual epochs are not tracked here
        
        
        # test stage
        model.eval()
        val_true = []
        val_pred = []
        
        with torch.no_grad():
            for inputs, labels in fold_val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_true.extend(labels.cpu().numpy())
                val_pred.extend(predicted.cpu().numpy())
            
        val_accuracy = accuracy_score(val_true, val_pred)  # epoch test accuracy
        
        
        
        # Early stopping check (without saving the model)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    
    # Final evaluation for this fold, after the Epoch loop has ended
    model.eval()
    val_true = []
    val_pred = []
    
    with torch.no_grad():
        for inputs, labels in fold_val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_true.extend(labels.cpu().numpy())
            val_pred.extend(predicted.cpu().numpy())
    
    
    
    # Calculate and store metrics 
    fold_accuracy = accuracy_score(val_true, val_pred)
    fold_precision = precision_score(val_true, val_pred, average='weighted')
    fold_recall = recall_score(val_true, val_pred, average='weighted')
    fold_f1 = f1_score(val_true, val_pred, average='weighted')
    
    fold_accuracies.append(fold_accuracy)
    fold_precisions.append(fold_precision)
    fold_recalls.append(fold_recall)
    fold_f1_scores.append(fold_f1)
    
    print(f"Fold {current_fold} - Accuracy: {fold_accuracy:.4f}, Precision: {fold_precision:.4f}, "
          f"Recall: {fold_recall:.4f}, F1: {fold_f1:.4f}")


# Final metrics across all the folds
print("\nK-Fold Cross-Validation Results:")
print(f"Average Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
print(f"Average Precision: {np.mean(fold_precisions):.4f} ± {np.std(fold_precisions):.4f}")
print(f"Average Recall: {np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}")
print(f"Average F1 Score: {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}")



#%% Optional1: Save the current model



torch.save(model, 'alzheimer_mri_alexnet2.pth') # saves the whole structure


# You will need to specify the model architecture to use it later.



"""
If you want to save only the weights of the model, this can be used:
    
torch.save(model.state_dict(), 'alzheimer_mri_alexnet1.pth')
    
"""



#%% Optional2: Perform debug if needed



# CUDA version
torch.cuda.is_available()  # should return True
torch.cuda.current_device()  # should return the current GPU ID
torch.cuda.get_device_name(torch.cuda.current_device())  # should return the GPU name





for _, labels in train_loader:
    print(labels.dtype)  # label data format should be torch.int64 (tensor)
    break



# Data labels
print(f"Classes: {dataset.classes}")
print(f"Class-to-Index Mapping: {dataset.class_to_idx}")
print(f"Labels range: {min(dataset.targets)} to {max(dataset.targets)}")



# To investigate the data
for inputs, labels in train_loader:
    print(inputs.shape)  # Should be [batch_size, 1, 248, 496]
    break




#%% 7) Demo: Use the saved model



# RUN THE SECTIONS 1) AND 3) 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# ---- 7.1) Select an image from the dataset arbitrarily
image_path = "Data/Very mild Dementia/OAS1_0023_MR1_mpr-1_133.jpg"  

#Moderate Dementia/OAS1_0308_MR1_mpr-2_100.jpg


image = cv2.imread(image_path)    # to read the image, it outputs a numpy array

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(f"Testing Image: {image_path}")
plt.show()




# 7.2) Apply the transforms for consistency

transform = transforms.Compose([
    transforms.ToPILImage(),                              # PIL convertion is manually performed
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])


image_tensor = transform(image).unsqueeze(0).to(device)   # moved to GPU




# ---- 7.3) Load the saved model------------copy-paste the model architecture to reuse----------

class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2))  
        self.relu1 = nn.SiLU()  
        self.lrn1 = nn.BatchNorm2d(32)  
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  
       
        self.conv2 = nn.Conv2d(32, 96, kernel_size=3, padding=1)  
        self.relu2 = nn.SiLU()  
        self.lrn2 = nn.BatchNorm2d(96)  
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  
       
        self.conv3 = nn.Conv2d(96, 288, kernel_size=3, padding=1)  
        self.relu3 = nn.SiLU()  
       
        self.conv4 = nn.Conv2d(288, 288, kernel_size=3, padding=1)  
        self.relu4 = nn.SiLU()  
       
        self.conv5 = nn.Conv2d(288, 288, kernel_size=3, padding=1)  
        self.relu5 = nn.SiLU()  
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  
    
        self.flatten = nn.Flatten()   
     
        self.fc1 = nn.Linear(288*15*30, 4096)
        self.relu6 = nn.SiLU()  
        self.dropout1 = nn.Dropout(p=0.2)  
        self.fc2 = nn.Linear(4096, 512)  
        self.relu7 = nn.SiLU()  
        self.dropout2 = nn.Dropout(p=0.1)  
        self.fc3 = nn.Linear(512, 4)  

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.lrn1(x)
        x = self.pool1(x)
       
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.lrn2(x)
        x = self.pool2(x)
       
        x = self.conv3(x)
        x = self.relu3(x)
       
        x = self.conv4(x)
        x = self.relu4(x)
        
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
       
        x = self.flatten(x)  
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu7(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x



model = torch.load('alzheimer_mri_alexnet1.pth', map_location=device, weights_only=False)





# 7.4) Use the model in testing state------------------------------------------
model.eval()
with torch.no_grad():
    output = model(image_tensor)          # apply the model by forward pass
    _, predicted = torch.max(output, 1)   # probability to label
    predicted_idx = predicted.item()      # get the predicted class id (0,1,2,3)






# 7.5) Print the predicted class name

 
class_names = dataset.classes
predicted_class = class_names[predicted_idx]   # getting the corresponding name


print(f"Predicted class: {predicted_class}")
print(f"Predicted class ID: {predicted_idx}")



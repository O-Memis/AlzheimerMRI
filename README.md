# AlzheimerMRI
 Classification of T1-Weighted MRI images of Alzheimer's Disease, in Python. Obtained from an open-source dataset. Supervised classification is done by designing a CNN model by using the PyTorch library. 

<br><br>
May 2025 - Oguzhan Memis

<br><br>

:tr: Türkçe: Derin Öğrenme ile Alzheimer MR görüntülerinin sınıflandırılması. Aksiyal (tranverse) eksende T1-Weighted MR görüntüleri çekilerek hazırlanmış olan, erişime açık bir veri seti kullanılmıştır. Bu veri setinde 4 adet sınıf bulunmaktadır. PyTorch kütüphanesi ile yazılmış bir CNN modeli kullanılarak, bu görüntüler başarıyla sınıflandırılmıştır.

<br><br>

## Content
1) Dataset Description
2) Code Organization
3) Current Results


<br><br>



## 1-Dataset Description

OASIS-1 DATASET (2007) : Alzheimer's Disease (AD) Brain MRI Images <br>

Obtained from here for educational purpose: 
[Kaggle](https://www.kaggle.com/datasets/ninadaithal/imagesoasis) <br>

Original source: [OasisBrains](https://sites.wustl.edu/oasisbrains/home/oasis-1/) <br><br>


**Data characteristics:** 

 T1-W protocol of MRI images in Transverse plane. <br>
    
 Cross-sectional data collection of 461 subjects, 18-96 yrs. <br>
    
 86k images (2D slices) converted to .JPG from NIFTI <br>
    
 8-bit and [248, 496] dimensions of **gray-scale** images, but saved in 3 channels. <br>

 The classes are unbalanced. The Non-Demented class dominates the others. <br>
    
 There are **4 classes** to classify AD: <br>
                                        **non-demented** <br>
                                        **mild-demented** <br>
                                        **modereate demented** <br>
                                        **very demented** <br>

<br><br>

# 2-Code Organization

 The codes are separated into 7 different cells based on general Deep Learning workflow. <br>
 Read the comments and run the codes cell by cell. You can also run the whole code at once, if you want. <br><br>
    
    
   The cells are as follows: <br><br>
        1) Importing general libraries for modular usage <br>
        2) Inspect the image attributes <br>
        3) Image processing for further analysis <br>
        4) CNN model training <br>
        5) Testing and evaluation <br>
        6) Cross-validation (not done yet) <br>
        7) Demo: Use the saved model <br>


Cells are constructed by the command " **#%%** " in Spyder IDE. <br><br>


## 3-Current Results

 %99,8 Test accuracy  <br>
 %99,8 Test precision <br>
 %99,8 Test F1 score <br>

 Obtained by slightly modified AlexNet CNN model.

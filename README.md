# Food Image and News Classification

## 📌 Overview
This project aims to classify food images and news articles into relevant categories. It consists of two main parts: image classification and text classification.

In the image classification section, various deep learning models including InceptionV3, ResNet, DenseNet, MobileNet, and Xception were implemented using Keras. Techniques such as data augmentation and handling missing labels were employed to enhance model performance. Data was downloaded from this [link](https://drive.google.com/file/d/15CHt2ueS4c7emHpmzFHC3c0TGd51Mnvz/view?usp=drive_link).

For text classification, two datasets ([AG News](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) and [BBC News](https://www.kaggle.com/competitions/learn-ai-bbc/data)) were utilized to tokenize and train the NLP model.

## 🎯 Objectives
- Implement deep learning models for image classification to classify food images.
- Develop an NLP model for text classification to categorize news articles effectively.
- Explore techniques such as data augmentation and tokenization to improve model performance.
- Present results through visualizations and evaluation metrics to assess model effectiveness.

## Image Classification

In the image classification section, various models including InceptionV3, ResNet, DenseNet, MobileNet, and Xception were implemented using Keras. Techniques such as data augmentation and handling missing labels were employed to enhance model performance.

### File Structure:
```
Image Classification
│   main.py
│   Q1_Image_Classification.ipynb
|
└───Models
│   DenseNet201_augmented_imagenet.h5
│   DenseNet201_augmented_imagenet.json
│   DenseNet201_imagenet.h5
│   DenseNet201_imagenet.json
│   history_DenseNet201_augmented_imagenet.npy
│   history_DenseNet201_imagenet.npy
│   history_Xception_augmented_imagenet.npy
│   Xception_augmented_imagenet.h5
│   Xception_augmented_imagenet.json
|
└───Pytorch ResNet
        test.ipynb
        train.ipynb
```

## Text Classification

For text classification, two datasets (AG News and BBC News) were utilized to tokenize and train the NLP model.

### File Structure:
```
Text Classification
│   main.py
│   Project2_P2_NLP.ipynb
│
├───Data
│       BBC News Train.csv
│       bbc_test.csv
│       data_news.csv
│       test.csv
│       test_ag.csv
│       train.csv
│
└───Models
        my_model.h5
        tokenizer_model.json
```

## 🚀 Getting Started
- For image classification, navigate to the `Image Classification` directory and run `main.py`.
- For text classification, navigate to the `Text Classification` directory and run `main.py`.

## 📊 Results
- Image classification results can be found in `Q1_Image_Classification.ipynb`.
- Text classification results can be found in `Project2_P2_NLP.ipynb`.

## 🤝 Feedback
We welcome any feedback, bug reports, and suggestions. Please let us know if you encounter any issues or have ideas for improvement.
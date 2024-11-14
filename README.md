# Crop Disease Outbreak Prediction

This project is designed to predict crop disease outbreaks by analyzing historical disease and weather data. By leveraging machine learning, it identifies patterns and correlations between environmental factors and disease outbreaks, supporting early detection and prevention efforts for farmers.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model](#model)

## Installation
To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/pallavikailas/crop-disease.git
cd crop-disease
pip install -r requirements.txt
```

## Usage
To start the application, run the following command:

```bash
cd src
streamlit run dashboard.py
```

## Project Structure
- `data/`: Contains the dataset files for training and validation.
- `src/`: Source code for data preprocessing, model training, and evaluation.
- `dashboard.py`: The Streamlit dashboard to visualize and interact with predictions.
- `requirements.txt`: List of dependencies needed to run the project.

## Dataset
The dataset includes the following features:
- **Temperature**
- **Humidity**
- **Rainfall**
- **Soil Moisture**
- **Wind Speed**
- **Sunlight Hours**
- **Soil pH**
- **Growth Stage**
- **Surrounding Crop Diversity**
- **Crop Type**
- **Disease Type**

Unique disease types include: Root Rot, Leaf Spot, Fungal Wilt, Stem Rot, Rust, Spot, Bacterial Blight, Anthracnose, Blight, Mildew, Powdery Mildew, Downy Mildew, and Wilt.

## Model
The model is trained on historical weather and disease data to predict the most likely disease outbreak and potential treatments. The goal is to enable farmers and stakeholders to take proactive steps in crop protection.

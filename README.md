# Brain-Tumor-Detection

📂 Dataset Details

Dataset Source: Cloned from Raashidh-Rizvi/Brain-Tumor-Detection
.

Dataset Path:

Training data → /content/Brain-Tumor-Detection/data/Training

Testing data → /content/Brain-Tumor-Detection/data/Testing

Classes (4 categories):

Glioma

Meningioma

Pituitary

No Tumor

The dataset is organized into folders by class and is suitable for Keras ImageDataGenerator or PyTorch ImageFolder.

👥 Group Member Roles

Each member was responsible for one preprocessing technique and an EDA visualization:

Handling Missing/Corrupted Images

Checked if all images can be loaded using OpenCV.

Removed corrupted/unreadable files.

Counted and visualized dataset class distribution.

Image Resizing & Standardization

Resized all images to 224x224 pixels.

Standardized image arrays for consistent model input.

Normalization / Scaling

Scaled pixel values from [0,255] → [0,1].

Justification: Improves convergence and stabilizes model training.

Data Augmentation (Feature Engineering)

Applied rotation, flips, zoom, and brightness adjustments.

Purpose: Increase effective dataset size and prevent overfitting.

Outlier / Noise Removal

Identified abnormal or noisy images using pixel intensity histograms.

Removed images that were too bright, blank, or corrupted.

Dimensionality Reduction (EDA Visualization)

Applied PCA on flattened image vectors.

Visualized clustering patterns of tumor vs non-tumor samples.

🛠️ How to Run the Code
1. Clone Repository & Setup
!git clone https://github.com/Raashidh-Rizvi/Brain-Tumor-Detection.git

2. Install Dependencies
pip install tensorflow matplotlib seaborn scikit-learn opencv-python

3. Run Individual Notebooks

Each member has their own notebook in notebooks/ folder:

notebooks/
├── IT24102388_HandlingMissingImages.ipynb
├── IT24104191_ImageResizingStandardization.ipynb
├── IT24102772 _NormalizationScaling.ipynb
├── IT24103670 _DataAugmentation.ipynb
├── IT24103178_OutlierNoiseRemoval.ipynb


4. Run Integrated Group Pipeline

group_pipeline.ipynb integrates all preprocessing steps in sequence:

Load raw dataset

Handle missing/corrupted images

Resize & standardize

Normalize pixel values

Augment data

Remove outliers

Dimensionality reduction visualization

Save cleaned dataset into results/outputs/

📊 Results & Outputs

EDA Visualizations: Stored in results/eda_visualizations/

Logs: Stored in results/logs/ (optional)

Final Processed Dataset: results/outputs/final_dataset.npy

⚖️ Notes for Viva (Progress Review I)

Each member will explain their preprocessing technique, why it was needed, show code + visualization, and interpret results.

Group will present the integrated preprocessing pipeline.

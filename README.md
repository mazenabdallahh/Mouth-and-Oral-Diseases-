# Oral Disease Classification using Deep Learning

This project leverages state-of-the-art Deep Learning architectures to automate the detection and classification of various mouth and oral cavity diseases. By utilizing transfer learning and fine-tuning on specialized medical imagery, the model provides a scalable tool for early oral health diagnosis.

##  About the Dataset
The project uses the **Mouth and Oral Diseases (MOD)** dataset, available on [Kaggle](https://www.kaggle.com/datasets/javedrashid/mouth-and-oral-diseases-mod). It contains high-quality clinical images labeled by expert practitioners across seven categories:

* **CaS:** Canker Sores
* **CoS:** Cold Sores
* **Gum:** Gingivostomatitis
* **MC:** Mouth Cancer
* **OC:** Oral Cancer
* **OLP:** Oral Lichen Planus
* **OT:** Oral Thrush

##  Project Overview
The core objective was to build a robust classifier capable of distinguishing between these seven conditions, which often share similar visual characteristics.

### Key Features:
* **Multi-Model Approach:** Evaluated performance using both **ResNet101** and **EfficientNet-B3** architectures.
* **Transfer Learning:** Utilized ImageNet pre-trained weights to jumpstart the learning process on medical-specific data.
* **Optimization:** Implemented **Mixed Precision training** for faster computation and **Early Stopping** to prevent overfitting.
* **Data Augmentation:** Applied random flips and normalization to improve model generalization.

##  Results
The final model, based on **EfficientNet-B3**, achieved a validation accuracy of approximately **89%**, demonstrating its reliability in identifying complex oral conditions.

| Metric | EfficientNet-B3 Value |
| :--- | :--- |
| **Training Accuracy** | ~85% |
| **Validation Accuracy** | **89.14%** |
| **Validation Loss** | 0.3365 |

## 🛠️ Tech Stack
* **Language:** Python
* **Framework:** PyTorch & Torchvision
* **Tools:** OpenCV, Matplotlib, Seaborn, Scikit-learn
* **Accelerator:** CUDA (NVIDIA GPU)

##  How to Use
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/oral-disease-classification.git](https://github.com/yourusername/oral-disease-classification.git)
    ```
2.  **Install Dependencies:**
    ```bash
    pip install torch torchvision opencv-python matplotlib seaborn scikit-learn tqdm
    ```
3.  **Run the Notebook:**
    Open `Teeth.ipynb` in Jupyter or Google Colab to train the model or run inference on new oral images.


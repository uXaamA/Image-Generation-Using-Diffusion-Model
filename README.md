# Diffusion Model - Image Generation



---

## 🚀 Project Overview

This project implements an **Image Generation system using Diffusion Models**. Diffusion models are state-of-the-art generative models that learn to create images by gradually removing noise from a random input, mimicking a reverse diffusion process.

This assignment demonstrates:

* Forward noise diffusion on clean animal images
* Training a denoising neural network
* Reversing the noise process to reconstruct images from noise

> ⚡ Powered entirely using **PyTorch**

---

## 📁 Dataset

* **Classes Used**: 5 animal classes (e.g., cat, dog, lion, tiger, elephant)
* **Images per class**: 10–20
* **Total training images**: 50–100
* **Image Size**: Resized to 64×64

Example structure:

```
animal_data/
├── cat/
├── dog/
├── lion/
├── tiger/
└── elephant/
```

> ⚠ Ensure the dataset is well-balanced and formatted correctly before training.

---

## 🔧 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python M_Usama_MSDS24045.py --data_path animal_data --mode train
```

* Trains the model
* Saves noisy/clean/predicted images
* Logs loss curve

### 3. Generate New Image from Noise

```bash
python M_Usama_MSDS24045.py --data_path animal_data --mode sample
```

* Starts from pure noise and generates a denoised image
* Output saved in `samples/generated_sample.png`

### 4. Evaluation Notebook

Open `test_single_sample.ipynb` to:

* Load trained model
* Pass noise
* Visualize clean vs predicted image

---

## 🧠 Model Architecture

### Forward Process (Noise Addition)

* Custom function applies Gaussian noise progressively over `T=1000` timesteps
* Ensures images reach near-complete noise

### Denoising Model

* CNN-based model designed to remove noise at each step
* Learns reverse of the forward process

### Loss Function

* Custom **L2 Loss (MSE)** between predicted and actual clean image

---

## 🧪 Training Configuration

| Parameter     | Value        |
| ------------- | ------------ |
| Epochs        | 10           |
| Batch Size    | 32           |
| Image Size    | 64x64        |
| Timesteps (T) | 1000         |
| Optimizer     | Adam         |
| Loss Function | MSE (Custom) |
| Framework     | PyTorch      |

---

## 📈 Results & Outputs

### ✅ Outputs Saved:

* Noisy samples per epoch: `samples/noisy_epochX.png`
* Denoised predictions: `samples/pred_epochX.png`
* Ground truths: `samples/clean_epochX.png`
* Loss curve: `samples/loss_graph.png`

### 📊 Training Loss:

* Loss decreases across epochs, indicating successful learning

* ![loss_graph](https://github.com/user-attachments/assets/eac541ea-b2c4-402c-96a1-636d659745bd)


### 🖼️ Sample Results (Epoch 10):

| Clean Image | Noisy Input | Predicted Output |
| ----------- | ----------- | ---------------- |
| ✅           | ❌           | ✅                |

![clean_epoch1](https://github.com/user-attachments/assets/93f7b581-7072-4346-b50c-c07bf2f509bd)
![noisy_epoch10](https://github.com/user-attachments/assets/e739f5b8-67f2-4e9a-8c44-2bfe16a20a4a)
![pred_epoch10](https://github.com/user-attachments/assets/6e8e1997-6c36-4fbc-9d4a-da4bc9968f9e)
---

## ⚠️ Challenges Faced

* Shape mismatches during noise scheduling
* Debugging across multiple files was hard → switched to single file structure
* Careful reshaping and normalization crucial to success

---

## 💡 Learnings

* Deepened understanding of **forward and reverse diffusion**
* Experienced the challenges of training generative models
* Understood importance of noise scheduling
* Hands-on implementation of noise scheduling + denoising pipeline

---

## 🧪 Folder Structure

```
M_Usama_MSDS24045/
├── M_Usama_MSDS24045.py         # Full pipeline
├── test_single_sample.ipynb     # Evaluation notebook
├── Report.pdf                   # Detailed report + results
├── requirements.txt             # Dependencies
├── saved_models/
│   └── denoise_model.pt         # Trained model
└── samples/
    ├── noisy_epochX.png
    ├── clean_epochX.png
    ├── pred_epochX.png
    └── loss_graph.png
```

---

## 📚 References

* [A Very Short Introduction to Diffusion Models](https://kailashahirwar.medium.com/a-very-short-introduction-to-diffusion-models-a84235e4e9ae)
* [Diffusion Models Made Easy](https://towardsdatascience.com/diffusion-models-made-easy-8414298ce4da)
* [SuperAnnotate on Diffusion Models](https://www.superannotate.com/blog/diffusion-models)
* [CMU Slides on Diffusion Models](https://deeplearning.cs.cmu.edu/S24/document/slides/Diffusion_Models.pdf)

---

## ✍️ Author

**Name**: Muhammad Usama
MS Data Science 
**Institute**:Information Technology University Lahore
**Profession**: Technical Data Anlayst Haier Pakistan 

---

> *This project was completed as part of the Deep Learning coursework under academic guidance. All results are reproducible and open for inspection.*

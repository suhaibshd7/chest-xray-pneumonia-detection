# Chest X-ray Pneumonia Detection using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A deep learning model that detects pneumonia from chest X-ray images with 95.20% AUC-ROC and 98.72% recall.**

![Demo](images/gradcam_visualizations.png)
*Grad-CAM visualization showing model attention on pneumonia-affected lung regions*

---

## 🎯 Project Overview

This project develops an AI-powered diagnostic assistant for pneumonia detection from chest X-ray images. The model serves as a screening tool to help radiologists identify potential pneumonia cases quickly and accurately.

### Key Results
- **98.72% Recall** - Catches 99% of pneumonia cases (critical for healthcare screening)
- **95.20% AUC-ROC** - Excellent discrimination between normal and pneumonia X-rays
- **85.26% Accuracy** - Strong overall performance on test set

### Clinical Significance
Early and accurate pneumonia detection is crucial for patient outcomes. This AI model can:
- Accelerate radiologist workflow by pre-screening X-rays
- Reduce false negatives in busy clinical settings
- Provide visual explanations (Grad-CAM) for model predictions
- Serve as a second opinion for junior radiologists

---

## 📊 Results

### Performance Metrics

| Metric | Score | Clinical Interpretation |
|--------|-------|------------------------|
| **Accuracy** | 85.26% | Overall correctness across all predictions |
| **Precision** | 81.57% | When model predicts pneumonia, it's correct 82% of the time |
| **Recall** | 98.72% | Detects 99% of actual pneumonia cases ✅ |
| **F1 Score** | 0.8933 | Balanced harmonic mean of precision and recall |
| **AUC-ROC** | 0.9520 | **Excellent discrimination capability** ⭐ |

### Confusion Matrix

```
                Predicted
              NORMAL  PNEUMONIA
    NORMAL      147       87      (62.8% specificity)
Actual
  PNEUMONIA       5      385      (98.7% sensitivity)
```

**Clinical Trade-off Analysis:**
- ✅ **High Recall Priority:** Only 5 missed pneumonia cases (1.3% false negative rate)
- ⚠️ **False Positives:** 87 false alarms (acceptable for screening tool)
- 💡 **Design Decision:** Prioritized catching all pneumonia cases over minimizing false positives, as missing pneumonia is clinically more dangerous than a false alarm

---

## 🏗️ Technical Architecture

### Model Design
- **Base Architecture:** ResNet-18 (18-layer Residual Network)
- **Transfer Learning:** Pre-trained on ImageNet, fine-tuned on chest X-rays
- **Output:** Binary classifier (Normal vs Pneumonia)
- **Parameters:** ~11 million trainable parameters

**Why ResNet-18?**
1. ✅ Faster training compared to deeper models (ResNet-50, ResNet-101)
2. ✅ Less prone to overfitting on small medical datasets (~5K images)
3. ✅ Proven effectiveness on medical imaging tasks
4. ✅ Good balance between performance and computational efficiency

### Training Strategy

**1. Data Augmentation**
- Random horizontal flip (mirrors X-ray)
- Random rotation (±10 degrees)
- Color jitter (brightness & contrast adjustment)

**Purpose:** Increases data diversity and reduces overfitting

**2. Class Imbalance Handling**
- Training set: 3,875 Pneumonia vs 1,341 Normal (2.9:1 ratio)
- Solution: Weighted loss function (weight=3.0 for Normal, weight=1.0 for Pneumonia)
- This prevents the model from simply predicting "Pneumonia" for everything

**3. Optimization**
- **Optimizer:** Adam with learning rate 0.0001
- **Regularization:** Weight decay (L2) = 0.0001
- **Early Stopping:** Patience = 5 epochs (stops if no improvement)
- **Batch Size:** 32 images per batch
- **Training Duration:** 12 epochs (stopped early to prevent overfitting)

### Dataset Details
- **Source:** [Chest X-Ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Training Set:** 5,216 images (1,341 Normal, 3,875 Pneumonia)
- **Validation Set:** 16 images (8 Normal, 8 Pneumonia)
- **Test Set:** 624 images (234 Normal, 390 Pneumonia)
- **Image Format:** Grayscale chest X-rays, resized to 224×224 pixels

---

## 🔍 Model Interpretability - Grad-CAM Analysis

Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes which regions of the X-ray the model focuses on when making predictions. This transparency is essential for clinical trust and validation.

![Grad-CAM Examples](images/gradcam_visualizations.png)

### Observations

**✅ Pneumonia Cases (Correct Focus)**
- Model correctly attends to lung infiltrates and opacity regions
- Highlights consolidation patterns typical of pneumonia
- Focus aligns with radiological findings

**⚠️ Normal Cases (Artifact Detection)**
- Some attention on image edges and corners (outside lung regions)
- Suggests the model may be learning subtle artifacts or image boundaries
- Despite this, model still achieves good performance on normal cases

**Validation Strategy:**
Using Grad-CAM, I validated that while the model does learn some image artifacts (evident in normal X-rays focusing on edges), it primarily attends to clinically relevant lung regions in pneumonia cases. This demonstrates the model is learning meaningful features, not just dataset biases.

---

## 🛠️ Tech Stack

- **Deep Learning Framework:** PyTorch 2.0+
- **Computer Vision:** torchvision, OpenCV
- **Data Processing:** NumPy, PIL
- **Visualization:** Matplotlib, Seaborn
- **Evaluation Metrics:** scikit-learn
- **Experiment Tracking:** TensorBoard
- **Development Environment:** Kaggle Notebooks (GPU: Tesla P100)

---

## 📁 Project Structure

```
chest-xray-pneumonia-detection/
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── notebook/
│   └── chest_xray_pneumonia.ipynb # Complete training & evaluation notebook
├── images/
│   ├── confusion_matrix.png        # Model performance visualization
│   ├── roc_curve.png               # ROC curve plot
│   └── gradcam_visualizations.png  # Interpretability examples
├── .gitignore                      # Git ignore rules
└── LICENSE                         # MIT License
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU (recommended) or CPU
- 8GB RAM minimum

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/suhaibshd7/chest-xray-pneumonia-detection.git
cd chest-xray-pneumonia-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download dataset**
- Visit [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Download and extract to your working directory

4. **Run the notebook**
- Open `notebook/chest_xray_pneumonia.ipynb`
- Run all cells to train and evaluate the model

---

## 📈 Training Details

**Training Progress:**
- Initial validation accuracy: 62.50% (Epoch 1)
- Best validation accuracy: 100.00% (Epoch 7)
- Final training accuracy: 99.16% (Epoch 12)
- Best validation loss: 0.0414 (Epoch 7)

**Early Stopping:**
Training automatically stopped at epoch 12 after no improvement for 5 consecutive epochs, preventing overfitting while maintaining strong validation performance.

---

## 💡 Technical Decisions & Rationale

### Q: Why ResNet-18 over larger architectures?
**A:** With only 5,216 training images, deeper models like ResNet-50 or ResNet-101 would likely overfit. ResNet-18 provides sufficient capacity for binary classification while training faster and generalizing better on limited medical data.

### Q: What about the false positives?
**A:** 87 false positives (37% false positive rate among normals) could be improved by:
- Trying different architectures (DenseNet, EfficientNet)
- Increasing penalty for false positives in loss function
- Collecting more diverse normal X-ray data

However, for a **screening tool**, high recall (98.72%) is prioritized over precision. False positives are reviewed by radiologists anyway, so the cost of a false alarm is lower than missing a pneumonia case (false negative).

### Q: How did you validate the model isn't just learning artifacts?
**A:** Used Grad-CAM interpretability analysis:
- ✅ Pneumonia images: Model focuses on lung infiltrates (correct)
- ⚠️ Normal images: Some attention to image borders (artifact detection)
- 💡 Conclusion: While model does learn some artifacts, primary features are clinically relevant

Further validation would involve testing on external datasets from different hospitals/X-ray machines.

---

## 🎓 What I Learned

### Technical Skills Developed
- **Transfer Learning:** Adapting pre-trained CNNs to medical imaging domain
- **Class Imbalance:** Implementing weighted loss functions for skewed datasets
- **Model Interpretability:** Using Grad-CAM for explainable AI in healthcare
- **Medical Imaging:** Understanding chest X-ray preprocessing and augmentation
- **PyTorch:** Custom dataset classes, training loops, and model evaluation

### Domain Knowledge Gained
- **Medical AI Ethics:** Balance between false positives and false negatives
- **Clinical Workflow:** How AI tools integrate into radiologist's daily practice
- **Healthcare Priorities:** Recall > Precision for screening applications

---

## 🔮 Future Improvements

### Short-term Enhancements
1. **External Testing:** Evaluate on X-rays from different hospitals to test generalization
2. **Address Artifacts:** Investigate why model focuses on image borders in normal cases
3. **Threshold Tuning:** Adjust classification threshold to optimize recall vs. precision trade-off

### Long-term Developments
1. **Multi-class Classification:** Detect specific pneumonia types (bacterial vs. viral)
2. **Localization:** Add bounding boxes or segmentation masks for affected lung regions
3. **Ensemble Methods:** Combine multiple models for improved robustness

---

## 📝 Reflections

### What Worked Well ✅
- Transfer learning significantly reduced training time and improved performance
- Weighted loss function effectively addressed class imbalance
- High recall (98.72%) achieved through strategic loss weighting
- Grad-CAM provided actionable insights into model behavior
- Early stopping prevented overfitting despite small dataset

### Challenges Encountered ⚠️
- Tiny validation set (16 images) caused unstable validation metrics
- Model learned some image artifacts (edge detection in normal X-rays)
- False positive rate (37%) may be too high for deployment without modification
- Limited dataset diversity (single source) raises generalization concerns

### Key Takeaway 💡
Building medical AI requires balancing **technical performance** with **clinical priorities**. A 98.72% recall is excellent for a screening tool, but deployment would require:
- Larger validation studies
- Regulatory approval (FDA clearance)
- Integration with hospital IT systems
- Continuous monitoring for performance drift

This project demonstrates end-to-end ML pipeline development and domain-aware decision-making essential for healthcare AI.

---

## 📚 References & Resources

### Dataset
- **Source:** Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018). "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification." Mendeley Data, v2.
- **Kaggle:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

### Technical Papers
- **Grad-CAM:** Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV 2017.
- **ResNet:** He et al. (2016). "Deep Residual Learning for Image Recognition." CVPR 2016.

---

## 👤 About Me

**Suhaib Shdefat**  
Medical Imaging Student | Aspiring AI Researcher

I'm a bachelor's student in Medical Imaging at The Hashemite University, Jordan, passionate about applying artificial intelligence to healthcare challenges. My goal is to bridge the gap between medical imaging technology and AI research.

**Research Interests:**
- Medical image analysis and computer-aided diagnosis
- Deep learning for healthcare applications
- Explainable AI in clinical settings

**Connect with Me:**
- 🐙 GitHub: [github.com/suhaibshd7](https://github.com/suhaibshd7)
- 📧 Email: suhaib.shdefat@gmail.com
- 🎓 University: The Hashemite University, Jordan
- 📍 Location: Amman, Jordan

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

**Note:** This is a research/educational project. Any clinical deployment would require proper validation, regulatory approval, and adherence to medical device regulations.

---

**⭐ If you found this project helpful, please star the repository!**

---

*Last Updated: November 2024*

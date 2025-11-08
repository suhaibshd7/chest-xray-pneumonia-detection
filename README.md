# Chest X-ray Pneumonia Detection using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A deep learning model that detects pneumonia from chest X-ray images with 95.28% AUC-ROC and 99.49% recall.**

![Demo](images/gradcam_visualizations.png)
*Grad-CAM visualization showing model attention on pneumonia-affected lung regions*

---

## 🎯 Project Overview

This project develops an AI-powered diagnostic assistant for pneumonia detection from chest X-ray images. The model serves as a screening tool to help radiologists identify potential pneumonia cases quickly and accurately.

### Key Results
- **99.49% Recall** - Catches nearly all pneumonia cases (critical for healthcare screening)
- **95.28% AUC-ROC** - Excellent discrimination between normal and pneumonia X-rays
- **81.89% Accuracy** - Strong overall performance on test set

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
| **Accuracy** | 81.89% | Overall correctness across all predictions |
| **Precision** | 77.76% | When model predicts pneumonia, it's correct 78% of the time |
| **Recall** | 99.49% | Detects 99.5% of actual pneumonia cases ✅ |
| **F1 Score** | 0.8729 | Balanced harmonic mean of precision and recall |
| **AUC-ROC** | 0.9528 | **Excellent discrimination capability** ⭐ |

### Confusion Matrix

```
                Predicted
              NORMAL  PNEUMONIA
    NORMAL      123      111      (52.6% specificity)
Actual
  PNEUMONIA       2      388      (99.5% sensitivity)
```

**Clinical Trade-off Analysis:**
- ✅ **High Recall Priority:** Only 2 missed pneumonia cases (0.5% false negative rate)
- ⚠️ **False Positives:** 111 false alarms (acceptable for screening tool)
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
- Some attention on bottom-right corner of images (outside lung regions)
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
chest-xray-pneumonia/
├── README.md                       # Project documentation
├── notebooks/
│   └── pneumonia_classifier.ipynb  # Main Kaggle notebook
├── images/
│   ├── confusion_matrix.png        # Model performance visualization
│   ├── roc_curve.png               # ROC curve plot
│   ├── gradcam_visualizations.png  # Interpretability examples
│   └── metrics_summary.png         # Overall metrics bar chart
├── checkpoints/
│   └── best_model.pth              # Trained model weights
├── requirements.txt                # Python dependencies
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
- Download and extract to `data/chest_xray/`

### Training

```bash
# Open the Kaggle notebook or run training script
# Training takes approximately 1-2 hours on GPU
```

### Evaluation

```bash
# Evaluation generates:
# - Confusion matrix
# - ROC curve
# - Grad-CAM visualizations
# - Performance metrics
```

---

## 📈 Training Details

**Training Progress:**
- Initial validation accuracy: 75.00% (Epoch 1)
- Best validation accuracy: 100.00% (Epoch 7)
- Final training accuracy: 99.23% (Epoch 12)
- Best validation loss: 0.1423 (Epoch 7)

**Early Stopping:**
Training automatically stopped at epoch 12 after no improvement for 5 consecutive epochs, preventing overfitting while maintaining strong validation performance.

---

## 💡 Technical Decisions & Rationale

### Q: Why ResNet-18 over larger architectures?
**A:** With only 5,216 training images, deeper models like ResNet-50 or ResNet-101 would likely overfit. ResNet-18 provides sufficient capacity for binary classification while training faster and generalizing better on limited medical data.

### Q: How would you deploy this in a hospital setting?
**A:** The model would integrate into the radiology workflow as a pre-screening tool:
1. When radiologist opens a chest X-ray in PACS (Picture Archiving and Communication System)
2. AI automatically analyzes the image in background
3. Prediction + Grad-CAM heatmap displays alongside the X-ray
4. Radiologist reviews AI suggestion and makes final diagnosis
5. AI serves as second opinion, NOT final arbiter

This accelerates workflow while maintaining human oversight for safety.

### Q: What about the false positives?
**A:** 111 false positives (47.4% false positive rate among normals) could be improved by:
- Trying different architectures (DenseNet, EfficientNet)
- Increasing penalty for false positives in loss function
- Collecting more diverse normal X-ray data

However, for a **screening tool**, high recall (99.49%) is prioritized over precision. False positives are reviewed by radiologists anyway, so the cost of a false alarm is lower than missing a pneumonia case (false negative).

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
- **Regulatory Awareness:** Importance of interpretability for FDA approval
- **Healthcare Priorities:** Recall > Precision for screening applications

### Software Engineering Practices
- **Experiment Tracking:** Using TensorBoard for monitoring training
- **Model Checkpointing:** Saving best models based on validation performance
- **Code Organization:** Structuring ML projects with clear separation of concerns
- **Reproducibility:** Setting random seeds, documenting hyperparameters

---

## 🔮 Future Improvements

### Short-term Enhancements
1. **Expand Validation Set:** Current validation set (16 images) is too small for reliable monitoring
2. **External Testing:** Evaluate on X-rays from different hospitals to test generalization
3. **Address Artifacts:** Investigate why model focuses on image borders in normal cases
4. **Threshold Tuning:** Adjust classification threshold to optimize recall vs. precision trade-off

### Long-term Developments
1. **Multi-class Classification:** Detect specific pneumonia types (bacterial vs. viral)
2. **Localization:** Add bounding boxes or segmentation masks for affected lung regions
3. **Ensemble Methods:** Combine multiple models for improved robustness
4. **Uncertainty Quantification:** Provide confidence intervals for predictions
5. **Web Deployment:** Create Streamlit or Flask app for easy clinical testing
6. **Mobile Optimization:** Deploy lightweight model for point-of-care devices

### Research Directions
1. **Comparison Study:** Benchmark against other architectures (DenseNet, Vision Transformers)
2. **Attention Mechanisms:** Incorporate explicit attention layers for better interpretability
3. **Few-shot Learning:** Adapt model to rare pneumonia subtypes with limited data
4. **Clinical Trial:** Prospective study comparing AI + Radiologist vs. Radiologist alone

---

## 📝 Reflections

### What Worked Well ✅
- Transfer learning significantly reduced training time and improved performance
- Weighted loss function effectively addressed class imbalance
- High recall (99.49%) achieved through strategic loss weighting
- Grad-CAM provided actionable insights into model behavior
- Early stopping prevented overfitting despite small dataset

### Challenges Encountered ⚠️
- Tiny validation set (16 images) caused unstable validation metrics
- Model learned some image artifacts (edge detection in normal X-rays)
- False positive rate (47%) may be too high for deployment without modification
- Limited dataset diversity (single source) raises generalization concerns

### Key Takeaway 💡
Building medical AI requires balancing **technical performance** with **clinical priorities**. A 99.49% recall is excellent for a screening tool, but deployment would require:
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
- **Transfer Learning:** Yosinski et al. (2014). "How transferable are features in deep neural networks?" NIPS 2014.

### Learning Resources
- **PyTorch Documentation:** https://pytorch.org/docs/
- **Transfer Learning Tutorial:** https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- **Medical Imaging with Deep Learning:** https://www.coursera.org/learn/ai-for-medical-diagnosis

---

## 👤 About Me

**Suhaib Shdefat**  
Medical Imaging Student | Aspiring AI Researcher

I'm a bachelor's student in Medical Imaging at The Hashemite University, Jordan, passionate about applying artificial intelligence to healthcare challenges. My goal is to bridge the gap between medical imaging technology and AI research.

**Research Interests:**
- Medical image analysis and computer-aided diagnosis
- Deep learning for healthcare applications
- Explainable AI in clinical settings
- Transfer learning for limited medical data

**Academic Goals:**
- **Short-term:** Secure a Master's scholarship in AI/Medical Imaging
- **Long-term:** Become an AI researcher and university professor, contributing to the advancement of medical imaging technology

**Why This Project?**
As a medical imaging student, I understand the critical role of accurate and timely diagnosis in patient care. This project combines my academic background with AI skills to create a tool that could genuinely impact radiologist workflows. The focus on interpretability (Grad-CAM) reflects the importance of trust and transparency in medical AI.

**Current Focus:**
Building a portfolio of AI projects in medical imaging to demonstrate research potential for graduate school applications. Exploring applications of deep learning in radiology, pathology, and diagnostic imaging.

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

## 🙏 Acknowledgments

- The Hashemite University Medical Imaging Department for foundational knowledge
- Kaggle community for providing the chest X-ray dataset
- PyTorch team for excellent deep learning framework and documentation
- Medical imaging researchers whose work inspired this project

---

## 📊 Citation

If you use this project in your research or work, please cite:

```bibtex
@misc{shdefat2024pneumonia,
  author = {Shdefat, Suhaib},
  title = {Chest X-ray Pneumonia Detection using Deep Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/suhaibshd7/chest-xray-pneumonia-detection}
}
```

---

**⭐ If you found this project helpful for learning medical AI or building your own portfolio, please star the repository!**

---

*Last Updated: November 2024*

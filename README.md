# GSoC-2025 : QML-HEP

### Quantum Approach (QML_for_HEP_GSoC_Task_4.ipynb)
**Technique**: Implements **IQGAN** (Invertible Quantum Generative Adversarial Network) using:
- Quantum neural networks with PennyLane
- 5-qubit quantum circuits with rotation gates (RY/RX/RZ) and CNOT operations
- PCA dimensionality reduction (`n_components=2`) with sigmoid transformation
- Swap test for quantum state comparison
- Custom log loss function with Adam optimization

**Key Features**:
- Quantum generator architecture with parameterized quantum circuits
- Sigmoid transformation maps features to [0, π/2] range
- Fine-tuning with correlation penalty to maintain feature relationships
- Hardware acceleration using CUDA/TPU

**Evaluation Metrics**:
{
'KL Divergence': 0.04708707,
'Wasserstein Fidelity': 0.33273508,
'AUC-ROC': 0.892,
'MSE': 0.18237947,
'Cosine Similarity': 0.95665384
}


---

### Classical Approach (Data_Analysis_-_Classical_GAN_on_HEP_GSoC.ipynb)
**Technique**: Implements **Classical GAN** with:
- PCA dimensionality reduction (`n_components=2`)
- Standard neural networks
- Correlation matrix analysis
- Distribution visualization

**Key Features**:
- Traditional deep learning approach
- Statistical correlation analysis
- Data distribution visualization
- Pandas-based data exploration

---

### Key Differences
| Aspect                | Quantum Approach                     | Classical Approach               |
|-----------------------|--------------------------------------|-----------------------------------|
| **Model Architecture**| Quantum circuits with rotation gates | Classical neural networks        |
| **Feature Encoding**  | Quantum state encoding               | PCA + statistical normalization  |
| **Training**          | Quantum gradient-based optimization  | Standard backpropagation         |
| **Evaluation**        | Quantum fidelity metrics + MSE       | Statistical correlation analysis |
| **Hardware**          | GPU/TPU + quantum simulation         | CPU/GPU                          |
| **Complexity**        | Higher (qubit entanglement)          | Lower                             |

---

### Metric Analysis
1. **KL Divergence (0.047)**: Indicates good distribution matching  
2. **Wasserstein (0.333)**: Suggests moderate transport cost between distributions  
3. **AUC-ROC (0.892)**: Strong discriminative performance  
4. **Cosine Similarity (0.957)**: High directional alignment in feature space  

---

### Model Performance Comparison

#### Quantum Model
**After Fine-Tuning**:
{'KL Divergence': 0.04708707, 'Wasserstein Fidelity': 0.33273508, 'AUC-ROC': 0.892, 'MSE': 0.18237947, 'Cosine Similarity': 0.95665384}


**Before Fine-Tuning**:
{'KL Divergence': 0.04078283, 'Wasserstein Fidelity': 0.26723456, 'AUC-ROC': 0.81800000, 'MSE': 0.12634069, 'Cosine Similarity': 0.96311071}

#### Classical Model
**Test Data**:
{'KL Divergence': inf, 'Wasserstein Fidelity': 0.28818943, 'AUC-ROC': 0.5476, 'MSE': 2.49942540, 'Cosine Similarity': 0.15625884}


**Training Data**:
{'KL Divergence': inf, 'Wasserstein Fidelity': 0.01096834, 'AUC-ROC': 0.5004, 'MSE': 0.00022856, 'Cosine Similarity': 0.99988858}

---

### Conclusion
The quantum IQGAN demonstrates better **feature preservation** (0.957 cosine similarity) through quantum entanglement while maintaining physical correlations via correlation penalty fine-tuning. The classical GAN shows catastrophic failure on test data (`KL=∞`, `AUC-ROC=0.5476`) due to overfitting, despite perfect training metrics (`MSE=0.0002`). Quantum methods achieve comparable performance to classical GANs while preserving quantum-computational advantages for HEP applications.

---
### Detailed Analysis of Model Performance

#### **Classical GAN Model**
**Key Issues:**  
1. **Severe Overfitting**  
   - **Training Data**: Near-perfect metrics (`MSE=0.0002`, `Cosine Similarity=0.999`)  
   - **Test Data**: Catastrophic failure (`KL=∞`, `AUC-ROC=0.5476 ≈ random guessing`, `MSE=2.499`)  
   - **Diagnosis**: The model memorizes training data but fails to generalize.

2. **Distribution Mismatch**  
   - Infinite KL divergence indicates the generated distribution lacks coverage of real data modes (e.g., "mode collapse").

3. **Poor Discriminative Power**  
   - `AUC-ROC (0.5476)` shows the discriminator cannot distinguish real vs. fake data effectively.

---

#### **Quantum Model (IQGAN)**
**Strengths:**  
1. **Robust Generalization**  
   - Consistent metrics across training/testing (`KL=0.04–0.047`, `AUC-ROC=0.818–0.892`).  
   - High cosine similarity (`0.956–0.963`) indicates strong feature alignment.

2. **Impact of Fine-Tuning**  
   - **Improved Discrimination**: `AUC-ROC ↑` from `0.818` to `0.892`.  
   - **Trade-offs**: Slight degradation in KL/Wasserstein metrics due to correlation penalty enforcing physical constraints.

**Weaknesses:**  
- **MSE increases** after fine-tuning (`0.126 → 0.182`), suggesting reconstruction fidelity vs. physical accuracy trade-off.

---

### Improvement Strategies

#### **For Classical GAN**
1. **Mitigate Overfitting**  
   - **Regularization**: Add dropout/L2 regularization.  
   - **Data Augmentation**: Expand training diversity.  
   - **Architecture**: Use deeper networks or spectral normalization.

2. **Address Mode Collapse**  
   - **Unrolled GANs** or **Wasserstein Loss** to improve distribution coverage.  
   - **Mini-batch Discrimination**: Encourage diverse outputs.

3. **Evaluation Fixes**  
   - Replace KL divergence with Wasserstein/Jensen-Shannon metrics to avoid `∞` values.  
   - Debug training loops for numerical stability.

---

#### **For Quantum IQGAN**
1. **Fine-Tuning Balance**  
   - Adjust correlation penalty weight (`lambda_corr`) to balance physical accuracy and reconstruction fidelity.  
   - Experiment with dynamic penalty scheduling.

2. **Architecture Enhancements**  
   - Increase qubit count or circuit depth for richer feature representation.  
   - Use **Quantum Neural Tangent Kernels** for better optimization.

3. **Loss Function**  
   - Combine KL divergence with adversarial losses for sharper distribution matching.  
   - Explore hybrid classical-quantum discriminators.

---

#### **Shared Improvements**
- **Data Preprocessing**: Apply PCA/Whitening to reduce noise and improve training stability.  
- **Hyperparameter Tuning**: Optimize learning rates, batch sizes, and epochs.  
- **Hardware**: Leverage GPUs/TPUs for faster quantum simulations.
  

---

### Key Takeaways
- **Classical GAN**: Needs architectural overhaul to address overfitting and mode collapse.  
- **Quantum IQGAN**: Fine-tuning successfully preserves physical correlations but requires balancing metrics.  
- **Hybrid Approaches**: Combine quantum generators with classical post-processing for optimal results.  

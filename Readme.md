# UQ Group

## Traffic flow learning


## Physics-informed Neural Operators

Folder of "PI-DCON" contains the implementation of Physics-informed discretization-independent deep compositional operator network. If you think that the work of the PI-DCON is useful in your research, please consider citing our paper in your manuscript:
```
@article{zhong2024physics,
  title={Physics-informed discretization-independent deep compositional operator network},
  author={Zhong, Weiheng and Meidani, Hadi},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={431},
  pages={117274},
  year={2024},
  publisher={Elsevier}
}
```

Folder of "PI-GANO" contains the implementation of Physics-Informed Geometry-Aware Neural Operator. If you think that the work of the PI-GANO is useful in your research, please consider citing our paper in your manuscript:
```
@article{zhong2025physics,
  title={Physics-Informed Geometry-Aware Neural Operator},
  author={Zhong, Weiheng and Meidani, Hadi},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={434},
  pages={117540},
  year={2025},
  publisher={Elsevier}
}
```
## How to Obtain Pretrained Models such as figCovnNet and Geometry-Aware Operator Transformer

**figCovnNet:**  
There are no direct references in the provided search results for a model named "figCovnNet." It is possible that the name is misspelled or not widely recognized in the literature or open-source repositories as of now. If you meant a different model (such as "FigConvNet" or a similarly named architecture), please clarify or provide more context.

**Geometry-Aware Operator Transformer (GAOT):**

- The Geometry-Aware Operator Transformer (GAOT) is a recent model designed for learning partial differential equations (PDEs) on arbitrary domains, combining geometry embeddings with transformer architectures for high accuracy and efficiency[5].
- The official paper is available on arXiv, but the search results do not provide a direct link to a code repository or pretrained model download for GAOT[5]. Typically, authors release code and pretrained weights on platforms like GitHub, but you may need to check the arXiv paper for supplementary materials or contact the authors directly for access[5].

**Related Geometry-Aware Models:**

- **PI-GANO (Physics-Informed Geometry-Aware Neural Operator):**  
  The official implementation is available on GitHub. To use their pretrained models:
  - Download the dataset and pretrained models into the specified folders.
  - Install the required Python packages.
  - Run the provided scripts in "test" mode to evaluate the pretrained models or "train" mode to retrain them[1].

- **GTA (Geometry-Aware Attention Mechanism for Multi-View Transformers):**  
  The official code is available on GitHub. For pretrained models:
  - Follow the repository instructions to set up the environment and download datasets.
  - The authors mention that pretrained models for the MSN-Hard dataset will be uploaded soon, so check back on their repository or contact the maintainers for updates[2].

**General Steps to Obtain Pretrained Models:**

1. **Find the Official Repository:**  
   Search for the model's name on GitHub or the project page linked in the paper.
2. **Check for Pretrained Weights:**  
   Look for a "releases," "models," or "checkpoints" section in the repository, or instructions in the README for downloading pretrained weights.
3. **Contact the Authors:**  
   If pretrained models are not publicly available, email the corresponding author (contact info is usually in the arXiv or conference paper).
4. **Reproduce Training:**  
   If no pretrained model is available, you may need to train the model from scratch using the provided code and datasets.

**Summary Table**

| Model Name                         | Pretrained Model Availability | How to Obtain                |
|-------------------------------------|------------------------------|------------------------------|
| figCovnNet                         | Not found in search results  | Clarify or check spelling    |
| Geometry-Aware Operator Transformer | Not directly available       | Check arXiv paper or contact authors[5] |
| PI-GANO                            | Available                    | Download from GitHub, follow instructions[1] |
| GTA (Geometry-Aware Transformer)    | To be uploaded               | Check GitHub, follow updates[2] |

If you need instructions for a specific model not listed here, please provide the correct name or a link to the relevant paper or repository.

Citations:
[1] https://github.com/WeihengZ/PI-GANO
[2] https://github.com/autonomousvision/gta
[3] https://arxiv.org/html/2411.00164v1
[4] https://proceedings.neurips.cc/paper_files/paper/2024/file/030cf55d506515f39c042e63ba0376dd-Paper-Conference.pdf
[5] https://arxiv.org/abs/2505.18781
[6] https://ojs.aaai.org/index.php/AAAI/article/view/28323/28635
[7] https://openreview.net/forum?id=exGOXqxR0L
[8] https://arxiv.org/pdf/2310.03059.pdf


## How Physics-Informed Data Improves Darcy Flow Predictions Compared to Traditional Methods

### 1. Incorporation of Physical Laws Enhances Accuracy and Generalization

Physics-informed models, such as Physics-Informed Neural Networks (PINNs), embed the governing equations of Darcy flow directly into the neural network’s training process. This means the model is not just fitting to data, but is also constrained to produce solutions that satisfy the physical laws (e.g., conservation of mass, Darcy’s law) everywhere in the domain. This dual constraint leads to:
- Improved predictive accuracy, especially in regions where data is sparse or noisy, because the model cannot produce physically implausible solutions[1][4][5].
- Better generalization to unseen geometries, boundary conditions, or material heterogeneities, as the model "learns" the underlying physics, not just the data distribution[1][4].

### 2. Data Efficiency and Robustness to Sparse Measurements

Traditional numerical methods (like FEM or FDM) require a full specification of the domain and parameters for each new simulation, and their accuracy is tied to mesh resolution and computational cost. In contrast, physics-informed models can:
- Reconstruct full flow fields from limited or sparse measurements by leveraging the embedded physical constraints[4][5].
- Reduce the need for dense labeled data, since the physics-based loss supplements the data-driven loss, allowing meaningful predictions even with fewer data points[4].

### 3. Computational Efficiency After Training

- Once trained, physics-informed neural networks can provide rapid predictions for new permeability fields or boundary conditions, orders of magnitude faster than running a new FEM simulation for each scenario[2][8].
- This makes them particularly attractive for applications requiring repeated or real-time predictions, such as uncertainty quantification or optimization[2][8].

### 4. Reduction of Numerical Artifacts

- Traditional numerical methods can suffer from numerical diffusion, dispersion, and other artifacts, especially on coarse meshes or complex geometries.
- Physics-informed models, by learning from high-fidelity data and enforcing physical constraints, can mitigate some of these issues and produce smoother, more physically consistent solutions[5].

### 5. Quantitative Improvements

- Studies have shown that adding physics constraints to deep neural network training can increase the accuracy of parameter estimation (e.g., hydraulic conductivity) by up to 50% compared to purely data-driven models[5].
- Physics-informed approaches often outperform both traditional machine learning and purely numerical methods in predicting pressure and flow fields, especially in heterogeneous or poorly characterized media[1][5].

### Summary Table

| Aspect                  | Physics-Informed Models                    | Traditional Numerical Methods (FEM, FDM)         |
|-------------------------|--------------------------------------------|--------------------------------------------------|
| Uses physical laws      | Yes (in loss function)                     | Yes (in discretized equations)                   |
| Data requirement        | Can work with sparse data                  | Requires full specification for each simulation   |
| Generalization          | Good to unseen domains/conditions          | Each case solved independently                   |
| Computational speed     | Fast inference after training              | Slow for each new scenario                       |
| Handles noise/sparsity  | Robust due to physics constraints          | Sensitive to input data quality                  |
| Numerical artifacts     | Fewer due to learned representations       | Can suffer from diffusion, dispersion, etc.      |
| Accuracy (with physics) | Up to 50% better in parameter estimation   | High, but depends on mesh and solver             |

---

**In summary:**  
Physics-informed models improve Darcy flow predictions by enforcing physical laws during training, enabling accurate, generalizable, and efficient predictions even with limited data. They can outperform traditional methods in speed and robustness, especially for repeated or real-time applications, while maintaining or even enhancing predictive accuracy[1][4][5][8].

Citations:
[1] https://pubs.aip.org/aip/pof/article/37/1/013605/3329262/Physics-informed-radial-basis-function-neural
[2] https://www.sciencedirect.com/science/article/abs/pii/S0021999123000141
[3] https://www.sciencedirect.com/science/article/pii/S0957417424015458
[4] https://arxiv.org/html/2406.19939v1
[5] https://www.extrica.com/article/23174
[6] https://henry.baw.de/bitstreams/b59b4542-e4ed-437f-9257-7e9c03f3f100/download
[7] https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023WR035064
[8] https://arxiv.org/html/2401.02363v1

---

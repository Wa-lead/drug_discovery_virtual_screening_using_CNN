# Drug Discovery Virtual Screening using Chemception

A CNN-based virtual screening pipeline that predicts biological activity of chemical compounds using molecular image representations, built on InceptionV3 transfer learning with custom molecular featurization.

## Architecture

The model uses a modified InceptionV3 backbone with transfer learning for binary activity classification.

<div align="center">
<img src="images/model_arch.png" width="700"/>
</div>

## Pipeline

```
SMILES → RDKit Mol → 4-Channel Featurization → Weighted Fusion (3-Ch RGB) → InceptionV3 → Active/Inactive
```

1. **Data preparation** — Parse SMILES strings into RDKit molecule objects ([`data/data_prepare.ipynb`](data/data_prepare.ipynb))
2. **Featurization** — Convert molecules to 4-channel images encoding bond order, atom type, hybridization, and Gasteiger charges
3. **Weighted fusion** — Fuse 4 channels into 3 (RGB) for InceptionV3 compatibility
4. **Training** — Fine-tune InceptionV3 with BOHB hyperparameter optimization
5. **Evaluation** — ROC-AUC on held-out test set
6. **Inference** — Gradio app with Grad-CAM interpretability

## Featurization

Each molecule is rendered as a multi-channel image where pixel values encode chemical properties:

<div align="center">
<img src="images/Featurizer.png" width="700"/>
</div>

The 4 channels are fused into 3 via weighted combination to leverage ImageNet-pretrained weights:

<div align="center">
<img src="images/after_fusion.png" width="700"/>
</div>

## Results

**ROC-AUC: 0.69** on the PubChem AID686978 bioassay dataset.

<div align="center">
<img src="images/ROC.png" width="400"/>
</div>

### Kernel Visualization

<div align="center">
<img src="images/kernel.png" width="500"/>
</div>

### Grad-CAM

Highlights which molecular substructures drive the model's prediction:

<div align="center">
<img src="images/GradCAM.png" width="400"/>
</div>

## Hyperparameter Optimization

Hyperparameters were tuned using BOHB (Bayesian Optimization + HyperBand) via the `hpbandster` library.

<div align="center">
<img src="images/hpbandster.png" width="500"/>
</div>

| Hyperparameter | Value  |
|---------------|--------|
| dense_layers  | 1      |
| dropout       | 0.1    |
| lr            | 0.0001 |
| neurons       | 128    |

## How to Run

```bash
# Clone
git clone https://github.com/Wa-lead/drug_discovery_virtual_screening_using_CNN.git
cd drug_discovery_virtual_screening_using_CNN

# Install
pip install -r requirements.txt

# Training
jupyter notebook notebooks/training.ipynb

# Inference (Gradio app)
jupyter notebook notebooks/inference.ipynb
```

## Project Structure

```
├── chemception/          # Model and featurizer package
│   ├── featurizer.py     # Molecular image featurizer
│   ├── model.py          # Custom Chemception CNN
│   └── model_transfer.py # InceptionV3 transfer learning model
├── hpbandster_opt/       # Hyperparameter optimization
│   ├── worker.py         # BOHB worker (custom Chemception)
│   ├── worker_transfer.py# BOHB worker (transfer learning)
│   └── server.py         # BOHB server
├── notebooks/
│   ├── training.ipynb    # End-to-end training pipeline
│   └── inference.ipynb   # Gradio inference app
├── models/               # Trained weights
├── data/                 # PubChem AID686978 bioassay data
└── images/               # Figures and visualizations
```

## References

- [Chemception: A Deep Neural Network with Minimal Chemistry Knowledge Matches the Performance of Expert-developed QSAR/QSPR Models](https://www.researchgate.net/publication/317732180)
- [PubChem Bioassay AID686978](https://pubchem.ncbi.nlm.nih.gov/bioassay/686978)

<div align="center">
<img src="images/inference.png" width="700"/>
</div>

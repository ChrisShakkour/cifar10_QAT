# cifar10_QAT

> **Quantization‑Aware Training (QAT)** of neural networks on the **CIFAR‑10 dataset** — enabling compact and efficient models suitable for deployment on low‑resource devices.

---

## 📌 Overview

This repository implements **Quantization‑Aware Training (QAT)** for image classification models on the CIFAR‑10 dataset.
QAT simulates fixed‑point quantization of weights and activations during training to help models maintain accuracy after quantization and deployment on hardware with limited precision.

---

## 🚀 Features

✔ Support for training and evaluating models with QAT
✔ YAML configuration for flexible experiment setup
✔ Example scripts for training baseline and quantized models
✔ Modular project structure for models, quantization helpers, and utilities

---

## 📁 Repository Structure

```plaintext
.
├── docs/                         # Documentation (usage, concepts)
├── examples/                     # Example scripts (e.g., LSQ quantization)
├── model/                        # Model definitions
├── quan/                         # QAT / quantization modules
├── util/                         # Utility scripts (data loading, metrics)
├── main.py                       # Training entrypoint
├── main_analytical.py            # Analytical experiments
├── config.yaml                   # Default configuration
├── set_cifar10*.yaml             # CIFAR‑10 specific configs
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

---

## 📦 Installation
1. **Fix interactive shell:**

   ```bash
   exec bash
   ```

2. **Clone the repository:**

   ```bash
   git clone https://github.com/ChrisShakkour/cifar10_QAT.git
   cd cifar10_QAT
   ```

3. **Set up a Python environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```
4. **Or Set up a miniconda3 environment:**

   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   source ~/.bashrc
   conda create --name venv
   ```
   
5. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 Usage

### 🔹 validation of a Baseline Model in 32-bit

```bash
python main.py set_cifar10_baseline.yaml
```
### 🔹 training of a pre-trained Baseline Model in 32-bit

```bash
python main.py set_cifar10_baseline_training.yaml
```

### 🔹 Train with Quantization‑Aware Training (QAT)

```bash
python main.py set_cifar10.yaml
```

Configuration files control training hyperparameters such as learning rate, batch size, number of epochs, and quantization settings.

---

## 📊 Evaluation & Results

During and after training, logs, evaluation metrics, and model checkpoints are saved to the output directory specified in the configuration files.

You can compare:

* Full‑precision (baseline) accuracy
* Quantization‑aware training accuracy
* Accuracy vs. model size / deployment efficiency trade‑offs

*(Add quantitative results, plots, or tables here if available.)*

---

## 🧰 Configuration

All experiments are configured using YAML files. Example configuration:

```yaml
see set_cifar10.yaml
```

Adjust parameters according to your experiment needs.

---

## 📚 About CIFAR‑10

The **CIFAR‑10** dataset consists of **60,000 32×32 color images** across **10 classes**:

* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck

It is split into **50,000 training images** and **10,000 test images** and is commonly used for benchmarking image classification models.

---

## 🧪 Dependencies

Key dependencies include:

* Python 3.x
* PyTorch (or the deep‑learning framework used in this repository)
* PyYAML
* NumPy
* Additional packages listed in `requirements.txt`

---

## 🛠️ Contributing

Contributions are welcome. Feel free to open an issue or submit a pull request for improvements, bug fixes, or new features.

---

## 📄 License

This project is released under the **MIT License**. See the `LICENSE` file for details.

---

## 🙏 Acknowledgements

This project builds on ideas from the quantization and efficient deep‑learning research community. Thanks to all open‑source contributors whose work made this project possible.

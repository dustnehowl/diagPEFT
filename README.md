# LLaMA 3.1 LoRA Fine-tuning with KoAlpaca

This repository contains code and utilities for fine-tuning the LLaMA 3.1 model using [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation) on the Korean instruction dataset [beomi/KoAlpaca-v1.1a](https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a). The goal is to benchmark LoRA-based PEFT and eventually compare it to diagonal-matrix-based alternatives.

---

## 🧾 Project Structure

```
.
├── data/                          # (optional custom data)
├── train.py                       # Training script
├── inference.py                   # Inference/test script
├── preprocess.py                  # KoAlpaca formatting util
├── requirements.txt               # Python dependencies
└── README.md                      # Project guide
```

---

## ✅ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment
- Python 3.10+
- PyTorch with GPU support (8bit/4bit quantization)
- LLaMA 3.1 HF model access granted (Meta license)

---

## 🏋️‍♂️ Training

```bash
python train.py
```

- Uses `meta-llama/Meta-Llama-3-8B-Instruct` with 8-bit loading.
- Applies LoRA to attention projection layers.
- Fine-tunes on KoAlpaca v1.1a Korean instruction dataset.

---

## 🤖 Inference

```bash
python inference.py
```

Modify the prompt inside `inference.py` to test different queries.


---

## 🚧 TODO Checklist

- [x] Integrate LoRA into LLaMA 3.1 fine-tuning
- [x] Load and preprocess KoAlpaca dataset
- [x] Train on LoRA adapter and save checkpoints
- [x] Run inference using LoRA adapter
- [ ] Add diagonal-matrix PEFT baseline
- [ ] Log training metrics (e.g., via TensorBoard or wandb)
- [ ] Evaluate models on benchmark tasks
- [ ] Export adapter-only weights
- [ ] Hugging Face Hub integration
- [ ] Support for full/half precision inference
- [ ] Add evaluation script (`evaluate.py`)

---

## 🔖 Credits
- Based on Meta LLaMA 3.1 model
- Dataset: [beomi/KoAlpaca-v1.1a](https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a)
- LoRA from [PEFT library](https://github.com/huggingface/peft)

---

Feel free to open issues or pull requests for improvements or extensions!

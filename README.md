# GPT-LLaMA2 Trainer

This project contains a modular pipeline for fine-tuning LLaMA2-based language models using Hugging Face's `transformers`, `peft`, and `trl` libraries.

## 📦 Features

- Parameter-efficient fine-tuning (LoRA)
- Hugging Face Dataset loading
- OpenAI integration (optional)
- Supports Colab and local environments

## 📁 Directory Structure

```
gpt_llama2_trainer/
│
├── gpt_llama2_trainer/
│   ├── __init__.py
│   └── trainer.py
│
├── scripts/
│   └── train.py
│
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Installation

```bash
git clone <repo-url>
cd gpt_llama2_trainer
pip install -r requirements.txt
```

### Usage

```bash
python scripts/train.py
```

## 🧠 Notes

- Colab-specific code is present; modify or remove for production environments.
- Make sure to configure datasets and model paths as needed in `trainer.py`.

---

Made with ❤️ by Eldar Hacohen   

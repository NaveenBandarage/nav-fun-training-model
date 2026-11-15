# Nav Fun Training Model

A small, hands-on Jupyter notebook for fine-tuning a language model on a playful message-classification task. It uses fictional examples and runs locally on Apple Silicon with MLX.

The model learns to label messages as **greeting**, **question**, or **request**. The notebook compares the base model with a LoRA adapter, then lets you try your own messages. The tiny dataset is for learning, not a reliable measure of real-world accuracy.

## Get started

You need an Apple Silicon Mac, Python 3.11 or newer, and an internet connection for the initial model download.

```sh
git clone https://github.com/NaveenBandarage/nav-fun-training-model.git
cd nav-fun-training-model
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab notebooks/fun_training.ipynb
```

Run the notebook cells in order. The first model load downloads `mlx-community/Qwen3.5-0.8B-4bit`; the training cell creates a new adapter under `adapters/`. Model weights, adapters, notebook checkpoints, and local environments are intentionally excluded from Git. The model download uses its own license; see the [model card](https://huggingface.co/Qwen/Qwen3.5-0.8B).

## Explore

- `notebooks/fun_training.ipynb`: the guided experiment.
- `data/train.jsonl` and `data/valid.jsonl`: fictional examples in chat format.
- `scripts/prepare_data.py`: regenerates those example files; it overwrites them.

The validation examples are monitored during training, so they are not an independent test. For a stronger evaluation, use a larger dataset with separate, independently written test examples.

Want to use this approach for a message-routing assistant? See the [adaptation guide](docs/adapting-to-an-assistant.md) for data preparation, training, evaluation, and integration.

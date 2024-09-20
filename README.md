# Responsible Generative AI Tools

Proof of concept library of Responsible Generative AI Tools.

## Installation

Temporarily, you can install this library locally in its own virtual
environment after cloning the repository:

```bash
git clone https://github.com/owahltinez/rgai-tools && cd rgai-tools
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Usage: Agile Classifier

To train your own agile classifier, you'll need a dataset with records
containing `text` and `label` keys. The set of all possible classes will be
inferred from all labels present in the data. The data should be in JSONL
format, with one record per line:

```bash
# Write dataset content to `dataset.jsonl`.
tee dataset.jsonl <<EOF > /dev/null
{'text': 'it has four wheels', 'label': 'car'}
{'text': 'it is human powered', 'label': 'bike'}
{'text': 'it goes on water', 'label': 'boat'}
EOF
# Pipe the dataset to the agile classifier for training.
cat dataset.jsonl | python -m rgai_tools.agile_classifier \
    --model_preset='gemma2_instruct_2b_en' \
    --model_output=/path/to/output.lora.h5
```

The fine-tuned model will be available at the specified location and can be
loaded using:

```python
from rgai_tools.common import model_loader
model = model_loader.load_gemma_model("gemma2_instruct_2b_en")
model.backbone.load_lora_weights("/path/to/output.lora.h5")
```

NOTE: Your kaggle credentials need to be [properly set][kaggle-setup] first.

## Usage: ShieldGemma

To determine whether some text is in violation of one of the policy types
supported by ShieldGemma, you can do the following:

```bash
python -m rgai_tools.shield_gemma \
    --harm_type=HATE \
    --use_case=PROMPT_ONLY \
    --user_content='text to evaluate'
```

NOTE: Your kaggle credentials need to be [properly set][kaggle-setup] first.

## Usage: Model Aligner

Model alignment is done via the [model-alignment][model-alignment] package.
It runs as an interactive terminal application that lets you iteratively
improve prompts based on user or auto-generated feedback. To start the process,
run:

```bash
python -m rgai_tools.model_aligner
```

NOTE: You will need to set a `GEMINI_API` environment variable with your API
key.

[kaggle-setup]: https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#api-credentials
[model-alignment]: https://github.com/PAIR-code/model-alignment

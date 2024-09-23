# Responsible Generative AI Tools

Proof of concept library of Responsible Generative AI Tools.

## Installation

You can install `rgai_tools` directly using `pipx`:

```bash
pipx install 'git+https://github.com/owahltinez/rgai-tools.git'
```

If you are using an M1/M2/M3 Mac device, you will need to provide a URL for
prebuilt binaries of `tensorflow-text` and additionally install
`tensorflow-metal` to enable hardware acceleration:

```bash
TF_TEXT='https://github.com/sun1638650145/Libraries-and-Extensions-for-TensorFlow-for-Apple-Silicon/releases/download/v2.17/tensorflow-2.17.0-cp311-cp311-macosx_14_0_arm64.whl'
pipx install 'git+https://github.com/owahltinez/rgai-tools.git' \
    --preinstall setuptools \
    --preinstall "$TF_TEXT" \
    --preinstall tensorflow-metal \
    --python $(which python3.11)
```

After installing, there will be a command `rgai-tools` available in your shell
which will be the main entrypoint for the various components.

For development, you can install this library locally in its own virtual
environment after cloning the repository:

```bash
git clone https://github.com/owahltinez/rgai-tools && cd rgai-tools
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Usage: Agile Classifier

Agile classifiers and related subcomponents are available under the
`rgai-tools agile-classifier` subcommand.

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
cat dataset.jsonl | rgai-tools agile-classifier train \
    --labels='car,bike,boat' \
    --model-preset='gemma2_instruct_2b_en' \
    --model-output=/path/to/output.lora.h5
```

The fine-tuned model will be available at the specified location and can be
loaded using:

```python
from rgai_tools.agile_classifier import model_wrapper
from rgai_tools.common import model_loader

# Load model from LoRA checkpoint.
labels = ["car", "bike", "boat"]
model = model_loader.load_gemma_model("gemma2_instruct_2b_en")
model.backbone.load_lora_weights("/path/to/output.lora.h5")
classifier = model_wrapper.AgileClassifier(model=model, labels=labels)

# Perform inference using model wrapper.
pred = classifier.predict(["it has two wheels"])
```

NOTE: Your kaggle credentials need to be [properly set][kaggle-setup] first.

## Usage: ShieldGemma

ShieldGemma and related subcomponents are available under the
`rgai-tools shieldgemma` subcommand.

To determine whether some text is in violation of one of the policy types
supported by ShieldGemma, you can do the following:

```bash
echo "{'harm_type': 'HATE', 'user_content': 'have a nice day'}" | rgai-tools shieldgemma
```

NOTE: Your kaggle credentials need to be [properly set][kaggle-setup] first.

## Usage: Model Aligner

Model Aligner and related subcomponents are available under the
`rgai-tools model-aligner` subcommand.

Model alignment is done via the [model-alignment][model-alignment] package.
It runs as an interactive terminal application that lets you iteratively
improve prompts based on user or auto-generated feedback. To start the process,
run:

```bash
rgai-tools model-aligner
```

NOTE: You will need to set a `GEMINI_API` environment variable with your API
key.

[kaggle-setup]: https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#api-credentials
[model-alignment]: https://github.com/PAIR-code/model-alignment

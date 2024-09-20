import sys
import json5

from absl import app
from absl import flags
from absl import logging

from rgai_tools.common import model_loader
from rgai_tools.agile_classifier import model_wrapper


_DEFAULT_MODEL_PRESET = "gemma_instruct_2b_en"


_MODEL_OUTPUT = flags.DEFINE_string(
    name="model_output",
    default=None,
    help="Path to save the model.",
    required=True,
)
_MODEL_PRESET = flags.DEFINE_string(
    name="model_preset",
    default=_DEFAULT_MODEL_PRESET,
    help="Preset (name) of the model, or path to local keras model.",
)
_EPOCHS = flags.DEFINE_integer(
    name="epochs",
    default=1,
    help="Number of epochs to train the classifier.",
)
_MAX_SEQ_LEN = flags.DEFINE_integer(
    name="max_sequence_length",
    default=128,
    help="Maximum sequence length for the model's preprocessor.",
)


def main(_) -> None:
  # The model output path should end with ".lora.h5".
  model_output_path = _MODEL_OUTPUT.value
  if not model_output_path.endswith(".lora.h5"):
    raise ValueError("The model output path should end with '.lora.h5'.")

  # Load the LLM model.
  llm = model_loader.load_gemma_model(
      preset=_MODEL_PRESET.value,
      max_sequence_length=_MAX_SEQ_LEN.value,
  )

  # Read the data from stdin.
  records = []
  for line in sys.stdin.readlines():
    line = line.strip()
    if line:
      try:
        record = json5.loads(line)
        records.append({"text": record["text"], "label": record["label"]})
      except Exception as exc:
        logging.error("Failed to parse input line: %s", line)
        logging.error("Expected format:")
        logging.error('{"text": "text content", "label": "label"}')
        raise exc

  # Train the classifier.
  classifier = model_wrapper.train_agile_classifier(
      model=llm,
      x_train=[x["text"] for x in records],
      y_train=[x["label"] for x in records],
      epochs=_EPOCHS.value,
  )

  # Save the model (only the LoRA wegiths).
  classifier.model.backbone.save_lora_weights(model_output_path)

  # Log information about how we can re-load the model.
  logging.info(
      "The model can be re-loaded using the following code:\n"
      "model = model_loader.load_gemma_model(preset='%s')\n"
      "model.backbone.load_lora_weights('%s')",
      _MODEL_PRESET.value,
      model_output_path,
  )


if __name__ == "__main__":
  app.run(main)

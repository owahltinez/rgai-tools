import sys
import json5
import click

from rgai_tools.common import model_loader
from rgai_tools.agile_classifier import model_wrapper

_DEFAULT_MODEL_PRESET = "gemma_instruct_2b_en"


@click.command()
@click.option(
    "--labels",
    required=True,
    help="Comma-separated list of labels for the classifier.",
)
@click.option(
    "--model-output",
    required=True,
    help="Path to save the model. Should end with '.lora.h5'.",
)
@click.option(
    "--model-preset",
    default=_DEFAULT_MODEL_PRESET,
    help="Preset (name) of the model, or path to local keras model.",
)
@click.option(
    "--epochs",
    default=1,
    help="Number of epochs to train the classifier.",
)
@click.option(
    "--max-sequence-length",
    default=128,
    help="Maximum sequence length for the model's preprocessor.",
)
def main(labels, model_output, model_preset, epochs, max_sequence_length):
  # The model output path should end with ".lora.h5".
  if not model_output.endswith(".lora.h5"):
    raise ValueError("The model output path should end with '.lora.h5'.")

  # Load the LLM model.
  llm = model_loader.load_gemma_model(
      preset=model_preset,
      max_sequence_length=max_sequence_length,
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
        click.echo(f"Failed to parse input line: {line}", err=True)
        click.echo("Expected format:", err=True)
        click.echo('{"text": "text content", "label": "label"}', err=True)
        raise exc

  # Train the classifier.
  classifier = model_wrapper.train_agile_classifier(
      labels=labels.split(","),
      model=llm,
      x_train=[x["text"] for x in records],
      y_train=[x["label"] for x in records],
      epochs=epochs,
  )

  # Save the model (only the LoRA weights).
  classifier.model.backbone.save_lora_weights(model_output)

  # Log information about how we can re-load the model.
  click.echo(
      f"The model can be re-loaded using the following code:\n"
      f"model = model_loader.load_gemma_model(preset='{model_preset}')\n"
      f"model.backbone.load_lora_weights('{model_output}')"
  )


if __name__ == "__main__":
  main()

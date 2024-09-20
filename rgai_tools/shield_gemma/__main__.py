import sys
import json5
import click

from ..common import model_loader
from . import model_wrapper
from . import text_processing

_DEFAULT_MODEL_PRESET = "shieldgemma_2b_en"


@click.command()
@click.option(
    "--model_preset",
    default=_DEFAULT_MODEL_PRESET,
    help="Preset (name) of the model, or path to local keras model.",
)
def main(model_preset):
  # Load model and wrapper.
  base_model = model_loader.load_gemma_model(model_preset)
  shieldgemma = model_wrapper.ShieldGemma(base_model)
  click.echo(f"Loaded ShieldGemma model from preset {model_preset}")

  # Read stdin for the user content.
  click.echo(
      "Expected format: {'harm_type': 'HATE', 'user_content': 'content'}\n"
      "Reading user content from stdin. You can pipe input from another "
      "command or type it in the terminal followed by [CTRL] + D."
  )
  prompts = []
  for line in sys.stdin.readlines():
    line = line.strip()
    if line:
      try:
        record = json5.loads(line)
        # Parse the HarmType enum from the enum name (not value).
        record["harm_type"] = text_processing.HarmType[record["harm_type"]]
        prompts.append(text_processing.build_prompt(**record))
      except Exception as exc:
        click.echo(f"Failed to parse input line: {line}. Error: {exc}", err=True)
        raise exc

  # Predict and output the policy violation probability.
  outputs = shieldgemma.predict_score(prompts)
  for output in outputs:
    click.echo(output[0])


if __name__ == "__main__":
  main()

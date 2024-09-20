import sys

from absl import app
from absl import flags
from absl import logging
import json5

from rgai_tools.common import model_loader
from rgai_tools.shield_gemma import model_wrapper
from rgai_tools.shield_gemma import text_processing


_DEFAULT_MODEL_PRESET = "shieldgemma_2b_en"

_MODEL_PRESET = flags.DEFINE_string(
    name="model_preset",
    default=_DEFAULT_MODEL_PRESET,
    help="Preset (name) of the model, or path to local keras model.",
)


def main(_) -> None:
  # Load model and wrapper.
  base_model = model_loader.load_gemma_model(_MODEL_PRESET.value)
  shieldgemma = model_wrapper.ShieldGemma(base_model)
  logging.info("Loaded ShieldGemma model from preset %s", _MODEL_PRESET.value)

  # Read stdin for the user content.
  logging.info(
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
        logging.error("Failed to parse input line: %s. Error: %r", line, exc)
        logging.error("Expected format:")
        logging.error('{"harm_type": "HATE", "user_content": "user content"}')
        raise exc

  # Predict and output the policy violation probability.
  outputs = shieldgemma.predict_score(prompts)
  for output in outputs:
    print(output[0])


if __name__ == "__main__":
  app.run(main)

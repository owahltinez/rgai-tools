from absl import logging
import keras_nlp


def load_gemma_model(
    preset: str,
    max_sequence_length: int = 512,
) -> keras_nlp.models.CausalLM:
  # Load the model from preset.
  logging.info("Loading model from preset %s", preset)
  model = keras_nlp.models.GemmaCausalLM.from_preset(preset)

  # Update the model's sequence length to ensure it doesn't run out of memory.
  model.preprocessor.sequence_length = max_sequence_length

  return model

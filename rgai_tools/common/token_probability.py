import keras
import keras_nlp


class TokenProbabilityLayer(keras.layers.Layer):
  """Layer that returns relative probabilities for a token set."""

  def __init__(self, token_set_idx: list[int], **kwargs):
    super().__init__(**kwargs)
    self.token_set_idx = token_set_idx

  def call(self, logits, padding_mask):
    offset = keras.ops.sum(padding_mask, axis=1) - 1
    last_prompt_index = keras.ops.cast(offset, "int32")
    last_logits = keras.ops.take(logits, last_prompt_index, axis=1)[:, 0]
    token_logits_list = [last_logits[:, idx] for idx in self.token_set_idx]
    token_logits_stacked = keras.ops.stack(token_logits_list, axis=1)
    return keras.ops.softmax(token_logits_stacked, axis=1)


def build_token_probability_model(
    model: keras_nlp.models.CausalLM,
    token_set: list[str],
) -> keras.Model:
  token_to_id = model.preprocessor.tokenizer.token_to_id
  token_set_idx = [token_to_id(token) for token in token_set]
  inputs = model.input
  x = model(inputs)
  x = TokenProbabilityLayer(token_set_idx)(x, inputs["padding_mask"])
  return keras.Model(inputs=inputs, outputs=x)

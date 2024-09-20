from typing import Iterable

import keras
import keras_nlp
import numpy
import tensorflow as tf

from rgai_tools.agile_classifier import text_processing
from rgai_tools.common import token_probability


_DEFAULT_PROMPT = "Classify the following text into one of the following classes"


class AgileClassifier:
  """Agile classifier to be wrapped around an LLM."""

  def __init__(
      self,
      model: keras_nlp.models.CausalLM,
      labels: tuple[str, ...],
      instructions: str = _DEFAULT_PROMPT,
      separator_token: str = "<separator>",
      end_of_text_token: str = "<eos>",
  ):
    self.model = model
    self.labels = labels
    self.instructions = instructions
    self.separator_token = separator_token
    self.end_of_text_token = end_of_text_token
    self.probability_model = token_probability.build_token_probability_model(
        model=model,
        token_set=labels,
    )

  def _encode_for_prediction(self, x_text: str) -> str:
    return text_processing.build_prompt(
        text=x_text,
        labels=self.labels,
        instructions=self.instructions,
        separator=self.separator_token,
    )

  def _encode_for_training(self, x_text: str, y_label: str) -> str:
    return self._encode_for_prediction(x_text) + y_label + self.end_of_text_token

  def fit(
      self,
      x_train: list[str],
      y_train: list[str],
      batch_size: int = 1,
      **fit_opts,
  ) -> keras.callbacks.History:
    records = list(map(self._encode_for_training, x_train, y_train))
    ds_train = tf.data.Dataset.from_tensor_slices(records).batch(batch_size)
    return self.model.fit(ds_train, **fit_opts)

  def predict_score(self, x_text: Iterable[str]) -> list[tuple[float, float]]:
    """Predicts the probabilities for the label tokens."""
    prompts = [self._encode_for_prediction(text) for text in x_text]
    inputs = self.model.preprocessor.generate_preprocess(prompts)
    return self.probability_model.predict(inputs, verbose=0)

  def predict(self, x_text: Iterable[str]) -> list[str]:
    idx = numpy.argmax(self.predict_score(x_text), axis=1)
    return [self.labels[i] for i in idx]


def train_agile_classifier(
    labels: tuple[str, ...],
    model: keras_nlp.models.CausalLM,
    x_train: list[str],
    y_train: list[str],
    epochs: int = 1,
    batch_size: int = 1,
    lora_rank: int = 4,
) -> AgileClassifier:
  # Create an instance of the AgileClassifier.
  agile_classifier = AgileClassifier(model=model, labels=labels)

  # Enable LoRA for efficient parameter fine-tuning.
  model.backbone.enable_lora(rank=lora_rank)

  # Compile the model using the Adam optimizer and appropriate loss function.
  model.compile(
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=keras.optimizers.Adam(learning_rate=0.0005),
      weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
  )

  # Begin training.
  agile_classifier.fit(
      x_train,
      y_train,
      epochs=epochs,
      batch_size=batch_size,
  )

  # Return the trained AgileClassifier.
  return agile_classifier

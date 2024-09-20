from typing import Iterable
import keras_nlp

from rgai_tools.common import token_probability


class ShieldGemma:

  def __init__(self, model: keras_nlp.models.CausalLM):
    self.model = model
    self.probability_model = token_probability.build_token_probability_model(
        model=model,
        token_set=["Yes", "No"],
    )

  def predict_score(self, x_text: Iterable[str]) -> list[tuple[float, float]]:
    """Predicts the probabilities for the "Yes" and "No" tokens."""
    inputs = self.model.preprocessor.generate_preprocess(x_text)
    return self.probability_model.predict(inputs, verbose=0)

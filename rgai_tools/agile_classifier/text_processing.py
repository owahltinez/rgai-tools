def build_prompt(
    text: str,
    labels: list[str],
    instructions: str,
    separator: str = "<separator>",
) -> str:
  prompt = f'{instructions}:[{",".join(labels)}]'
  return separator.join([prompt, f"Text:{text}", "Prediction:"])

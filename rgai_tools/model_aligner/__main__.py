import os

from absl import app
import json5

from model_alignment import model_helper
from model_alignment import single_run


def print_indented(text: str, indent: int = 1) -> None:
  print("\t" * indent + text.replace("\n", "\n" + "\t" * indent))


def prompt_for_inputs() -> dict[str, str]:
  input_instance = {}
  input_prompt = 'Enter a JSON object with {"variable": "value"} values: '
  while not input_instance:
    json_data = input(input_prompt).strip()
    try:
      input_instance = json5.loads(json_data)
    except ValueError as exc:
      print(f"Error: {exc}")

  return input_instance


def print_model_response(
    aligner: single_run.AlignableSingleRun,
    input_instance: dict[str, str],
) -> None:
  response = aligner.send_input(input_instance)
  print("\nModel response:\n")
  print_indented(response.text.strip(), indent=1)
  print("\n\n")


def main(*_) -> None:
  gemini_key = os.getenv("GEMINI_KEY")
  if not gemini_key:
    raise ValueError("GEMINI_KEY environment variable must be set")

  context = ""
  input_prompt = "Enter the model prompt with any {variable} in curly braces: "
  while not context:
    context = input(input_prompt).strip()

  gemini_model = model_helper.GeminiModelHelper(api_key=gemini_key)
  aligner = single_run.AlignableSingleRun(gemini_model)
  aligner.set_model_description(context)

  input_instance = prompt_for_inputs()
  print_model_response(aligner, input_instance)

  options = {
      "0": "Finish alignment",
      "1": "Critique the response",
      "2": "Praise the response",
      "3": "Auto generate critiques for the response",
      "4": "Auto generate praises for the response",
      "5": "Re-enter different variables",
  }
  input_prompt = "Please select an option to continue alignment:\n"
  input_prompt += "\n".join([f"[{k}] {v}" for k, v in options.items()])
  input_prompt += "\n\nEnter the selected option: "
  while True:
    selection = ""
    while selection not in options:
      selection = input(input_prompt).strip()

    if selection == "0":
      break

    elif selection == "1":
      critique = input("Enter a critique for the response: ")
      principles = aligner.critique_response(critique)
      print(f"\nPrinciples generated from critique: {principles}")
      aligner.update_model_description_from_principles()
      print_model_response(aligner, input_instance)

    elif selection == "2":
      praise = input("Enter praise for the response: ")
      principles = aligner.kudos_response(praise)
      print(f"\nPrinciples generated from praise: {principles}")
      aligner.update_model_description_from_principles()
      print_model_response(aligner, input_instance)

    elif selection == "3":
      critiques = aligner.generate_critiques()
      critiques_str = [f"\n{i + 1}. {c}" for i, c in enumerate(critiques)]
      print(f"\nGenerated critiques:{''.join(critiques_str)}\n")

    elif selection == "4":
      praises = aligner.generate_kudos()
      praises_str = [f"\n{i + 1}. {c}" for i, c in enumerate(praises)]
      print(f"\nGenerated praises:{''.join(praises_str)}\n")

    elif selection == "5":
      input_instance = prompt_for_inputs()
      print_model_response(aligner, input_instance)

  print("\nAlignment complete.")
  print("Final model description and principles:\n")
  print_indented(aligner.get_model_description_with_principles() + "\n", indent=1)


if __name__ == "__main__":
  app.run(main)

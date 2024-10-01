import sys

from absl import logging
import click
import json5
from tqdm import auto as tqdm

from llm_comparator import comparison
from llm_comparator import types as llm_types
from llm_comparator import llm_judge_runner
from llm_comparator import model_helper
from llm_comparator import rationale_bullet_generator
from llm_comparator import rationale_cluster_generator
from rgai_tools.common import model_loader
from rgai_tools.llm_comparator import simple_server


@click.group()
def llm_comparator():
  pass


@llm_comparator.command()
@click.option(
    "--config-data",
    type=click.STRING,
    help="JSON string with LLM comparator config data.",
)
@click.option(
    "--config-file",
    type=click.STRING,
    help="Path to the saved LLM comparator config file.",
)
@click.option(
    "--port",
    type=click.INT,
    default=8080,
    help="Port to use for the LLM Comparator UI server.",
)
def launch(
    *,
    config_data: str,
    config_file: str,
    port: int,
) -> None:
  simple_server.serve_llmc(
      config_data=config_data,
      config_file=config_file,
      port=port,
  )


@llm_comparator.command()
@click.option(
    "--model-a",
    type=click.STRING,
    required=True,
    help="Preset (name) of the first model, or path to local keras model.",
)
@click.option(
    "--model-b",
    type=click.STRING,
    required=True,
    help="Preset (name) of the first model, or path to local keras model.",
)
@click.option(
    "--max-token-count",
    type=click.INT,
    default=512,
    help="Maximum number of tokens to generate.",
)
@click.option(
    "--model-judge",
    type=click.STRING,
    help="Preset (name) of the LLM judge model, or path to local keras model.",
)
@click.option(
    "--model-judge-count",
    type=click.INT,
    default=3,
    help="Number of individual raters to use for the model judge.",
)
@click.option(
    "--model-judge-prompt",
    type=click.STRING,
    help="Prompt template to use for the LLM judge model.",
)
@click.option(
    "--output-file",
    type=click.STRING,
    help=(
        "Path to the saved LLM comparator config file. If none is provided, "
        "the config will be saved as a temporary file and deleted on exit."
    ),
)
@click.option(
    "--port",
    type=click.INT,
    default=8080,
    help="Port to use for the LLM Comparator UI server.",
)
@click.option(
    "--serve",
    is_flag=True,
    help="Start a server to serve the LLM Comparator UI.",
)
def compare(
    *,
    model_a: str,
    model_b: str,
    max_token_count: int,
    model_judge: str,
    model_judge_prompt: str,
    model_judge_count: int,
    output_file: str,
    port: int,
    serve: bool,
) -> None:
  models = [{"name": model_a}, {"name": model_b}]
  metadata = dict(source_path="rgai-tools", custom_fields_schema=[])
  config = dict(models=models, metadata=metadata, examples=[])

  # Load the model judge.
  if model_judge:
    generator = model_helper.VertexGenerationModelHelper(model_name=model_judge)
    llm_judge_opts = dict()
    if model_judge_prompt:
      llm_judge_opts["llm_judge_prompt_template"] = model_judge_prompt
    llm_judge = llm_judge_runner.LLMJudgeRunner(generator, **llm_judge_opts)

    # The embedding model can be any text embedder provided by Vertex AI. We default
    # to 'textembedding-gecko@003' but you can change this with the `model_name=`
    # param
    embedder = model_helper.VertexEmbeddingModelHelper()

    # The `bulletizer` condenses the results provided by the judge into a set of
    # bullets to make them easier to understand and consume in the UI.
    bulletizer = rationale_bullet_generator.RationaleBulletGenerator(generator)

    # The `clusterer` takes the bullets, embeds them, groups them into clusters
    # based on embedding similarity, and generates a label for those clusters.
    clusterer = rationale_cluster_generator.RationaleClusterGenerator(
        gen_model_helper=generator,
        emb_model_helper=embedder,
    )

  # Pre-define variables for the LLM models that might be needed later.
  llm_a: model_loader.IModel | None = None
  llm_b: model_loader.IModel | None = None

  # Read stdin for the user content.
  click.echo(
      """
Expected format, in a single line:
  {
    'input': '<input prompt>',
    'tags': [<list of keywords for categorizing prompts>],
    'output_text_a': '<output text for model A, if no model A is given>',
    'output_text_b': '<output text for model B, if no model B is given>',
    'score': <numeric score for the example, if no LLM judge is given>,
    'individual_rater_scores': [<list of individual rater scores, optional>],
    'radionale_list': [<list of individual rater rationale, optional>],
  }

Reading user content from stdin. You can pipe input from another command or
command or type it in the terminal followed by [CTRL + D]."""
  )
  # NOTE: This will produce prompt outputs one record at a time, which is a
  # very inefficient way to use the LLM models. In the future, batching should
  # be implemented.
  for line in tqdm.tqdm(sys.stdin.readlines(), desc="Processing inputs"):
    line = line.strip()
    if line:
      try:
        record = json5.loads(line)

        # Produce the output for model A.
        if "output_text_a" not in record:
          if not model_a:
            raise ValueError("Expected 'output_text_a' field in input when no model A is given.")
          if not llm_a:
            llm_a = model_loader.load_gemma_model(model_a)
          record["output_text_a"] = llm_a.generate(
              [record["input"]],
              max_length=max_token_count,
          )[0]

        # Produce the output for model B.
        if "output_text_b" not in record:
          if not model_b:
            raise ValueError("Expected 'output_text_b' field in input when no model B is given.")
          if not llm_b:
            llm_b = model_loader.load_gemma_model(model_b)
          record["output_text_b"] = llm_b.generate(
              [record["input"]],
              max_length=max_token_count,
          )[0]

        # Produce the score from the LLM judge.
        if "score" not in record:
          if not model_judge:
            raise ValueError("Expected 'score' field in input when no LLM judge is given.")

          llm_judge_input = llm_types.LLMJudgeInput(
              prompt=record["input"],
              response_a=record["output_text_a"],
              response_b=record["output_text_b"],
          )
          llm_judge_output = comparison.run(
              inputs=[llm_judge_input],
              judge=llm_judge,
              bulletizer=bulletizer,
              clusterer=clusterer,
              model_names=(model_a, model_b),
              judge_opts=dict(num_repeats=model_judge_count),
          )
          record = dict(record, **llm_judge_output["examples"][0])

        # Append the record to the examples list.
        config["examples"].append(record)
      except Exception as exc:
        click.echo(f"Failed to parse input line: {line}. Error: {exc}", err=True)
        raise exc

  # Save the config to the output file.
  if output_file is not None:
    with open(output_file, "w") as f:
      logging.info("Saving LLM comparator config to %s", output_file)
      json5.dump(config, f)

  if serve:
    simple_server.serve_llmc(
        config_data=config,
        port=port,
    )


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  llm_comparator()

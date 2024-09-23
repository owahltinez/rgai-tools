import click
from rgai_tools.agile_classifier.cli import agile_classifier
from rgai_tools.model_aligner.cli import model_aligner
from rgai_tools.shieldgemma.cli import shieldgemma


@click.group()
def cli():
  pass


# Add all the subcommands to the main CLI instance.
cli.add_command(agile_classifier)
cli.add_command(model_aligner)
cli.add_command(shieldgemma)

if __name__ == "__main__":
  cli()

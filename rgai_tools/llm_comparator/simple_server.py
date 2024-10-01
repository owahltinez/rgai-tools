import json
import os
import tempfile
import shutil
from typing import Any

from absl import logging
import bottle
import llm_comparator


def static_server(
    directory: str,
    port: int = 8080,
    load_message: str | None = None,
) -> None:
  """Starts a simple server to serve static files from the specified directory.

  Args:
    directory: The directory to serve static files from.
    port: The port to serve the files on.
    load_message: The message to display when the server is started.

  Returns:
    None
  """

  @bottle.route("/")
  def serve_index():
    return bottle.static_file("index.html", root=directory)

  @bottle.route("/<filepath:path>")
  def serve_static(filepath: str):
    return bottle.static_file(filepath, root=directory)

  print(load_message or f"Serving {directory} at http://localhost:{port}")
  bottle.run(host="localhost", port=port, quiet=True)


def serve_llmc(
    *,
    config_file: str | None = None,
    config_data: dict[str, Any] | None = None,
    port: int = 8080,
) -> None:
  if (not config_data and not config_file) or (config_file and config_data):
    raise ValueError("Either config_file or config_data must be provided")

  # Get the website root directory.
  www_root = os.path.join(llm_comparator.__path__[0], "data")

  # Determine the base URL for the server.
  base_url = f"http://localhost:{port}"

  # Create a temporary folder, and serve files from there..
  with tempfile.TemporaryDirectory() as temp_dir:
    logging.info(f"Creating temporary directory at {temp_dir}")

    # Copy all of the static contents.
    shutil.copytree(www_root, temp_dir, dirs_exist_ok=True)

    # Write the config file to the temporary directory, if given as data.
    if config_data:
      with open(os.path.join(temp_dir, "config.json"), "w") as f:
        json.dump(config_data, f)

      # The UI expects a URL, so we need to make the path relative.
      config_file = f"{base_url}/config.json"

    # Write a message to the terminal so users can click it.
    query_string = f"?results_path={config_file}"
    load_msg = f"Serving the LLM Comparator: {base_url}/{query_string}"

    # Instantiate the server.
    static_server(directory=temp_dir, port=port, load_message=load_msg)

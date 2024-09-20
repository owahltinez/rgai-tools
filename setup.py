import setuptools

# Read the requirements from requirements.txt
with open("requirements.txt") as f:
  requirements = f.read().splitlines()

setuptools.setup(
    name="rgai_tools",
    version="0.1.0",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "agile_classifier = rgai_tools.agile_classifier.__main__:main",
            "model_aligner = rgai_tools.model_aligner.__main__:main",
            "shieldgemma = rgai_tools.shieldgemma.__main__:main",
        ],
    },
)

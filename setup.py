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
            "rgai-tools = rgai_tools.__main__:cli",
        ],
    },
)

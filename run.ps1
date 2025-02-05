# Activate the virtual environment
. .\venv\Scripts\Activate.ps1

# Install the package (in editable mode, if necessary)
pip install -e .

# Run the `__main__.py` file
python -m GRPO

"""
Module 'definitions' is the module with global constants. It is reserved for
special truths that is absolute anywhere in the project.

Constants that are applicable to specialised must not be placed here.
"""

from pathlib import Path

# The SURIMI-market's root directory.
ROOT_DIR = Path(__file__).parent.parent.resolve()

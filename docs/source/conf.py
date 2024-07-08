# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "adoptODE"
copyright = "2024, Leon Lettermann, Sebastian Herzog"
author = "Leon Lettermann, Sebastian Herzog"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx.ext.doctest",
]

templates_path = ["_templates"]
exclude_patterns = []

# don't copy the dollar signs, >>>, etc.
copybutton_prompt_text = r">>> |\.\.\. |^([\w -]+@[\w.-]+):(\/[\w\/.-]+)(\$|#)\s* |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

master_doc = "contents"

add_module_names = False

modindex_common_prefix = [f"{project}."]
toc_object_entries = False
show_authors = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "display_version": True,
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
}

html_static_path = ["_static"]


# -- Extension Configuration
autoclass_content = "both"  # include __init__ docstring in class description

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = None
todo_include_todos = True

# Configuration for intersphinx
intersphinx_mapping = {
    "h5py": ("https://docs.h5py.org/en/latest", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numba": ("https://numba.pydata.org/numba-doc/latest/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "pytest": ("https://docs.pytest.org/en/latest/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}
import sys

sys.path.append("../../../adoptODE/")
sys.path.append(".")
import adoptODE
from run_autodoc import main

main()

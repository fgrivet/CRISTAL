# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# pylint: disable=unused-variable
import locale
from datetime import datetime

from cristal import __date__, __license__, __version__

project = "CRISTAL"
copyright = "2026, Florian Grivet"
author = "Florian Grivet"
release = __version__
locale.setlocale(locale.LC_TIME, "C")
release_date = datetime.strptime(__date__, "%Y-%m-%d").strftime("%d %B %Y")
license = __license__

rst_epilog = f"""
.. |project| replace:: {project}
.. |author| replace:: **{author}**
.. |release| replace:: **{release}**
.. |release_date| replace:: **{release_date}**
.. |license| replace:: **{license}**

.. role:: add-badge
   :class: badge add-badge

.. role:: improve-badge
   :class: badge improve-badge

.. role:: change-badge
   :class: badge change-badge

.. role:: fix-badge
   :class: badge fix-badge

.. |Add| replace:: :add-badge:`Add`
.. |Improve| replace:: :improve-badge:`Improve`
.. |Change| replace:: :change-badge:`API Change`
.. |Fix| replace:: :fix-badge:`Fix`
"""


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_nb",
    "sphinx_collections",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
exclude_patterns = []
modindex_common_prefix = ["cristal."]


bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "label"

autosummary_generate = True
autosummary_imported_members = False
autosummary_ignore_module_all = False
autodoc_inherit_docstrings = True
numpydoc_show_class_members = False

autodoc_default_options = {
    "exclude-members": "set_inverse_transform_request, get_metadata_routing, set_output",
}

# Collection of Sphinx extensions
collections = {
    "notebooks": {
        "driver": "copy_folder",
        "source": "../examples",
        "target": "notebooks/",
        "ignore": ["*.py", ".sh"],
    }
}
nbsphinx_execute_timeout = 120

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_title = "CRISTAL Documentation"
html_logo = "_static/logo_small.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "external_links": [],
    "icon_links_label": "Icon Links",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/fgrivet/CRISTAL",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
    "show_prev_next": False,
    "search_bar_text": "Search the docs ...",
    "navigation_with_keys": False,
    "collapse_navigation": False,
    # "navigation_depth": 3,
    # "show_nav_level": 1, # Levels on the left
    "show_toc_level": 3,  # Levels on the right
    "navbar_align": "left",
    "logo": {
        "alt_text": "CRISTAL Documentation",
        "image_relative": "logo.svg",
        "image_light": "logo.svg",
        "image_dark": "logo.svg",
    },
    "header_links_before_dropdown": 6,  # Number of header before "More"
    # -- Template placement in theme layouts ----------------------------------
    "navbar_start": ["navbar-logo"],
    # Note that the alignment of navbar_center is controlled by navbar_align
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    # navbar_persistent is persistent right (even when on mobiles)
    "navbar_persistent": ["search-button"],
    "article_header_start": ["breadcrumbs"],
    "article_header_end": [],
    "article_footer_items": ["prev-next"],
    "content_footer_items": [],
    # Use html_sidebars that map page patterns to list of sidebar templates
    "primary_sidebar_end": [],
    "footer_start": ["copyright"],
    "footer_center": [],
    "footer_end": [],
}


# pylint: disable=unused-argument
def remove_module_docstring(app, what, name, obj, options, lines):
    """Removes the docstring from modules without :no-members: in automodule."""
    if what == "module" and options.get("members") is not None:
        del lines[:]


def filter_inherited_members(app, what, name, obj, options, lines):
    """Force l'inclusion des membres définis dans la classe et exclut ceux indésirables."""
    if what == "class":
        # List of modules to exclude (e.g., sklearn, numpy, pandas)
        excluded_modules = {"sklearn", "numpy", "pandas"}

        # Retrieves the names of members defined directly in the current class
        current_class_members = set(obj.__dict__.keys())

        # Initialize inclusion/exclusion sets
        include_set = set(current_class_members)  # Include ALL members defined in the class
        exclude_set = set()

        # Iterates through base classes to exclude members inherited from external modules
        for base_class in obj.__bases__:
            for member_name, member in base_class.__dict__.items():
                if hasattr(member, "__module__") and member.__module__:
                    member_module = member.__module__.split(".")[0]
                    if member_module in excluded_modules and member_name not in current_class_members:
                        exclude_set.add(member_name)

        # Clean up: if a member is in both include_list and exclude_list, exclude it.
        final_include = include_set - exclude_set

        # Update options for Sphinx
        # 1. Force the inclusion of members defined in the class
        if final_include:
            # Convertir options.members en set si c'est une chaîne ou une liste
            current_include = getattr(options, "members", [])
            if isinstance(current_include, str):
                current_include = {current_include}
            elif isinstance(current_include, list):
                current_include = set(current_include)
            elif hasattr(current_include, "__iter__"):
                current_include = set(current_include)
            else:
                current_include = set()
            current_include.update(final_include)
            options.members = list(current_include)  # Sphinx attend une liste

        # 2. Explicitly exclude unwanted members
        if exclude_set:
            current_exclude = getattr(options, "exclude_members", [])
            if isinstance(current_exclude, str):
                current_exclude = {current_exclude}
            elif isinstance(current_exclude, list):
                current_exclude = set(current_exclude)
            elif hasattr(current_exclude, "__iter__"):
                current_exclude = set(current_exclude)
            else:
                current_exclude = set()
            current_exclude.update(exclude_set)
            options.exclude_members = list(current_exclude)  # Sphinx attend une liste

        # Disable automatic hiding of undocumented members
        options.undoc_members = False


def setup(app):
    app.connect("autodoc-process-docstring", remove_module_docstring)
    app.connect("autodoc-process-docstring", filter_inherited_members)

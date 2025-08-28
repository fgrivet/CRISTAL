# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from cristal import __version__

project = "CRISTAL"
copyright = "2025, Florian Grivet"
author = "Florian Grivet"
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_nb",
    "sphinxcontrib.collections",
]

templates_path = ["_templates"]
exclude_patterns = []
modindex_common_prefix = ["cristal."]


autodoc_default_options = {"member": True, "undoc-members": True, "special-members": "__init__", "show-inheritance": True, "inherited-members": True}
autosummary_generate = False

# Collection of Sphinx extensions
collections = {
    "notebooks": {
        "driver": "copy_folder",
        "source": "../examples",
        "target": "notebooks/",
        "ignore": ["*.py", ".sh"],
    }
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_title = "CRISTAL Documentation"
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
    # "show_nav_level": 1,
    # "show_toc_level": 1,
    "navbar_align": "left",
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

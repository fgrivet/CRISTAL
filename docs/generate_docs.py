"""
Automatic generation of .rst files for Sphinx.

Usage:
    python generate_docs.py <package_name> <output_folder>

Example:
    python generate_docs.py cristal source/API

Generated structure:
    - Package/subpackage → list-table of members + toctree (without displaying the contents)
    - Leaf module → automodule with :members: (full contents)
"""

import importlib
import inspect
import os
import pkgutil
import re
import shutil
import sys
from typing import TypeVar

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_package(module) -> bool:
    return hasattr(module, "__path__")


def get_public_members(module) -> list[str]:
    """Returns __all__ if it exists, otherwise the locally defined public names."""
    if hasattr(module, "__all__"):
        return list(module.__all__)
    return [
        name
        for name, obj in inspect.getmembers(module)
        if not name.startswith("_") and (inspect.isclass(obj) or inspect.isfunction(obj)) and getattr(obj, "__module__", None) == module.__name__
    ]


def find_member_module(pkg_name: str, member_name: str, submodule_names: list[str]) -> str:
    """
    Finds the submodule that actually defines `member_name`.
    Iterates through the direct submodules and returns the first one that contains the member.
    Fallback: pkg_name itself.
    """
    for fullname in submodule_names:
        try:
            sub = importlib.import_module(fullname)
            if hasattr(sub, member_name):
                obj_in_sub = getattr(sub, member_name)
                obj_module = getattr(obj_in_sub, "__module__", None)
                # The member is defined in this submodule (or one of its children)
                if obj_module and obj_module.startswith(fullname):
                    return obj_module
        except Exception:
            continue
    return pkg_name


def get_inline_comment(module, name: str) -> str:
    """Extracts the comment `#:` from the definition line of a member."""
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError):
        return ""
    # Search for a line containing `name = ... #: <comment>`
    pattern = rf"^\s*{re.escape(name)}\s*[=:][^#\n]*#:\s*(.+)$"
    match = re.search(pattern, source, re.MULTILINE)
    return match.group(1).strip() if match else ""


import ast


def get_attribute_docstring(module, name: str) -> str:
    """
    Extracts the docstring from a module attribute according to the PEP 257 convention:
        X = ...
        \"\"\"Docstring.\"\"\"
    Also works for comments #:.
    """
    # Attempt via the AST source
    try:
        source = inspect.getsource(module)
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return ""

    for i, node in enumerate(tree.body):
        # Looking for an assignment whose name matches
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue

        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if name not in names:
            continue

        # Checks if the next node is an Expr containing a constant str
        if i + 1 < len(tree.body):
            next_node = tree.body[i + 1]
            if isinstance(next_node, ast.Expr) and isinstance(next_node.value, ast.Constant) and isinstance(next_node.value.value, str):
                return next_node.value.value.splitlines()[0].strip()

    return ""


def get_first_doc_line(obj, module=None, name: str = "") -> str:
    if isinstance(obj, TypeVar):
        if module and name:
            comment = get_inline_comment(module, name) or get_attribute_docstring(module, name)
            return comment if comment else f"TypeVar({', '.join(c.__name__ for c in obj.__constraints__)})"
        return "TypeVar"

    doc = inspect.getdoc(obj) or ""
    first = doc.splitlines()[0] if doc else ""
    if first:
        return first

    # Fallback : #: then docstring PEP 257 standalone
    if module and name:
        return get_inline_comment(module, name) or get_attribute_docstring(module, name) or "—"
    return "—"


def underline(title: str, char: str = "=") -> str:
    return f"{title}\n{char * len(title)}"


# ---------------------------------------------------------------------------
# RST Content Generators
# ---------------------------------------------------------------------------


def make_package_rst(pkg_name: str, module, submodule_names: list[str]) -> str:
    """RST for a package or sub-package: list-table + toctree."""
    title = pkg_name.replace("_", r"\_")
    lines = [underline(title), "", f".. automodule:: {pkg_name}", "   :no-members:", ""]

    # ---- list-table of public members (classes/functions of __init__) ----
    members = get_public_members(module)
    if members:
        lines += [
            "Classes",
            "-------",
            "",
            ".. list-table::",
            "   :widths: 40 60",
            "   :header-rows: 1",
            "",
            "   * - Class",
            "     - Description",
        ]
        for name in members:
            obj = getattr(module, name, None)
            desc = get_first_doc_line(obj) if obj else "—"
            # Chooses :class: or :func: depending on the type
            role = "class" if inspect.isclass(obj) else "func"
            # Full actual path of the member (may be in a submodule)
            real_module = find_member_module(pkg_name, name, submodule_names)
            full_path = f"{real_module}.{name}"
            label = f"{pkg_name}.{name}"
            # Sphinx Syntax: `Displayed label <actual target>`
            if label == full_path:
                lines.append(f"   * - :{role}:`{label}`")
            else:
                lines.append(f"   * - :{role}:`{label} <{full_path}>`")
            lines.append(f"     - {desc}")
        lines.append("")

    # ---- list-table of submodules/subpackages ----
    if submodule_names:
        lines += [
            "Submodules / Subpackages",
            "-----------------------------",
            "",
            ".. list-table::",
            "   :widths: 40 60",
            "   :header-rows: 1",
            "",
            "   * - Module",
            "     - Description",
        ]
        for fullname in submodule_names:
            try:
                sub = importlib.import_module(fullname)
                desc = get_first_doc_line(sub)
            except Exception:
                desc = "—"
            short = fullname.split(".")[-1].replace("_", r"\_")
            lines.append(f"   * - :mod:`{fullname}`")
            lines.append(f"     - {desc}")
        lines.append("")

    # ---- hidden toctree so that Sphinx can follow the links ----
    lines += [
        ".. toctree::",
        "   :hidden:",
        "",
    ]
    for fullname in submodule_names:
        lines.append(f"   {fullname}")
    lines.append("")

    return "\n".join(lines)


def make_module_rst(mod_name: str) -> str:
    """
    RST for a leaf module:
        1. List table summarizing all members (classes, functions, variables)
        2. Automodule with a complete description of each object
    """
    title = mod_name.replace("_", r"\_")
    lines = [underline(title), ""]
    special_members = "__init__, __call__"

    try:
        module = importlib.import_module(mod_name)
    except Exception:
        # In case of import failure, a minimal stub is generated
        lines += [
            f".. automodule:: {mod_name}",
            "   :members:",
            "   :show-inheritance:",
            f"   :special-members: {special_members}",
            "   :private-members: _compute_scores",
            "",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # 0. Module docstring without members                                #
    # ------------------------------------------------------------------ #
    lines += [f".. automodule:: {mod_name}", "   :no-members:", "   :no-index:", ""]

    # ------------------------------------------------------------------ #
    # 1. Collection of members by category                               #
    # ------------------------------------------------------------------ #
    classes = []
    functions = []
    variables = []

    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        # Only keep what is defined in this module
        obj_module = getattr(obj, "__module__", None)
        is_implemented_const_in_types = mod_name == "cristal.types" and name.startswith("IMPLEMENTED_")
        if obj_module and obj_module != mod_name and not is_implemented_const_in_types:
            continue

        if inspect.isclass(obj):
            classes.append((name, obj))
        elif inspect.isfunction(obj):
            functions.append((name, obj))
        elif not inspect.ismodule(obj):
            if name != "TYPE_CHECKING":
                variables.append((name, obj))

    # ------------------------------------------------------------------ #
    # 2. Summary list-table by category                                  #
    # ------------------------------------------------------------------ #
    def _table_section(section_title: str, items: list, role: str | None, lines: list[str], mod):
        lines.append(section_title)
        lines.append("-" * len(section_title))
        lines += [
            "",
            ".. list-table::",
            "   :widths: 35 65",
            "   :header-rows: 1",
            "",
            "   * - Name",
            "     - Description",
        ]
        for name, obj in items:
            desc = get_first_doc_line(obj, module=mod, name=name)
            full = f"{mod_name}.{name}"
            if role:
                lines.append(f"   * - :{role}:`{name} <{full}>`")
            else:
                lines.append(f"   * - ``{name}``")
            lines.append(f"     - {desc}")
        lines.append("")

    if classes:
        _table_section("Classes", classes, role="class", lines=lines, mod=module)
    if functions:
        _table_section("Functions", functions, role="func", lines=lines, mod=module)
    if variables:
        _table_section("Variables / Constants", variables, role="const", lines=lines, mod=module)

    # ------------------------------------------------------------------ #
    # 3. automodule — complete description of all objects                #
    # ------------------------------------------------------------------ #
    # Collect TypeVar names to exclude from automodule (numpydoc crash)
    typevar_names = [name for name, obj in variables if isinstance(obj, TypeVar)]

    lines += ["Detailed reference", "------------------", ""]

    # First the TypeVar (ArrayLike, DTypeLike)
    for name in typevar_names:
        obj = getattr(module, name)
        # Retrieves the constraints of the TypeVar
        constraints = ", ".join(f"``{c.__name__}``" for c in obj.__constraints__)
        # Get the description (#: or standalone docstring)
        desc = get_inline_comment(module, name) or get_attribute_docstring(module, name)

        lines += [
            f".. py:data:: {name}",
            f"   :type: TypeVar",
            f"   :canonical: {mod_name}.{name}",
            "",
        ]
        if desc:
            lines += [f"   {desc}", ""]
        if constraints:
            lines += [f"   Constrained to {constraints}.", ""]
        if obj.__bound__:
            # str(obj.__bound__) is of format <class 'cristal.config.detector_config.DetectorConfig'>
            inner = str(obj.__bound__).strip().replace("<class '", "").replace("'>", "")
            parts = inner.rsplit(".", 1)
            class_name = parts[-1]
            lines += [f"   Bound to :class:`{class_name} <{inner}>`.", ""]

    # Then the automodule
    lines += [
        f".. automodule:: {mod_name}",
        "   :members:",
        "   :show-inheritance:",
        f"   :special-members: {special_members}",
        "   :private-members: _compute_scores",
    ]
    # Excluding the TypeVar written before
    if typevar_names:
        lines.append(f"   :exclude-members: {', '.join(typevar_names)}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Recursive traversal of the package
# ---------------------------------------------------------------------------


def walk_package(pkg_name: str, output_dir: str):
    """
    Recursively traverses the package and generates one .rst file per module/package.
    Returns a list of all created rst filenames (relative path).
    """
    created = []

    def _walk(name: str):
        try:
            module = importlib.import_module(name)
        except Exception as e:
            print(f"  [SKIP] {name} — impossible to import : {e}")
            return

        rst_path = os.path.join(output_dir, f"{name}.rst")

        if is_package(module):
            # Retrieves direct submodules (1 level)
            submodule_names = []
            for _, subname, _ in pkgutil.iter_modules(module.__path__):
                if subname.startswith("_"):  # ignore __init__, __main__…
                    continue
                fullname = f"{name}.{subname}"
                submodule_names.append(fullname)

            content = make_package_rst(name, module, submodule_names)
            _write(rst_path, content)
            created.append(name)

            # Recursion on children
            for fullname in submodule_names:
                _walk(fullname)
        else:
            content = make_module_rst(name)
            _write(rst_path, content)
            created.append(name)

    _walk(pkg_name)
    return created


def _write(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [OK]  {path}")


# ---------------------------------------------------------------------------
# Main func
# ---------------------------------------------------------------------------


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    pkg_name = sys.argv[1]  # ex: my_package
    output_dir = sys.argv[2]  # ex: source/API

    # Remove dir to recreate it
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    print(f"\n=== Generating .rst for '{pkg_name}' in '{output_dir}' ===\n")

    # The package must be importable; the current directory is added to the path.
    sys.path.insert(0, os.getcwd())

    created = walk_package(pkg_name, output_dir)

    print(f"\n✓ {len(created)} file(s) .rst generated in '{output_dir}'.")


if __name__ == "__main__":
    main()
    import subprocess

    subprocess.run(["make", "clean"], check=True, cwd=os.getcwd(), shell=True)
    subprocess.run(["make", "html"], check=True, cwd=os.getcwd(), shell=True)

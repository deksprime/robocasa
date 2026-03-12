import os

assets_root = os.path.join(os.path.dirname(__file__), "assets")

# Extensions directory — for custom objects, fixtures, scenes, environments.
# Default: sibling directory of wherever robocasa is installed.
# e.g. if robocasa is at /tmp/robocasa, extensions default to /tmp/robocasa-extensions
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
extensions_root = os.environ.get(
    "ROBOCASA_EXTENSIONS_DIR",
    os.path.join(os.path.dirname(_pkg_root), "robocasa-extensions"),
)
extensions_assets = os.path.join(extensions_root, "assets")


def get_asset_paths(subdir: str) -> list[str]:
    """Return list of asset directories to search for a given subdirectory.

    Checks both the built-in assets_root and the extensions directory.
    Upstream assets take priority (listed first).

    Args:
        subdir: subdirectory name, e.g. "objects", "fixtures", "scenes"

    Returns:
        List of existing directory paths to search.
    """
    paths = []
    builtin = os.path.join(assets_root, subdir)
    if os.path.isdir(builtin):
        paths.append(builtin)
    ext = os.path.join(extensions_assets, subdir)
    if os.path.isdir(ext):
        paths.append(ext)
    return paths

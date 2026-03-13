"""CLI command to list installed model-clinic plugins.

Usage:
    model-clinic plugins
"""

import argparse

from model_clinic._plugins import list_plugins, load_plugins
from model_clinic.clinic import REGISTRY


def main():
    parser = argparse.ArgumentParser(
        prog="model-clinic plugins",
        description="List installed model-clinic plugins",
    )
    parser.parse_args()

    plugins = list_plugins()

    if not plugins:
        print("Installed plugins:")
        print("  (none installed)")
        print()
        print("To create a plugin: model-clinic new-plugin <name>")
        return

    # Load plugins to count detectors each one contributes
    before_detectors = set(REGISTRY._detectors.keys())
    loaded = load_plugins(REGISTRY)

    print("Installed plugins:")
    for p in plugins:
        version_str = f"v{p['version']}" if p["version"] else "v?.?.?"
        # Figure out which detectors this plugin added
        # (rough: we can't easily attribute per-plugin, so show total)
        line = f"  {p['name']:<24s} {version_str:<10s}"
        if p["module"]:
            line += f" ({p['module']})"
        print(line)

    after_detectors = set(REGISTRY._detectors.keys())
    new_detectors = after_detectors - before_detectors
    if new_detectors:
        print()
        print(f"Plugin detectors: {', '.join(sorted(new_detectors))}")
    print()


if __name__ == "__main__":
    main()

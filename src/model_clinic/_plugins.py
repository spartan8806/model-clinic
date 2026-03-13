"""Plugin discovery and loading for model-clinic.

Plugins are Python packages that register via setuptools entry points:

    # In plugin's pyproject.toml:
    [project.entry-points."model_clinic.plugins"]
    my_plugin = "my_package.clinic_plugin:register"

The register(registry) function receives the ConditionRegistry and can call
registry.register(...) to add custom detectors and prescribers.
"""

import importlib.metadata
import warnings


_PLUGIN_GROUP = "model_clinic.plugins"
_plugins_loaded = False


def load_plugins(registry=None):
    """Discover and load all installed model-clinic plugins.

    Parameters
    ----------
    registry : ConditionRegistry, optional
        Registry to pass to each plugin's register() function.
        If None, uses the global REGISTRY from model_clinic.clinic.

    Returns
    -------
    list[str]
        Names of successfully loaded plugins.
    """
    global _plugins_loaded
    if registry is None:
        from model_clinic.clinic import REGISTRY
        registry = REGISTRY

    loaded = []
    for ep in importlib.metadata.entry_points(group=_PLUGIN_GROUP):
        try:
            fn = ep.load()
            fn(registry)
            loaded.append(ep.name)
        except Exception as e:
            warnings.warn(f"Failed to load model-clinic plugin '{ep.name}': {e}")
    _plugins_loaded = True
    return loaded


def plugins_loaded():
    """Return True if load_plugins() has already been called."""
    return _plugins_loaded


def reset_plugins_loaded():
    """Reset the loaded flag (for testing)."""
    global _plugins_loaded
    _plugins_loaded = False


def list_plugins():
    """List all installed model-clinic plugins with metadata.

    Returns
    -------
    list[dict]
        Each dict has keys: name, module, distribution, version.
    """
    plugins = []
    for ep in importlib.metadata.entry_points(group=_PLUGIN_GROUP):
        info = {
            "name": ep.name,
            "module": ep.value,
            "distribution": None,
            "version": None,
        }
        # Try to get distribution info
        try:
            if hasattr(ep, "dist") and ep.dist is not None:
                info["distribution"] = ep.dist.name
                info["version"] = ep.dist.metadata["Version"]
        except Exception:
            pass
        plugins.append(info)
    return plugins

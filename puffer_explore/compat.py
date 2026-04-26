"""PufferLib version compatibility layer.

Auto-detects PufferLib 3.0 vs 4.0 and exposes a unified API for:
- Importing PuffeRL trainer class
- Loading default config

Usage:
    from puffer_explore.compat import PuffeRL, load_config, PUFFERLIB_VERSION
"""

from __future__ import annotations


def _detect_pufferlib_version() -> str:
    """Detect installed PufferLib version. Returns "3.0", "4.0", or "unknown"."""
    try:
        import pufferlib
        version = getattr(pufferlib, "__version__", None)
        # PufferLib 3.0 stores __version__ as a float (3.0), 4.0 as string
        version_str = str(version) if version is not None else "0"
        if version_str.startswith("4"):
            return "4.0"
        if version_str.startswith("3"):
            return "3.0"
    except ImportError:
        return "unknown"
    # Fall back to module probing
    try:
        import pufferlib.torch_pufferl  # noqa: F401
        return "4.0"
    except ImportError:
        pass
    try:
        import pufferlib.pufferl  # noqa: F401
        return "3.0"
    except ImportError:
        pass
    return "unknown"


PUFFERLIB_VERSION = _detect_pufferlib_version()


def get_pufferl():
    """Return the PuffeRL trainer class for the installed PufferLib version."""
    if PUFFERLIB_VERSION == "4.0":
        from pufferlib.torch_pufferl import PuffeRL
        return PuffeRL
    elif PUFFERLIB_VERSION == "3.0":
        from pufferlib.pufferl import PuffeRL
        return PuffeRL
    raise RuntimeError(
        "Could not detect PufferLib. Install pufferlib (3.x or 4.x)."
    )


def load_config(env_name: str) -> dict:
    """Load PufferLib's default config for the given env (3.0 + 4.0)."""
    if PUFFERLIB_VERSION == "4.0":
        from pufferlib.pufferl import load_config as _load
        return _load(env_name)
    elif PUFFERLIB_VERSION == "3.0":
        from pufferlib.pufferl import load_config as _load
        return _load(env_name)
    raise RuntimeError("PufferLib not installed")


# Re-export PuffeRL eagerly for convenience
try:
    PuffeRL = get_pufferl()
except RuntimeError:
    PuffeRL = None


"""pyFMM package initializer.

Auto-discovers subpackages (moment, plotting, utils, ...) and re-exports
their public names. New modules added to the subpackages will be picked up
automatically if they expose public symbols.
"""

import importlib
import pkgutil
import inspect

__all__ = []

# Discover and import subpackages automatically (moment, plotting, utils, ...)
# Build an ordered list: each subpackage name followed by its public symbols
ordered = []
for finder, name, ispkg in pkgutil.iter_modules(__path__):
    if not ispkg:
        continue
    subpkg = importlib.import_module(f".{name}", __name__)
    # expose the subpackage module object
    globals()[name] = subpkg
    ordered.append(name)

    # Pull public names from subpackage (respect subpkg.__all__ if present)
    try:
        public = list(getattr(subpkg, "__all__"))
    except Exception:
        public = [n for n in dir(subpkg) if not n.startswith("_")]

    for n in public:
        # avoid overwriting an existing top-level name
        if n in globals():
            continue
        try:
            obj = getattr(subpkg, n)
        except AttributeError:
            continue
        globals()[n] = obj
        ordered.append(n)

# assign ordered list to __all__ (will be filtered later)
__all__ = ordered

# Import whatever plotting declares as public; fall back to all non-private names.
# Keep __all__ unique and ordered
_seen = []
__all__ = [x for x in __all__ if not (x in _seen or _seen.append(x))]

# Filter exports: only expose subpackages (modules under pyFMM) and
# functions/classes defined in pyFMM submodules. This avoids leaking
# implementation helpers like importlib, pkgutil, inspect, numpy, etc.
_filtered = []
for name in __all__:
    obj = globals().get(name, None)
    if obj is None:
        continue
    # keep modules that are part of this package
    if inspect.ismodule(obj):
        if getattr(obj, '__package__', '').startswith(__name__):
            _filtered.append(name)
        continue
    # keep functions and classes defined in pyFMM.*
    if inspect.isfunction(obj) or inspect.isclass(obj):
        modname = getattr(obj, '__module__', '')
        if modname.startswith(__name__ + '.'):
            _filtered.append(name)

__all__ = _filtered

# clean temporary names from module globals to avoid leaking them
for _tmp in ("finder", "name", "ispkg", "subpkg", "public", "ordered", "_seen", "_filtered", "obj", "modname", "n"):
    globals().pop(_tmp, None)

# Remove helper modules from the top-level namespace so they don't show up
# in `dir(pyFMM)` for end users.
for _helper in ("importlib", "pkgutil", "inspect"):
    globals().pop(_helper, None)
"""pot_eval subpackage auto-importer.

Automatically imports modules under `pyFMM.pot_eval` and re-exports their
public symbols. Module objects are also added to __all__ so the top-level
package may re-export them if desired.
"""
import importlib
import pkgutil
import inspect

__all__ = []

for finder, name, ispkg in pkgutil.iter_modules(__path__):
    if ispkg:
        continue
    mod = importlib.import_module(f".{name}", __name__)
    globals()[name] = mod
    # expose the module name so top-level pyFMM can re-export it
    try:
        __all__.append(name)
    except Exception:
        pass

    names = getattr(mod, "__all__", None)
    if names is None:
        names = [n for n in dir(mod) if not n.startswith("_")]

    for n in names:
        try:
            obj = getattr(mod, n)
        except AttributeError:
            continue
        # export only functions and classes defined in the module itself
        if (inspect.isfunction(obj) or inspect.isclass(obj)) and getattr(obj, "__module__", None) == mod.__name__:
            existing = globals().get(n)
            if existing is not None and inspect.ismodule(existing):
                continue
            globals()[n] = obj
            __all__.append(n)

# remove duplicates while preserving order
_seen = []
__all__ = [x for x in __all__ if not (x in _seen or _seen.append(x))]

# clean temporary names from module globals to avoid leaking them
for _tmp in ("finder", "name", "ispkg", "mod", "names", "obj", "_seen"):
    globals().pop(_tmp, None)

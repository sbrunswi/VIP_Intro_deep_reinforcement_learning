"""
Custom factory that selects either the stock no_ros2 env or Erin's local copy.
This allows you to experiment inside the "Individual Workspace/Erin" folder without
modifying the upstream package.
"""
from importlib import import_module

from no_ros2.environments.env_factory import make_pylon_env as _base_factory


def make_pylon_env(use_ros2=False, use_erin=False, **kwargs):
    """Return a wrapped pylon env.

    If ``use_erin`` is True then we import ``Erin.mock_pylon_env.MockPylonRacingEnv``
    and wrap it.  Otherwise we fall back to the normal no_ros2 factory.  This
    mirrors the interface used by the existing scripts, so you can simply pass
    ``use_erin=True`` from your command line or training code.
    """
    if use_erin:
        mod = import_module("Erin.mock_pylon_env")
        base = mod.MockPylonRacingEnv(**kwargs)
    else:
        base = _base_factory(use_ros2=use_ros2, **kwargs)
    # we still wrap with the original PylonRacingWrapper so the interface stays the same
    from no_ros2.environments.pylon_wrapper import PylonRacingWrapper

    return PylonRacingWrapper(base)

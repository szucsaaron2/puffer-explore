#!/usr/bin/env python3
"""Probe PufferLib internals to discover exact buffer attribute names.

Run this INSIDE the Docker container to inspect the trainer object
and figure out the correct attribute names for integration.py.
"""

import sys


def probe_pufferlib():
    """Create a PufferLib trainer and inspect its attributes."""
    try:
        import pufferlib
        print(f"PufferLib version: {pufferlib.__version__}")
    except ImportError:
        print("ERROR: PufferLib not installed!")
        sys.exit(1)

    # Try to import the main training module
    try:
        from pufferlib import pufferl
        print(f"pufferl module loaded: {pufferl}")
    except ImportError:
        print("No pufferlib.pufferl module")
        try:
            from pufferlib import cleanrl
            print(f"pufferlib.cleanrl module loaded: {cleanrl}")
        except ImportError:
            print("No pufferlib.cleanrl module either")

    # List all public modules
    print("\n=== PufferLib public modules ===")
    for attr in sorted(dir(pufferlib)):
        if not attr.startswith("_"):
            print(f"  pufferlib.{attr}")

    # Try to set up a simple environment
    print("\n=== Attempting Atari env ===")
    try:
        import pufferlib.environments.atari as atari_env
        print(f"Atari env module: {atari_env}")
        print(f"  dir: {[a for a in dir(atari_env) if not a.startswith('_')]}")
    except Exception as e:
        print(f"Atari env import failed: {e}")

    # Try CartPole as simplest env
    print("\n=== Attempting simple env setup ===")
    try:
        import gymnasium as gym
        env = gym.make("CartPole-v1")
        print(f"CartPole obs space: {env.observation_space}")
        print(f"CartPole act space: {env.action_space}")
        env.close()
    except Exception as e:
        print(f"CartPole failed: {e}")

    # Try to create a PufferLib trainer
    print("\n=== Attempting PufferLib trainer creation ===")
    try:
        # PufferLib 3.x API
        from pufferlib import pufferl
        config = pufferl.load_config("breakout")
        print(f"Config loaded: {list(config.keys())}")

        # Create trainer
        trainer = pufferl.make(**config)
        print(f"Trainer type: {type(trainer)}")
        print("\n=== Trainer attributes (buffers) ===")
        for attr in sorted(dir(trainer)):
            if not attr.startswith("_"):
                val = getattr(trainer, attr, "?")
                t = type(val).__name__
                if hasattr(val, "shape"):
                    print(f"  {attr}: {t} shape={val.shape}")
                elif isinstance(val, (int, float, str, bool)):
                    print(f"  {attr}: {t} = {val}")
                elif callable(val):
                    print(f"  {attr}: method")
                else:
                    print(f"  {attr}: {t}")

        # Try running one evaluate step
        print("\n=== Running evaluate() ===")
        trainer.evaluate()
        print("evaluate() succeeded!")

        # Now check buffer attributes again
        print("\n=== Post-evaluate trainer attributes ===")
        for attr in sorted(dir(trainer)):
            if not attr.startswith("_"):
                val = getattr(trainer, attr, "?")
                if hasattr(val, "shape"):
                    print(f"  {attr}: shape={val.shape} dtype={val.dtype}")

        trainer.close()
        print("\nTrainer closed successfully.")

    except Exception as e:
        print(f"Trainer creation failed: {e}")
        import traceback
        traceback.print_exc()

    # Try PufferLib 4.x API
    print("\n=== Trying PufferLib 4.x API ===")
    try:
        from pufferlib import train
        print(f"pufferlib.train module: {train}")
    except ImportError:
        print("No pufferlib.train module (not 4.x)")


def probe_torch():
    """Check torch + CUDA availability."""
    import torch
    print("\n=== PyTorch ===")
    print(f"Version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


if __name__ == "__main__":
    probe_torch()
    probe_pufferlib()

"""Command-line entry point for training and exporting the three physics U-Nets."""

try:
    from .neural_physics import _main
except ImportError:
    from neural_physics import _main


if __name__ == "__main__":
    _main()

# IDM Data Processing

This folder provides utilities for:

- converting LIBERO demonstrations into CALVIN-style episodes;
- converting Open X-Embodiment TFDS datasets into videos;
- extracting CoTracker flow targets;
- extracting SAM semantic targets.

Most scripts expose `--data_root`, `--save_path`, and `--checkpoint` arguments. The repository root `.env.example` lists the matching environment variables.

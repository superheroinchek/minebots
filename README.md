# MineBots

MineBots is a high performance Minecraft-inspired automation agent built on top of modern transformer models. The agent combines a long term planner, episodic memory and a fast visual processing stack to navigate a lightweight simulated world autonomously.

## Features

- **Transformer powered control loop** – Uses Hugging Face models for rapid action generation with optional 8-bit loading and FlashAttention.
- **Juven simulation environment** – Runs on a lightweight grid world with procedurally scattered resources, removing the need for a live Minecraft client.
- **Vision pipeline** – Normalizes captured frames for downstream multimodal reasoning.
- **Episodic memory** – Maintains a rolling buffer of observations and executed actions for plan adaptation.
- **Adaptive planning** – Generates a goal driven plan that reprioritizes based on memory summaries.
- **Scene snapshots** – Stores ASCII overviews and rendered frames summarizing the approximate state of the simulated world.

## Quick start

1. Install dependencies (including YAML support):
   ```bash
   pip install -U transformers torch sentence-transformers pillow numpy pyyaml
   ```
2. (Optional) Create a YAML configuration file:
   ```bash
   cp config.example.yaml minebots.yaml
   # edit minebots.yaml with your preferred settings
   ```
3. Export configuration if needed:
   ```bash
   export MINEBOTS_SIMULATION='{"world_size": 20, "resource_clusters": 8}'
   export MINEBOTS_TRANSFORMER='{"model_name": "microsoft/phi-2", "temperature": 0.1}'
   ```
4. Run the agent:
   ```bash
   python -m minebots.main --log-level INFO --config minebots.yaml
   ```

The agent will launch the Juven simulation, continually refine its plan, and execute high value actions directly against the simulated world. Each cycle records an approximate view of the surroundings using both images and ASCII summaries.

## Configuration

Configuration can be provided through environment variables containing JSON objects,
or via a YAML file supplied to the CLI with `--config`.

### YAML configuration

YAML configuration files accept the following structure (see `config.example.yaml`):

```yaml
simulation:
  world_size: 16
  resource_clusters: 6
  vision_resolution: [192, 192]
vision:
  frame_store: ./frames
  downscale_factor: 2
transformer:
  model_name: microsoft/phi-2
memory:
  max_events: 2048
goal: "Complete the Minecraft playthrough efficiently and safely."
max_cycles: 50
```

### Environment variables

When present, environment variables override any values defined in YAML:

- `MINEBOTS_SIMULATION` – Juven world settings (`world_size`, `resource_clusters`, `max_cluster_size`, `alert_threshold`, `vision_resolution`, `vision_noise`, `seed`).
- `MINEBOTS_VISION` – Vision parameters (`frame_store`, `max_history`, `downscale_factor`).
- `MINEBOTS_TRANSFORMER` – Transformer options (`model_name`, `max_new_tokens`, `temperature`, `use_8bit`, `use_flash_attention`).
- `MINEBOTS_MEMORY` – Memory options (`max_events`, `summary_model_name`).
- `MINEBOTS_GOAL` – Custom primary goal string.

## Development notes

- The default controller uses the CPU when CUDA is not available.
- The vision system stores frames in `./frames` unless configured otherwise.
- Optional dependencies such as `torch`, `transformers`, `sentence-transformers`, and `pillow` are loaded lazily and only required when the corresponding subsystem is used.
- Vision snapshots are written to the configured `frame_store` directory and include both saved images and ASCII overviews for remote monitoring.

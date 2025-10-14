#!/usr/bin/env python3
"""
Debug script to investigate WandB config structure
"""

import wandb
from wandb import Api
import json

# Initialize WandB API
api = Api()

project_name = "danastar"
group_name = "DanaStar_MK4_Small_Sweep_formula9"

print(f"Investigating runs from project: {project_name}, group: {group_name}")
print("=" * 80)

# Get all runs in the specified group
runs = api.runs(f"ep-rmt-ml-opt/{project_name}", filters={"group": group_name})

# Look at first few runs
for i, run in enumerate(runs):
    if i >= 3:  # Only examine first 3 runs
        break

    print(f"\nRun {i+1}: {run.name}")
    print("-" * 80)

    # Examine the config object
    config = run.config
    print(f"Type of config: {type(config)}")
    print(f"Config value: {config}")

    # Check if it has a _items attribute (which WandB configs sometimes have)
    if hasattr(config, '_items'):
        print(f"Config has _items attribute: {config._items}")

    # Check if it has keys() method
    if hasattr(config, 'keys'):
        print(f"Config has keys() method")
        try:
            print(f"Config keys: {list(config.keys())[:10]}")  # First 10 keys
        except Exception as e:
            print(f"Error calling keys(): {e}")

    # Try to access it as dict
    print(f"\nAttempting dict access methods:")
    try:
        print(f"  config['lr'] = {config['lr']}")
    except Exception as e:
        print(f"  config['lr'] failed: {type(e).__name__}: {e}")

    try:
        print(f"  config.get('lr') = {config.get('lr')}")
    except Exception as e:
        print(f"  config.get('lr') failed: {type(e).__name__}: {e}")

    # Check for alternative attributes
    print(f"\nChecking run object attributes:")
    print(f"  dir(run): {[attr for attr in dir(run) if not attr.startswith('_')][:20]}")

    # Check if there's a raw_config or something similar
    if hasattr(run, 'raw_config'):
        print(f"  run.raw_config type: {type(run.raw_config)}")
        print(f"  run.raw_config: {run.raw_config}")

    # Check json_config
    if hasattr(run, 'json_config'):
        print(f"\n  run.json_config type: {type(run.json_config)}")
        print(f"  run.json_config (first 500 chars): {str(run.json_config)[:500]}")
        try:
            # Try to parse the config string as JSON
            config_dict = json.loads(config)
            print(f"\n  Successfully parsed config as JSON!")
            print(f"  Type after parsing: {type(config_dict)}")
            print(f"  lr value: {config_dict.get('lr', {}).get('value', 'NOT FOUND')}")
        except Exception as e:
            print(f"  Failed to parse config as JSON: {e}")

    # Try to get summary too
    summary = run.summary
    print(f"\nType of summary: {type(summary)}")
    print(f"Summary dir: {[attr for attr in dir(summary) if not attr.startswith('_')][:20]}")

    # Check if it has _json_dict or similar
    if hasattr(summary, '_json_dict'):
        print(f"Summary has _json_dict: {type(summary._json_dict)}")
    if hasattr(summary, '_dict'):
        print(f"Summary has _dict: {type(summary._dict)}, value (first 200 chars): {str(summary._dict)[:200]}")

    if hasattr(summary, 'keys'):
        try:
            print(f"Summary keys (first 10): {list(summary.keys())[:10]}")
        except Exception as e:
            print(f"Error getting summary keys: {e}")

    # Try direct attribute access
    try:
        val_loss_attr = getattr(summary, 'val/loss', None)
        print(f"summary['val/loss'] via getattr: {val_loss_attr}")
    except Exception as e:
        print(f"Error with getattr: {e}")

    # Try item access
    try:
        val_loss_item = summary['val/loss']
        print(f"summary['val/loss'] via __getitem__: {val_loss_item}")
    except Exception as e:
        print(f"Error with __getitem__: {e}")

print("\n" + "=" * 80)
print("Investigation complete")

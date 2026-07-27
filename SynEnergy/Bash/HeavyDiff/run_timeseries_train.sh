#!/bin/bash
cd "$(dirname "$0")/../.."
python main.py --name power_heavy_50000 --config_file Config/Extreme/power.yaml --train --heavy --condition

#!/bin/bash
cd "$(dirname "$0")/../.."
python main.py --name power --config_file Config/Extreme/power.yaml --train --condition

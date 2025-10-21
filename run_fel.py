# tools/fel_run.py
import argparse, yaml, sys
from src.loop.sim_runner import run_sim

def main():
    ap = argparse.ArgumentParser("FEL simulator")
    ap.add_argument("--config", required=True, help="Path to FEL YAML config")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    run_sim(cfg)

if __name__ == "__main__":
    sys.exit(main())


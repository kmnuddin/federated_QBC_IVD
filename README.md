# FEL Simulation (Rest-only Evaluation)

Run:
```bash
pip install -r requirements.txt
python run_sim.py --config configs/sim_meta_summary.yaml
```
Inputs expected in `data/path_embed_rsna.csv` and `data/path_embed_mendeley.csv`.
Each CSV must have: `embedding_path`, `id`, `ivd_level`.

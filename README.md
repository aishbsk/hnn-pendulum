## Hamiltonian Pendulum

Learn a Hamiltonian from data for a single pendulum and show near-constant energy in long rollouts.

``` bash
hnn-pendulum/
├── README.md
├── requirements.txt
├── src/
│   ├── physics.py        
│   ├── data.py            
│   ├── models/
│   │   ├── hnn.py         
│   │   └── mlp.py          
│   ├── train.py            
│   └── eval.py             
├── scripts/
│   ├── gen_data.py        
│   └── make_figures.py     
├── data/                  
├── reports/                
└── tests/
    ├── test_physics.py     
    └── test_models.py     
```

## Quickstart

```bash
uv python install 3.11
uv run -m scripts.gen_data
uv venv .venv
uv pip install -r requirements.txt
```

```bash
# 1) Generate data (clean set, T=200, dt=0.05)
uv run -m scripts.gen_data \
  --outdir data --n-train 200 --n-test 50 \
  --length 200 --dt 0.05 --eps 0.0 --noise 0.0

# 2) Train HNN 
uv run -m src.train \
  --model hnn --sigma 0.0 --angle-embed \
  --epochs 2000 --lr 1e-3 --batch 1024 --seed 0

# 3) Evaluate long rollouts
uv run -m src.eval \
  --model hnn --sigma 0.0 --angle-embed \
  --horizon 200 --dt 0.05
```
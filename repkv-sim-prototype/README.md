# RepKV Simulation Prototype

> PROTOTYPE — this project checks whether RepKV's control logic is worth pursuing. It is not a hardware-accurate performance model.

## Question

When multi-turn chat sessions share limited edge compute, memory, and network capacity, can proactive KV preparation increase TTFT SLO-goodput with less resource use than full pre-copying? The prototype compares three policies on the same generated workload:

- `on_demand`: restore KV only after a request arrives;
- `eager_full`: predict the next request and copy its full KV to one backup node;
- `repkv`: predict the next request, prepare only the KV needed to make one backup node useful, and reclaim low-value idle KV when memory is full.

## Run

```bash
python3 run.py
```

The command prints one comparison table and writes per-policy summaries and per-request records to `outputs/`.

Useful options:

```bash
python3 run.py --seeds 1,2,3,4,5
python3 run.py --nodes 4 --sessions 80 --horizon 180
python3 run.py --prediction-error 8 --hbm-blocks 180
python3 run.py --help
```

## What the prototype models

- multi-turn sessions whose KV grows after every turn;
- a queue and KV cache at each edge node;
- request-time KV transfer or recomputation;
- noisy predictions of the next request time and prompt size;
- background bandwidth and background recomputation limits;
- proactive preparation in small KV batches;
- HBM pressure and idle-KV reclamation;
- TTFT, SLO-goodput, transfer blocks per successful request, recomputed blocks per successful request, memory residency per successful request, and unused preparation.

## What it does not establish

The simulator cannot prove absolute latency or deployment speedup. Its rates must later be replaced by measurements from the Ascend 910B testbed. The useful outputs at this stage are comparative trends, failure cases, sensitivity to prediction error, and whether the proposed controller behaves as intended.

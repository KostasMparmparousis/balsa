# NEO/Balsa Experiment Reproduction Guide

This document provides the specific commands required to replicate the experiments presented in our paper using the NEO/Balsa optimizer.

-   All paths should be relative to the root of the repository.
-   `<path/to/huggingface/models/>` refers to the directory where you downloaded the pretrained model checkpoints.

---

## (E1) End-to-End Performance & Value Model Fidelity

### Training

```bash
python run.py --run Neo_JOBRandomSplit1 --local \
    --workload_dir ../../experiments/experiment1/job/train/ \
    --test_workload_dir ../../experiments/experiment1/job/test/ \
    --workload_order ../../experiments/experiment1/job/train/queries/ascending_latency_query_order.txt \
    --target_checkpoint_dir ../../experiments/experiment1/job/train/models/neo/
```

### Testing

python test_model.py --run Neo_JOBRandomSplit1 \
    --checkpoint_dir <path/to/huggingface/models/>/E1/neo_job/ \
    --workload_dir ../../experiments/experiment1/job/test/ \
    --test_workload_dir ../../experiments/experiment1/job/test/


## (E2) Sensitivity & Execution Stability

### Training

```bash
# Add your training command for E2 here
```

### Testing
```bash
# Add your testing command for E2 here
```

---

## (E3) Learning Trajectory & Convergence

### Training

```bash
# Add your training command for E2 here
```

### Testing
```bash
# Add your testing command for E2 here
```
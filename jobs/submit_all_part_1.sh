#!/bin/bash
# filepath: jobs/submit_all_experiments.sh

echo "Submitting all experiments..."
echo "Total: 16 jobs (2 models x 2 datasets x 4 losses)"
echo ""

# U-Net experiments
echo "=== U-Net Experiments ==="
for DATASET in PH2 DRIVE; do
    for LOSS in bce dice focal weighted_bce; do
        echo "Submitting: U-Net + ${DATASET} + ${LOSS}"
        DATASET=$DATASET LOSS=$LOSS bsub -app c02516_1g.10gb < jobs/train_u_net.sh
        sleep 1
    done
done

echo ""

# EncDec experiments
echo "=== EncDec Experiments ==="
for DATASET in PH2 DRIVE; do
    for LOSS in bce dice focal weighted_bce; do
        echo "Submitting: EncDec + ${DATASET} + ${LOSS}"
        DATASET=$DATASET LOSS=$LOSS bsub -app c02516_1g.10gb < jobs/train_enc_dec.sh
        sleep 1
    done
done

echo ""
echo "All 16 jobs submitted!"
echo ""
echo "Monitor with: bstat"
echo "Check outputs in: outputs/"
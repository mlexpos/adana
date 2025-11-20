#!/bin/bash

# Multi-regime runtime analysis for BigHead AdamW
#python bighead_adamw_multiregime_runtime.py

python compare_scaling_rules.py --scaling-rules Enoki_Scaled --optimizers adamw mk4 d-muon manau ademamix --fit-relative-to-adamw
python compare_scaling_rules.py --scaling-rules Eryngii_Scaled --optimizers adamw mk4 d-muon ademamix --fit-relative-to-adamw
python compare_scaling_rules.py --scaling-rules Enoki Enoki_Scaled Eryngii Eryngii_Scaled --optimizers adamw mk4 d-muon manau ademamix --fit-relative-to-adamw


python bighead_lr_scaling.py --scaling-rule Enoki_Scaled --optimizer adamw --target-omega 4.0 --top-k 7
python bighead_lr_scaling.py --scaling-rule Enoki_Scaled --optimizer mk4 --target-omega 4.0 --top-k 7 
python bighead_lr_scaling.py --scaling-rule Enoki_Scaled --optimizer d-muon --target-omega 4.0 --top-k 7
python bighead_lr_scaling.py --scaling-rule Enoki_Scaled --optimizer manau --target-omega 4.0 --top-k 7
python bighead_lr_scaling.py --scaling-rule Enoki_Scaled --optimizer ademamix --target-omega 4.0 --top-k 7

python bighead_lr_scaling.py --scaling-rule Eryngii_Scaled --optimizer adamw --target-omega 4.0 --top-k 7
python bighead_lr_scaling.py --scaling-rule Eryngii_Scaled --optimizer mk4 --target-omega 4.0 --top-k 7
python bighead_lr_scaling.py --scaling-rule Eryngii_Scaled --optimizer d-muon --target-omega 4.0 --top-k 7
python bighead_lr_scaling.py --scaling-rule Eryngii_Scaled --optimizer manau --target-omega 4.0 --top-k 7
python bighead_lr_scaling.py --scaling-rule Eryngii_Scaled --optimizer ademamix --target-omega 4.0 --top-k 7

# python compare_scaling_rules.py --scaling-rules BigHead --optimizers adamw mk4 d-muon manau ademamix
python compare_scaling_rules.py --scaling-rules Enoki_Scaled --optimizers adamw mk4 d-muon manau ademamix
python compare_scaling_rules.py --scaling-rules Eryngii_Scaled --optimizers adamw mk4 d-muon ademamix
python compare_scaling_rules.py --scaling-rules Enoki Enoki_Scaled Eryngii Eryngii_Scaled --optimizers adamw mk4


# python bighead_lr_scaling.py --scaling-rule BigHead --optimizer adamw --target-omega 4.0 --top-k 5
# python bighead_lr_scaling.py --scaling-rule BigHead --optimizer mk4 --target-omega 4.0 --top-k 5
# python bighead_lr_scaling.py --scaling-rule BigHead --optimizer d-muon --target-omega 4.0 --top-k 5
# python bighead_lr_scaling.py --scaling-rule BigHead --optimizer manau --target-omega 4.0 --top-k 5
# python bighead_lr_scaling.py --scaling-rule BigHead --optimizer ademamix --target-omega 4.0 --top-k 5


# python bighead_lr_scaling.py --scaling-rule Enoki --optimizer adamw --target-omega 4.0 --top-k 5
# python bighead_lr_scaling.py --scaling-rule Enoki --optimizer mk4 --target-omega 4.0 --top-k 5 --target-clipsnr 2.0
# python bighead_lr_scaling.py --scaling-rule Enoki --optimizer d-muon --target-omega 4.0 --top-k 5
# python bighead_lr_scaling.py --scaling-rule Enoki --optimizer manau --target-omega 4.0 --top-k 5
# python bighead_lr_scaling.py --scaling-rule Enoki --optimizer ademamix --target-omega 4.0 --top-k 5

# python bighead_lr_scaling.py --scaling-rule Eryngii --optimizer adamw --target-omega 4.0 --top-k 5
# python bighead_lr_scaling.py --scaling-rule Eryngii --optimizer d-muon --target-omega 4.0 --top-k 5
# python bighead_lr_scaling.py --scaling-rule Eryngii --optimizer manau --target-omega 4.0 --top-k 5
# python bighead_lr_scaling.py --scaling-rule Eryngii --optimizer ademamix --target-omega 4.0 --top-k 5
# python bighead_lr_scaling.py --scaling-rule Eryngii --optimizer mk4 --target-omega 4.0 --top-k 5

# python compare_scaling_rules.py --scaling-rules BigHead --optimizers adamw mk4 d-muon manau ademamix
# python compare_scaling_rules.py --scaling-rules Enoki --optimizers adamw mk4 d-muon manau ademamix
# python compare_scaling_rules.py --scaling-rules Eryngii --optimizers adamw mk4 d-muon ademamix


# python bighead_lr_scaling.py --scaling-rule Eryngii --optimizer adamw --target-omega 4.0 --top-k 5
# python bighead_lr_scaling.py --scaling-rule Eryngii --optimizer d-muon --target-omega 4.0 --top-k 5
# python bighead_lr_scaling.py --scaling-rule Eryngii --optimizer manau --target-omega 4.0 --top-k 5
# python bighead_lr_scaling.py --scaling-rule Eryngii --optimizer ademamix --target-omega 4.0 --top-k 5
# python bighead_lr_scaling.py --scaling-rule Eryngii --optimizer mk4 --target-omega 4.0 --top-k 5
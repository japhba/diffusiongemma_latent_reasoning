#!/bin/bash
cd "$(dirname "$0")"
python3 suscept.py --phase both --n-clean 5 --rhos 0,0.25,0.5,0.75,1.0 --corr-seeds 5 >> run.log 2>&1
if [ -f anchor_dg.py ]; then python3 anchor_dg.py >> anchor_dg.log 2>&1; fi
echo CHAIN_DONE >> run.log

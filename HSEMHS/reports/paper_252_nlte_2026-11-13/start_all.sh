#!/bin/bash
# waits for the NLTE table, then starts every NLTE case at once (one MHS run per node)
until grep -q -e "report written" /mn/stornext/d9/data/harshm/MHS/HSEMHS/out/nlte_eos_1d_2026-11-13/build_nlte_table_v4.log; do sleep 15; done
grep -q "saved" /mn/stornext/d9/data/harshm/MHS/HSEMHS/out/nlte_eos_1d_2026-11-13/build_nlte_table_v4.log || exit 1
s() { ssh -o BatchMode=yes $1 "nohup bash -c '$2' > /dev/null 2>&1 < /dev/null &"; }
s hyades14 "/mn/stornext/d9/data/harshm/MHS/HSEMHS/out/paper_252_nlte_2026-11-13/launch.sh 3 32 nlte_mhs_eos_mu0.5_az0 nlte_colhse_eos_mu0.5_az0"
s hyades21 "/mn/stornext/d9/data/harshm/MHS/HSEMHS/out/paper_252_nlte_2026-11-13/launch.sh 3 32 nlte_mhs_eos_mu0.707_az0 nlte_colhse_eos_mu0.707_az0"
s hyades20 "/mn/stornext/d9/data/harshm/MHS/HSEMHS/out/paper_252_nlte_2026-11-13/launch.sh 3 32 nlte_mhspp_eos_mu0.5_az0 nlte_colhse_logm_mu1.0"
s hyades17 "/mn/stornext/d9/data/harshm/MHS/HSEMHS/out/paper_252_nlte_2026-11-13/launch.sh 3 24 nlte_mhs_eos_mu1.0 nlte_colhse_eos_mu1.0"
s hyades18 "/mn/stornext/d9/data/harshm/MHS/HSEMHS/out/paper_252_nlte_2026-11-13/launch.sh 3 24 nlte_mhs_m3d_mu1.0 nlte_colhse_m3d_mu1.0 nlte_mhs_logm_mu1.0"
s hyades12 "/mn/stornext/d9/data/harshm/MHS/IO/tau252_nlte/build_nlte.sh && /mn/stornext/d9/data/harshm/MHS/HSEMHS/out/paper_252_nlte_2026-11-13/launch.sh 3 32 nlteB_colhse_eosnlte_mu1.0 nlteB_mhs_eosnlte_cut3_mu1.0 nlteB_mhs_eosnlte_mu1.0"
echo "launched $(date -Is)"

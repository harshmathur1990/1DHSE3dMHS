#!/bin/bash -l
# μ = 1 τ input with τ integrated from the NLTE table's χ500 (branch nlte-eos).
export PATH=/tmp/julia-1.13.1/bin:$PATH
cd /mn/stornext/d9/data/harshm/MHS_nlte/HSEMHS
M3D=/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385
NC=/mn/stornext/d9/data/harshm/MHS/IO/BIFROST_en024048_hion_0_504_0_504.nc
O=/mn/stornext/d9/data/harshm/MHS/IO/tau252_nlte
julia --project=. -t 64 scripts/inputs/make_tau_input.jl $M3D $NC $O/input_eosnlte_mu1.0.nc $O/truth_eosnlte_mu1.0.h5 mu=1 table=cache/eos_table_nlte.h5 > $O/build_eosnlte_mu1.0.log 2>&1
echo "eosnlte_mu1.0 exit $?" >> $O/build_nlte.done

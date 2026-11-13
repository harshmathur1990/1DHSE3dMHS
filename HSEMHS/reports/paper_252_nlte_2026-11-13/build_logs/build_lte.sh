#!/bin/bash -l
# The paper's τ inputs rebuilt on branch nlte-eos (truth now with lnne), with the LTE table.
source ~/loadgcc.sh >/dev/null 2>&1
cd /mn/stornext/d9/data/harshm/MHS_nlte/HSEMHS
M3D=/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385
NC=/mn/stornext/d9/data/harshm/MHS/IO/BIFROST_en024048_hion_0_504_0_504.nc
O=/mn/stornext/d9/data/harshm/MHS/IO/tau252_nlte
b() { name=$1; shift; julia --project=. -t 64 scripts/inputs/make_tau_input.jl $M3D $NC $O/input_$name.nc $O/truth_$name.h5 "$@" > $O/build_$name.log 2>&1; echo "$name exit $?" >> $O/build_lte.done; }
b eos_mu1.0 mu=1
b m3d_mu1.0 mu=1 tau=multi3d
b eos_mu0.707_az0 mu=0.707 azimuth_deg=0
b eos_mu0.5_az0 mu=0.5 azimuth_deg=0

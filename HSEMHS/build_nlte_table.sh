#!/bin/bash
# The HSEMHS NLTE EOS table (cache/eos_table_nlte.h5) for the FFNOInversion forward of ar098192 with the final66 l004p model.
source ~/loadgcc.sh >/dev/null 2>&1
export JULIA_PKG_USE_CLI_GIT=true
cd /mn/stornext/d9/data/harshm/HSEMHS/HSEMHS
/tmp/julia-1.12.3/bin/julia --project=. -e 'using Pkg; Pkg.instantiate()' 2>&1 | tail -3
echo "build start $(date)"
/tmp/julia-1.12.3/bin/julia --project=. -t 120 scripts/run.jl eos_table_nlte_build.toml
echo "EXIT $? $(date)"

#!/usr/bin/env python3
"""
Python oracle for the Wittmann direct solver at the shim's tolerances.

witt.py hard-codes prec = 1e-5, 250 outer and 5 inner iterations. To compare at identical
settings (spec Gate 1), this script loads witt.py's source and applies exactly the same
changes as deps/witt/build.jl makes to the C++ code (prec, outer cap, both f5 loops),
asserting each substitution count. The physics is untouched.

usage: witt_oracle.py WITT_DIR IN.txt OUT.txt [prec] [maxit]
  IN.txt  : lines "T P_pa"
  OUT.txt : lines "rho_kg_m3 ne_m3 pe_Pa"   (pe in Pa)
  opacity mode: witt_oracle.py WITT_DIR IN.txt OUT.txt --opacity
  OUT.txt : lines "T xna xne n1..n17 opac scat" (cgs) at 5000 A, Pe from the same solver
  --nofix : do not apply the gasc H2 correction (FIX_GASC_H2) to the Python code
"""
import sys, os, types
import numpy as np

wdir, fin, fout = sys.argv[1:4]
opacity = "--opacity" in sys.argv
rest = [a for a in sys.argv[4:] if not a.startswith("--")]
prec = float(rest[0]) if len(rest) > 0 else 1e-10
maxit = int(rest[1]) if len(rest) > 1 else 5000

src = open(os.path.join(wdir, "witt.py")).read()


def sub(s, old, new, count):
    n = s.count(old)
    assert n == count, f"expected {count} of {old!r}, found {n}"
    return s.replace(old, new)


src = sub(src, "import witt as this", "", 1)   # bound to the patched module below
src = sub(src, "self.prec = 1.e-5", "self.prec = prec", 1)
src = sub(src, "while((abs(dif) > self.prec) and (it < 250)):", f"while((abs(dif) > self.prec) and (it < {maxit})):", 1)
src = sub(src, "while((diff > 1.e-5) and (it < 5)):", "while((diff > self.prec) and (it < 50)):", 2)
if "--nofix" not in sys.argv:
    # same correction as deps/witt/build.jl FIX_GASC_H2: gasc uses pe_pg's H2 constants
    src = sub(src, "g4 = 10.0**cmol[0]; g5 = 10.0**cmol[1]",
              "cmol[0] = self.acota(cmol[0], -30., 30.); cmol[1] = self.acota(cmol[1], -30., 30.); "
              "g4 = pe*10.0**cmol[0]; g5 = pe*10.0**cmol[1]", 1)
mod = types.ModuleType("witt_patched")
mod.__file__ = os.path.join(wdir, "witt.py")
mod.__dict__["prec"] = prec
exec(compile(src, mod.__file__, "exec"), mod.__dict__)
sys.modules["witt_patched"] = mod
mod.this = mod                                   # contOpacity refers to `this.cop`
w = mod.witt(prec=prec)

data = np.loadtxt(fin, ndmin=2)
with open(fout, "w") as fo:
    for T, P in data:
        pg = P * 10.0
        pe = w.pe_from_pg(T, pg)
        if opacity:
            n = w.getBackgroundPartials(T, pg, pe, divide_by_u=True)
            TK = T * w.BK
            opac, scat = mod.cop(T, TK / w.EV, TK, w.HH / TK, np.log(T), (pg - pe) / TK, pe / TK,
                                 np.array([5000.0]), *n)
            vals = [T, (pg - pe) / TK, pe / TK] + [float(v) for v in n] + [float(opac[0]), float(scat[0])]
            fo.write(" ".join(repr(float(v)) for v in vals) + "\n")
        else:
            rho = w.rho_from_pe(T, pe)
            fo.write(f"{float(rho*1e3)!r} {float(pe/(w.BK*T)*1e6)!r} {float(pe*0.1)!r}\n")

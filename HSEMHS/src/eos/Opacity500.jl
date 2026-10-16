# Julia port of the Wittmann continuum opacity (wittman-eos/witt.py, lines 804–1363),
# one function per Python function, scalar, one wavelength (implementation doc §7.7).
#
# CGS throughout, numeric literals verbatim. Python 0-based array indices are translated to
# Julia 1-based ones; comments give the Python expression where the translation is not
# obvious. `cop` returns (absorption + scattering, scattering); use only the first, which
# already includes scattering. HOTOP() is a placeholder (0), as in the original.

const WL500 = 5000.0                     # Å

seaton(freq0, xsect, power, a, freq) =
    xsect * (a + (1.0 - a) * (freq0 / freq)) * (freq0 / freq)^(floor(2.0 * power + 0.01) * 0.5)

const Z4LOG = (0.0, 1.20412, 1.90849, 2.40824, 2.79588, 3.11261)
const A0_DATA = [
    5.53, 5.49, 5.46, 5.43, 5.40, 5.25, 5.00, 4.69, 4.48, 4.16, 3.85,
    4.91, 4.87, 4.84, 4.80, 4.77, 4.63, 4.40, 4.13, 3.87, 3.52, 3.27,
    4.29, 4.25, 4.22, 4.18, 4.15, 4.02, 3.80, 3.57, 3.27, 2.98, 2.70,
    3.64, 3.61, 3.59, 3.56, 3.54, 3.41, 3.22, 2.97, 2.70, 2.45, 2.20,
    3.00, 2.98, 2.97, 2.95, 2.94, 2.81, 2.65, 2.44, 2.21, 2.01, 1.81,
    2.41, 2.41, 2.41, 2.41, 2.41, 2.32, 2.19, 2.02, 1.84, 1.67, 1.50,
    1.87, 1.89, 1.91, 1.93, 1.95, 1.90, 1.80, 1.68, 1.52, 1.41, 1.30,
    1.33, 1.39, 1.44, 1.49, 1.55, 1.56, 1.51, 1.42, 1.33, 1.25, 1.17,
    0.90, 0.95, 1.00, 1.08, 1.17, 1.30, 1.32, 1.30, 1.20, 1.15, 1.11,
    0.55, 0.58, 0.62, 0.70, 0.85, 1.01, 1.15, 1.18, 1.15, 1.11, 1.08,
    0.33, 0.36, 0.39, 0.46, 0.59, 0.76, 0.97, 1.09, 1.13, 1.10, 1.08,
    0.19, 0.21, 0.24, 0.28, 0.38, 0.53, 0.76, 0.96, 1.08, 1.09, 1.09]
const A0 = permutedims(reshape(A0_DATA, 11, 12))          # 12×11, Python A0[r, c] = A0[r+1, c+1]

function coulff(tlog, freqlg, nz)
    gamlog = 10.39638 - tlog / 1.15129 + Z4LOG[nz]
    igam = min(trunc(Int, gamlog + 7.0), 10)
    igam < 1 && (igam = 1)
    hvktlg = (freqlg - tlog) / 1.15129 - 20.63764
    ihvkt = min(trunc(Int, hvktlg + 9.0), 11)
    ihvkt < 1 && (ihvkt = 1)
    p = gamlog - (igam - 7)
    q = hvktlg - (ihvkt - 9)
    # Python: (1-P)*((1-Q)*A0[IHVKT-1,IGAM-1]+Q*A0[IHVKT,IGAM-1])+P*((1-Q)*A0[IHVKT-1,IGAM]+Q*A0[IHVKT,IGAM])
    return (1.0 - p) * ((1.0 - q) * A0[ihvkt, igam] + q * A0[ihvkt+1, igam]) +
           p * ((1.0 - q) * A0[ihvkt, igam+1] + q * A0[ihvkt+1, igam+1])
end

const COULX_A1 = (0.9916, 1.105, 1.101, 1.101, 1.102, 1.0986)
const COULX_B1 = (2.719e3, -2.375e4, -9.863e3, -5.765e3, -3.909e3, -2.704e3)
const COULX_C1 = (-2.268e10, 4.077e8, 1.035e8, 4.593e7, 2.371e7, 1.229e7)

# N is the Python 0-based level index
function coulx(N::Int, freq, Z)
    n = (N + 1.0)^2
    if freq >= (Z * Z * 3.28805e15 / n)
        freq1 = freq * 1.0e-10
        clx = 0.2815 / freq1 / freq1 / freq1 / n / n / (N + 1.0) * Z * Z * Z * Z
        N >= 6 && return clx
        clx *= (COULX_A1[N+1] + (COULX_B1[N+1] + COULX_C1[N+1] * (Z * Z / freq1)) * (Z * Z / freq1))
        return clx
    end
    return 0.0
end

function hop(xne, xh1, xh2, freq, freqlg, t, tlog, tkev, stim, ehvkt)
    freq3 = (freq * 1.0e-10)^3
    cfree = 3.6919e-22 / freq3
    freet = xne * cfree * xh2 / sqrt(t)
    xr = xh1 / 13.595 * tkev
    boltex = exp(-13.427 / tkev) * xr
    exlim = exp(-13.595 / tkev) * xr
    c = 0.2815 / freq3
    freq < 4.05933e13 && (boltex = exlim / ehvkt)
    bolt(N) = (n1 = (N + 1.0)^2; exp(-13.595 * (1.0 - 1.0 / n1) / tkev) * 2.0 * n1 * xh1)
    h = (coulx(6, freq, 1.0) * bolt(6) + coulx(7, freq, 1.0) * bolt(7) + (boltex - exlim) * c +
         coulff(tlog, freqlg, 1) * freet) * stim
    s = 0.0
    for N in 0:5
        s += coulx(N, freq, 1.0) * bolt(N)
    end
    h += s * (1.0 - ehvkt)
    return h
end

function hrayop(xh1, freq)
    wave = min(freq, 2.463e15)
    wave = 2.997925e18 / wave
    ww = wave * wave
    ww2 = ww * ww
    sig = (5.799e-13 + 1.422e-6 / ww + 2.784 / (ww2)) / (ww2)
    return sig * xh1 * 2.0
end

function h2plop(xh1, xh2, freq, freqlg, freq15, tkev, stim)
    freq > 3.28805e15 && return 0.0
    fr = -3.0233e3 + (3.7797e2 + (-1.82496e1 + (3.9207e-1 - 3.1672e-3 * freqlg) * freqlg) * freqlg) * freqlg
    es = -7.342e-3 + (-2.409 + (1.028 + (-0.4230 + (0.1224 - 0.01351 * freq15) * freq15) * freq15) * freq15) * freq15
    return exp(-es / tkev + fr) * 2.0 * xh1 * xh2 * stim
end

function hminop(xh1, xhmin, freq, t, tkev, xne, ehvkt)
    freq1 = freq * 1.0e-10
    b = (1.3727e-15 + 4.3748 / freq) / freq1
    c = -2.5993e-7 / freq1^2
    hminbf = if freq <= 1.8259e14
        0.0
    elseif freq >= 2.111e14
        6.801e-10 + (5.358e-3 + (1.481e3 + (-5.519e7 + 4.808e11 / freq1) / freq1) / freq1) / freq1
    else
        3.695e-6 + (-1.251e-1 + 1.052e3 / freq1) / freq1
    end
    hminff = (b + c / t) * xh1 * xne * 2.0e-20
    hmin = t < 7730.0 ? xhmin : exp(0.7552 / tkev) / (2.0 * 2.4148e15 * t * sqrt(t)) * xh1 * xne
    h = hminbf * (1 - ehvkt) * hmin * 1.0e-10
    return h + hminff
end

const HE_G0 = (1.0, 3.0, 1.0, 9.0, 3.0, 3.0, 1.0, 9.0, 20.0, 3.0)
const HEFREQ0 = (5.9452090e15, 1.1528440e15, 0.9803331e15, 0.8761076e15, 0.8147100e15,
                 0.4519048e15, 0.4030971e15, 0.8321191e15, 0.3660215e15, 0.3627891e15)
const HE_CHI0 = (0.0, 19.819, 20.615, 20.964, 21.217, 22.718, 22.920, 23.006, 23.073, 23.086)

function he1op(xhe1, xhe2, xne, freq, freqlg, t, tkev, tlog, ehvkt, stim)
    freet = xne * 1.0e-10 * xhe2 * 1.0e-10 / sqrt(t) * 1.0e-10
    xrlog = log(xhe1 * (2.0 / 13.595) * tkev)
    boltex = exp(-23.730 / tkev + xrlog)
    exlim = exp(-24.587 / tkev + xrlog)
    freq3 = (freq * 1.0e-10)^3
    cfree = 3.6919e8 / freq3
    c = 2.815e-1 / freq3
    # Python: for NMIN in range(10): if HEFREQ0[NMIN] <= FREQ: break   (falls through to 9)
    nmin = 10
    for i in 1:10
        if HEFREQ0[i] <= freq
            nmin = i
            break
        end
    end
    dum = (33.32 - 2.0 * freqlg, -390.026 + (21.035 - 0.318 * freqlg) * freqlg, 26.83 - 1.91 * freqlg,
           61.21 - 2.9 * freqlg, 81.35 - 3.5 * freqlg, 12.69 - 1.54 * freqlg, 23.85 - 1.86 * freqlg,
           49.30 - 2.60 * freqlg, 85.20 - 3.69 * freqlg, 58.81 - 2.89 * freqlg)
    ex = boltex
    freq < 2.055e14 && (ex = exlim / ehvkt)
    he1 = (ex - exlim) * c
    s = 0.0
    for i in nmin:10
        s += exp(dum[i]) * exp(-HE_CHI0[i] / tkev) * HE_G0[i] * xhe1
    end
    he1 += s
    return (he1 + coulff(tlog, freqlg, 1) * freet * cfree) * stim
end

function he2op(xhe2, xhe3, xne, freq, freqlg, t, tkev, tlog, ehvkt, stim)
    freet = xne * xhe3 / sqrt(t)
    xr = xhe2 / 13.595 * tkev
    boltex = exp(-53.859 / tkev) * xr
    exlim = exp(-54.403 / tkev) * xr
    freq3 = (freq * 1.0e-5)^3
    cfree = 3.6919e-07 / freq3 * 4.0
    c = 2.815e14 * 2.0 * 2.0 / freq3
    ex = boltex
    freq < 1.31522e14 && (ex = exlim / ehvkt)
    he2 = (ex - exlim) * c
    s = 0.0
    for N in 0:8
        n12 = (N + 1.0)^2
        s += coulx(N, freq, 2.0) * exp(-(54.403 - 54.403 / n12) / tkev) * 2.0 * n12 * xhe2
    end
    he2 += s
    he2 = (he2 + coulff(tlog, freqlg, 2) * cfree * freet) * stim
    return he2 >= 1.0e-20 ? he2 : 0.0
end

function hemiop(xhe1, freq, t, xne)
    a = 3.397e-26 + (-5.216e-11 + 7.039e05 / freq) / freq
    b = -4.116e-22 + (1.067e-06 + 8.135e09 / freq) / freq
    c = 5.081e-17 + (-8.724e-03 - 5.659e12 / freq) / freq
    return (a * t + b + c / t) * xne * xhe1 * 1.0e-20
end

function heraop(xhe1, freq)
    ww = (2.997925e+03 / min(freq * 1.0e-15, 5.15))^2
    arg = 1.0 + (2.44e5 + 5.94e10 / (ww - 2.90e5)) / ww
    sig = 5.484e-14 / ww / ww * arg * arg
    return sig * xhe1
end

const PEACH0 = permutedims(reshape([
    -42.474, -42.350, -42.109, -41.795, -41.467, -41.159, -40.883,
    -41.808, -41.735, -41.582, -41.363, -41.115, -40.866, -40.631,
    -41.273, -41.223, -41.114, -40.951, -40.755, -40.549, -40.347,
    -45.583, -44.008, -42.957, -42.205, -41.639, -41.198, -40.841,
    -44.324, -42.747, -41.694, -40.939, -40.370, -39.925, -39.566,
    -50.969, -48.388, -46.630, -45.344, -44.355, -43.568, -42.924,
    -50.633, -48.026, -46.220, -44.859, -43.803, -42.957, -42.264,
    -53.028, -49.643, -47.367, -45.729, -44.491, -43.520, -42.736,
    -51.785, -48.352, -46.050, -44.393, -43.140, -42.157, -41.363,
    -52.285, -48.797, -46.453, -44.765, -43.486, -42.480, -41.668,
    -52.028, -48.540, -46.196, -44.507, -43.227, -42.222, -41.408,
    -52.384, -48.876, -46.513, -44.806, -43.509, -42.488, -41.660,
    -52.363, -48.856, -46.493, -44.786, -43.489, -42.467, -41.639,
    -54.704, -50.772, -48.107, -46.176, -44.707, -43.549, -42.611,
    -54.359, -50.349, -47.643, -45.685, -44.198, -43.027, -42.418], 7, 15))   # 15×7
const FREQMG = (1.9341452e15, 1.8488510e15, 1.1925797e15, 7.9804046e14, 4.5772110e14, 4.1440977e14, 4.1113514e14)
const FLOG0 = (35.32123, 35.19844, 35.15334, 34.71490, 34.31318, 33.75728, 33.65788, 33.64994, 33.43947)
const TLG0 = (8.29405, 8.51719, 8.69951, 8.85367, 8.98720, 9.10498, 9.21034)

"""First Python index N in 0:nmax-1 with freq > tab[N], else nmax-1 (loop fall-through)."""
function py_break_index(freq, tab, nmax)
    for N in 0:nmax-1
        freq > tab[N+1] && return N
    end
    return nmax - 1
end

function mg1op(freq, freqlg, t, tlog)
    nt = min(6, trunc(Int, floor(t / 1000.0)) - 3)
    nt < 1 && (nt = 1)
    dt = (tlog - TLG0[nt]) / (TLG0[nt+1] - TLG0[nt])        # Python TLG0[NT-1], TLG0[NT]
    N = py_break_index(freq, FREQMG, 7)
    d = (freqlg - FLOG0[N+1]) / (FLOG0[N+2] - FLOG0[N+1])     # Python FLOG0[N], FLOG0[N+1]
    N > 1 && (N = 2N - 1)
    d1 = 1.0 - d
    xwl1 = PEACH0[N+2, nt] * d + PEACH0[N+1, nt] * d1        # Python PEACH0[N+1,NT-1], PEACH0[N,NT-1]
    xwl2 = PEACH0[N+2, nt+1] * d + PEACH0[N+1, nt+1] * d1
    return exp(xwl1 * (1.0 - dt) + xwl2 * dt)
end

function c1op(freq, tkev)
    c1240 = 5.0 * exp(-1.264 / tkev)
    c1444 = exp(-2.683 / tkev)
    x1444 = 0.0; x1240 = 0.0; x1100 = 0.0
    freq >= 2.7254e15 && (x1100 = seaton(2.7254e15, 1.219e-17, 2.0e0, 3.317e0, freq))
    freq >= 2.4196e15 && (x1240 = seaton(2.4196e15, 1.030e-17, 1.5e0, 2.789e0, freq))
    freq >= 2.0761e15 && (x1444 = seaton(2.0761e15, 9.590e-18, 1.5e0, 3.501e0, freq))
    return x1100 * 9.0 + x1240 * c1240 + x1444 * c1444
end

al1op(freq) = freq > 1.443e15 ? 2.1e-17 * ((1.443e15 / freq)^3) * 6.0 : 0.0

const PEACH1 = permutedims(reshape([
    38.136, 38.138, 38.140, 38.141, 38.143, 38.144, 38.144, 38.145, 38.145,
    37.834, 37.839, 37.843, 37.847, 37.850, 37.853, 37.855, 37.857, 37.858,
    37.898, 37.898, 37.897, 37.897, 37.897, 37.896, 37.895, 37.895, 37.894,
    40.737, 40.319, 40.047, 39.855, 39.714, 39.604, 39.517, 39.445, 39.385,
    40.581, 40.164, 39.893, 39.702, 39.561, 39.452, 39.366, 39.295, 39.235,
    45.521, 44.456, 43.753, 43.254, 42.878, 42.580, 42.332, 42.119, 41.930,
    45.520, 44.455, 43.752, 43.251, 42.871, 42.569, 42.315, 42.094, 41.896,
    55.068, 51.783, 49.553, 47.942, 46.723, 45.768, 44.997, 44.360, 43.823,
    53.868, 50.369, 48.031, 46.355, 45.092, 44.104, 43.308, 42.652, 42.100,
    54.133, 50.597, 48.233, 46.539, 45.261, 44.262, 43.456, 42.790, 42.230,
    54.051, 50.514, 48.150, 46.454, 45.176, 44.175, 43.368, 42.702, 42.141,
    54.442, 50.854, 48.455, 46.733, 45.433, 44.415, 43.592, 42.912, 42.340,
    54.320, 50.722, 48.313, 46.583, 45.277, 44.251, 43.423, 42.738, 42.160,
    55.691, 51.965, 49.444, 47.615, 46.221, 45.119, 44.223, 43.478, 42.848,
    55.661, 51.933, 49.412, 47.582, 46.188, 45.085, 44.189, 43.445, 42.813,
    55.973, 52.193, 49.630, 47.769, 46.349, 45.226, 44.314, 43.555, 42.913,
    55.922, 52.141, 49.577, 47.715, 46.295, 45.172, 44.259, 43.500, 42.858,
    56.828, 52.821, 50.110, 48.146, 46.654, 45.477, 44.522, 43.730, 43.061,
    56.657, 52.653, 49.944, 47.983, 46.491, 45.315, 44.360, 43.569, 42.901], 9, 19))   # 19×9
const FREQSI1 = (2.1413750e15, 1.97231650e15, 1.7879689e15, 1.5152920e15, 0.55723927e15, 5.3295914e14,
                 4.7886458e14, 4.72164220e14, 4.6185133e14)
const FLOG1 = (35.45438, 35.30022, 35.21799, 35.11986, 34.95438, 33.95402, 33.90947, 33.80244, 33.78835,
               33.76626, 33.70518)
const TLG1 = (8.29405, 8.51719, 8.69951, 8.85367, 8.98720, 9.10498, 9.21034, 9.30565, 9.39266)

function si1op(freq, freqlg, t, tlog)
    nt = min(8, trunc(Int, floor(t / 1000.0)) - 3)
    nt < 1 && (nt = 1)
    dt = (tlog - TLG1[nt]) / (TLG1[nt+1] - TLG1[nt])
    N = py_break_index(freq, FREQSI1, 9)
    d = (freqlg - FLOG1[N+1]) / (FLOG1[N+2] - FLOG1[N+1])
    N > 1 && (N = 2N - 1)
    dd = 1.0 - d
    xwl1 = PEACH1[N+2, nt] * d + PEACH1[N+1, nt] * dd
    xwl2 = PEACH1[N+2, nt+1] * d + PEACH1[N+1, nt+1] * dd
    return exp(-(xwl1 * (1.0 - dt) + xwl2 * dt)) * 9.0
end

const FE_G1 = (25.0, 35.0, 21.0, 15.0, 9.0, 35.0, 33.0, 21.0, 27.0, 49.0, 9.0, 21.0,
               27.0, 9.0, 9.0, 25.0, 33.0, 15.0, 35.0, 3.0, 5.0, 11.0, 15.0, 13.0,
               15.0, 9.0, 21.0, 15.0, 21.0, 25.0, 35.0, 9.0, 5.0, 45.0, 27.0, 21.0,
               15.0, 21.0, 15.0, 25.0, 21.0, 35.0, 5.0, 15.0, 45.0, 35.0, 55.0, 25.0)
const FE_E1 = (500.0, 7500.0, 12500.0, 17500.0, 19000.0, 19500.0, 19500.0,
               21000.0, 22000.0, 23000.0, 23000.0, 24000.0, 24000.0, 24500.0,
               24500.0, 26000.0, 26500.0, 26500.0, 27000.0, 27500.0, 28500.0,
               29000.0, 29500.0, 29500.0, 29500.0, 30000.0, 31500.0, 31500.0,
               33500.0, 33500.0, 34000.0, 34500.0, 34500.0, 35000.0, 35500.0,
               37000.0, 37000.0, 37000.0, 38500.0, 40000.0, 40000.0, 41000.0,
               41000.0, 43000.0, 43000.0, 43000.0, 43000.0, 44000.0)
const FE_WNO1 = (63500.0, 58500.0, 53500.0, 59500.0, 45000.0, 44500.0, 44500.0,
                 43000.0, 58000.0, 41000.0, 54000.0, 40000.0, 40000.0, 57500.0,
                 55500.0, 38000.0, 57500.0, 57500.0, 37000.0, 54500.0, 53500.0,
                 55000.0, 34500.0, 34500.0, 34500.0, 34000.0, 32500.0, 32500.0,
                 32500.0, 32500.0, 32000.0, 29500.0, 29500.0, 31000.0, 30500.0,
                 29000.0, 27000.0, 54000.0, 27500.0, 24000.0, 47000.0, 23000.0,
                 44000.0, 42000.0, 42000.0, 21000.0, 42000.0, 42000.0)

function fe1op(freq, hkt)
    waveno = freq / 2.99792458e10
    waveno < 21000.0 && return 0.0
    s = 0.0
    for i in 1:48
        bolt = FE_G1[i] * exp(-FE_E1[i] * 2.99792458e10 * hkt)
        xxx = ((FE_WNO1[i] + 3000.0 - waveno) / FE_WNO1[i] / 0.1)
        xsect = FE_WNO1[i] < waveno ? 3.0e-18 / (1.0 + xxx^4) : 0.0
        s += xsect * bolt
    end
    return s
end

coolop(xc1, xmg1, xal1, xsi1, xfe1, stim, freq, freqlg, t, tlog, tkev, hkt) =
    (c1op(freq, tkev) * xc1 + mg1op(freq, freqlg, t, tlog) * xmg1 + al1op(freq) * xal1 +
     si1op(freq, freqlg, t, tlog) * xsi1 + fe1op(freq, hkt) * xfe1) * stim

function n1op(freq, tkev)
    c1130 = 6.0 * exp(-3.575 / tkev)
    c1020 = 10.0 * exp(-2.384 / tkev)
    x1130 = 0.0; x1020 = 0.0; x853 = 0.0
    freq >= 3.517915e15 && (x853 = seaton(3.517915e15, 1.142e-17, 2.0e0, 4.29e0, freq))
    freq >= 2.941534e15 && (x1020 = seaton(2.941534e15, 4.410e-18, 1.5e0, 3.85e0, freq))
    freq >= 2.653317e15 && (x1130 = seaton(2.653317e15, 4.200e-18, 1.5e0, 4.34e0, freq))
    return x853 * 4.0 + x1020 * c1020 + x1130 * c1130
end

o1op(freq) = freq >= 3.28805e15 ? 9.0 * seaton(3.28805e15, 2.94e-18, 1.0e0, 2.66e0, freq) : 0.0

function mg2op(freq, tkev)
    c1169 = 6.0 * exp(-4.43 / tkev)
    x1169 = 0.0; x824 = 0.0
    freq >= 3.635492e15 && (x824 = seaton(3.635492e15, 1.40e-19, 4.0e0, 6.7e0, freq))
    freq >= 2.564306e15 && (x1169 = 5.11e-19 * (2.564306e15 / freq)^3)
    return x824 * 2.0 + x1169 * c1169
end

const PEACH2 = permutedims(reshape([
    -43.8941, -43.8941, -43.8941, -43.8941, -43.8941, -43.8941,
    -42.2444, -42.2444, -42.2444, -42.2444, -42.2444, -42.2444,
    -40.6054, -40.6054, -40.6054, -40.6054, -40.6054, -40.6054,
    -54.2389, -52.2906, -50.8799, -49.8033, -48.9485, -48.2490,
    -50.4108, -48.4892, -47.1090, -46.0672, -45.2510, -44.5933,
    -52.0936, -50.0741, -48.5999, -47.4676, -46.5649, -45.8246,
    -51.9548, -49.9371, -48.4647, -47.3340, -46.4333, -45.6947,
    -54.2407, -51.7319, -49.9178, -48.5395, -47.4529, -46.5709,
    -52.7355, -50.2218, -48.4059, -47.0267, -45.9402, -45.0592,
    -53.5387, -50.9189, -49.0200, -47.5750, -46.4341, -45.5082,
    -53.2417, -50.6234, -48.7252, -47.2810, -46.1410, -45.2153,
    -53.5097, -50.8535, -48.9263, -47.4586, -46.2994, -45.3581,
    -54.0561, -51.2365, -49.1980, -47.6497, -46.4302, -45.4414,
    -53.8469, -51.0256, -48.9860, -47.4368, -46.2162, -45.2266], 6, 14))   # 14×6
const FREQSI2 = (4.9965417e15, 3.9466738e15, 1.5736321e15, 1.5171539e15, 9.2378947e14, 8.3825004e14, 7.6869872e14)
const FLOG2 = (36.32984, 36.14752, 35.91165, 34.99216, 34.95561, 34.45941, 34.36234, 34.27572, 34.20161)
const TLG2 = (9.21034, 9.39266, 9.54681, 9.68034, 9.79813, 9.90349)

function si2op(freq, freqlg, t, tlog)
    nt = min(5, trunc(Int, floor(t / 2000.0)) - 4)
    nt < 1 && (nt = 1)
    dt = (tlog - TLG2[nt]) / (TLG2[nt+1] - TLG2[nt])
    N = py_break_index(freq, FREQSI2, 7)
    d = (freqlg - FLOG2[N+1]) / (FLOG2[N+2] - FLOG2[N+1])
    N > 1 && (N = 2N - 2)
    N == 13 && (N = 12)
    d1 = 1.0 - d
    xwl1 = PEACH2[N+2, nt] * d + PEACH2[N+1, nt] * d1
    xwl2 = PEACH2[N+2, nt+1] * d + PEACH2[N+1, nt+1] * d1
    return exp(xwl1 * (1.0 - dt) + xwl2 * dt) * 6.0
end

function ca2op(freq, tkev)
    c1218 = 10.0 * exp(-1.697 / tkev)
    c1420 = 6.0 * exp(-3.142 / tkev)
    x1044 = 0.0; x1218 = 0.0; x1420 = 0.0
    if freq >= 2.870454e15
        xxx = (2.870454e15 / freq)^3
        x1044 = 1.08e-19 * xxx
    end
    freq >= 2.460127e15 && (x1218 = 1.64e-17 * sqrt(2.460127e15 / freq))
    freq >= 2.110779e15 && (x1420 = seaton(2.110779e15, 4.13e-18, 3.0, 0.69, freq))
    return x1044 + x1218 * c1218 + x1420 * c1420
end

lukeop(xn1, xo1, xmg2, xsi2, xca2, stim, freq, freqlg, t, tlog, tkev) =
    (n1op(freq, tkev) * xn1 + o1op(freq) * xo1 + mg2op(freq, tkev) * xmg2 +
     si2op(freq, freqlg, t, tlog) * xsi2 + ca2op(freq, tkev) * xca2) * stim

hotop() = 0.0                       # placeholder in the original (hence OPACITY_INCOMPLETE)
elecop(xne) = 0.6653e-24 * xne

function h2raop(xh1, freq, t, tkev, tlog)
    ww = (2.997925e18 / min(freq, 2.922e15))^2
    ww2 = ww * ww
    sig = (8.14e-13 + 1.28e-6 / ww + 1.61e0 / ww2) / ww2
    arg = 4.477 / tkev - 4.6628e1 + (1.8031e-3 + (-5.023e-7 + (8.1424e-11 - 5.0501e-15 * t) * t) * t) * t - 1.5 * tlog
    h1 = xh1 * 2.0
    return arg > -80.0 ? exp(arg) * h1 * h1 * sig : 0.0
end

"""
    cop(T, TKEV, TK, HKT, TLOG, XNA, XNE, WL, n17) -> (extinction, scattering)  [cm⁻¹]

`n17` are the background partials (H1, H2, HMIN, HE1, HE2, HE3, C1, AL1, SI1, SI2, CA1,
CA2, MG1, MG2, FE1, N1, O1), cm⁻³, divided by partition functions.
"""
function cop(t, tkev, tk, hkt, tlog, xna, xne, wl, n::AbstractVector)
    h1, h2, hmin, he1, he2, he3, c1, al1, si1, si2, ca1, ca2, mg1, mg2, fe1, n1, o1 = ntuple(i -> n[i], 17)
    freq = 2.997925e18 / wl
    freqlg = log(freq)
    freq15 = freq * 1.0e-15
    ehvkt = exp(-freq * hkt)
    stim = 1.0 - ehvkt
    acool = 0.0; aluke = 0.0
    ahyd = hop(xne, h1, h2, freq, freqlg, t, tlog, tkev, stim, ehvkt)
    ah2p = h2plop(h1, h2, freq, freqlg, freq15, tkev, stim)
    ahmin = hminop(h1, hmin, freq, t, tkev, xne, ehvkt)
    sigh = hrayop(h1, freq)
    ahe1 = he1op(he1, he2, xne, freq, freqlg, t, tkev, tlog, ehvkt, stim)
    ahe2 = he2op(he2, he3, xne, freq, freqlg, t, tkev, tlog, ehvkt, stim)
    ahemin = hemiop(he1, freq, t, xne)
    sighe = heraop(he1, freq)
    t < 12000.0 && (acool = coolop(c1, mg1, al1, si1, fe1, stim, freq, freqlg, t, tlog, tkev, hkt))
    t < 30000.0 && (aluke = lukeop(n1, o1, mg2, si2, ca2, stim, freq, freqlg, t, tlog, tkev))
    ahot = hotop()
    sigel = elecop(xne)
    sigh2 = h2raop(h1, freq, t, tkev, tlog)
    a = ahyd + ahmin + ah2p + ahe1 + ahe2 + ahemin + acool + aluke + ahot
    b = sigh + sighe + sigel + sigh2
    return a + b, b
end

"""Extinction at 5000 Å in m⁻¹ (absorption + scattering, counted once)."""
function chi500_m(T, n17::AbstractVector, xna, xne)
    tk = T * BK_CGS
    tkev = tk / EV_CGS
    opac, _ = cop(T, tkev, tk, HH_CGS / tk, log(T), xna, xne, WL500, n17)
    return 100.0 * opac
end

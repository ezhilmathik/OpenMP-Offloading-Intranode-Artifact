# collect.awk -- one dataset (site x size x rep) -> printed table,   [v5: plausibility checks]
#                rows appended to summary.csv and to plot_<site>.dat
#
# Called by collect.sh with the dataset's files as arguments.
# Needs GNU awk (FPAT for quoted CSV fields, asort for medians, match arrays).
#
# Sources used for each table row
#   time             nsys kernel trace (GPU timestamps), warm-up launch excluded
#   registers, smem  driver launch record (kern_*.csv, exact bytes); fallback:
#                    ptxas for CUDA, nsys trace (rounded to 0.001 MB) for OpenMP
#   launch shape     nsys trace
#   occupancy        theoretical: computed here, same formula for every kernel
#                    achieved: nsys GPU metrics (all) and ncu (CUDA)
#   memory           nsys GPU metrics (all); L1/L2 hit rates from ncu (CUDA)
#   launch overhead  the programs' own empty-launch measurement + nsys API medians
#   kernel mode      LIBOMPTARGET_INFO probe
#   correctness      the programs' own CPU check

function uq(s)   { gsub(/^[ \t]*"|"[ \t]*$/, "", s); gsub(/""/, "\"", s); return s }
function num(s)  { s = uq(s); gsub(/,/, "", s); return s + 0 }
function col(prefix,   k) { for (k in H) if (index(k, prefix) == 1) return H[k]; return 0 }
function unit_scale_ms(h) { return (h ~ /\(ns\)/) ? 1e-6 : (h ~ /\(us\)/) ? 1e-3 : (h ~ /\(s\)/) ? 1e3 : 1 }
function hname(i,   k) { for (k in H) if (H[k] == i) return k; return "" }

function classify(name, bx, by) {
    if (name ~ /matrixMultiplyOpenMP/) return "omp"
    if (name ~ /dgemm_naive/) {
        if (bx == 32  && by == 8) return "cuda_naive"
        if (bx == 128 && by == 1) return "cuda_rep1d"
        return ""
    }
    if (name ~ /dgemm_tiled32/) return "cuda_tiled32"
    if (name ~ /dgemm_regtile/) return (name ~ /128/) ? "cuda_rt128" : "cuda_rt64"
    return ""
}

# ptxas entry name belonging to an implementation
function ptx_entry(im,   e, re) {
    re = (im ~ /naive|rep1d/) ? "dgemm_naive" : (im == "cuda_tiled32") ? "dgemm_tiled32" :
         (im == "cuda_rt64") ? "regtile.*ILi64E" : (im == "cuda_rt128") ? "regtile.*ILi128E" : "^$"
    for (e in PREG) if (e ~ re) return e
    return ""
}

# Theoretical occupancy (%), CUDA occupancy-calculator rules for sm_80/sm_90:
# 65536 registers and 64 warps per SM, max 32 blocks, registers allocated per
# warp in units of 256, 1 KB shared memory reserved per block.
function occ_theory(regs, threads, smem, gpu,   wpb, rpw, bregs, bwarp, bsmem, smem_sm, b) {
    wpb = int((threads + 31) / 32)
    rpw = int((regs * 32 + 255) / 256) * 256
    if (rpw < 256) rpw = 256
    bregs = int(int(65536 / rpw) / wpb)
    bwarp = int(64 / wpb)
    smem_sm = (gpu ~ /H100|H200|GH200/) ? 233472 : 167936
    bsmem = (smem > 0) ? int(smem_sm / (smem + 1024)) : 32
    b = bregs
    if (bwarp < b) b = bwarp
    if (bsmem < b) b = bsmem
    if (b > 32) b = 32
    return 100.0 * b * wpb / 64
}

# Average of one GPU metric over the timed launches (warm-up launch excluded).
function gm_avg(im, met,   k, s, c, first) {
    if (!(im in GMN)) return ""
    first = (GMN[im] > 1) ? 2 : 1
    s = 0; c = 0
    for (k = first; k <= GMN[im]; k++)
        if ((im, met, k) in GMV) { s += GMV[im, met, k]; c++ }
    return c ? s / c : ""
}
function gm_pick(im, re,   met, v) {
    for (met in GMMET)
        if (tolower(met) ~ re) { v = gm_avg(im, met); if (v != "") return v }
    return ""
}

function f1(v)   { return (v == "") ? "n/a" : sprintf("%.1f", v) }
function gmv(im, v) { return (GMST[im] == "invalid") ? "invalid (nsys)" : f1(v) }
function ncv(im, v) { return (NCST[im] == "invalid") ? "invalid (ncu)" : (NCST[im] == "suspect" && v != "n/a") ? v " *" : v }
# median [min-max] in ms below 1 s, otherwise in s
function tfmt3(md, lo, hi,   u, d) {
    u = (md >= 1000) ? "s" : "ms"; d = (u == "s") ? 1000 : 1
    if (lo == "") return sprintf("%.2f %s", md / d, u)
    return sprintf("%.2f [%.2f-%.2f] %s", md / d, lo / d, hi / d, u)
}
function nan(v)  { return (v == "") ? "NaN" : v }

BEGINFILE {
    fn = FILENAME; sub(/.*\//, "", fn)
    if (fn ~ /\.csv$/) FPAT = "([^,]*)|(\"([^\"]|\"\")*\")"
    else               FS = " "
    delete H; hdr = 0
    prog = (fn ~ /^omp/ || fn ~ /_omp/) ? "omp" : "cuda"
}

{ if (sub(/\r$/, "")) $0 = $0 }          # tolerate CRLF line endings in any input

fn == "meta.txt" {
    k = $0; sub(/:.*/, "", k); v = $0; sub(/^[^:]*: ?/, "", v)
    if (!(k in META)) META[k] = v
    next
}

fn ~ /^build_cuda/ {
    if (match($0, /entry function '([^']+)'/, m)) entry = m[1]
    else if (match($0, /Used ([0-9]+) registers/, m)) {
        PREG[entry] = m[1]
        PSMEM[entry] = match($0, /([0-9]+) bytes smem/, s) ? s[1] : 0
    }
    next
}

fn == "cuda.out" || fn == "omp.out" {
    im = ""
    if      ($0 ~ /^naive \(32x8/)    im = "cuda_naive"
    else if ($0 ~ /^naive \(128x1/)   im = "cuda_rep1d"
    else if ($0 ~ /^tiled32/)         im = "cuda_tiled32"
    else if ($0 ~ /^regtile 64x64/)   im = "cuda_rt64"
    else if ($0 ~ /^regtile 128x128/) im = "cuda_rt128"
    else if ($0 ~ /^omp collapse/)    im = "omp"
    if (im != "" && match($0, /([0-9.]+) ms +([0-9.]+) +(.*)$/, m)) {
        PROG_MS[im] = m[1]
        v = m[3]; sub(/ *\(.*$/, "", v); CORRECT[im] = v
    }
    if ($0 ~ /^launch overhead/ && match($0, /([0-9.]+) us/, m))     LAUNCH[prog] = m[1]
    if ($0 ~ /^first call/      && match($0, /([0-9.]+) ms/, m))     FIRST[prog]  = m[1]
    next
}

fn == "omp_info.txt" {
    if ($0 ~ /matrixMultiplyOpenMP/ && match($0, / in (.+) mode/, m)) MODE = m[1]
    next
}

fn ~ /cuda_gpu_trace/ {
    if (!hdr) {
        if (NF < 3) next
        for (i = 1; i <= NF; i++) H[uq($i)] = i
        hdr = 1
        tS = col("Start"); tD = col("Duration"); tGX = col("GrdX"); tGY = col("GrdY")
        tBX = col("BlkX"); tBY = col("BlkY"); tR = col("Reg"); tSS = col("StcSMem")
        tDS = col("DymSMem"); tN = col("Name")
        durScale = unit_scale_ms(hname(tD))
        h = hname(tSS); smScale = (h ~ /MB/) ? 1e6 : (h ~ /KB|KiB/) ? 1e3 : 1
        next
    }
    if (!tGX || !tN || num($tGX) == 0) next            # memcpy / memset rows
    im = classify(uq($tN), num($tBX), num($tBY))
    if (im == "") next
    n = ++TN[im]
    TD[im, n] = num($tD) * durScale
    if (n == 1) {
        GX[im] = num($tGX); GY[im] = num($tGY); BX[im] = num($tBX); BY[im] = num($tBY)
        TREG[im] = num($tR); TSMEM[im] = int((num($tSS) + num($tDS)) * smScale + 0.5)
    }
    next
}

fn ~ /cuda_api_sum/ {
    if (!hdr) {
        if (NF < 3) next
        for (i = 1; i <= NF; i++) H[uq($i)] = i
        hdr = 1
        aMed = col("Med"); aName = col("Name")
        apiScale = unit_scale_ms(hname(aMed)) * 1e3   # -> microseconds
        next
    }
    nm = uq($aName)
    if (nm ~ /^(cudaLaunchKernel|cuLaunchKernel)/)          API_LAUNCH[prog] = num($aMed) * apiScale
    if (nm ~ /^(cudaDeviceSynchronize|cuStreamSynchronize)/) API_SYNC[prog]  = num($aMed) * apiScale
    next
}

fn ~ /^kern_/ {
    if (!hdr) { for (i = 1; i <= NF; i++) H[uq($i)] = i; hdr = 1; next }
    im = classify(uq($(H["name"])), num($(H["blockX"])), num($(H["blockY"])))
    if (im == "") next
    KREG[im]  = num($(H["regs"]))
    KSMEM[im] = num($(H["static_smem_B"])) + num($(H["dyn_smem_B"]))
    KLOC[im]  = uq($(H["local_B"]))
    next
}

fn ~ /^gm_/ {
    if (!hdr) { for (i = 1; i <= NF; i++) H[uq($i)] = i; hdr = 1; next }
    im = classify(uq($(H["name"])), num($(H["blockX"])), num($(H["blockY"])))
    if (im == "") next
    st = uq($(H["start_ns"]))
    if (!((im, st) in GMI)) GMI[im, st] = ++GMN[im]
    met = uq($(H["metric"]))
    GMV[im, met, GMI[im, st]] = num($(H["value"]))
    GMMET[met] = 1
    next
}

fn ~ /^ncu_/ {
    if (!hdr) {
        if (NF < 5) next
        for (i = 1; i <= NF; i++) H[uq($i)] = i
        hdr = 1
        nK = col("Kernel Name"); nB = col("Block Size"); nMN = col("Metric Name"); nMV = col("Metric Value")
        next
    }
    if (!nK || !nB || !nMN || !nMV) next
    b = uq($nB); gsub(/[^0-9]+/, " ", b); split(b, bb, " ")
    im = classify(uq($nK), bb[1] + 0, bb[2] + 0)
    if (im == "") next
    NCU[im, uq($nMN)] = num($nMV)
    if (col("Metric Unit")) NCUU[im, uq($nMN)] = uq($(col("Metric Unit")))
    next
}

END {
    site = META["site"]; gpu = META["gpu"]; gb = META["size_gb"]; N = META["N"] + 0; rep = META["rep"]
    flops = 2.0 * N * N * N
    nimp = split("cuda_naive cuda_tiled32 omp cuda_rep1d cuda_rt64 cuda_rt128", IMP, " ")

    for (i = 1; i <= nimp; i++) {
        im = IMP[i]
        n = TN[im] + 0
        if (n > 0) {                                  # GPU trace; launch 1 is the warm-up
            first = (n > 1) ? 2 : 1; cnt = 0; delete a
            for (j = first; j <= n; j++) a[++cnt] = TD[im, j]
            asort(a)
            TMED[im] = (cnt % 2) ? a[(cnt + 1) / 2] : (a[cnt / 2] + a[cnt / 2 + 1]) / 2
            TMIN[im] = a[1]; TMAX[im] = a[cnt]; TCNT[im] = cnt
        } else if (im in PROG_MS) {                   # no trace: program's mean
            TMED[im] = PROG_MS[im]; TCNT[im] = 0
        }
        if ((im in TMED) && TMED[im] > 0) GF[im] = flops / (TMED[im] / 1e3) / 1e9

        e = ptx_entry(im)
        if (im in KREG)       { REG[im] = KREG[im]; SMEM[im] = KSMEM[im]; LOC[im] = KLOC[im] }
        else if (e != "")     { REG[im] = PREG[e];  SMEM[im] = PSMEM[e] }
        else if (im in TREG)  { REG[im] = TREG[im]; SMEM[im] = TSMEM[im] }
        if (im in GX) {
            thr = BX[im] * BY[im]
            EPT[im] = (N * N) / (GX[im] * GY[im] * thr)
            if (im in REG) OCC[im] = occ_theory(REG[im], thr, SMEM[im], gpu)
        }

        r = gm_pick(im, "dram read.*%"); w = gm_pick(im, "dram write.*%")
        DRAM_GM[im] = (r == "") ? "" : r + (w == "" ? 0 : w)
        OCC_GM[im]  = gm_pick(im, "warps in flight.*throughput %")
        if (OCC_GM[im] == "") OCC_GM[im] = gm_pick(im, "warp occupancy.*%")
        SMACT_GM[im] = gm_pick(im, "sms active.*%")

        L1[im]   = ((im, "l1tex__t_sector_hit_rate.pct") in NCU) ? NCU[im, "l1tex__t_sector_hit_rate.pct"] : ""
        L2[im]   = ((im, "lts__t_sector_hit_rate.pct") in NCU) ? NCU[im, "lts__t_sector_hit_rate.pct"] : ""
        DRAMN[im]= ((im, "dram__throughput.avg.pct_of_peak_sustained_elapsed") in NCU) ? NCU[im, "dram__throughput.avg.pct_of_peak_sustained_elapsed"] : ""
        SMN[im]  = ((im, "sm__throughput.avg.pct_of_peak_sustained_elapsed") in NCU) ? NCU[im, "sm__throughput.avg.pct_of_peak_sustained_elapsed"] : ""
        OCCN[im] = ((im, "sm__warps_active.avg.pct_of_peak_sustained_active") in NCU) ? NCU[im, "sm__warps_active.avg.pct_of_peak_sustained_active"] : ""
        sec = NCU[im, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"]
        req = NCU[im, "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"]
        SPR[im] = (req > 0) ? sec / req : ""
        pg = (im == "omp") ? "omp" : "cuda"
        LW[im] = LAUNCH[pg]; LA[im] = API_LAUNCH[pg]; LS[im] = API_SYNC[pg]

        # Sanity checks. A kernel running alone on the GPU keeps nearly all SMs
        # busy; if nsys says otherwise, its samples missed the kernel. ncu values
        # of exactly 0 for occupancy or SM throughput are impossible for these
        # kernels (counter overflow in very long kernels); values from kernels
        # longer than 25 s at ncu's locked clock are kept but flagged.
        GMST[im] = (SMACT_GM[im] != "" && SMACT_GM[im] < 50) ? "invalid" : ""
        if (SMACT_GM[im] == "" && OCC_GM[im] != "" && OCC_GM[im] < 1) GMST[im] = "invalid"   # no "SMs Active" metric
        if (GMST[im] == "invalid") { OCC_GM[im] = ""; DRAM_GM[im] = "" }
        NCST[im] = ""
        if (OCCN[im] != "" || SMN[im] != "") {
            if ((OCCN[im] != "" && (OCCN[im] + 0 <= 0 || OCCN[im] + 0 > 110)) ||   # 0% or far above 100%
                (SMN[im]  != "" && SMN[im] + 0 <= 0) ||
                (SPR[im]  != "" && SPR[im] + 0 > 32) ||                              # a warp touches <= 32 sectors
                (L2[im]   != "" && L2[im] + 0 >= 99.9))                              # GEMM on GBs of data
                NCST[im] = "invalid"
            else {
                d = NCU[im, "gpu__time_duration.sum"] + 0; u = NCUU[im, "gpu__time_duration.sum"]
                ds = (u ~ /^n/) ? d / 1e9 : (u ~ /^u/) ? d / 1e6 : (u ~ /^m/) ? d / 1e3 : d
                if (ds > 25) NCST[im] = "suspect"
            }
        }
        if (NCST[im] == "invalid") { L1[im] = ""; L2[im] = ""; DRAMN[im] = ""; SMN[im] = ""; OCCN[im] = ""; SPR[im] = "" }
    }

    # ---------------- table ----------------
    nc = split("cuda_naive cuda_tiled32 omp cuda_rep1d", C, " ")
    LBL["cuda_naive"] = "CUDA naive"; LBL["cuda_tiled32"] = "CUDA tiled32"
    LBL["omp"] = "OMP collapse(2)";   LBL["cuda_rep1d"] = "CUDA 1-D (ref)"
    printf "=== %s | %s | %s GB (N = %d) | rep %s ===\n", site, gpu, gb, N, rep
    printf "%-28s", ""; for (c = 1; c <= nc; c++) printf "  %-25s", LBL[C[c]]; print ""

    for (row = 1; row <= 17; row++) {
        name = ""
        for (c = 1; c <= nc; c++) {
            im = C[c]; v = "n/a"
            if (row == 1)  { name = "Time, median [min-max]"
                             if (im in TMED) v = (TCNT[im] > 0) ? tfmt3(TMED[im], TMIN[im], TMAX[im]) \
                                                                : tfmt3(TMED[im], "", "") " prog mean" }
            if (row == 2)  { name = "GFLOP/s";                  if (im in GF) v = sprintf("%.0f", GF[im]) }
            if (row == 3)  { name = "OMP slower by"
                             v = (im == "omp") ? "-" : ((im in TMED) && ("omp" in TMED)) ? sprintf("%.2fx", TMED["omp"] / TMED[im]) : "n/a" }
            if (row == 4)  { name = "Regs / smem B / local B"
                             if (im in REG) v = REG[im] " / " SMEM[im] " / " ((LOC[im] == "") ? "?" : LOC[im]) }
            if (row == 5)  { name = "Launch: blocks x threads"; if (im in GX) v = sprintf("%d x %d (%dx%d)", GX[im] * GY[im], BX[im] * BY[im], BX[im], BY[im]) }
            if (row == 6)  { name = "Elements per thread";      if (im in EPT) v = sprintf("%.0f", EPT[im]) }
            if (row == 7)  { name = "Occupancy theoretical %";  if (im in OCC) v = sprintf("%.1f", OCC[im]) }
            if (row == 8)  { name = "Occupancy achieved % nsys"; v = gmv(im, OCC_GM[im]) }
            if (row == 9)  { name = "Occupancy achieved % ncu";  v = ncv(im, f1(OCCN[im])) }
            if (row == 10) { name = "DRAM throughput % nsys";    v = gmv(im, DRAM_GM[im]) }
            if (row == 11) { name = "DRAM throughput % ncu";     v = ncv(im, f1(DRAMN[im])) }
            if (row == 12) { name = "L1 / L2 hit % ncu";         v = ncv(im, (L1[im] == "") ? "n/a" : sprintf("%.1f / %.1f", L1[im], L2[im])) }
            if (row == 13) { name = "SM throughput % ncu";       v = ncv(im, f1(SMN[im])) }
            if (row == 14) { name = "Sectors per request ncu";   v = ncv(im, (SPR[im] == "") ? "n/a" : sprintf("%.2f", SPR[im])) }
            if (row == 15) { name = "Launch overhead us (wall)"
                             v = (im == "cuda_naive" || im == "omp") ? f1(LW[im]) : "same as naive" }
            if (row == 16) { name = "Launch / sync call med us"
                             v = (im == "cuda_naive" || im == "omp") ? (f1(LA[im]) " / " f1(LS[im])) : "same as naive" }
            if (row == 17) { name = "Kernel mode | correctness"
                             v = ((im == "omp") ? ((MODE == "") ? "?" : MODE) : "CUDA") " | " ((im in CORRECT) ? CORRECT[im] : "n/a") }
            val[c] = v
        }
        printf "%-28s", name; for (c = 1; c <= nc; c++) printf "  %-25s", val[c]; print ""
    }
    if ("omp" in FIRST) printf "OpenMP first call (warm-up, incl. runtime setup): %s ms\n", FIRST["omp"]
    for (c = 1; c <= nc; c++) { if (GMST[C[c]] == "invalid") fg = 1; if (NCST[C[c]] == "invalid") fn_ = 1; if (NCST[C[c]] == "suspect") fs = 1 }
    if (fg)  print "invalid (nsys): SMs Active < 50% during the kernel, so the GPU-metric samples missed it"
    if (fn_) print "invalid (ncu):  impossible values (0% or >110% occupancy, 0% SM throughput, >32 sectors/request or 100% L2 hits): counter overflow in a very long kernel"
    if (fs)  print "*  ncu value from a kernel longer than 25 s at ncu's clock; treat with caution"

    # ---------------- summary.csv ----------------
    head = "site,gpu,size_gb,N,rep,impl,t_med_ms,t_min_ms,t_max_ms,n_timed,t_prog_mean_ms,gflops," \
           "omp_slowdown_vs_this,regs,smem_B,local_B,blocks,threads,block_shape,elems_per_thread," \
           "occ_theory_pct,occ_ach_nsys_pct,occ_ach_ncu_pct,sm_active_nsys_pct,dram_nsys_pct," \
           "dram_ncu_pct,l1_hit_ncu_pct,l2_hit_ncu_pct,sm_thr_ncu_pct,sectors_per_req," \
           "launch_wall_us,launch_call_med_us,sync_call_med_us,omp_mode,correctness,nsys_gm_state,ncu_state"
    if ((getline tmp < summary) <= 0) print head > summary
    close(summary)
    for (i = 1; i <= nimp; i++) {
        im = IMP[i]
        if (!(im in TMED) && !(im in GX)) continue
        slow = ((im in TMED) && ("omp" in TMED) && im != "omp") ? sprintf("%.4f", TMED["omp"] / TMED[im]) : ""
        printf "%s,\"%s\",%s,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
            site, gpu, gb, N, rep, im, TMED[im], TMIN[im], TMAX[im], TCNT[im], PROG_MS[im],
            (im in GF) ? sprintf("%.1f", GF[im]) : "", slow, REG[im], SMEM[im], LOC[im],
            (im in GX) ? GX[im] * GY[im] : "", (im in GX) ? BX[im] * BY[im] : "",
            (im in GX) ? BX[im] "x" BY[im] : "", (im in EPT) ? sprintf("%.1f", EPT[im]) : "",
            (im in OCC) ? sprintf("%.1f", OCC[im]) : "", OCC_GM[im], OCCN[im], SMACT_GM[im],
            DRAM_GM[im], DRAMN[im], L1[im], L2[im], SMN[im], SPR[im],
            LW[im], LA[im], LS[im], (im == "omp") ? MODE : "", CORRECT[im],
            (GMST[im] == "") ? "ok" : GMST[im], (NCST[im] == "") ? ((OCCN[im] == "") ? "none" : "ok") : NCST[im] >> summary
    }
    close(summary)

    # ---------------- plot row (chosen repetition only) ----------------
    if (rep == plotrep) {
        phead = "GB N gf_naive gf_tiled32 gf_omp gf_rep1d gf_rt64 gf_rt128 slow_naive slow_tiled32 " \
                "dram_naive dram_tiled32 dram_omp dram_rep1d occth_naive occth_tiled32 occth_omp " \
                "occ_naive occ_tiled32 occ_omp l1_naive l1_tiled32 l1_rep1d launch_cuda launch_omp"
        if ((getline tmp < plotfile) <= 0) print phead > plotfile
        close(plotfile)
        sn = (("omp" in TMED) && ("cuda_naive" in TMED)) ? TMED["omp"] / TMED["cuda_naive"] : ""
        st = (("omp" in TMED) && ("cuda_tiled32" in TMED)) ? TMED["omp"] / TMED["cuda_tiled32"] : ""
        print gb, N, nan(GF["cuda_naive"]), nan(GF["cuda_tiled32"]), nan(GF["omp"]), nan(GF["cuda_rep1d"]),
              nan(GF["cuda_rt64"]), nan(GF["cuda_rt128"]), nan(sn), nan(st),
              nan(DRAM_GM["cuda_naive"]), nan(DRAM_GM["cuda_tiled32"]), nan(DRAM_GM["omp"]), nan(DRAM_GM["cuda_rep1d"]),
              nan(OCC["cuda_naive"]), nan(OCC["cuda_tiled32"]), nan(OCC["omp"]),
              nan(OCC_GM["cuda_naive"]), nan(OCC_GM["cuda_tiled32"]), nan(OCC_GM["omp"]),
              nan(L1["cuda_naive"]), nan(L1["cuda_tiled32"]), nan(L1["cuda_rep1d"]),
              nan(LAUNCH["cuda"]), nan(LAUNCH["omp"]) >> plotfile
        close(plotfile)
    }

    # ---------------- every GPU metric, for inspection ----------------
    if (gmfile != "") {
        printf "# per-kernel averages of all nsys GPU metrics (timed launches only)\n" > gmfile
        for (i = 1; i <= nimp; i++)
            for (met in GMMET) { v = gm_avg(IMP[i], met); if (v != "") printf "%-14s %-60s %10.3f\n", IMP[i], met, v > gmfile }
        close(gmfile)
    }
}

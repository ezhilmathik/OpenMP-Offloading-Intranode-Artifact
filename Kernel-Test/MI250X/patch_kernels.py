# patch_kernels.py -- add a kernel selector to mat-hip.cpp / mat-cuda.cu
#
#   python3 patch_kernels.py mat-hip.cpp
#
# Afterwards the program takes an optional third argument:
#   ./matmul_hip <N> [reps] [kernels]
# where "kernels" is a comma-separated list of naive, naive1d, tiled32,
# regtile64, regtile128 (default: all). The run script passes KERNELS=... here.
import sys

p = sys.argv[1]
s = open(p).read()
if 'const char *sel' in s:
    print(p, "already patched"); sys.exit(0)

reps_line = '    const int reps = argc > 2 ? atoi(argv[2]) : 5;'
assert reps_line in s, p + ": reps line not found"
s = s.replace(reps_line, reps_line + '''
    const char *sel = argc > 3 ? argv[3] : "";        // "" = all kernels
    auto want = [&](const char *key) {
        if (!*sel) return true;
        const size_t n = strlen(key);
        for (const char *q = strstr(sel, key); q; q = strstr(q + 1, key))
            if ((q == sel || q[-1] == ',') && (q[n] == '\\0' || q[n] == ',')) return true;
        return false;
    };''', 1)

run_old = '    auto run = [&](const char* name, auto launch) {'
assert run_old in s, p + ": run lambda not found"
s = s.replace(run_old, '''    auto run = [&](const char* key, const char* name, auto launch) {
        if (!want(key)) return;''', 1)

for call, key in [('run("naive (32x8 threads)"', 'naive'),
                  ('run("naive (128x1, OMP-like)"', 'naive1d'),
                  ('run("tiled32 (your original)"', 'tiled32'),
                  ('run("regtile 64x64x16, 4x4"', 'regtile64'),
                  ('run("regtile 128x128x8, 8x8"', 'regtile128')]:
    assert call in s, p + ": " + call
    s = s.replace(call, 'run("' + key + '", "' + call[5:], 1)

if '#include <cstring>' not in s:
    s = s.replace('#include <cstdlib>', '#include <cstdlib>\n#include <cstring>', 1)
s = s.replace('Usage: %s <N> [reps=5]\\n', 'Usage: %s <N> [reps=5] [kernels]\\n', 1)
open(p, 'w').write(s)
print("patched", p)

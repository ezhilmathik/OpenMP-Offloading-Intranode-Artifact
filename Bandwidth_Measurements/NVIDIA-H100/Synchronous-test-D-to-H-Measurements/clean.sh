#!/bin/bash
# Delete everything except source files, scripts, and the Makefile
find . -type f ! \( \
    -name "Makefile"  \
    -o -name "*.sh"   \
    -o -name "*.cu"   \
    -o -name "*.cc"   \
    -o -name "*.c"    \
    -o -name "*.py"   \
\) -delete
echo "Cleaned - kept: Makefile, *.sh, *.cu, *.cc, *.c, *.py (directories preserved)"

#!/bin/bash
# Build the WASM engine (cpp/wasm.cpp) → webapp/tgwasm.js (single file, wasm embedded).
# SIMD128 on; single-threaded (GitHub Pages sends no COOP/COEP headers, so no SharedArrayBuffer).
set -e
cd "$(dirname "$0")/.."
emcc -O3 -msimd128 -std=c++17 cpp/wasm.cpp -o webapp/tgwasm.js \
  -s MODULARIZE=1 -s EXPORT_NAME=TGWasm -s SINGLE_FILE=1 \
  -s ENVIRONMENT=web,worker,node -s ALLOW_MEMORY_GROWTH=1 \
  -s EXPORTED_FUNCTIONS=_tg_init,_tg_search,_tg_eval,_tg_n,_malloc,_free \
  -s EXPORTED_RUNTIME_METHODS=HEAPF32,HEAP32,HEAPU8 \
  -s ASSERTIONS=0 -s FILESYSTEM=0
ls -la webapp/tgwasm.js

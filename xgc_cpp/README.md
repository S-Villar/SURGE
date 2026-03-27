# XGC Surrogate C++ Inference

Standalone C++ module for running SURGE-trained XGC A_parallel models (Model1, Model12) via ONNX Runtime. **Does not modify** `/global/cfs/projectdirs/m499/sku/XGC-Devel`.

## Reference

- **XGC source:** `XGC-Devel/XGC_core/cpp/solvers/get_dAdt_guess.cpp` → `set_nn_correction_decomp()`
- This API provides a callable NN-based correction that can replace the linear correction in `set_nn_correction_decomp`.

## Input: 61 variables

Model1 and Model12 use the **first 61 columns** (current timestep only) from the OLCF hackathon data layout. See `docs/xgc/XGC_INPUT_STRUCTURE.md` and `docs/xgc/XGC_SUMMARY.md` Section 0.

## Export model to ONNX

```bash
python scripts/xgc_export_onnx.py --run-dir runs/xgc_model1_61cols
python scripts/xgc_export_onnx.py --run-dir runs/xgc_model12_finetune --model xgc_mlp_aparallel
```

Output: `run_dir/onnx/xgc_mlp_aparallel.onnx` and `run_dir/onnx/scaler_params.json`

## Build

```bash
cd xgc_cpp
mkdir build && cd build
cmake .. -DUSE_ONNXRUNTIME=ON -DONNXRUNTIME_ROOT=/path/to/onnxruntime
make
```

With system ONNX Runtime:

```bash
cmake .. -DUSE_ONNXRUNTIME=ON
make
```

## API

```c
#include "xgc_surrogate_infer.h"

// Init (load model + scalers)
xgc_surrogate_init("/path/to/model.onnx", "/path/to/scaler_params.json");

// Inference - 61 inputs, 2 outputs (outputs[1] = A_parallel)
float inputs[61];
float outputs[2];
xgc_surrogate_infer_single(inputs, 61, outputs, 2);

// Switch to Model12
xgc_surrogate_switch_model("/path/to/model12.onnx", "/path/to/model12/scaler_params.json");

xgc_surrogate_finalize();
```

## Integration into XGC

To replace `set_nn_correction_decomp` in XGC:

1. Copy this `xgc_cpp/` directory into your XGC build tree (outside XGC-Devel).
2. Build `libxgc_surrogate_infer.a` with ONNX Runtime.
3. Create an alternative `get_dAdt_guess_nn.cpp` that:
   - Calls `xgc_surrogate_init()` at startup.
   - In the NN correction branch, builds the 61 input features from `input.correction_previous`, `input.dAhdt`, and other XGC state.
   - Calls `xgc_surrogate_infer_single()` and writes the result into `correction`.
4. Link XGC against `libxgc_surrogate_infer.a` and ONNX Runtime.

The mapping from XGC state to the 61 inputs must match the data preprocessing used in SURGE training.

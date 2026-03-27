/**
 * XGC Surrogate C++ Inference API
 *
 * Loads SURGE-trained Model1 (61 cols) or Model12 (finetuned) via ONNX Runtime
 * and provides inference for A_parallel prediction. Does not modify XGC-Devel.
 *
 * Reference: XGC-Devel/XGC_core/cpp/solvers/get_dAdt_guess.cpp -> set_nn_correction_decomp()
 */

#ifndef XGC_SURROGATE_INFER_H
#define XGC_SURROGATE_INFER_H

#ifdef __cplusplus
extern "C" {
#endif

/** Initialize surrogate: load ONNX model and scaler params from JSON.
 *  onnx_path: path to .onnx file (e.g. runs/xgc_model1_61cols/onnx/xgc_mlp_aparallel.onnx)
 *  scaler_path: path to scaler_params.json in same onnx dir
 *  Returns 0 on success, -1 on error. */
int xgc_surrogate_init(const char* onnx_path, const char* scaler_path);

/** Run inference for a single sample. Inputs must be 61 floats (Model1/Model12).
 *  Applies input scaling, runs ONNX, applies output unscaling.
 *  outputs[0] = output_0, outputs[1] = A_parallel
 *  Returns 0 on success, -1 on error. */
int xgc_surrogate_infer_single(const float* inputs, int n_inputs, float* outputs, int n_outputs);

/** Switch to a different model (e.g. Model12). Same init semantics. */
int xgc_surrogate_switch_model(const char* onnx_path, const char* scaler_path);

/** Release resources. */
void xgc_surrogate_finalize(void);

#ifdef __cplusplus
}
#endif

#endif /* XGC_SURROGATE_INFER_H */

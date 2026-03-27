/**
 * Minimal test for xgc_surrogate_infer.
 * Run from SURGE root: ./xgc_surrogate_test
 * Expects: runs/xgc_model1_61cols/onnx/xgc_mlp_aparallel.onnx and scaler_params.json
 */

#include "xgc_surrogate_infer.h"
#include <stdio.h>
#include <cstring>

int main(int argc, char** argv) {
    const char* onnx = (argc > 1) ? argv[1] : "runs/xgc_model1_61cols/onnx/xgc_mlp_aparallel.onnx";
    const char* scaler = (argc > 2) ? argv[2] : "runs/xgc_model1_61cols/onnx/scaler_params.json";

    if (xgc_surrogate_init(onnx, scaler) != 0) {
        fprintf(stderr, "Failed to init surrogate (onnx=%s, scaler=%s)\n", onnx, scaler);
        return 1;
    }

    float inputs[61] = {0};
    for (int i = 0; i < 61; i++) inputs[i] = 0.1f * (i % 10);

    float outputs[2];
    if (xgc_surrogate_infer_single(inputs, 61, outputs, 2) != 0) {
        fprintf(stderr, "Inference failed\n");
        xgc_surrogate_finalize();
        return 1;
    }

    printf("output_0=%.6f, A_parallel=output_1=%.6f\n", outputs[0], outputs[1]);
    xgc_surrogate_finalize();
    return 0;
}

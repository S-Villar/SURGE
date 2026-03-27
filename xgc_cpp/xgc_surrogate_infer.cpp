/**
 * XGC Surrogate C++ Inference - ONNX Runtime implementation
 *
 * Uses 61 input variables (Model1/Model12). Input scaling and output unscaling
 * from scaler_params.json. output[1] = A_parallel.
 */

#include "xgc_surrogate_infer.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>

#ifdef USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

static bool g_initialized = false;
static int g_n_inputs = 61;
static int g_n_outputs = 2;
static std::vector<float> g_input_mean;
static std::vector<float> g_input_scale;
static std::vector<float> g_output_mean;
static std::vector<float> g_output_scale;

#ifdef USE_ONNXRUNTIME
static Ort::Env* g_env = nullptr;
static Ort::Session* g_session = nullptr;
static Ort::MemoryInfo g_mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
#endif

/* Minimal JSON parsing for scaler_params.json - extract float arrays by key */
static int parse_scaler_json(const char* path) {
    std::ifstream f(path);
    if (!f) return -1;
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    f.close();

    auto extract_array = [&s](const char* key, std::vector<float>& out) -> bool {
        std::string k(key);
        size_t pos = s.find("\"" + k + "\"");
        if (pos == std::string::npos) return false;
        pos = s.find("[", pos);
        if (pos == std::string::npos) return false;
        size_t end = s.find("]", pos);
        if (end == std::string::npos) return false;
        std::string arr = s.substr(pos + 1, end - pos - 1);
        out.clear();
        for (size_t i = 0; i < arr.size(); ) {
            while (i < arr.size() && (arr[i] == ',' || arr[i] == ' ' || arr[i] == '\n' || arr[i] == '\r')) i++;
            if (i >= arr.size()) break;
            size_t j = i;
            while (j < arr.size() && arr[j] != ',' && arr[j] != ']') j++;
            float v = (float)std::atof(arr.substr(i, j - i).c_str());
            out.push_back(v);
            i = j;
        }
        return !out.empty();
    };

    if (!extract_array("input_mean", g_input_mean) ||
        !extract_array("input_scale", g_input_scale) ||
        !extract_array("output_mean", g_output_mean) ||
        !extract_array("output_scale", g_output_scale)) {
        return -1;
    }
    return 0;
}

int xgc_surrogate_init(const char* onnx_path, const char* scaler_path) {
#ifdef USE_ONNXRUNTIME
    if (g_initialized) {
        if (g_session) delete g_session;
        g_session = nullptr;
    }

    if (parse_scaler_json(scaler_path) != 0) return -1;
    g_n_inputs = (int)g_input_mean.size();
    g_n_outputs = (int)g_output_mean.size();

    try {
        if (!g_env) g_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "xgc_surrogate");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        g_session = new Ort::Session(*g_env, onnx_path, opts);
        g_initialized = true;
        return 0;
    } catch (...) {
        return -1;
    }
#else
    (void)onnx_path;
    (void)scaler_path;
    return -1;  /* ONNX Runtime not linked */
#endif
}

int xgc_surrogate_switch_model(const char* onnx_path, const char* scaler_path) {
    return xgc_surrogate_init(onnx_path, scaler_path);
}

int xgc_surrogate_infer_single(const float* inputs, int n_inputs, float* outputs, int n_outputs) {
#ifdef USE_ONNXRUNTIME
    if (!g_initialized || !g_session || n_inputs != g_n_inputs || n_outputs != g_n_outputs)
        return -1;

    /* Input scaling: (x - mean) / scale */
    std::vector<float> scaled_input(g_n_inputs);
    for (int i = 0; i < g_n_inputs; i++) {
        float scale = g_input_scale[i] + 1e-10f;
        scaled_input[i] = (inputs[i] - g_input_mean[i]) / scale;
    }

    try {
        std::vector<int64_t> shape = {1, g_n_inputs};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            g_mem_info, scaled_input.data(), scaled_input.size(),
            shape.data(), shape.size());

        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};
        auto out = g_session->Run(Ort::RunOptions{nullptr},
                                 input_names, &input_tensor, 1,
                                 output_names, 1);

        float* raw_out = out[0].GetTensorMutableData<float>();
        size_t out_len = out[0].GetTensorTypeAndShapeInfo().GetElementCount();

        /* Output unscaling: y * scale + mean */
        for (size_t i = 0; i < (size_t)n_outputs && i < out_len; i++) {
            float scale = g_output_scale[i] + 1e-10f;
            outputs[i] = raw_out[i] * scale + g_output_mean[i];
        }
        return 0;
    } catch (...) {
        return -1;
    }
#else
    (void)inputs;
    (void)n_inputs;
    (void)outputs;
    (void)n_outputs;
    return -1;
#endif
}

void xgc_surrogate_finalize(void) {
#ifdef USE_ONNXRUNTIME
    if (g_session) {
        delete g_session;
        g_session = nullptr;
    }
    if (g_env) {
        delete g_env;
        g_env = nullptr;
    }
#endif
    g_initialized = false;
}

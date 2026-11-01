#include <torch/extension.h>

torch::Tensor bc1_decode_cuda(torch::Tensor endpoints, torch::Tensor weights,
                             torch::Tensor modes, int64_t height, int64_t width, int64_t channels);
std::vector<torch::Tensor> bc1_backward_cuda(torch::Tensor gradient, torch::Tensor endpoints,
    torch::Tensor weights, torch::Tensor modes, torch::Tensor temperature,
    int64_t height, int64_t width, int64_t channels);
std::vector<torch::Tensor> bc1_usage_cuda(torch::Tensor weights, torch::Tensor modes,
                                       torch::Tensor temperature, int64_t surfaces);
std::vector<torch::Tensor> bc1_usage_backward_cuda(torch::Tensor gradient, torch::Tensor weights,
    torch::Tensor modes, torch::Tensor temperature, torch::Tensor occupancy_gradient, int64_t surfaces);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode", &bc1_decode_cuda);
    m.def("backward", &bc1_backward_cuda);
    m.def("usage", &bc1_usage_cuda);
    m.def("usage_backward", &bc1_usage_backward_cuda);
}

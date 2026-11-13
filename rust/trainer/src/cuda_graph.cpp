// CUDA graphs for the self-play network: tch has no binding for
// at::cuda::CUDAGraph, so this is a minimal C ABI around it.
//
// Capture: rv_graph_begin() makes a side stream current and starts capturing
// it; every libtorch op the calling thread issues until rv_graph_end() is
// recorded instead of run. rv_graph_replay() then launches the recorded
// kernels with one call on the current stream. All graphs share one memory
// pool: they are replayed one at a time, on one thread.

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>

#include <cstring>
#include <exception>
#include <optional>

namespace {

struct Graph {
  at::cuda::CUDAGraph graph;
  std::optional<c10::cuda::CUDAStream> capture_stream;
};

void set_error(char* error, size_t len, const char* message) {
  if (error != nullptr && len > 0) {
    std::strncpy(error, message, len - 1);
    error[len - 1] = '\0';
  }
}

}  // namespace

extern "C" {

// A new graph whose capture has started on the current device, or null (and
// `error` set). `pool` is shared by all graphs: pass the same handle.
void* rv_graph_begin(int64_t pool_a, int64_t pool_b, char* error, size_t error_len) {
  Graph* g = nullptr;
  try {
    g = new Graph();
    // The capture stream starts after everything queued so far (captures
    // are rare: a full synchronization is fine).
    c10::cuda::getCurrentCUDAStream().synchronize();
    auto stream = c10::cuda::getStreamFromPool();
    g->capture_stream = stream;
    c10::cuda::setCurrentCUDAStream(stream);
    g->graph.capture_begin({static_cast<uint64_t>(pool_a), static_cast<uint64_t>(pool_b)},
                           cudaStreamCaptureModeThreadLocal);
    return g;
  } catch (const std::exception& e) {
    set_error(error, error_len, e.what());
    c10::cuda::setCurrentCUDAStream(c10::cuda::getDefaultCUDAStream());
    delete g;
    return nullptr;
  }
}

// End the capture and restore the default stream; false (and `error` set) on
// failure, after which the graph must only be freed.
bool rv_graph_end(void* handle, char* error, size_t error_len) {
  auto* g = static_cast<Graph*>(handle);
  bool ok = true;
  try {
    g->graph.capture_end();
  } catch (const std::exception& e) {
    set_error(error, error_len, e.what());
    ok = false;
  }
  c10::cuda::setCurrentCUDAStream(c10::cuda::getDefaultCUDAStream());
  if (ok) {
    g->capture_stream->synchronize();
  }
  return ok;
}

bool rv_graph_replay(void* handle, char* error, size_t error_len) {
  try {
    static_cast<Graph*>(handle)->graph.replay();
    return true;
  } catch (const std::exception& e) {
    set_error(error, error_len, e.what());
    return false;
  }
}

void rv_graph_free(void* handle) {
  try {
    delete static_cast<Graph*>(handle);
  } catch (...) {
  }
}

// A memory pool handle for rv_graph_begin.
void rv_graph_pool(int64_t* a, int64_t* b) {
  auto pool = at::cuda::graph_pool_handle();
  *a = static_cast<int64_t>(pool.first);
  *b = static_cast<int64_t>(pool.second);
}

}  // extern "C"

//! CUDA graphs for the network's forward pass (`cuda_graph.cpp`).
//!
//! A forward pass of this small network is ~30 kernels that each run for a
//! few microseconds; at self-play batch sizes (~800 positions) launching them
//! costs as much as running them. A graph records the kernels once per batch
//! size and launches them with a single call: on an RTX 3080 Laptop, 768
//! positions took 1.07 ms instead of 1.55.
//!
//! A graph keeps the addresses it was captured with: the input and output
//! tensors stay in place (new positions are copied into the input), and new
//! weights are copied into the network's tensors
//! ([`TorchNet::copy_weights_from`]) instead of replacing them.

use std::collections::HashMap;

use tch::Kind;
use tch::Tensor;

use crate::log_selfplay;
use crate::net::TorchNet;

#[cfg(rv_cuda_graphs)]
mod ffi {
    use std::ffi::c_char;
    use std::ffi::c_void;

    unsafe extern "C" {
        pub fn rv_graph_begin(
            pool_a: i64,
            pool_b: i64,
            error: *mut c_char,
            error_len: usize,
        ) -> *mut c_void;
        pub fn rv_graph_end(handle: *mut c_void, error: *mut c_char, error_len: usize) -> bool;
        pub fn rv_graph_replay(handle: *mut c_void, error: *mut c_char, error_len: usize) -> bool;
        pub fn rv_graph_free(handle: *mut c_void);
        pub fn rv_graph_pool(a: *mut i64, b: *mut i64);
    }
}

/// One captured graph.
struct CudaGraph(*mut std::ffi::c_void);

#[cfg(rv_cuda_graphs)]
fn error_text(buffer: &[u8]) -> String {
    let end = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
    String::from_utf8_lossy(&buffer[..end]).into_owned()
}

impl CudaGraph {
    /// Record the CUDA work of `f` (issued on this thread) into a graph.
    #[cfg(rv_cuda_graphs)]
    fn capture<R>(pool: (i64, i64), f: impl FnOnce() -> R) -> Result<(CudaGraph, R), String> {
        let mut error = [0u8; 512];
        // SAFETY: the buffer outlives the call and its length is passed.
        let handle =
            unsafe { ffi::rv_graph_begin(pool.0, pool.1, error.as_mut_ptr().cast(), error.len()) };
        if handle.is_null() {
            return Err(error_text(&error));
        }
        let graph = CudaGraph(handle);
        let result = f();
        // SAFETY: `handle` came from rv_graph_begin and is ended once.
        if unsafe { ffi::rv_graph_end(graph.0, error.as_mut_ptr().cast(), error.len()) } {
            Ok((graph, result))
        } else {
            Err(error_text(&error))
        }
    }

    #[cfg(not(rv_cuda_graphs))]
    fn capture<R>(_pool: (i64, i64), _f: impl FnOnce() -> R) -> Result<(CudaGraph, R), String> {
        Err("built without CUDA graph support".into())
    }

    #[cfg(rv_cuda_graphs)]
    fn replay(&self) -> Result<(), String> {
        let mut error = [0u8; 512];
        // SAFETY: a successfully captured graph; the buffer outlives the call.
        if unsafe { ffi::rv_graph_replay(self.0, error.as_mut_ptr().cast(), error.len()) } {
            Ok(())
        } else {
            Err(error_text(&error))
        }
    }

    #[cfg(not(rv_cuda_graphs))]
    fn replay(&self) -> Result<(), String> {
        unreachable!("no graph is ever captured")
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        #[cfg(rv_cuda_graphs)]
        // SAFETY: the handle came from rv_graph_begin and is freed once.
        unsafe {
            ffi::rv_graph_free(self.0)
        }
    }
}

#[cfg(rv_cuda_graphs)]
fn new_pool() -> (i64, i64) {
    let (mut a, mut b) = (0, 0);
    // SAFETY: two valid out-pointers.
    unsafe { ffi::rv_graph_pool(&mut a, &mut b) };
    (a, b)
}

#[cfg(not(rv_cuda_graphs))]
fn new_pool() -> (i64, i64) {
    (0, 0)
}

struct Captured {
    graph: CudaGraph,
    /// Packed positions `(n, 4)` int64 on the device: the graph's input.
    input: Tensor,
    /// `(n, 65)` f32 on the device: the graph's output.
    output: Tensor,
}

/// The network's forward pass on packed positions, through one CUDA graph
/// per padded batch size, captured on first use. Falls back to running the
/// network directly when graphs are off or a capture fails.
pub struct GraphedForward {
    enabled: bool,
    pool: Option<(i64, i64)>,
    graphs: HashMap<usize, Captured>,
}

impl GraphedForward {
    pub fn new(enabled: bool) -> Self {
        GraphedForward {
            enabled: enabled && cfg!(rv_cuda_graphs),
            pool: None,
            graphs: HashMap::new(),
        }
    }

    /// Forget the captured graphs (the network was replaced).
    pub fn reset(&mut self) {
        self.graphs.clear();
    }

    /// Network outputs for `packed` (a padded batch, 4 int64 per position)
    /// into `out` (65 f32 per position).
    pub fn run(&mut self, net: &TorchNet, packed: &[i64], out: &mut [f32]) {
        let n = packed.len() / 4;
        if self.enabled {
            match self.replay(net, packed, n, out) {
                Ok(()) => return,
                Err(error) => {
                    log_selfplay!(
                        "cuda-graphs-off",
                        process = 0,
                        batch = n,
                        error = format!("{error:?}")
                    );
                    self.enabled = false;
                    self.graphs.clear();
                }
            }
        }
        net.forward(&net.unpack(packed)).copy_data(out, n * 65);
    }

    fn replay(
        &mut self,
        net: &TorchNet,
        packed: &[i64],
        n: usize,
        out: &mut [f32],
    ) -> Result<(), String> {
        if !self.graphs.contains_key(&n) {
            let captured = self.capture(net, n)?;
            self.graphs.insert(n, captured);
        }
        let captured = self.graphs.get_mut(&n).expect("just inserted");
        let host = Tensor::from_slice(packed).view([n as i64, 4]);
        captured.input.copy_(&host);
        captured.graph.replay()?;
        captured.output.copy_data(out, n * 65);
        Ok(())
    }

    fn capture(&mut self, net: &TorchNet, n: usize) -> Result<Captured, String> {
        let pool = *self.pool.get_or_insert_with(new_pool);
        let input = Tensor::zeros([n as i64, 4], (Kind::Int64, net.device()));
        // Outside the capture first: cuDNN picks its algorithms and the
        // allocator warms up for this shape.
        for _ in 0..3 {
            let _ = net.forward(&net.unpack_device(&input));
        }
        let (graph, output) = CudaGraph::capture(pool, || net.forward(&net.unpack_device(&input)))?;
        Ok(Captured { graph, input, output })
    }
}

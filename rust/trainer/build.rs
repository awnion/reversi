//! Link flags for libtorch from the Python environment.
//!
//! - rpath to the libtorch directory, so the binary starts without
//!   `LD_LIBRARY_PATH` (the learner sets it anyway);
//! - on Linux, `libtorch_cuda.so` with `--no-as-needed`: nothing in the
//!   binary references it directly, so the default `--as-needed` would drop
//!   it and libtorch would silently come up without CUDA;
//! - with CUDA, `src/cuda_graph.cpp` (CUDA graphs, which tch does not bind)
//!   compiled against the same libtorch headers, and `cfg(rv_cuda_graphs)`.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cuda_graph.cpp");
    println!("cargo::rustc-check-cfg=cfg(rv_cuda_graphs)");
    // Published by torch-sys (`links = "tch"`, `cargo:libtorch_lib=...`).
    let Ok(lib_dir) = env::var("DEP_TCH_LIBTORCH_LIB") else {
        return;
    };
    let lib_dir = PathBuf::from(lib_dir);
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" || target_os == "macos" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    }
    if target_os == "linux" && lib_dir.join("libtorch_cuda.so").exists() {
        println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
        println!("cargo:rustc-link-arg=-L{}", lib_dir.display());
        println!("cargo:rustc-link-arg=-ltorch_cuda");
        println!("cargo:rustc-link-arg=-Wl,--as-needed");
        build_cuda_graphs(&lib_dir);
    }
}

/// The CUDA runtime headers: the ones of the CUDA wheels next to torch
/// (`site-packages/nvidia/*/include`), else `$CUDA_HOME` or /usr/local/cuda.
fn cuda_include(lib_dir: &std::path::Path) -> Option<PathBuf> {
    let has_runtime = |dir: &PathBuf| dir.join("cuda_runtime_api.h").is_file();
    let nvidia = lib_dir.join("../../nvidia");
    let mut wheels: Vec<PathBuf> = std::fs::read_dir(&nvidia)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path().join("include"))
        .filter(has_runtime)
        .collect();
    wheels.sort();
    let home = env::var("CUDA_HOME").map(|h| PathBuf::from(h).join("include"));
    wheels
        .pop()
        .or(home.ok().filter(has_runtime))
        .or_else(|| Some(PathBuf::from("/usr/local/cuda/include")).filter(has_runtime))
}

fn build_cuda_graphs(lib_dir: &std::path::Path) {
    let Some(cuda) = cuda_include(lib_dir) else {
        println!("cargo:warning=CUDA headers not found: building without CUDA graphs");
        return;
    };
    let torch_include = lib_dir.join("../include");
    let abi = env::var("LIBTORCH_CXX11_ABI").unwrap_or_else(|_| "1".to_owned());
    cc::Build::new()
        .cpp(true)
        .file("src/cuda_graph.cpp")
        .include(&torch_include)
        .include(torch_include.join("torch/csrc/api/include"))
        .include(&cuda)
        // The wheel's headers lack `crt/` (compiler configuration, the same
        // across CUDA versions): take it from a CUDA toolkit.
        .include("/usr/local/cuda/include")
        .flag("-std=c++20")
        .flag(format!("-D_GLIBCXX_USE_CXX11_ABI={abi}"))
        .flag("-DGLOG_USE_GLOG_EXPORT")
        .warnings(false)
        .compile("rv_cuda_graph");
    for lib in ["torch_cuda", "c10_cuda", "c10"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-cfg=rv_cuda_graphs");
}

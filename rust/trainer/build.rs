//! Link flags for libtorch from the Python environment.
//!
//! - rpath to the libtorch directory, so the binary starts without
//!   `LD_LIBRARY_PATH` (the learner sets it anyway);
//! - on Linux, `libtorch_cuda.so` with `--no-as-needed`: nothing in the
//!   binary references it directly, so the default `--as-needed` would drop
//!   it and libtorch would silently come up without CUDA.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
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
    }
}

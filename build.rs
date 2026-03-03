fn main() {
    #[cfg(feature = "coolprop-static")]
    coolprop_static::build();

    #[cfg(feature = "coolprop-dylib")]
    coolprop_dylib::link();
}

#[cfg(feature = "coolprop-static")]
mod coolprop_static {
    use std::{
        env,
        path::{Path, PathBuf},
        process::Command,
    };

    pub fn build() {
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let source_dir = manifest_dir.join("vendor/CoolProp");
        let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

        // Rebuild if the build script or the CoolProp header changes.
        // The submodule is pinned to a specific commit, so source changes
        // only happen through a deliberate submodule update.
        // There is no need to watch individual source files.
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-changed=vendor/CoolProp/include/CoolPropLib.h");

        if target_arch == "wasm32" {
            build_wasm(&source_dir, &out_dir);
        } else {
            build_native(&source_dir, &out_dir);
        }
    }

    /// Build `CoolProp` as a static library for native targets using the `cmake` crate.
    fn build_native(source_dir: &Path, out_dir: &Path) {
        // CoolProp's CMakeLists.txt forces CMAKE_INSTALL_PREFIX to its own
        // default, so we cannot rely on cmake's install step to put the
        // library where cargo expects it.
        // Set CMAKE_ARCHIVE_OUTPUT_DIRECTORY to a fixed location instead.
        let lib_dir = out_dir.join("lib");

        // -DCOOLPROP_LIB tells CoolPropLib.h to mark the API functions with
        // `extern "C"`, giving us C linkage in the static library.
        // The shared-library cmake target normally adds this flag;
        // the static-library target does not, so we supply it manually.
        cmake::Config::new(source_dir)
            .define("COOLPROP_STATIC_LIBRARY", "ON")
            .define("COOLPROP_SHARED_LIBRARY", "OFF")
            .define("BUILD_TESTING", "OFF")
            .define("CMAKE_BUILD_TYPE", "Release")
            .cxxflag("-DCOOLPROP_LIB")
            .define(
                "CMAKE_ARCHIVE_OUTPUT_DIRECTORY",
                lib_dir.to_str().expect("lib_dir path is valid UTF-8"),
            )
            .build();

        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=static=CoolProp");

        // Link the C++ standard library — CoolProp's internals are in C++.
        let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
        match target_os.as_str() {
            "macos" => println!("cargo:rustc-link-lib=c++"),
            _ => println!("cargo:rustc-link-lib=stdc++"),
        }
    }

    /// Build `CoolProp` as a static library for WASM via Emscripten.
    ///
    /// The `cmake` crate cannot be used here — it injects
    /// `--target=wasm32-unknown-unknown` and `-fno-exceptions`, both
    /// incompatible with Emscripten.
    /// We invoke cmake directly via `std::process::Command`.
    fn build_wasm(source_dir: &Path, out_dir: &Path) {
        let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
        assert!(
            target_os == "emscripten",
            "CoolProp requires Emscripten for WASM builds. \
             Use target `wasm32-unknown-emscripten`, \
             not `wasm32-unknown-unknown`."
        );

        let emscripten_root = discover_emscripten_root();
        let toolchain_file = emscripten_root.join("cmake/Modules/Platform/Emscripten.cmake");

        assert!(
            toolchain_file.exists(),
            "Emscripten toolchain file not found at {}",
            toolchain_file.display()
        );

        let build_dir = out_dir.join("coolprop-wasm-build");
        let lib_dir = out_dir.join("lib");

        std::fs::create_dir_all(&build_dir).expect("failed to create WASM build directory");

        // CoolProp CXX flags:
        // - `-fwasm-exceptions`: Align with Rust's emscripten runtime which
        //   uses the WASM EH proposal.
        //   Without this, CoolProp uses legacy sjlj EH and exceptions escape
        //   past `catch(...)`.
        // - `-DCOOLPROP_NO_INCBIN`: Preprocessor define that disables the
        //   `incbin` assembly directive — WASM assembler rejects GAS syntax.
        // - `-DCOOLPROP_LIB`: C linkage for the API functions (same as native).
        let cxx_flags = "-fwasm-exceptions -DCOOLPROP_NO_INCBIN -DCOOLPROP_LIB";
        let c_flags = "-DCOOLPROP_NO_INCBIN";

        // Configure.
        let configure_status = Command::new("cmake")
            .current_dir(&build_dir)
            .args([
                "-S",
                source_dir.to_str().expect("source_dir path is valid UTF-8"),
                "-B",
                ".",
                &format!(
                    "-DCMAKE_TOOLCHAIN_FILE={}",
                    toolchain_file
                        .to_str()
                        .expect("toolchain path is valid UTF-8")
                ),
                "-DCOOLPROP_STATIC_LIBRARY=ON",
                "-DCOOLPROP_SHARED_LIBRARY=OFF",
                "-DBUILD_TESTING=OFF",
                "-DCMAKE_BUILD_TYPE=Release",
                &format!("-DCMAKE_CXX_FLAGS={cxx_flags}"),
                &format!("-DCMAKE_C_FLAGS={c_flags}"),
                &format!(
                    "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={}",
                    lib_dir.to_str().expect("lib_dir path is valid UTF-8")
                ),
            ])
            .status()
            .expect("failed to run cmake configure for WASM");

        assert!(
            configure_status.success(),
            "cmake configure failed for WASM CoolProp build"
        );

        // Build.
        let build_status = Command::new("cmake")
            .current_dir(&build_dir)
            .args(["--build", ".", "--config", "Release"])
            .status()
            .expect("failed to run cmake build for WASM");

        assert!(
            build_status.success(),
            "cmake build failed for WASM CoolProp build"
        );

        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=static=CoolProp");

        // No explicit C++ stdlib link needed — Emscripten's toolchain links
        // its own C++ runtime automatically during final linking.

        // CoolProp decompresses ~25 MB of fluid data at init, exceeding the
        // default 16 MB WASM heap.
        println!("cargo:rustc-link-arg=-sALLOW_MEMORY_GROWTH=1");
    }

    /// Discover the Emscripten root directory via `em-config`.
    fn discover_emscripten_root() -> PathBuf {
        let output = Command::new("em-config")
            .arg("EMSCRIPTEN_ROOT")
            .output()
            .expect(
                "failed to run `em-config`. \
                 Is the Emscripten SDK installed and `em-config` on PATH?",
            );

        assert!(
            output.status.success(),
            "`em-config EMSCRIPTEN_ROOT` failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let root = String::from_utf8(output.stdout).expect("`em-config` output is not valid UTF-8");

        PathBuf::from(root.trim())
    }
}

#[cfg(feature = "coolprop-dylib")]
mod coolprop_dylib {
    /// Tell the linker to link against the prebuilt CoolProp shared library.
    ///
    /// The library search path is already provided by the platform-specific
    /// `coolprop-sys-*` dependency crate. We only need to emit the link
    /// directive so the symbols from our `extern "C"` declarations resolve.
    pub fn link() {
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rustc-link-lib=dylib=CoolProp");
    }
}

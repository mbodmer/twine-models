fn main() {
    #[cfg(feature = "coolprop-static")]
    coolprop_static::build();
}

#[cfg(feature = "coolprop-static")]
mod coolprop_static {
    use std::{env, path::PathBuf};

    pub fn build() {
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let source_dir = manifest_dir.join("vendor/CoolProp");

        // Rebuild if the build script or the CoolProp header changes.
        // The submodule is pinned to a specific commit, so source changes
        // only happen through a deliberate submodule update. Because of this,
        // there is no need to watch individual source files.
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-changed=vendor/CoolProp/include/CoolPropLib.h");

        // CoolProp's CMakeLists.txt forces CMAKE_INSTALL_PREFIX to its own
        // default, so we cannot rely on cmake's install step to put the
        // library where cargo expects it. Instead, set
        // CMAKE_ARCHIVE_OUTPUT_DIRECTORY to a fixed location and tell cargo
        // to search there.
        let lib_dir = out_dir.join("lib");

        // -DCOOLPROP_LIB tells CoolPropLib.h to mark the API functions with
        // `extern "C"`, giving us C linkage in the static library. The
        // shared-library cmake target normally adds this flag; the
        // static-library target does not, so we supply it manually here.
        cmake::Config::new(&source_dir)
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
}

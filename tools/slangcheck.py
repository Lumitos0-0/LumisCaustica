#!/usr/bin/env python3
"""Headless stand-in for the Vulkan SDK's slangc, for compile-checking shaders/**.

The Gradle `compileShaders` task shells out to a real slangc, which needs the Vulkan SDK.
This drives the same compiler through Slang's exported C ABI (`spProcessCommandLineArguments`
+ `spCompile`) from whatever libslang happens to be on disk, so shader edits can be
type-checked in a container with no SDK and no GPU.

It passes the exact argument list build.gradle uses, minus `-g` (debug info needs the
source on disk in the same layout, which it is, but the blob is large and unused here)
and minus spirv-val, which is a separate binary.

  SLANG_LIB=/path/to/libslang.so python3 tools/slangcheck.py [file-or-dir ...]

With no arguments it checks every stage entry point under shaders/.
"""
import ctypes
import glob
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHADERS = ROOT / "shaders"
STAGES = ("rgen", "rchit", "rahit", "rmiss", "rcall", "comp", "vert", "frag")

# Variants build.gradle compiles a second time under extra flags. Keep in lock-step with
# CompileShaders.compile()'s worldIndirectRaygen branch.
EXTRA_VARIANTS = {
    "pipelines/world/indirect.rgen.slang": [
        ["-DCAUSTICA_ENABLE_EXT_SER", "-capability", "spvShaderInvocationReorderEXT"],
    ],
}


def find_lib():
    explicit = os.environ.get("SLANG_LIB")
    if explicit:
        return explicit
    for pattern in (
        "/tmp/tc/py/slangpy/libslang-compiler.so*",
        "/tmp/tc/slang/lib/libslang.so",
        os.path.expanduser("~/.local/lib/libslang.so"),
    ):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    raise SystemExit("no libslang found; set SLANG_LIB")


def load(lib_path):
    # RTLD_GLOBAL so the core-module and glslang side libraries resolve against it.
    lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    lib.slang_createGlobalSession.argtypes = [ctypes.c_int64, ctypes.POINTER(ctypes.c_void_p)]
    lib.slang_createGlobalSession.restype = ctypes.c_int32
    lib.spCreateCompileRequest.argtypes = [ctypes.c_void_p]
    lib.spCreateCompileRequest.restype = ctypes.c_void_p
    lib.spProcessCommandLineArguments.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int]
    lib.spProcessCommandLineArguments.restype = ctypes.c_int32
    lib.spCompile.argtypes = [ctypes.c_void_p]
    lib.spCompile.restype = ctypes.c_int32
    lib.spGetDiagnosticOutput.argtypes = [ctypes.c_void_p]
    lib.spGetDiagnosticOutput.restype = ctypes.c_char_p
    lib.spDestroyCompileRequest.argtypes = [ctypes.c_void_p]
    lib.spDestroyCompileRequest.restype = None
    return lib


def stage_sources(targets):
    def is_stage(f):
        return len(f.suffixes) >= 2 and f.suffixes[-2][1:] in STAGES

    if not targets:
        return [f for f in sorted(SHADERS.rglob("*.slang")) if is_stage(f)]
    out = []
    for t in targets:
        p = pathlib.Path(t)
        if p.is_dir():
            out += [f for f in sorted(p.rglob("*.slang")) if is_stage(f)]
        else:
            out.append(p)
    return out


def main():
    lib = load(find_lib())
    session = ctypes.c_void_p()
    if lib.slang_createGlobalSession(0, ctypes.byref(session)) < 0:
        raise SystemExit("slang_createGlobalSession failed")

    include_dirs = sorted({str(p.parent) for p in SHADERS.rglob("*.slang")})
    out_dir = pathlib.Path(os.environ.get("SLANGCHECK_OUT", "/tmp/slangcheck"))
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for src in stage_sources(sys.argv[1:]):
        rel = src.resolve().relative_to(SHADERS).as_posix()
        variants = [[]] + EXTRA_VARIANTS.get(rel, [])
        for index, extra in enumerate(variants):
            includes = [str(src.resolve().parent)] + [d for d in include_dirs
                                                      if d != str(src.resolve().parent)]
            args = [str(src.resolve()), "-target", "spirv", "-profile", "spirv_1_5",
                    "-matrix-layout-column-major",
                    "-warnings-as-errors", "all", "-warnings-disable", "41012"]
            for d in includes:
                args += ["-I", d]
            args += extra
            if os.environ.get("SLANGCHECK_REFLECT"):
                args += ["-reflection-json", os.environ["SLANGCHECK_REFLECT"]]
            args += ["-o", str(out_dir / (rel.replace("/", "_") + f".{index}.spv"))]

            argv = (ctypes.c_char_p * len(args))(*[a.encode() for a in args])
            req = lib.spCreateCompileRequest(session)
            result = lib.spProcessCommandLineArguments(req, argv, len(args))
            if result >= 0:
                result = lib.spCompile(req)
            diagnostics = (lib.spGetDiagnosticOutput(req) or b"").decode(errors="replace")
            lib.spDestroyCompileRequest(req)

            label = rel if not extra else f"{rel} [{' '.join(extra)}]"
            if result < 0:
                failures += 1
                print(f"FAIL  {label}\n{diagnostics.rstrip()}\n")
            else:
                print(f"ok    {label}")
                if diagnostics.strip():
                    print(diagnostics.rstrip())

    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

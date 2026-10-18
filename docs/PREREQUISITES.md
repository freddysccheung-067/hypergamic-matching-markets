# Prerequisites

Verified against this machine on 2026-10-18.

## What is already installed

| Tool | Version | Path |
|------|---------|------|
| CMake | 4.4.4 | on `PATH` |
| Python | 3.12.10 | `C:\Users\Freddy\AppData\Local\Programs\Python\Python312` |
| pip | 25.0.1 | — |
| git | 2.55.0 | on `PATH` |
| pybind11 | present | `...\Python312\Lib\site-packages\pybind11` |

## What is missing — install this before Day 1

**No C++ compiler is installed.** `cl`, `g++`, and `clang++` are all absent, and there is no
Visual Studio installer and no `C:\msys64`. Nothing in `src/cpp/` can compile until this is fixed.

### Option A — MSVC Build Tools (recommended)

The path of least resistance on Windows: it matches the python.org CPython build already on this
machine, needs no `PATH` surgery, and OpenMP support is built in.

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools --source winget `
  --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --includeRecommended"
```

Then **open a new terminal** — the environment only refreshes on shell start. Verify:

```powershell
& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property displayName
cmake --version
```

You do not need `cl.exe` on `PATH`. CMake locates the toolchain itself once VS is installed;
just configure with no `-G` flag and CMake picks the Visual Studio generator.

### Option B — MSYS2 / MinGW-w64

Better OpenMP conformance (MSVC only implements OpenMP 2.0, so `collapse` and task directives are
unavailable), at the cost of more setup and a non-native Python ABI risk.

```powershell
winget install --id MSYS2.MSYS2 --source winget
```

Then in an MSYS2 **UCRT64** shell: `pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake`,
and configure with `-G Ninja` from that shell.

Pick A unless you specifically need OpenMP 4+ constructs. The plan below assumes A.

## Configure and build

Multi-config generators put the extension in a per-configuration subdirectory, which is the usual
reason a "successful" build still throws `ModuleNotFoundError`.

```powershell
# from the project root
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel

# confirm the module actually exists
Get-ChildItem -Recurse build -Filter "_hypergamic*" | Select-Object FullName
```

If the file is under `build\Release\`, that directory is what goes on `PYTHONPATH`.

## Verify the toolchain end to end

```powershell
python -c "import sys; sys.path.insert(0, 'build/Release'); import _hypergamic; print('ok')"
```

Run this before writing any simulation code. A green smoke test here means Days 1–2 are about the
model, not about the build.

## Version pins, and why

`requirements.txt` pins `torch==2.3.1` and `numpy==1.26.4`. Both publish `cp312` Windows wheels, so
they install cleanly on Python 3.12. If you later move to Python 3.13, `numpy` must go to 2.x and
`torch` to 2.5+, and the pin set needs revisiting — do not assume it carries over.

## OpenMP note

`find_package(OpenMP QUIET)` — the build succeeds without it and the epoch loop runs serially.
Everything is still correct, just slower. Do not let a missing OpenMP stall Day 1; check with
`cmake --build build --config Release` output for `OpenMP_CXX_FOUND`.

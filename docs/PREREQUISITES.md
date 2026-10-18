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
& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property displayName
cmake --version
```

Note `-products *`, not `-product *`. The singular form is not a valid switch and vswhere
answers with `Error 0x57: Unknown parameter: product`.

You do not need `cl.exe` on `PATH`. CMake locates the toolchain itself once VS is installed;
just configure with no `-G` flag and CMake picks the Visual Studio generator.

A successful configure prints the toolchain it chose:

```
-- Building for: Visual Studio 17 2022
-- The CXX compiler identification is MSVC 19.44.35229.0
-- Found Python3: ...\python.exe (found version "3.12.10")
-- Found pybind11: ... (found version "2.12.0")
```

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

### Use a virtual environment — do not install into the global interpreter

This machine runs a **shared global Python 3.12** that other projects depend on (the global
site-packages contains `open-webui`, `kubernetes`, and `opencv-python-headless`). Installing
`requirements.txt` globally **downgrades** their pinned dependencies:

| Package | Global version | requirements.txt pin |
|---------|---------------|----------------------|
| numpy | 2.5.4 | 1.26.4 |
| pandas | 3.0.3 | 2.2.2 |
| torch | 2.14.1 | 2.3.1 |
| scipy | 1.18.1 | 1.13.1 |
| pyyaml | 6.0.3 | 6.0.1 |
| pybind11 | 3.1.0 | 2.12.0 |
| pytest | 9.1.1 | 8.2.2 |
| networkx | 3.7 | 3.3 |
| tqdm | 4.70.1 | 4.66.4 |

A global install breaks `open-webui` (it pins `pandas==3.0.3`) and `opencv-python-headless`
(it requires `numpy>=2`). Always work inside a venv.

```powershell
# from the project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `Activate.ps1` is blocked, allow scripts for the current user once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Point CMake at the venv interpreter

CMake caches the Python it found at configure time. If `build/` was configured against the
global interpreter, **delete it** and reconfigure from an activated venv, otherwise the
extension is built against a different Python than the one that imports it:

```powershell
Remove-Item -Recurse -Force build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

Confirm the configure output names `.venv`:

```
-- Found Python3: D:/hypergamic-matching-markets/.venv/Scripts/python.exe ...
```

If it still names the global Python, pass it explicitly:

```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$PWD\.venv\Scripts\python.exe"
```

Multi-config generators put the extension in a per-configuration subdirectory, which is the usual
reason a "successful" build still throws `ModuleNotFoundError`.

```powershell
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

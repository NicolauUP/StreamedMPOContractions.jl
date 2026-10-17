# StreamedMPOContractions.jl

Memory-efficient multiplication of dense ITensorMPS MPOs on CPU or NVIDIA GPU.
The package provides one public function: `streamed_product`.

## Install from GitHub

In your application's Julia environment:

```julia
using Pkg
Pkg.add(url="https://github.com/NicolauUP/StreamedMPOContractions.jl", rev="v0.1.0")
```

Then:

```julia
using StreamedMPOContractions
C = streamed_product(A, B; maxdim=768, cutoff=1e-12)
```

The version tag fixes the source revision used by your project. Julia records
that dependency in the application's Project.toml and Manifest.toml.
For development, follow the local installation below instead.

## Install locally

In the Julia environment where you want to use it, run once:

```julia
using Pkg
Pkg.develop(path=joinpath(homedir(), "Documents", "StreamedMPOContractions.jl"))
```

Then use it normally:

```julia
using StreamedMPOContractions

C = streamed_product(A, B; maxdim=768, cutoff=1e-12)
X2 = streamed_product(X, X; maxdim=768, cutoff=1e-12, block_right=16)
```

`A`, `B`, and `X` are ITensorMPS `MPO` objects with matching physical indices.
Inputs remain unchanged. The result uses the standard physical prime levels
0 and 1, like `ITensorMPS.apply(A, B)`. No library methods are replaced.

This is a local, unregistered package. `Pkg.add("StreamedMPOContractions")` from
the General registry is not available yet.

## CUDA

Add CUDA to your application's environment, then load it alongside this package:

```julia
using CUDA, ITensors, ITensorMPS, StreamedMPOContractions
CUDA.allowscalar(false)

Agpu = ITensors.adapt(CuArray, A)
Bgpu = ITensors.adapt(CuArray, B)
Cgpu = streamed_product(Agpu, Bgpu; maxdim=768, cutoff=1e-12)
```

The extension loads automatically when CUDA and the package are both loaded.
Inputs and output stay on the GPU. CPU users do not need CUDA. The CUDA
extension retires only owned temporary buffers; caller tensors are borrowed.
No full contraction tensors are transferred to the host. Singular-value
truncation follows NDTensors' policy. There are no added synchronization barriers
or forced garbage collections inside the contraction.

## Options

| Keyword | Meaning |
|---|---|
| `maxdim` | Retained bond cap; defaults to the product of input maximum bonds. |
| `cutoff=1e-14` | NDTensors squared-singular-value truncation cutoff. |
| `mindim=1` | Minimum retained bond dimension. |
| `block_right=16` | Number of outgoing right-factor bond states per tile. |
| `truncate_kwargs=(;)` | Overrides for the final ITensorMPS `truncate!`. |
| `observer=nothing` | Optional callback receiving scalar metadata per streamed bond. |

Smaller tiles reduce scratch space and may take longer. A hard bond cap can
prevent reaching the requested cutoff. The cutoff is not a global error bound.
An observer is optional; the package stores no global diagnostic history.

## Method and scope

Instead of forming the whole local product matrix `M`, the first pass generates
column tiles and reduces `M'` by QR, merging the small triangular factors. An
SVD of the final small factor determines the retained left singular subspace of
`M`. A second pass regenerates the tiles and computes `U' * M`. Tile and QR
buffers are reused. The last two tensors are split and the MPO is rounded with
ITensorMPS. This is a local SVD-equivalent reduction up to floating-point error,
not a globally optimal variational fit.

The full local product matrix is avoided, but the compressed carry still has
size proportional to the output bond times both input bonds. Streaming trades
extra contraction/QR work for lower temporary memory; it is not promised to be
faster than zipup.

Supported: open-chain dense MPOs, one virtual bond between neighboring tensors,
paired input/output physical indices at prime levels 0 and 1, matching index
identities, a consistent scalar type across both inputs, and Float32/Float64
or their complex counterparts. CUDA tests cover
Float64 and ComplexF64. QN/block-sparse MPOs and other GPU backends are outside
the current scope. Compatibility currently targets Julia 1.12, ITensors 0.9,
ITensorMPS 0.3, NDTensors 0.4, and optional CUDA 6.2.

The package contains the streamed contraction and its optional CUDA backend;
it has no Hartree–Fock, SP2 solver, checkpoint, or cluster dependencies.

## Test

```julia
using Pkg
Pkg.test("StreamedMPOContractions")
```

Tests compare with independent dense multiplication and SVD, native zipup,
real/complex inputs, small and ragged tiles, truncation, input immutability,
output lifetime, and invalid inputs. `examples/basic.jl` is a complete CPU example.
For an otherwise empty application environment, also add `ITensors` and
`ITensorMPS` with `Pkg.add(["ITensors", "ITensorMPS"])` to run that example.

On an NVIDIA machine, in an environment with CUDA and this package installed
(also add ITensors, ITensorMPS, and NDTensors as direct dependencies for the
standalone test script):

```julia
using StreamedMPOContractions
include(joinpath(pkgdir(StreamedMPOContractions), "test", "runtests_gpu.jl"))
```

The CUDA test suite requires a functional GPU and fails explicitly otherwise.

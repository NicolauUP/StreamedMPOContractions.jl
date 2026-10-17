# Optional: run this file in an environment containing this package and CUDA.
using CUDA
CUDA.functional() || error("GPU tests require a functional NVIDIA GPU")
CUDA.allowscalar(false)
include("common.jl")
BLAS.set_num_threads(1)

@testset "CUDA extension" begin
    @test !isnothing(Base.get_extension(StreamedMPOContractions,:StreamedMPOContractionsCUDAExt))
    @test SMC._backend(CUDA.zeros(1))==:cuda
end
test_factorization(CuArray,Array)
test_products(A->ITensors.adapt(CuArray,A),ITensors.cpu)

@testset "Output remains on GPU" begin
    sites=siteinds("Qubit",5)
    A=ITensors.adapt(CuArray,represent(randn(MersenneTwister(887),32,32),sites))
    C=streamed_product(A,A;maxdim=8,cutoff=1e-12)
    @test all(i->ITensors.array(C[i]) isa CuArray,eachindex(C))
    @test_throws ArgumentError streamed_product(A,ITensors.cpu(A))
end
CUDA.synchronize()

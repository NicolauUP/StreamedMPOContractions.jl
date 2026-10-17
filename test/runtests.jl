include("common.jl")
BLAS.set_num_threads(1)

@testset "CPU-only package loading" begin
    @test isnothing(Base.get_extension(StreamedMPOContractions,:StreamedMPOContractionsCUDAExt))
end
test_factorization(copy,identity)
test_products(deepcopy,identity)
test_products(deepcopy,identity;types=(Float32,ComplexF32),tolerance=8e-5)

@testset "Argument validation" begin
    s=siteinds("Qubit",3);A=represent(Matrix{Float64}(I,8,8),s)
    @test_throws ArgumentError streamed_product(A,A;block_right=0)
    @test_throws ArgumentError streamed_product(A,A;cutoff=-1)
    @test_throws ArgumentError streamed_product(A,A;cutoff=NaN)
    @test_throws ArgumentError streamed_product(A,A;maxdim=0)
    @test_throws ArgumentError streamed_product(A,A;maxdim=2,mindim=3)
    @test_throws ArgumentError streamed_product(A,A;truncate_kwargs=(;cutoff=-1))
    B=represent(Matrix{Float64}(I,4,4),s[1:2])
    @test_throws DimensionMismatch streamed_product(A,B)
    other=represent(Matrix{Float64}(I,8,8),siteinds("Qubit",3))
    @test_throws ArgumentError streamed_product(A,other)
    @test_throws ArgumentError streamed_product(prime(A),prime(A))
    complex_A=represent(Matrix{ComplexF64}(I,8,8),s)
    @test_throws ArgumentError streamed_product(A,complex_A)
    qsites=siteinds("S=1/2",3;conserve_qns=true)
    Q=MPO(qsites,"Id")
    @test_throws ArgumentError streamed_product(Q,Q)
end

@testset "Zero operator" begin
    s=siteinds("Qubit",4);Z=represent(zeros(16,16),s)
    C=streamed_product(Z,Z;maxdim=4,cutoff=1e-12)
    @test norm(dense_matrix(C,s,identity))==0
end

using StreamedMPOContractions, ITensors, ITensorMPS, LinearAlgebra, Random

sites=siteinds("Qubit",4)
rng=MersenneTwister(17)
M=randn(rng,16,16)
A=MPO(ITensor(reshape(M,ntuple(_->2,8)),prime.(sites)...,sites...),sites;cutoff=0.0)
C=streamed_product(A,A;maxdim=256,cutoff=1e-12,block_right=3)
represented=reshape(Array(reduce(*,C),prime.(sites)...,sites...),16,16)
println("Relative error against dense A²: ",norm(represented-M*M)/norm(M*M))

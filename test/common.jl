using StreamedMPOContractions, ITensors, ITensorMPS, LinearAlgebra, Random, Test
import NDTensors
const SMC=StreamedMPOContractions

function represent(M,sites)
    dims=Tuple(dim.(vcat(sites,sites)))
    MPO(ITensor(reshape(M,dims),prime.(sites)...,sites...),sites;cutoff=0.0)
end
function dense_matrix(A,sites,to_host)
    n=prod(dim,sites)
    reshape(Array(reduce(*,to_host(A)),prime.(sites)...,sites...),n,n)
end

function test_products(to_device,to_host;types=(Float64,ComplexF64),tolerance=3e-11)
    rng=MersenneTwister(20261017)
    @testset "Operator products, compression, and input/output lifetime" begin
        for T in types,N in (1,2,3,5)
            sites=siteinds("Qubit",N);n=2^N
            matrices=[randn(rng,T,n,n) for _ in 1:2]
            A,B=to_device.(represent.(matrices,Ref(sites)))
            exact=matrices[1]*matrices[2];scale=norm(exact)
            native=apply(A,B;maxdim=3,cutoff=1e-12)
            for cap in (3,n^2),width in (1,3)
                records=Dict{String,Any}[]
                C=streamed_product(A,B;maxdim=cap,cutoff=1e-12,block_right=width,observer=x->push!(records,x))
                represented=dense_matrix(C,sites,to_host)
                @test all(isfinite,represented)
                @test maxlinkdim(C)<=cap
                @test all(i->isapprox(dense_matrix((A,B)[i],sites,to_host),matrices[i];rtol=tolerance),1:2)
                @test issetequal(siteinds(C,1),siteinds(A,1))
                @test length(records)==max(N-2,0)
                if cap==n^2
                    @test norm(represented-exact)/scale<tolerance
                else
                    @test norm(represented-dense_matrix(native,sites,to_host))/scale<tolerance
                end
                # A later contraction must not invalidate the first output.
                second=streamed_product(A,B;maxdim=cap,cutoff=1e-12,block_right=width+1)
                GC.gc()
                @test dense_matrix(C,sites,to_host)==represented
                @test norm(dense_matrix(second,sites,to_host)-represented)/scale<tolerance
            end
        end
    end
    @testset "Complex Hermitian square and nonbinary physical dimension" begin
        sites=siteinds("Qubit",4);n=16
        Q=Matrix(qr(randn(rng,ComplexF64,n,n)).Q)
        X=Q*Diagonal(range(.05,.95;length=n))*Q'
        A=to_device(represent(X,sites));C=streamed_product(A,A;maxdim=256,cutoff=1e-12)
        @test norm(dense_matrix(C,sites,to_host)-X*X)/norm(X*X)<tolerance
        @test norm(dense_matrix(A,sites,to_host)-X)/norm(X)<tolerance
        sites=[Index(3,"Site,n=$i") for i in 1:3]
        M=randn(rng,27,27);A=to_device(represent(M,sites))
        @test norm(dense_matrix(streamed_product(A,A;cutoff=0.0),sites,to_host)-M*M)/norm(M*M)<tolerance
    end
end

function test_factorization(to_array,to_host_array;types=(Float64,ComplexF64),tolerance=3e-11)
    rng=MersenneTwister(7717)
    @testset "Streamed factorization versus independent dense SVD" begin
        for T in types,(m,n) in ((8,37),(12,49),(8,5)),width in (5,16),kind in (:random,:ill_conditioned,:zero)
            A=randn(rng,T,m,n);k=min(m,n)
            if kind!=:random
                L=Matrix(qr(randn(rng,T,m,k)).Q)[:,1:k]
                V=Matrix(qr(randn(rng,T,n,k)).Q)[:,1:k]
                vals=kind==:zero ? zeros(k) : 10.0.^range(0,-12;length=k)
                A=L*Diagonal(vals)*V'
            end
            source=to_array(A);record=Dict{String,Any}()
            choose=S->begin
                P,err,_=NDTensors.truncate!!(S.^2;maxdim=3,cutoff=1.7e-8)
                P,err
            end
            U,C=SMC._project_tiles(J->copy(view(source,:,J)),m,n;
                block_columns=width,select=choose,event=record)
            F=svd(A);P,err=choose(to_array(F.S));r=length(P)
            reference=F.U[:,1:r]*Diagonal(F.S[1:r])*F.V[:,1:r]'
            result=to_host_array(U*C);scale=max(norm(A),1)
            @test size(U,2)==r
            @test norm(result-reference)/scale<tolerance
            @test abs(norm(result-A)-norm(F.S[r+1:end]))/scale<tolerance
            @test to_host_array(source)==A
            @test record["tile_calls"]==2cld(n,width)
            @test record["maximum_tile_bytes"]<=m*min(n,width)*sizeof(T)
            @test abs(record["truncation_error"]-err)<tolerance
        end
    end
    @testset "Borrowed tiles and ragged panels" begin
        for T in types
            A=randn(rng,T,19,83);source=to_array(A)
            buffers=Dict(n=>to_array(zeros(T,19,n)) for n in (16,3))
            tile=J->begin
                buffer=buffers[length(J)];copyto!(buffer,view(source,:,J));buffer
            end
            select=S->begin
                P,err,_=NDTensors.truncate!!(S.^2;maxdim=7,cutoff=1e-12)
                P,err
            end
            U,C=SMC._project_tiles(tile,19,83;block_columns=16,select,owned_tiles=false)
            F=svd(A);expected=F.U[:,1:7]*Diagonal(F.S[1:7])*F.V[:,1:7]'
            before=to_host_array(U*C)
            for buffer in values(buffers);fill!(buffer,T(991));end
            @test norm(to_host_array(U*C)-expected)/norm(A)<tolerance
            @test to_host_array(U*C)==before
            @test to_host_array(source)==A
        end
    end
end

# CPU needs no explicit retirement. A separate backend file supplies CUDA methods.
_release_owned!(x)=nothing
_geqrf_owned!(A,tau)=LinearAlgebra.LAPACK.geqrf!(A,tau)
_small_svd(R)=svd!(R;alg=LinearAlgebra.QRIteration())
_backend(::Array)=:cpu
_backend(A)=throw(ArgumentError("unsupported array backend $(typeof(A)); use dense CPU arrays or CUDA.CuArray with CUDA loaded"))

mutable struct _Scratch
    arrays::Vector{Any}
end
_Scratch()=_Scratch(Any[])
_hold!(s::_Scratch,x)=(push!(s.arrays,x);x)
_shape!(s,x,dims...)=begin
    y=reshape(x,dims...)
    y===x ? y : _hold!(s,y)
end
_alias!(s,x,parent)=x===parent ? x : _hold!(s,x)
function _close!(s::_Scratch)
    # Derived CuArrays own reference-counted aliases. Release these as well as
    # roots, in reverse order, so stale reshapes cannot pin a large allocation.
    for x in Iterators.reverse(s.arrays)
        _release_owned!(x)
    end
    empty!(s.arrays)
    return nothing
end

function _core_array(X,i,row,col)
    left=i==1 ? Index[] : [linkind(X,i-1)]
    right=i==length(X) ? Index[] : [linkind(X,i)]
    data=ITensors.array(X[i],left...,row,col,right...)
    return reshape(data,isempty(left) ? 1 : dim(only(left)),dim(row),dim(col),
        isempty(right) ? 1 : dim(only(right)))
end

"""One fixed-size set of tile buffers. Only full and ragged widths are needed."""
function _tile_buffers!(s,G,B,d,ap,w)
    g,a,b=size(G);_,u,t,_=size(B)
    T=promote_type(eltype(G),eltype(B))
    alloc(dims...)=_hold!(s,similar(G,T,dims...))
    panel=alloc(b,u,t,w)
    D=alloc(g,a,u,t,w);Dp=alloc(g,t,w,a,u)
    E=alloc(g,t,w,d,ap);M=alloc(g,d,t,ap,w)
    return (;panel,D,Dp,E,M,
        panelmat=_shape!(s,panel,b,u*t*w),
        Dmat=_shape!(s,D,g*a,u*t*w),Dpmat=_shape!(s,Dp,g*t*w,a*u),
        Emat=_shape!(s,E,g*t*w,d*ap),Mmat=_shape!(s,M,g*d*t,ap*w))
end

function _product_tile!(buffers,Gmat,Amat,B,ap,J)
    (first(J)-1)%ap==0 && last(J)%ap==0 || error("unaligned outgoing-bond tile")
    K=cld(first(J),ap):div(last(J),ap)
    panel_view=view(B,:,:,:,K)
    try
        copyto!(buffers.panel,panel_view)
    finally
        panel_view===B || _release_owned!(panel_view)
    end
    mul!(buffers.Dmat,Gmat,buffers.panelmat)
    permutedims!(buffers.Dp,buffers.D,(1,4,5,2,3))
    mul!(buffers.Emat,buffers.Dpmat,Amat)
    permutedims!(buffers.M,buffers.E,(1,4,2,5,3))
    return buffers.Mmat  # borrowed until the next use of these buffers
end

function _qr_buffers!(s,prototype,height,columns)
    packed=_hold!(s,similar(prototype,height,columns))
    tau=_hold!(s,similar(prototype,min(height,columns)))
    R=_hold!(s,similar(prototype,min(height,columns),columns))
    return (;packed,tau,R)
end
function _triangularize!(s,buffers)
    _geqrf_owned!(buffers.packed,buffers.tau)
    source=_alias!(s,view(buffers.packed,1:size(buffers.R,1),:),buffers.packed)
    copyto!(buffers.R,source)
    triu!(buffers.R)
    return buffers.R
end

"""Two-pass reduction. Owned tile callbacks return fresh arrays by default;
production tile storage is borrowed and reused, hence owned_tiles=false there.
"""
function _project_tiles(tile,rows,columns;block_columns,select,
    small_svd=_small_svd,
    owned_tiles=true,event=nothing)
    block_columns>0 || error("positive tile width required")
    rows>0 && columns>0 || error("nonempty local matrix required")
    scratch=_Scratch();local_buffers=Dict{Int,Any}();merge_buffers=Dict{Int,Any}()
    Rstore=nothing;Rrows=0;max_tile=0;calls=0
    U=nothing;C=nothing;L=nothing;sigma=nothing;V=nothing;P=nothing
    try
        for first in 1:block_columns:columns
            J=first:min(columns,first+block_columns-1)
            M=tile(J);calls+=1
            size(M)==(rows,length(J)) || error("incorrect product tile shape")
            max_tile=max(max_tile,length(M))
            if isnothing(Rstore)
                Rstore=_hold!(scratch,similar(M,rows,rows))
            end
            local_qr=get!(local_buffers,length(J)) do
                _qr_buffers!(scratch,M,length(J),rows)
            end
            try
                copyto!(local_qr.packed,adjoint(M))
            finally
                owned_tiles && _release_owned!(M)
            end
            local_R=_triangularize!(scratch,local_qr)
            local_rows=size(local_R,1)
            if Rrows==0
                Rrows=local_rows
                target=_alias!(scratch,view(Rstore,1:Rrows,:),Rstore)
                copyto!(target,local_R)
            else
                height=Rrows+local_rows
                merge_qr=get!(merge_buffers,height) do
                    _qr_buffers!(scratch,Rstore,height,rows)
                end
                old=_alias!(scratch,view(Rstore,1:Rrows,:),Rstore)
                top=_alias!(scratch,view(merge_qr.packed,1:Rrows,:),merge_qr.packed)
                bottom=_alias!(scratch,view(merge_qr.packed,Rrows+1:height,:),merge_qr.packed)
                copyto!(top,old);copyto!(bottom,local_R)
                merged_R=_triangularize!(scratch,merge_qr)
                Rrows=min(height,rows)
                target=_alias!(scratch,view(Rstore,1:Rrows,:),Rstore)
                copyto!(target,merged_R)
            end
        end
        # The SVD may overwrite its input; never overwrite the reusable R store.
        Rsmall=_hold!(scratch,similar(Rstore,Rrows,rows))
        source=_alias!(scratch,view(Rstore,1:Rrows,:),Rstore)
        copyto!(Rsmall,source)
        L,sigma,V=small_svd(Rsmall)
        P,truncerr=select(sigma);r=length(P);full_rank=length(sigma)
        sourceU=view(V,:,1:r)
        try
            U=copy(sourceU)
        finally
            sourceU===V || _release_owned!(sourceU)
        end
        # U and C own storage outside scratch and are the only returned arrays.
        C=similar(U,r,columns)
        _release_owned!(L);L=nothing
        _release_owned!(sigma);sigma=nothing
        _release_owned!(V);V=nothing
        _release_owned!(P);P=nothing
        # Packed QR/merge storage is not needed by projection.
        _close!(scratch)
        for first in 1:block_columns:columns
            J=first:min(columns,first+block_columns-1)
            M=tile(J);calls+=1
            target=view(C,:,J)
            try
                mul!(target,adjoint(U),M)
            finally
                target===C || _release_owned!(target)
                owned_tiles && _release_owned!(M)
            end
        end
        isnothing(event) || merge!(event,Dict("rows"=>rows,"columns"=>columns,"retained_rank"=>r,
            "full_rank"=>full_rank,"tile_calls"=>calls,
            "maximum_tile_bytes"=>max_tile*sizeof(eltype(U)),
            "full_matrix_bytes_avoided"=>rows*columns*sizeof(eltype(U)),
            "compressed_remainder_bytes"=>r*columns*sizeof(eltype(U)),
            "small_factor_rows"=>Rrows,"small_factor_columns"=>rows,
            "truncation_error"=>truncerr,"element_type"=>string(eltype(U)),
            "buffer_reuse"=>true,"distinct_tile_qr_shapes"=>length(local_buffers),
            "distinct_merge_qr_shapes"=>length(merge_buffers)))
        return U,C
    catch
        _release_owned!(U);_release_owned!(C)
        rethrow()
    finally
        _release_owned!(L);_release_owned!(sigma);_release_owned!(V);_release_owned!(P)
        _close!(scratch)
    end
end

"""Dense single-bond MPO product, with reusable tile/QR scratch.
Internal contraction assumes matching intermediate physical indices.
"""
function _contract(A0::MPO,B0::MPO;cutoff=1e-14,
    maxdim=maxlinkdim(A0)*maxlinkdim(B0),mindim=1,
    truncate_kwargs=(;cutoff,maxdim,mindim),block_right=16,
    small_svd=_small_svd,observer=nothing)
    N=length(A0);N==length(B0) || throw(DimensionMismatch("MPO lengths differ"))
    N==1 && return MPO([A0[1]*B0[1]])
    block_right>0 || error("positive outgoing-bond block required")
    A=sim(linkinds,orthogonalize(A0,1));B=orthogonalize(B0,1)
    rows=[only(siteinds(uniqueinds,A,B,i)) for i in 1:N]
    cols=[only(siteinds(uniqueinds,B,A,i)) for i in 1:N]
    common=[only(siteinds(commoninds,A,B,i)) for i in 1:N]
    C=MPO(N)
    prototype=ITensors.array(A[1]);T=promote_type(eltype(A[1]),eltype(B[1]))
    G=similar(prototype,T,1,1,1);fill!(G,one(T))
    previous=nothing
    for i in 1:N-2
        a=_core_array(A,i,rows[i],common[i]);b=_core_array(B,i,common[i],cols[i])
        g,al,bl=size(G);ap=size(a,4);bp=size(b,4);d=dim(rows[i])
        scratch=_Scratch();buffers=Dict{Int,Any}()
        event=isnothing(observer) ? nothing : Dict{String,Any}("bond"=>i,
            "incoming_remainder_bytes"=>length(G)*sizeof(eltype(G)),"block_right"=>block_right)
        choose=S->begin
            P,err,_=NDTensors.truncate!!(S.^2;mindim,maxdim,cutoff)
            (P,err)
        end
        U,remainder=try
            Amat_data=_hold!(scratch,permutedims(a,(1,3,2,4)))
            Amat=_shape!(scratch,Amat_data,al*dim(common[i]),d*ap)
            Gmat=_shape!(scratch,G,g*al,bl)
            tile=J->begin
                w=div(length(J),ap)
                storage=get!(buffers,w) do
                    _tile_buffers!(scratch,G,b,d,ap,w)
                end
                _product_tile!(storage,Gmat,Amat,b,ap,J)
            end
            _project_tiles(tile,g*d*dim(cols[i]),ap*bp;
                block_columns=ap*block_right,select=choose,small_svd,
                owned_tiles=false,event)
        finally
            _close!(scratch)
        end
        isnothing(observer) || observer(event)
        r=size(U,2);link=Index(r,ITensors.commontags(linkinds(A,i)))
        U=reshape(U,g,d,dim(cols[i]),r)
        C[i]=isnothing(previous) ? ITensor(reshape(U,d,dim(cols[i]),r),rows[i],cols[i],link) :
            ITensor(U,previous,rows[i],cols[i],link)
        # The old carry is ours, never caller input. Its borrowed reshape has
        # been retired by _close!(scratch), so its storage can now be released.
        _release_owned!(G)
        G=reshape(remainder,r,ap,bp)
        G===remainder || _release_owned!(remainder)
        previous=link
    end
    R=isnothing(previous) ? ITensor(true) : ITensor(G,previous,linkind(A,N-2),linkind(B,N-2))
    final=R*A[N-1]*B[N-1]*A[N]*B[N]
    left=isnothing(previous) ? [rows[N-1],cols[N-1]] : [rows[N-1],cols[N-1],previous]
    C[N-1],C[N]=factorize(final,left;ortho="right",tags=ITensors.commontags(linkinds(A,N-1)),
        cutoff,maxdim,mindim)
    truncate!(C;truncate_kwargs...)
    return C
end

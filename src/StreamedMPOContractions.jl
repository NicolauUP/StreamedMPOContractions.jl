module StreamedMPOContractions

using ITensors, ITensorMPS, LinearAlgebra
import NDTensors

export streamed_product

include("contraction.jl")

function _validate_inputs(A::MPO,B::MPO)
    length(A)==length(B) || throw(DimensionMismatch("MPO lengths differ"))
    length(A)>0 || throw(ArgumentError("MPOs must contain at least one tensor"))
    backend=nothing;scalar_type=nothing
    for X in (A,B),i in eachindex(X)
        hasqns(X[i]) && throw(ArgumentError("QN/block-sparse MPOs are not supported"))
        s=collect(siteinds(X,i))
        length(s)==2 || throw(ArgumentError("each MPO tensor must have one input and one output physical index"))
        input=filter(j->plev(j)==0,s)
        length(input)==1 && prime(only(input)) in s || throw(ArgumentError("physical indices must be paired at prime levels 0 and 1"))
        if i<length(X)
            length(linkinds(X,i))==1 || throw(ArgumentError("exactly one virtual bond between neighbors is required"))
        end
        data=ITensors.array(X[i])
        eltype(data) in (Float32,Float64,ComplexF32,ComplexF64) || throw(ArgumentError("use Float32, Float64, ComplexF32, or ComplexF64 tensors"))
        current=_backend(data)
        if isnothing(backend)
            backend=current
            scalar_type=eltype(data)
        else
            current==backend || throw(ArgumentError("both MPOs must use the same array backend"))
            eltype(data)==scalar_type || throw(ArgumentError("both MPOs must use the same scalar type throughout"))
        end
    end
    for i in eachindex(A)
        issetequal(siteinds(A,i),siteinds(B,i)) || throw(ArgumentError("MPO physical indices must match at each site"))
    end
    return nothing
end

"""
    streamed_product(A::MPO, B::MPO; cutoff=1e-14, maxdim=nothing,
                     mindim=1, block_right=16, truncate_kwargs=(;), observer=nothing)

Approximate the operator product `A * B` as an MPO using a two-pass streamed
QR/SVD contraction. Inputs are not modified; output physical indices use the
usual prime levels 0 and 1. This is a separate function and does not change
ITensorMPS's `apply` or `contract` methods.

`maxdim` caps intermediate retained bonds and the final rounding; its default is
the product of the input maximum bonds. `cutoff` uses NDTensors' squared-singular-
value truncation policy. `truncate_kwargs` overrides options for final `truncate!`.
`block_right` is the number of outgoing right-factor bond states per tile.
Smaller tiles reduce scratch memory but may increase runtime.

Only open-chain dense MPOs with one virtual bond between neighbors and paired
physical indices at prime levels 0 and 1 are supported. Load CUDA to enable the
optional CuArray backend; CPU execution does not require CUDA.

If supplied, `observer(record)` receives scalar metadata after each streamed
bond. No records are retained by the package. The observer must not mutate the
inputs. Records describe local tiles, not the total GPU memory peak.

The local factorization is SVD-equivalent up to floating-point error. The final
compressed MPO is not guaranteed to be the globally optimal approximation, and
`cutoff` is not a certified global relative error.
"""
function streamed_product(A::MPO,B::MPO;cutoff::Real=1e-14,
    maxdim::Union{Nothing,Integer}=nothing,mindim::Integer=1,
    block_right::Integer=16,truncate_kwargs::NamedTuple=(;),observer=nothing)
    _validate_inputs(A,B)
    isfinite(cutoff) && cutoff>=0 || throw(ArgumentError("cutoff must be finite and nonnegative"))
    block_right>0 || throw(ArgumentError("block_right must be positive"))
    cap=isnothing(maxdim) ? maxlinkdim(A)*maxlinkdim(B) : maxdim
    1<=mindim<=cap || throw(ArgumentError("require 1 <= mindim <= maxdim"))
    final_options=merge((;cutoff,maxdim=cap,mindim),truncate_kwargs)
    final_cutoff=get(final_options,:cutoff,cutoff)
    isfinite(final_cutoff) && final_cutoff>=0 || throw(ArgumentError("final cutoff must be finite and nonnegative"))
    1<=get(final_options,:mindim,mindim)<=get(final_options,:maxdim,cap) || throw(ArgumentError("invalid final truncation dimensions"))
    return replaceprime(_contract(prime(A),B;cutoff,maxdim=cap,mindim,
        block_right,truncate_kwargs=final_options,observer),2=>1)
end

end

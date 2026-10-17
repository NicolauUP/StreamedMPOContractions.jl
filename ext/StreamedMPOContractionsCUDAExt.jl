module StreamedMPOContractionsCUDAExt

using CUDA, LinearAlgebra
import StreamedMPOContractions as SMC

SMC._backend(::CUDA.CuArray)=:cuda
SMC._release_owned!(A::CUDA.CuArray)=CUDA.unsafe_free!(A)
SMC._geqrf_owned!(A::CUDA.CuMatrix,tau::CUDA.CuVector)=CUDA.CUSOLVER.geqrf!(A,tau)

function SMC._small_svd(R::CUDA.CuMatrix)
    # CUDA's QR-based gesvd accepts tall matrices. R can be wide near an edge.
    if size(R,1)<size(R,2)
        packed=copy(adjoint(R))
        try
            F=svd!(packed;alg=CUDA.CUSOLVER.QRAlgorithm())
            left=copy(adjoint(F.Vt))
            CUDA.unsafe_free!(F.Vt)
            return left,F.S,F.U
        finally
            CUDA.unsafe_free!(packed)
        end
    end
    F=svd!(R;alg=CUDA.CUSOLVER.QRAlgorithm())
    right=copy(adjoint(F.Vt))
    CUDA.unsafe_free!(F.Vt)
    return F.U,F.S,right
end

end

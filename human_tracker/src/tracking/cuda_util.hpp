#ifndef CUDA_UTIL_HPP
#define CUDA_UTIL_HPP

#include <cublas_v2.h>
#include <cublas_api.h>
#include <cuda.h>

#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
#define CublasSafeCall( err ) __cublasSafeCall( err, __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cublasSafeCall( cublasStatus_t err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( CUBLAS_STATUS_SUCCESS != err )
    {
        fprintf( stderr, "cublasSafeCall() failed at %s:%i : %i\n",
                 file, line, err );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}


#define CUDA_BLOCK_SIZE 192
#define TPTR(x) thrust::raw_pointer_cast(&x[0])

#endif

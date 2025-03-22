import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

from pycuda.compiler import SourceModule
import time

# Matrix size
N = 10  # Adjust as needed

# Define the parallel matrix multiplication kernel
kernel_code = """
__global__ void matmul(float *A, float *B, float *C, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0.0;
        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}
"""
def read_csv_to_list_rating(file_path):
    """Read a CSV file into a list of lists with specific column types."""
    with open(file_path, 'r',encoding='utf-8') as file:
        matrix = []
        header = file.readline().strip().split(',')
        for line in file:
            values = line.strip().split(',')
            row = [int(values[0]), int(values[1]), float(values[2]), int(values[3])]
            matrix.append(row)
        return header,matrix
def read_csv_to_list_movies(file_path):
    """Read a CSV file into a list of lists with specific column types."""
    with open(file_path, 'r',encoding='utf-8') as file:
        matrix = []
        header = file.readline().strip().split(',')
        for line in file:
            values = line.strip().split(',')
            row = [int(values[0]), str(values[1]), str(values[2])]
            matrix.append(row)
        return header,matrix
# Example usage
headerRating,matrixrating = read_csv_to_list_rating('ratings.csv')
headerMovies,matrixMovies = read_csv_to_list_movies('movies.csv')
totalRaters = int(matrixrating[-1][0]) #total rows


# Initialize matrices
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros_like(A, dtype=np.float32)

# Allocate memory on the GPU
A_gpu = cuda.mem_alloc(A.nbytes)
B_gpu = cuda.mem_alloc(B.nbytes)
C_gpu = cuda.mem_alloc(C.nbytes)

# Copy matrices to the GPU
cuda.memcpy_htod(A_gpu, A)
cuda.memcpy_htod(B_gpu, B)

# Compile the kernel
mod = SourceModule(kernel_code)
matmul = mod.get_function("matmul")

# Define grid and block sizes
BLOCK_SIZE = 16  # Common size for CUDA blocks (16x16 threads)
grid_size = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, (N + BLOCK_SIZE - 1) // BLOCK_SIZE, 1)
block_size = (BLOCK_SIZE, BLOCK_SIZE, 1)

# Run the parallel CUDA kernel
cuda.Context.synchronize()  # Ensure synchronization before execution
start_time = time.time()
matmul(A_gpu, B_gpu, C_gpu, np.int32(N), block=block_size, grid=grid_size)
cuda.Context.synchronize()  # Ensure synchronization after execution

# Copy result back to CPU
cuda.memcpy_dtoh(C, C_gpu)

# Track execution time for CUDA
end_time = time.time()
print(f"Parallel GPU Execution Time (CUDA): {end_time - start_time:.5f} seconds")

# Compute correct result using NumPy
C_cpu = np.dot(A, B)

# Check if the results are close
if np.allclose(C, C_cpu, atol=1e-2):  # Adjust the tolerance for precision
    print("CUDA result matches NumPy result!")
else:
    print("CUDA result does not match NumPy result.")


import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import ctypes

# Define the Row structure
class Row(ctypes.Structure):
    _fields_ = [("col1", ctypes.c_int), ("col2", ctypes.c_float)]

# Create some example row data
rows_data = [
    
]
rows_data.append(Row(1, 2.5))
rows_data.append(Row(2, 3.5))
rows_data.append(Row(3, 4.5))
i=4
f=5.5
rows_data.append(Row(i,f))
# Convert rows data to ctypes array
rows = (Row * len(rows_data))(*rows_data)

# Allocate memory on the GPU for the rows
rows_gpu = cuda.mem_alloc(ctypes.sizeof(rows))

# Transfer the rows data to the GPU
cuda.memcpy_htod(rows_gpu, rows)

# Define kernel code
kernel_code = """
struct Row {
    int col1;
    float col2;
};

__global__ void matmul(float *A, float *B, float *C, int N, Row *rows) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0.0;
        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }

    // Access Row data and print it
    if (row < N && col<N) {
        int col1_value = rows[row].col1;
        float col2_value = rows[row].col2;
        printf("Row %d Col %d-> col1: %d, col2: %f\\n", row,col, col1_value, col2_value);
    }
}
"""

# Compile the kernel
mod = SourceModule(kernel_code)

# Allocate matrices for A, B, and C
N = 4
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

# Allocate memory for matrices on the GPU
A_gpu = cuda.mem_alloc(A.nbytes)
B_gpu = cuda.mem_alloc(B.nbytes)
C_gpu = cuda.mem_alloc(C.nbytes)

# Transfer matrices to the GPU
cuda.memcpy_htod(A_gpu, A)
cuda.memcpy_htod(B_gpu, B)

# Get the kernel function
matmul = mod.get_function("matmul")

# Launch the kernel
block_size = (2, 2, 1)
grid_size = (2, 2)

matmul(A_gpu, B_gpu, C_gpu, np.int32(N), rows_gpu, block=block_size, grid=grid_size)

# Transfer result back to host
cuda.memcpy_dtoh(C, C_gpu)

# Print the result matrix
print("Resulting Matrix C:")
print(C)

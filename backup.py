import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import ctypes
from pycuda.compiler import SourceModule
import time

N = 50  # Adjust as needed

# Define the Row structure
class Row(ctypes.Structure):
    _fields_ = [("col1", ctypes.c_int), ("col2", ctypes.c_float)]

# Create some example row data
rows_data = []

# Define the parallel matrix multiplication kernel
kernel_code = """
struct Row {
    int col1;
    float col2;
};

__global__ void matmul(float *A, float *B, int N, int raters, int movies, Row *rows) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int found=0;
    // Access Row data and print it
    int workingID;
    float workingRating;
    if (row < movies && col < raters && col>0 && row>0) {
        for(int i=0;i<100;i++){
            int col1_value = rows[(col-1)*100+i].col1;
            float col2_value = rows[(col-1)*100+i].col2;
            if (col1_value == row){
                found=1;
                workingID=col1_value;
                int index=(col-1)*100+i;
                workingRating=col2_value;
                atomicExch(&rows[index].col1, -10);
                atomicExch(&rows[index].col2, -10.0f);
                break;
            }
        }
    
        if (found) {
            float value = 0.0;
            for (int k = 0; k < N; k++) {
                value += A[row * N + k] * B[k * movies + col];
            }
            //printf("%f\\n",value);
        }
    }
}
"""

mod = SourceModule(kernel_code)

def read_csv_to_list_rating(file_path):
    """Read a CSV file into a list of lists with specific column types."""
    with open(file_path, 'r', encoding='utf-8') as file:
        matrix = []
        header = file.readline().strip().split(',')
        for line in file:
            values = line.strip().split(',')
            row = [int(values[0]), int(values[1]), float(values[2]), int(values[3])]
            matrix.append(row)
        return header, matrix

def read_csv_to_list_movies(file_path):
    """Read a CSV file into a list of lists with specific column types."""
    with open(file_path, 'r', encoding='utf-8') as file:
        matrix = []
        header = file.readline().strip().split(',')
        for line in file:
            values = line.strip().split(',')
            row = [int(values[0]), str(values[1]), str(values[2])]
            matrix.append(row)
        return header, matrix

headerRating, matrixrating = read_csv_to_list_rating('ratings.csv')
headerMovies, matrixMovies = read_csv_to_list_movies('movies.csv')

totalRaters = int(matrixrating[-1][0])+1  # total rows
totalMovies = int(matrixMovies[-1][0])+1
print(totalRaters)
print(totalMovies)

d = {}
# row 0 is userID

for row in matrixrating:
    if row[0] not in d:
        d[row[0]] = []
    else:
        d[row[0]].append(list(row[1:4]))

for key, val in d.items():
    d[key] = sorted(val, key=lambda x: x[2])

for key, val in d.items():
    d[key] = d[key][::-1]

newD = {}

for key, val in d.items():
    newD[key] = d[key][:100]

totalUsers = len(newD)

rows_data = []
for i in range(1, (len(newD) + 1)):
    row = newD[i]
    for j in range(100):
        if j < len(row):
            rows_data.append(Row(row[j][0], row[j][1]))
        else:
            rows_data.append(Row(0, 0.0))

matrixAMovies = np.zeros((totalMovies, N), dtype=np.float32)
matrixBUsers = np.zeros((N, totalRaters), dtype=np.float32)
print()
print(len(matrixAMovies))
print(len(matrixAMovies[0]))
print()
print(len(matrixBUsers))
print(len(matrixBUsers[0]))

# Fill matrices with random values in the range -0.01 to 0.01
matrixAMovies = np.random.uniform(-0.01, 0.01, size=(totalMovies, N)).astype(np.float32)
matrixBUsers = np.random.uniform(-0.01, 0.01, size=(N, totalRaters)).astype(np.float32)

A_gpu_matrixAMovies = cuda.mem_alloc(matrixAMovies.nbytes)
B_gpu_matrixBUsers = cuda.mem_alloc(matrixBUsers.nbytes)

if A_gpu_matrixAMovies is None or B_gpu_matrixBUsers is None:
    print("Damn")

rows = (Row * len(rows_data))(*rows_data)

# Allocate memory on the GPU for the rows
rows_gpu = cuda.mem_alloc(ctypes.sizeof(rows))
if rows_gpu is None:
    print("damn")

# Transfer the rows data to the GPU
cuda.memcpy_htod(rows_gpu, rows)

cuda.memcpy_htod(A_gpu_matrixAMovies, matrixAMovies)
cuda.memcpy_htod(B_gpu_matrixBUsers, matrixBUsers)


matmul = mod.get_function("matmul")

# Define grid and block sizes
block_size = (16, 16, 1)  # 16x16 threads per block
grid_size = ((totalRaters + block_size[1] - 1) // block_size[1], 
             (totalMovies + block_size[0] - 1) // block_size[0])


print(f"Matrix A shape: {matrixAMovies.shape}")
print(f"Matrix B shape: {matrixBUsers.shape}")

# Calculate the shared memory size
shared_mem_size = ctypes.sizeof(Row) * min(totalRaters * 100, block_size[0] * block_size[1])

# Run the parallel CUDA kernel
cuda.Context.synchronize()  # Ensure synchronization before execution
start_time = time.time()
matmul(A_gpu_matrixAMovies, B_gpu_matrixBUsers, np.int32(N), np.int32(totalRaters), np.int32(totalMovies), rows_gpu,
       block=block_size, grid=grid_size, shared=shared_mem_size)
cuda.Context.synchronize()  # Ensure synchronization after execution
end_time = time.time()
# Copy result back to CPU
rows_result = np.zeros(len(rows_data), dtype=[('col1', np.int32), ('col2', np.float32)])
cuda.memcpy_dtoh(rows_result, rows_gpu)


count=0
# Check the values in rows_result
for i in range(len(rows_result)):
    if not ((rows_result[i]['col1'] == 0 and rows_result[i]['col2'] == 0.0) or (rows_result[i]['col1'] == -10 and rows_result[i]['col2'] == -10.0)):
        print(f"Row {i}: col1 = {rows_result[i]['col1']}, col2 = {rows_result[i]['col2']}")
        count+=1
print(count)

# Track execution time for CUDA

print(f"Parallel GPU Execution Time (CUDA): {end_time - start_time:.5f} seconds")
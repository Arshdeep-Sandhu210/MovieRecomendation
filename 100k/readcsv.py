import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import ctypes
from pycuda.compiler import SourceModule
import time
#define CLIP(x, min, max) (fminf(fmaxf(x, min), max))

N = 50  # Increase to capture more user preferences
  # Adjust as needed

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

__global__ void matmul(float *A, float *B, int N, int raters, int movies, Row *rows, float learning_rate) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Movies
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Raters

    if (row < movies && col < raters && col > 0 && row > 0) {
        int found = 0;
        float actual_rating = 0.0;

        // Find actual rating from stored data
        for (int i = 0; i < 100; i++) {
            if (rows[(col - 1) * 100 + i].col1 == row) {
                found = 1;
                actual_rating = rows[(col - 1) * 100 + i].col2;
                break;
            }
        }

        if (found) {
            float predicted_rating = 0.0;
            for (int k = 0; k < N; k++) {
                predicted_rating += A[row * N + k] * B[k * raters + col];
            }

            float error = actual_rating - predicted_rating;
            
            for (int k = 0; k < N; k++) {
                float lambda = 0.01;
                float regularization = lambda * (A[row * N + k] + B[k * raters + col]);
                float grad_A = -2.0 * error * B[k * raters + col] + regularization;
                float grad_B = -2.0 * error * A[row * N + k] + regularization;
                                
                // Apply updates
                atomicAdd(&A[row * N + k], -learning_rate * grad_A);
                atomicAdd(&B[k * raters + col], -learning_rate * grad_B);

                atomicExch(&A[row * N + k], fminf(fmaxf(A[row * N + k], -1.0f), 1.0f));
                atomicExch(&B[k * raters + col], fminf(fmaxf(B[k * raters + col], -1.0f), 1.0f));
            }

            

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
matrixAMovies = np.random.uniform(-0.1, 0.1, size=(totalMovies, N)).astype(np.float32)
matrixBUsers = np.random.uniform(-0.1, 0.1, size=(N, totalRaters)).astype(np.float32)

matrixAMovies = np.ascontiguousarray(matrixAMovies, dtype=np.float32)
matrixBUsers = np.ascontiguousarray(matrixBUsers, dtype=np.float32)

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
cuda.Context.synchronize()  # Ensure synchronization before execution]
print("A GPU Memory Address:", int(A_gpu_matrixAMovies))
print("B GPU Memory Address:", int(B_gpu_matrixBUsers))
print("Rows GPU Memory Address:", int(rows_gpu))

# start_time = time.time()
# matmul(A_gpu_matrixAMovies, B_gpu_matrixBUsers, np.int32(N), np.int32(totalRaters), np.int32(totalMovies), rows_gpu,learning_rate,
#        block=block_size, grid=grid_size, shared=shared_mem_size)
# cuda.Context.synchronize()  # Ensure synchronization after execution
# end_time = time.time()
# # Copy result back to CPU

# # 🚀 First, copy updated A and B back to CPU
# matrixAMovies_result = np.empty_like(matrixAMovies)
# matrixBUsers_result = np.empty_like(matrixBUsers)

# cuda.memcpy_dtoh(matrixAMovies_result, A_gpu_matrixAMovies)
# cuda.memcpy_dtoh(matrixBUsers_result, B_gpu_matrixBUsers)
# rows_result = np.zeros(len(rows_data), dtype=[('col1', np.int32), ('col2', np.float32)])
# cuda.memcpy_dtoh(rows_result, rows_gpu)


epochs = 100  # Number of iterations

initial_lr = np.float32(0.001)  # Start slightly lower than 0.003
decay_factor = np.float32(0.95)   # Reduce by 10% every 5 epochs
decay_step = 5

min_lr = np.float32(0.0005)  # Ensure learning rate never goes too low
def normalize_embeddings(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)  # Compute row-wise norm
    norms[norms == 0] = 1  # Prevent division by zero
    return matrix / norms
learning_rate = np.float32(0.001)
for epoch in range(epochs):
    # learning_rate = np.float32(initial_lr * np.exp(-0.05 * epoch))



    matrixAMovies_result = np.empty_like(matrixAMovies)
    matrixBUsers_result = np.empty_like(matrixBUsers)
    rows_result = np.zeros(len(rows_data), dtype=[('col1', np.int32), ('col2', np.float32)])
    start_time = time.time()
    # if epoch % 5 == 0:  # Every 5 epochs, reduce learning rate
    #     learning_rate *= 0.9  # Reduce by 10%

    matmul(A_gpu_matrixAMovies, B_gpu_matrixBUsers, np.int32(N), np.int32(totalRaters),
           np.int32(totalMovies), rows_gpu, learning_rate, block=block_size, grid=grid_size,
           shared=shared_mem_size)
    
    cuda.Context.synchronize()  # Wait for CUDA execution

    # Copy updated matrices and rows back to CPU
    cuda.memcpy_dtoh(matrixAMovies_result, A_gpu_matrixAMovies)
    cuda.memcpy_dtoh(matrixBUsers_result, B_gpu_matrixBUsers)
    cuda.memcpy_dtoh(rows_result, rows_gpu)


    if np.isnan(matrixAMovies_result).any() or np.isnan(matrixBUsers_result).any():
        print("Error: NaN detected in A or B matrix!")
        exit()

    if np.isinf(matrixAMovies_result).any() or np.isinf(matrixBUsers_result).any():
        print("Error: Inf detected in A or B matrix!")
        exit()

    # Compute loss
    mse = 0
    count = 0
    for i in range(len(rows_result)):
        if rows_result[i]['col1'] > 0:
            actual = rows_result[i]['col2']
            predicted = np.dot(matrixAMovies_result[rows_result[i]['col1'], :], matrixBUsers_result[:, i // 100])
            
            if np.isnan(predicted) or np.isinf(predicted):
                print(f"Error in prediction: row {i}, actual {actual}, predicted {predicted}")
                exit()

            mse += (actual - predicted) ** 2
            count += 1

    mse = mse / count if count > 0 else float('inf')  # Avoid division by zero

    end_time = time.time()
    print(f"Epoch {epoch+1}/{epochs}, MSE: {mse:.5f}, Time: {end_time - start_time:.5f} sec")





# Track execution time for CUDA

print(f"Parallel GPU Execution Time (CUDA): {end_time - start_time:.5f} seconds")


user_id = 2  # Example user (change as needed)
movie_id = rows_result[17]['col1']  # Get the movie ID from rows_result

predicted = np.dot(matrixAMovies_result[movie_id, :], matrixBUsers_result[:, user_id])
print(f"Predicted rating for user {user_id} and movie {movie_id}: {predicted:.2f}")







listUnused =[]
for m in (matrixrating[55:180]):
    hit=0
    for i in range(100,200):
        if int(m[1]) ==int(rows_data[i].col1):
            
            hit=1
            break
    if hit==0:
        listUnused.append([int(m[1]),int(m[2])])

listUnused2 =[]
for m in (matrixrating[55:180]):
    hit=0
    for i in range(100,200):
        if int(m[1]) ==int(rows_data[i].col1):
            
            hit=1
            break
    if hit==1:
        listUnused2.append([int(m[1]),int(m[2])])


for i in range(len(listUnused)):
    user_id = 2  # Example user (change as needed)
    movie_id = listUnused[i][0]  # Get the movie ID from rows_result
    actualPred = listUnused[i][1]
    predicted = np.dot(matrixAMovies_result[movie_id, :], matrixBUsers_result[:, user_id])
    print(f"Predicted rating for user {user_id} and movie {movie_id}: {predicted:.2f}     Actual {actualPred}")

print()

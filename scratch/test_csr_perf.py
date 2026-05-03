import time
import numpy as np
from scipy.sparse import csr_matrix

N = 9597
L = 60140

# Mock CSR with typical sparsity for koko (based on nnz ~ 200 per row for predictions)
nnz_per_row = 200
data = np.random.rand(N * nnz_per_row).astype(np.float32)
indices = np.random.randint(0, L, size=N * nnz_per_row)
indptr = np.arange(0, (N + 1) * nnz_per_row, nnz_per_row)
csr = csr_matrix((data, indices, indptr), shape=(N, L))

print(f"Testing CSR row access for N={N}, L={L}, nnz_per_row={nnz_per_row}")

t0 = time.time()
n_test = 1000
for i in range(n_test):
    _ = csr[i].toarray()
dt = time.time() - t0
print(f"Time for {n_test} rows: {dt:.4f}s ({dt/n_test:.6f}s per row)")
print(f"Estimated time for full epoch ({N} rows): {dt/n_test * N:.2f}s")

# Test batch access
batch_size = 256
t0 = time.time()
for i in range(0, n_test, batch_size):
    _ = csr[i:i+batch_size].toarray()
dt = time.time() - t0
print(f"Time for {n_test} rows (batched {batch_size}): {dt:.4f}s ({dt/n_test:.6f}s per row)")
print(f"Estimated time for full epoch ({N} rows) batched: {dt/n_test * N:.2f}s")

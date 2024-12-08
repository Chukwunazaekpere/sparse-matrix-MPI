"""
Author: Chukwunazaekpere Emmanuel Obioma
Lecture: Parallelism
Nationality: Biafran
Email-1: chukwunazaekpere.obioma@ue-germany.de 
Email-2: ceo.naza.tech@gmail.com
************************************************
Implementation: to implement the sequential multiplication of squared matrices
Course: Multi-core Programming
Written: Dec 8th 2024
Due: Dec 9th 2024
"""
from scipy.sparse import csr_matrix, csc_matrix
import mpi4py
from mpi4py import MPI # import the 'MPI' module
import numpy as np
from datetime import datetime
import random
from time import time
import logging
logger = logging.getLogger(__name__)
log_date = datetime.now()
logging.basicConfig(level=logging.INFO)

mpi4py.rc.initialize = False  # do not initialize MPI automatically
mpi4py.rc.finalize = False    # do not finalize MPI automatically


MPI.Init()      # manual initialization of the MPI environment

class Sequential_Matrix_Multiplication():
    def __init__(self):
        self.row_dim = 0
        self.col_dim = 0

    """Get an input from user as to the size of the square matrix to be generated"""
    def get_mat_size(self):
        logging.info(msg=f"\n\t {datetime.now()} This is the sequential matrix multiplication algorithm...")
        try:
            matrix_row = input("\n\t Please enter a number for matrix row: ")
            row_dim = int(matrix_row)
            if row_dim:
                self.row_dim = row_dim
                matrix_cols = input("\n\t Please enter a number for matrix columns: ")
                col_dim = int(matrix_cols)
                if col_dim:
                    self.col_dim = col_dim

        except Exception as err:
            return f"Matrix dimension unacceptable. Please enter a number."

    def generate_sparse_matrix(self):
        """Sparse matrix generation"""
        data = np.array(self.generate_dense_vector(self.col_dim))
        row = np.array(self.generate_dense_vector(self.col_dim))
        col = np.array(self.generate_dense_vector(self.col_dim))
        matrix = csc_matrix((data, (row, col)), dtype = np.int8).toarray() 
        return matrix
    
    def generate_dense_vector(self, dim:int):
        """Dense matrix generation"""
        vector = []
        for val in range(0, dim): # use column index as matrices are only conformal for multiplication, if there inner dimensions are the same
            value = random.randint(2, 100) # generate random vector elements
            vector.append(value)
        return vector
    
    def sequential_multiplication(self):
        sparse_matrix = self.generate_sparse_matrix()
        dense_vector = self.generate_dense_vector(len(sparse_matrix[0]))
        logging.info(msg=f"The generated sparse matrix is: {sparse_matrix}")
        logging.info(msg=f"The generated dense vector is: {dense_vector}")
        dot_product = []
        multiply_comp = 0
        for matrix_A_rows in sparse_matrix:
            index_of_row_in_A = 0
            matrix_B_row_index= 0
            matrix_B_col_index= 0
            row_sums = []
            while True:
                multiply_comp+= (matrix_A_rows[matrix_B_row_index]*(dense_vector[matrix_B_row_index]))
                add_row_B_index = True
                if matrix_B_row_index == self.row_dim-1:
                    index_of_row_in_A+=1
                    matrix_B_col_index+=1
                    matrix_B_row_index=0
                    add_row_B_index = False
                    row_sums.append(multiply_comp)
                    multiply_comp = 0
                if index_of_row_in_A == self.row_dim:
                    dot_product.append(row_sums)
                    break
                matrix_B_row_index+= 1 if add_row_B_index else 0
        # print("\n\t matrices: ", matrix_A, matrix_B)
        return dot_product


if __name__ == "__main__":
    mat_mult = Sequential_Matrix_Multiplication()
    start_time = time()
    mat_dim = mat_mult.get_mat_size()
    if mat_dim == None:
        dim = mat_mult.sequential_multiplication()
        end_time = time()
        print(f"\n\t total time taken: {(end_time-start_time)} secs")
    else:
        print(mat_dim)
MPI.Finalize()  # manual finalization of the MPI environment

    



    

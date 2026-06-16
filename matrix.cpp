#include <iostream>
#include <vector>
using namespace std;

//Matrix class
class matrix {
public:
    //Size and matrix information
    int row;
    int col;
    vector<vector<float>> val;

    //Constructor based on size
    matrix(int row, int col){
        this->row = row;
        this->col = col;
        val = vector<vector<float>> (row, vector<float>(col, 0));
    }

    //Constructor from vector
    matrix(int row, int col, vector<float> flat){
        this->row = row;
        this->col = col;
        val = vector<vector<float>> (row, vector<float>(col, 0));
        for (int i = 0; i < row; i++){
            for (int j = 0; j < col; j++){
                val[i][j] = flat[i*col + j];
            }
        }
    }

    //Transpose
    matrix T() const{
        matrix temp(col, row);
        for (int i = 0; i < row; i++){
            for (int j = 0; j < col; j++){
                temp.val[j][i] = val[i][j];
            }
        }
        return temp;
    }


    //Prints out the matrix
    void print(){
        for (int i = 0; i < row; i++){
            for (int j = 0; j < col; j++){
                cout << val[i][j] << " ";
            }
            cout << endl;
        }
    }

    //Matrix -> vector
    vector<float> flatten(){
        vector<float> output(row*col);
        for (int i = 0; i < row; i++){
            for (int j = 0; j < col; j++){
                output[i*col + j] = val[i][j];
            }
        }
        return output;
    }

    matrix unflatten(int row, int col, vector<float>& flat){
        if (row*col != flat.size()){
            throw invalid_argument("Row col values do not align with flattened vector");
        }
        matrix temp(row, col);
        for(int i = 0; i < flat.size(); i++){
            temp.val[i/col][i%col] = flat[i];
        }
        return temp;
    }

    //Reshape matrix
    matrix reshape(int row, int col){
        if (row*col != row*col){
            throw invalid_argument("Row col values do not align with original matrix");
        }
        vector<float> temp = this->flatten();
        matrix output = unflatten(row, col, temp);
        return output;
    }

};

namespace matrixOps {
    //Naive matrix multiplication
    matrix matmul_naive(const matrix& A, const matrix& B){
        if (A.col != B.row){
            throw invalid_argument("Matrix 1 col does not match matrix 2 row");
        }

        matrix output(A.row, B.col);
        for (int i = 0; i < A.row; i++){
            for (int j = 0; j < A.col; j ++){
                for (int k = 0; k < B.col; k++){
                    output.val[i][k] += A.val[i][j]*B.val[j][k];
                }
            }
        }
        return output;
    }
    
    //Naive matrix multiplication with transposition of second matrix
    matrix matmul_naive_transposed(const matrix& A, const matrix& B){
        if (A.col != B.row){
            throw invalid_argument("Matrix 1 col does not match matrix 2 row");
        }

        matrix C = B.T();
        matrix output(A.row, B.col);
        for (int i = 0; i < A.row; i++){
            for (int j = 0; j < A.col; j ++){
                for (int k = 0; k < B.col; k++){
                    output.val[i][k] += A.val[i][j]*C.val[k][j];
                }
            }
        }
        return output;
    }

    //Vectorized matrix multiplication
    matrix matmul_vec(matrix& A, matrix& B){
        if (A.col != B.row){
            throw invalid_argument("Matrix 1 col does not match matrix 2 row");
        }

        vector<float> vecA = A.flatten();
        vector<float> vecB = B.T().flatten();
        
        vector<float> output(A.row*B.col);


        for (int i = 0; i < A.row; i++){
            for (int j = 0; j < B.col; j++){
                for (int k = 0; k < A.col; k++){
                    output[i*B.col + j] += vecA[i*A.col + k]*vecB[j*B.row + k]; 
                }
            }
        }
        
        matrix mat(A.row, B.col, output);
        return mat;
    }
};


int main(){
    matrix test(1000,1000);
    int count = 1;
    for (int i = 0; i < test.row; i++){
        for (int j = 0; j < test.col; j++){
            test.val[i][j] = count++;
        }
    }
    matrix testT = test.T();
    matrix multed = matrixOps::matmul_vec(test, testT);

    return 0;
}

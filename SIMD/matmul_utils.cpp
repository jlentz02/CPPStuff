#include "matmul_utils.h"
#include <vector>

using namespace std;

vector<float> pad(const vector<float>& matrix, int rowA, int colA){
    if (rowA%8 == 0 & colA%8 == 0){
        return matrix;
    }

    int rowA_pad = rowA;
    int colA_pad = colA;

    if (rowA%8 != 0){
        rowA_pad += 8 - rowA%8;
    }

    if (colA%8 != 0){
        colA_pad += 8 - colA%8;
    }

    vector<float> output(rowA_pad*colA_pad, 0.0f);

    for (int i = 0; i < rowA_pad; i++){
        for (int j = 0; j < colA_pad; j++){
            if (i < rowA & j < colA){
                output[i*colA_pad + j] = matrix[i*colA + j];
            }
        }
    }

    return output;

}

vector<float> unpad(const vector<float>&matrix, int rowA, int colA){
    return matrix;
}
#include "matmul_utils.h"
#include <vector>
#include <string>
#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <filesystem>

using namespace std;

//Pads the vectorized matrix by adding zeros in the right spots
//Returns a tuple of new matrix size for use in the matmul function
vector<int> pad(vector<float>& matrix, int rowA, int colA){
    if (rowA%8 == 0 & colA%8 == 0){
        return vector<int> {rowA,colA};
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

    matrix = output;

    return vector<int> {rowA_pad, colA_pad};
}

//Different version of padding function
void pad(std::vector<float>& matrix, int rowA, int colA, int rowA_pad, int colA_pad){

    vector<float> output(rowA_pad*colA_pad, 0.0f);

    for (int i = 0; i < rowA_pad; i++){
        for (int j = 0; j < colA_pad; j++){
            if (i < rowA & j < colA){
                output[i*colA_pad + j] = matrix[i*colA + j];
            }
        }
    }

    matrix = output;
}

void unpad(vector<float>& matrix, int rowA, int colA, int rowPad){
    
    vector<float> output(rowA*colA, 0.0f);

    for (int i = 0; i < rowA; i++){
        for (int j = 0; j < colA; j++){
            output[colA*i + j] = matrix[(colA + rowPad)*i + j];
        }

    }

    matrix = output;
}

void print_matrix(vector<float>& matrix, int row, int col){
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            cout << matrix[col*i + j] << " ";
        }
        cout << endl;
    }
}

void record_experiment(string algo_name, vector<int> times){
    const char *path = "C:/Users/joshu/Documents/Coding Projects/CPPStuff/experiments.csv";
    ofstream myfile(path, ios::app);

    if (!myfile){
        cerr << "Error opening file" << endl;
    }

    vector<int> sizes = {16,32,64,128,256,384,512,640,768,896,1024,1280,1536,1792,2048};

    myfile << algo_name;

    for (int i = 0; i < times.size(); i++){
        myfile << "," << times[i];
    }

    myfile << "\n";

    myfile.close();

    cout << "Writing complete";
}
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

    //Transpose
    void T(){
        vector<vector<float>> temp (col, vector<float>(row,0));
        for (int i = 0; i < row; i++){
            for (int j = 0; j < col; j++){
                temp[j][i] = val[i][j];
            }
        }
        int temp_int = row;
        row = col;
        col = temp_int;
        val = temp;
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

    vector<vector<float>> unflatten(int row, int col, vector<float> flat){
        if (row*col != flat.size()){
            throw invalid_argument("Row col values do not align with flattened vector");
        }
        vector<vector<float>> temp(row, vector<float>(col, 0));
        for(int i = 0; i < flat.size(); i++){
            temp[i/col][i%col] = flat[i];
        }
        return temp;
    }

    //Reshape matrix
    void reshape(int row, int col){
        if (row*col != row*col){
            throw invalid_argument("Row col values do not align with original matrix");
        }
        vector<float> temp = this->flatten();
        vector<vector<float>> output = unflatten(row, col, temp);
        this->row = row;
        this->col = col;
        val = output;
    }
    
};

int main(){
    matrix test(4,6);
    int count = 0;
    for (int i = 0; i < test.row; i++){
        for (int j = 0; j < test.col; j++){
            test.val[i][j] = count++;
        }
    }

    test.print();

    return 0;
}

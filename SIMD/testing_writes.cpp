#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include "matmul_utils.h"
using namespace std;
int main(){
    

    vector<int> sizes = {16,32,64,128,256,384,512,640,768,896,1024,1280,1536,1792,2048};
    vector<int> test_times = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14};

    string alogname = "testingwritefunction";

    record_experiment(alogname, test_times);

    return 0;
}
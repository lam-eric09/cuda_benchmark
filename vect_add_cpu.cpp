#include <vector>
#include <iostream>

constexpr size_t N = 100000000;

void add_vector(double *out, double *a, double *b){
    for (int i=0; i<N; i++){
        out[i] = a[i] + b[i];
    }
}

int main(){
    std::vector<double> a(N);
    std::vector<double> b(N);
    std::vector<double> out(N);

    for (int i=0; i<N; i++){
        a[i] = 1.0;
        b[i] = 2.0;
    }

    add_vector(out.data(), a.data(), b.data());
    
    return 0;
}
// LinearRegressionPytorch.cpp : Defines the entry point for the application.
//

#include "LinearRegressionPytorch.h"

using namespace std;


auto forward(torch::Tensor w,torch::Tensor x) {
    return w * x;
}

auto loss(torch::Tensor y, torch::Tensor y_pred) {
    auto sum = pow((y - y_pred), 2);
    at::Tensor x = sum;
    at::Tensor x_mean = x.mean();
    torch::Tensor res = x_mean;
    return res;
}

int main() {
    torch::Tensor X = torch::tensor({ 1,2,3,4 });
    torch::Tensor Y = torch::tensor({ 3,6,9,12 });
    
    auto w = torch::tensor(0.0,torch::requires_grad());
    /*auto y_pred = forward(w, X);
    
    auto l = loss(Y, y_pred);
    l.backward();
    cout << w.grad() << endl;*/
    
    int i_ter = 100;
    float learningRate = 0.01;
   
    for (int i = 0; i < i_ter; i++) {
        auto y_pred = forward(w, X);
        auto l = loss(Y, y_pred);
        cout << i << " " << y_pred << " " << "Loss = " <<  l << endl;
        l.backward();
        torch::NoGradGuard guard;
        w -= learningRate * w.grad();

        w.mutable_grad() = torch::tensor(0.0);
        
        
    }

    
    return 0;
}
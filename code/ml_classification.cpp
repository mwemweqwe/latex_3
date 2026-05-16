#include <iostream>
#include <vector>
#include <random>

// Простая реализация логистической регрессии
class LogisticRegression {
private:
    std::vector<double> weights;
    double learning_rate;
    int n_iterations;
    
    double sigmoid(double z) {
        return 1.0 / (1.0 + exp(-z));
    }
    
public:
    LogisticRegression(double lr = 0.01, int iter = 1000) 
        : learning_rate(lr), n_iterations(iter) {}
    
    void fit(const std::vector<std::vector<double>>& X, 
             const std::vector<int>& y) {
        int n_features = X[0].size();
        weights.assign(n_features, 0.0);
        
        for (int iter = 0; iter < n_iterations; iter++) {
            std::vector<double> gradients(n_features, 0.0);
            
            for (size_t i = 0; i < X.size(); i++) {
                double z = 0.0;
                for (size_t j = 0; j < weights.size(); j++) {
                    z += weights[j] * X[i][j];
                }
                double pred = sigmoid(z);
                double error = pred - y[i];
                
                for (size_t j = 0; j < weights.size(); j++) {
                    gradients[j] += error * X[i][j];
                }
            }
            
            for (size_t j = 0; j < weights.size(); j++) {
                weights[j] -= learning_rate * gradients[j] / X.size();
            }
        }
    }
    
    double predict(const std::vector<double>& x) {
        double z = 0.0;
        for (size_t j = 0; j < weights.size(); j++) {
            z += weights[j] * x[j];
        }
        return sigmoid(z) > 0.5 ? 1.0 : 0.0;
    }
};

int main() {
    std::cout << "Machine Learning Classification in C++" << std::endl;
    std::cout << "Logistic Regression implementation" << std::endl;
    return 0;
}
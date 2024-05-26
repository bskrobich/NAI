#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <tuple>

using namespace std;
using namespace std::chrono;

struct Item {
    int weight;
    int value;
};

pair<int, vector<Item>> readInput(const string& filename) {
    ifstream file(filename);
    int totalCapacity;
    vector<Item> items;

    if (!file.is_open()) {
        cerr << "Unable to open file: " << filename << endl;
    }
    file >> totalCapacity;

    int weight, value;
    while (file >> weight >> value) {
        items.push_back({weight, value});
    }

    file.close();
    return {totalCapacity, items};
}

tuple<vector<int>, int, int> knapsackBruteForce(int totalCapacity, const vector<Item>& items) {
    int numberOfItems = items.size();
    int bestValue = 0;
    int bestWeight = 0;
    vector<int> bestCombination;

    int totalCombinations = pow(2, numberOfItems);

    for (int i = 0; i < totalCombinations; ++i) {
        vector<int> combination(numberOfItems, 0);
        int totalWeight = 0;
        int totalValue = 0;

        for (int j = 0; j < numberOfItems; ++j) {
            if ((i & (1 << j)) != 0) {
                combination[j] = 1;
                totalWeight += items[j].weight;
                totalValue += items[j].value;
            }
        }
        if (totalWeight <= totalCapacity && totalValue > bestValue) {
            bestValue = totalValue;
            bestWeight = totalWeight;
            bestCombination = combination;
        }
    }
    return {bestCombination, bestValue, bestWeight};
}

int main() {
    string filename = "../data.txt";
    auto [totalCapacity, items] = readInput(filename);

    auto start = high_resolution_clock::now();
    auto [bestCombination, bestValue, totalWeight] = knapsackBruteForce(totalCapacity, items);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "Best combination:";
    for (int i : bestCombination) {
        cout << " " << i;
    }
    cout << "\nBest value: " << bestValue << endl;
    cout << "Total weight: " << totalWeight << endl;
    cout << "Execution time: " << duration.count() / 1000.0 << " seconds" << endl;

    return 0;
}

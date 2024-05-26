#include <iostream>
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

int main() {
    string filename;
    cout << "Enter data file absolute path: ";
    cin >> filename;
    auto [totalCapacity, items] = readInput(filename);

    cout << "Total capacity: " << totalCapacity << endl;
    cout << "Number of items: " << items.size() << endl;

}

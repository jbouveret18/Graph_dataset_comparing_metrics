#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <unordered_set>
#include <cmath>
#include <algorithm>


struct MatrixEntry {
    int row, col;
    double value;
    bool operator==(const MatrixEntry& other) const {
        return row == other.row && col == other.col && value == other.value;
    }
};

namespace std {
    template <>
    struct hash<MatrixEntry> {
        size_t operator()(const MatrixEntry& entry) const {
            size_t hashValue = 17;
            hashValue = hashValue * 31 + std::hash<int>{}(entry.row);
            hashValue = hashValue * 31 + std::hash<int>{}(entry.col);
            hashValue = hashValue * 31 + std::hash<double>{}(entry.value);
            return hashValue;
        }
    };
}

void readMatrixFile(const std::string& filename, std::vector<MatrixEntry>& matrixEntries, int& rows, int& cols, int& entries) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    std::string line;

    // Read header
    while (getline(file, line)) {
        if (line[0] != '%') {
            std::stringstream ss(line);
            ss >> rows >> cols >> entries;
            break;
        }
    }

    MatrixEntry entry;
    while (getline(file, line)) {
        std::stringstream local_ss(line);
        local_ss >> entry.row >> entry.col;
        if (local_ss >> entry.value) {
            // Value is provided in the file
        } else {
            entry.value = 0.0;  // Set a default value if not provided
        }
        matrixEntries.push_back(entry);
    }

    file.close();
}


void populateNeighborSets(const std::vector<MatrixEntry>& matrixEntries, std::vector<std::unordered_set<int>>& neighbors, int rows, int cols) {
#pragma omp parallel for
    for (size_t i = 0; i < matrixEntries.size(); ++i) {
        const auto& entry = matrixEntries[i];
        int maxRowCol = std::max(entry.row, entry.col);
        if (maxRowCol <= rows) {
#pragma omp critical
            {
                neighbors[entry.row - 1].insert(entry.col - 1); // Adjust indices to start from 0
                neighbors[entry.col - 1].insert(entry.row - 1); // For symmetric matrices
            }
        }
    }
}

double calculateCosineSimilarity(const std::unordered_set<int>& set1, const std::unordered_set<int>& set2) {
    // Calculate intersection size
    int intersection = 0;
    for (int elem : set1) {
        if (set2.find(elem) != set2.end()) {
            intersection++;
        }
    }

    // Calculate magnitudes
    double magnitude1 = std::sqrt(set1.size());
    double magnitude2 = std::sqrt(set2.size());

    // Ensure division by zero is avoided
    if (magnitude1 == 0 || magnitude2 == 0) {
        return 0.0; // Return zero similarity if any set has zero magnitude
    }

    // Calculate cosine similarity
    double similarity = intersection / (magnitude1 * magnitude2);
    return similarity;
}

void calculateDegreeCentrality(const std::vector<std::unordered_set<int>>& neighbors, std::vector<double>& centrality) {
#pragma omp parallel for
    for (size_t i = 0; i < neighbors.size(); ++i) {
        centrality[i] = static_cast<double>(neighbors[i].size());
    }
}

void compareCentrality(const std::vector<double>& centrality1, const std::vector<double>& centrality2) {
    std::cout << "Node\tGraph 1 Centrality\tGraph 2 Centrality\n";
    int numberOfCommonCenters = 0;

    size_t minSize = std::min(centrality1.size(), centrality2.size());

#pragma omp parallel for reduction(+:numberOfCommonCenters)
    for (size_t i = 0; i < minSize; ++i) {
        if (i < centrality1.size() && i < centrality2.size() && centrality1[i] == centrality2[i]){
#pragma omp critical
            {
                std::cout << i << "\t" << centrality1[i] << "\t\t\t" << centrality2[i] << std::endl;
            }
            numberOfCommonCenters++;
        }
    }
    std::cout << "The number of common centers: " << numberOfCommonCenters << std::endl;
    std::cout << "Total number of centers: " << centrality1.size()+centrality2.size() << std::endl;
    double percentage = static_cast<double>(numberOfCommonCenters) / (centrality1.size()+centrality2.size());
    std::cout << "Percentage of common centers: " << percentage*100 << "%" << std::endl;
}

std::vector<std::vector<double>> calculateCosineSimilarityMatrix(const std::vector<std::unordered_set<int>>& vectors1, const std::vector<std::unordered_set<int>>& vectors2) {
    std::vector<std::vector<double>> similarityMatrix(vectors1.size(), std::vector<double>(vectors2.size(), 0.0));

    // It's harder to parallelize this part due to data dependencies and updates to shared matrix
    // Parallelizing this part might not yield significant speedup and might require more complex strategies

    for (size_t i = 0; i < vectors1.size(); ++i) {
        for (size_t j = 0; j < vectors2.size(); ++j) {
            similarityMatrix[i][j] = calculateCosineSimilarity(vectors1[i], vectors2[j]);
        }
    }

    return similarityMatrix;
}

// Function to check if node 'v' is a neighbor of node 'u'
bool isNeighbor(const std::vector<MatrixEntry>& matrix, int u, int v) {
    // Assuming the matrix is symmetric (u and v can be swapped)
    return std::any_of(matrix.begin(), matrix.end(), [u, v](const MatrixEntry& entry) {
        return (entry.row == u && entry.col == v) || (entry.row == v && entry.col == u);
    });
}

// Function to compute the neighborhood overlap between two nodes in matrices represented by vectors
double neighborhoodOverlap(const std::vector<MatrixEntry>& matrix1, const std::vector<MatrixEntry>& matrix2, int node1, int node2) {
    int count = 0;

    // Check for each node in matrix 1 if it is a neighbor of the given node
#pragma omp parallel for reduction(+:count)
    for (const auto& entry : matrix1) {
        if ((entry.row == node1 && isNeighbor(matrix2, node2, entry.col)) ||
            (entry.col == node1 && isNeighbor(matrix2, node2, entry.row))) {
            count++;
        }
    }

    // Calculate overlap as the ratio of common neighbors to total neighbors of node1
    double overlap = static_cast<double>(count) / matrix1.size(); // Assuming matrix1.size() as the total possible neighbors
    return overlap;
}

int main() {
    std::vector<MatrixEntry> alphaMatrix, betaMatrix;
    int alphaRows, alphaCols, alphaEntries;
    int betaRows, betaCols, betaEntries;

    readMatrixFile("C:\\Users\\jbouv\\OneDrive\\Desktop\\Ecole\\ISEP\\Cycle_ingenieur\\A3\\ITU\\Classes\\Parallel Algorithms\\Project\\Project_Code\\data\\road-euroroad.edges", alphaMatrix, alphaRows, alphaCols, alphaEntries);
    readMatrixFile("C:\\Users\\jbouv\\OneDrive\\Desktop\\Ecole\\ISEP\\Cycle_ingenieur\\A3\\ITU\\Classes\\Parallel Algorithms\\Project\\Project_Code\\data\\road-minnesota.mtx", betaMatrix, betaRows, betaCols, betaEntries);

    std::cout << "alpha Matrix Entries: " << alphaEntries << std::endl;
    std::cout << "beta Matrix Entries: " << betaEntries << std::endl;

    std::vector<std::unordered_set<int>> alphaNeighbors(alphaRows);
    std::vector<std::unordered_set<int>> betaNeighbors(betaRows);

#pragma omp parallel sections
    {
#pragma omp section
        {
            populateNeighborSets(alphaMatrix, alphaNeighbors, alphaRows, alphaCols);
        }
#pragma omp section
        {
            populateNeighborSets(betaMatrix, betaNeighbors, betaRows, betaCols);
        }
    }
    std::vector<std::vector<double>> similarityMatrix = calculateCosineSimilarityMatrix(alphaNeighbors, betaNeighbors);

    double similarityNumber = 0.0;
    // Display similarity matrix
#pragma omp parallel for reduction(+:similarityNumber)
    for (size_t i = 0; i < similarityMatrix.size(); ++i) {
        for (size_t j = 0; j < similarityMatrix[i].size(); ++j) {
            if(similarityMatrix[i][j]!=0){
                std::cout << "Similarity between alpha node " << i << " and beta node " << j << ": " << similarityMatrix[i][j] << std::endl;
                similarityNumber+=similarityMatrix[i][j];
            }
        }
    }
    std::cout << "Total Cosine similarity is " << (similarityNumber/(betaEntries*alphaEntries))*100 << "%" << std::endl;

    std::vector<double> alphaCentrality(alphaRows, 0.0);

    std::vector<double> betaCentrality(betaRows, 0.0);

    // Calculate degree centrality for alpha graph
    calculateDegreeCentrality(alphaNeighbors, alphaCentrality);

    // Calculate degree centrality for beta graph
    calculateDegreeCentrality(betaNeighbors, betaCentrality);

    compareCentrality(alphaCentrality,betaCentrality);

    // Calculate and print neighborhood overlap for all nodes in both matrices
    double totalNodeNumber = alphaMatrix.size()*betaMatrix.size();
    double overlappingNeighborhood = 0.0;
#pragma omp parallel for reduction(+:overlappingNeighborhood)
    for (size_t i = 0; i < alphaMatrix.size(); ++i) {
#pragma omp parallel for reduction(+:overlappingNeighborhood)
        for (size_t j = 0; j < betaMatrix.size(); ++j) {
            double overlap = neighborhoodOverlap(alphaMatrix, betaMatrix, i, j);
            if(overlap!=0){
#pragma omp critical
                std::cout << "Neighborhood overlap between node " << i << " in 1st Matrix and node " << j << " in 2nd Matrix: " << overlap << std::endl;
                overlappingNeighborhood+=overlap;
            }

        }
    }
    std::cout << "Total overlapping neighborhood: " << overlappingNeighborhood << std::endl;
    std::cout << "Total nodes compared: " << totalNodeNumber << std::endl;
    std::cout <<"Total percentage of overlapping: "<< (overlappingNeighborhood/totalNodeNumber)*100 <<"%"<<std::endl;
    return 0;
}

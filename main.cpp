#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <chrono>

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
    // Hash function for std::pair<int, int>
    template <>
    struct hash<std::pair<int, int>> {
        size_t operator()(const std::pair<int, int>& p) const {
            size_t hashValue = 17;
            hashValue = hashValue * 31 + std::hash<int>{}(p.first);
            hashValue = hashValue * 31 + std::hash<int>{}(p.second);
            return hashValue;
        }
    };
}
// Function to extract edges from MatrixEntry instances
std::unordered_set<std::pair<int, int>> extractEdges(const std::vector<MatrixEntry>& entries) {
    std::unordered_set<std::pair<int, int>> edges;

#pragma omp parallel
    {
        std::unordered_set<std::pair<int, int>> localEdges;

#pragma omp for
        for (size_t i = 0; i < entries.size(); ++i) {
            const auto& entry = entries[i];
            localEdges.insert({entry.row, entry.col});
            localEdges.insert({entry.col, entry.row});
        }

        // Merge local sets into the global set
#pragma omp critical
        {
            edges.insert(localEdges.begin(), localEdges.end());
        }
    }

    return edges;
}

int graphEditDistance(const std::unordered_set<std::pair<int, int>>& graph1_edges,
                      const std::unordered_set<std::pair<int, int>>& graph2_edges) {
    int differences = 0;

#pragma omp parallel for reduction(+:differences)
    for (size_t i = 0; i < graph1_edges.size(); ++i) {
        auto it = graph1_edges.begin();
        std::advance(it, i);
        const auto& edge = *it;

        if (graph2_edges.find(edge) == graph2_edges.end()) {
            differences++;
        }
    }

#pragma omp parallel for reduction(+:differences)
    for (size_t i = 0; i < graph2_edges.size(); ++i) {
        auto it = graph2_edges.begin();
        std::advance(it, i);
        const auto& edge = *it;

        if (graph1_edges.find(edge) != graph1_edges.end()) {
            differences++;
        }
    }

    return differences;
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

double commonCentersPercentage;
int numberOfCommonCenters = 0;

void compareCentrality(const std::vector<double>& centrality1, const std::vector<double>& centrality2) {
    //std::cout << "Node\tGraph 1 Centrality\tGraph 2 Centrality\n";
    size_t minSize = std::min(centrality1.size(), centrality2.size());

#pragma omp parallel for reduction(+:numberOfCommonCenters)
    for (size_t i = 0; i < minSize; ++i) {
        if (i < centrality1.size() && i < centrality2.size() && centrality1[i] == centrality2[i]){
#pragma omp critical
            {
                //std::cout << i << "\t" << centrality1[i] << "\t\t\t" << centrality2[i] << std::endl;
            }
            numberOfCommonCenters++;
        }
    }
    std::cout << "The number of common centers: " << numberOfCommonCenters << std::endl;
    std::cout << "Total number of centers: " << centrality1.size()+centrality2.size() << std::endl;
    commonCentersPercentage = static_cast<double>(numberOfCommonCenters) / (centrality1.size()+centrality2.size());
    std::cout << "Percentage of common centers: " << commonCentersPercentage*100 << "% " << std::ends;
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

void runGraphComparisonAnalysis(std::string& filenameAlpha,std::string& filenameBeta){
    std::cout << "Comparing: " << filenameAlpha << " with " << filenameBeta << std::endl;
    std::vector<MatrixEntry> alphaMatrix, betaMatrix;
    int alphaRows, alphaCols, alphaEntries;
    int betaRows, betaCols, betaEntries;
    std::ifstream file1(filenameAlpha);
    std::ifstream file2(filenameBeta);
    if (!file1.is_open() || !file2.is_open()) {
        std::cerr << "Error: Could not open files " << filenameAlpha << " or " << filenameBeta << std::endl;
        return;
    }
    auto startTimeRead1 = std::chrono::high_resolution_clock::now();
    readMatrixFile(filenameAlpha, alphaMatrix, alphaRows, alphaCols, alphaEntries);
    auto endTimeRead1 = std::chrono::high_resolution_clock::now();
    auto startTimeRead2 = std::chrono::high_resolution_clock::now();
    readMatrixFile(filenameBeta, betaMatrix, betaRows, betaCols, betaEntries);
    auto endTimeRead2 = std::chrono::high_resolution_clock::now();

    auto durationRead1 = std::chrono::duration_cast<std::chrono::microseconds>(endTimeRead1 - startTimeRead1);
    std::cout << "alpha Matrix Entries: " << alphaEntries << "\ttook: "<< durationRead1.count() << " microseconds" << std::endl;
    auto durationRead2 = std::chrono::duration_cast<std::chrono::microseconds>(endTimeRead2 - startTimeRead2);
    std::cout << "beta Matrix Entries: " << betaEntries << "\ttook: " << durationRead2.count() << " microseconds" << std::endl;
    double totalNodeNumber = alphaMatrix.size()*betaMatrix.size();
    double overlappingNeighborhood = 0.0;
    double similarityNumber = 0.0;

    // Define weights for each metric
    const double cosineSimilarityWeight = 0.4;
    const double commonCentersWeight = 0.15;
    const double overlappingNeighborhoodWeight = 0.15;
    const double graphDistanceWeight = 0.3;

    std::vector<std::unordered_set<int>> alphaNeighbors(alphaRows);
    std::vector<std::unordered_set<int>> betaNeighbors(betaRows);
    std::vector<double> alphaCentrality(alphaRows, 0.0);
    std::vector<double> betaCentrality(betaRows, 0.0);
    auto startTimeExtractEdges1 = std::chrono::high_resolution_clock::now();
    std::unordered_set<std::pair<int, int>> alphaEdges(extractEdges(alphaMatrix));
    auto endTimeExtractEdges1 = std::chrono::high_resolution_clock::now();
    auto startTimeExtractEdges2 = std::chrono::high_resolution_clock::now();
    std::unordered_set<std::pair<int, int>> betaEdges(extractEdges(betaMatrix));
    auto endTimeExtractEdges2 = std::chrono::high_resolution_clock::now();

    auto durationExtractionEdges1 = std::chrono::duration_cast<std::chrono::microseconds>(endTimeExtractEdges1 - startTimeExtractEdges1);
    auto durationExtractionEdges2 = std::chrono::duration_cast<std::chrono::microseconds>(endTimeExtractEdges2 - startTimeExtractEdges2);

    int alphaEdgesSize = alphaEdges.size();
    int betaEdgesSize  = betaEdges.size();
    int totalEdgesSize = alphaEdgesSize+betaEdgesSize;
    int totalGraphEditDifferences = graphEditDistance(alphaEdges,betaEdges);
    double graphEditDifferencePercentage = static_cast<double>(totalGraphEditDifferences)*100 / totalEdgesSize;



#pragma omp parallel sections
    {
#pragma omp section
        {
            auto startTimeFunction1 = std::chrono::high_resolution_clock::now();
            populateNeighborSets(alphaMatrix, alphaNeighbors, alphaRows, alphaCols);
            auto endTimeFunction1 = std::chrono::high_resolution_clock::now();
            auto durationFunction1 = std::chrono::duration_cast<std::chrono::microseconds>(endTimeFunction1 - startTimeFunction1);
            std::cout << "populateNeighborSets for alphaMatrix \ttook: " << durationFunction1.count() << " microseconds." << std::endl;

        }
#pragma omp section
        {
            auto startTimeFunction2 = std::chrono::high_resolution_clock::now();
            populateNeighborSets(betaMatrix, betaNeighbors, betaRows, betaCols);
            auto endTimeFunction2 = std::chrono::high_resolution_clock::now();
            auto durationFunction2 = std::chrono::duration_cast<std::chrono::microseconds>(endTimeFunction2 - startTimeFunction2);
            std::cout << "populateNeighborSets for betaMatrix \ttook: " << durationFunction2.count() << " microseconds." << std::endl;
        }
    }
    auto startTimeCosineSimilarity = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<double>> similarityMatrix = calculateCosineSimilarityMatrix(alphaNeighbors, betaNeighbors);

// Display similarity matrix
#pragma omp parallel for reduction(+:similarityNumber)
    for (size_t i = 0; i < similarityMatrix.size(); ++i) {
        for (size_t j = 0; j < similarityMatrix[i].size(); ++j) {
            if (similarityMatrix[i][j] != 0) {
                similarityNumber += similarityMatrix[i][j];
            }
        }
    }
    auto endTimeCosineSimilarity = std::chrono::high_resolution_clock::now();
    auto durationCosineSimilarity = std::chrono::duration_cast<std::chrono::microseconds>(endTimeCosineSimilarity - startTimeCosineSimilarity);
    std::cout << "Total Cosine similarity is " << (similarityNumber/(betaEntries*alphaEntries))*100 << "% \ttook: " << durationCosineSimilarity.count() << " microseconds"<< std::endl;

    auto startTimeDegreeCentrality = std::chrono::high_resolution_clock::now();
// Calculate degree centrality for alpha graph
    calculateDegreeCentrality(alphaNeighbors, alphaCentrality);
// Calculate degree centrality for beta graph
    calculateDegreeCentrality(betaNeighbors, betaCentrality);
// Compare centrality
    compareCentrality(alphaCentrality, betaCentrality);
    auto endTimeDegreeCentrality = std::chrono::high_resolution_clock::now();
    auto durationDegreeCentrality = std::chrono::duration_cast<std::chrono::microseconds>(endTimeDegreeCentrality - startTimeDegreeCentrality);
    std::cout<<"took: "<< durationDegreeCentrality.count()<<" microseconds"<<std::endl;
    // Calculate and print neighborhood overlap for all nodes in both matrices
    auto startTimeOverlappingNeighborhood = std::chrono::high_resolution_clock::now();
#pragma omp parallel for reduction(+:overlappingNeighborhood)
    for (size_t i = 0; i < alphaMatrix.size(); ++i) {
#pragma omp parallel for reduction(+:overlappingNeighborhood)
        for (size_t j = 0; j < betaMatrix.size(); ++j) {
            double overlap = neighborhoodOverlap(alphaMatrix, betaMatrix, i, j);
            if(overlap!=0){
#pragma omp critical
                //std::cout << "Neighborhood overlap between node " << i << " in 1st Matrix and node " << j << " in 2nd Matrix: " << overlap << std::endl;
                overlappingNeighborhood+=overlap;
            }

        }
    }
    auto endTimeOverlappingNeighborhood = std::chrono::high_resolution_clock::now();
    auto durationOverlappingNeighborhood = std::chrono::duration_cast<std::chrono::milliseconds>(endTimeOverlappingNeighborhood - startTimeOverlappingNeighborhood);

    double cosineSimilarity = (similarityNumber / (betaEntries * alphaEntries)) * 100;
    double overlappingNeighborhoodPercentage = (overlappingNeighborhood / totalNodeNumber) * 100;
    double aggregateResemblanceScore = (cosineSimilarity * cosineSimilarityWeight) +
                                       (commonCentersPercentage * commonCentersWeight) +
                                       (overlappingNeighborhoodPercentage * overlappingNeighborhoodWeight) +
                                       (graphEditDifferencePercentage*graphDistanceWeight);
    std::cout << "Total overlapping neighborhood: " << overlappingNeighborhood << std::endl;
    std::cout << "Total nodes compared: " << totalNodeNumber << std::endl;
    std::cout <<"Total percentage of overlapping: "<< (overlappingNeighborhood/totalNodeNumber)*100 <<"%"<<std::endl;
    std::cout <<"Graph Edit Distance:" << totalGraphEditDifferences <<std::endl;
    std::cout <<"Number of Edges for Matrix 1:" << alphaEdgesSize << "\ttook: " << durationExtractionEdges1.count() <<" microseconds"<<std::endl;
    std::cout <<"Number of Edges for Matrix 2:" << betaEdgesSize << "\ttook: " << durationExtractionEdges2.count() << " microseconds"<<std::endl;
    std::cout <<"Total number of Edges:" <<totalEdgesSize <<std::endl;
    std::cout << "Percentage of Graph Edit difference percentage:"<<graphEditDifferencePercentage<<"%"<<std::endl;
    std::cout << "Cosine Similarity Weight: " << cosineSimilarityWeight << std::endl;
    std::cout << "Common Centers Weight: " << commonCentersWeight << std::endl;
    std::cout << "Overlapping Neighborhood Weight: " << overlappingNeighborhoodWeight << "\ttook: " << durationOverlappingNeighborhood.count() << " microseconds" << std::endl;
    std::cout << "Graph Distance Weight: " << graphDistanceWeight << std::endl;
    std::cout << "Aggregate Resemblance Score: " << aggregateResemblanceScore << "%" << std::endl;
}


int main() {
    std::string stringArray[] = {"C:\\Users\\jbouv\\OneDrive\\Desktop\\Ecole\\ISEP\\Cycle_ingenieur\\A3\\ITU\\Classes\\Parallel Algorithms\\Project\\Project_Code\\data\\test1.mtx",
                                 "C:\\Users\\jbouv\\OneDrive\\Desktop\\Ecole\\ISEP\\Cycle_ingenieur\\A3\\ITU\\Classes\\Parallel Algorithms\\Project\\Project_Code\\data\\test2.mtx",
                                 "C:\\Users\\jbouv\\OneDrive\\Desktop\\Ecole\\ISEP\\Cycle_ingenieur\\A3\\ITU\\Classes\\Parallel Algorithms\\Project\\Project_Code\\data\\road-euroroad.edges",
                                 "C:\\Users\\jbouv\\OneDrive\\Desktop\\Ecole\\ISEP\\Cycle_ingenieur\\A3\\ITU\\Classes\\Parallel Algorithms\\Project\\Project_Code\\data\\road-minnesota.mtx"};
    // Compare each pair of strings exactly once
    for (std::size_t i = 0; i < std::size(stringArray); ++i) {
        for (std::size_t j = i + 1; j < std::size(stringArray); ++j) {
            runGraphComparisonAnalysis(stringArray[i], stringArray[j]);
        }
    }
    return 0;
}

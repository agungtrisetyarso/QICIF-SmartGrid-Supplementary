#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <map>

// Simple structure for one time step
struct FreqSample {
    std::string timestamp;
    std::map<std::string, double> freq_mHz;   // location → deviation in mHz
    std::map<std::string, int>    qi;         // location → quality indicator
};

// ------------------------------------------------------------
// 1. Load the CSV (semicolon-separated)
// ------------------------------------------------------------
std::vector<FreqSample> load_sync01(const std::string& filename) {
    std::vector<FreqSample> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open " << filename << std::endl;
        return data;
    }

    std::string line;
    // Read header
    std::getline(file, line);
    std::stringstream header_ss(line);
    std::string col;
    std::vector<std::string> headers;
    while (std::getline(header_ss, col, ';')) {
        // remove possible quotes or spaces
        col.erase(0, col.find_first_not_of(" \t\""));
        col.erase(col.find_last_not_of(" \t\"") + 1);
        headers.push_back(col);
    }

    // Identify frequency and QI columns
    std::vector<std::pair<std::string, size_t>> freq_cols; // location name → column index
    std::vector<std::pair<std::string, size_t>> qi_cols;

    for (size_t i = 1; i < headers.size(); ++i) {
        if (headers[i].find("f50_") == 0 || headers[i].find("f60_") == 0) {
            std::string loc = headers[i].substr(4); // remove "f50_" or "f60_"
            freq_cols.emplace_back(loc, i);
        } else if (headers[i].find("QI_") == 0) {
            std::string loc = headers[i].substr(3);
            qi_cols.emplace_back(loc, i);
        }
    }

    std::cout << "Detected locations: ";
    for (const auto& p : freq_cols) std::cout << p.first << " ";
    std::cout << "\n";

    // Read data rows
    size_t line_count = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(ss, token, ';')) {
            tokens.push_back(token);
        }

        if (tokens.size() < 2) continue;

        FreqSample sample;
        sample.timestamp = tokens[0];

        for (const auto& [loc, idx] : freq_cols) {
            if (idx < tokens.size() && !tokens[idx].empty() && tokens[idx] != "NaN") {
                try {
                    sample.freq_mHz[loc] = std::stod(tokens[idx]);
                } catch (...) {
                    sample.freq_mHz[loc] = std::nan("");
                }
            } else {
                sample.freq_mHz[loc] = std::nan("");
            }
        }

        for (const auto& [loc, idx] : qi_cols) {
            if (idx < tokens.size() && !tokens[idx].empty()) {
                try {
                    sample.qi[loc] = std::stoi(tokens[idx]);
                } catch (...) {
                    sample.qi[loc] = -1;
                }
            }
        }

        data.push_back(sample);
        ++line_count;
        if (line_count % 100000 == 0) {
            std::cout << "Loaded " << line_count << " samples...\r" << std::flush;
        }
    }

    std::cout << "\nTotal samples loaded: " << data.size() << std::endl;
    return data;
}

// ------------------------------------------------------------
// 2. Basic statistics (useful for QICIF / damping studies)
// ------------------------------------------------------------
void compute_basic_stats(const std::vector<FreqSample>& data) {
    if (data.empty()) return;

    // Collect all location names
    std::vector<std::string> locations;
    for (const auto& p : data[0].freq_mHz) {
        locations.push_back(p.first);
    }

    for (const auto& loc : locations) {
        std::vector<double> vals;
        for (const auto& s : data) {
            double v = s.freq_mHz.at(loc);
            if (!std::isnan(v)) vals.push_back(v);
        }

        if (vals.empty()) continue;

        double mean = std::accumulate(vals.begin(), vals.end(), 0.0) / vals.size();
        double sq_sum = 0.0;
        for (double v : vals) sq_sum += (v - mean) * (v - mean);
        double stddev = std::sqrt(sq_sum / vals.size());

        auto [min_it, max_it] = std::minmax_element(vals.begin(), vals.end());

        std::cout << "\n=== " << loc << " ===\n";
        std::cout << "  Valid samples : " << vals.size() << "\n";
        std::cout << "  Mean (mHz)    : " << std::fixed << std::setprecision(3) << mean << "\n";
        std::cout << "  Std Dev (mHz) : " << stddev << "\n";
        std::cout << "  Min / Max     : " << *min_it << " / " << *max_it << "\n";
    }
}

// ------------------------------------------------------------
// 3. Simple RoCoF (Rate of Change of Frequency) calculation
//    Important for low-inertia / damping analysis
// ------------------------------------------------------------
void compute_rocof(const std::vector<FreqSample>& data, const std::string& location, int window = 1) {
    std::cout << "\n=== RoCoF for " << location << " (window = " << window << " s) ===\n";

    std::vector<double> rocof;
    for (size_t i = window; i < data.size(); ++i) {
        double f1 = data[i - window].freq_mHz.at(location);
        double f2 = data[i].freq_mHz.at(location);
        if (!std::isnan(f1) && !std::isnan(f2)) {
            // convert mHz → Hz, then / Δt
            double df_hz = (f2 - f1) / 1000.0;
            rocof.push_back(df_hz / window);   // Hz/s
        }
    }

    if (rocof.empty()) return;

    double mean = std::accumulate(rocof.begin(), rocof.end(), 0.0) / rocof.size();
    double max_abs = 0.0;
    for (double r : rocof) max_abs = std::max(max_abs, std::abs(r));

    std::cout << "  Mean RoCoF   : " << mean << " Hz/s\n";
    std::cout << "  Max |RoCoF|  : " << max_abs << " Hz/s\n";
}

// ------------------------------------------------------------
// 4. Main
// ------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string filename = "data/SYNC01.csv";
    if (argc > 1) filename = argv[1];

    std::cout << "Loading " << filename << " ...\n";
    auto data = load_sync01(filename);

    if (data.empty()) {
        std::cerr << "No data loaded. Check file path and format.\n";
        return 1;
    }

    // Basic statistics (good starting point for your QICIF novelty)
    compute_basic_stats(data);

    // Example RoCoF for the first detected location
    if (!data[0].freq_mHz.empty()) {
        std::string first_loc = data[0].freq_mHz.begin()->first;
        compute_rocof(data, first_loc, 1);   // 1-second RoCoF
        compute_rocof(data, first_loc, 5);   // 5-second smoothed RoCoF
    }

    std::cout << "\nProcessing finished.\n";
    return 0;
}

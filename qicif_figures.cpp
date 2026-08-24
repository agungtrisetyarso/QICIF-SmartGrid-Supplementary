// qicif_figures.cpp
// Generates data for QICIF placeholder figures (esp. f5 / 3D trajectories,
// eigenvalue migration, time-domain envelopes) matching the manuscript model.
// Compile: g++ -std=c++17 -O2 qicif_figures.cpp -o qicif_figures
// Run:     ./qicif_figures

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>

using namespace std;

// Parameters chosen to reproduce Table 4 frequencies / damping ratios
const double OMEGA0 = 4.08;          // rad/s  (~0.65 Hz)
const double L      = 1.0;           // normalized inductance
const double C      = 1.0 / (OMEGA0 * OMEGA0);
const double R0     = 0.005;         // baseline (very small)

// R values from the paper
const vector<double> R_VALUES = {0.00, 0.04, 0.08};
const vector<string> R_LABELS = {"Baseline_R0", "Moderate_R004", "High_R008"};

// Damping ratio ζ = R/2 * sqrt(C/L)
double zeta(double R) {
    return (R / 2.0) * sqrt(C / L);
}

// Real and imag parts of dominant eigenvalue (matches Table 4 style)
pair<double,double> eigenvalue(double R) {
    double z = zeta(R);
    double real = -z * OMEGA0;
    double imag = OMEGA0 * sqrt(max(0.0, 1.0 - z*z));
    return {real, imag};
}

// Analytical solution for verification (under-damped)
double analytical_Q(double t, double R, double A = 1.0, double phi = 0.0) {
    double z = zeta(R);
    double wd = OMEGA0 * sqrt(max(0.0, 1.0 - z*z));
    return A * exp(-z * OMEGA0 * t) * sin(wd * t + phi);
}

// RK4 step for state [Q, dQ/dt]
void rk4_step(double& Q, double& dQ, double R, double dt) {
    auto deriv = [&](double q, double dq) {
        double d2q = -(R/L)*dq - (1.0/(L*C))*q;
        return make_pair(dq, d2q);
    };

    auto k1 = deriv(Q, dQ);
    auto k2 = deriv(Q + 0.5*dt*k1.first, dQ + 0.5*dt*k1.second);
    auto k3 = deriv(Q + 0.5*dt*k2.first, dQ + 0.5*dt*k2.second);
    auto k4 = deriv(Q + dt*k3.first,     dQ + dt*k3.second);

    Q  += (dt/6.0)*(k1.first  + 2*k2.first  + 2*k3.first  + k4.first);
    dQ += (dt/6.0)*(k1.second + 2*k2.second + 2*k3.second + k4.second);
}

int main() {
    cout << fixed << setprecision(6);
    cout << "QICIF figure data generator (C++)\n";
    cout << "L = " << L << ", C = " << C << ", omega0 = " << OMEGA0 << " rad/s\n\n";

    // ------------------------------------------------------------------
    // 1. Eigenvalue migration data (for Figure 2 style plot)
    // ------------------------------------------------------------------
    ofstream eig("eigenvalue_migration.csv");
    eig << "R,real,imag,frequency_Hz,damping_ratio_zeta\n";
    for (double R : R_VALUES) {
        auto [re, im] = eigenvalue(R);
        double f = im / (2.0 * M_PI);
        double z = zeta(R);
        eig << R << "," << re << "," << im << "," << f << "," << z << "\n";
        cout << "R=" << R << "  eigenvalue = " << re << " ± j" << im
             << "  f=" << f << " Hz  zeta=" << z << "\n";
    }
    eig.close();
    cout << "Wrote eigenvalue_migration.csv\n\n";

    // ------------------------------------------------------------------
    // 2. Time-domain envelopes / trajectories (for Figure 5 and PMU-style plots)
    // ------------------------------------------------------------------
    const double T_END = 300.0;   // seconds (paper horizon)
    const double DT    = 0.01;
    const int    N     = static_cast<int>(T_END / DT) + 1;

    for (size_t i = 0; i < R_VALUES.size(); ++i) {
        double R = R_VALUES[i];
        string fname = "time_domain_" + R_LABELS[i] + ".csv";
        ofstream out(fname);
        out << "t,Q,dQ,envelope,analytical_Q\n";

        double Q  = 1.0;          // initial deviation
        double dQ = 0.0;
        for (int k = 0; k < N; ++k) {
            double t = k * DT;
            double env = abs(Q);  // simple envelope proxy
            double qa  = analytical_Q(t, R);
            out << t << "," << Q << "," << dQ << "," << env << "," << qa << "\n";
            rk4_step(Q, dQ, R, DT);
        }
        out.close();
        cout << "Wrote " << fname << "\n";
    }

    // ------------------------------------------------------------------
    // 3. 3-D oscillatory trajectories in normalized labor-capital-stability space
    //    (replacement for placeholder f5 / Figure 1)
    //    Labor   ~ normalized human/adaptive capacity proxy (related to C / reskilling)
    //    Capital ~ automation / inductive flow proxy
    //    Stability ~ instantaneous damping effectiveness (higher R → faster decay)
    // ------------------------------------------------------------------
    ofstream traj3d("trajectories_3d_labor_capital_stability.csv");
    traj3d << "t,Labor,Capital,Stability,R,trajectory_id\n";

    // Several initial conditions + three R values → families of trajectories
    vector<pair<double,double>> inits = {
        {1.0, 0.0}, {0.8, 0.3}, {0.6, -0.2}, {1.2, 0.1}, {0.5, 0.4}
    };

    int traj_id = 0;
    for (double R : R_VALUES) {
        for (auto [Q0, dQ0] : inits) {
            double Q  = Q0;
            double dQ = dQ0;
            for (int k = 0; k < N; k += 5) {          // subsample for smaller file
                double t = k * DT;
                // Normalized coordinates (illustrative mapping consistent with paper spirit)
                double Labor     = 0.5 + 0.4 * Q + 0.1 * R;               // human capacity axis
                double Capital   = 0.5 + 0.3 * dQ - 0.15 * R;             // automation axis
                double Stability = 1.0 - exp(-zeta(R) * OMEGA0 * t);      // approaches 1 under higher R
                traj3d << t << "," << Labor << "," << Capital << "," << Stability
                       << "," << R << "," << traj_id << "\n";
                rk4_step(Q, dQ, R, DT);
            }
            ++traj_id;
        }
    }
    traj3d.close();
    cout << "Wrote trajectories_3d_labor_capital_stability.csv  (use for f5 / Fig.1 style 3-D plot)\n\n";

    // ------------------------------------------------------------------
    // 4. Summary damping improvement (for bar charts / right panel of Fig.2)
    // ------------------------------------------------------------------
    ofstream damp("damping_improvement.csv");
    damp << "R,zeta,improvement_percent_vs_baseline\n";
    double z0 = zeta(0.0);
    for (double R : R_VALUES) {
        double z = zeta(R);
        double imp = (z0 > 1e-9) ? 100.0 * (z - z0) / z0 : 0.0;
        damp << R << "," << z << "," << imp << "\n";
    }
    damp.close();
    cout << "Wrote damping_improvement.csv\n";

    cout << "\nAll CSV files generated successfully.\n"
         << "Next: plot them (see suggested Python / gnuplot snippets below).\n";
    return 0;
}

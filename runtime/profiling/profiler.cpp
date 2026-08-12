#include "profiler.h"

#include <algorithm>
#include <iomanip>

namespace rt::profiling {

void Profiler::print_summary(std::ostream& os, const std::string& title) const {
    if (!title.empty()) os << title << "\n";

    double grand_total = 0.0;
    for (const auto& [label, stats] : stats_) grand_total += stats.total_ms;

    std::vector<std::pair<std::string, OpStats>> sorted(stats_.begin(), stats_.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second.total_ms > b.second.total_ms; });

    os << std::left << std::setw(20) << "label" << std::right << std::setw(10) << "calls"
       << std::setw(14) << "total_ms" << std::setw(12) << "avg_ms" << std::setw(8) << "%\n";
    for (const auto& [label, stats] : sorted) {
        double pct = grand_total > 0.0 ? 100.0 * stats.total_ms / grand_total : 0.0;
        os << std::left << std::setw(20) << label << std::right << std::setw(10) << stats.calls
           << std::setw(14) << std::fixed << std::setprecision(4) << stats.total_ms
           << std::setw(12) << std::setprecision(5) << stats.avg_ms() << std::setw(7)
           << std::setprecision(1) << pct << "%\n";
    }
    os << std::left << std::setw(20) << "TOTAL" << std::right << std::setw(10) << ""
       << std::setw(14) << std::fixed << std::setprecision(4) << grand_total << "\n";
}

}  // namespace rt::profiling

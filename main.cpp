#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

class RandomStreamGen {
  std::string allowed_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-";
  std::mt19937 gen;
  std::uniform_int_distribution<> len_dis;
  std::uniform_int_distribution<> char_dis;

public:
  RandomStreamGen() : gen(std::random_device{}()), len_dis(1, 30), char_dis(0, allowed_chars.size() - 1) {
  }

  RandomStreamGen(uint32_t seed) : gen(seed), len_dis(1, 30), char_dis(0, allowed_chars.size() - 1) {
  }

  std::stringstream GenerateStream(int length) {
    std::stringstream stream{};
    stream.clear();
    for (int i = 0; i < length; ++i) {
      int word_len = len_dis(gen);
      for (int j = 0; j < word_len; ++j) {
        stream << allowed_chars[char_dis(gen)];
      }
      stream << '\n';
    }
    return stream;
  }

  std::vector<std::stringstream> GenerateSplittedStream(int length, int parts) {
    std::vector<std::stringstream> streams;
    if (parts <= 0) {
      return streams;
    }
    streams.reserve(parts);
    int base = length / parts;
    int rem = length % parts;
    for (int i = 0; i < parts; ++i) {
      int chunk = base + (i < rem ? 1 : 0);
      streams.emplace_back(GenerateStream(chunk));
    }
    return streams;
  }
};

class HashFuncGen {
public:
  // функция была проверена на равномерность при сдаче дз 23.01.2025 + Mix равномерности хуже не делает
  uint32_t strFoldAndMix(std::string key, uint32_t n, uint32_t m) {
    uint64_t s = 0;
    uint64_t power = 1;
    for (size_t i = 0; i < key.size(); ++i) {
      s += key[i] * power;
      s %= m;
      if (i % n == n - 1) {
        power = 1;
      } else {
        power = (power * 256) % m;
      }
    }
    uint32_t h = static_cast<uint32_t>(s) % m;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  }
};

class HyperLogLog {
  int B;
  size_t q;
  std::vector<uint32_t> reg;
  HashFuncGen hasher;

  static int CountLeadingZeros(uint32_t x) {
    if (x == 0) {
      return 32;
    }
    int n = 0;
    uint64_t mask = 1ull << 31;
    while ((x & mask) == 0) {
      ++n;
      mask >>= 1ull;
    }
    return n;
  }

  double Alpha() const {
    if (q == 16) return 0.673;
    if (q == 32) return 0.697;
    if (q == 64) return 0.709;
    return 0.7213 / (1.0 + 1.079 / static_cast<double>(q));
  }

public:
  explicit HyperLogLog(int precision_bits)
      : B(precision_bits), q(1ull << precision_bits), reg(q, 0) {
  }

  void Add(const std::string &key) {
    uint32_t hash = hasher.strFoldAndMix(key, 4, (1ull << 32) - 5);
    
    uint32_t index = hash & static_cast<uint32_t>(q - 1);
    uint32_t w = hash >> B;

    int rank = 0;
    if (w == 0) {
      rank = 32 - B + 1;
    } else {
      rank = CountLeadingZeros(w) - B + 1;
    }

    if (reg[index] < rank) {
      reg[index] = static_cast<uint32_t>(rank);
    }
  }

  double Estimate() const {
    double z = 0.0;
    int zeros = 0;
    for (uint32_t v : reg) {
      z += std::pow(2.0, -static_cast<int>(v));
      if (v == 0) {
        ++zeros;
      }
    }
    double estimate = Alpha() * static_cast<double>(q) * static_cast<double>(q) / z;

    /*// small-range correction (linear counting)
    if (estimate <= 2.5 * static_cast<double>(q) && zeros > 0) {
      estimate = static_cast<double>(q) * std::log(static_cast<double>(q) / static_cast<double>(zeros));
    } else {
      // large-range correction for 32-bit hash space
      const double two32 = std::pow(2.0, 32.0);
      if (estimate > (two32 / 30.0)) {
        estimate = -two32 * std::log(1.0 - estimate / two32);
      }
    }*/
    return estimate;
  }
};

std::vector<std::string> ReadLines(std::stringstream &stream) {
  std::vector<std::string> items;
  std::string line;
  while (std::getline(stream, line)) {
    items.push_back(line);
  }
  return items;
}

std::vector<int> BuildPrefixSizes(int length, int parts) {
  std::vector<int> sizes;
  sizes.reserve(parts);
  if (parts <= 0) {
    return sizes;
  }
  int base = length / parts;
  int rem = length % parts;
  int acc = 0;
  for (int i = 0; i < parts; ++i) {
    acc += base + (i < rem ? 1 : 0);
    sizes.push_back(acc);
  }
  return sizes;
}

int main() {
  int streams_count = 5;
  int stream_length = 100000;
  int parts = 10;
  int precision_bits = 6;
  int seed = 4011505;

  RandomStreamGen gen(static_cast<uint32_t>(seed));

  std::vector<int> prefix_sizes = BuildPrefixSizes(stream_length, parts);

  std::vector<double> sum_est(parts, 0.0);
  std::vector<double> sum_sq_est(parts, 0.0);

  std::ofstream per_stream_csv("per_stream.csv");
  per_stream_csv << "stream_id,step,prefix_size,exact,estimate\n";

  for (int s = 0; s < streams_count; ++s) {
    std::vector<std::stringstream> stream_parts = gen.GenerateSplittedStream(stream_length, parts);

    HyperLogLog hll(precision_bits);
    std::unordered_set<std::string> exact_set;

    int processed = 0;
    for (int step = 0; step < parts; ++step) {
      std::vector<std::string> items = ReadLines(stream_parts[step]);
      for (const std::string &key : items) {
        exact_set.insert(key);
        hll.Add(key);
        ++processed;
      }
      int target = prefix_sizes[step];
      double estimate = hll.Estimate();
      int exact = static_cast<int>(exact_set.size());

      per_stream_csv << s << ',' << (step + 1) << ',' << target << ',' << exact << ',' << estimate << '\n';

      sum_est[step] += estimate;
      sum_sq_est[step] += estimate * estimate;
    }
  }

  std::ofstream stats_csv("stats.csv");
  stats_csv << "step,prefix_size,mean_estimate,stddev_estimate,mean_plus_std,mean_minus_std\n";

  for (int step = 0; step < parts; ++step) {
    double mean = sum_est[step] / static_cast<double>(streams_count);
    double variance = 0.0;
    if (streams_count > 1) {
      variance = (sum_sq_est[step] - (sum_est[step] * sum_est[step]) / streams_count) /
                 static_cast<double>(streams_count - 1);
      if (variance < 0.0) {
        variance = 0.0;
      }
    }
    double stddev = std::sqrt(variance);
    double mean_plus = mean + stddev;
    double mean_minus = mean - stddev;
    stats_csv << (step + 1) << ',' << prefix_sizes[step] << ',' << mean << ',' << stddev << ','
              << mean_plus << ',' << mean_minus << '\n';
  }

  std::cout << "Done. Wrote per_stream.csv and stats.csv\n";
  return 0;
}
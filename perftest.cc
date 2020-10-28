#include <iostream>
#include <random>
#include <vector>
#include <thread>
#include <algorithm>
#include <pthread.h>
#include <time.h>
#include <gflags/gflags.h>

#include "config.h"
#include "compiler.hh"

#include "masstree.hh"
#include "kvthread.hh"
#include "masstree_tcursor.hh"
#include "masstree_insert.hh"
#include "masstree_print.hh"
#include "masstree_remove.hh"
#include "masstree_scan.hh"
#include "masstree_stats.hh"
#include "string.hh"

#define NUM_THREADS 4

using gflags::RegisterFlagValidator;

DEFINE_string(benchmarks, "fillrandom",
              "Comma-separated list of benchmarks to run. Options:\n"
              "\tfillrandom             -- write N random values\n"
              "\tfillseq                -- write N values in sequential order\n"
              "\tconcurrentfillonlyrandom -- one or more write threads write N values concurrently\n"
              "\treadrandom             -- read N values in random order\n"
              "\treadseq                -- scan the DB\n"
              "\treadwrite              -- 1 thread writes while N - 1 threads "
              "do random\n"
              "\t                          reads\n"
              "\tseqreadwrite           -- 1 thread writes while N - 1 threads "
              "do scans\n");

DEFINE_int32(
    num_threads, 4,
    "Number of concurrent threads to run. If the benchmark includes writes,\n"
    "then at most one thread will be a writer");

DEFINE_uint32(num_operations, 1000000,
             "Number of operations to do for write and random read benchmarks");

DEFINE_int32(num_scans, 1,
             "Number of times for each thread to scan the inlineskiplist for "
             "sequential read "
             "benchmarks");

DEFINE_int32(item_size, 8, "Number of bytes each item should be");

DEFINE_int64(seed, 0,
             "Seed base for random number generators. "
             "When 0 it is deterministic.");

class key_unparse_unsigned {
public:
    static int unparse_key(Masstree::key<uint64_t> key, char* buf, int buflen) {
        return snprintf(buf, buflen, "%" PRIu64, key.ikey());
    }
};

enum KeygenMode { SEQUENTIAL, RANDOM, UNIQUE_RANDOM };

// A good 64-bit random number generator based on std::mt19937_64
class Random64 {
 private:
  std::mt19937_64 generator_;

 public:
  explicit Random64(uint64_t s) : generator_(s) { }

  // Generates the next random number
  uint64_t Next() { return generator_(); }

  // Returns a uniformly distributed value in the range [0..n-1]
  // REQUIRES: n > 0
  uint64_t Uniform(uint64_t n) {
    return std::uniform_int_distribution<uint64_t>(0, n - 1)(generator_);
  }

  // Randomly returns true ~"1/n" of the time, and false otherwise.
  // REQUIRES: n > 0
  bool OneIn(uint64_t n) { return Uniform(n) == 0; }

  // Skewed: pick "base" uniformly from range [0,max_log] and then
  // return "base" random bits.  The effect is to pick a number in the
  // range [0,2^max_log-1] with exponential bias towards smaller numbers.
  uint64_t Skewed(int max_log) {
    return Uniform(uint64_t(1) << Uniform(max_log + 1));
  }
};

// A seeded replacement for removed std::random_shuffle
template <class RandomIt>
void RandomShuffle(RandomIt first, RandomIt last, uint32_t seed) {
  std::mt19937 rng(seed);
  std::shuffle(first, last, rng);
}

// A replacement for removed std::random_shuffle
template <class RandomIt>
void RandomShuffle(RandomIt first, RandomIt last) {
  RandomShuffle(first, last, std::random_device{}());
}

int64_t GetSysTimeMicros() {
  timeval tv;
  gettimeofday(&tv, 0);
  return (int64_t)tv.tv_sec * 1000000 + (int64_t)tv.tv_usec;
}

class KeyGenerator {
 public:
  KeyGenerator(Random64* write_rand, Random64* read_rand, KeygenMode write_mode, KeygenMode read_mode, uint64_t num)
      : write_rand_(write_rand), read_rand_(read_rand), write_mode_(write_mode), read_mode_(read_mode), num_(num), write_next_(0), read_next_(0) {
    if ((write_mode_ == UNIQUE_RANDOM) || (read_mode_ == UNIQUE_RANDOM)) {
      values_.resize(num_);
      for (uint64_t i = 0; i < num_; ++i) {
        values_[i] = i;
      }

      RandomShuffle(values_.begin(), values_.end(),
                    static_cast<uint32_t>(FLAGS_seed));
    }
  }

  uint64_t NextForWrite() {
    switch (write_mode_) {
      case SEQUENTIAL:
        return write_next_++;
      case RANDOM:
        return write_rand_->Next() % num_;
      case UNIQUE_RANDOM:
        return values_[write_next_++];
    }
    assert(false);
    return std::numeric_limits<uint64_t>::max();
  }

  uint64_t NextForWrite(uint64_t index ) {
    switch (write_mode_) {
      case UNIQUE_RANDOM:
        return values_[index];
      default:
        assert(false);
        return std::numeric_limits<uint64_t>::max();
    }
  }
  
  uint64_t NextForRead() {
    switch (read_mode_) {
      case SEQUENTIAL:
        return read_next_++;
      case RANDOM:
        return read_rand_->Next() % num_;
      case UNIQUE_RANDOM:
        return values_[read_next_++];
    }
    assert(false);
    return std::numeric_limits<uint64_t>::max();
  }

 private:
  Random64* write_rand_;
  Random64* read_rand_;
  KeygenMode write_mode_;
  KeygenMode read_mode_;
  const uint64_t num_;
  uint64_t write_next_;
  uint64_t read_next_;
  std::vector<uint64_t> values_;
};

class MasstreeWrapper {
public:
    struct table_params : public Masstree::nodeparams<15,15> {
        typedef uint64_t value_type;
        typedef Masstree::value_print<value_type> value_print_type;
        typedef threadinfo threadinfo_type;
        typedef key_unparse_unsigned key_unparse_type;
        static constexpr ssize_t print_max_indent_depth = 12;
    };

    typedef Masstree::Str Str;
    typedef Masstree::basic_table<table_params> table_type;
    typedef Masstree::unlocked_tcursor<table_params> unlocked_cursor_type;
    typedef Masstree::tcursor<table_params> cursor_type;
    typedef Masstree::leaf<table_params> leaf_type;
    typedef Masstree::internode<table_params> internode_type;

    typedef typename table_type::node_type node_type;
    typedef typename unlocked_cursor_type::nodeversion_value_type nodeversion_value_type;

    static __thread typename table_params::threadinfo_type *ti;

    MasstreeWrapper(KeyGenerator* key_gen) : key_gen_(key_gen) {
        this->table_init();
    }

    void table_init() {
        if (ti == nullptr)
            ti = threadinfo::make(threadinfo::TI_MAIN, -1);
        table_.initialize(*ti);
    }

    void keygen_reset(KeyGenerator* key_gen) {
        key_gen_ = key_gen;
    }

    static void thread_init(int thread_id) {
        if (ti == nullptr)
            ti = threadinfo::make(threadinfo::TI_PROCESS, thread_id);
    }

    void insert_test(int thread_id) {
        for (unsigned int i = 0; i < FLAGS_num_operations; i++) {
            auto int_key = key_gen_->NextForWrite();
            uint64_t key_buf;
            Str key = make_key(int_key, key_buf);

            cursor_type lp(table_, key);
            bool found = lp.find_insert(*ti);
            always_assert(!found, "keys should all be unique");

            lp.value() = int_key;

            fence();
            lp.finish(1, *ti);
        }
    }

    void insert_concurrent_test(int thread_id, uint32_t num_operations, uint32_t start) {
        for (unsigned int i = 0; i < num_operations; i++) {
            auto int_key = key_gen_->NextForWrite(start++);
            uint64_t key_buf;
            Str key = make_key(int_key, key_buf);

            cursor_type lp(table_, key);
            bool found = lp.find_insert(*ti);
            always_assert(!found, "keys should all be unique");

            lp.value() = int_key;

            fence();
            lp.finish(1, *ti);
        }
    }

    // random read
    void get_test(int thread_id) {
        unsigned int num_read_operations = FLAGS_num_operations / (FLAGS_num_threads - 1);
        unsigned int read_hits = 0;
        for (unsigned int i = 0; i < num_read_operations; i++) {
            auto int_key = key_gen_->NextForRead();
            uint64_t key_buf;
            Str key = make_key(int_key, key_buf);
            unlocked_cursor_type lp(table_, key);
            bool found = lp.find_unlocked(*ti);
            if (found) {
              read_hits++; 
            }
        }

        double hit_rate = read_hits % num_read_operations;
        std::cout << "Thread " << thread_id << " read hit rate: " << hit_rate << std::endl;
    }

    // sequential read
    void scan_test(int thread_id) {
        unsigned int num_read_operations = FLAGS_num_operations / (FLAGS_num_threads - 1);
        for (int i = 0; i < FLAGS_num_scans; i++) {
            for (unsigned int i = 0; i < num_read_operations; i++) {
                //TODO
            }
        }  
    }

private:
    table_type table_;
    KeyGenerator* key_gen_;
    static bool stopping;
    static uint32_t printing;

    static inline Str make_key(uint64_t int_key, uint64_t& key_buf) {
        key_buf = __builtin_bswap64(int_key);
        return Str((const char *)&key_buf, sizeof(key_buf));
    }
};

__thread typename MasstreeWrapper::table_params::threadinfo_type* MasstreeWrapper::ti = nullptr;
bool MasstreeWrapper::stopping = false;
uint32_t MasstreeWrapper::printing = 0;

volatile mrcu_epoch_type active_epoch = 1;
volatile uint64_t globalepoch = 1;
volatile bool recovering = false;

void insert_thread(MasstreeWrapper* mt, int thread_id) {
    mt->thread_init(thread_id);
    mt->insert_test(thread_id);
}

void insert_concurrent_thread(MasstreeWrapper* mt, int thread_id) {
    unsigned int num_operations_per_thread = (FLAGS_num_operations / FLAGS_num_threads);
    int remaining = (FLAGS_num_operations % FLAGS_num_threads);
    unsigned int start = 0;
    mt->thread_init(thread_id);
    if (thread_id < remaining) {
        start = (num_operations_per_thread + 1) * thread_id;
        mt->insert_concurrent_test(thread_id, (num_operations_per_thread + 1), start);
    } else {
        start = (num_operations_per_thread * thread_id + remaining);
        mt->insert_concurrent_test(thread_id, num_operations_per_thread, start);
    }
}

void get_thread(MasstreeWrapper* mt, int thread_id) {
    mt->thread_init(thread_id);
    mt->get_test(thread_id);
}

void scan_thread(MasstreeWrapper* mt, int thread_id) {
    mt->thread_init(thread_id);
    mt->scan_test(thread_id);
}

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(std::string("\nUSAGE:\n") + std::string(argv[0]) +
                  " [OPTIONS]...");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  Random64 write_rng(FLAGS_seed);
  Random64 read_rng(FLAGS_seed);
  const char* name = FLAGS_benchmarks.c_str();
  std::unique_ptr<KeyGenerator> key_gen;
  int64_t begin, end, duration;
  if (0 == strcmp(name, "fillseq")) {
    std::cout << "Running " << name << std::endl;
    key_gen.reset(new KeyGenerator(
        &write_rng, &read_rng, SEQUENTIAL, SEQUENTIAL, FLAGS_num_operations));
    auto mt = new MasstreeWrapper(key_gen.get());
    begin = GetSysTimeMicros();
    std::vector<std::thread> ths;
    for (int i = 0; i < 1; ++i)
        ths.emplace_back(insert_thread, mt, i);
    for (auto& t : ths)
        t.join();
  } else if (0 == strcmp(name, "fillrandom")) {
    std::cout << "Running " << name << std::endl;
    key_gen.reset(new KeyGenerator(
        &write_rng, &read_rng, UNIQUE_RANDOM, SEQUENTIAL, FLAGS_num_operations));
    auto mt = new MasstreeWrapper(key_gen.get());
    begin = GetSysTimeMicros();
    std::vector<std::thread> ths;
    for (int i = 0; i < 1; ++i)
        ths.emplace_back(insert_thread, mt, i);
    for (auto& t : ths)
        t.join();
  } else if (0 == strcmp(name, "concurrentfillonlyrandom")) {
    std::cout << "Running " << name << std::endl;
    key_gen.reset(new KeyGenerator(
        &write_rng, &read_rng, UNIQUE_RANDOM, SEQUENTIAL, FLAGS_num_operations));
    auto mt = new MasstreeWrapper(key_gen.get());
    begin = GetSysTimeMicros();
    std::vector<std::thread> ths;
    for (int i = 0; i < FLAGS_num_threads; ++i)
        ths.emplace_back(insert_concurrent_thread, mt, i);
    for (auto& t : ths)
        t.join();
  } else if (0 == strcmp(name, "readrandom")) {
    std::cout << "Running " << name << std::endl;
    key_gen.reset(new KeyGenerator(
        &write_rng, &read_rng, SEQUENTIAL, RANDOM, FLAGS_num_operations));
    auto mt = new MasstreeWrapper(key_gen.get());
    // prepare data for masstree
    std::vector<std::thread> wrths;
    for (int i = 0; i < 1; ++i)
        wrths.emplace_back(insert_thread, mt, i);
    for (auto& t : wrths)
        t.join();

    begin = GetSysTimeMicros();
    std::vector<std::thread> rdths;
    for (int i = 1; i < FLAGS_num_threads; ++i)
        rdths.emplace_back(get_thread, mt, i);
    for (auto& t : rdths)
        t.join();
  } else if (0 == strcmp(name, "readseq")) {
    std::cout << "Running " << name << std::endl;
    key_gen.reset(new KeyGenerator(
        &write_rng, &read_rng, SEQUENTIAL, SEQUENTIAL, FLAGS_num_operations));
    auto mt = new MasstreeWrapper(key_gen.get());
    // prepare data for masstree 
    std::vector<std::thread> wrths;
    for (int i = 0; i < 1; ++i)
        wrths.emplace_back(insert_thread, mt, i);
    for (auto& t : wrths)
        t.join();

    begin = GetSysTimeMicros();
    std::vector<std::thread> rdths;
    for (int i = 1; i < FLAGS_num_threads; ++i)
        rdths.emplace_back(scan_thread, mt, i);
    for (auto& t : rdths)
        t.join();
  } else if (0 == strcmp(name, "readwrite")) {
    std::cout << "Running " << name << std::endl;
    // use UNIQUE_RANDOM to avoid the same key insertion which may not be inserted
    key_gen.reset(new KeyGenerator(
        &write_rng, &read_rng, UNIQUE_RANDOM, UNIQUE_RANDOM, FLAGS_num_operations));
    auto mt = new MasstreeWrapper(key_gen.get());
    begin = GetSysTimeMicros();
    std::vector<std::thread> ths;
    for (int i = 0; i < 1; ++i)
        ths.emplace_back(insert_thread, mt, i);

    for (int i = 1; i < FLAGS_num_threads; ++i)
        ths.emplace_back(get_thread, mt, i);
    for (auto& t : ths)
        t.join();
  } else if (0 == strcmp(name, "seqreadwrite")) {
    std::cout << "Running " << name << std::endl;
    // use UNIQUE_RANDOM to avoid the same key insertion which may not be inserted
    key_gen.reset(new KeyGenerator(
        &write_rng, &read_rng, UNIQUE_RANDOM, SEQUENTIAL, FLAGS_num_operations));
    auto mt = new MasstreeWrapper(key_gen.get());
    begin = GetSysTimeMicros();
    std::vector<std::thread> ths;
    for (int i = 0; i < 1; ++i)
        ths.emplace_back(insert_thread, mt, i);
    for (int i = 1; i < FLAGS_num_threads; ++i)
        ths.emplace_back(scan_thread, mt, i);
    for (auto& t : ths)
        t.join();
  } else {
    std::cout << "WARNING: skipping unknown benchmark '" << name
              << std::endl;
    begin = GetSysTimeMicros();
  }

  end = GetSysTimeMicros();
  duration = end - begin;
  std::cout << "test done, elapsed " << duration << " us" << std::endl;
  return 0;
}

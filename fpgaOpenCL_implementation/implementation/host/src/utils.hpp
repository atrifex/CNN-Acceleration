#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <chrono>

#ifdef __GNUC__
#define unused __attribute__((unused))
#else // __GNUC__
#define unused
#endif // __GNUC__

template <typename T>
static bool check_success(const T &err);

template <>
bool check_success<herr_t>(const herr_t &err) {
  const auto res = err >= static_cast<herr_t>(0);
  if (res == true) {
    return res;
  }
  std::cout << "Failed in HDFS..." << std::endl;
  assert(res);
  return res;
}

template <typename T, size_t N>
constexpr size_t array_size(const T (&)[N]) {
  return N;
}

template <typename SzTy, size_t N>
static size_t flattened_length(const SzTy (&idims)[N]) {
  const auto dims = std::valarray<SzTy>(idims, N);
  size_t len      = std::accumulate(std::begin(dims), std::end(dims), 1,
                               std::multiplies<SzTy>());
  return len;
}

template <typename SzTy>
static size_t flattened_length(const SzTy n) {
  return static_cast<size_t>(n);
}

template <typename T, typename SzTy>
static T *allocate(const SzTy len) {
  T *res = new T[len];
  return res;
}

template <typename T, typename SzTy, size_t N>
static T *allocate(const SzTy (&dims)[N]) {
  const auto len = flattened_length(dims);
  return allocate<T>(len);
}

template <typename T, typename SzTy>
static T *zeros(const SzTy len) {
  T *res = allocate<T, SzTy>(len);
  std::fill(res, res + len, static_cast<T>(0));
  return res;
}

template <typename T, typename SzTy, size_t N>
static T *zeros(const SzTy (&dims)[N]) {
  const auto len = flattened_length(dims);
  T *res         = allocate<T, SzTy>(len);
  std::fill(res, res + len, static_cast<T>(0));
  return res;
}

static std::chrono::time_point<std::chrono::high_resolution_clock> now() {
  return std::chrono::high_resolution_clock::now();
}

#endif // __UTILS_HPP__

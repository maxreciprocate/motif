#ifndef QUEUE_H
#define QUEUE_H

#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>

template <class T>
class Queue {
private:
  std::vector<T> queue;
  std::condition_variable takecv;
  std::condition_variable addcv;
  std::mutex mx;
  std::atomic<int> count;
  size_t idx;
  bool done;

public:
  Queue(uint64_t size) : queue(size + 1), idx(0), done(false), count(0) {}

  inline void sub() { ++count; }

  void push(std::string sourcefn);
  void push(T pair);
  T pop();
  void finish();
};

#endif

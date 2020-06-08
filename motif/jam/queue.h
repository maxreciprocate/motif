#ifndef QUEUE_H
#define QUEUE_H

#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>

#define MAX_SOURCE_QUEUE_SIZE 32

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

  void push(std::string sourcefn) {
    std::unique_lock<std::mutex> lock(mx);

    while (idx == queue.capacity() - 1)
      takecv.wait(lock);

    queue[++idx].first = sourcefn;
    readfile(sourcefn, queue[idx].second);

    lock.unlock();
    addcv.notify_one();
  }

  void push(T pair) {
    std::unique_lock<std::mutex> lock(mx);

    while (idx == queue.capacity() - 1)
      takecv.wait(lock);

    queue[++idx] = pair;

    lock.unlock();
    addcv.notify_one();
  }

  T pop() {
    std::unique_lock<std::mutex> lock(mx);

    while (!done && idx == 0)
      addcv.wait(lock);

    if (done && idx == 0) {
      return T();
    }

    auto element = queue[idx--];

    lock.unlock();
    takecv.notify_one();

    return element;
  }

  void finish() {
    if (--count > 0) return;

    done = true;
    addcv.notify_all();
  }
};

#endif

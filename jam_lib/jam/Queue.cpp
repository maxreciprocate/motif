#include "Queue.h"

template <class T>
void Queue<T>::push(std::string sourcefn) {
  std::unique_lock<std::mutex> lock(mx);

  while (idx == queue.capacity() - 1)
    takecv.wait(lock);

  queue[++idx].first = sourcefn;
  readfile(sourcefn, queue[idx].second);

  lock.unlock();
  addcv.notify_one();
}

template <class T>
void Queue<T>::push(T pair) {
  std::unique_lock<std::mutex> lock(mx);

  while (idx == queue.capacity() - 1)
    takecv.wait(lock);

  queue[++idx] = pair;

  lock.unlock();
  addcv.notify_one();
}

template <class T>
T Queue<T>::pop() {
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

template <class T>
void Queue<T>::finish() {
  if (--count > 0) return;

  done = true;
  addcv.notify_all();
}

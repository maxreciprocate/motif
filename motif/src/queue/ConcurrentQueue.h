//
// Created by Denys Maletsden on 08.04.2020.
//

#ifndef _CONCURRENTQUEUE_H
#define _CONCURRENTQUEUE_H
#include <iostream>
#include <thread>
#include <vector>
#include <deque>
#include <queue>
#include <mutex>
#include <condition_variable>

#include "../readers/archive_reader.h"
#include "../readers/file_readers.h"

template <typename T>
class ConcurrentQueue {
private:
    std::queue<T> queue_;
    std::mutex mtx;
    std::condition_variable cv;
public:

    T pop() {
        std::unique_lock<std::mutex> mlock(mtx);
        while (queue_.empty()) {
            cv.wait(mlock);
        }
        auto item = queue_.front();
        queue_.pop();
        return item;
    }

    void push(const T item) {
        std::unique_lock<std::mutex> mlock(mtx);
        queue_.push(item);
        mlock.unlock();
        cv.notify_one();
    }
};

#endif //_CONCURRENTQUEUE_H

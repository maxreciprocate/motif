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

#define MAX_QUEUE_SIZE          (1024ull * 1024ull * 1024ull * 4ull)     // 4GB
#define MAX_ELEMENTS_NUM         20
template <typename T>
class ConcurrentQueue {
private:
    std::queue<T> queue_;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<uint64_t> max_queue_size = MAX_QUEUE_SIZE;
    std::atomic<uint16_t> max_elements_num = MAX_ELEMENTS_NUM;
    std::atomic<uint64_t> queue_size = 0;
    std::condition_variable read_cv;

public:
    void set_queue_limit(uint32_t new_max_queue_size) {
        max_queue_size = new_max_queue_size;
    }
    T pop() {
        std::unique_lock<std::mutex> mlock(mtx);
        while (queue_.empty()) {
            cv.wait(mlock);
        }
        auto item = queue_.front();
        queue_.pop();
        queue_size -= item.content.size();

        mlock.unlock();
        read_cv.notify_one();
        return item;
    }

    void push(T item) {
        std::unique_lock<std::mutex> mlock(mtx);

        // check whether limit is not reached
        while (
            (item.content.size() + queue_size.load()) > max_queue_size.load() ||
            (queue_.size() + 1) > max_elements_num.load()
        ) {
            read_cv.wait(mlock);
        }
        queue_size += item.content.size();

        queue_.push(std::move(item));
        mlock.unlock();
        cv.notify_one();
    }
};

#endif //_CONCURRENTQUEUE_H

//
// Created by root on 25-7-20.
//

// threadPool.cpp
#include "threadPool.h"

threadPool::threadPool(size_t num_threads) : active_tasks_(0), stop_(false) {
    for (size_t i = 0; i < num_threads; ++i) {
        // 创建线程，并让它们立即开始执行 worker_loop
        workers_.emplace_back([this] { this->worker_loop(); });
    }
}

threadPool::~threadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true; // 设置停止标志
    }
    cv_task_available_.notify_all(); // 唤醒所有线程，让它们检查停止标志并退出
    for (std::thread &worker : workers_) {
        worker.join(); // 等待每个线程安全退出
    }
}

void threadPool::worker_loop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            // 等待，直到有任务或收到停止信号
            cv_task_available_.wait(lock, [this] { return !this->tasks_.empty() || this->stop_; });

            // 如果是收到停止信号且任务队列已空，则退出循环
            if (this->stop_ && this->tasks_.empty()) {
                return;
            }

            task = std::move(tasks_.front());
            tasks_.pop();
        } // 锁在这里释放

        task(); // 执行任务（在没有锁的情况下）

        // 任务完成，减少活跃任务计数
        if (--active_tasks_ == 0) {
            // 如果这是最后一个任务，通知等待的线程
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_tasks_finished_.notify_all();
        }
    }
}

void threadPool::submit(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (stop_) {
            throw std::runtime_error("submit on stopped threadPool");
        }
        tasks_.emplace(std::move(task));
        active_tasks_++; // 增加活跃任务计数
    }
    cv_task_available_.notify_one(); // 唤醒一个线程来处理任务
}

void threadPool::wait_for_completion() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    // 等待，直到活跃任务计数为0
    cv_tasks_finished_.wait(lock, [this] { return this->active_tasks_ == 0; });
}
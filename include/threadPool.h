//
// Created by root on 25-7-20.
//

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

class threadPool {
public:
    // 构造函数：创建指定数量的工作线程
    threadPool(size_t num_threads);

    // 析构函数：确保所有线程都已完成并退出
    ~threadPool();

    // 提交任务到任务队列
    void submit(std::function<void()> task);

    // 等待所有当前提交的任务完成
    void wait_for_completion();

private:
    // 工作线程的主循环函数
    void worker_loop();

    std::vector<std::thread> workers_;          // 存储所有工作线程
    std::queue<std::function<void()>> tasks_;   // 任务队列
    std::mutex queue_mutex_;                    // 保护任务队列的互斥锁
    std::condition_variable cv_task_available_; // 条件变量，用于唤醒等待任务的线程
    std::condition_variable cv_tasks_finished_; // 条件变量，用于通知任务已全部完成
    std::atomic<int> active_tasks_;             // 正在执行或等待执行的任务计数
    std::atomic<bool> stop_;                    // 停止标志，用于通知线程退出
};

#endif

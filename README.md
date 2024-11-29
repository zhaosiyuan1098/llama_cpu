# yuangine

## 介绍

在x86和arm架构下，使用KVcache、SIMD、多线程、循环展开等方法对llama2推理加速

纯c++实现，执行效率高

## 依赖

- [xmake](https://github.com/xmake-io/xmake)

## 使用方法

1. 安装[xmake](https://github.com/xmake-io/xmake)
2. 克隆仓库并进入目录：
    ```bash
    git clone https://github.com/zhaosiyuan1098/yuangine.git
    cd yuangine
    ```

3. 下载所需模型：
* 使用curl下载
    ```bash
    
    cd ./model

    ```

    * x86:
    ```bash
    curl -L -o LLaMA_7B_2_chat.zip "https://www.dropbox.com/scl/fi/vu7wnes1c7gkcegg854ys/LLaMA_7B_2_chat.zip?rlkey=q61o8fpc954g1ke6g2eaot7cf&dl=1"
    ```
    * ARM:
    ```bash
    curl -L -o LLaMA_7B_2_chat.zip "https://www.dropbox.com/scl/fi/1trpw92vmh4czvl28hkv0/LLaMA_7B_2_chat.zip?rlkey=dy1pdek0147gnuxdzpodi6pkt&dl=1"
    ```
    解压
    ```bash
    unzip LLaMA_7B_2_chat.zip
    ```

* 使用python下载其他模型（可选）
    ```bash
    conda create -n yuangine python=3.10
    conda activate yuangine
    pip install -r requirenments.txt
    cd ./model

    python download_model.py --model 想要下载的模型名 --QM 对应的架构
    ```
3. 编译项目：
    ```bash
    cd ..
    xmake
    ```
4. 运行项目：
    ```bash
    xmake run
    ```

## 结构
参照llama2原始结构实现
![](./pic/llama2_structure.png)

[具体代码架构](./structure.txt)

## 效果展示

### 使用各种方法加速效果对比
| 方法 | x86 加速比 | ARM 加速比 | 备注 |
|------|------------|------------|------|
| SIMD+多线程+循环展开 | 16.16x | 18.3x | 使用缓存加速 |
| SIMD | 8.83x | 10.24x | 单指令多数据 |
| 多线程 | 2.99x | 3.17x | 并行计算 |
| 循环展开 | 1.04x | 1.06x | 减少循环开销 |

### 运行结果
SIMD+多线程+循环展开:
![](./pic/speedup.png)
SIMD:
![](./pic/simd.png)
多线程 
![](./pic/multithread.png)
循环展开
![](./pic/unrolling.png)
**本教程不得用於任何形式的商業用途，如果需要轉載請與作者SCP-173聯繫，如果發現未經允許複製轉載，將保留追求其法律責任的權利。 **



---
# 關於計算機的硬件配置說明
## **推薦配置**
如果您是高校學生或者高級研究人員，並且實驗室或者個人資金充沛，建議您採用如下配置：

 - 主板：X299型號或Z270型號
 - CPU: i7-6950X或i7-7700K 及其以上高級型號
 - 內存：品牌內存，總容量32G以上，根據主板組成4通道或8通道
 - SSD： 品牌固態硬盤，容量256G以上
 - <font color=#FF0000>顯卡：NVIDIA GTX TITAN(XP) NVIDIA GTX 1080ti、NVIDIA GTX TITAN、NVIDIA GTX 1080、NVIDIA GTX 1070、NVIDIA GTX 1060 (順序為優先建議，並且建議同一顯卡，可以根據主板插槽數量購買多塊，例如X299型號主板最多可以採用×4的顯卡)</font>
 - 電源：由主機機容量的確定，一般有顯卡總容量後再加200W即可
## **最低配置**
如果您是僅僅用於自學或代碼調試，亦或是條件所限僅採用自己現有的設備進行開發，那麼您的電腦至少滿足以下幾點：

 - CPU：Intel第三代i5和i7以上系列產品或同性能AMD公司產品
 - 內存：總容量4G以上

## <font color=#FF0000>CPU說明</font>
 - 大多數CPU目前支持多核多線程，那麼如果您採用CPU加速，就可以使用多線程運算。這方面的優勢對於服務器CPU志強系列尤為關鍵
## <font color=#FF0000>顯卡說明</font>
 - 如果您的顯卡是非NVIDIA公司的產品或是NVIDIA GTX系列中型號的第一個數字低於6或NVIDIA的GT系列，都不建議您採用此類顯卡進行加速計算，例如`NVIDIA GT 910`、 `NVIDIA GTX 460` 等等。
 - 如果您的顯卡為筆記本上的GTX移動顯卡（型號後面帶有標識M），那麼請您慎重使用顯卡加速，因為移動版GPU容易發生過熱燒毀現象。
 - 如果您的顯卡，顯示的是諸如 `HD5000`,`ATI 5650` 等類型的顯卡，那麼您只能使用CPU加速
 - 如果您的顯卡芯片為Pascal架構（`NVIDIA GTX 1080`,`NVIDIA GTX 1070`等），您只能在之後的配置中選擇`CUDA 8.0`
 ---

# 基本開發環境搭建
## 1. Linux 發行版
linux有很多發行版，本文強烈建議讀者採用新版的`Ubuntu 16.04 LTS`
一方面，對於大多數新手來說Ubuntu具有很好的圖形界面，與樂觀的開源社區；另一方面，Ubuntu是Nvidia官方以及絕大多數深度學習框架默認開發環境。
個人不建議使用Ubuntu其他版本，由於GCC編譯器版本不同，會導致很多依賴無法有效安裝。
Ubuntu 16.04 LTS<font color=#FF0000>下載地址</font>：http://www.ubuntu.org.cn/download/desktop
![](../images/keras_ubuntu_1.png)
通過U盤安裝好後，進行初始化環境設置。
## 2. Ubuntu初始環境設置

 - 安裝開發包
打開`終端`輸入：
```bash
# 系統升級
>>> sudo apt update
>>> sudo apt upgrade
# 安裝python基礎開發包
>>> sudo apt install -y python-dev python-pip python-nose gcc g++ git gfortran vim
```

 - 安裝運算加速庫
打開`終端`輸入：
```
>>> sudo apt install -y libopenblas-dev liblapack-dev libatlas-base-dev
```

## 3. CUDA開發環境的搭建(CPU加速跳過)
***如果您的僅僅採用cpu加速，可跳過此步驟***
 - 下載CUDA8.0

下載地址：https://developer.nvidia.com/cuda-downloads
![](../images/keras_ubuntu_2.png)

之後打開`終端`輸入：

```
>>> sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
>>> sudo apt update
>>> sudo apt -y install cuda
```
自動配置成功就好。

 - 將CUDA路徑添加至環境變量
在`終端`輸入：
```
>>> sudo gedit /etc/profile
```
在`profile`文件中添加：
```bash
export CUDA_HOME=/usr/local/cuda-8.0
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
之後`source /etc/profile`即可

 - 測試
在`終端`輸入：
```
>>> nvcc -V
```
會得到相應的nvcc編譯器相應的信息，那麼CUDA配置成功了。 (**記得重啟系統**)

如果要進行`cuda性能測試`，可以進行：
```shell
>>> cd /usr/local/cuda/samples
>>> sudo make -j8
```
編譯完成後，可以進`samples/bin/.../.../...`的底層目錄，運行各類實例。


## 4. 加速庫cuDNN（可選）
從官網下載需要註冊賬號申請，兩三天批准。網盤搜索一般也能找到最新版。
Linux目前最新的版本是cudnn V6，但對於tensorflow的預編譯版本還不支持這個最近版本，建議採用5.1版本，即是cudnn-8.0-win-x64-v5.1-prod.zip。
下載解壓出來是名為cuda的文件夾，裡面有bin、include、lib，將三個文件夾複製到安裝CUDA的地方覆蓋對應文件夾，在終端中輸入：
```shell
>>> sudo cp include/cudnn.h /usr/local/cuda/include/
>>> sudo cp lib64/* /usr/local/cuda/lib64/
>>> cd /usr/local/cuda/lib64
>>> sudo ln -sf libcudnn.so.5.1.10 libcudnn.so.5
>>> sudo ln -sf libcudnn.so.5 libcudnn.so
>>> sudo ldconfig -v
```

# Keras框架搭建

## 相關開發包安裝
在`終端`中輸入:
```shell
>>> sudo pip install -U --pre pip setuptools wheel
>>> sudo pip install -U --pre numpy scipy matplotlib scikit-learn scikit-image
>>> sudo pip install -U --pre tensorflow-gpu
# >>> sudo pip install -U --pre tensorflow ## CPU版本
>>> sudo pip install -U --pre keras
```
安裝完畢後，輸入`python`，然後輸入：
```python
>>> import tensorflow
>>> import keras
```
無錯輸出即可


## Keras中mnist數據集測試
 下載Keras開發包
```shell
>>> git clone https://github.com/fchollet/keras.git
>>> cd keras/examples/
>>> python mnist_mlp.py
```
程序無錯進行，至此，keras安裝完成。


## 聲明與聯繫方式 ##

由於作者水平和研究方向所限，無法對所有模塊都非常精通，因此文檔中不可避免的會出現各種錯誤、疏漏和不足之處。如果您在使用過程中有任何意見、建議和疑問，歡迎發送郵件到scp173.cool@gmail.com與作者取得聯繫.

**本教程不得用於任何形式的商業用途，如果需要轉載請與作者或中文文檔作者聯繫，如果發現未經允許複製轉載，將保留追求其法律責任的權利。 **

作者：[SCP-173](https://github.com/KaiwenXiao)
E-mail ：scp173.cool@gmail.com
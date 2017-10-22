# Keras:基於Python的深度學習庫



## 這就是Keras
Keras是一個高層神經網絡API，Keras由純Python編寫而成並基[Tensorflow](https://github.com/tensorflow/tensorflow)、[Theano](https://github.com/Theano/Theano)以及[CNTK](https://github.com/Microsoft/cntk)後端。 Keras
為支持快速實驗而生，能夠把你的idea迅速轉換為結果，如果你有如下需求，請選擇Keras：

* 簡易和快速的原型設計（keras具有高度模塊化，極簡，和可擴充特性）
* 支持CNN和RNN，或二者的結合
* 無縫CPU和GPU切換

Keras適用的Python版本是：Python 2.7-3.5

Keras的設計原則是

* **用戶友好**：Keras是為人類而不是天頂星人設計的API。用戶的使用體驗始終是我們考慮的首要和中心內容。 Keras遵循減少認知困難的最佳實踐：Keras提供一致而簡潔的API， 能夠極大減少一般應用下用戶的工作量，同時，Keras提供清晰和具有實踐意義的bug反饋。

* **模組化**：模型可理解為一個層的序列或數據的運算圖，完全可配置的模塊可以用最少的代價自由組合在一起。具體而言，網絡層、損失函數、優化器、初始化策略、激活函數、正則化方法都是獨立的模塊，你可以使用它們來構建自己的模型。

* **易擴展性**：添加新模塊超級容易，只需要仿照現有的模塊編寫新的類或函數即可。創建新模塊的便利性使得Keras更適合於先進的研究工作。

* **與Python協作**：Keras沒有單獨的模型配置文件類型（作為對比，caffe有），模型由python代碼描述，使其更緊湊和更易debug，並提供了擴展的便利性。


***

## 關於Keras-tw

本文檔是Keras文檔的中文繁體版(主要是從keras-cn中文簡體版中轉譯過來)，內容包括了[keras.io](http://keras.io/)的全部內容，以及由keras-cn貢獻者所撰寫的例子、解釋和建議

現在，keras-tw的版本號將簡單的跟隨最新的keras release版本

文檔中不可避免的會出現各種錯誤、疏漏和不足之處。如果您在使用過程中有任何意見、建議和疑問，歡迎發送郵件到erhwenkuo@gmail.com與我取得聯繫或回覆至原簡體作者moyan_work@foxmail.com。

您對文檔的任何貢獻，包括文檔的翻譯、查缺補漏、概念解釋、發現和修改問題、貢獻示例程序等，均會被記錄在[致謝](acknowledgement)，十分感謝您對Keras中文文檔的貢獻！

如果你發現本文檔缺失了官方文檔的部分內容，請積極聯繫我補充。

本文檔相對於原文檔有更多的使用指導和概念澄清，請在使用時關注文檔中的Tips，特別的，本文檔的額外模塊還有：

* Keras新手指南：我們新提供了“Keras新手指南”的頁面，在這裡我們對Keras進行了感​​性介紹，並簡單介紹了Keras配置方法、一些小知識與使用陷阱，新手在使用前應該先閱讀本部分的文檔。

* Keras資源：在這個頁面，我們羅列一些Keras可用的資源，本頁面會不定期更新，請注意關注

* 深度學習與Keras：位於導航欄最下方的該模塊翻譯了來自Keras作者博客[keras.io](http://blog.keras.io/)
和其他Keras相關博客的文章，該欄目的文章提供了對深度學習的理解和大量使用Keras的例子，您也可以向這個欄目投稿。
所有的文章均在醒目位置標誌標明來源與作者，本文檔對該欄目文章的原文不具有任何處置權。如您仍覺不妥，請聯繫本人（erhwenkuo@gmail.com）刪除。

***

## 當前版本與更新

如果你發現本文檔提供的信息有誤，有兩種可能：

* 你的Keras版本過低：記住Keras是一個發展迅速的深度學習框架，請保持你的Keras與官方最新的release版本相符

* 我們的中文文檔沒有及時更新：如果是這種情況，請發郵件給我，我會盡快更新

目前文檔的版本號是2.0.8，對應於官方的2.0.8 release 版本：

* FAQ新增了關於可複現模型的支持
* application中新增了模型MobileNet
* constraints新增min_max_norm
* 新增了激活函數selu和與之配合的層AlphaDropout
* 新增損失函數categorical_hinge
* 由於年久失修，**深度學習與Keras**欄目中的很多內容的代碼已經不再可用，我們決定在新的文檔中移除這部分。仍然想訪問這些內容（以及已經被移除的一些層，如Maxout）的文檔的同學，請下載[中文文檔](https://github.com/MoyanZitto/keras-cn)的legacy文件夾，並使用文本編輯器（如sublime）打開對應.md文件。
* 修正了一些錯誤，感謝@zhangxiaoyu，@Yang Song，@唐文威，@Jackie，@銹子，的寶貴意見
* 此外，感謝@zh777k製作了Keras2.0.4中文文檔的離線版本，對於許多用戶而言，這個版本的keras已經足夠使用了。下載地址在[百度雲盤](http://pan.baidu.com/s/1geHmOpH)

注意，keras在github上的master往往要高於當前的release版本，如果你從源碼編譯keras，可能某些模塊與文檔說明不相符，請以官方Github代碼為準

***

##快速開始：30秒上手Keras

Keras的核心數據結構是“模型”，模型是一種組織網絡層的方式。 Keras中主要的模型是Sequential模型，Sequential是一系列網絡層按順序構成的棧。你也可以查看[函數式模型](getting_started/functional_API.md)來學習建立更複雜的模型

Sequential模型如下
```python
from keras.models import Sequential

model = Sequential()
```
將一些網絡層通過<code>.add\(\)</code>堆疊起來，就構成了一個模型：
```python
from keras.layers import Dense, Activation

model.add(Dense(units=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(units=10))
model.add(Activation("softmax"))
```
完成模型的搭建後，我們需要使用<code>.compile\(\)</code>方法來編譯模型：
```python
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```
編譯模型時必須指明損失函數和優化器，如果你需要的話，也可以自己定制損失函數。 Keras的一個核心理念就是簡明易用同時，保證用戶對Keras的絕對控制力度，用戶可以根據自己的需要定制自己的模型、網絡層，甚至修改源代碼。
```python
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
```
完成模型編譯後，我們在訓練數據上按batch進行一定次數的迭代來訓練網絡


```python
model.fit(x_train, y_train, epochs=5, batch_size=32)
```
當然，我們也可以手動將一個個batch的數據送入網絡中訓練，這時候需要使用：
```python
model.train_on_batch(x_batch, y_batch)
```
隨後，我們可以使用一行代碼對我們的模型進行評估，看看模型的指標是否滿足我們的要求：
```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```
或者，我們可以使用我們的模型，對新的數據進行預測：
```python
classes = model.predict(x_test, batch_size=128)
```
搭建一個問答系統、圖像分類模型，或神經圖靈機、word2vec詞嵌入器就是這麼快。支撐深度學習的基本想法本就是簡單的，現在讓我們把它的實現也變的簡單起來！

為了更深入的了解Keras，我們建議你查看一下下面的兩個tutorial

* [快速開始Sequntial模型](getting_started/sequential_model)
* [快速開始函數式模型](getting_started/functional_API)

還有我們的新手教程，雖然是面向新手的，但我們閱讀它們總是有益的：

* [Keras新手指南](for_beginners/concepts)

在Keras代碼包的examples文件夾裡，我們提供了一些更高級的模型：基於記憶網絡的問答系統、基於LSTM的文本的文本生成等。

***

##安裝

Keras使用了下面的依賴包，三種後端必須至少選擇一種，我們建議選擇tensorflow。

* numpy，scipy

* pyyaml

* HDF5, h5py（可選，僅在模型的save/load函數中使用）

* 如果使用CNN的推薦安裝cuDNN

當使用TensorFlow為後端時：

* [TensorFlow](https://www.tensorflow.org/install/)

當使用Theano作為後端時：

* [Theano](http://deeplearning.net/software/theano/install.html#install)

當使用CNTK作為後端時：

* [CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine)


“後端”翻譯自backend，指的是Keras依賴於完成底層的張量運算的軟件包。


從源碼安裝Keras時，首先git clone keras的代碼：
```sh
git clone https://github.com/fchollet/keras.git
```
接著<code>cd</code>到Keras的文件夾中，並運行下面的安裝命令：
```python
sudo python setup.py install
```
你也可以使用PyPI來安裝Keras
```python
sudo pip install keras
```
如果你用的是virtualenv虛擬環境，不要用sudo就好。
**詳細的Windows和Linux安裝教程請參考“Keras新手指南”中給出的安裝教程，特別鳴謝SCP-173編寫了這些教程**

***

##在Theano、CNTK、TensorFlow間切換

Keras預設使用TensorFlow作為後端來進行張量操作，如需切換到Theano，請查看[這裡](backend)

***

##技術支持

你可以在下列網址提問或加入Keras開發討論:

- [Keras Google group](https://groups.google.com/forum/#!forum/keras-users)
- [Keras Slack channel](https://kerasteam.slack.com/),[點擊這裡](https://keras-slack-autojoin.herokuapp.com/)獲得邀請.

你也可以在[Github issues](https://github.com/fchollet/keras/issues)裡提問或請求新特性。在提問之前請確保你閱讀過我們的[原則](https://github.com/fchollet/keras/blob/master/CONTRIBUTING.md)

***

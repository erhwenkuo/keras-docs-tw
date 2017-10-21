# Scikit-Learn接口包裝器

我們可以通過包裝器將```Sequential```模型（僅有一個輸入）作為Scikit-Learn工作流的一部分，相關的包裝器定義在```keras.wrappers.scikit_learn.py```中

目前，有兩個包裝器可用：

```keras.wrappers.scikit_learn.KerasClassifier(build_fn=None, **sk_params)```實現了sklearn的分類器接口


```keras.wrappers.scikit_learn.KerasRegressor(build_fn=None, **sk_params)```實現了sklearn的回歸器接口

## 參數

* build_fn：可調用的函數或類對象

* sk_params：模型參數和訓練參數

```build_fn```應構造、編譯並返回一個Keras模型，該模型將稍後用於訓練/測試。 ```build_fn```的值可能為下列三種之一：

1. 一個函數

2. 一個具有```call```方法的類對象

3. None，代表你的類繼承自```KerasClassifier```或```KerasRegressor```，其```call```方法為其父類的```call```方法

```sk_params```以模型參數和訓練（超）參數作為參數。合法的模型參數為```build_fn```的參數。注意，‘build_fn’應提供其參數的預設值。所以我們不傳遞任何值給```sk_params```也可以創建一個分類器/回歸器

```sk_params```還接受用於調用```fit```，```predict```，```predict_proba```和```score```方法的參數，如`` `nb_epoch```，```batch_size```等。這些用於訓練或預測的參數按如下順序選擇：

1. 傳遞給```fit```，```predict```，```predict_proba```和```score```的字典參數

2. 傳遞個```sk_params```的參數

3. ```keras.models.Sequential```，```fit```，```predict```，```predict_proba```和```score```的預設值

當使用scikit-learn的```grid_search```接口時，合法的可轉換參數是你可以傳遞給```sk_params```的參數，包括訓練參數。即，你可以使用```grid_search```來搜索最佳的```batch_size```或```nb_epoch```以及其他模型參數
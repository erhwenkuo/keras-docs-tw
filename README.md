# Traditional Chinese translation of the Keras documentation

This is the repository for the translated `.md` sources files of [keras.io](http://keras.io/). 
Also, this repository includes some extra useful information (tips, environment setup...etc) 
regarding to Keras.

---


# keras-docs-tw

本專案旨在建立一個[keras.io](keras.io)的中文版文檔，提供給深度學習開發者對Keras的API參考與相關範例。
本項目目前已完成2.8版本的轉譯，繁體版文檔網址為[keras-docs-tw](http://eighty20.cc/keras-docs-tw/)。

簡體版文檔網址為[keras-cn](http://keras-cn.readthedocs.io/en/latest/)是本專案初始轉譯的版本(由BigMoyan於2016-4-29發起)。

這個文檔專案是基於Mkdocs來生成靜態網頁

## 構建文檔
- 在你/妳的電腦上安裝python與pip
- 安裝MkDocs: ```pip install mkdocs```
- 執行:
  - mkdocs serve # 開始一個本地的網頁伺服器: localhost:8000
  - mkdocs build # 構建靜態網頁到"site"目錄

## 貢獻
如果你想為文檔做出貢獻，請使用Markdown編寫文檔並遵守以下約定。

### 0.字體顏色

保持預設的字體顏色和字號，錨點的顏色為預設，超鏈接的顏色為預設

當使用```<a name='something'></a>```來設置錨點時，可能引起字體顏色改變，如果顏色發生改變，需要使用font修正字體，預設字體的顏色是#404040

### 1.標題級別

頁面大標題為一級，一般每個文件只有一個一級標題 #
頁面內的小節是二級標題 ##
小節的小節，如examples裡各個example是三級標題 ###
  
### 2.代碼塊規則

成塊的代碼使用

\`\`\`python

code

\`\`\`

的形式顯式指明
段中代碼使用\`\`\`code\`\`\`的形式指明

### 3. 超鏈接

鏈接到本項目其他頁面的超鏈接使用相對路徑，一個例子是
```[<font color='#FF0000'>text</font>](../models/about_model.md)```
鏈接到其他外站的鏈接形式與此相同，只不過圓括號中是絕對地址

### 4.圖片

圖片保存在docs/images中，插入的例子是：

```

![text](../images/image_name.png)

```

### 5.分割線

每個二級標題之間使用
\*\*\*
產生一個分割線

# 參考網站

## Markdown繁體中文教程

[Markdown](http://markdown.tw/)

## MkDocs簡體中文教程

[MkDocs](http://markdown-docs-zh.readthedocs.io/zh_CN/latest/)

## Keras官方文檔(英文)

[Keras](http://keras.io/)

感謝參與！
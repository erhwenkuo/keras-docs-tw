# 模型可視化

```keras.utils.vis_utils```模塊提供了畫出Keras模型的函數（利用graphviz）

該函數將畫出模型結構圖，並保存成圖片：

```python
from keras.utils import plot_model
plot_model(model, to_file='model.png')
```

```plot_model```接收兩個可選參數：

* ```show_shapes```：指定是否顯示輸出數據的形狀，默認為```False```
* ```show_layer_names```:指定是否顯示層名稱,默認為```True```

我們也可以直接獲取一個```pydot.Graph```對象，然後按照自己的需要配置它，例如，如果要在ipython中展示圖片
```python
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
```

【Tips】依賴 pydot-ng 和 graphviz，若出現錯誤，用命令行輸入```pip install pydot-ng & brew install graphviz```
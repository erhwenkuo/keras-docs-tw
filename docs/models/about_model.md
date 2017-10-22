# 關於Keras模型

Keras有兩種類型的模型，[序貫模型（Sequential）](sequential.md)和[函數式模型（Model）](model.md)，函數式模型應用更為廣泛，序貫模型是函數式模型的一種特殊情況。

兩類模型有一些方法是相同的：

* ```model.summary()```：打印出模型概況

* ```model.get_config()```:返回包含模型配置信息的Python字典。模型也可以從它的config信息中重構回去


```python
config = model.get_config()
model = Model.from_config(config)
# or, for Sequential:
model = Sequential.from_config(config)
```

* ```model.get_layer()```：依據層名或下標獲得層對象

* ```model.get_weights()```：返回模型權重張量的列表，類型為numpy array

* ```model.set_weights()```：從numpy array裡將權重載入給模型，要求數組具有與```model.get_weights()```相同的形狀。

* ```model.to_json```：返回代表模型的JSON字符串，僅包含網絡結構，不包含權值。可以從JSON字符串中重構原模型：

```python
from models import model_from_json

json_string = model.to_json()
model = model_from_json(json_string)
```

* ```model.to_yaml```：與```model.to_json```類似，同樣可以從產生的YAML字符串中重構模型

```python
from models import model_from_yaml

yaml_string = model.to_yaml()
model = model_from_yaml(yaml_string)
```

* ```model.save_weights(filepath)```：將模型權重保存到指定路徑，文件類型是HDF5（後綴是.h5）

* ```model.load_weights(filepath, by_name=False)```：從HDF5文件中加載權重到當前模型中, 預設情況下模型的結構將保持不變。如果想將權重載入不同的模型（有些層相同）中，則設置```by_name=True```，只有名字匹配的層才會載入權重
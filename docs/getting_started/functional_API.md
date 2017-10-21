# 快速開始函數式（Functional）模型

我們起初將Functional一詞譯作泛型，想要表達該類模型能夠表達任意張量映射的含義，但表達的不是很精確，在Keras 2裡我們將這個詞改譯為“函數式”，對函數式編程有所了解的同學應能夠快速get到該類模型想要表達的含義。函數式模型稱作Functional，但它的類名是Model，因此我們有時候也用Model來代表函數式模型。

Keras函數式模型接口是用戶定義多輸出模型、非循環有向模型或具有共享層的模型等複雜模型的途徑。一句話，只要你的模型不是類似VGG一樣一條路走到黑的模型，或者你的模型需要多於一個的輸出，那麼你總應該選擇函數式模型。函數式模型是最廣泛的一類模型，序貫模型（Sequential）只是它的一種特殊情況。

這部分的文檔假設你已經對Sequential模型已經比較熟悉

讓我們從簡單一點的模型開始

## 第一個模型：全連接網絡

```Sequential```當然是實現全連接網絡的最好方式，但我們從簡單的全連接網絡開始，有助於我們學習這部分的內容。在開始前，有幾個概念需要澄清：

* 層對象接受張量為參數，返回一個張量。

* 輸入是張量，輸出也是張量的一個框架就是一個模型，通過```Model```定義。

* 這樣的模型可以被像Keras的```Sequential```一樣被訓練

```python
from keras.layers import Input, Dense
from keras.models import Model

# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels) # starts training
```

***

## 所有的模型都是可調用的，就像層一樣

利用函數式模型的接口，我們可以很容易的重用已經訓練好的模型：你可以把模型當作一個層一樣，通過提供一個tensor來調用它。注意當你調用一個模型時，你不僅僅重用了它的結構，也重用了它的權重。

```python
x = Input(shape=(784,))
# This works, and returns the 10-way softmax we defined above.
y = model(x)
```

這種方式可以允許你快速的創建能處理序列信號的模型，你可以很快將一個圖像分類的模型變為一個對視頻分類的模型，只需要一行代碼：

```python
from keras.layers import TimeDistributed

# Input tensor for sequences of 20 timesteps,
# each containing a 784-dimensional vector
input_sequences = Input(shape=(20, 784))

# This applies our previous model to every timestep in the input sequences.
# the output of the previous model was a 10-way softmax,
# so the output of the layer below will be a sequence of 20 vectors of size 10.
processed_sequences = TimeDistributed(model)(input_sequences)
```

***

## 多輸入和多輸出模型

使用函數式模型的一個典型場景是搭建多輸入、多輸出的模型。

考慮這樣一個模型。我們希望預測Twitter上一條新聞會被轉發和點贊多少次。模型的主要輸入是新聞本身，也就是一個詞語的序列。但我們還可以擁有額外的輸入，如新聞發布的日期等。這個模型的損失函數將由兩部分組成，輔助的損失函數評估僅僅基於新聞本身做出預測的情況，主損失函數評估基於新聞和額外信息的預測的情況，即使來自主損失函數的梯度發生彌散，來自輔助損失函數的信息也能夠訓練Embeddding和LSTM層。在模型中早點使用主要的損失函數是對於深度網絡的一個良好的正則方法。總而言之，該模型框圖如下：

![multi-input-multi-output-graph](../images/multi-input-multi-output-graph.png)

讓我們用函數式模型來實現這個框圖

主要的輸入接收新聞本身，即一個整數的序列（每個整數編碼了一個詞）。這些整數位於1到10，000之間（即我們的字典有10，000個詞）。這個序列有100個單詞。

```python
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
main_input = Input(shape=(100,), dtype='int32', name='main_input')

# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# A LSTM will tr​​ansform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = LSTM(32)(x)
```

然後，我們插入一個額外的損失，使得即使在主損失很高的情況下，LSTM和Embedding層也可以平滑的訓練。

```python
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
```

再然後，我們將LSTM與額外的輸入數據串聯起來組成輸入，送入模型中：

```python
auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# And finally we add the main logistic regression layer
main_output = Dense(1, activation='sigmoid', name='main_output')(x)
```
最後，我們定義整個2輸入，2輸出的模型：

```python
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
```

模型定義完畢，下一步編譯模型。我們給額外的損失賦0.2的權重。我們可以通過關鍵字參數```loss_weights```或```loss```來為不同的輸出設置不同的損失函數或權值。這兩個參數均可為Python的列表或字典。這裡我們給```loss```傳遞單個損失函數，這個損失函數會被應用於所有輸出上。

```python
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1., 0.2])
```
編譯完成後，我們通過傳遞訓練數據和目標值訓練該模型：

```python
model.fit([headline_data, additional_data], [labels, labels],
          epochs=50, batch_size=32)

```
因為我們輸入和輸出是被命名過的（在定義時傳遞了“name”參數），我們也可以用下面的方式編譯和訓練模型：

```python
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})

# And trained it via:
model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': labels, 'aux_output': labels},
          epochs=50, batch_size=32)
```

***

<a name='node'>
<font color='#404040'>
## 共享層
另一個使用函數式模型的場合是使用共享層的時候。

考慮微博數據，我們希望建立模型來判別兩條微博是否是來自同一個用戶，這個需求同樣可以用來判斷一個用戶的兩條微博的相似性。

一種實現方式是，我們建立一個模型，它分別將兩條微博的數據映射到兩個特徵向量上，然後將特徵向量串聯並加一個logistic回歸層，輸出它們來自同一個用戶的概率。這種模型的訓練數據是一對對的微博。

因為這個問題是對稱的，所以處理第一條微博的模型當然也能重用於處理第二條微博。所以這裡我們使用一個共享的LSTM層來進行映射。

首先，我們將微博的數據轉為（140，256）的矩陣，即每條微博有140個字符，每個單詞的特徵由一個256維的詞向量表示，向量的每個元素為1表示某個字符出現，為0表示不出現，這是一個one-hot編碼。

之所以是（140，256）是因為一條微博最多有140個字符，而擴展的ASCII碼表編碼了常見的256個字符。原文中此處為Tweet，所以對外國人而言這是合理的。如果考慮中文字符，那一個單詞的詞向量就不止256了。

```python
import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

tweet_a = Input(shape=(140, 256))
tweet_b = Input(shape=(140, 256))
```

若要對不同的輸入共享同一層，就初始化該層一次，然後多次調用它

```python
# This layer can take as input a matrix
# and will return a vector of size 64
shared_lstm = LSTM(64)

# When we reuse the same layer instance
# multiple times, the weights of the layer
# are also being reused
# (it is effectively *the same* layer)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

# We can then concatenate the two vectors:
merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)

# And add a logistic regression on top
predictions = Dense(1, activation='sigmoid')(merged_vector)

# We define a trainable model linking the
# tweet inputs to the predictions
model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([data_a, data_b], labels, epochs=10)
```

先暫停一下，看看共享層到底輸出了什麼，它的輸出數據shape又是什麼

***

## 層“節點”的概念

無論何時，當你在某個輸入上調用層時，你就創建了一個新的張量（即該層的輸出），同時你也在為這個層增加一個“（計算）節點”。這個節點將輸入張量映射為輸出張量。當你多次調用該層時，這個層就有了多個節點，其下標分別為0，1，2...

在上一版本的Keras中，你可以通過```layer.get_output()```方法來獲得層的輸出張量，或者通過```layer.output_shape```獲得其輸出張量的shape。這個版本的Keras你仍然可以這麼做（除了```layer.get_output()```被```output```替換）。但如果一個層與多個輸入相連，會出現什麼情況呢？

如果層只與一個輸入相連，那沒有任何困惑的地方。 ```.output```將會返回該層唯一的輸出

```python
a = Input(shape=(140, 256))

lstm = LSTM(32)
encoded_a = lstm(a)

assert lstm.output == encoded_a
```

但當層與多個輸入相連時，會出現問題

```
a = Input(shape=(140, 256))
b = Input(shape=(140, 256))

lstm = LSTM(32)
encoded_a = lstm(a)
encoded_b = lstm(b)

lstm.output
```

上面這段代碼會報錯

```python
>> AssertionError: Layer lstm_1 has multiple inbound nodes,
hence the notion of "layer output" is ill-defined.
Use `get_output_at(node_index)` instead.
```

通過下面這種調用方式即可解決

```python
assert lstm.get_output_at(0) == encoded_a
assert lstm.get_output_at(1) == encoded_b
```
</font>
</a>


對於```input_shape```和```output_shape```也是一樣，如果一個層只有一個節點，或所有的節點都有相同的輸入或輸出shape，那麼```input_shape```和`` `output_shape```都是沒有歧義的，並也只返回一個值。但是，例如你把一個相同的```Conv2D```應用於一個大小為\(32,32,3\)的數據，然後又將其應用於一個\(64,64,3\)的數據，那麼此時該層就具有了多個輸入和輸出的shape，你就需要顯式的指定節點的下標，來表明你想取的是哪個了

```python
a = Input(shape=(32, 32, 3))
b = Input(shape=(64, 64, 3))

conv = Conv2D(16, (3, 3), padding='same')
conved_a = conv(a)

# Only one input so far, the following will work:
assert conv.input_shape == (None, 32, 32, 3)

conved_b = conv(b)
# now the `.input_shape` property wouldn't work, but this does:
assert conv.get_input_shape_at(0) == (None, 32, 32, 3)
assert conv.get_input_shape_at(1) == (None, 64, 64, 3)
```
***

## 更多的例子

代碼示例依然是學習的最佳方式，這裡是更多的例子

### inception模型

inception的詳細結構參見Google的這篇論文：[Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842)

```python
from keras.layers import Conv2D, MaxPooling2D, Input

input_img = Input(shape=(256, 256, 3))

tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
```

### 卷積層的殘差連接

殘差網絡（Residual Network）的詳細信息請參考這篇文章：[Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385)

```python
from keras.layers import Conv2D, Input

# input tensor for a 3-channel 256x256 image
x = Input(shape=(256, 256, 3))
# 3x3 conv with 3 output channels (same as input channels)
y = Conv2D(3, (3, 3), padding='same')(x)
# this returns x + y.
z = keras.layers.add([x, y])
```

### 共享視覺模型

該模型在兩個輸入上重用了圖像處理的模型，用來判別兩個MNIST數字是否是相同的數字
```python
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model

# First, define the vision modules
digit_input = Input(shape=(27, 27, 1))
x = Conv2D(64, (3, 3))(digit_input)
x = Conv2D(64, (3, 3))(x)
x = MaxPooling2D((2, 2))(x)
out = Flatten()(x)

vision_model = Model(digit_input, out)

# Then define the tell-digits-apart model
digit_a = Input(shape=(27, 27, 1))
digit_b = Input(shape=(27, 27, 1))

# The vision model will be shared, weights and all
out_a = vision_model(digit_a)
out_b = vision_model(digit_b)

concatenated = keras.layers.concatenate([out_a, out_b])
out = Dense(1, activation='sigmoid')(concatenated)

classification_model = Model([digit_a, digit_b], out)
```

### 視覺問答模型

在針對一幅圖片使用自然語言進行提問時，該模型能夠提供關於該圖片的一個單詞的答案

這個模型將自然語言的問題和圖片分別映射為特徵向量，將二者合併後訓練一個logistic回歸層，從一系列可能的回答中挑選一個。
```python
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential

# First, let's define a vision model using a Sequential model.
# This model will encode an image into a vector.
vision_model = Sequential()
vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
vision_model.add(Conv2D(64, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(128, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Flatten())

# Now let's get a tensor with the output of our vision model:
image_input = Input(shape=(224, 224, 3))
encoded_image = vision_model(image_input)

# Next, let's define a language model to encode the question into a vector.
# Each question will be at most 100 word long,
# and we will index words as integers from 1 to 9999.
question_input = Input(shape=(100,), dtype='int32')
embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
encoded_question = LSTM(256)(embedded_question)

# Let's concatenate the question vector and the image vector:
merged = keras.layers.concatenate([encoded_question, encoded_image])

# And let's train a logistic regression over 1000 words on top:
output = Dense(1000, activation='softmax')(merged)

# This is our final model:
vqa_model = Model(inputs=[image_input, question_input], outputs=output)

# The next stage would be training this model on actual data.

```

### 視頻問答模型

在做完圖片問答模型後，我們可以快速將其轉為視頻問答的模型。在適當的訓練下，你可以為模型提供一個短視頻（如100幀）然後向模型提問一個關於該視頻的問題，如“what sport is the boy playing？”->“football”

```python
from keras.layers import TimeDistributed

video_input = Input(shape=(100, 224, 224, 3))
# This is our video encoded via the previously trained vision_model (weights are reused)
encoded_frame_sequence = TimeDistributed(vision_model)(video_input) # the output will be a sequence of vectors
encoded_video = LSTM(256)(encoded_frame_sequence) # the output will be a vector

# This is a model-level representation of the question encoder, reusing the same weights as before:
question_encoder = Model(inputs=question_input, outputs=encoded_question)

# Let's use it to encode the question:
video_question_input = Input(shape=(100,), dtype='int32')
encoded_video_question = question_encoder(video_question_input)

# And this is our video question answering model:
merged = keras.layers.concatenate([encoded_video, encoded_video_question])
output = Dense(1000, activation='softmax')(merged)
video_qa_model = Model(inputs=[video_input, video_question_input], outputs=output)
```
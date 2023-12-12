# 회전기계 고장유형 진단

@ author : Han Minwoong

@ Mechanical and control engineering

@ Purpose :

* Determination of failure type based on time series vibration data of rotating machinery
* Directly identify features contained in the data, such as the relationship between vibration data and failure types. Build an analysis framework that can extract and classify state types



[TOC]



## 1. Data specifications

* rotor testbed를 대상 설비로 하고, 가속도 센서를 이용하여 대상 설비로부터 진동데이터를 수집
* 해당 데이터는 rotor testbed의 상태에 따라 4개의 센서로부터 취득한 시계열 데이터

<img src="https://github.com/HanMinung/DLIP/assets/99113269/9925b44c-a776-4853-8628-2ec7a3a694ef" alt="image" style="zoom:80%;" />

* 4개의 센서 부착 위치에서, 약 140초의 가속도 신호를 수집



### 1.1. Data type

* 정상 상태 (Normal)
* 질량 불균형 (unbalance - Type 1)
  * 회전기기의 진동 원인으로 여러 가지가 있으나 그 중 질량 불균형이 차지하는 비중이 가장 크다. 이는 회전기기에서 rotor의 질량 중심이 회전 중심(축 중심)과 일치하지 않을 때 나타나는 현상이다. 따라서 비대칭적인 질량분포가 있는 rotor를 회전시켰을 때 발생하는 비대칭적 원심력은 회전기계에서 진동(소음)을 발생시킨다.
* 지지 불량 (Mechanical Looseness - Type 2)
  * 회전기계를 지면 혹은 평판 등에 견고하게 지지하지 않은 경우, 기계의 수평이 틀어지거나 장비의기울어짐 등이 확인될 수 있다. 특히 기계 지지가 구조적으로 이완되고 및 취약한 경우, 다른 고장 유형과 유사한 진동 스펙트럼을 나타내므로 불평형, 축 정렬 불량 등으로 잘못 진단할 수 있다
* Type 3 : Type 1 + Type 2 

<img src="https://github.com/HanMinung/DLIP/assets/99113269/923bb245-8fca-40c4-af8b-8dd91890485d" alt="image" style="zoom:80%;" />

위 데이터는 group 1 sensor 1에 대한 데이터 파일예시이다.

* 첫 번째 열 : 시간 데이터 (0 ~ 약 140초 / sampling frequency)
* 두 번째 열 : Normal data 
* 세 번째 열 : Type 1
* 네 번째 열 : Type 2
* 다섯 번째 열 : Type 3



## 2. Data preprocessing

### 2.1. 전처리의 필요성

다수의 센서를 사용하는 경우, 각 센서로부터 수집된 데이터의 저장 간격이 일정하지 않을 수 있다. 그렇기 때문에, 계측 시간의 동일화 (resampling)이 필요하다. 사용된 모든 센서는 같으은 정류의 가속도 센서이지만, sampling rate가 다르다.

* sensor 1 : 0.000736
* sensor 2: 0.000760
* sensor 3 : 0.000714
* sensor 4 : 0.000761

모터 최대 속도가 3000RPM인 회전 기계의 최소 sampling rate는 Nyquist 이론에 의해 회전기계의 최대주파수 (25 Hz)의 2배 이상으로 설정해야 한다. 또한, 계측 데이터의 신호를 분석하기 위해서는 최대주파수의 5-10배의 계측 sampling rate가 요구된다. 여기서, 센서의 계측 데이터를 최대한 보존하는 범위에서 신호 분석이 가능하도록 resampling을 하기 위해 0.001 sec의 sampling period로 down-sampling을 진행한다.



### 2.2. 전처리의 진행

* Resampling : down sampling
* Moving average filter
* Min-max scaling



### 2.3. 코드 구현

* Resampling with interpolation
  
  ```python
  x_new  = np.arange(0, 140, 0.001)
  y_new1, y_new2, y_new3, y_new4 = [], [], [], []
  
  for item in ['normal', 'type1', 'type2', 'type3']:
      
      f_linear1 = interpolate.interp1d(sensor1['time'], sensor1[item], kind='linear')
      y_new1.append(f_linear1(x_new))
      f_linear2 = interpolate.interp1d(sensor2['time'], sensor2[item], kind='linear')
      y_new2.append(f_linear2(x_new))
      f_linear3 = interpolate.interp1d(sensor3['time'], sensor3[item], kind='linear')
      y_new3.append(f_linear3(x_new))
      f_linear4 = interpolate.interp1d(sensor4['time'], sensor4[item], kind='linear')
      y_new4.append(f_linear4(x_new))
   
  sensor1 = pd.DataFrame(np.array(y_new1).T, columns = ['normal', 'type1', 'type2', 'type3'])
  sensor2 = pd.DataFrame(np.array(y_new2).T, columns = ['normal', 'type1', 'type2', 'type3'])
  sensor3 = pd.DataFrame(np.array(y_new3).T, columns = ['normal', 'type1', 'type2', 'type3'])
  sensor4 = pd.DataFrame(np.array(y_new4).T, columns = ['normal', 'type1', 'type2', 'type3'])
  ```
  
  <img src="https://github.com/HanMinung/DLIP/assets/99113269/2eeff910-27ae-4938-b242-52b17532fb71" alt="image" style="zoom:80%;" />
  
* Moving average filter

  ```python
  M = 15
  normal_s1 = np.convolve(normal_['s1'], np.ones(M), 'valid') / M;    normal_s1 = normal_s1.reshape(len(normal_s1),1)
  normal_s2 = np.convolve(normal_['s2'], np.ones(M), 'valid') / M;    normal_s2 = normal_s2.reshape(len(normal_s2),1)
  normal_s3 = np.convolve(normal_['s3'], np.ones(M), 'valid') / M;    normal_s3 = normal_s3.reshape(len(normal_s3),1)
  normal_s4 = np.convolve(normal_['s4'], np.ones(M), 'valid') / M;    normal_s4 = normal_s4.reshape(len(normal_s4),1)
  
  type1_s1 = np.convolve(type1_['s1'], np.ones(M), 'valid') / M;  type1_s1 = type1_s1.reshape(len(type1_s1),1)
  type1_s2 = np.convolve(type1_['s2'], np.ones(M), 'valid') / M;  type1_s2 = type1_s2.reshape(len(type1_s2),1)
  type1_s3 = np.convolve(type1_['s3'], np.ones(M), 'valid') / M;  type1_s3 = type1_s3.reshape(len(type1_s3),1)
  type1_s4 = np.convolve(type1_['s4'], np.ones(M), 'valid') / M;  type1_s4 = type1_s4.reshape(len(type1_s4),1)
  
  type2_s1 = np.convolve(type2_['s1'], np.ones(M), 'valid') / M;  type2_s1 = type2_s1.reshape(len(type2_s1),1)
  type2_s2 = np.convolve(type2_['s2'], np.ones(M), 'valid') / M;  type2_s2 = type2_s2.reshape(len(type2_s2),1)
  type2_s3 = np.convolve(type2_['s3'], np.ones(M), 'valid') / M;  type2_s3 = type2_s3.reshape(len(type2_s3),1)
  type2_s4 = np.convolve(type2_['s4'], np.ones(M), 'valid') / M;  type2_s4 = type2_s4.reshape(len(type2_s4),1)
  
  type3_s1 = np.convolve(type3_['s1'], np.ones(M), 'valid') / M;  type3_s1 = type3_s1.reshape(len(type3_s1),1)
  type3_s2 = np.convolve(type3_['s2'], np.ones(M), 'valid') / M;  type3_s2 = type3_s2.reshape(len(type3_s2),1)
  type3_s3 = np.convolve(type3_['s3'], np.ones(M), 'valid') / M;  type3_s3 = type3_s3.reshape(len(type3_s3),1)
  type3_s4 = np.convolve(type3_['s4'], np.ones(M), 'valid') / M;  type3_s4 = type3_s4.reshape(len(type3_s4),1)
  
  normal_temp = np.concatenate((normal_s1, normal_s2, normal_s3, normal_s4), axis =1)
  type1_temp  = np.concatenate((type1_s1, type1_s2, type1_s3, type1_s4), axis =1)
  type2_temp  = np.concatenate((type2_s1, type2_s2, type2_s3, type2_s4), axis =1)
  type3_temp  = np.concatenate((type3_s1, type3_s2, type3_s3, type3_s4), axis =1)
  ```

* Min-Max scaler

  ```python
  scaler = MinMaxScaler()
  scaler.fit(normal_)
  normal = scaler.transform(normal_temp)
  type1  = scaler.transform(type1_temp)
  type2  = scaler.transform(type2_temp)
  type3  = scaler.transform(type3_temp)
  ```

* Preprocessed data

  <img src="https://github.com/HanMinung/DLIP/assets/99113269/0e21aa7f-c5c8-4725-b44a-dbba5bc1e5bd" alt="image" style="zoom:80%;" />

  기존의 140,000개의 행에서 14개가 빠진 형태인데, 이는 처음 14개의 값들이 평균구하는데 활용이 되기 때문이다.

* Data labeling

  <img src="https://github.com/HanMinung/DLIP/assets/99113269/957418ac-b9c3-4ede-b1b4-3f6748d98091" alt="image" style="zoom:80%;" />

  ```python
  train_label = np.concatenate((np.full((60000,1),0), np.full((60000,1),1), np.full((60000,1),2), np.full((60000,1),3)))
  valid_label = np.concatenate((np.full((20000,1),0), np.full((20000,1),1), np.full((20000,1),2), np.full((20000,1),3)))
  test_label  = np.concatenate((np.full((20000,1),0), np.full((20000,1),1), np.full((20000,1),2), np.full((20000,1),3)))
  ```

  ```python
  # data shuffle
  idx = np.arange(train.shape[0]); np.random.shuffle(idx)
  train = train[:][idx]; train_label = train_label[:][idx]
  
  idx_v = np.arange(valid.shape[0]); np.random.shuffle(idx_v)
  valid = valid[:][idx_v]; valid_label = valid_label[:][idx_v]
  
  idx_t = np.arange(test.shape[0]); np.random.shuffle(idx_t)
  test = test[:][idx_t]; test_label = test_label[:][idx_t]
  ```

  * 여기까지 진행하게 되면, index는 shuffle 되어서 랜덤하게 배정이 되었으며, 각 라벨링 된 데이터도 같은 인덱스로 배정된다.





## 3. Model selection

### 3.1 .Background

라벨링 된 데이터이기 때문에 지도학습에 해당하고, 지도학습에 효과적인 모델을 선정하는 것이 중요하다.

* DNN (Deep Neural Network)

  CNN에서 모델 내 은닉층을 많이 늘려서 학습의 결과를 향상시키는 방법

  <img src="https://github.com/HanMinung/DLIP/assets/99113269/b04ea808-b735-4afb-9c7f-5d64bc793da2" alt="image" style="zoom:80%;" />

* CNN (Convolutional Neural Network)

  주로 Matrix 데이터나 이미지 데이터에 대하여 특징을 추출하는데 사용

* RNN (Recurrent Neural Network)

  인공신경망 종류 중 하나로, 과거의 학습을 가중치를 통해 현재 학습에 반영하는 특징을 가진다. 진동데이터와 같은 순차적인 데이터 학습에 사용된다.

  

### 3.2. Data shape conversion

* Pytorch 머신러닝 모듈 : tensor 형태의 데이터를 입력으로 받기 때문에, 데이터의 형태를 array에서 tensor로 변환

* torch.from_numpy 함수 : pytorch 텐서로 변환

  ```python
  x_train = torch.from_numpy(train).float()
  y_train = torch.from_numpy(train_label).float().T[0]
  x_valid = torch.from_numpy(valid).float()
  y_valid = torch.from_numpy(valid_label).float().T[0]
  x_test = torch.from_numpy(test).float()
  y_test = torch.from_numpy(test_label).float().T[0]
  ```

  ```python
  train = TensorDataset(x_train, y_train)
  train_dataloader = DataLoader(train, batch_size =5000, shuffle=True)
  valid = TensorDataset(x_valid, y_valid)
  valid_dataloader = DataLoader(valid, batch_size =len(x_valid), shuffle=False)
  test = TensorDataset(x_test, y_test)
  test_dataloader = DataLoader(test, batch_size =len(x_valid), shuffle=False)
  ```

* Pytorch에는 Dataset과 DataLoader라는 모듈이 있어서 미니 배치 학습이나 데이터 셔플 등의 기능을 제공해준다.

* TensorDataset : 데이터 x_data와 레이블 y_data를 묶어 놓는 컨테이너이다. 

* DataLoader : TensorDataset의 결과가 전달되는데, 이를 통해 batch size나 데이터를 섞을지 여부 등을 결정해 준다.

* training data의 경우 학습을 위해 batch size를 5000으로, 학습을 반복할 때마다 데이터를 다시 섞어주기 위해 shuffle을 사용하였다.



### 3.3. DNN model

* 모델 내 은닉층을 많이 늘려 학습의 결과를 향상시키는 방법

* Supervised classification, regression에서 주로 활용

  ```python
  class KAMP_DNN(nn.Module):
      
   def __init__(self):
   	super(KAMP_DNN, self).__init__()
   	self.layer1 = nn.Linear(in_features =4, out_features =100)
  	self.layer2 = nn.Linear(in_features =100, out_features =100)
   	self.layer3 = nn.Linear(in_features =100, out_features =100)
   	self.layer4 = nn.Linear(in_features =100, out_features =4)
   
   	self.dropout = nn.Dropout(0.2)
   	self.relu = nn.ReLU()
      
   def forward(self, input):
      
   	out =self.layer1(input)
   	out =self.relu(out)
   	out =self.dropout(out)
   
   	out =self.layer2(out)
   	out =self.relu(out)
   	out =self.dropout(out)
   
   	out =self.layer3(out)
   	out =self.relu(out)
   	out =self.dropout(out)
   
   	out =self.layer4(out)
      
   return out
   
  model_check = KAMP_DNN()
  print(model_check)
  ```

* __init__에서 레이어를 초기화

  * ###### nn.Linear() : 선형계층으로 weight와 bias을 사용하여 입력에 선형 변환을 적용하는 모듈

    * in_features : 신경망으로 입력되는 size
    * out_features : 신경망에서 출력되는 size

* nn.Dropout() : 드롭아웃 계층으로 신경망 일부를 의도적으로 학습에서 제외함으로써 모델을 일반화하는 역할

<img src="https://github.com/HanMinung/DLIP/assets/99113269/5c25d638-c7e5-4b08-b1ed-0729ce6fbb6f" alt="image" style="zoom:80%;" />

* forward 메서드 : 초기화된 layer에 입력 데이터를 전달하고 최종 출력을 얻는다

* Algorithm structure

  * Layer 1 : Layer 1은 입력 특징이 4개이고, 출력 특징이 100개인 fully connected layer이다. 이는 각 입력 특징들을 100개의 뉴런으로 변환하는 과정으로, 입력 데이터로부터 다양한 특징을 추출하는 역할을 한다. 따라서 여기서는 입력으로부터 feature extraction이 이루어진다.

  * Layer 2 : 마지막 Layer는 100개의 입력 특징을 받아 4개의 출력 특징으로 변환한다. 이 부분에서는 최종적으로 추출된 특징들을 각 클래스에 대한 확률로 변환하는 작업이 이루어진다. 따라서 이 부분은 최종적인 classification을 수행한다.

  * Dropout : Dropout은 모델이 과적합(overfitting)되는 것을 방지하기 위한 regularization 기법 중 하나이다. 안에 들어가는 실수의 확률로 뉴런을 제외하면서 과적합을 방지한다.

  * RELU activation function : 입력값이 양수인 경우 --> 그대로 출력 | 음수인 경우 --> 0을 출력

    주된 사용이유 : 네트워크에 비선형성을 추가하여 다양한 특징들을 학습할 수 있게 하기위함. vanishing gradient 문제를 완화하는데 도움을 준다.



### 3.4. CNN model

* Matrix data or 이미지 데이터에서의 특징 추출에 쓰인다.

* ```python
  class KAMP_CNN(nn.Module):
      
   def __init__(self):
      
   	super(KAMP_CNN, self).__init__()
   	self.conv1 = nn.Sequential(
   	nn.Conv1d(in_channels = 1, out_channels=100, kernel_size=2, stride=1, padding='same'),
   	nn.BatchNorm1d(100),
   	nn.ReLU(),
   	nn.MaxPool1d(kernel_size=1, stride=1),
   	nn.Dropout(p=0.2))
   
      # self.conv2 = nn.Sequential(
      # nn.Conv1d(in_channels=100, out_channels=100, kernel_size=2, stride=1, padding='same'),
      # nn.BatchNorm1d(100),
      # nn.ReLU(),
      # nn.MaxPool1d(kernel_size=1, stride=1),
      # nn.Dropout(p=0.2))
  
      # self.conv3 = nn.Sequential(
      # nn.Conv1d(in_channels=100, out_channels=100, kernel_size=2, stride=1, padding='same'),
      # nn.BatchNorm1d(100),
      # nn.ReLU(),
      # nn.MaxPool1d(kernel_size=1, stride=1),
      # nn.Dropout(p=0.2))
   
      self.conv4 = nn.Sequential(
      nn.Conv1d(in_channels=100, out_channels=4, kernel_size=2, stride=1, padding='same'),
      nn.BatchNorm1d(4),
      nn.ReLU(),
      nn.MaxPool1d(kernel_size=1, stride=1))
   
      self.final_pool = nn.AdaptiveAvgPool1d(1)
      self.linear = nn.Linear(4, 4)
      
  def forward(self, input):
          
      input = input.unsqueeze(1)
      out =self.conv1(input)
      # out = self.conv2(out)
      # out = self.conv3(out)
      out =self.conv4(out)
      out =self.final_pool(out)
      out =self.linear(out.squeeze(-1))
      
      return out
      
  model_check = KAMP_CNN()
  print(model_check)
  ```

* Algorithm structure

  * in channels : 신경망으로 입력되는 데이터의 channel size
  * out channels : convolution을 수행한 feature extraction을 진행한 뒤, 생성되는 feature map의 두께
  * Batch normalization : Batch Normalization은 네트워크의 안정성을 높이고 학습 속도를 가속화하는 데 도움을 주는 정규화 기법. `nn.BatchNorm1d`는 1D 입력에 대한 Batch Normalization을 수행.
  * Max pooling : Max Pooling은 주어진 영역에서 가장 큰 값을 선택하여 다운샘플링하는 연산이다. 이를 통해 공간 차원을 줄이고 중요한 특징을 강조하게 된다. 
  * 결국, 각 convolutional layer는 1D convolution + batchnormalization + activation function + pooling 기법에 대한 정보를 포함하고 있다. 
  * Final layer
    * `nn.Conv1d`: 100개의 입력 feature map에서 4개의 출력 feature map으로 변환하는데, 이 과정에서 각 클래스에 대한 특징을 추출한다.
    * `nn.AdaptiveAvgPool1d(1)`: Adaptive Average Pooling을 통해 각 feature map의 평균을 계산한다. 이는 각 클래스에 대한 특징을 평균화하여 클래스를 구분하는 데 도움을 준다.



### 3.5. RNN model

* RNN, LSTM, GRU는 과거의 학습을 가중치를 통해 현재 학습에 반영하는 특징이 존재한다. 따라서, 순차적으로 변화하는 시계열 데이터의 학습에 탁월한 효과를 갖는다.

* Pytorch의 LSTM을 활용하면 쉽게 활용가능

* ```python
  class KAMP_RNN(nn.Module):
      
   def __init__(self):
      
   	super(KAMP_RNN, self).__init__()
   	self.lstm = nn.LSTM(input_size =4, hidden_size =100, num_layers =2, batch_first=True, dropout =0.2)
   	self.fc = nn.Linear(in_features =100, out_features =4)
   
   def forward(self, input):
          
   	input = input.unsqueeze(1)
   	out, _ =self.lstm(input)
   	out = out.view(-1,100)
   	output =self.fc(out)
      
   	return output
   
  model_check = KAMP_RNN()
  print(model_check)
  ```

* LSTM : long short - memory network 주로 시계열 데이터 처리에 많이 사용된다. 

* input shape : torch.nn.LSTM의 입력 사이즈는 [Batch size, Sequence length, Input size]이며, 학습 데이터에 따라 입력사이즈를 조절하면 된다.

  [Batch size = 80000, sequence length = 1, input size = 4]



### 3.6. Model training

```python
def train_model(model, criterion, optimizer, num_epoch, train_dataloader, PATH):
    
    loss_values = []
    loss_values_v = []
    check =0
    accuracy_past = 0
    
    for epoch in range(1, num_epoch +1) :
        
        model.train()
        batch_number =0
        running_loss =0.0
        
        for batch_idx, samples in enumerate(train_dataloader) :
            
            x_train, y_train = samples
            optimizer.zero_grad()
            y_hat = model.forward(x_train)
            loss = criterion(y_hat,y_train.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_number +=1
        
        loss_values.append(running_loss / batch_number)

    #---------------------- 모델 검증 ---------------------#
    model.eval()
    accuracy =0.0
    total =0.0
    
    for batch_idx, data in enumerate(valid_dataloader) :
        
        x_valid, y_valid = data
        
        v_hat = model.forward(x_valid)
        v_loss = criterion(v_hat,y_valid.long())
        _, predicted = torch.max(v_hat.data, 1)
        total += y_valid.size(0)
        accuracy += (predicted == y_valid).sum().item()
    
    loss_values_v.append(loss.item())
    accuracy = (accuracy / total)
    
    #----------------Check for early stopping---------------#
    if epoch % 1 ==0:
        print('[Epoch {}/{}] [Train_Loss: {:.6f} /Valid_Loss: {:.6f}]'.format(epoch, num_epoch, loss.item(),v_loss.item()))
        print('[Epoch {}/{}] [Accuracy : {:.6f}]'.format(epoch, num_epoch, accuracy)) 
        
    if accuracy_past > accuracy:
        check +=1  
        
    else:
        check =0
        accuracy_past = accuracy 
            
    if check >50:
        print('This is time to do early stopping')        

    torch.save(model, PATH +'model.pt')
    
    return loss_values, loss_values_v
```

* criterion : loss 계산을 위한 함수 (cross entropy loss)

  * label이 존재하는 supervised learning에서 기계학습 모델은 예측값(y_hat)이 실제값(label)에 가까워지도록 학습이 이루어진다. 이때의 차이를 loss로 활용

  <img src="https://github.com/HanMinung/DLIP/assets/99113269/73f8eb2c-842e-4a84-81d6-a1d4c672dc49" alt="image" style="zoom:80%;" />

* optimizer : Adam을 활용

  * 기계학습에서 네트워크 오차를 줄이고 알고리즘의 정확도를 높이기 위해 모델의 가중치를 변화시켜주는 역할을 수행

  <img src="https://github.com/HanMinung/DLIP/assets/99113269/1d57ea51-7982-4b02-8721-7a72fc02ba2f" alt="image" style="zoom:80%;" />

* train_dataloader : 전처리가 완료된 데이터

* 출력 데아터 : loss_values (학습 손실 값) & loss_values_v (검증 손실 값)



```python
def train_model(model, criterion, optimizer, num_epoch, train_dataloader, PATH):
    
    loss_values = []
    loss_values_v = []
    check = 0
    accuracy_past = 0
    
    for epoch in range(1, num_epoch +1):
        
        #---------------------- 모델 학습 ---------------------#
        model.train()
        batch_number = 0
        running_loss = 0.0
        
        for batch_idx, samples in enumerate(train_dataloader):
            
            x_train, y_train = samples
            
            optimizer.zero_grad()
            y_hat = model.forward(x_train)
            loss = criterion(y_hat,y_train.long())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_number +=1
            loss_values.append(running_loss / batch_number)
```

* model.train() : training 과정에서 사용해야 하는 layer들을 알아서 사용하도록 하는 함수
* train_dataloader : 학습데이터는 배치를 나누어 사용하게 되며, enumerate() 함수를 통해 batch index(batch_idx)와 해당 데이터(samples)를 동시에 사용한다. 그 결과 학습에 사용되는 데이터는 각 배치별(x_train, y_train)이다.
* loss_values.append(running_loss / batch_number) : training loss 값을 저장



### 3.7. Training of 3 models

```python
DNN_model = KAMP_DNN()
num_epochs =1000
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(DNN_model.parameters())
PATH ='save/DNN/'
DNN_loss_values, DNN_loss_values_v = train_model(DNN_model, criterion, optimizer,
num_epochs, train_dataloader, PATH)
```

```python
CNN_model = KAMP_CNN()
num_epochs =1000
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNN_model.parameters())
PATH ='save/CNN/'
CNN_loss_values, CNN_loss_values_v = train_model(CNN_model, criterion, optimizer,
num_epochs, train_dataloader, PATH)
```

```python
RNN_model = KAMP_RNN()
num_epochs =1000
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(RNN_model.parameters())
PATH ='save/RNN/'
RNN_loss_values, RNN_loss_values_v = train_model(RNN_model, criterion, optimizer,
num_epochs, train_dataloader, PATH)
```



## 4. Performance evaluation

```python
def test_model(model, PATH):
    model = torch.load(PATH +'model.pt')
    #---------------------- 모델 시험 ---------------------#
    model.eval()
    total =0.0
    accuracy =0.0
    
    for batch_idx, data in enumerate(test_dataloader):
        x_test, y_test = data
        t_hat = model(x_test)
        _, predicted = torch.max(t_hat.data, 1)
        total += y_test.size(0)
        accuracy += (predicted == y_test).sum().item()
    accuracy = (accuracy / total)
    #------------------------------------------------------#
    print(accuracy)
```

* 모델의 성능을 평가할 때에는 다음 두 가지를 봐야 한다.
  * epoch에 따른 model의 loss 개선
  * confusion matrix




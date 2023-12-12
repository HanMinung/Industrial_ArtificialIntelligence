# 로봇 용접 예지보전 AI 데이터 분석

@ author : Han Minwoong

@ School of mechanical and control engineering

@ Handong global university

@ subject : PHM of welding robot based on vibration & electric current datas



## 1. Problem statement

* 대상 공정은 엘리베이터 비상정지 장치 제조공정 중 로봇 용접 공정으로 로봇 용접기는 핵심 설비이다.
* 공정상의 문제 현황
  1) 현재, 로봇 용접기 등 공정 핵심 설비의 고장징후를 이상 소음 발생 등 관리자의 운영 경험에 의존해 대응하는 수준으로, 해당 소음과 이상징후의 직접적인 관련성을 판단하기 어렵다.
  2) 또한, 다양한 형태의 용접 불량상황이 발견되고 있지만 이에 대한 사후대처도 현장 관리자의 경험에 전적으로 의존하는 상황이다.
  3) 주요 설비의 원활한 운영을 위해서는 객관적인 데이터에 근거한 정확한 예측을 통해 설비의 가동 중단에 따른 생산 손실을 줄일 수 있는 대책이 요구된다.
  4) 결과적으로, 로봇 용접기의 고장 예방과 품질 불량을 감지할 수 있는 AI 기반 공정 감시 시스템 구축이 필요하다.
* 로봇 용접기의 고장과 품질 불량은 다양한 요인으로 발생하기 때문에, 상대적으로 성능이 우수한 지도학습 알고리듬을 적용하기 어렵다.



## 2. Data anlaysis

* ###### 센서 부착 위치

  * 진동 센서 : 로봇 용접기에 부착
  * 전류 센서 : 로봇 용접기의 주 전원선에 부착

* 정상 진동 데이터 및 비정상 진동 데이터

  <img src="https://github.com/HanMinung/NumericalProgramming/assets/99113269/e24cba25-37ba-4e2c-8fe1-5036dcff1526" alt="image" style="zoom: 80%;" />

* 진동 데이터 : **3200 Hz**로 측정된다.

  * 0.3초 동안 수집된 512개의 진동 측정치로 구성된다. (주파수 도메인)

* 정싱 전류 데이터 및 비정상 전류 데이터

  <img src="https://github.com/HanMinung/NumericalProgramming/assets/99113269/f5d0341a-f895-4aa4-ad7b-5f333a747313" alt="image" style="zoom:80%;" />

* 전류 데이터 : **4000 Hz**로 측정된다.

  * 약 0.5초 동안 수집된 1024개의 측정치로 구성된다. (주파수 도메인)

* trainset

  * vibration_normal.csv : 정상 진동데이터
  * vibration_anomaly.csv : 비정상 진동데이터

* current_normal.csv : 정상 전류데이터

  * current_anomaly.csv : 비정상 전류데이터

  

## 3.  Data standardization

* AI 모델의 학습, 평가를 위해서는 학습용 데이터셋에 대한 표준화를 진행하고, 평가데이터셋에 이를 적용해야 한다.

* 표준화는 모든 학습 데이터를 평균 0, 분산 1인 정규분포로 변환하는 것으로 sklearn.preprocessing 패키지에 포함된 StandardScaler 객체를 이용한다.

* new value = (x-mean)/(standard deviation)

* Data standardization : Standardization is another scaling method where the values are centered around the mean with a unit standard deviation. This means that the mean of the attribute becomes zero, and the resultant distribution has a unit standard deviation.

* fit transform method : 인자로 입력된 데이터셋의 평균과 표준편차를 구하고, 이를 이용해 표준화된 데이터셋을 반환

  <img src="https://github.com/HanMinung/DLIP/assets/99113269/3efb615b-a457-46bb-aaf2-73a9bcfde052" alt="image" style="zoom:67%;" />

  <img src="https://github.com/HanMinung/DLIP/assets/99113269/2115b008-bf2e-4658-8e27-375c3fed095c" alt="image" style="zoom: 80%;" />

  

  



## 4. Model 

* 사용된 모델 : convolutional auto encoder
  * 입력 : data size, time setp, features
  * 따라서, train set & test set의 형태를 3차원으로 변경해주어야 한다. 



* 



## 5. adda







## reference

https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/




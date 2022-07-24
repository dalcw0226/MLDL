# 슬롯 머신 문제
# 슬롯 머신 문제는 강화학습에서의 첫번째 주제이다.
# 5개의 슬롯 머신에서 가장 좋은 슬롯머신을 찾는다.

# 라이브러리 임포트
# 라이브러리는 numpy 한가지만 이용한다.
import numpy as np

# 시뮬레이터를 만든다.
# 시뮬레이터는 모델을 훈련시킬 훈련데이터들을 의미한다.
conversionRate = [0.15, 0.04, 0.13, 0.11, 0.05]  # 5개의 슬롯머신이 돈을 줄 확률을 담고 있다.
N = 2000 # 학습을 시킬 데이터의 수 = 데이터의 행을 의미한다.
d = len(conversionRate) # 슬롯머신의 개수를 가지고 있다 = 데이터의 열을 의미한다.


# 슬롯머신의 승패의 결과를 정의
# 지금 만들 데이터가 훈련을 위한 데이터이다.
X = np.zeros((N, d))  # (행, 열) = (N, d), 이게 무슨 행렬이냐면 학습용 데이터를 만들기 위한 행렬이다.
for i in range(N):
    for j in range(d):
        if np.random.rand() < conversionRate[j]:  # 랜덤값을 반환을 해서 슬롯머신이 가지고 있는 전환률(확률)보다 작다면 해당 슬롯머신에서는 승리를 할 수 있다.
            X[i][j] = 1  # 그 슬롯 머신에서 게임을 한다면 돈을 얻을 수 있다는 표시인 1을 남겨준다.
            
# 슬롯머신마다 게임한 결과를 기록할 배열을 선언한다.
nPosReward = np.zeros(d)  # 승리한 경우, 즉 보상을 받은 경우가 된다.
nNegReward = np.zeros(d)  # 패배한 경우, 즉 보상을 받지 못한 경우가 된다.

# 실제로 모델을 생성하고 학습까지 진행해보자
# 이전 인공지능에서 학습했던 방식과는 조금 느낌이 다르다.
# 모델이 이긴 경우에는 보상을 받고, 모델이 진 경우에는 보상을 받지 못한다.
for i in range(N):  # 행을 이동하기 위해 만든 반복문
    selected = 0  # 어느 슬롯멋니이 선택되었는지 기록한다.
    maxRandom = 0  # 가장 높은 베타분포의 결과값을 기록한다.
    for j in range(d):
        randomBeta = np.random.beta(nPosReward[j]+1, nNegReward[j]+1)
        # ß 분포에 따라서 랜덤으로 갑을 추출해낸다.
        # ß 분포를 따른다면, 첫번째인수의 값이 클수록 더 큰 확률값을 출력해 내고, 두번째 인수의 값이 클 수록 더 작은 값을 추출해 낸다.
        # 따라서, 슬롯머신에서 승리할 확률이 높은 머신에는 ß분포의 출력 값이 크게 나온다.
        # ß분포 가장 큰 슬롯머신을 선택해야한다.
        if maxRandom < randomBeta:
            maxRandom = randomBeta
            selected = j
    # nPosReward와 nNegReward를 업데이트 한다.
    if X[i][selected] == 1:
        nPosReward[selected] += 1
    else:
        nNegReward[selected] += 1

# 굳이 한해도 상관없지만, 모델이 학습하는 상황을 지켜보기 위해서 출력을 해준다.
nSelected = nPosReward + nNegReward
for i in range(d):
    print(f"Machine number {i+1} was selected {nSelected[i]} times")
print(f"Conclusion : Best machine is machine number {np.argmax(nSelected)+1}")
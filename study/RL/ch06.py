# 실제 예시를 적용해 보자!
# 쇼핑몰에서 가장 적당한 방법을 이용해서 사람을 유치시키기 

# 톰슨 모델의 작동원리
# 1. 베타분포에서 무작위 값을 뽑는다.
# 2. 가장 높은 무작위 값을 갖는 전략을 선택한다.
# 3. 결과값과 일치하면 보상을 주고 아닌경우에는 처벌을 한다.

# 여기서도 중요한 개념이 베터 분포이다.

# 모델 생성 - 22.07.25
# 이번에는 저번 예시와는 조금 차이를 둬서 모델의 평가까지 함께한다.

# 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt
import random

# 매개변수 생성
N = 10000 # 고객 10000명
d = 9 # 9개의 전략

# 시뮬레이터 구성
conversion_rates = [0.05, 0.13, 0.09, 0.16, 0.11, 0.04, 0.20, 0.08, 0.01]
X = np.array(np.zeros([N, d]))
for i in range(N):
    for j in range(d):
        if np.random.rand() <= conversion_rates[j]:
            X[i, j] = 1

# 모델의 성능평가와 가장 좋은 전략을 판단하기 위한 변수 선언
strategies_selected_rs = [] # 무작위 알고리즘에 의해 선택된 전략
strategies_selected_ts = [] # 톰슨 모델에 의해 선택된 전략
total_reward_rs = 0 # 무작위 알고리즘에 의해 누적된 보상
total_reward_ts = 0 # 톰슨 모델에 의해 누적된 보상
numbers_of_rewards_1 = [0] * d # 보상으로 1을 받은 횟수
numbers_of_rewards_0 = [0] * d # 보상으로 0을 받은 횟수


# 모델 작동
for n in range(N):
    # 무작위 알고리즘 작동
    strategy_rs = random.randrange(d)
    strategies_selected_rs.append(strategy_rs)
    # 개인적으로 굉장하다고 생각하는 알고리즘
    reward_rs = X[n, strategy_rs]
    total_reward_rs = total_reward_rs + reward_rs


    # 톰슨 모델의 샘플링
    strategy_ts = 0
    max_random = 0
    for i in range(d):
        random_beta = np.random.beta(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)

        if random_beta > max_random:
            max_random = random_beta
            strategy_ts = i

    # 보상과 처벌 업데이트
    reward_ts = X[n, strategy_ts]
    if reward_ts == 1:
        numbers_of_rewards_1[strategy_ts] = numbers_of_rewards_1[strategy_ts] + 1
    else:
        numbers_of_rewards_0[strategy_ts] = numbers_of_rewards_0[strategy_ts] + 1

    # 선택한 전략을 전략리스트에 추가한다
    strategies_selected_ts.append(strategy_ts)
    # 총 보상 계산하기
    total_reward_ts = total_reward_ts + reward_ts

# 상태 출력
nSelected = np.array(numbers_of_rewards_1) + np.array(numbers_of_rewards_0)
for i in range(d):
    print(f"Stratagy number {i+1} was selected {nSelected[i]} times")
print(f"Conclusion : Best Stratagy is number {np.argmax(nSelected)+1}")

# 상대 수익률 계산
relative_return  = (total_reward_ts - total_reward_rs) / total_reward_rs * 100
print(f"relative return : {relative_return}")

num_list = [0] * 9
for i in strategies_selected_ts:
    num_list[i] += 1
print(num_list)
# 물류창고에서 일하는 로봇
# 우리의 목표는 로봇의 위치가 어디있던지 간에 최단경로를 반환해야한다.

# 환경구성
# 상태 정의 : 시간 t 에서 롭소이 있는 위치
# 행동 정의 : 로봇이 갈 수 있는 다음 목적지이다.
# 보상 정의 : 보상행렬을 만든다.

# AI 모델
# 1. 초기화 : 모든 Q 값을 0으로 초기화 한다.
# 2. 반복
    # (1) 임의의 상태 si 를 선택한다.
    # (2) 다음 이어질 수 있는 행동을 수행한다.
    # (3) 다음 상태에 이르고 보상으로 R을 얻는다
    # (4) TD를 계산한다.
    # (5) Q 값을 업데이트 한다.


# 라이브러리 임포트
import numpy as np

# Q 러닝에 사용할 매개변수를 설정한다.
gamma = 0.75
alpha = 0.9

# part 1 - 환경구성

# 상태정의
location_to_state = {
    'A' : 0,
    'B' : 1,
    'C' : 2,
    'D' : 3,
    'E' : 4,
    'F' : 5,
    'G' : 6,
    'H' : 7,
    'I' : 8,
    'J' : 9,
    'K' : 10,
    'L' : 11
}

# 행동 정의
actions = [0,1,2,3,4,5,6,7,8,9,10,11]

# 보상 행렬 정의
R = np.array([
    [0,1,0,0,0,0,0,0,0,0,0,0],
    [1,0,1,0,0,1,0,0,0,0,0,0],
    [0,1,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0],
    [0,1,0,0,0,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,1000,1,0,0,0,0],
    [0,0,0,1,0,0,1,0,0,0,0,1],
    [0,0,0,0,1,0,0,0,0,1,0,0],
    [0,0,0,0,0,1,0,0,1,0,1,0],
    [0,0,0,0,0,0,0,0,0,1,0,1],
    [0,0,0,0,0,0,0,1,0,0,1,0],
])

# part 2 - Q 러닝으로 AI 솔루션 구성

# Q 값 초기화
Q = np.array(np.zero([12,12]))

# Q 러닝 프로세스 구현
for i in range(1000):
    current_state = np.random.randint(0, 12)  # 랜덤으로 한개의 정수를 현재 상태로 정의한다.
    playable_actions = []
    for j in range(12):
        if R[current_state, j] > 0:
            playable_actions.append(j)
    next_state = np.random.choice(playable_actions)  # 선택가능한 것들 중 랜덤으로 1개를 정한다.
    TD = R[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state, ])] - Q[current_state, next_state]
    Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD

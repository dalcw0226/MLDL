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
Q = np.array(np.zeros([12,12]))

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

print("Q-values : ")
print(Q.astype(int))


# part 3 - 운영 시작(추론모드)
# 시작 위치에서 최우선 순위 위치까지 최적의 경로를 계산한다.
# route 함수를 구현한다.

# 위치를 인덱스 대신 문자열로 입력할 것이기 때문에 내부적으로 위치 인덱스를 문자에 매핑한 딕셔너리가 필요하다.

# 인덱스에서 문자로 매핑 생성
state_to_location = {state : location for location, state in location_to_state.items()}  # 리스트 컴프리헨션 기법

# 최적 경로를 반환하는 최종 함수를 생성한다.
def route(starting_location, ending_location):
    route = [starting_location]
    next_location = starting_location
    while (next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state, ])
        # print(np.argmax(Q[starting_state, ]))
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route

# 최종 경로 출력
print('Route : ')
print(route('E', 'G'))

# 여기서 드는 의문! 이 강화학습 알고리즘을 백준 최단 경로 문제풀이에 사용할 수 있을까?
# 또한 강화학습으로 푼 최단 경로 문제와 다익스트라 알고리즘간의 차이는 무엇일까?
# 두개의 결과가 같다면 다익스트라 알고리즘이 효율성 측면에서는 더 좋은 것 아닐까?

# 개선 - 1
# 최우선순위에 있는 위치에 높은 보상을 부여하는 것을 자동화해 수작업을 없애는 것

# route 함수를 개선하기
def route(starting_location, ending_location):
    # 복사를 왜 하는가?
    # 원본을 훼손하면 안된다. => 나중에 ending_location이 변경되었을때를 대비해야한다.
    R_new = np.copy(R)
    ending_state = location_to_state[ending_location]
    R_new[ending_location, ending_location] = 1000 # => ending_location에 Agent가 몀추게 된다.

    # Q러닝 전체 알고리즘을 포함시켜야 한다.
    Q = np.array(np.zeros([12,12]))
    for i in range(1000):
        current_state = np.random.randint(0, 12)  # 랜덤으로 한개의 정수를 현재 상태로 정의한다.
        playable_actions = []
        for j in range(12):
            if R[current_state, j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)  # 선택가능한 것들 중 랜덤으로 1개를 정한다.
        TD = R[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state, ])] - Q[current_state, next_state]
        Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD
    route = [starting_location]
    next_location = starting_location
    while (next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state, ])
        # print(np.argmax(Q[starting_state, ]))
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route

# 개선  - 2
# 중간 목표 추가 : 문제를 풀이하기 위해서는 크게 3가지 방법이 있다.

# 해법 1, 2
# J에서 K로 이끄는 보상을 제공한다.
# 이 보상은 1 ~ 1000 사이여야한다.

# 보상을 주거나, 처벌을 주어서 해결한다.
# 단 자동화를 하는 측면에서는 어려움이 있다.

# 따라서 해법 3을 생각한다.
# 시작, 중간, 종류 위치의 세 개의 입력을 취하는 추가적인 best_route()함수를 만든다.

# 최적의 경로를 반환하는 최종 함수 생성
def best_route(starting_location, intermediary_location, ending_location):
    return route(starting_location, intermediary_location) + route(intermediary_location, ending_location)[1:]

print("Route : ")
print(best_route('E', 'K', 'G'))
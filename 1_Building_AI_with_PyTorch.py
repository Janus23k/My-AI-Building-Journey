import torch
import torch.nn as nn
import torch.optim as optim

# 1. 데이터 준비 (공부시간, 잠잔시간)
# [공부시간, 잠잔시간] -> [합격 여부(1=합격, 0=불합격)]
X = torch.tensor([[10, 5], [2, 9], [8, 4], [1, 8]], dtype=torch.float32)
y = torch.tensor([[1], [0], [1], [0]], dtype=torch.float32)

# 2. 아주 간단한 '인공 뇌' 설계하기
class SimpleBrain(nn.Module):
    def __init__(self):
        super(SimpleBrain, self).__init__()
        # 입력(2개) -> 중간(5개 신경세포) -> 출력(1개)
        self.layer1 = nn.Linear(2, 5) 
        self.layer2 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid() # 0~1 사이 확률로 변환

    def forward(self, x):
        x = torch.relu(self.layer1(x)) # 첫 번째 층 통과 (ReLU 스위치)
        x = self.sigmoid(self.layer2(x)) # 두 번째 층 통과 후 확률 출력
        return x

model = SimpleBrain()

# 3. 어떻게 공부할지 정하기 (학습 설정)
criterion = nn.BCELoss() # 틀린 정도를 계산하는 공식 (Loss)
optimizer = optim.SGD(model.parameters(), lr=0.1) # 틀렸을 때 수정하는 도구 (Backpropagation)

# 4. 학습 시작 (1,000번 반복 공부)
print("학습 중...")
for epoch in range(1000):
    optimizer.zero_grad()    # 기록 초기화
    output = model(X)        # 1) 예측하기 (Forward)
    loss = criterion(output, y) # 2) 얼마나 틀렸나 계산
    loss.backward()          # 3) 뒤로 가며 수정사항 찾기 (Backward)
    optimizer.step()         # 4) 실제로 가중치 수정

print("학습 완료!\n")

# 5. 새로운 데이터로 테스트해보기
# 질문: "7시간 공부하고 6시간 잔 친구는 합격할까?"
test_data = torch.tensor([[7, 6]], dtype=torch.float32)
prediction = model(test_data)

print(f"예측 결과(확률): {prediction.item():.4f}")
if prediction.item() > 0.5:
    print("AI: 이 친구는 합격할 것 같습니다! 🎉")
else:
    print("AI: 아쉽지만 불합격할 것 같네요. 😢")
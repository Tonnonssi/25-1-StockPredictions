import torch
import torch.nn as nn
import torch.nn.functional as F

class ARNet(nn.Module):
    """
    Autoregressive Neural Network (ARNet)

    이 모델은 간단한 선형 autoregressive 모델로,
    입력된 시계열의 이전 p 시점의 값을 기반으로 다음 시점의 값을 예측한다.
    
    - 입력: (batch_size, p) 크기의 텐서 (p는 시계열의 lookback window 크기)
    - 출력: (batch_size, 1) 크기의 예측값

    파라미터:
        p (int): AR 계수의 개수 (lookback 길이)
    """
    def __init__(self, p):
        super().__init__()
        self.w = nn.Linear(p, 1, bias=False)  # AR 계수만 학습 (bias 없음)

    def forward(self, x):
        y = self.w(x)
        return y


def regulated_mse_loss(pred, y, model, s, c_lam, c1, c2):
    """
    희소성 유도 정규화가 포함된 MSE 손실 함수

    이 함수는 예측 오차(MSE)와 함께, 가중치에 희소성을 유도하는 정규화 항 R(θ)를 포함한 총 손실을 계산한다.

    Loss = MSE(pred, y) + λ(s) * R(θ)

    - λ(s) = c_lam * (1/s - 1) : 사용자가 기대하는 sparsity 수준(s)을 기준으로 동적으로 조정되는 정규화 강도
    - R(θ): 가중치의 크기를 부드럽게 희소화하는 함수

    파라미터:
        pred (Tensor): 모델의 예측값
        y (Tensor): 정답값
        model (ARNet): ARNet 모델 객체
        s (float): 희소성 기대 비율 (예: 0.25 → 전체 중 25%만 유의미하길 기대)
        c_lam (float): 정규화 강도 조절 계수 (예: sqrt(MSE)/100)
        c1 (float): 정규화 함수의 경사 조절 (크면 빠르게 0/1 전이)
        c2 (float): 가중치 크기의 스케일링 지수

    반환값:
        total_loss (Tensor): 희소성 정규화가 포함된 총 손실값 (scalar)
    """
    mse_loss = F.mse_loss(pred, y)

    lamb = c_lam * (1 / s - 1)

    R_theta = torch.mean(2 / (1 + torch.exp(torch.abs(model.w.weight)**(1 / c2) * c1))) - 1

    total_loss = mse_loss + lamb * R_theta

    return total_loss
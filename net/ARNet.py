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

    Args:
        p (int): AR 계수의 개수 (lookback 길이)
    """
    def __init__(self, p):
        super().__init__()
        self.w = nn.Linear(p, 1, bias=False)  # AR 계수만 학습 (bias 없음)

    def forward(self, x):
        y = self.w(x)
        return y


class RegulatedMSELoss:
    """
    RegulatedMSELoss(s, c_lam, c1, c2)

    손실 함수 클래스로, MSE 손실에 sparsity를 유도하는 정규화 항을 추가하여
    희소한 AR 계수를 학습할 수 있도록 설계되었다.

    주로 AR-Net 모델에서 사용되며, 불필요한 시차(lag)에 해당하는 계수를 0에 가깝게 억제하는 방식으로
    모델의 해석 가능성과 계산 효율성을 높인다.

    Args:
        s (float): AR 계수의 희소성 정도를 나타내는 하이퍼파라미터.
                   데이터 생성 시 사용된 유효 계수 수 p_data와 모델의 차수 p_model의 비율로 정의됨.
                   예: s = p_data / p_model

        c_lam (float): 정규화 항의 강도를 조절하는 계수.
                       보통 c_lam ≈ sqrt(L) / 100 으로 설정하며, 추정 오차의 크기에 비례함.

        c1 (float): 작은 값(0 근처의 계수)을 강하게 억제하기 위한 파라미터.
                    sigmoid 함수의 기울기를 결정하여, 작은 계수를 빠르게 0으로 수렴시킴.

        c2 (float): 큰 값에 대한 정규화를 약하게 만들어주는 파라미터.
                    절댓값의 제곱근 루트를 조정해 큰 계수에는 덜 민감하게 작용함.


    Methods:
        __call__(pred, y, model):
            예측값(pred)과 실제값(y), 그리고 AR 계수를 포함한 모델을 입력으로 받아
            MSE 손실과 sparsity regularization이 적용된 총 손실값을 반환한다.

            Parameters:
                pred (torch.Tensor): 모델의 예측값
                y (torch.Tensor): 실제 정답값
                model (torch.nn.Module): w.weight 속성에 AR 계수를 포함한 모델

            Returns:
                total_loss (torch.Tensor): MSE 손실과 정규화 손실을 더한 총 손실값

    Example:
        >>> loss_fn = RegulatedMSELoss(s=0.1, c_lam=0.05, c1=30, c2=3)
        >>> loss = loss_fn(predictions, targets, model)
    """
    def __init__(self, s, c_lam, c1, c2):
        self.s = s
        self.c_lam = c_lam
        self.c1, self.c2 = c1, c2

    def __call__(self, pred, y, model):
        mse_loss = F.mse_loss(pred, y)

        lamb = self.c_lam * (1/self.s - 1)
    
        R_theta = torch.mean(2 / (1 + torch.exp(torch.abs(model.w.weight)**(1/self.c2) * self.c1))) - 1

        total_loss = mse_loss + lamb * R_theta

        return total_loss
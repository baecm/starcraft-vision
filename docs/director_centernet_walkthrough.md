# Director-CenterNet: Implementation Walkthrough & Technical Guide

StarCraft: Brood War 이스포츠 옵저버를 위한 **Multi-Region Viewport Prediction (MRVP)** 모델인 **Director-CenterNet**의 학습 및 추론 파이프라인 구현 상세 문서입니다.

---

## 1. 연구 배경 및 문제 정의

- **기존 한계**: 기존 단일 영역 예측(Single-Region Viewport) 모델은 다수의 인간 관찰자가 서로 다른 곳을 보고 있을 때(Multimodal attention) 이를 단순 노이즈로 취급하여 하나의 합의 영역으로 붕괴(collapse)시켰습니다. 이로 인해 두 모드 사이에서 카메라가 요동치는 thrashing 현상이 발생했습니다.
- **MRVP 재정의**: 프레임당 순위가 매겨진 $K$개의 비중복 영역을 예측하여 동시다발적인 주요 교전 및 전략적 움직임을 포괄하도록 합니다.
- **Director-CenterNet 접근법**: CenterNet 기반 dense heatmap regression을 활용하여 $3 \times 3$ max-pooling 기반의 가변 peak 추출을 수행하며, 4가지 전용 학습 손실 함수를 통해 주 영역(Top-1)과 부 영역(Top-2+)을 분리 감독합니다.

---

## 2. 시스템 아키텍처 및 파이프라인 개요

```
+-----------------------------------------------------------------------------------+
| 1. Ranked Mode Preprocessing Cache (src/dataset/mode_cache.py)                    |
|    - Uses src/metrics/modes.py::extract_modes directly                            |
|    - Caches M_t = (m_t^(1), m_t^(2), ...) with centers & supports (U=5)           |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| 2. Continuous Frame Pairing Dataset (src/dataset/starcraft_windows.py)           |
|    - pair_mode=True: fetches (X_t, X_{t+1}) windows within same replay            |
|    - Supports trajectory smoothness (L_smooth) under random batch shuffling       |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| 3. Director-CenterNet Model (src/models/backbones/director_centernet.py)          |
|    - Backbone: ResNet-50 + FPN (stride s=4)                                       |
|    - Heads: Heatmap Head, Sub-pixel Offset Head, Size Head                        |
|                                                                                   |
|    [Training Objectives] (src/losses/director_losses.py)                          |
|      * L_hcm   : Modified focal loss on Top-1 Gaussian target Y_t^(1)             |
|      * L_rmc   : Masked MSE on auxiliary target Y_t^- over Omega_t = {q: A_t^(1)=0}|
|      * L_rep   : Differentiable pairwise IoU penalty on predicted boxes           |
|      * L_smooth: L2 displacement squared of primary center (t vs t+1)             |
|      * L_off, L_sz: L1 regression on valid mode locations                         |
|      * Joint Loss: L = L_hcm + 1.0*L_off + 0.1*L_sz + 0.5*L_rmc + 0.3*L_rep + 0.2*L_sm|
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| 4. Ranked Multi-Peak Extraction (Inference)                                       |
|    - 3x3 Max-pooling local NMS                                                    |
|    - Confidence threshold tau=0.2, Top-K=3 peaks in descending order               |
|    - Standard COCO JSON output compatible with scripts/mode_disagreement.py        |
+-----------------------------------------------------------------------------------+
```

---

## 3. 세부 컴포넌트 및 수식 매핑

### (1) 공통 설정 및 상수 관리 ([`src/config.py`](../src/config.py))
모드 추출 및 학습/추론 하이퍼파라미터를 단일 위치에서 관리합니다:
- `VIEWPORT_SIZE_HW = (12, 20)`: 관찰자 뷰포트 크기 (타일 단위 $12 \times 20$)
- `MODE_EXTRACTION_SIGMA = 4.0`, `MODE_EXTRACTION_MIN_SEP = 12.0`, `MODE_EXTRACTION_REL_THRESHOLD = 0.35`, `MODE_EXTRACTION_MAX_MODES = 5`: `src/metrics/modes.py` 모드 추출 파라미터 일원화
- `NUM_OBSERVERS_U = 5`: 풀 관찰자 수
- `DIRECTOR_RENDER_SIGMA = 2.0`: 논문 Table B.8 가우시안 타깃 렌더링 $\sigma$
- `DIRECTOR_K = 3`, `DIRECTOR_TAU = 0.2`: 피크 추출 수 $K$ 및 신뢰도 임계치 $\tau$

---

### (2) 순위 모드 전처리 캐시 ([`src/dataset/mode_cache.py`](../src/dataset/mode_cache.py))
- **코드 재사용**: `src/metrics/modes.py`의 `extract_modes()` 함수를 그대로 호출하여 평가와 학습이 100% 동일한 정답 모드를 참조하도록 보장합니다.
- **자동 캐싱**: `ensure_mode_cache()` 유틸을 통해 학습 시작 시 replay별 `modes_{label_method}_sig4.0_sep12.0_th0.35_k5.pkl` 캐시를 멀티프로세싱으로 자동 생성합니다.
- **경량 직렬화**: 각 프레임에 대해 `centers: (M, 2) [row, col]`, `support: (M,)`, `n_observers`만 저장하여 I/O 오버헤드를 극소화했습니다.

---

### (3) 타깃 히트맵 렌더러 및 손실 함수 ([`src/losses/director_losses.py`](../src/losses/director_losses.py))

#### ① Support 가중 Gaussian 타깃 렌더러 (논문 식 12)
$$Y_t(q) = \max_k \omega_k \exp\left(-\frac{\|q - m_t^{(k)}\|^2}{2\sigma^2}\right), \quad \omega_k = \frac{n_t(m_t^{(k)})}{U} \quad (U=5)$$
- **$Y_t^{(1)}$**: 1순위 주 관심 모드 $m_t^{(1)}$만으로 렌더링된 히트맵.
- **$Y_t^-$**: 2순위 이하 부 관심 모드 $M_t^- = \{m_t^{(2)}, \dots, m_t^{(K)}\}$로 렌더링된 히트맵.
- **주 영역 마스크 $A_t^{(1)}$**: 정답 1순위 모드 중심 기준 $12 \times 20$ 뷰포트 영역.
- **보조 영역 마스크 $\Omega_t$ (식 16)**:
  $$\Omega_t = \{q : A_t^{(1)}(q) = 0\}$$

#### ② $L_{hcm}$ (Human Consensus Match loss, 식 13, 14)
- 주 영역을 Top-1 모드에 고정하기 위한 CornerNet 계열 modified focal loss:
  $$\ell_t(q) = \begin{cases} (1 - \hat{Y}_t(q))^{\alpha_f} \log \hat{Y}_t(q) & \text{if } Y_t^{(1)}(q) = 1 \\ (1 - Y_t^{(1)}(q))^{\beta_f} \hat{Y}_t(q)^{\alpha_f} \log(1 - \hat{Y}_t(q)) & \text{otherwise} \end{cases}$$
  $$L_{hcm} = -\frac{1}{N} \sum_q \ell_t(q) \quad (\alpha_f=2, \beta_f=4)$$

#### ③ $L_{rmc}$ (Ranked Mode Coverage loss, 식 15)
- 주 영역 밖($\Omega_t$)에서 2순위 이하 모드를 학습하도록 유도하는 masked MSE:
  $$L_{rmc} = \frac{1}{|\Omega_t|} \sum_{q \in \Omega_t} (\hat{Y}_t(q) - Y_t^-(q))^2$$
- **특징**: 주 영역($A_t^{(1)}$) 내부 타일을 마스킹하여 $L_{hcm}$과의 적대적 경쟁을 원천 차단하며, $M_t^-$가 빈 프레임에서는 정확히 $0.0$을 반환합니다.

#### ④ $L_{rep}$ (Spatial Repulsion loss, 식 17)
- 예측된 영역 박스 간의 중복을 억제하는 미분 가능한 pairwise IoU 합:
  $$L_{rep} = \sum_{k < l} \text{IoU}(V(\hat{p}_t^{(k)}), V(\hat{p}_t^{(l)}))$$

#### ⑤ $L_{smooth}$ (Trajectory Smoothness loss, 식 18)
- 연속 프레임 간 주 영역(Top-1) 중심의 프레임 간 L2 변위 제곱 페널티:
  $$L_{smooth} = \|\bar{p}_t^{(1)} - \bar{p}_{t+1}^{(1)}\|_2^2$$

#### ⑥ 결합 손실 (식 19)
$$L = L_{hcm} + \lambda_{off} L_{off} + \lambda_{sz} L_{size} + \lambda_{rmc} L_{rmc} + \lambda_{rep} L_{rep} + \lambda_{sm} L_{smooth}$$
- 기본 가중치: $\lambda_{off}=1.0, \lambda_{sz}=0.1, \lambda_{rmc}=0.5, \lambda_{rep}=0.3, \lambda_{sm}=0.2$.

---

### (4) 연속 프레임 페어링 샘플러 ([`src/dataset/starcraft_windows.py`](../src/dataset/starcraft_windows.py), [`src/dataset/loader.py`](../src/dataset/loader.py))
- 기존 `DataLoader`는 `shuffle=True`로 프레임들을 무작위로 섞기 때문에, 배치 내에서 연속 프레임을 추출할 수 없었습니다.
- `StarCraftWindowDataset`에 `pair_mode=True` 옵션을 추가하여 각 샘플 로드 시 동일 replay 내의 연속된 다음 윈도우 $(t, t+1)$ 텐서와 모드 정보를 함께 로드하도록 구현했습니다:
  - `target["next_image"]`: 다음 프레임 윈도우 텐서
  - `target["next_modes"]`: 다음 프레임 순위 모드 메타데이터
  - `target["next_valid"]`: 실제 후속 윈도우 존재 여부. replay의 마지막 윈도우는 후속이 없어 자기 자신과 페어링되므로(텐서 shape 유지 목적), 이 플래그가 `False`가 되어 $L_{smooth}$의 `valid_mask`에서 제외됩니다. 이것이 없으면 replay마다 변위 0인 가짜 쌍이 손실에 섞입니다.
- 이를 통해 배치가 무작위 셔플되어도 $L_{smooth}$를 1:1로 안전하고 정확하게 역전파할 수 있습니다.
- $L_{smooth}$ 활성화 조건은 `next_image`의 존재 여부로 판단합니다 (`next_modes`는 mode cache 적재 실패 시 없을 수 있어 게이트로 부적합).

---

### (5) 추론 시 가변 다중 Peak 추출 및 기존 평가 규약 호환 ([`src/inference.py`](../src/inference.py))
- $3 \times 3$ max-pooling으로 local NMS 수행.
- Score $\tau \ge 0.2$를 만족하는 피크 중 상위 $K=3$개를 내림차순 정렬하여 반환 (가변 개수).
- `save_predictions_as_coco()`를 통해 기존 COCO JSON 스키마를 유지하므로, 별도 변환 없이 [`scripts/mode_disagreement.py`](../scripts/mode_disagreement.py) 및 [`src/evaluate.py`](../src/evaluate.py)에서 즉시 평가 가능합니다.

---

## 4. 단위 테스트 검증 결과 ([`tests/test_director_losses.py`](../tests/test_director_losses.py))

다음 6가지 핵심 항목에 대한 단위 테스트가 작성되어 있습니다:
1. `test_render_targets_and_masking`: $Y_t^{(1)}, Y_t^-$ 분리 생성, 관찰자 지지도 스케일링($\omega_k$), 주 영역 내부 마스킹($\Omega_t = 0$) 검증.
2. `test_l_rmc_empty_when_no_auxiliary_modes`: 보조 모드가 없는 단일 모드 프레임에서 $L_{rmc} == 0.0$ 검증.
3. `test_l_rmc_masks_out_primary_region`: 주 영역 내부에 인위적 대형 오차를 주입해도 $\Omega_t$에 의해 $L_{rmc}$에 반영되지 않음을 검증.
4. `test_l_hcm_focal_loss_backward`: CornerNet modified focal loss 계산 및 역전파(gradient) 검증.
5. `test_l_rep_spatial_repulsion`: 비중복 시 IoU = 0.0, 완전 중복 시 IoU = 1.0 및 역전파 검증.
6. `test_l_smooth_trajectory`: 동일 좌표 시 0.0, 이동 시 L2 거리 제곱 계산 및 역전파 검증.

---

## 5. 실행 커맨드 가이드 ([`commands.sh`](../commands.sh))

### ① Baseline vs Director-CenterNet Full Model
```bash
# Baseline: Vanilla CenterNet (단일 영역)
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="architecture=centernet batch_size=16 max_epoch=30 window_size=4 num_workers=16 dataset=fold1 seed=123"
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="architecture=centernet batch_size=16 max_epoch=30 window_size=4 num_workers=16 dataset=fold1 seed=123"

# Proposed: Director-CenterNet (Full Model: L_hcm + L_rmc + L_rep + L_smooth)
NVIDIA_VISIBLE_DEVICES=0 make train ARGS="architecture=director_centernet batch_size=16 max_epoch=30 window_size=4 num_workers=16 dataset=fold1 seed=123"
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="architecture=director_centernet batch_size=16 max_epoch=30 window_size=4 num_workers=16 dataset=fold1 seed=123"
```

### ⓪ Ablation 설계 근거 (논문 실험 설정에 그대로 옮길 것)

설계한 손실은 4개($L_{hcm}$, $L_{rmc}$, $L_{rep}$, $L_{smooth}$)지만 **ablation 축은 3개**다. $L_{off}$ / $L_{size}$는 표준 CenterNet 회귀 항이라 기여도 측정 대상이 아니라 전제다.

**$L_{hcm}$은 기저이지 축이 아니다.** $\lambda_{hcm}=0$이면 히트맵에 양성 감독이 한 곳도 남지 않는다. $L_{rmc}$의 도메인은 보조 support로 한정되어 있어 주 영역 셀도 배경도 감독하지 않으므로, 학습 자체가 성립하지 않는 퇴화 설정이다. 따라서 격자는 $2^4=16$이 아니라 $2^3=8$이다.

8개 중 **2개는 돌리지 않으며, 그 이유가 논문에 들어가야 한다.**

- **$L_{hcm}+L_{rep}$ (미실행): 설계상 무의미하다.** $L_{rep}$는 *이미 존재하는* 영역 쌍의 IoU를 벌하는 항이라 영역을 생성하지 못한다. $L_{rmc}$가 없으면 프레임당 예측 영역이 사실상 1개이므로 밀어낼 대상이 존재하지 않고, 이 조합은 $L_{hcm}$ 단독과 구분되지 않는다. 즉 **$L_{rep}$의 효과는 $L_{rmc}$에 조건부로만 정의된다.** 이는 누락이 아니라 두 항의 의존 구조이며, Table에서 해당 칸을 비우는 근거로 명시할 것.
- **$L_{hcm}+L_{smooth}$ (미실행): 선택적.** 다중 영역 기계 없이 시간적 안정화만의 효과를 보는 설정으로, 기존 단일 영역 연구에 가장 가깝다. $L_{smooth}$의 작동 여부는 VD 지표로 Row 3 vs Row 4에서 직접 읽히므로 필수는 아니다.

**누적 비교군 Row 3과 leave-one-out의 `w/o L_smooth`는 같은 실행이다** ($L_{hcm}+L_{rmc}+L_{rep}$). 아래 ②-3과 ③-3은 `id_string`만 다른 중복이므로 한 번만 돌리고 두 표에 같은 수치를 싣는다.

임계값 관련: `score_threshold` 스윕은 재학습이 필요 없다. [`src/inference.py`](../src/inference.py)가 임계값별로 `model_NNN_th<x>` 폴더를 분리하므로 학습된 체크포인트에 추론만 다시 돌리면 된다. 하한은 모델 내부 $\tau$(`DIRECTOR_TAU`)이며, 그 아래 peak은 필터에 도달하기 전에 이미 버려진다.

### ② Loss Ablation Studies (논문 Table 7 누적 비교군)
```bash
# 1) L_hcm only (Row 1: w/o L_rmc, L_rep, L_smooth)
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="architecture=director_centernet architecture.loss_weights.lambda_rmc=0.0 architecture.loss_weights.lambda_rep=0.0 architecture.loss_weights.lambda_sm=0.0 batch_size=16 max_epoch=30 window_size=4 num_workers=16 dataset=fold1 seed=123 id_string=director_centernet_abl_hcm_only_fold1_s123"

# 2) L_hcm + L_rmc (Row 2: w/o L_rep, L_smooth)
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="architecture=director_centernet architecture.loss_weights.lambda_rep=0.0 architecture.loss_weights.lambda_sm=0.0 batch_size=16 max_epoch=30 window_size=4 num_workers=16 dataset=fold1 seed=123 id_string=director_centernet_abl_hcm_rmc_fold1_s123"

# 3) L_hcm + L_rmc + L_rep (Row 3: w/o L_smooth)
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="architecture=director_centernet architecture.loss_weights.lambda_sm=0.0 batch_size=16 max_epoch=30 window_size=4 num_workers=16 dataset=fold1 seed=123 id_string=director_centernet_abl_hcm_rmc_rep_fold1_s123"

# 4) Full Director-CenterNet (Row 4: L_hcm + L_rmc + L_rep + L_smooth)
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="architecture=director_centernet batch_size=16 max_epoch=30 window_size=4 num_workers=16 dataset=fold1 seed=123 id_string=director_centernet_full_fold1_s123"
```

### ③ Leave-One-Out Ablations
```bash
# w/o L_rmc (Ranked Mode Coverage 제외)
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="architecture=director_centernet architecture.loss_weights.lambda_rmc=0.0 batch_size=16 max_epoch=30 window_size=4 num_workers=16 dataset=fold1 seed=123 id_string=director_centernet_abl_no_rmc_fold1_s123"

# w/o L_rep (Spatial Repulsion 제외)
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="architecture=director_centernet architecture.loss_weights.lambda_rep=0.0 batch_size=16 max_epoch=30 window_size=4 num_workers=16 dataset=fold1 seed=123 id_string=director_centernet_abl_no_rep_fold1_s123"

# w/o L_smooth (Trajectory Smoothness 제외)
NVIDIA_VISIBLE_DEVICES=0 make run ARGS="architecture=director_centernet architecture.loss_weights.lambda_sm=0.0 batch_size=16 max_epoch=30 window_size=4 num_workers=16 dataset=fold1 seed=123 id_string=director_centernet_abl_no_smooth_fold1_s123"
```

### ④ 모드 진단 스크립트 실행 ([`scripts/mode_disagreement.py`](../scripts/mode_disagreement.py))
```bash
python scripts/mode_disagreement.py \
    --replays 275 1725 3613 4520 4664 \
    --label-method all_correct \
    --model centernet=centernet_vanilla_win1_fold1_s123_20260818_071116 \
    --model director=director_centernet_full_fold1_s123 \
    --epoch 30 \
    --outdir results/mode_disagreement/fold1
```

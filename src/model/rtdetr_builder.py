# src/model/rtdetr_builder.py
import os
import argparse
from .CustomRTDETR import CustomRTDETR
from .KBRS_RTDETR import KBRS_RTDETR

def get_model_instance_rtdetr(num_classes: int,
                              version: str = "v2", 
                              model_size: str = "l", 
                              in_channels: int = 3,
                              use_kbrs: bool = False,         # 추가됨
                              kbrs_params: dict = None,       # 추가됨
                              loss_weights: dict = None):     # 추가됨
    
    # ---------------------------------------------------------
    # 1. 환경변수에서 캐시 디렉토리를 읽어옵니다.
    # ---------------------------------------------------------
    base_cache_dir = os.environ.get("TORCH_CACHE_DIR", "/workspace/.torch_cache")
    weight_dir = os.path.join(base_cache_dir, "ultralytics")
    os.makedirs(weight_dir, exist_ok=True)

    if version.lower() == "v2":
        weights = os.path.join(weight_dir, f"rtdetrv2-{model_size}.pt")
    else:
        weights = os.path.join(weight_dir, f"rtdetr-{model_size}.pt")

    print(f"Building RT-DETR (Ultralytics) - Version: {version}, Size: {model_size}, Classes: {num_classes}")
    print(f"Target Weights Path: {weights}")
    
    # 2. 파일이 없을 경우 직접 다운로드 (권한 에러 우회)
    if not os.path.exists(weights):
        print(f"Weights not found at {weights}. Downloading...")
        import urllib.request
        # v8.3.0 또는 작동하는 릴리즈 URL 사용
        url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{os.path.basename(weights)}"
        try:
            urllib.request.urlretrieve(url, weights)
            print("Download complete.")
        except Exception as e:
            raise RuntimeError(f"Failed to download {os.path.basename(weights)} from {url}. Error: {e}")

    # 3. KBRS 사용 여부에 따른 모델 분기
    if use_kbrs:
        print("Mode         : KBRS integrated RT-DETR (Forward Hooking)")
        model = KBRS_RTDETR(
            weights=weights, 
            num_classes=num_classes,
            in_channels=in_channels,
            kbrs_params=kbrs_params,
            loss_weights=loss_weights
        )
    else:
        print("Mode         : Vanilla RT-DETR")
        model = CustomRTDETR(
            weights=weights, 
            num_classes=num_classes,
            in_channels=in_channels
        )    
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--version", choices=["v1", "v2"], default="v2", help="RT-DETR 버전 선택")
    parser.add_argument("--size", choices=["s", "m", "l", "x"], default="l", help="모델 크기")
    parser.add_argument("--in-channels", type=int, default=3)
    args = parser.parse_args()

    model = get_model_instance_rtdetr(
        num_classes=args.num_classes,
        version=args.version,
        model_size=args.size,
        in_channels=args.in_channels
    )

    print("\n===== RT-DETR Model Structure =====")
    print(model)
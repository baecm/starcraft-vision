# src/model/factory.py
from .maskrcnn_builder import get_model_instance_segmentation
from .rtdetr_builder import get_model_instance_rtdetr
from .KBRS_RTDETR import KBRS_RTDETR  # (이전 답변에서 만든 Hook 기반 통합 모델)

def build_model(args):
    """
    args에 따라 4가지(세부 6가지) 실험 조합 중 하나를 생성하여 반환합니다.
    """
    model_name = args.model_name.lower() # "maskrcnn" or "rtdetr"
    use_kbrs = args.use_kbrs

    print(f"========== Building Model ==========")
    print(f"Architecture : {model_name.upper()}")
    print(f"Use KBRS     : {use_kbrs}")
    
    if model_name == "maskrcnn":
        # 1 & 2: Mask R-CNN (Vanilla or KBRS)
        model = get_model_instance_segmentation(
            num_classes=args.num_classes,
            window_size=args.window_size,
            in_channels=args.in_channels,
            use_kbrs=use_kbrs,
            kbrs_params=args.kbrs_params if use_kbrs else None,
            # 기타 args 생략
        )
        return model

    elif model_name == "rtdetr":
        print(f"RT-DETR Ver  : {args.rtdetr_version}")
        
        # -----------------------------------------------------------------
        # [수정] 가중치 다운로드 및 절대 경로 확보 로직을 factory에서 처리합니다.
        # -----------------------------------------------------------------
        import os
        import urllib.request

        base_cache_dir = os.environ.get("TORCH_CACHE_DIR", "/workspace/.torch_cache")
        weight_dir = os.path.join(base_cache_dir, "ultralytics")
        os.makedirs(weight_dir, exist_ok=True)

        weight_filename = f"rtdetrv2-{args.rtdetr_size}.pt" if args.rtdetr_version == "v2" else f"rtdetr-{args.rtdetr_size}.pt"
        weights_path = os.path.join(weight_dir, weight_filename)

        if not os.path.exists(weights_path):
            print(f"Weights not found at {weights_path}. Downloading...")
            # 에러 로그에 찍힌 v8.3.0 URL 반영
            url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{weight_filename}"
            try:
                urllib.request.urlretrieve(url, weights_path)
                print("Download complete.")
            except Exception as e:
                raise RuntimeError(f"Failed to download {weight_filename} from {url}. Error: {e}")
        # -----------------------------------------------------------------

        if use_kbrs:
            # 4: RT-DETR (v1/v2) + KBRS
            print("Mode         : KBRS integrated RT-DETR (Forward Hooking)")
            model = KBRS_RTDETR(
                weights=weights_path,  # <--- 파일명이 아닌 "절대 경로" 전달
                num_classes=args.num_classes,
                in_channels=args.in_channels,
                kbrs_params=args.kbrs_params,
                loss_weights=args.loss_weights
            )
        else:
            # 3: RT-DETR (v1/v2) Baseline
            print("Mode         : Vanilla RT-DETR")
            
            # 주의: 만약 rtdetr_builder.py의 get_model_instance_rtdetr가 
            # weights 인자로 절대 경로를 제대로 받도록 되어 있는지 확인해주세요.
            model = get_model_instance_rtdetr(
                num_classes=args.num_classes,
                version=args.rtdetr_version,
                model_size=args.rtdetr_size,
                in_channels=args.in_channels
                # (rtdetr_builder.py 내부에서도 동일한 캐시 경로를 쓰도록 
                # 앞서 수정해두었으므로 정상 동작할 것입니다.)
            )
        return model

    else:
        raise ValueError(f"Unknown model_name: {model_name}. Use 'maskrcnn' or 'rtdetr'.")
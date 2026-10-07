"""The end-to-end benchmark runner that lived in src/evaluate.py until 2026-10.
It ran the legacy ProbabilisticVideoDETR comparison (`make estimate ARGS="--benchmark ..."`).
Kept for reference only; it is not importable from src/ any more."""

# =====================================================================
# 5. E2E Benchmark Runner (Integrated from run_benchmark.py)
# =====================================================================

def run_e2e_benchmark(args: argparse.Namespace):
    """
    Executes End-to-End model benchmark inference & Proposed model comparison.
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys_paths = [root_dir, os.path.join(root_dir, "src")]
    for p in sys_paths:
        if p not in sys.path:
            sys.path.insert(0, p)

    from dataset.custom_penn_fudan import CustomPennFudanDataset
    from models import build_model, ProbabilisticVideoDETR

    window_size = getattr(args, "window_size", None)
    if window_size is None:
        model_lower = (args.model_name or "").lower()
        if "win1" in model_lower:
            window_size = 1
        elif "win4" in model_lower:
            window_size = 4
        else:
            window_size = 4

    out_json = getattr(args, "output_json", None)
    if not out_json:
        bench_dir = "/workspace/results/benchmark"
        os.makedirs(bench_dir, exist_ok=True)
        out_json = os.path.join(bench_dir, f"{args.model_name}_e{args.epoch}.json")

    print("=" * 85)
    print(f"🚀 Running Evaluation Benchmark (Task Mode: {args.task.upper()})")
    print(f"[*] Target Model    : {args.model_name} (Epoch {args.epoch})")
    print(f"[*] Window Size     : {window_size}")
    print(f"[*] Output Path     : {out_json}")
    print(f"[*] Target Replays  : {args.replays}")
    print("=" * 85)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    print("[*] Loading dataset windows...")
    try:
        ds = CustomPennFudanDataset(
            input_root=args.input_root,
            label_root=args.label_root,
            label_method=args.label_method,
            training_ids=args.replays,
            window_size=window_size,
            include_components=getattr(args, "include_components", None),
            interval=1,
            training=False,
            verbose=True,
        )
    except Exception as e:
        local_input = os.path.join(root_dir, "data/input/dst")
        local_label = os.path.join(root_dir, "data/label/dst")
        ds = CustomPennFudanDataset(
            input_root=local_input,
            label_root=local_label,
            label_method=args.label_method,
            training_ids=args.replays,
            window_size=window_size,
            include_components=getattr(args, "include_components", None),
            interval=1,
            training=False,
            verbose=True,
        )

    # Replay metrics computation
    replay_results = []
    base_avg: Dict[str, float] = {}

    for replay_id in args.replays:
        replay_id = str(replay_id)
        coco_gt = load_coco_gt(args.label_root, replay_id, args.label_method)
        preds_by_img = load_coco_preds(args.pred_root, args.model_name, args.epoch, replay_id, args.label_method)

        ic_row = compute_ic_for_replay(
            replay_id=replay_id,
            mode="model",
            coco_gt=coco_gt,
            args=args,
            preds_all=preds_by_img,
            model_tag=f"{args.model_name}_e{args.epoch}",
        )

        multi_row = compute_multi_region_for_replay(
            replay_id=replay_id,
            coco_gt=coco_gt,
            preds_by_img=preds_by_img,
            multi_topk=getattr(args, "multi_topk", 3),
        )

        combined = dict(ic_row)
        combined.update(multi_row)
        combined["replay"] = replay_id
        replay_results.append(combined)

    df_res = pd.DataFrame(replay_results)
    for col in df_res.select_dtypes(include=[np.number]).columns:
        base_avg[col] = float(df_res[col].dropna().mean()) if not df_res[col].dropna().empty else float("nan")

    # Display comparison if requested
    print_section_header(f"BENCHMARK RESULTS SUMMARY: {args.model_name} (Epoch {args.epoch})")
    cols = list(df_res.columns)
    rb = ReportBlock(
        title=f"Benchmark Summary ({args.model_name})",
        columns=cols,
        aligns=["left"] + ["right"] * (len(cols) - 1),
    )
    for _, r in df_res.iterrows():
        rb.add_row(*[r.get(c) for c in cols])
    rb.print()

    # Save outputs
    out_dir = os.path.dirname(out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"replays": replay_results, "summary_averages": base_avg}, f, indent=2)
    print(f"[*] Saved benchmark summary JSON: {out_json}")


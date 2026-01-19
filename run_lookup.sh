# ground truth labels
NVIDIA_VISIBLE_DEVICES=0 make lookup ARGS=" \
  --replays 275 1725 3613 4520 4664 1559 1628 2351 6219 11251 36 212 438 522 1660 \
  --label-source gt \
  --label-method all_correct \
  --skip-missing-npz \
  --force-row-on-error \
  --num-workers 16 \
  --log-level log \
  "

NVIDIA_VISIBLE_DEVICES=0 make lookup ARGS=" \
  --replays 6219 11251 \
  --label-source gt \
  --label-method all_correct \
  --skip-missing-npz \
  --force-row-on-error \
  --num-workers 16 \
  --log-level log \
  "

# predicted labels
# fold1 models
NVIDIA_VISIBLE_DEVICES=0 make lookup ARGS=" \
  --replays 275 1725 3613 4520 4664 \
  --label-source pred \
  --id-string \
    all_correct_win4_vanilla_fold1_s123_20251201_072032 \
    all_correct_win4_vanilla_fold1_s456_20251202_003952 \
    all_correct_win4_vanilla_fold1_s789_20251202_163217 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_base_score_20251219_080334 \
    all_correct_win4_kbrs_fold1_s456_kbrs025_base_score_20260101_175816 \
    all_correct_win4_kbrs_fold1_s789_kbrs025_base_score_20251230_004200 \
    all_correct_win4_kbrs_fold1_s123_kbrs050_base_score_20251204_075521 \
    all_correct_win4_kbrs_fold1_s123_kbrs075_base_score_20251215_013131 \
    all_correct_win4_kbrs_fold1_s123_kbrs100_base_score_20251218_144314 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_density_only_010_20260106_020536 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_density_only_020_20260107_002100 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_density_only_030_20260105_055837 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_density_only_040_20260107_213711 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_density_only_050_20260108_192700 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_density_only_060_20260112_054512 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_density_only_070_20260113_021018 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_density_only_080_20260112_054522 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_density_only_090_20260113_021512 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_density_000_20260104_094953 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_density_010_20251221_003040 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_density_020_20251204_074954 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_density_040_20251219_025812 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_density_050_20251225_164224 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_density_060_20251226_040303 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_density_070_20251226_232953 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_density_080_20251227_190348 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_density_090_20251228_144100 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_centeredness_only_010_20260106_025719 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_centeredness_only_020_20260106_161035 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_centeredness_only_030_20260103_083718 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_centeredness_only_040_20260108_065035 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_centeredness_only_050_20260107_084411 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_centeredness_only_060_20260112_054141 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_centeredness_only_070_20260115_010156 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_centeredness_only_080_20260115_133852 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_centeredness_only_090_20260115_133852 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_centeredness_000_20260102_074149 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_centeredness_010_20251216_063600 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_centeredness_020_20251204_074940 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_centeredness_040_20251224_064547 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_centeredness_050_20251219_101225 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_centeredness_060_20260112_054140 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_centeredness_070_20260115_011113 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_centeredness_080_20260114_042003 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_centeredness_090_20260117_032809 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_mixture_only_050_20260117_032841 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_mixture_only_100_20260110_003945 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_mixture_only_150_20260109_034032 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_mixture_only_200_20260108_063029 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_mixture_only_250_20260107_082026 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_mixture_only_300_20260105_030111 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_mixture_only_350_20260107_082433 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_mixture_only_400_20260108_063029 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_mixture_only_450_20260109_034029 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_mixture_only_500_20260110_003827 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_mixture_000_20260104_071547 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_mixture_050_20260106_034845 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_mixture_100_20260107_114216 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_mixture_150_20260108_190414 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_mixture_200_20251223_094128 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_mixture_250_20251204_074947 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_mixture_350_20251222_073231 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_mixture_400_20251224_140321 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_mixture_450_20260109_220242 \
    all_correct_win4_kbrs_fold1_s123_kbrs025_mixture_500_20260117_032417 \
  --epoch 30 \
  --label-method all_correct \
  --skip-missing-npz \
  --force-row-on-error \
  --num-workers 64 \
  --log-level log \
  "

# fold2 models
NVIDIA_VISIBLE_DEVICES=0 make lookup ARGS=" \
  --replays 1559 1628 2351 6219 11251 \
  --label-source pred \
  --id-string \
    all_correct_win4_vanilla_fold2_s123_20251203_081925 \
    all_correct_win4_vanilla_fold2_s456_20251204_002252 \
    all_correct_win4_vanilla_fold2_s789_20251204_163313 \
    all_correct_win4_kbrs_fold2_s123_kbrs025_base_score_20251222_080550 \
    all_correct_win4_kbrs_fold2_s456_kbrs025_base_score_20251231_031613 \
    all_correct_win4_kbrs_fold2_s789_kbrs025_base_score_20251231_215319 \
  --epoch 30 \
  --label-method all_correct \
  --skip-missing-npz \
  --force-row-on-error \
  --num-workers 64 \
  --log-level log \
  "

# fold3 models
NVIDIA_VISIBLE_DEVICES=0 make lookup ARGS=" \
  --replays 36 212 438 522 1660 \
  --label-source pred \
  --id-string \
    all_correct_win4_vanilla_fold3_s123_20251205_081306 \
    all_correct_win4_vanilla_fold3_s456_20251205_235535 \
    all_correct_win4_vanilla_fold3_s789_20251206_161212 \
    all_correct_win4_kbrs_fold3_s123_kbrs025_base_score_20251230_083127 \
    all_correct_win4_kbrs_fold3_s456_kbrs025_base_score_20251231_151012 \
    all_correct_win4_kbrs_fold3_s789_kbrs025_base_score_20260102_163643 \
  --epoch 30 \
  --label-method all_correct \
  --skip-missing-npz \
  --force-row-on-error \
  --num-workers 64 \
  --log-level log \
  "

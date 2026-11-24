from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    data_file: str
    data_key: str
    label_file: str
    label_key: str
    train_rate: float


DATASETS = {
    "Indian": DatasetSpec(
        "Indian_pines_corrected.mat",
        "indian_pines_corrected",
        "Indian_pines_gt.mat",
        "indian_pines_gt",
        0.05,
    ),
    "PaviaU": DatasetSpec("PaviaU.mat", "paviaU", "PaviaU_gt.mat", "paviaU_gt", 0.03),
    "Houston": DatasetSpec("Houston.mat", "houston", "Houston_gt.mat", "houston_gt", 0.03),
    "WHU_LK": DatasetSpec(
        "WHU_Hi_LongKou.mat",
        "WHU_Hi_LongKou",
        "WHU_Hi_LongKou_gt.mat",
        "WHU_Hi_LongKou_gt",
        0.005,
    ),
}


PAPER_BANDS = {
    "Indian": [7, 85, 109, 26, 80, 181, 153, 36, 32, 28, 5, 99, 9, 78, 161, 82, 148, 11, 48, 159],
    "PaviaU": [16, 99, 14, 101, 91, 93, 12, 19, 89, 65, 23, 87, 21, 96, 84, 62, 81, 7, 25, 10],
    "Houston": [35, 11, 122, 8, 37, 120, 32, 6, 44, 13, 104, 110, 18, 84, 22, 82, 131, 80, 68, 53],
    "WHU_LK": [5, 151, 9, 146, 144, 1, 269, 126, 3, 13, 258, 7, 157, 261, 264, 123, 266, 128, 167, 149],
}

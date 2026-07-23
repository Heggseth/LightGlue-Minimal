import os
import torch
import cv2
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

# rbd 用来去掉数据里的 batch 维度，把形状里 batchsize 那一个维度去掉，便于后续处理


def visualize_matches(image0_path, image1_path, points0, points1, output_path):
    """
    可视化两张图片之间的匹配点，并保存结果图像
    points0: 第一张图中的匹配点坐标，形状 [K, 2]
    points1: 第二张图中的匹配点坐标，形状 [K, 2]
    """

    # 1. 把 torch.Tensor 转成 numpy，方便 OpenCV 使用
    points0 = points0.detach().cpu().numpy()
    points1 = points1.detach().cpu().numpy()

    # 2. 用 OpenCV 读取原始图片
    img0 = cv2.imread(image0_path)
    img1 = cv2.imread(image1_path)

    if img0 is None:
        raise FileNotFoundError(f"Cannot read image0: {image0_path}")
    if img1 is None:
        raise FileNotFoundError(f"Cannot read image1: {image1_path}")

    # 3. 把点坐标转换成 OpenCV 的 KeyPoint 格式
    kpts0 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in points0]
    kpts1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in points1]

    # 4. 因为 points0[i] 和 points1[i] 已经是一对匹配点
    # 所以这里手动构造 DMatch
    cv_matches = [
        cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0)
        for i in range(len(kpts0))
    ]

    # 5. 画出匹配线
    match_img = cv2.drawMatches(
        img0,
        kpts0,
        img1,
        kpts1,
        cv_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # 6. 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, match_img)

    print(f"Saved visualization to: {output_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # extractor 是基于 SuperPoint 的特征提取器，matcher 是匹配器
    extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    image0_path = "images/image0.jpg"
    image1_path = "images/image1.jpg"

    image0 = load_image(image0_path).to(device)
    image1 = load_image(image1_path).to(device)

    print("image0 shape:", image0.shape)
    print("image1 shape:", image1.shape)

    # feats 是图的特征，它们都是字典类型
    # keypoints：关键点坐标
    # keypoint_scores：每个关键点的置信度 / 分数
    # descriptors：每个关键点对应的特征描述子
    # image_size：原始图像尺寸信息
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)

    print("feats0 keys:", feats0.keys())
    print("feats1 keys:", feats1.keys())

    # matches01 代表第一张图和第二张图之间的匹配关系
    matches01 = matcher({"image0": feats0, "image1": feats1})

    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    # matches：匹配索引
    # points0：第一张图里的匹配点坐标
    # points1：第二张图里的匹配点坐标
    matches = matches01["matches"]
    points0 = feats0["keypoints"][matches[..., 0]]
    points1 = feats1["keypoints"][matches[..., 1]]

    print("matches shape:", matches.shape)
    print("points0 shape:", points0.shape)
    print("points1 shape:", points1.shape)
    print("Number of matches:", len(matches))

    # 可视化匹配结果
    visualize_matches(
        image0_path=image0_path,
        image1_path=image1_path,
        points0=points0,
        points1=points1,
        output_path="outputs/matches_output.jpg",
    )


if __name__ == "__main__":
    main()

import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
#rbd用来去掉数据里的 batch 维度，把形状里batchsize那一个维度去掉，便于后续处理

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    #extractor是基于SuperPoint的特征提取器  matcher是匹配器
    extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    image0 = load_image("images/image0.jpg").to(device)
    image1 = load_image("images/image1.jpg").to(device)

    print("image0 shape:", image0.shape)
    print("image1 shape:", image1.shape)
    #feats是图的特征，它们都是字典类型，keypoints：关键点坐标 keypoint_scores：每个关键点的置信度/分数 descriptors：每个关键点对应的特征描述子 image_size：原始图像尺寸信息

    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)

    print("feats0 keys:", feats0.keys())
    print("feats1 keys:", feats1.keys())
    #matches01代表第一张图和第二张图之间的匹配关系
    matches01 = matcher({"image0": feats0, "image1": feats1})

    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
#matches：匹配索引
#points0：第一张图里的匹配点坐标
#points1：第二张图里的匹配点坐标

    matches = matches01["matches"]
    points0 = feats0["keypoints"][matches[..., 0]]
    points1 = feats1["keypoints"][matches[..., 1]]

    print("matches shape:", matches.shape)
    print("points0 shape:", points0.shape)
    print("points1 shape:", points1.shape)
    print("Number of matches:", len(matches))


if __name__ == "__main__":
    main()
#创建 SuperPoint 特征提取器
#↓
#创建 LightGlue 匹配器
#↓
#读取 image0 和 image1
#↓
#对 image0 提取关键点和描述子
#↓
#对 image1 提取关键点和描述子
#↓
#LightGlue 匹配两张图的特征
#↓
#得到 matches
#↓
#根据 matches 找到 points0 和 points1
#  ↓
#打印匹配数量
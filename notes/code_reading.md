 'from lightglue.utils import load_image, rbd'
what is rbd used for?  rbd is used to remove the bash dimension from data, getting rid of the batchsize dimension in the shape,making it easier for subsequent processing


' matches01 = matcher({"image0": feats0, "image1": feats1})'
matches01 represents the matching relationship between the first and second images
like matches01 = [15,2]. It means the 15st point in image0 matches the 2nd point in image1.

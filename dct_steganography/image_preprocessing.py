import numpy as np

luminance_quant_table = np.asarray([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 36, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
],
    dtype=np.float32)


#deli sliku na blokove velicine 8x8
def divide_img_to_8x8(image):
    blocks = []
    for i in np.vsplit(image, int(image.shape[0] / 8)):
        for j in np.hsplit(i, int(image.shape[1] / 8)):
            blocks.append(j)
    return blocks


#spaja 8x8 blokove u sliku
def connect_8x8_blocks(img_width, block_segments):
    image_rows = []
    temp = []
    for i in range(len(block_segments)):
        if i > 0 and not(i % int(img_width / 8)):
            image_rows.append(temp)
            temp = [block_segments[i]]
        else:
            temp.append(block_segments[i])
    image_rows.append(temp)

    return np.block(image_rows)


#kontejnerska klasa za cuvanje slike u YCbCr prostoru boja pri cemu je svaka komponenta slike
#podeljena na blokove
class YCrCb(object):
    def __init__(self, cover_image):
        self.height, self.width = cover_image.shape[:2]
        self.channels = [
                         divide_img_to_8x8(cover_image[:, :, 0]),
                         divide_img_to_8x8(cover_image[:, :, 1]),
                         divide_img_to_8x8(cover_image[:, :, 2]),
                        ]

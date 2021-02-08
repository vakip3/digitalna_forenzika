import cv2
from numpy import *
import image_preprocessing as preprocess
set_printoptions(formatter={'float': '{: 0.3f}'.format})

M = 0
N = 0


def u_coeff(idx):
    if idx > 0:
        return math.sqrt(2 / M)
    return 1 / math.sqrt(M)


def v_coeff(idx):
    if idx > 0:
        return math.sqrt(2 / N)
    return 1 / math.sqrt(N)


def sum_of_sum(matrix, M, N, row, col):
    return sum([sum([matrix[i, j] * math.cos(((2 * i + 1) * row * math.pi) / (2 * M)) * math.cos(
        ((2 * j + 1) * col * math.pi) / (2 * N)) for j in range(N)]) for i in range(M)])


def isum_of_sum(matrix, M, N, r, s):
    return sum([sum([u_coeff(i) * v_coeff(j) * matrix[i, j] * math.cos(((2 * r + 1) * i * math.pi) / (2 * M)) * math.cos(
        ((2 * s + 1) * j * math.pi) / (2 * N)) for j in range(N)]) for i in range(M)])


def dct2(img):
    global M
    global N
    image = img
    original_matrix = image
    M = 8
    N = 8
    dct_matrix = zeros([M, N])
    for row in range(M):
        for col in range(N):
            dct_matrix[row, col] = u_coeff(row) * v_coeff(col) * sum_of_sum(original_matrix, M, N, row, col)
    return around(dct_matrix)


def idct2(dct_mat):
    idct_matrix = zeros([M, N])
    for row in range(M):
        for col in range(N):
            idct_matrix[row, col] = isum_of_sum(dct_mat, M, N, row, col)
    return idct_matrix


# sampl = around(random.uniform(low=0, high=255, size=(8,8)))


raw_cover_image = cv2.imread("lenna.jpg", flags=cv2.IMREAD_COLOR)
height, width = raw_cover_image.shape[:2]
# ako dimenzije slike nisu deljive sa 8, povecavamo ih tako da budu
while height % 8:
    height += 1
while width % 8:
    width += 1
valid_dim = (width, height)
padded_image = cv2.resize(raw_cover_image, valid_dim)
cover_image_f32 = float32(padded_image)
# konvertujemo sliku u YCbCr format
cover_image_YCC = preprocess.YCrCb(cv2.cvtColor(cover_image_f32, cv2.COLOR_BGR2YCrCb))
obrada = cover_image_YCC.channels[0]
item = obrada[0]
#
# matrix_after_dct = dct2(item)
#
# print(idct2(matrix_after_dct))
#
# print("\n")
# print("\n")
# print(cv2.idct(matrix_after_dct))

def toBits(message):
    bits = []

    for char in message:
        binval = bin(ord(char))[2:].rjust(8, '0')

        # for bit in binval:
        bits.append(binval)

    numBits = bin(len(bits))[2:].rjust(8, '0')
    return bits, numBits


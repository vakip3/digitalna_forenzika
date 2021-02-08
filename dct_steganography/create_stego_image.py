import cv2
import bitstring
import numpy as np
import zigzag_matrix_scan as zz
import image_preprocessing as preprocess
import dct as dct
import multiprocessing
from joblib import Parallel, delayed


num_cores = multiprocessing.cpu_count()

stego_file = "./stego_image.png"
cover_file = "./lenna.jpg"
message = "i want to tell you something i am here and i know things that would hurt you"


def hide_message(message_bits, dct_blocks):
    data_complete = False
    message_bits.pos = 0
    encoded_data_len = bitstring.pack('uint:32', len(message_bits))
    encoded_blocks = []
    for dct_block in dct_blocks:
        # obilaze se svi AC koeficijenti, DC preskacemo
        for i in range(1, len(dct_block)):
            acc_coeff = np.int32(dct_block[i])
            if acc_coeff > 1:
                acc_coeff = np.uint8(dct_block[i])
                if message_bits.pos == (len(message_bits) - 1):
                    data_complete = True
                    break
                packed_coeff = bitstring.pack('uint:8', acc_coeff)
                if encoded_data_len.pos <= len(encoded_data_len) - 1:
                    packed_coeff[-1] = encoded_data_len.read(1)
                else:
                    packed_coeff[-1] = message_bits.read(1)
                dct_block[i] = np.float32(packed_coeff.read('uint:8'))
        encoded_blocks.append(dct_block)

    if not data_complete:
        raise ValueError("Message is too long!")

    return encoded_blocks


def stego(cover_img, message):
    num_channels = 3
    cover_image_path = cover_img
    secret_message = message

    raw_cover_image = cv2.imread(cover_image_path, flags=cv2.IMREAD_COLOR)
    height, width = raw_cover_image.shape[:2]
    # ako dimenzije slike nisu deljive sa 8, povecavamo ih tako da budu
    while height % 8:
        height += 1
    while width % 8:
        width += 1
    valid_dim = (width, height)
    padded_image = cv2.resize(raw_cover_image, valid_dim)
    cover_image_f32 = np.float32(padded_image)
    # konvertujemo sliku u YCbCr format
    cover_image_YCC = preprocess.YCrCb(cv2.cvtColor(cover_image_f32, cv2.COLOR_BGR2YCrCb))
    stego_image = np.empty_like(cover_image_f32)

    for channel in range(num_channels):
        # primenjujemo dct nad blokovima

        dct_blocks = Parallel(n_jobs=num_cores)(delayed(dct.dct2)(block) for block in cover_image_YCC.channels[channel])
        # kvantizacija blokova
        dct_quants = [np.around(np.divide(item, preprocess.luminance_quant_table)) for item in dct_blocks]

        # koeficijenti u bloku se obilaze cik-cak i sortiraju po energiji
        sorted_coefficients = [zz.zigzag(block) for block in dct_quants]

        # podatke sakrivamo u luminance sloju jer su tu promene najmanje primetne
        if channel == 0:
            secret_data = ""
            for char in secret_message.encode('ascii'):
                secret_data += bitstring.pack('uint:8', char)
            embedded_dct_blocks = hide_message(secret_data, sorted_coefficients)
            desorted_coefficients = [zz.inverse_zigzag(block, max_width=8, max_height=8) for block in
                                     embedded_dct_blocks]
        else:
            # koeficijenti se vracaju u originalni raspored
            desorted_coefficients = [zz.inverse_zigzag(block, max_width=8, max_height=8) for block in
                                     sorted_coefficients]

        # dekvantizacija blokova
        dct_dequants = [np.multiply(data, preprocess.luminance_quant_table) for data in desorted_coefficients]

        # inverzni dct

        idct_blocks = Parallel(n_jobs=num_cores)(delayed(dct.idct2)(block) for block in dct_dequants)

        # spajanje blokova u sliku
        stego_image[:, :, channel] = np.asarray(preprocess.connect_8x8_blocks(cover_image_YCC.width, idct_blocks))

    return stego_image


# stego_image = stego(cover_file, message)
# # slika se konvertuje nazad u RGB format
# stego_image_BGR = cv2.cvtColor(stego_image, cv2.COLOR_YCR_CB2BGR)
#
# # pikseli na vrednost od 0 do 255
# final_stego_image = np.uint8(np.clip(stego_image_BGR, 0, 255))
#
# # cuvanje stego-slike
# cv2.imwrite(stego_file, final_stego_image)
#
# original = cv2.imread(cover_file)
# cv2.imshow("Cover image", original)
# stego = cv2.imread(stego_file)
# cv2.imshow("Stego", stego)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

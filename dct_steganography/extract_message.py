import cv2
import struct
import bitstring
import numpy as np
import zigzag_matrix_scan as zz
import create_stego_image as src
import image_preprocessing as preprocess
import dct as dct

stego_image = cv2.imread(src.stego_file, flags=cv2.IMREAD_COLOR)
stego_image_f32 = np.float32(stego_image)
stego_image_YCC = preprocess.YCrCb(cv2.cvtColor(stego_image_f32, cv2.COLOR_BGR2YCrCb))


def read_message_from_stego(dct_blocks):
    message = ""
    for block in dct_blocks:
        for i in range(1, len(block)):
            ac_coeff = np.int32(block[i])
            if ac_coeff > 1:
                message += bitstring.pack('uint:1', np.uint8(block[i]) & 0x01)
    return message


def inverse_stego(image):
    #podaci su u luminance sloju, pa samo njega obradjujem
    dct_blocks = [dct.dct2(block) for block in image.channels[0]]

    #kvantizacija blokova
    dct_quants = [np.around(np.divide(item, preprocess.luminance_quant_table)) for item in dct_blocks]

    #cik-cak obilazak za sortiranje koeficijenata po frekvenciji
    sorted_coefficients = [zz.zigzag(block) for block in dct_quants]

    #citanje poruke
    data_from_coeff = read_message_from_stego(sorted_coefficients)

    #odredjivanje duzine poruke
    data_len = int(data_from_coeff.read('uint:32') / 8)

    # Extract secret message from DCT coefficients
    extracted_data = bytes()

    for _ in range(data_len):
        extracted_data += struct.pack('>B', data_from_coeff.read('uint:8'))

    return extracted_data.decode('ascii')


# extracted = inverse_stego(stego_image_YCC)
# print(extracted)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





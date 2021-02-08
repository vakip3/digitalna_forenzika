import create_stego_image as stego
import extract_message as reverse_stego
import cv2
import numpy as np
import image_preprocessing as preprocess

stego_file = "./stego_image.png"
cover_file = "./lenna.jpg"
message = "i want to tell you something i am here and i know things that would hurt you"

stego_image = stego.stego(cover_file, message)
# slika se konvertuje nazad u RGB format
stego_image_BGR = cv2.cvtColor(stego_image, cv2.COLOR_YCR_CB2BGR)

# pikseli na vrednost od 0 do 255
final_stego_image = np.uint8(np.clip(stego_image_BGR, 0, 255))

# cuvanje stego-slike
cv2.imwrite(stego_file, final_stego_image)

original = cv2.imread(cover_file)
cv2.imshow("Cover image", original)

stego_image = cv2.imread(stego.stego_file, flags=cv2.IMREAD_COLOR)
cv2.imshow("Stego", stego_image)

stego_image_f32 = np.float32(stego_image)
stego_image_YCC = preprocess.YCrCb(cv2.cvtColor(stego_image_f32, cv2.COLOR_BGR2YCrCb))

message = reverse_stego.inverse_stego(stego_image_YCC)
print(message)

height, width = original.shape[:2]
while height % 8:
    height += 1
while width % 8:
    width += 1
valid_dim = (width, height)
padded_image = cv2.resize(original, valid_dim)
cover_image_f32 = np.float32(padded_image)
psnr = cv2.PSNR(stego_image, padded_image)
print(psnr)


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


mse_err = mse(stego_image, padded_image)
print(mse_err)

cv2.waitKey(0)
cv2.destroyAllWindows()
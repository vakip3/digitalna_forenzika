import numpy as np


def zigzag(matrix):
    height = 0
    width = 0

    min_width = 0
    min_height = 0

    max_width = matrix.shape[0]
    max_height = matrix.shape[1]

    i = 0

    output = np.zeros((max_width * max_height))

    while (width < max_width) and (height < max_height):
        # smer navise
        if ((height + width) % 2) == 0:

            if width == min_width:
                # prvi red
                output[i] = matrix[width, height]

                if height == max_height:
                    width = width + 1
                else:
                    height = height + 1

                i = i + 1

            # poslednja kolona
            elif (height == max_height - 1) and (width < max_width):
                output[i] = matrix[width, height]
                width = width + 1
                i = i + 1

            # svi ostali slucajevi
            elif (width > min_width) and (height < max_height - 1):
                output[i] = matrix[width, height]
                width = width - 1
                height = height + 1
                i = i + 1

        # smer nanize
        else:
            # poslednja vrsta
            if (width == max_width - 1) and (height <= max_height - 1):
                output[i] = matrix[width, height]
                height = height + 1
                i = i + 1

            # prva kolona
            elif height == min_height:
                output[i] = matrix[width, height]

                if width == max_width - 1:
                    height = height + 1
                else:
                    width = width + 1

                i = i + 1

            # sve ostalo
            elif (width < max_width - 1) and (height > min_height):
                # print(6)
                output[i] = matrix[width, height]
                width = width + 1
                height = height - 1
                i = i + 1

        # donji desni element(poslednji)
        if (width == max_width - 1) and (height == max_height - 1):
            output[i] = matrix[width, height]
            break

    return output


def inverse_zigzag(matrix, max_width, max_height):
    height = 0
    width = 0

    min_width = 0
    min_height = 0

    output = np.zeros((max_width, max_height))

    i = 0
    # ----------------------------------

    while (width < max_width) and (height < max_height):
        if ((height + width) % 2) == 0:

            if width == min_width:
                output[width, height] = matrix[i]

                if height == max_height:
                    width = width + 1
                else:
                    height = height + 1

                i = i + 1

            elif (height == max_height - 1) and (width < max_width):
                output[width, height] = matrix[i]
                width = width + 1
                i = i + 1

            elif (width > min_width) and (height < max_height - 1):
                output[width, height] = matrix[i]
                width = width - 1
                height = height + 1
                i = i + 1

        else:

            if (width == max_width - 1) and (height <= max_height - 1):
                output[width, height] = matrix[i]
                height = height + 1
                i = i + 1

            elif height == min_height:
                output[width, height] = matrix[i]
                if width == max_width - 1:
                    height = height + 1
                else:
                    width = width + 1
                i = i + 1

            elif (width < max_width - 1) and (height > min_height):
                output[width, height] = matrix[i]
                width = width + 1
                height = height - 1
                i = i + 1

        if (width == max_width - 1) and (height == max_height - 1):
            output[width, height] = matrix[i]
            break

    return output

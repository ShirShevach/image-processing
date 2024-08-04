import cv2
import numpy as np

# q1:
def blend_images(image1, image2, mask, num_levels=6):
    # Split input images into color channels
    channels_img1 = cv2.split(image1)
    channels_img2 = cv2.split(image2)

    # Initialize an empty list to store the blended channels
    blended_channels = []

    # Blend each color channel separately
    for channel_img1, channel_img2 in zip(channels_img1, channels_img2):
        blended_channel = blend_channel(channel_img1, channel_img2, mask, num_levels)
        blended_channels.append(blended_channel)

    # Merge the blended color channels into the final blended image
    blended_image = cv2.merge(blended_channels)
    return blended_image


def blend_channel(channel_img1, channel_img2, mask, num_levels):
    # Build Gaussian pyramids for the two input color channels and the mask
    gaussian_pyr_img1 = [channel_img1.astype(np.float32)]
    gaussian_pyr_img2 = [channel_img2.astype(np.float32)]
    gaussian_pyr_mask = [mask.astype(np.float32) / 255.0]  # Normalize mask to range [0, 1]
    laplacian_pyr_img1 = []
    laplacian_pyr_img2 = []

    for _ in range(num_levels - 1):
        channel_img1 = cv2.pyrDown(channel_img1)
        channel_img2 = cv2.pyrDown(channel_img2)
        mask = cv2.pyrDown(mask)
        gaussian_pyr_img1.append(channel_img1.astype(np.float32))
        gaussian_pyr_img2.append(channel_img2.astype(np.float32))
        gaussian_pyr_mask.append(mask.astype(np.float32) / 255.0)  # Normalize mask to range [0, 1]

    # Reconstruct the Laplacian pyramid
    for i in range(num_levels - 1):
        expanded_img1 = cv2.pyrUp(gaussian_pyr_img1[i + 1],
                             dstsize=(gaussian_pyr_img1[i].shape[1], gaussian_pyr_img1[i].shape[0]))
        expanded_img2 = cv2.pyrUp(gaussian_pyr_img2[i + 1],
                             dstsize=(gaussian_pyr_img2[i].shape[1], gaussian_pyr_img2[i].shape[0]))

        laplacian_img1 = cv2.subtract(gaussian_pyr_img1[i], expanded_img1)
        laplacian_img2 = cv2.subtract(gaussian_pyr_img2[i], expanded_img2)

        laplacian_pyr_img1.append(laplacian_img1)
        laplacian_pyr_img2.append(laplacian_img2)

    # Lowest level of the Laplacian pyramid is same as the lowest level of the Gaussian pyramid
    laplacian_pyr_img1.append(gaussian_pyr_img1[-1])
    laplacian_pyr_img2.append(gaussian_pyr_img2[-1])

    # Initialize the Laplacian pyramid for the blended channel
    laplacian_pyr_blend = []

    # Blend the color channels at each level of the pyramid
    for img1, img2, mask in zip(laplacian_pyr_img1, laplacian_pyr_img2, gaussian_pyr_mask):
        blended_channel = img1 * (1 - mask) + img2 * mask
        laplacian_pyr_blend.append(blended_channel)

    blended_channel = laplacian_pyr_blend[-1]
    for i in range(num_levels - 2, -1, -1):
        blended_channel = cv2.pyrUp(blended_channel)
        blended_channel += laplacian_pyr_blend[i]

    blended_channel = np.clip(blended_channel, 0, 255).astype(np.uint8)

    return blended_channel

# q2:
def hybrid_images(image1, image2, num_levels=6):
    # Split input images into color channels
    channels_img1 = cv2.split(image1)
    channels_img2 = cv2.split(image2)

    # Initialize an empty list to store the blended channels
    hybrid_channels = []

    # Blend each color channel separately
    for channel_img1, channel_img2 in zip(channels_img1, channels_img2):
        hybrid_img_channel = hybrid_channel(channel_img1, channel_img2, num_levels)
        hybrid_channels.append(hybrid_img_channel)

    # Merge the blended color channels into the final blended image
    blended_image = cv2.merge(hybrid_channels)
    return blended_image

def hybrid_channel(channel_img1, channel_img2, num_levels):
    height = channel_img1.shape[0]
    width = channel_img1.shape[1]

    # Build Gaussian pyramids for the two input color channels and the mask
    gaussian_pyr_img1 = [channel_img1.astype(np.float32)]
    gaussian_pyr_img2 = [channel_img2.astype(np.float32)]
    laplacian_pyr_img1 = []
    laplacian_pyr_img2 = []

    for i in range(num_levels - 1):
        channel_img1 = cv2.pyrDown(channel_img1)
        channel_img2 = cv2.pyrDown(channel_img2)
        gaussian_pyr_img1.append(channel_img1.astype(np.float32))
        gaussian_pyr_img2.append(channel_img2.astype(np.float32))

    # Reconstruct the Laplacian pyramid
    for i in range(num_levels - 1):
        expanded_img1 = cv2.pyrUp(gaussian_pyr_img1[i + 1],
                             dstsize=(gaussian_pyr_img1[i].shape[1], gaussian_pyr_img1[i].shape[0]))
        expanded_img2 = cv2.pyrUp(gaussian_pyr_img2[i + 1],
                             dstsize=(gaussian_pyr_img2[i].shape[1], gaussian_pyr_img2[i].shape[0]))

        laplacian_img1 = cv2.subtract(gaussian_pyr_img1[i], expanded_img1)
        laplacian_img2 = cv2.subtract(gaussian_pyr_img2[i], expanded_img2)

        laplacian_pyr_img1.append(laplacian_img1)
        laplacian_pyr_img2.append(laplacian_img2)

    laplacian_pyr_img1.append(gaussian_pyr_img1[-1])
    laplacian_pyr_img2.append(gaussian_pyr_img2[-1])

    # Initialize the Laplacian pyramid for the blended channel
    laplacian_pyr_blend = []

    for i in range(num_levels):
        if i in [0, 1]:
            blended_channel = laplacian_pyr_img2[i]
        else:
            blended_channel = laplacian_pyr_img1[i]

        laplacian_pyr_blend.append(blended_channel)

    # Reconstruct the blended channel from the Laplacian pyramid
    blended_channel = laplacian_pyr_blend[-1]
    for i in range(num_levels - 2, -1, -1):
        blended_channel = cv2.pyrUp(blended_channel)
        blended_channel += laplacian_pyr_blend[i]

    # Ensure the values are within the valid range
    blended_channel = np.clip(blended_channel, 0, 255).astype(np.uint8)

    return blended_channel





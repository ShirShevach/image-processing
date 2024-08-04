import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


def compute_keypoints_and_descriptors(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Create SIFT object
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return keypoints, descriptors


def match_descriptors(descriptors1, descriptors2):
    from sklearn.neighbors import NearestNeighbors
    # Create NearestNeighbors model
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(descriptors2)

    # Find nearest neighbors for each descriptor in descriptor1
    distances, indices = nbrs.kneighbors(descriptors1)

    # Extract distances to the closest and second closest neighbors
    closest_distances = distances[:, 0]
    second_closest_distances = distances[:, 1]

    # Compute the ratio of distances
    ratios = closest_distances / second_closest_distances

    # Best match threshold
    threshold = 0.9

    # Filter matches based on threshold
    below_threshold_indices = np.where(ratios < threshold)[0]

    # Convert indices to DMatch objects
    matches = [cv2.DMatch(i, indices[i, 0], 0) for i in below_threshold_indices]

    # Return the DMatch objects
    return matches


def transform_image(image1, image2, keypoints1, keypoints2, matches):
    # Extract matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # RANSAC Threshold
    ransac_reproj_threshold = 45

    # Find the perspective transformation using RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_threshold)

    # Apply the transformation to the first image
    transformed_image = cv2.warpPerspective(image1, M, (image2.shape[1], image2.shape[0]))

    return transformed_image


def create_mask(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to get a binary mask
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Normalize the mask to have values in the range [0, 1]
    mask = mask.astype(float) / 255

    return mask


def paste_two_images(image2, transformed_image):
    # Create the mask
    mask = create_mask(transformed_image)

    # Split channels
    channels_img1 = cv2.split(transformed_image)
    channels_img2 = cv2.split(image2)

    # Merge two images using mask
    blended_channels = []
    blended_channels.append(channels_img1[0] * mask + channels_img2[0] * (1 - mask))
    blended_channels.append(channels_img1[1] * mask + channels_img2[1] * (1 - mask))
    blended_channels.append(channels_img1[2] * mask + channels_img2[2] * (1 - mask))

    # Merge channels
    blended_image = cv2.merge(blended_channels)

    return blended_image


def main(name_img):
    # Path to the images
    image1_path = f'{name_img}_high_res.png'
    image2_path = f'{name_img}_low_res.jpg'
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Compute keypoints and descriptors for the first image
    keypoints1, descriptors1 = compute_keypoints_and_descriptors(image1_path)

    # Compute keypoints and descriptors for the second image
    keypoints2, descriptors2 = compute_keypoints_and_descriptors(image2_path)

    # Match descriptors between the images
    matches = match_descriptors(descriptors1, descriptors2)

    # Finding transformation by RANSAC algorithm
    transformed_image = transform_image(image1, image2, keypoints1, keypoints2, matches)

    blended_image = paste_two_images(image2, transformed_image)

    # Draw the matches
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the result
    cv2.imwrite(f'{name_img}_Matches.png', matched_image)
    cv2.imwrite(f'{name_img}_Transformed_image.png', transformed_image)
    cv2.imwrite(f'{name_img}_blended_image.jpg', blended_image)


if __name__ == "__main__":
    main("desert")
    main("lake")

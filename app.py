# Iimport requests
import numpy as np

def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img

# Using a different reliable source to avoid persistent 403 blocks
image_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/image2.jpg"
image_path = tf.keras.utils.get_file("project_sample_final.jpg", image_url)

img = load_img(image_path)
converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

start_time = time.time()
result = detector(converted_img)
end_time = time.time()

result = {key:value.numpy() for key,value in result.items()}

print(f"Found {len(result['detection_scores'])} objects.")
print(f"Inference time: {end_time - start_time:.2f}s")

image_with_boxes = draw_boxes(
    img.numpy(), result["detection_boxes"],
    result["detection_class_entities"], result["detection_scores"])

plt.figure(figsize=(12, 8))
plt.imshow(image_with_boxes)
plt.axis('off')
plt.show()

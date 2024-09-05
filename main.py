import cv2
import numpy as np
from PIL import Image
import time

def apply_blur(image, weight):
    if weight < 1 or weight > 10:
        raise ValueError(f"Weight must be between 1 and 10. Received value: {weight}")

    # Calculate kernel size
    kernel_size = weight * 2 + 1

    # Apply Gaussian blur multiple times
    blurred_image = image
    for _ in range((weight // 2) + 1):
        blurred_image = cv2.GaussianBlur(blurred_image, (kernel_size, kernel_size), cv2.BORDER_DEFAULT)
    
    return blurred_image

def apply_color_lut(image):
    lut_in = [0, 255, 80]
    lut_out = [0, 0, 255]
    model = np.arange(256)

    lut = np.interp(model, lut_in, lut_out).astype(np.uint8)
    return cv2.LUT(image, lut)

def scale_image(image, target_width, scale_up=True):
    height, width = image.shape[:2]
    ratio = width / height

    if scale_up:
        new_width = target_width
        new_height = int(new_width / ratio)
    else:
        new_height = target_width
        new_width = int(new_height * ratio)
    
    return cv2.resize(image, (new_width, new_height))

def enhance_logos(image, blur_weight, output_scale):
    image = Image.fromarray(image).convert('RGB')
    width, height = image.size
    
    image_np = np.array(image)
    
    # Step 1: High resolution upscale
    high_res_image = scale_image(image_np, width + 1000, scale_up=True)
    
    # Step 2: Apply blur
    blurred_image = apply_blur(high_res_image, blur_weight)
    
    # Step 3: Apply color LUT
    color_adjusted_image = apply_color_lut(blurred_image)
    
    # Step 4: Resize to final output scale
    final_image = scale_image(color_adjusted_image, output_scale)
    
    return final_image

def process_image(src_image, output_scale, blur_weight):
    # Original redimensionada
    original_resized = scale_image(src_image, output_scale, scale_up=True)
    
    # Aplicar Gaussian blur e redimensionar
    blurred_image = apply_blur(original_resized, blur_weight)
    
    # Imagem final com todas as técnicas
    final_image = enhance_logos(src_image, blur_weight=blur_weight, output_scale=output_scale)

    # Garantir que todas as imagens estão em BGR (sem alpha)
    if original_resized.shape[2] == 4:
        original_resized = cv2.cvtColor(original_resized, cv2.COLOR_BGRA2BGR)
    if blurred_image.shape[2] == 4:
        blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGRA2BGR)
    if final_image.shape[2] == 4:
        final_image = cv2.cvtColor(final_image, cv2.COLOR_BGRA2BGR)

    return original_resized, blurred_image, final_image

def main():
    start_time = time.time()
    
    src_image = cv2.imread('Assets/obama.jpg', cv2.IMREAD_UNCHANGED)

    # Parâmetros
    blur_weight = 10
    output_scale = 800
    
    # Processamento das imagens
    original_resized, blurred_image, final_image = process_image(src_image, output_scale, blur_weight)

    # Ajustar todas as imagens para o mesmo tamanho de altura
    common_height = min(original_resized.shape[0], blurred_image.shape[0], final_image.shape[0])

    original_resized = cv2.resize(original_resized, (int(original_resized.shape[1] * common_height / original_resized.shape[0]), common_height))
    blurred_image = cv2.resize(blurred_image, (int(blurred_image.shape[1] * common_height / blurred_image.shape[0]), common_height))
    # lut_image = cv2.resize(lut_image, (int(lut_image.shape[1] * common_height / lut_image.shape[0]), common_height))
    final_image = cv2.resize(final_image, (int(final_image.shape[1] * common_height / final_image.shape[0]), common_height))

    # Combinar as 4 imagens lado a lado
    combined_image = np.hstack((original_resized, blurred_image, final_image))
    
    end_time = time.time()
    cv2.imshow('Comparison: Original, Blur, Final', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Total time: {end_time - start_time:.5f} seconds")

if __name__ == "__main__":
    main()
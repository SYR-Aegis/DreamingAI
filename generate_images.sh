model=${1}
num_images=${2}
path=${3}

python3 generate_images.py ${model} ${num_images} ${path} && python3 upscale_image.py ${path};
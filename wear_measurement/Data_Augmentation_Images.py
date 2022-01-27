##############################################
########## Data_Augmentation_Images ##########
##############################################

##########FK Version: 10.01.2022##############


from PIL import Image, ImageFilter, ImageEnhance
import os

# folder = Dateipfad, in dem Originalbilder gespeichert sind
folder = "Z:/Elements/Fabian/KIT/Bachelor_Arbeit/Database/Semantic_Segmentation/New_Masks/"
# copy_to_path = Dateipfad, in dem die Augmendet Bilder gespeicert werden sollen
copy_to_path = "Z:/Elements/Fabian/KIT/Bachelor_Arbeit/Database/Semantic_Segmentation/Additional_Database_3_2/"

# Für Zoom Bilder um ...% beschneiden und auf benötigte Größe skalieren:
cropbox=(51, 51, 461, 461)

# Bilder müssen sortiert aus den Ordnern geladen werden und sortiert abgespeichert, damit Masken genau den Bildern entsprechen
images = []
i = 1
for filename in sorted(os.listdir(folder)):
    Original_Image = Image.open(folder+filename)
    print(filename)

    rotated_image0 = Original_Image.resize((512, 512))
    filename = os.path.splitext(filename)
    if i<10:
        new_name = 'Image_' + str(0) + str(0) + str(0) + str(i)               
    elif i >= 10 and i < 100:
        new_name = 'Image_' + str(0) + str(0) + str(i)
    elif 100<=i and i<1000:
        new_name = 'Image_' + str(0) + str(i)
    elif 1000<=i and i<10000:
        new_name = 'Image_' + str(i)
    rotated_image0.save(copy_to_path + new_name + '.tif')
    i=i+1

# Drehen um 10°
for filename in sorted(os.listdir(folder)):
    Original_Image = Image.open(folder+filename)
    print(filename)
    rotated_image0 = Original_Image.rotate(10).resize((512, 512))
    filename = os.path.splitext(filename)
    if i<10:
        new_name = 'Image_' + str(0) + str(0) + str(0) + str(i)
    elif i >= 10 and i < 100:
        new_name = 'Image_' + str(0) + str(0)+ str(i)
    elif 100<=i and i<1000:
        new_name = 'Image_' + str(0) + str(i)
    elif 1000<=i and i<10000:
        new_name = 'Image_' + str(i)
    rotated_image0.save(copy_to_path + new_name + '.tif')
    i=i+1

# Drehen um 180°
for filename in sorted(os.listdir(folder)):
    Original_Image = Image.open(folder+filename)
    print(filename)
    rotated_image0 = Original_Image.rotate(180).resize((512, 512))
    filename = os.path.splitext(filename)
    if i<10:
        new_name = 'Image_' + str(0) + str(0) + str(0) + str(i)
    elif i >= 10 and i < 100:
        new_name = 'Image_' + str(0) + str(0)+ str(i)
    elif 100<=i and i<1000:
        new_name = 'Image_' + str(0) + str(i)
    elif 1000<=i and i<10000:
        new_name = 'Image_' + str(i)
    rotated_image0.save(copy_to_path + new_name + '.tif')
    i=i+1

# drehen um -10°
for filename in sorted(os.listdir(folder)):
    Original_Image = Image.open(folder+filename)
    print(filename)
    rotated_image0 = Original_Image.rotate(350).resize((512, 512))
    filename = os.path.splitext(filename)
    if i<10:
        new_name = 'Image_' + str(0) + str(0) + str(0) + str(i)
    elif i >= 10 and i < 100:
        new_name = 'Image_' + str(0) + str(0)+ str(i)
    elif 100<=i and i<1000:
        new_name = 'Image_' + str(0) + str(i)
    elif 1000<=i and i<10000:
        new_name = 'Image_' + str(i)
    rotated_image0.save(copy_to_path + new_name + '.tif')
    i=i+1

# Anwendung Gaussian Blur
for filename in sorted(os.listdir(folder)):
    Original_Image = Image.open(folder+filename)
    print(filename)
    rotated_image0 = Original_Image.rotate(0).resize((512, 512)).filter(filter=ImageFilter.GaussianBlur)
    filename = os.path.splitext(filename)
    if i<10:
        new_name = 'Image_' + str(0) + str(0) + str(0) + str(i)
    elif i >= 10 and i < 100:
        new_name = 'Image_' + str(0) + str(0)+ str(i)
    elif 100<=i and i<1000:
        new_name = 'Image_' + str(0) + str(i)
    elif 1000<=i and i<10000:
        new_name = 'Image_' + str(i)
    rotated_image0.save(copy_to_path + new_name + '.tif')
    i=i+1

# ZOOM
for filename in sorted(os.listdir(folder)):
    Original_Image = Image.open(folder+filename)
    print(filename)
    rotated_image0 = Original_Image.rotate(0).crop(cropbox).resize((512, 512))
    filename = os.path.splitext(filename)
    if i<10:
        new_name = 'Image_' + str(0) + str(0) + str(0) + str(i)
    elif i >= 10 and i < 100:
        new_name = 'Image_' + str(0) + str(0)+ str(i)
    elif 100<=i and i<1000:
        new_name = 'Image_' + str(0) + str(i)
    elif 1000<=i and i<10000:
        new_name = 'Image_' + str(i)
    rotated_image0.save(copy_to_path + new_name + '.tif')
    i=i+1

# Kontrast erhöhen
for filename in sorted(os.listdir(folder)):
    Original_Image = Image.open(folder+filename)
    print(filename)
    rotated_image0 = Original_Image.rotate(0).resize((512, 512))
    enhancer = ImageEnhance.Contrast(rotated_image0)
    rotated_image0 = enhancer.enhance(factor=1.4)
    filename = os.path.splitext(filename)
    if i<10:
        new_name = 'Image_' + str(0) + str(0) + str(0) + str(i)
    elif i >= 10 and i < 100:
        new_name = 'Image_' + str(0) + str(0)+ str(i)
    elif 100<=i and i<1000:
        new_name = 'Image_' + str(0) + str(i)
    elif 1000<=i and i<10000:
        new_name = 'Image_' + str(i)
    rotated_image0.save(copy_to_path + new_name + '.tif')
    i=i+1

# Helligkeit erhöhen
for filename in sorted(os.listdir(folder)):
    Original_Image = Image.open(folder+filename)
    print(filename)
    rotated_image0 = Original_Image.rotate(0).resize((512, 512))
    enhancer = ImageEnhance.Brightness(rotated_image0)
    rotated_image0 = enhancer.enhance(factor=1.2)
    filename = os.path.splitext(filename)
    if i<10:
        new_name = 'Image_' + str(0) + str(0) + str(0) + str(i)
    elif i >= 10 and i < 100:
        new_name = 'Image_' + str(0) + str(0)+ str(i)
    elif 100<=i and i<1000:
        new_name = 'Image_' + str(0) + str(i)
    elif 1000<=i and i<10000:
        new_name = 'Image_' + str(i)
    rotated_image0.save(copy_to_path + new_name + '.tif')
    i=i+1

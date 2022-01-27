############################################
########## Data_Augmentation_Masks##########
############################################

##########FK Version: 10.01.2022##############

# Der Ablauf ist genau der gleiche wie bei Data_Augmentation_Images. Es werden aber die FIlter wie Gausian Blur,
# Kontrast und Helligkeit nicht verwendet, da dies kein Unterschied bei den Masken macht

from PIL import Image, ImageFilter, ImageEnhance
import os

# folder = Dateipfad, in dem Originalmasken gespeichert sind
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

# Drehen um -10°
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

# An dieser Stelle ist bei den Bildern Gaussian Blur
for filename in sorted(os.listdir(folder)):
    Original_Image = Image.open(folder+filename)
    print(filename)
    rotated_image0 = Original_Image.rotate(0).resize((512, 512))
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

# An dieser Stelle ist bei den Bildern Erhöhung des Kontrastes
for filename in sorted(os.listdir(folder)):
    Original_Image = Image.open(folder+filename)
    print(filename)
    rotated_image0 = Original_Image.rotate(0).resize((512, 512))
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

# An dieser Stelle ist bei den Bildern Erhöhung der Helligkeit
for filename in sorted(os.listdir(folder)):
    Original_Image = Image.open(folder+filename)
    print(filename)
    rotated_image0 = Original_Image.rotate(0).resize((512, 512))
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

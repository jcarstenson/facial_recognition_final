from pyrobot.vision import *
from commands import getstatusoutput as run
from copy import deepcopy
from random import randint
import math
from PIL import Image

# Useful functions for processing images for a neural network

img_width = 450
img_height = 450

def normalizeImage(data):
    """
    This function takes a grayscale image with values between 0 and a 
    maximum of 255, and normalizes the values between 0 and 1.
    """
    maxValue = float(max(data))
    for i in range(len(data)):
        data[i] = data[i]/maxValue
    
def getImages(directory, imagesFilenames, outputFile):
    """
    This function can be used to read in a series of pgm images from
    files and convert them into a format that is suitable for neural
    network training.  The directory parameter should be the location
    of the pgm files.  The imagesFilenames parameter should be the
    name of a file that contains image file names, one per line.  The
    outputFile should be the name of a file where the converted images
    will be written.
    """
    out = open(outputFile, "w")
    names = open(imagesFilenames, "r")
    while 1:
        name = names.readline().strip()
        if len(name) == 0: break
        image = PyrobotImage(depth=1)
        image.loadFromFile(directory+name) 
        normalizeImage(image.data)
        for value in image.data:
            out.write("%.4f " % value)
        out.write("\n")
    names.close()
    out.close()


"""
createData -- for each PPM image with filepath given in annotations_f with only one face,
                save one 20x20 PPM that is a face, and two 20x20 PPMs that are not faces,
                in new_dir. The filenames are integers, that correspond to the order of the
                images in target_output_f (each line is 0 (not face) or 1 (face)).
"""
def createData(old_dir, annotations_list, target_output_f, new_dir, output_img_names_f):
    target_output = open(target_output_f, "w")
    output_filenames = open(output_img_names_f, "w")

    counter = 1
    for annotation_f in annotations_list:
        annotations = open(annotation_f, "r")

        while 1:
            name = annotations.readline().strip()
            if len(name) == 0: break

            num_faces = int(annotations.readline().strip())
            for i in range(num_faces):
                ellipse = annotations.readline().strip() # need to read the line anyway

                if num_faces == 1:   # only deal with one face images so its easier to get the non-face squares
                    # we only enter this if statement once

                    features = ellipse.split()
                    maj_axis_r, min_axis_r, angle, cen_x, cen_y, detection_score = [float(feature) for feature in features]

                    image = Image.open(old_dir+name+".jpg")
                    # image = PyrobotImage(depth=3)
                    # image.loadFromFile(old_dir+name+".ppm") 

                    total_width, total_height = image.size

                    #### IMAGE WITH A FACE

                    # crop to a not-rotated square. TODO ROTATED?
                    left_x = int(round(cen_x - min_axis_r))
                    right_x = int(round(cen_x + min_axis_r))
                    top_y = int(round(cen_y - min_axis_r))
                    bot_y = int(round(cen_y + min_axis_r))

                    face_img = image.copy()
                    face_img = face_img.crop((left_x, top_y, right_x, bot_y))
                    
                    unscaled_width, unscaled_height = face_img.size

                    # shrink to a 20x20 square
                    face_img_sm = face_img.resize((20,20))

                    #### ONE IMAGE WITHOUT A FACE

                    if (unscaled_width + maj_axis_r < cen_x) or (cen_x + maj_axis_r + unscaled_width +1 < total_width):
                        

                        filename = name.replace("/","_")+".ppm"
                        #filename = str(counter)+".ppm"
                        output_filenames.write(filename+"\n")

                        face_img_sm.save(new_dir+filename)


                        counter += 1
                        
                        target_output.write("1"+"\n")


                        if unscaled_width + maj_axis_r < cen_x:
                            offset_l = randint(0, int(round(cen_x - unscaled_width - maj_axis_r)))

                        else: # we ahve (cen_x + maj_axis_r + unscaled_width < total_width)
                            offset_l = randint(int(round(cen_x + maj_axis_r +1)), int(round(total_width - unscaled_width)))

                        offset_top1 = randint(0,int(round(total_height - maj_axis_r*2)))

                        notface_1 = image.copy()
                        notface_1 = notface_1.crop((offset_l, offset_top1, offset_l + unscaled_width, offset_top1+unscaled_height))

                        # shrink to a 20x20 square
                        notface_1sm = notface_1.resize((20,20))

                        filename = name.replace("/","_")+"not.ppm"

                        #filename = str(counter)+"not.ppm"
                        output_filenames.write(filename+"\n")
                        notface_1sm.save(new_dir+filename)

                        target_output.write("0"+"\n")

                        counter += 1

                        """
                        todo:
                        * rotate the face
                        """

        annotations.close()

    output_filenames.close()
    target_output.close()

#getOnlyOneFace("FDDB-folds/FDDB-fold-01-ellipseList.txt","fold1oneface.txt") # gets a list of the ones we're dealing w/ to convert ot PPM
#convertToPPM("/local/rzevall1jcarste1cs63/", "fold1oneface.txt", "/local/rzevall1jcarste1cs63/")

annotationsList = []
for i in [1,4,5,6,7,8,9,10]:
    name = "FDDB-folds/%d.txt" % i
    annotationsList.append(name)

createData("/local/rzevall1jcarste1cs63/", annotationsList, "targets.dat", "/local/rzevall1jcarste1cs63/faceinputs/","FaceFilenames.txt")

getImages("/local/rzevall1jcarste1cs63/faceinputs/", "FaceFilenames.txt", "inputs.dat")
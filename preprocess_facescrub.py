import csv
from multiprocessing import Pool
from urllib.request import urlretrieve
import urllib.error
import os
import cv2
import ssl
import random
import hashlib
import numpy as np
import sys

# Name of folder under where Annotations and JPEGImages folders will be created and populated from Facescrub
facescrub_folder = "/path/to/outputDir/"
# Names of the Facescrub data files located in the facescrub_folder listed above 
# Get these files from the Facescrub team; Go to http://www.vintage.winklerbros.net/facescrub.html
file_list = ["facescrub_actors.txt","facescrub_actresses.txt"]
# Number of lines to process from the Facescrub files; Training on faces will require ~9000 images and many Facescrub entries fail
file_count_limit = 15000
# Multiprocessing threads to use
threads = 8 
# Image dimension limit to prevent memory issues from gigantic images in training set 
image_dim_limit = 800


# Returns a string in XML format aligned with VOC2007 definition
def create_xml(folder_name,file_name,face_loc,image_dims): 

    return '''
    <annotation>
        <folder>'''+folder_name+'''</folder>
        <filename>'''+file_name+'''</filename>
        <source>
            <database>FaceScrub</database>
            <annotation>FaceScrub</annotation>
            <image>FaceScrub</image>
            <flickrid>FaceScrub</flickrid>
        </source>
        <owner>
            <flickrid>FaceScrub</flickrid>
            <name>FaceScrub</name>
        </owner>
        <size>
            <width>'''+str(image_dims[1])+'''</width>
            <height>'''+str(image_dims[0])+'''</height>
            <depth>'''+str(image_dims[2])+'''</depth>
        </size>
        <segmented>0</segmented>
        <object>
            <name>face</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>'''+str(face_loc[0])+'''</xmin>
                <ymin>'''+str(face_loc[1])+'''</ymin>
                <xmax>'''+str(face_loc[2])+'''</xmax>
                <ymax>'''+str(face_loc[3])+'''</ymax>
            </bndbox>
        </object>
    </annotation>'''

# Generate sha256 hash of file content to match Facescrub data
def file_digest(in_filename):

    # Get SHA256 hash of file
    BLOCKSIZE = 65536
    hasher = hashlib.sha256()
    with open(in_filename, 'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)
    return hasher.hexdigest()

# Attempts to download image from url provided in Facescrub data file
def get_facescrub_image(entry):

    name, url, face_loc, hash_val = entry
    face_loc = [int(coord) for coord in face_loc]
    jpg_outfile = os.path.join(jpg_path,name + ".jpg")  #os.path.splitext(url)[-1])
    
    if os.path.isfile(jpg_outfile):
        #print("{0} already downloaded; skipping".format(name))
        return True
    else:
        try:
            # Get JPEG from URL and write it to disk
            _, headers = urlretrieve(url,jpg_outfile)
            # Check that an image is being returned
            if (headers["content-type"] is not None) and headers["content-type"].startswith("image"):
                img = cv2.imread(jpg_outfile)
                # Verify matching hash, 3 color channels before keeping image and writing XML
                if file_digest(jpg_outfile) == hash_val and img.shape[2]==3 \
                  and img.shape[0]<=image_dim_limit and img.shape[1]<=image_dim_limit:
                    # Ensure that future color operations will succeed on this image
                    test = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    test = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    xml_outfile = os.path.join(xml_path, name + ".xml")
                    with open(xml_outfile, "w") as xml_file:
                        xml_file.write(create_xml("VOC2007", name + ".jpg", face_loc, img.shape))
                    print("Processed {0}".format(name))
                    return True
                else:
                    #print("Hash does not match for {0}".format(name))
                    if os.path.isfile(jpg_outfile):
                        os.remove(jpg_outfile)
                    return False
            else:
                #print("Unexpected HTTP headers for {0}".format(name))
                if os.path.isfile(jpg_outfile):
                    os.remove(jpg_outfile)
                return False

        except urllib.error.HTTPError:
            #print("Caught HTTP error")
            if os.path.isfile(jpg_outfile):
                os.remove(jpg_outfile)
            return False
        except urllib.error.URLError:
            #print("Caught URL error")
            if os.path.isfile(jpg_outfile):
                os.remove(jpg_outfile)
            return False
        except ConnectionResetError:
            #print("Caught connection error")
            if os.path.isfile(jpg_outfile):
                os.remove(jpg_outfile)
            return False
        except ssl.CertificateError:
            #print("Caught SSL error")
            if os.path.isfile(jpg_outfile):
                os.remove(jpg_outfile)
            return False
        except ssl.SSLError:
            #print("Caught SSL error")
            if os.path.isfile(jpg_outfile):
                os.remove(jpg_outfile)
            return False
        except TimeoutError:
            #print("Caught timeout error")
            if os.path.isfile(jpg_outfile):
                os.remove(jpg_outfile)
            return False
        except:
            print("Caught unknown error:",sys.exc_info()[1])
            if os.path.isfile(jpg_outfile):
                os.remove(jpg_outfile)
            return False

facescrub = list()
stats = list()

# Set path for jpg and xml outputs
jpg_path = os.path.join(facescrub_folder,"JPEGImages")
if not os.path.isdir(jpg_path):
        os.makedirs(jpg_path, exist_ok=True)
xml_path = os.path.join(facescrub_folder,"Annotations")
if not os.path.isdir(xml_path):
        os.makedirs(xml_path, exist_ok=True)

# Read Facescrub actor and actresses files
for file_name in [os.path.join(facescrub_folder,item) for item in file_list]:
    facescrub += list(csv.reader(open(file_name),delimiter='\t'))

# Take a random sample of the actors and actresses in the Facescrub data set up to the file_count_limit
facescrub_randomized = random.sample(facescrub,file_count_limit)

# Remove spaces from name and add hash to the end to make unique file name
# (actorname_hash, file_ext, url, [face_loc], hash)
facescrub_formatted = [(a[0].replace(" ", "")+"_"+a[1],a[3],a[4].split(','),a[5]) for a in facescrub_randomized]

# Download the Facescrub images
print("Processing {0} of {1} Facescrub entries...".format(file_count_limit,len(facescrub)))

# Use multiprocessing; Large file_count_limit setting (ex: >3000) may cause it to hang on exit with file fragments
with Pool(threads) as p:
    stats = p.map(get_facescrub_image, facescrub_formatted)

# Or use standard iteration; This is very slow for large file_count_limit settings
# for entry in facescrub_formatted:
#     stats.append(get_facescrub_image(entry))

total = len(stats)
successes = len([a for a in stats if a is True])
print("{0} total entries attempted".format(total))
print("{0} entries processed successfully".format(successes))
print("{0} entries failed".format(total - successes))

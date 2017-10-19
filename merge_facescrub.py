import os
from collections import defaultdict
import random
import shutil

# List of VOC datasets
datasets = ["VOC2007","VOC2012"]
# Folder location of extracted original VOC datasets; See Preparation section of https://github.com/weiliu89/caffe/tree/ssd for links
VOC_folder = "/path/to/VOCdevkit/"
# Folder location of downloaded and preprocessed Facescrub data already obtained by executing preprocess_facescrub.py
facescrub_folder = "/path/to/preprocessed/facescrub_data/"
# Directory list of facescrub files in randomly shuffled order so sequential images in list are not the same person
facescrub_files = os.listdir(os.path.join(facescrub_folder,"JPEGImages"))
random.seed(41)
random.shuffle(facescrub_files)

for dataset in datasets:
    print("*** Dataset {0} ***".format(dataset))

    # Dictionary to track the number of entries needed to replenish each file after removing 'person' files
    replenish_files = defaultdict(int)
    VOC_dataset_main_folder=os.path.join(VOC_folder, dataset, "ImageSets", "Main")

    # Examine person files to keep non-person person images and remove person images
    person_stage_list = ["person_test.txt","person_trainval.txt","person_train.txt","person_val.txt"]
    person_file_remove = defaultdict(set)
    person_file_keep =  defaultdict(list)
    for label_stage in person_stage_list:
        if os.path.isfile(os.path.join(VOC_dataset_main_folder, label_stage)):
            with open(os.path.join(VOC_dataset_main_folder, label_stage), "r") as in_file:
                raw_file_contents = in_file.readlines()
                for line in raw_file_contents:
                    flag = line[-3:-1].replace(" ", "")
                    if flag == "-1":
                        person_file_keep[label_stage].append(line)
                    else:
                        if dataset == "VOC2007":
                            file_ref = line[0:6]
                            person_file_remove[label_stage].add(file_ref)
                        if dataset == "VOC2012":
                            file_ref = line[0:11]
                            person_file_remove[label_stage].add(file_ref)
                face_stage = label_stage.replace("person", "face")
                replenish_files[face_stage] = len(person_file_remove[label_stage])

    # Write out face file with images that don't have a person and delete the original person files
    for label_stage in person_stage_list:
        person_file_name = os.path.join(VOC_dataset_main_folder, label_stage)
        if os.path.isfile(person_file_name):
            face_stage = label_stage.replace("person","face")
            with open(os.path.join(VOC_dataset_main_folder, face_stage),"w") as out_file:
                out_file.writelines(person_file_keep[label_stage])
            os.remove(person_file_name)

    # Make union of all files that have a person in them so we can remove them everywhere
    all_person_files = person_file_remove["person_test.txt"] | person_file_remove["person_trainval.txt"] \
                       | person_file_remove["person_train.txt"] | person_file_remove["person_val.txt"]

    # List of all labels from VOC2007 and VOC2012 except person
    labels = ["aeroplane","bicycle","bird","boat","bottle","bus","car","chair","cow","diningtable","dog",\
    "horse","motorbike","pottedplant","sheep","sofa","train","tvmonitor"]
    labels_dict = {label: [label+"_test.txt",label+"_train.txt",label+"_trainval.txt",label+"_val.txt"] \
                   for label in labels}

    # VOC2012 includes an unfortunately named train.txt file in addition to the train object files
    if dataset == "VOC2012": labels_dict["train"].append("train.txt")

    # Include files that refer to stages rather than objects
    special_labels = ["test","trainval","val"]
    for special_label in special_labels:
        labels_dict[special_label]=[special_label+".txt"]

    # Rewrite label file contents to remove any images labeled as person
    for label in labels_dict:
        for label_stage in labels_dict[label]:
            if os.path.isfile(os.path.join(VOC_dataset_main_folder, label_stage)):
                with open(os.path.join(VOC_dataset_main_folder,label_stage),"r") as in_file:
                    raw_file_contents = in_file.readlines()

                # VOC2007 names files with 6 numbers
                if dataset=="VOC2007":
                    file_list_to_check = {line[0:6] for line in raw_file_contents}
                    file_list_to_remove = all_person_files.intersection(file_list_to_check)
                    file_list_to_write = [line for line in raw_file_contents if line[0:6] not in file_list_to_remove]
                #VOC2012 names files with 11 numbers
                if dataset == "VOC2012":
                    file_list_to_check = { line[0:11] for line in raw_file_contents }
                    file_list_to_remove = all_person_files.intersection(file_list_to_check)
                    file_list_to_write = [line for line in raw_file_contents if line[0:11] not in file_list_to_remove]

                replenish_files[label_stage] = len(file_list_to_remove)

                with open(os.path.join(VOC_dataset_main_folder,label_stage),"w") as out_file:
                    out_file.writelines(file_list_to_write)

    # Replenish trainval.txt file with images labeled as face
    trainval_options = list()
    trainval_goal = replenish_files.pop("trainval.txt")
    print("Adding {0} faces to trainval.txt".format(trainval_goal))

    if len(facescrub_files) < trainval_goal:
        print("Insufficent Facescrub JPG files. Have {0} Need {1}".format(len(facescrub_files),trainval_goal))
        quit()
    with open(os.path.join(VOC_dataset_main_folder, "trainval.txt"), "a") as out_file:
        for i in range(trainval_goal):
            candidate = facescrub_files[i]
            #out_file.write(candidate + "\n")
            out_file.write(os.path.splitext(candidate)[0]+"\n")
            trainval_options.append(candidate)

    # Slice off facescrub files reserved for trainval in this VOC dataset
    facescrub_files = facescrub_files[trainval_goal:]

    # Replenish test.txt file with images labeled as face
    test_options = list()

    if dataset=="VOC2007":
        test_goal = replenish_files.pop("test.txt")
        print("Adding {0} face to test.txt".format(test_goal))

        if len(facescrub_files) < test_goal:
            print("Insufficent Facescrub JPG files. Have {0} Need {1}".format(len(facescrub_files), test_goal))
            quit()

        with open(os.path.join(VOC_dataset_main_folder, "test.txt"), "a") as out_file:
            for i in range(test_goal):
                candidate = facescrub_files[i]
                #out_file.write(candidate + "\n")
                out_file.write(os.path.splitext(candidate)[0] + "\n")
                test_options.append(candidate)

        # Slice off facescrub files reserved for test in this VOC dataset
        facescrub_files = facescrub_files[test_goal:]

    # Replenish files with face images
    trainval_options_false = [os.path.splitext(item)[0] + " -1\n" for item in trainval_options]
    trainval_options_true = [os.path.splitext(item)[0] + "  1\n" for item in trainval_options]
    test_options_false = [os.path.splitext(item)[0] + " -1\n" for item in test_options]
    test_options_true = [os.path.splitext(item)[0] + "  1\n" for item in test_options]

    for category in replenish_files:
        num_needed = replenish_files[category]
        print("Adding {1} faces to {0}".format(category,num_needed))
        with open(os.path.join(VOC_dataset_main_folder, category), "a") as out_file:
            if category in ["face_trainval.txt", "face_train.txt", "face_val.txt"]:
                out_file.writelines(trainval_options_true[0:num_needed])
            elif category in ["face_test.txt"]:
                out_file.writelines(test_options_true[0:num_needed])
            elif category in ['val.txt', 'train.txt'] or category.split("_")[1] in \
                    ["trainval.txt", "val.txt", "train.txt"]:
                out_file.writelines(trainval_options_false[0:num_needed])
            elif category.split("_")[1] in ["test.txt"]:
                out_file.writelines(test_options_false[0:num_needed])

    # Copy utilized Facescrub files into JPEGImages and Annotations folders for VOC datasets
    facescrub_jpeg_folder = os.path.join(facescrub_folder,"JPEGImages")
    facescrub_annotations_folder = os.path.join(facescrub_folder,"Annotations")
    dataset_jpeg_folder = os.path.join(VOC_folder, dataset, "JPEGImages")
    dataset_annotations_folder = os.path.join(VOC_folder, dataset, "Annotations")
    files_to_copy = trainval_options + test_options
    print("Copying {0} facescrub files".format(len(files_to_copy)))
    for image_name in files_to_copy:
        shutil.copy2(os.path.join(facescrub_jpeg_folder,image_name),os.path.join(dataset_jpeg_folder,image_name))
        xml_file = os.path.splitext(image_name)[0] + ".xml"

        # Update folder name in pre-constructed XML when using file in VOC2012 rather than VOC2007
        if dataset == "VOC2012":
            with open(os.path.join(facescrub_annotations_folder,xml_file),"r") as read_xml:
                fix_xml = read_xml.read().replace("VOC2007","VOC2012")
            with open(os.path.join(facescrub_annotations_folder,xml_file),"w") as write_xml:
                write_xml.write(fix_xml)

        shutil.copy2(os.path.join(facescrub_annotations_folder,xml_file),os.path.join(dataset_annotations_folder,xml_file))

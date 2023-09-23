import os
import shutil
import xml.etree.ElementTree as ET
import glob

classes = ["without_mask", "with_mask", "mask_weared_incorrect"]

annotation_dir = "/data_face_mask/annotations"
label_yolo_dir = "/data_face_mask/label_yolo"
images_dir = "/data_face_mask/images"


def xml_to_yolo_bbox(bbox, w, h):
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h

    width = (x2 - x1) / w
    height = (y2 - y1) / h

    return [x_center, y_center, width, height]


def yolo_to_xml_bbox(bbox, w, h):
    x_center, y_center, width, height = bbox
    w_half_len = (width * w) / 2
    h_half_len = (height * h) / 2

    xmin = int((x_center * w) - w_half_len)
    ymin = int((y_center * h) - h_half_len)
    xmax = int((x_center * w) + w_half_len)
    ymax = int((y_center * h) + h_half_len)

    return [xmin, ymin, xmax, ymax]


def process_raw_data():
    if not os.path.isdir(label_yolo_dir):
        os.mkdir(label_yolo_dir)

    files = glob.glob(os.path.join(annotation_dir, "*.xml"))
    for file in files:
        base_name = os.path.basename(file)
        file_name = os.path.splitext(base_name)[0]
        if not os.path.exists(os.path.join(images_dir, f'{file_name}.png')):
            print(f"{file_name} images does not exist !!!")
            continue

        result = []

        tree = ET.parse(file)
        root = tree.getroot()
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)
        objects = root.findall("object")

        for object in objects:
            label = object.find("name").text

            if label not in classes:
                classes.append(label)

            index = classes.index(label)
            pil_bbox = [int(x.text) for x in object.find("bndbox")]
            yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)

            # convert data to string
            bbox_str = " ".join([str(x) for x in yolo_bbox])
            result.append(f"{index} {bbox_str}")

        if result:
            with open(os.path.join(label_yolo_dir, f"{file_name}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(result))


def prepare_data(main_txt_file, main_img_file, train_size, test_size, val_size):
    metarial = []
    for i in os.listdir("/data_face_mask/images/"):
        srt = i[:-4]
        metarial.append(srt)

    for i in range(0, train_size):
        source_txt = main_txt_file + "/" + metarial[i] + ".txt"
        source_img = main_img_file + "/" + metarial[i] + ".png"

        mstring = metarial[i]
        train_destination_txt = "/home/vudangitwork/Documents/FTECH_Reposity/face_mask/data/train/labels" + "/" + metarial[i] + ".txt"
        train_destination_png = "/home/vudangitwork/Documents/FTECH_Reposity/face_mask/data/train/images" + "/" + metarial[i] + ".png"

        shutil.copy(source_txt, train_destination_txt)
        shutil.copy(source_img, train_destination_png)

    for l in range(train_size, train_size + test_size):
        source_txt = main_txt_file + "/" + metarial[l] + ".txt"
        source_img = main_img_file + "/" + metarial[l] + ".png"

        mstring = metarial[l]
        test_destination_txt = "/home/vudangitwork/Documents/FTECH_Reposity/face_mask/data/test/labels" + "/" + metarial[l] + ".txt"
        test_destination_png = "/home/vudangitwork/Documents/FTECH_Reposity/face_mask/data/test/images" + "/" + metarial[l] + ".png"

        shutil.copy(source_txt, test_destination_txt)
        shutil.copy(source_img, test_destination_png)

    for n in range(train_size + test_size, train_size + test_size + val_size):
        source_txt = main_txt_file + "/" + metarial[n] + ".txt"
        source_img = main_img_file + "/" + metarial[n] + ".png"

        mstring = metarial[n]
        val_destination_txt = "/home/vudangitwork/Documents/FTECH_Reposity/face_mask/data/val/labels" + "/" + metarial[n] + ".txt"
        val_destination_png = "/home/vudangitwork/Documents/FTECH_Reposity/face_mask/data/val/images" + "/" + metarial[n] + ".png"

        shutil.copy(source_txt, val_destination_txt)
        shutil.copy(source_img, val_destination_png)


if __name__ == '__main__':
    prepare_data(
        main_txt_file="/data_face_mask/label_yolo",
        main_img_file="/data_face_mask/images",
        train_size=603,
        val_size=150,
        test_size=100,
    )

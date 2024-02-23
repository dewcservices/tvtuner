#!/bin/bash

set -e

echo "In order to use the Kaggle's public API, you must first authenticate using
an API token. Go to the 'Account' tab of your user profile and select 'Create
New Token'. This will trigger the download of kaggle.json, a file containing
your API credentials. Move this file to ~/.kaggle/kaggle.json on Linux, OSX, and
other UNIX-based operating systems, and at
C:\Users\<Windows-username>\.kaggle\kaggle.json on Windows.  If the token is not
there, an error will be raised. Hence, once you've downloaded the token, you
should move it from your Downloads folder to this folder."

read -n 1 -r -s -p "Once you have done this, press any key to continue..." key
echo

chmod 600 ~/.kaggle/kaggle.json

cd yolo

mkdir -p data/tomatoes
cd data/tomatoes
kaggle datasets download nexuswho/tomatod
unzip -q tomatod.zip -x sample_dataset.yaml
rm tomatod.zip
mkdir train val
mv images/train train/images
mv images/val val/images
mv labels/train train/labels
mv labels/val val/labels
rm -rf images labels annotations

cd ../..
mkdir -p data/fixed_wing
cd data/fixed_wing
kaggle datasets download nyahmet/fixed-wing-uav-dataset
unzip -q fixed-wing-uav-dataset.zip
rm fixed-wing-uav-dataset.zip
mkdir labels
mv images/*.txt labels/

cd ../..
mkdir -p data/quadcopter
cd data/quadcopter
kaggle datasets download sshikamaru/drone-yolo-detection
unzip -q drone-yolo-detection.zip -x yolov2-tiny-voc.cfg
rm drone-yolo-detection.zip
mkdir images labels
mv Database1/Database1/*.JPEG images
mv Database1/Database1/*.txt labels
rm images/video18_1999.JPEG images/video15_487.JPEG  # found using: ls images | cut -d "." -f 1 > img_ref; ls labels | cut -d "." -f 1 > labels_ref; diff img_ref labels_ref
cd images
find . -name '*.JPEG' -size 0 | cut -d "." -f 2 | cut -d "/" -f 2 > to_delete  # remove empty images
cd ..
cd labels
find . -name '*.txt' -size 0 | cut -d "." -f 2 | cut -d "/" -f 2 > to_delete  # remove empty labels
cd ..
cat images/to_delete | awk '{print "rm labels/"$0".txt"}' > to_delete.sh
cat images/to_delete | awk '{print "rm images/"$0".JPEG"}' >> to_delete.sh
cat labels/to_delete | awk '{print "rm labels/"$0".txt"}' >> to_delete.sh
cat labels/to_delete | awk '{print "rm images/"$0".JPEG"}' >> to_delete.sh
. to_delete.sh
rm to_delete.sh images/to_delete labels/to_delete
rm -rf Database1

# DTD
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -zxvf dtd-r1.0.1.tar.gz

# Food101
wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -zxvf food-101.tar.gz

# MIT67
mkdir -p ~/mit67
wget -P ./mit67/ http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar
wget -P ./mit67/ https://web.mit.edu/torralba/www/TrainImages.txt
wget -P ./mit67/ https://web.mit.edu/torralba/www/TestImages.txt
tar -xvf ./mit67/indoorCVPR_09.tar -C ./mit67/

# SUN397
wget http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
tar -zxvf SUN397.tar.gz
wget -P ./SUN397 https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip
unzip ./SUN397/Partitions.zip -d ./SUN397

# CUB200
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar -xvzf CUB_200_2011.tgz

# Dogs
mkdir -p ~/StanfordDogs
wget -P ./StanfordDogs/ http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
wget -P ./StanfordDogs/ http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar
wget -P ./StanfordDogs/ http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar
wget -P ./StanfordDogs/ http://vision.stanford.edu/aditya86/ImageNetDogs/train_data.mat
wget -P ./StanfordDogs/ http://vision.stanford.edu/aditya86/ImageNetDogs/test_data.mat
tar -xvf ./StanfordDogs/images.tar -C ./StanfordDogs/
tar -xvf ./StanfordDogs/annotation.tar -C ./StanfordDogs/
tar -xvf ./StanfordDogs/lists.tar -C ./StanfordDogs/

# Flowers
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
tar -xvzf 102flowers.tgz

# Pets
mkdir -p ~/pets
wget -P ./pets/ https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
wget -P ./pets/ https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz
tar -zxvf ./pets/images.tar.gz -C ./pets/
tar -zxvf ./pets/annotations.tar.gz -C ./pets/

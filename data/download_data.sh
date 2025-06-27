OPTIONS="-q --show-progress"
DATASET=https://darus.uni-stuttgart.de/api/access/datafile/:persistentId\?persistentId\=doi:10.18419/DARUS-5120
wget $OPTIONS $DATASET/1 -O data/feature_engineering_thermal_2D.h5
wget $OPTIONS $DATASET/2 -O data/feature_engineering_thermal_3D.h5


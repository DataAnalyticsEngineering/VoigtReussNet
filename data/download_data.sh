OPTIONS="-q --show-progress"
DARUS=https://darus.uni-stuttgart.de/api/access/datafile
wget $OPTIONS $DARUS/386680 -O feature_engineering_thermal_2D.h5
wget $OPTIONS $DARUS/387095 -O feature_engineering_thermal_3D.h5

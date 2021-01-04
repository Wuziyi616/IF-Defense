ROOT=..

export MESHFUSION_PATH=$ROOT/external/mesh-fusion
export HDF5_USE_FILE_LOCKING=FALSE # Workaround for NFS mounts

INPUT_PATH=$ROOT/data/external/MN40
CHOY2016_PATH=$ROOT/data/external/Choy2016
BUILD_PATH=$ROOT/data/MN40.build
OUTPUT_PATH=$ROOT/data/MN40

NPROC=12
TIMEOUT=180
N_VAL=2
N_TEST=2
N_AUG=50

declare -a CLASSES=(
tv_stand
bottle
cone
lamp
airplane
sofa
vase
range_hood
night_stand
toilet
bed
laptop
table
cup
person
chair
radio
keyboard
desk
bathtub
door
bookshelf
piano
curtain
tent
flower_pot
stool
dresser
xbox
guitar
stairs
monitor
wardrobe
bench
mantel
sink
car
bowl
plant
glass_box
)

# Utility functions
lsfilter() {
 folder=$1
 other_folder=$2
 ext=$3

 for f in $folder/*; do
   filename=$(basename $f)
   if [ ! -f $other_folder/$filename$ext ] && [ ! -d $other_folder/$filename$ext ]; then
    echo $filename
   fi
 done
}

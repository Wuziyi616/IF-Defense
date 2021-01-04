source dataset_mn40/config.sh

# Function for processing a single model
reorganize_choy2016() {
  modelname=$(basename -- $5)
  output_path="$4/$modelname"
  build_path=$3
  choy_vox_path=$2
  choy_img_path=$1

  points_file="$build_path/4_points/$modelname.npz"
  points_out_file="$output_path/points.npz"

  pointcloud_file="$build_path/4_pointcloud/$modelname.npz"
  pointcloud_out_file="$output_path/pointcloud.npz"

  # I don't need Vox or Img in IF-Defense, so I just comment them
  # vox_file="$choy_vox_path/$modelname/model.binvox"
  # vox_out_file="$output_path/model.binvox"

  # img_dir="$choy_img_path/$modelname/rendering"
  # img_out_dir="$output_path/img_choy2016"

  # metadata_file="$choy_img_path/$modelname/rendering/rendering_metadata.txt"
  # camera_out_file="$output_path/img_choy2016/cameras.npz"

  # echo "Copying model $output_path"
  mkdir -p $output_path   # $img_out_dir

  cp $points_file $points_out_file
  cp $pointcloud_file $pointcloud_out_file
  # cp $vox_file $vox_out_file

  # python dataset_mn40/get_r2n2_cameras.py $metadata_file $camera_out_file
  # counter=0
  # for f in $img_dir/*.png; do
    # outname="$(printf '%03d.jpg' $counter)"
    # echo $f
    # echo "$img_out_dir/$outname"
    # convert "$f" -background white -alpha remove "$img_out_dir/$outname"
    # counter=$(($counter+1))
  # done
}

copy_files() {
  modelname=$1
  output_path="$2/$modelname"
  build_path=$3

  mkdir -p $output_path

  points_file="${build_path}/4_points/${modelname}.npz"
  points_out_file="${output_path}/points.npz"

  pointcloud_file="${build_path}/4_pointcloud/${modelname}.npz"
  pointcloud_out_file="${output_path}/pointcloud.npz"

  mesh_file="${build_path}/4_watertight_scaled/${modelname}.off"
  mesh_out_file="${output_path}/mesh.off"

  cp $points_file $points_out_file
  cp $pointcloud_file $pointcloud_out_file
  cp $mesh_file $mesh_out_file
}

# export -f reorganize_choy2016
# export -f copy_files

# Make output directories
mkdir -p $OUTPUT_PATH

# Run build
for c in ${CLASSES[@]}; do
  echo "Parsing class $c"
  BUILD_PATH_C=$BUILD_PATH/$c
  OUTPUT_PATH_C=$OUTPUT_PATH/$c
  # CHOY2016_IMG_PATH_C="$CHOY2016_PATH/MN40Rendering/$c"
  # CHOY2016_VOX_PATH_C="$CHOY2016_PATH/MN40Vox32/$c"
  mkdir -p $OUTPUT_PATH_C

  # ls $CHOY2016_VOX_PATH_C | parallel -P $NPROC --timeout $TIMEOUT \
    # reorganize_choy2016 $CHOY2016_IMG_PATH_C $CHOY2016_VOX_PATH_C \
      # $BUILD_PATH_C $OUTPUT_PATH_C {}

  echo "Moving points, point clouds and meshes to output dir"
  for one_file in $(ls "${BUILD_PATH_C}/4_points")
  do
    filename=${one_file%.*}
    echo $filename
    copy_files $filename $OUTPUT_PATH_C $BUILD_PATH_C
  done

  echo "Creating split"
  python create_split.py $OUTPUT_PATH_C --r_val 0.1 --r_test 0.0
done

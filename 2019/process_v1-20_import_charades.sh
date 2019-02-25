 #!/bin/bash
 # chmod u+x scriptname
 # remove files where == 0 size
 # find . -size 0 -exec rm {} \;

source helper_scripts.sh

DEBUG=false
[ "$DEBUG" == 'true' ] && set -x

path_to_vector_processor="/home/barry/PycharmProjects/projectx/optical-flow-filter/demos/flowWebCam/build/flowWebCam"
temporary_files_path="/home/barry/processed/charades/"
LOG_FILE=process
files_inbound=process_files_inbound.txt
files_inbound_for_vectors=files_inbound_for_vectors.txt
quality_to_process=7
relevance_to_process=7
was_verified='Yes'
resolution_reduction_percent=100
fourcc_code='mp4v'
vector_output_type=$4

# Use
#barry@LenovoY70:~/PycharmProjects/projectx/2019$ ./process_v1-20_import_charades.sh /media/barry/Seagate\ Portable\ Drive/Magick/CharadesEgo/CharadesEgo/CharadesEgo_v1_train.csv /media/barry/Seagate\ Portable\ Drive/Magick/CharadesEgo_v1_480/CharadesEgo_v1_480/ /media/barry/Seagate\ Portable\ Drive/Magick/CharadesEgo_v1_480/CharadesEgo_v1_480/

# ./process_v1-20_import_charades.sh
#<path to original charades text file> $1
#<path to input files> $2
#<path to output files> $3

# Python call: 
# python <path to original charades text file> <file to create> <quality> <relevance> <verified>
# python process_v1-20_import_charades.py /media/barry/Seagate\ Portable\ Drive/Magick/CharadesEgo/CharadesEgo/CharadesEgo_v1_train.csv charades_to_process.txt 7 7 Yes


# GOAL: from the Charades dataset
# detect specific quailty, having specific relevance, and possibly verified videos 
# convert the selected videos to b/w and to a specific resolution
# convert the b/w to color where motion is represented by the color as direction of change 

createDirectory "$3"

# BEGIN GRAY CONVERSION PROCESS
echo "Start: Processing inbound video files..." | write_log

create_file_list "$1" "$files_inbound" $quality_to_process $relevance_to_process $was_verified

# Gray creation
echo "Start: Convert .mp4 files Gray" | write_log
while read -r line; do
    echo "Reading name from file - $line"  | write_log
    destination_gray="$2$line"_gray.mp4

    if [ ! -f "$destination_gray" ]; then
      original_file="$3$line".mp4

      echo "Creating Gray - $destination_gray from Original - $original_file"  | write_log
      set -x
      ffmpeg -nostdin -i "$original_file" -vf hue=s=0 "$destination_gray"
      set +x
    else
     echo "Destination Gray file exists - $destination_gray" | write_log
    fi
    # Either way, remove the line we processed from the files_inbound file
    sed -i '1d' $files_inbound
done < "$files_inbound"
echo "End  : Convert .mp4 files Gray" | write_log

remove_file "$files_inbound"

# BEGIN_VECTOR_PROCESS
# ./process_v1-20_import_charades.sh
# <path to original charades text file> $1
# <path to input files> $2
# <path to output files> $3

# UL9Z5EGO
# DOJODEGO
# GCQDHEGO
# GTWBTEGO

# Create temp file
> $files_inbound_for_vectors
create_file_list  "$1" "$files_inbound_for_vectors" $quality_to_process $relevance_to_process $was_verified

# delete temp execution file
echo "Deleting contents of temp vector processing execution file." | write_log
> helper_process_vectors.sh

sed -i "s/{vector_output_type}/$vector_output_type/g" helper_process_vectors.sh
sed -i "s/{fourcc_code}/$fourcc_code/g" helper_process_vectors.sh

# Vector generation
echo "Start: Creating vector conversion list" | write_log
while read line; do
    echo "Reading name from file - $line" | write_log
    original_file="$2""$line"_gray.mp4
    temp_destination_vect="$temporary_files_path""$line"_"$fourcc_code"_"$vector_output_type"_vect.mp4 
    if [ ! -e $temp_destination_vect ]
    then
      echo "rm -f $temp_destination_vect"
      echo "$path_to_vector_processor -i \"$original_file\" -o \"$temp_destination_vect\" -f '$vector_output_type' -c '$fourcc_code'" >> helper_process_vectors.sh
    fi 
done < "$files_inbound_for_vectors"
echo "End  : Creating vector conversion list" | write_log

echo "Begin: executing source helper_process_vectors.sh" | write_log
source helper_process_vectors.sh
echo "End  : executing source helper_process_vectors.sh" | write_log

echo "Deleting contents of temp vector files to copy." | write_log

echo "--------------------"
echo "Begin: moving files to destination" | write_log

  temp_destination_vect="$temporary_files_path"
  echo "Move fm: $temp_destination_vect"* | write_log
  
  final_destination_vect="$3"
  echo "Move to: $final_destination_vect" | write_log
  mv -f "$temp_destination_vect"* "$final_destination_vect"

echo "End  : moving files to destination" | write_log

echo "End  : Processing inbound video files." | write_logS`


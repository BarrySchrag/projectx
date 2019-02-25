 #!/bin/bash
source helper_scripts.sh
# goal 1029/02/22:  simplify and find out why I cannot loop and generate vector files!


LOG_FILE=process
files_inbound=process_files_inbound.txt
files_inbound_for_vectors=files_inbound_for_vectors.txt
quality_to_process=7
relevance_to_process=7
was_verified='Yes'
resolution_reduction_percent=100
fourcc_code='mp4v'
vector_output_type='RGBDegrees'

# reqsubstr="Invalid data"
# file="/home/barry/gdrive/MagickPOC/modet/modet_2019-02-20_08-51.mp4"
# echo "RC----1"
# RC=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $file 2>&1)
 
# #RC=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $file) 2>&1 
# echo "RC----2"
# echo $RC
# if [ -z "${RC##*$reqsubstr*}" ]; then
#     # Do something to handle the error.
#     echo "     : ffprobe error in file read for $file" 
#     #echo $file >> files_to_be_removed_from_google_drive.txt
# else
#     echo "     : ffprobe ok for $file" 
#     #echo $file >> $files_inbound
# fi
#./flowWebCam -i '/media/barry/Seagate Portable Drive/Magick/CharadesEgo_v1_480/CharadesEgo_v1_480/AQQJV_gray.mp4' -o '/media/barry/Seagate Portable Drive/Magick/CharadesEgo_v1_480/CharadesEgo_v1_480/AQQJV_vect.mp4' -f RGBDegrees
#++ ./flowWebCam -i '/media/barry/Seagate Portable Drive/Magick/CharadesEgo_v1_480/CharadesEgo_v1_480/GMRH8_gray.mp4' -o '/media/barry/Seagate Portable Drive/Magick/CharadesEgo_v1_480/CharadesEgo_v1_480/GMRH8_vect.mp4' -f RGBDegrees -c mp4v
#++ ./flowWebCam -i '/media/barry/Seagate Portable Drive/Magick/CharadesEgo_v1_480/CharadesEgo_v1_480/GMRH8EGO_gray.mp4' -o '/media/barry/Seagate Portable Drive/Magick/CharadesEgo_v1_480/CharadesEgo_v1_480/GMRH8EGO_vect.mp4' -f RGBDegrees -c mp4v
# Bad from script
#++ ./flowWebCam -i '/media/barry/Seagate Portable Drive/Magick/CharadesEgo_v1_480/CharadesEgo_v1_480/GMRH8EGO_gray.mp4' -o '/media/barry/Seagate Portable Drive/Magick/CharadesEgo_v1_480/CharadesEgo_v1_480/GMRH8EGO_mp4v_RGBDegrees_vect.mp4' -f RGBDegrees -c mp4v
# Good from cmdline
#   ./flowWebCam -i '/media/barry/Seagate Portable Drive/Magick/CharadesEgo_v1_480/CharadesEgo_v1_480/GMRH8EGO_gray.mp4' -o '/media/barry/Seagate Portable Drive/Magick/CharadesEgo_v1_480/CharadesEgo_v1_480/GMRH8EGO_mp4v_RGBDegrees_cmdline_vect.mp4' -f RGBDegrees -c mp4v

# ./process_v1-20_import_charades.sh
# <path to original charades text file> $1
# <path to input files> $2
# <path to output files> $3

# UL9Z5EGO
# DOJODEGO  *
# GCQDHEGO
# GTWBTEGO
#/media/barry/Seagate\ Portable\ Drive/Magick/CharadesEgo/CharadesEgo/CharadesEgo_v1_train.csv charades_to_process.txt 7 7 Yes

inbound1="/media/barry/Seagate Portable Drive/Magick/CharadesEgo_v1_480/CharadesEgo_v1_480/GMRH8EGO_gray.mp4"
inbound2="/media/barry/Seagate Portable Drive/Magick/CharadesEgo_v1_480/CharadesEgo_v1_480/AQQJV_gray.mp4"
inbound3="/media/barry/Seagate Portable Drive/Magick/CharadesEgo_v1_480/CharadesEgo_v1_480/DOJODEGO_gray.mp4"

outbound1="~/processed/charades/GMRH8EGO_1vect.mp4"
outbound2="~/processed/charades/AQQJV_2vect.mp4"
outbound3="~/processed/charades/DOJODEGO_3vect.mp4"

# rm -f "$outbound1"
# rm -f "$outbound2"
# rm -f "$outbound3"
#  set -x
# ./flowWebCam -i "$inbound1" -o "$outbound1" -f RGBDegrees -c mp4v
# ./flowWebCam -i "$inbound2" -o "$outbound2" -f RGBDegrees -c mp4v
# ./flowWebCam -i "$inbound3" -o "$outbound3" -f RGBDegrees -c mp4v
# set +x
#./flowWebCam -i ~/Desktop/AQQJV_gray.mp4 -o ~/Desktop/AQQJV_x264_a_vect.mp4 -f RGBDegrees -c x264


echo "Start: Vector conversion" | write_log
while read line; do

    echo "Reading name from file - $line"  | write_log
     
    destination_vect="$3$line"_"$fourcc_code"_"$vector_output_type"_vect.mp4
   
    remove_file "$destination_vect"
    if [ ! -f "$destination_vect" ]; then
    
    original_file="$2""$line"_gray.mp4

    echo "Name Gray   - $original_file" | write_log
    echo "Name Vector - $destination_vect"  | write_log
    
    # what I know:
    # Seems to eat the first character of $line every other call
    # flowWebCam is being killed by the loop every time so 1 frame is being written! 
    # ./flowWebCam -i "$original_file" -o "$destination_vect" -f "$vector_output_type" -c "$fourcc_code"
    # /home/barry/PycharmProjects/projectx/optical-flow-filter/demos/flowWebCam/build/flowWebCam -i "$original_file" -o "$destination_vect" -f "$vector_output_type" -c "$fourcc_code"
    echo "Begin: calling source helper_process_vectors.sh"  | write_log
    # sleep 3 does not eat the first $line character, so proves that the exe is the problem.
    sleep 3 #source helper_process_vectors.sh
    echo "End  : calling source helper_process_vectors.sh"  | write_log
     

    ret=$?
    if [ $ret -eq 0 ]; then
    echo "-----------The program exited normally"
    elif [ $ret -gt 128 ]; then
    echo "---------------The program died of signal $((ret-128)): $(kill -l $ret)"
    else
    echo "-------------The program failed with status $ret"
    fi
     
    while kill -0 $pid 2> /dev/null; do
    echo "Waiting for exit of process id: $pid"
    sleep 1
    done
   
      # Remove the line we processed from the files_inbound file
      sed -i '1d' $files_inbound_for_vectors
    else
      echo "Destination Vector file exists - $destination_vect" | write_log
    fi
done < "$files_inbound_for_vectors"
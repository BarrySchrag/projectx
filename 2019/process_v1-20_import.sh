#!/bin/bash
set -eu                # Always put this in Bourne shell scripts
DEBUG=false
[ "$DEBUG" == 'true' ] && set -x

LOG_FILE=process
files_inbound=process_files_inbound.txt
initial_inbound_folder="MagickPOC/modet"
# USE
# ./process_v1-20_import.sh ~/gdrive ~/processed
# ./process_v1-20_import.sh <inbound google drive folder> <location of processed files>

# Function to write to the Log file
###################################
write_log()
{
  while read text
  do
      LOGTIME=`date "+%Y-%m-%d %H:%M:%S"`
      # If log file is not defined, just echo the output
      if [ "$LOG_FILE" == "" ]; then
    echo $LOGTIME": $text";
      else
        LOG=$LOG_FILE.`date +%Y%m%d`
    touch $LOG
        if [ ! -f $LOG ]; then echo "ERROR!! Cannot create log file $LOG. Exiting."; exit 1; fi
    echo $LOGTIME": $text" | tee -a $LOG;
      fi
  done
}

createDirectory() {
    if [ ! -d $1 ]
        then
        mkdir -p $1
    fi
}
createDirectory $2

echo "Start: Processing inbound video files..." | write_log
if [ -f "$files_inbound" ]; then 
    echo "The file $files_inbound exists, processing" | write_log 
else
    # 0. Import google drive mp4 files  :  exec drive pull MagickPOC
    echo 'Start: Pulling inbound .mp4 files from Google Drive' | write_log
    set -x
    eval $(exec drive pull -ignore-conflict $1/$initial_inbound_folder )
    set +x
    echo 'End  : pulling inbound .mp4 files from Google Drive' | write_log

    echo "Start: Determine what .mp4 files to process in $1" | write_log

    while IFS= read -d $'\0' -r file ; do
            #printf '%s\n' "$file" 
            FILESIZE=$(stat -c%s "$file")
            #echo "Size of $file = $FILESIZE bytes."
            if [[ $FILESIZE -gt 0 ]]; then
              echo $file >> $files_inbound
            else
              echo "     : Zero byte file! $1" | write_log
              echo $file >> files_to_be_removed_from_google_drive.txt
            fi
    done < <(find $1 -iname '*.mp4' -print0 ) 
    echo "End  : Determine what .mp4 files to process" | write_log
fi

echo "Start: Processing files from $files_inbound..." | write_log

unset n
while read -r line; do
  echo '   Processing.. ' "$line" | write_log
  #1. Convert all inbound 1 minute RGB video to Grayscale - 2.6MB/7.6MB = 34% of original size, alternate format=gray 2% larger
  # Path_without_extension='/home/barry/gdrive/Google Photos/2019/20190209_134956_bathroom_pan_right'
  path_without_extension=$(echo "$line" | sed -e 's/\.[^.]*$//')
  name_without_extension=$(echo $(basename "${path_without_extension%.*}") )
  destination_file=$2/"$name_without_extension"_gray.mp4

  # Remove destination file if it was only half processed.
  rm -f $destination_file
  ffmpeg -nostdin -i "$line" -vf hue=s=0 $destination_file
  
  # Remove the line we processed from the files_inbound file
  sed -i '1d' $files_inbound
  
  # Remove the local inbound file
  rm -f $line

  # Remove from google drive to save storage space
  echo $line >> files_to_be_removed_from_google_drive.txt
done < @files_inbound

echo "End  : Processing files from $files_inbound..." | write_log

# cleanup inbound 
echo "Start: Cleaning inbound files log" | write_log
rm $files_inbound
echo "End  : Cleaning inbound files log" | write_log

echo "Start: cleaning google drive" | write_log
while read -r line; do
  echo '   Removing files.. ' "$line" | write_log
  rm $line
done < files_to_be_removed_from_google_drive.txt
echo "End  : cleaning google drive" | write_log

echo "End  : Processing inbound video files." | write_log

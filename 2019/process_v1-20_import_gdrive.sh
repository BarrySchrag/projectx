 #!/bin/bash
DEBUG=false
[ "$DEBUG" == 'true' ] && set -x

LOG_FILE=process
files_inbound=process_files_inbound.txt
files_to_be_removed_from_google_drive=files_to_be_removed_from_google_drive.txt
#initial_inbound_folder="MagickPOC/modet"
# USE
# ./process_v1-20_import.sh ~/gdrive ~/processed
# ./process_v1-20_import.sh <inbound google drive folder> <location of processed files>
# TODO:
# Only process files which are older than 5 minutes (or they will not be ready yet)
#  Check the drive stat <file> copyable flag

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
    #set -x
    drive pull -force -ignore-checksum -no-prompt $1 
    #set +x
    echo 'End  : pulling inbound .mp4 files from Google Drive' | write_log

    echo "Start: Determine what .mp4 files to process in $1" | write_log
    reqsubstr="Invalid data"
    while IFS= read -d $'\0' -r file ; do
            # is the file ready?
            set -x
            RC=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $file) 2>&1 
            echo $RC
            if [ -z "${RC##*$reqsubstr*}" ]; then
               # Do something to handle the error.
              echo "     : ffprobe error in file read for $file" | write_log
              echo $file >> $files_to_be_removed_from_google_drive
            else
              echo "     : ffprobe ok for $file" | write_log
              echo $file >> $files_inbound
            fi
            set +x
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
  echo $line >> $files_to_be_removed_from_google_drive
done < $files_inbound

echo "End  : Processing files from $files_inbound..." | write_log

# cleanup inbound 
echo "Start: Cleaning inbound files log" | write_log
rm -f $files_inbound
echo "End  : Cleaning inbound files log" | write_log

echo "Start: Cleaning google drive mapped to $1" | write_log
unset n
while read -r line; do 
  echo '    Begin removing file: ' "$line" | write_log
  rm -f $line
   echo '    End removing file: ' "$line" | write_log
done < $files_to_be_removed_from_google_drive
echo '    Pushing changes' | write_log
drive push -force -ignore-checksum -no-prompt $1
echo "End  : Cleaning google drive mapped to $1" | write_log

rm -f $files_to_be_removed_from_google_drive
echo "End  : Processing inbound video files." | write_log

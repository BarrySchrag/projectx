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

createDirectory()
{
  [ -d "$1" ] || mkdir "$1"
}

create_file_list()
{
  echo 'Start: python file list creation' | write_log
  #set -x
  # <path to original charades text file> <file to create> <quality> <relevance> <verified>
  python process_v1-20_import_charades.py "$1" "$2" $3 $4 $5
  #set +x
  echo 'End  : python file list creation' | write_log
}

remove_file()
{
  # cleanup inbound 
  echo "Start: Cleaning inbound files log" | write_log
  rm -f "$1"
  echo "End  : Cleaning inbound files log" | write_log
}

escape_name()
{
  $(printf %q "$1")
}
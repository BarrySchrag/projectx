# -*- coding: utf-8 -*-

# creates a file with the paths to the files determined by the paramaters
# files_inbound is the spredsheet provided by AI2 identifying the specifics of the Charades dataset
# python is used here only to extract the relevent info from that file and create a paired down file with the list for later processing
# python process_v1-20_import_charades.py original_list new_list quality_to_process relevance_to_process was_verified
import sys
import csv

print( sys.argv[1] )
original_list = sys.argv[1]

print( sys.argv[2] )
new_list = sys.argv[2]

print( sys.argv[3] )
quality_to_process=int(sys.argv[3])

print( sys.argv[4] )
relevance_to_process=int(sys.argv[4])

print( sys.argv[5] )
was_verified=sys.argv[5]

count = 0
outbound_f = open(new_list, 'a+')
with open(original_list) as f:
    reader = csv.DictReader(f)
    for row in reader:
        quality = int(row['quality'] or 0)
        relevance = int(row['relevance'] or 0)
        verified = row['verified']
        if (quality == quality_to_process and relevance == relevance_to_process and verified == was_verified):
          #print( "quality %d, id %s" % (quality, row['id']))
          outbound_f.write(row['id'] + "\n")
          count+=1
print('Count Processed: %s' % str(count))
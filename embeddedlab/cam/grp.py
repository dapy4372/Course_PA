#! /usr/bin/python2.7
import time, requests, operator, glob, sys, os, csv
import numpy as np

_key = '57f0b5bd23354bb2b6283543b34da840'
_maxNumRetries = 20

if( len(sys.argv) != 4 ):
    print "Usage:"
    print ""
    print "      " + sys.argv[0] + " <image dir> <group list filename> <image faceId map>"
    sys.exit(1)

path_to_watch = sys.argv[1]
group_list_filename = sys.argv[2]
imgPath_faceId_map_filename = sys.argv[3]

def processRequest( url, json, data, headers, params ):

    """
    Helper function to process the request to Project Oxford

    Parameters:
    json: Used when processing images from its URL. See API Documentation
    data: Used when processing image read from disk. See API Documentation
    headers: Used to pass the key information and the data type request
    """

    retries = 0
    result = None

    while True:

        response = requests.request( 'post', url = url, json = json, data = data, headers = headers, params = params )
        print("res = ", response)

        if response.status_code == 429: 

            print( "Message: %s" % ( response.json()['error']['message'] ) )

            if retries <= _maxNumRetries: 
                time.sleep(1) 
                retries += 1
                continue
            else: 
                print( 'Error: failed after retrying!' )
                break

        elif response.status_code == 200 or response.status_code == 201:

            if 'content-length' in response.headers and int(response.headers['content-length']) == 0: 
                result = None 
            elif 'content-type' in response.headers and isinstance(response.headers['content-type'], str): 
                if 'application/json' in response.headers['content-type'].lower(): 
                    result = response.json() if response.content else None 
                elif 'image' in response.headers['content-type'].lower(): 
                    result = response.content
        else:
            print( "Error code: %d" % ( response.status_code ) )
            print( "Message: %s" % ( response.json()['error']['message'] ) )

        break
        
    return result

dect_headers = dict()
dect_headers['Ocp-Apim-Subscription-Key'] = _key
dect_headers['Content-Type'] = 'application/octet-stream'
dect_url = 'https://api.projectoxford.ai/face/v1.0/detect'
dect_params = { 'returnFaceId': 'true' }

grp_headers = dict()
grp_headers['Ocp-Apim-Subscription-Key'] = _key
grp_headers['Content-Type'] = 'application/json'
grp_url = 'https://api.projectoxford.ai/face/v1.0/group'

faceIds = [] 
imgPath_faceId_map = dict()

img_filename_list = glob.glob(path_to_watch + "/*.jpg")

# check for first time
if( os.path.exists( imgPath_faceId_map_filename ) ):
    with open( imgPath_faceId_map_filename, "r" ) as f:
        reader = csv.reader( f, delimiter = " " )
        for row in reader:
            # row[0]: key, faceId
            # row[1]: value, img path
            imgPath_faceId_map[row[0]] = row[1]

    # append existed faceId 
    faceIds += imgPath_faceId_map.keys()

if img_filename_list:
    for img_filename in img_filename_list:
        time.sleep(3)
        with open( img_filename, 'rb' ) as f:
            data = f.read()

        result = processRequest( dect_url, None, data, dect_headers, dect_params )
        if result: # check if result is empty
            imgPath_faceId_map[result[0]['faceId']] = os.path.basename(img_filename)
            faceIds.append(result[0]['faceId'])
        else:
            sys.stderr.write(img_filename + " detection request return error!!\n")

    time.sleep(3)

    if len(faceIds) > 2:
        grp_json = dict()
        grp_json["faceIds"] = faceIds

        grp_result = processRequest( grp_url, grp_json, None, grp_headers, None )

        with open( group_list_filename, "w" ) as f:
            writer = csv.writer( f, delimiter = " " )

            for idx, grp in enumerate( grp_result["groups"] ):
                for faceId in grp:
                    writer.writerow( [idx, imgPath_faceId_map[faceId] ] )
                    #print( "%d %s" % (idx, imgPath_faceId_map[faceId]) )

            for faceId in grp_result["messyGroup"]:
                    writer.writerow( [-1, imgPath_faceId_map[faceId] ] )
                    #f.write( "%d %s\n" % (messy_idx, imgPath_faceId_map[faceId]) )

    with open( imgPath_faceId_map_filename, "wb" ) as f:
        writer = csv.writer( f, delimiter = " " )
        for key, value in imgPath_faceId_map.iteritems():
            writer.writerow( [key, value] )

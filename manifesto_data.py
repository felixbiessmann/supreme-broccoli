import urllib, itertools, json, os
import urllib.request
import pandas as pd

BASEURL = "https://manifesto-project.wzb.eu/tools"
VERSION = "MPDS2018a"
APIKEY  = "9cd9104a725f26bcae04da3eed6bdd40" #AN API KEY STRING FROM https://manifestoproject.wzb.eu/information/documents/api
COUNTRY = "Germany"
DATADIR = "manifesto"

os.makedirs(DATADIR, exist_ok=True)

# manifestoproject codes for left/right orientation
LABEL2RIGHTLEFT = {104: 'right',
 201: 'right',
 203: 'right',
 305: 'right',
 401: 'right',
 402: 'right',
 407: 'right',
 414: 'right',
 505: 'right',
 601: 'right',
 603: 'right',
 605: 'right',
 606: 'right',
 103: 'left',
 105: 'left',
 106: 'left',
 107: 'left',
 403: 'left',
 404: 'left',
 406: 'left',
 412: 'left',
 413: 'left',
 504: 'left',
 506: 'left',
 701: 'left',
 202: 'left'}

# manifestoproject codes (integer divided by 100) for political domain
LABEL2DOMAIN = {1: 'External Relations',
 2: 'Freedom and Democracy',
 3: 'Political System',
 4: 'Economy',
 5: 'Welfare and Quality of Life',
 6: 'Fabric of Society'}



MANIFESTOCODE2LABEL = {
        int(x[3:6]):x[8:-2] for x in \
            open(DATADIR+"/manifestolabels.txt").readlines()
        }

def get_url(url):
    return urllib.request.urlopen(url).read().decode()

def get_latest_version():
    '''
    Get the latest version id of the Corpus
    '''
    versionsUrl = BASEURL+"/api_list_metadata_versions.json?&api_key="+APIKEY
    versions = json.loads(get_url(versionsUrl))
    return versions['versions'][-1]

def get_manifesto_id(text_id,version):
    '''
    Get manifesto id of a text given the text id and a version id
    '''
    textKeyUrl = BASEURL+"/api_metadata?keys[]="+text_id+"&version="+version+"&api_key="+APIKEY
    textMetaData = json.loads(get_url(textKeyUrl))
    return textMetaData['items'][0]['manifesto_id']

def get_core(version = VERSION):
    '''
    Downloads core data set, including information about all parties
    https://manifestoproject.wzb.eu/information/documents/api
    '''
    url = BASEURL + "/api_get_core?key=" + VERSION + "&api_key=" + APIKEY
    return json.loads(get_url(url))

def get_text_keys(country=COUNTRY):
    d = get_core()
    return [p[5:7] for p in d if p[1]==country]

def get_text(text_id):
    '''
    Retrieves the latest version of the manifesto text with corresponding labels
    '''
    # get the latest version of this text
    version = get_latest_version()
    # get the text metadata and manifesto ID
    manifestoId = get_manifesto_id(text_id,version)
    textUrl = BASEURL + "/api_texts_and_annotations.json?keys[]="+manifestoId+"&version="+version+"&api_key="+APIKEY
    textData = json.loads(get_url(textUrl))
    try:
        text = [(t['cmp_code'],t['text']) for t in textData['items'][0]['items']]
        print('Downloaded %d texts for %s'%(len(textData['items'][0]['items']),text_id))
        return text
    except:
        print('Could not get text %s'%text_id)

def get_texts_per_party(country=COUNTRY):
    # get all tuples of party/date corresponding to a manifesto text in this country
    textKeys = get_text_keys(country)
    # get the texts
    texts = {t[1]+"_"+t[0]:get_text(t[1]+"_"+t[0]) for t in textKeys}
    texts = {k: v for k, v in texts.items() if v}
    print("Downloaded %d/%d annotated texts"%(len(texts),len(textKeys)))
    return texts

def get_texts(country=COUNTRY):
    texts = get_texts_per_party(country)
    return [x for x in list(itertools.chain(*texts.values())) if x[0]!='NA' and x[0]!='0']

def get_manifesto_texts(country = "Germany",
        folder=DATADIR,
        min_len=10,
        force_download = False):
    fn = folder + "/manifesto-%s.csv"%country
    if (force_download == False) and os.path.isfile(fn):
        print("Loading %s"%fn)
        df = pd.read_csv(fn)
    else:
        print("Downloading texts from manifestoproject.")
        manifestotexts = get_texts(country)
        df = pd.DataFrame(manifestotexts,columns=['cmp_code','content'])
        df.to_csv(fn,index=False)
    df = df[df.content.apply(lambda x: len(str(x)) > min_len)]
    return df['content'].map(str).tolist(),df['cmp_code'].map(int).tolist()

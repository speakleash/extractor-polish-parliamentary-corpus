import requests
from tqdm import tqdm
import os
from lm_dataformat import Archive
import shutil
import spacy
import json
import glob
import tarfile
from multiprocessing.pool import Pool

URL = 'http://mozart.ipipan.waw.pl/~rtuora/resources/PPC.tgz'

def download_file(url):

    ok = True
    file_name = './'+url.split('/')[-1]
    
    print("Downloading data...")
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(file_name, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        ok = False
    
    return ok, file_name
        
        
def generate_data(file):
    with tarfile.open(file) as archive:
        for member in archive:
            if member.isfile():
                yield json.load(archive.extractfile(member))


def get_word_stats(txt):
    if not txt:
        return 0, 0, 0, 0, 0, 0, 0

    sentences = 0
    words = 0
    verbs = 0
    nouns = 0
    punctuations = 0
    symbols = 0
    stopwords = 0

    doc = nlp(txt)

    sentences = len(list(doc.sents))
    words = len([token.text for token in doc if not token.is_punct])
    nouns = len([token.text for token in doc if (not token.is_stop and not token.is_punct and token.pos_ == "NOUN")])
    verbs = len([token.text for token in doc if (not token.is_stop and not token.is_punct and token.pos_ == "VERB")])
    punctuations = len([token.text for token in doc if (token.is_punct or token.pos_ == "PUNCT")])
    symbols = len([token.text for token in doc if (token.pos_ == "SYM")])
    stopwords = len([token.text for token in doc if token.is_stop])

    return sentences, words, verbs, nouns, punctuations, symbols, stopwords


def process_item(document):

    meta = {}
    txt = document['text']
    l = len(txt.strip())
    if l > 100000:
        nlp.max_length = len(txt) + 100
    sentences, words, verbs, nouns, punctuations, symbols, stopwords = get_word_stats(txt.strip())
    meta = {'publisher' : document['metadata']['publisher'], 'title': document['metadata']['title'], 'length': l, 'sentences': sentences, 'words': words, 'verbs': verbs, 'nouns': nouns, 'punctuations': punctuations, 'symbols': symbols, 'stopwords': stopwords}
    return txt.strip(), meta

def initialize_worker():

    print('Initializing worker...')   

    #Each worker node needs to have its own resources.

    global nlp

    #Disabling some unused model features speeds things up to 20%
    nlp = spacy.load("pl_core_news_md", disable=('ner','lemmatizer','textcat','entity_linker'))
   
#Main

if __name__ == '__main__':


    ar = Archive('./data')

    file_name_zst = './PPC_corpus.jsonl.zst'
    file_name_manifest = './PPC_corpus.manifest'

    #Donwnload file

    ok, source_file = download_file(URL)

    if  not ok:
        raise Exception("Downloading data failed...")


    total_len = 0
    total_docs = 0
    total_sentences = 0
    total_words = 0
    total_verbs = 0
    total_nouns = 0
    total_punctuations = 0
    total_symbols = 0
    total_stopwords = 0

 
    with Pool(initializer=initialize_worker) as pool:
        # issue tasks to the process pool
        print('Processing...')
       
        for txt, meta in pool.imap(process_item, generate_data(source_file)):
        

            total_words += meta['words']
            total_verbs += meta['verbs']
            total_nouns += meta['nouns']
            total_len += meta['length']
            total_docs += 1
            total_sentences += meta['sentences']
            total_punctuations += meta['punctuations']
            total_symbols += meta['symbols']
            total_stopwords += meta['stopwords']
            ar.add_data(txt, meta = meta)
            try:
                print("Added " + meta.get('title'))
            #Some documents has None tile
            except:
                print("Added...")
        # Close the process pool
        
        pool.close()
        # Wait for all tasks to complete
        pool.join()
    ar.commit()


    data_files= glob.glob('./data/*')
    file_size = 0

    #This solves an issue where data_files remained locked after ar commiting, causing error on cleanup
    ar = None

    for f in data_files:
        if f.endswith('.zst'):
            shutil.copy(f, os.path.join(file_name_zst))
            file_size = os.path.getsize(file_name_zst)

        os.remove(f)

    manifest = {"project" : "SpeakLeash", "name": "The Polish Parliamentary Corpus", "description": "The Polish Parliamentary Corpus (PPC) is a large collection of documents from the proceedings of the Polish Parliament, Sejm and Senate, both plenary and committee sittings, interpellations and questions.", "license": "Public Domain", "language": "pl", "file_size" : file_size, "sources": [{"name": "The Polish Parliamentary Corpus", "url": "http://clip.ipipan.waw.pl/PPC", "license": "Public Domain"}], "stats": {"documents": total_docs, "sentences": total_sentences, "words" : total_words, "nouns" : total_nouns, "verbs" : total_verbs, "characters": total_len, "punctuations" : total_punctuations, "symbols" : total_symbols, "stopwords": total_stopwords}}
    json_manifest = json.dumps(manifest, indent = 4) 

    with open(file_name_manifest, 'w') as mf:
        mf.write(json_manifest)


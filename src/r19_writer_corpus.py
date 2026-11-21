"""Analysis log R19, helper for B-39: writer text that the attribution never labelled. Abstracts from the Hugging Face
dataset gfissore/arxiv-abstracts-2021 (any category), 12 pages of 100 rows at offsets drawn with seed 2193, kept if
they have at least 40 tokens; written outside the repository (the texts are not redistributed).
Usage: python -I src/r19_writer_corpus.py OUT_JSON"""
import sys, json, random, re, urllib.request
R = random.Random(2193); out = []
URL = 'https://datasets-server.huggingface.co/rows?dataset=gfissore/arxiv-abstracts-2021&config=default&split=train&offset={}&length=100'
for off in sorted(R.sample(range(0, 1_990_000, 100), 12)):
    d = json.load(urllib.request.urlopen(URL.format(off), timeout=60))
    for r in d['rows']:
        a = ' '.join(str(r['row']['abstract']).split())
        if len(re.findall(r'[a-z0-9]+', a.lower())) >= 40: out.append({'id': r['row']['id'], 'abstract': a})
json.dump(out, open(sys.argv[1], 'w')); print(len(out))

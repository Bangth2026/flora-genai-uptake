"""Analysis log R19, helper: the X subtypes of every FLoRA token (src/r16_flora.py executed up to its F-20 block, as in
src/r17_flora_xsub.py), written with the tokens to a pickle outside the repository for src/r19_flora.py.
Nothing in the subtypes changes. The output contains student text from FLoRA (CC BY 4.0).
Usage: EMB_CACHE=FILE python -I src/r19_flora_sub.py DATA_DIR R8_TOKEN_CLASSES OUT_PKL"""
import sys, os, pickle
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[3]
src = open(os.path.join(HERE, 'r16_flora.py')).read()
cut = src.index("out['F20_x_subtypes'] = o20")
sys.argv = sys.argv[:3]
G = {'__name__': 'r16_defs', '__file__': os.path.join(HERE, 'r16_flora.py')}
os.chdir(os.path.join(HERE, '..'))
exec(compile(src[:cut], 'r16_flora.py', 'exec'), G)
pickle.dump({u: dict(A=list(G['EV'][u]['A']), sub=list(sub), cls=list(G['EV'][u]['cls'])) for u, sub in G['SUB'].items()}, open(OUT, 'wb'))
print('saved', len(G['SUB']))

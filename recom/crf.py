import pycrfsuite
from django.conf import settings

tagger = pycrfsuite.Tagger()
tagger.open(settings.BASE_DIR+'/recom/data/spacing_for_movie2.model')

def generate_templates(begin=-2, end=2, min_range_length=3, max_range_length=5):
    templates = []
    for b in range(begin,end):
        for e in range(b,end+1):
            length = (e-b+1)
            if length < min_range_length or length > max_range_length:
                continue
            if b*e > 0:
                continue
            templates.append((b,e))
    return templates

class CharacterFeatureTransformer:
    def __init__(self, templates):
        self.templates = templates
    def __call__(self, chars, tags=None):
        x = []
        for i in range(len(chars)):
            xi = []
            e_max = len(chars)
            for t in self.templates:
                b = i + t[0]
                e = i + t[1] + 1
                if b < 0 or e > e_max:
                    continue
                xi.append(('X[%d,%d]'%(t[0],t[1]),chars[b:e]))
            x.append(xi)
        return x

def sent_to_chartags(sent, nonspace='0', space='1'):
    chars = sent.replace(' ', '')
    if not chars:
        return '', []
    tags = [nonspace]*(len(chars)-1)+[space]
    idx = 0
    for c in sent:
        if c == ' ':
            tags[idx-1] = space
        else:
            idx += 1
    return chars, tags

def sent_to_xy(sent, feature_transformer):
    chars, tags = sent_to_chartags(sent)
    x = [['%s=%s'%(xij[0],xij[1]) for xij in xi] for xi in feature_transformer(chars,tags)]
    y = [t for t in tags]
    return x,y

templates = generate_templates(-2, 2, 3, 3)
transformer = CharacterFeatureTransformer(templates)

def correct(sent, feature_transformer):
    x,y = sent_to_xy(sent, feature_transformer)
    pred = tagger.tag(x)
    char = sent.replace(' ','')
    corrected = ''.join([c if tag=='0' else c+' ' for c,tag in zip(char,pred)]).strip()
    return corrected

correcter = lambda x: correct(x,transformer)
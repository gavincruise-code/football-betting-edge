import difflib, unicodedata

def norm_team(name):
    if not name: return ''
    s = str(name).lower().strip()
    n = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return n.replace('ifk ','').replace('fc ','').replace('sk ','').replace('ac ','').replace('cd ','').strip()

# ESPN sends 'LASK Linz', dataset has 'Lask Linz'
espn = norm_team('LASK Linz')
db   = norm_team('Lask Linz')
print(f'ESPN: "{espn}"')
print(f'DB:   "{db}"')
ratio = difflib.SequenceMatcher(None, espn, db).ratio()
print(f'Similarity: {ratio:.3f}  (cutoff=0.6 passes: {ratio >= 0.6})')

espn2 = norm_team('SV Josko Ried')
print(f'\nESPN "SV Josko Ried" normalised: "{espn2}" -> not in DB (newly promoted)')

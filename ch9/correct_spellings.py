from difflib import SequenceMatcher

def string_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def correct_wrong_word(cw,gw,cap):
    '''
    Spelling correction logic
    This is a very simple logic that replaces
    words with incorrect spelling with the word that highest
    similarity. Some words are manually corrected as the words
    found to be most similar semantically did not match.
    '''
    correct_word = None
    found_similar_word = False
    sim = string_similarity(gw,cw)
    if sim>0.9:
        if cw != 'stting' and cw != 'sittign' and cw != 'smilling' and \
            cw!='skiies' and cw!='childi' and cw!='sittion' and cw!='peacefuly' and cw!='stainding' and\
            cw != 'staning' and cw!='lating' and cw!='sking' and cw!='trolly' and cw!='umping' and cw!='earing' and \
            cw !='baters' and cw !='talkes' and cw !='trowing' and cw !='convered' and cw !='onsie' and cw !='slying':
            print(gw,' ',cw,' ',sim,' (',cap,')')
            correct_word = gw
            found_similar_word = True
        elif cw == 'stting' or cw == 'sittign' or cw == 'sittion':
            correct_word = 'sitting'
            found_similar_word = True
        elif cw == 'smilling':
            correct_word = 'smiling'
            found_similar_word = True
        elif cw == 'skiies':
            correct_word = 'skis'
            found_similar_word = True
        elif cw == 'childi':
            correct_word = 'child'
            found_similar_word = True
        elif cw == 'peacefuly':
            correct_word = 'peacefully'
            found_similar_word = True
        elif cw == 'stainding' or cw == 'staning':
            correct_word = 'standing'
            found_similar_word = True
        elif cw == 'lating':
            correct_word = 'laying'
            found_similar_word = True
        elif cw == 'sking':
            correct_word = 'skiing'
            found_similar_word = True
        elif cw == 'trolly':
            correct_word = 'trolley'
            found_similar_word = True
        elif cw == 'umping':
            correct_word = 'jumping'
            found_similar_word = True
        elif cw == 'earing':
            correct_word = 'eating'
            found_similar_word = True
        elif cw == 'baters':
            correct_word = 'batters'
            found_similar_word = True
        elif cw == 'talkes':
            correct_word = 'talks'
            found_similar_word = True
        elif cw == 'trowing':
            correct_word = 'throwing'
            found_similar_word = True
        elif cw =='convered':
            correct_word = 'covered'
            found_similar_word = True
        elif cw == 'onsie':
            correct_word = cw
            found_similar_word = True
        elif cw =='slying':
            correct_word = 'flying'
            found_similar_word = True
        else:
            raise NotImplementedError
    else:
        correct_word = cw
        found_similar_word = False
        
    return correct_word, found_similar_word
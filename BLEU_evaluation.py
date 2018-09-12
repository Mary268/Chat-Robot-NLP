import math
import numpy as np

def matched_count(cand_d, ref_ds):
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count


def calculate_refer_length(ref_l, cand_l):
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best

def bleu_pair_ngram(candidate, reference, n):
    #print('--------%d-grams-------'%n)
    # the count that the n-grams matches between reference and candidate
    matched_counts = 0
    # the count that the n-grams in reference
    count = 0
    count_r = 0
    count_c = 0
    #for si in range(len(candidate)):
    # Calculate precision for each sentence
    ref_counts = []
    ref_lengths = []
    # Build dictionary of ngram counts
    #for reference in references:
    ref_sentence = reference
    refer_dict = {}
    words = ref_sentence.strip().split()
    ref_lengths.append(len(words))
    limits = len(words) - n + 1
    # loop through the sentance consider the ngram length
    for i in range(limits):
        ngram = ' '.join(words[i:i+n]).lower()
        #print(ngram)
        if ngram in refer_dict.keys():
            refer_dict[ngram] += 1
        else:
            refer_dict[ngram] = 1
    ref_counts.append(refer_dict)
    count_r += len(words)
    #print('refer_dict')
    #print(refer_dict)
    # candidate
    cand_sentence = candidate
    cand_dict = {}
    words = cand_sentence.strip().split()
    limits = len(words) - n + 1
    for i in range(0, limits):
        ngram = ' '.join(words[i:i + n]).lower()
        #print(ngram)
        if ngram in cand_dict:
            cand_dict[ngram] += 1
        else:
            cand_dict[ngram] = 1
    matched_counts += matched_count(cand_dict, ref_counts)
    #print('cand_dict')
    #print(cand_dict)
    #print('matched_counts')
    #print(matched_counts)
    count += limits
    #words count in reference: define length of reference: 
    #in my case only one reference:ref_lengths[0]/length of first reference
    
    # words count in candidate
    count_c += len(words)
    #print('counts in reference')
    #print(count)
    if matched_counts == 0:
        bleu_n = 0
    else:
        # the percentage matches for n-gram is BLEU score
        bleu_n = float(matched_counts) / count
    # if count_r and count_c differentated too much, give brevity penalty
    #print('count_c: %d'%count_c)
    #print('count_r: %d'%count_r)
    bp = brevity_penalty(count_c, count_r)
    return bleu_n, bp

def brevity_penalty(count_c, count_r):
    #if count_c > count_r, candidate is longer than reference, no penalty in this case
    if count_c > count_r:
        bp = 1
    #if count_c <= count_r, generate brevity penalty: bp >1 in this case
    else:
        bp = math.exp(1-(float(count_r)/count_c))
    return bp

#by using the product of their values:nth root of the product of n numbers
def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

# rerurn average BLEU score and its 1-4 gram specific BLEU score
def BLEU_EVA(reference, candidate):
    bleu_n_s = []
    for i in range(4):
        bleu_n, bp = bleu_pair_ngram(candidate, reference, i+1)
        bleu_n_s.append(bleu_n)
    bleu = geo_mean(bleu_n_s) * bp
    return bleu, bleu_n_s


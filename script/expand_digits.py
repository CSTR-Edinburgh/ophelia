#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: SCRIPT - December 2018 
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk


import sys
import os
import glob
import re
import regex
# import fileinput
from argparse import ArgumentParser

# from lxml import etree

## Check required executables are available:
# 
# from distutils.spawn import find_executable
# 
# required_executables = ['sox', 'ch_wave']
# 
# for executable in required_executables:
#     if not find_executable(executable):
#         sys.exit('%s command line tool must be on system path '%(executable))
    




# def main_work():

    #################################################
      
    # ======== Get stuff from command line ==========

    # a = ArgumentParser()
    # a.add_argument('-o', dest='outdir', required=True, \
    #                 help= "Put output here: make it if it doesn't exist")
    # a.add_argument('-c', dest='clear', action='store_true', \
    #                 help= "clear any previous training data first")
    # a.add_argument('-p', dest='max_cores', required=False, type=int, help="maximum number of CPU cores to use in parallel")
    # opts = a.parse_args()
    
    # # ===============================================
    
    # for direc in [opts.outdir]:
    #     if not os.path.isdir(direc):
    #         os.makedirs(direc)

    # parser = etree.HTMLParser()
    # f = open(infile, 'r')
    # tree   = etree.parse(f, parser)

    # r = etree.tostring(tree.getroot(),
    #                 pretty_print=True, method="html")









# ten_map = {
#                 '1': 'kumi',
#                 '2': 'ishirini',
#                 '3': 'thelathini',
#                 '4': 'arobaini',
#                 '5': 'hamsini',
#                 '6': 'sitini',
#                 '7': 'sabini',
#                 '8': 'themanini',
#                 '9': 'tisini'
#     }
    
# one_map = {
#                 '0': 'sifuri',  ## not used in full expansions
#                 '1': 'moja',
#                 '2': 'pili',
#                 '3': 'tatu',
#                 '4': 'nne',
#                 '5': 'tano',
#                 '6': 'sita',
#                 '7': 'saba',
#                 '8': 'nane',
#                 '9': 'tisa'
# }



if 1:
    ten_map = {
                    '0': '',    
                    '1': 'goma',
                    '2': 'ashirin',
                    '3': 'talatin',
                    '4': "arba'in",
                    '5': 'hamsin',
                    '6': 'sittin',
                    '7': "saba'in",
                    '8': 'tamanin',
                    '9': "casa'in"
        }
        
    one_map = {
                    '0': '',  ## not used in full expansions
                    '1': 'daya',
                    '2': 'biyu',
                    '3': 'uku',
                    '4': 'hudu',
                    '5': 'biyar',
                    '6': 'shida',
                    '7': 'bakwai',
                    '8': 'takwas',
                    '9': 'tara'
    }

    hundred = 'dari'
    thousand = 'dubu'
    hundred_map = dict([(digit, '%s %s '%(hundred, one_map[digit])) for digit in one_map])
    #print hundred_map
    thousand_map = dict([(digit, '%s %s '%(thousand, one_map[digit])) for digit in one_map])
    hundred_map['0'] = ''
    hundred_map['1'] = 'dari '    
    thousand_map['0'] = ''

    zero = 'sifiri'

    teenmap = {
        '11':  'goma sha daya',
        '12':  'goma sha biyu',
        '13':  'goma sha uku',
        '14':  'goma sha hudu',
        '15':  'goma sha biyar',
        '16':  'goma sha shida',
        '17':  'goma sha bakwai',
        '18':  'goma sha takwas',
        '19':  'goma sha tara'
    }

    andword = 'da '
if 0:


    ten_map = {
                    '0': '',
                    '1': 'ten',
                    '2': 'twenty',
                    '3': 'thirty',
                    '4': 'forty',
                    '5': 'fifty',
                    '6': 'sixty',
                    '7': 'seventy',
                    '8': 'eighty',
                    '9': 'ninty'
        }
        
    one_map = {
                    #'0': 'zero', 
                    '0': '',
                    '1': 'one',
                    '2': 'two',
                    '3': 'three',
                    '4': 'four',
                    '5': 'five',
                    '6': 'six',
                    '7': 'seven',
                    '8': 'eight',
                    '9': 'nine'
    }

    hundred = 'hundred'
    thousand = 'thousand'

    hundred_map = dict([(digit, '%s %s '%(one_map[digit], hundred)) for digit in one_map])
    # print hundred_map
    thousand_map = dict([(digit, '%s %s '%(one_map[digit], thousand)) for digit in one_map])
    hundred_map['0'] = ''
    thousand_map['0'] = ''

    andword = 'and '


    teenmap = {'11': 'eleven', \
               '12': 'twelve', \
               '13': 'thirteen', \
               '14': 'fourteen', \
               '15': 'fifteen', \
               '16': 'sixteen', \
               '17': 'seventeen', \
               '18': 'eighteen', \
               '19': 'nineteen'}

def expand_digit_sequence(number):
    maxplaces = 4
    number = str(int(number))
    #assert int(number) < 200
    assert len(number) <= maxplaces

    number = number.zfill(maxplaces)
    thousands, hundreds, tens, ones = list(number)

    and_pattern = ['','','and ','-'] ## add an 'and' before this place?
    #and_thousands, and_hundreds, and_tens, and_ones = and_pattern


    output = []
    output.append(thousand_map[thousands] )
    #output += andword
    output.append(hundred_map[hundreds] )
    #output += andword
    if number[-2:] in teenmap:
        output.append(teenmap[number[-2:]] )
        output.append('')
    else:
        output.append(ten_map[tens] )
        output.append(one_map[ones] )

    # print output


    outstring = ''
    for (i, (chunk, add_and)) in enumerate(zip(output, and_pattern)):
        remaining = ''.join(output[i:])
        if add_and:
            if remaining:                
                if outstring:
                    outstring += add_and
        outstring += chunk
    print outstring

    # if hundreds == '1':
    #     text_num.append('mia moja')
    # if tens != '0':
    #     text_num.append(ten_map[tens])
    # if ones != '0':
    #     text_num.append(one_map[ones])

    # return ' na '.join(text_num)


def expand_hausa_digit_sequence(number):
    maxplaces = 4
    number = str(int(number))
    #assert int(number) < 200
    assert len(number) <= maxplaces

    number = number.zfill(maxplaces)
    thousands, hundreds, tens, ones = list(number)

    #and_pattern = ['','','and ','-'] ## add an 'and' before this place?
    #and_thousands, and_hundreds, and_tens, and_ones = and_pattern


    output = []
    output.append(thousand_map[thousands] )
    #output += andword
    output.append(hundred_map[hundreds] )
    #output += andword
    if number[-2:] in teenmap:
        output.append(teenmap[number[-2:]] )
        #output.append('')
    else:
        output.append(ten_map[tens] )
        output.append(one_map[ones] )

    # print output
    output = [t for t in output if t]  ## remove ''

    outstring = andword.join(output)
    if not outstring:
        outstring = zero
    return outstring

    # if hundreds == '1':
    #     text_num.append('mia moja')
    # if tens != '0':
    #     text_num.append(ten_map[tens])
    # if ones != '0':
    #     text_num.append(one_map[ones])

    # return ' na '.join(text_num)
    
    


prereplacements = {
                '.0': ' 0',    ### football scores!
              'G7': 'ji seven',
             'MI5': 'em ai five',
             'MI-5': 'em ai five',
             '@': ' at ',
             'R&F': 'ar and ef',
             ' AS ': ' a es ',     # not 'ei'
             'CSKA': 'si es ka',
             'BOko': 'boko',
             'BBCHausa.com': 'bi bi si hausa dot com',             
             'BBCHausa': 'bi bi si hausa',
             'BATE': 'bet',
             'CESLAC': 'seslac',
             'IPOB': 'ai pob',
             "MPs": "em pis",
             'MURRAY': 'murray'
                    }


exceptions = {'04': 'zero four',
             

                    }



letters = {
    'A': 'ei',
    'B': 'bi',
    'C': 'si',
    'D': 'di',   
    'E': 'i',   
    'F': 'ef',   
    'G': 'ji',   
    'H': 'eich',   
    'I': 'ai',   
    'J': 'jei',   
    'K': 'kei',   
    'L': 'el',   
    'M': 'em',   
    'N': 'en',   
    'O': 'o',   
    'P': 'pi',   
    'Q': 'kyu',   
    'R': 'ar',   
    'S': 'es',   
    'T': 'ti',   
    'U': 'yu',                                
    'V': 'vi',
    'W': 'double yu',
    'X': 'eks',
    'Y': 'wai',
    'Z': 'zed'                
}


acro = ['CAF', 'COSATU', 'ECOWAS', 'FIFA', 'FIFAr', 'MUJAO', 'NASA', 'NATO', 'OPEC', 'UEFA', 'UNICEF', 'WADA']
asletters = ['BBC', 'CBN','DUP', 'EFCC', 'FA', 'FBI', 'FC', 'HIV' , 'IS', 'MI', 'OSCE', 'US', 'WHO', "KRC", "LRA", "MDD", "MDDr", "MI", "MSF", "MTN", "NBS", "NNPC", "NNPCn", "NPA", "PJD", "PKK", "PSG", "SIPG", "TB", "UT", "VX", "WADA", "WFP", "WHO", "YPG"]
        
def txt_to_tokens(text):
    # f = codecs.open(text_file, 'r', encoding='utf8')
    # text = f.read()
    # f.close()
    text = text.strip(' \n\r')

    ## TODO: for hausa: better solution?
    text = text.replace('\n\n', ' ')
    text = text.replace('\n', ' ')

    tokens = regex.split('([\p{P}\p{Z}]+)', text)   # P - punctuation      Z - separators (incl. space)
    tokens = [t for t in tokens if t != '']

    return tokens



def norm_hausa(text):
    ## pre
    for (f,t) in prereplacements.items():
        text = text.replace(f, t)

    tokens = txt_to_tokens(text)
    output = []
    for token in tokens:
        #print [token]
        #print asletters
        if token in exceptions:
            output.append(exceptions[token])
    
        if re.search('\d', token):
            digits = re.sub('[^\d]', '', token)
            expanded = expand_hausa_digit_sequence(digits)
            if token.endswith('n'):
                expanded += 'n'
            output.append(expanded)
        elif token in asletters:
            expanded = ' '.join([letters.get(char, char) for char in token])
            output.append(expanded)
        else:
            output.append(token)

    output = u''.join(output)

    return output


def test():
    for n in [u'0', u'04', u'1', u'10', u'100', u'105', u'11', u'110', u'12', u'120', u'13', u'14', u'15', u'150', u'16', u'17', u'175', u'1780', u'18', u'19', u'194', u'1947', u'1957', u'1991', u'1996', u'1998', u'2', u'20', u'200', u'2000', u'2004', u'2005', u'2008', u'2009', u'2010', u'2011', u'2012', u'2013', u'2014', u'2015', u'2016', u'2018', u'2019', u'2020', u'2025', u'2026', u'2030', u'2050', u'2070', u'21', u'22', u'23', u'230', u'237', u'24', u'25', u'250', u'26', u'260', u'261', u'27', u'28', u'29', u'3', u'30', u'300', u'31', u'318', u'326', u'33', u'34', u'35', u'36', u'37', u'38', u'39', u'4', u'40', u'400', u'43', u'44', u'47', u'48', u'49', u'5', u'50', u'52', u'53', u'57', u'59', u'6', u'60', u'61', u'62', u'66', u'67', u'68', u'7', u'70', u'713', u'72', u'74', u'75', u'8', u'80', u'800', u'814', u'82', u'9', u'90', u'95', u'96', u'98']:
        print '===='
        print n
        expand_hausa_digit_sequence(n)
    # expand_digit_sequence(50)
    # expand_digit_sequence(1074)
    # expand_digit_sequence(1014)
    # expand_digit_sequence(9020)    


#  u'2n'   u'3n',  6n   82n   , u'G7', u'MI5'

if __name__=="__main__":

    #test()
    # main_work()
    print norm_hausa('''Hukumar kare Hakkin BilAdama ta MDD, ta ce ta samu rahotanni masu tushe da ke cewa mayakan IS sun hallaka fararen hula 'yan Iraki fiye da 230 a yayinda suke kokarin tserewa daga Mosul a cikin makonni 2 da suka wuce. Hukumar ta kuma ce tana gudanar da bincike a kan rahotannin da ke cewa an kashe fararen hula 80 a wasu hare-hare ta sama a Mosul a makon jiya.''')












import sys, codecs, re, regex


f = codecs.open(sys.argv[1], encoding='utf8', errors='ignore')
text = f.read()
f.close()

lines = text.split('\n')


all_seps = set()
for line in lines:
    phones = line.split('|')[-1].strip('\n\r ').split(' ')
    seps = set([phone for phone in phones if phone.startswith('<') and phone.endswith('>')])
    all_seps.update(seps)
    #print seps
    #print phones



# print all_seps

badseps = []
for sep in all_seps:

    if regex.match('\A[\p{P}\p{Z}]+\Z', sep.strip('<>')):
        #puncs.append(sep)
        pass
    elif sep in ["<'s>",  '<_END_>', '<_START_>']:
        pass
    else:
        badseps.append(sep)

for sep in badseps:
    text = text.replace(sep, '<>')





#bad_strings = sys.argv[2:]


# for bad in bad_strings:
#     lines = lines.replace('<'+bad+'>', '<>')

print text.encode('utf8')
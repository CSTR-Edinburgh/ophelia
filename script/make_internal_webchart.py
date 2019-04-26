#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: SCRIPT - March 2018 
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk


import sys, os
from string import strip
from argparse import ArgumentParser

def main_work(voice_dirs, names=[], pattern='', outfile='', title=''):

    #for v in voice_dirs:
    #    print v
    #sys.exit('wvswv')
    #voice_dirs = opts.d
    voice_dirs = [string for string in voice_dirs if os.path.isdir(string)]

    if names:
        #names = opts.n
        if not len(names) == len(voice_dirs):
            print '------'
            for name in names:
                print name
            for v in  voice_dirs:
                print v
            sys.exit('len(names) != len(voice_dirs)')
    else:
        names = [direc.strip('/').split('/')[-1] for direc in voice_dirs]
    print names

    # for i in range(0,len(inputs),2):
    #     name = inputs[i]
    #     voice_dir = inputs[i+1]
    #     names.append(name)
    #     voice_dirs.append(voice_dir)


    #################################################

    ### only keep utts appearing in all conditions
    uttnames=[]
    all_utts = []
    for voice_dir in voice_dirs:
        print voice_dir
        print os.listdir(voice_dir)
        print '-----'
        all_utts.extend(os.listdir(voice_dir))

    for unique_utt in set(all_utts):
        if unique_utt.endswith('.wav'):
            if all_utts.count(unique_utt) == len(names):
                uttnames.append(unique_utt)


    if pattern:
        uttnames = [name for name in uttnames if pattern in name]

    # for voice_dir in voice_dirs:
    #     for uttname in os.listdir(voice_dir):
    #             if uttname not in uttnames:
    #                     uttnames.append(uttname)
                    

    if len(uttnames) == 0:
        sys.exit('no utterances found in common!')


    output = ''


    if title:
        output += '<h2>' + title + '</h2>\n'

    ## table top and toprow
    output += '<TABLE BORDER="1" CELLSPACING=2 CELLPADDING=7 WIDTH=1046 height="3">\n'
    output += '<!-- First (header) row -->\n'
    output += "<TR>\n"
    output += '<TD WIDTH="1" VALIGN="TOP" height="1"> <FONT FACE="Verdana" SIZE=2> <B><P ALIGN="CENTER">Condition</B></FONT> </TD>\n'
    for (name,voice_dir) in zip(names, voice_dirs):
            _, voice = os.path.split(voice_dir)
            #output += voice


            output += '<TD WIDTH="1" VALIGN="TOP" height="1"><FONT FACE="Verdana" SIZE=2><B><P ALIGN="CENTER">%s</B></FONT> </TD>\n'%(name)
    output += '</TR>\n'
    
    for uttname in sorted(uttnames):
    
            output += "<TR>\n"        
            
            output += '<TD WIDTH="1" VALIGN="TOP" height="1"><FONT FACE="Verdana" SIZE=2><B><P ALIGN="CENTER">%s</B></FONT></TD>\n'%(uttname.replace(".wav", ""))
            for voice_dir in voice_dirs:

                wavename=os.path.join(voice_dir, uttname)
                output += '<TD WIDTH="1" VALIGN="TOP" height="1">\n'
                output += get_audio_control(wavename)

            output += "</TR>\n"                        
    output += '</table>\n'
    output += '<p>&nbsp;</p>\n'


    if outfile:
        f = open(outfile, 'w')
        f.write(output)
        f.close()
    else:
        print output




def get_audio_control(fname):
    return '''<p><a onclick="this.firstChild.play()"><audio ><source src="%s"/></audio><img width="30" alt="" src="http://pngimg.com/uploads/ear/ear_PNG35710.png" height="30" /></a></p>\n'''%(fname)




if __name__=="__main__": 

    #################################################
      
    # ======== Get stuff from command line ==========

    a = ArgumentParser()
    a.add_argument('-o', dest='outfile', default='', type=str, \
                    help= "If not given, print to console")
    a.add_argument('-d', nargs='+', required=True, help='list of directories with samples')
    a.add_argument('-n', nargs='+', required=False, help='list of names -- use directory names if not given')
    a.add_argument('-p', dest='pattern', default='', type=str)
    a.add_argument('-title', default='', type=str)

    opts = a.parse_args()
    

    # ===============================================

    main_work(opts.d, opts.n, opts.pattern, opts.outfile, opts.title)








#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: SCRIPT - November 2018
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk


import sys, os, re, glob, urllib
import cgitb; cgitb.enable() ## for debugging
import time

this_script = 'http://homepages.inf.ed.ac.uk/cgi/owatts/tts_demo_control_01.cgi'
outdir = '/public/homepages/owatts/web/tts_demo_01/output/'                   # /group/project/script_tts/html/'
outdir_url = 'http://homepages.inf.ed.ac.uk/owatts/tts_demo_01/output/'


def print_html_header():
	print "Content-type: text/html"
	print ""
	print "<html>"
	print "<head><title>CGI Results</title></head>"
	print "<body>"


def print_html_footer():
	print ""
	print "</body>"
	print "</html>"

 
def get_form_data_from_stdin():

    f = sys.stdin
    data=f.read()
    f.close()

    if data=='':
        return {} 
    else:
        data = data.split("&")
        data = [item.split("=") for item in data]
        data = dict(data)
        return data


def unique_file(fname):
	suffix=0
	while os.path.isfile(fname.replace('.txt','_%s.txt'%(suffix))):
		suffix+=1
	return fname.replace('.txt','_%s.txt'%(suffix))

def print_initial_textbox():
    print_textbox_plus_audio('Type your text here')

def print_textbox_plus_audio(text, wavefile=''):
    print "Content-type: text/html"
    print ""
    print '''<html><title>Demo</title>
    <style>
    body {padding: 16px; font-family: sans-serif; font-size: 14px; color: #444}
    input {font-size: 14px; padding: 8px 12px; outline: none; border: 1px solid #ddd}
    input:focus {box-shadow: 0 1px 2px rgba(0,0,0,.15)}
    p {padding: 12px}
    button {background: #28d; padding: 9px 14px; margin-left: 8px; border: none; outline: none;
            color: #fff; font-size: 14px; border-radius: 4px; cursor: pointer;}
    button:hover {box-shadow: 0 1px 2px rgba(0,0,0,.15); opacity: 0.9;}
    button:active {background: #29f;}
    button[disabled] {opacity: 0.4; cursor: default}
    </style>
    <body>
    <form action="%s" method="post">  
      <input id="text" type="text" size="40" placeholder="%s", name="text_to_synth">
      <input id="control_vector_1" type="text" size="8" placeholder="0.0" name="control_vector_1">
      <input id="control_vector_2" type="text" size="8" placeholder="0.0" name="control_vector_2">      
      <button id="button" name="synthesize">Speak</button>
    </form>
    <p id="message"></p>'''%(this_script, text)
    if wavefile:
        print '''<audio id="audio" controls autoplay><source src="%s" type="audio/wav"></audio>
        '''%(wavefile)
    print '</body></html>'

data = get_form_data_from_stdin()
if data=={}:
    print_initial_textbox()
    sys.exit(0)



text = urllib.unquote_plus(data.get('text_to_synth', 'example'))

# password = data.get('password', '')

# password_list = [''] # ['9pw8qf']
# if password not in password_list:
#     os.system('sleep 3')
#     print_initial_textbox()
#     sys.exit(0)    

timestr = time.strftime("%Y%m%d-%H%M%S")

fname = unique_file(os.path.join(outdir, timestr + '.txt'))

f = open(fname, 'w')
f.write(text)
f.close()

## Write control vector to text file:-
cv1 = urllib.unquote_plus(data.get('control_vector_1', '0.0'))
cv2 = urllib.unquote_plus(data.get('control_vector_2', '0.0'))
## default is empty strings - replace these
if not cv1:
    cv1 = '0.0'
if not cv2:
    cv2 = '0.0'

vec_fname = fname.replace('.txt','.vec')
f = open(vec_fname, 'w')
f.write('%s %s'%(cv1, cv2))
f.close()


wavfile = fname.replace('.txt','.wav')

while not os.path.isfile(wavfile):
    os.system('sleep 1')

os.system('chmod 744 %s'%(wavfile))  ## allow it to be read by all

wavurl = wavfile.replace(outdir,outdir_url)


if 0: ## debug
    print_html_header()
    print data
    print
    print wavurl
    print 'BLAHHH!!!!'
    print_html_footer()
    sys.exit(0)


### send only waveform url (for scripted use) or view of textbox?
try:
    return_wave_url = int(data.get('get_wave_url', 0))
except:
    return_wave_url = 0

#return_wave_url = 1

if return_wave_url:
    print "Location: %s" % wavurl
    print
else:
    print_textbox_plus_audio(text, wavurl)
sys.exit(0)




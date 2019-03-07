#load LJSpeech script
#corrupt its transcript
#save back as a new corrupted script

import argparse
import os
import numpy as np

def load_transcript(transcript_path):
    '''load transcript, return a list of lines'''
    with open(transcript_path, 'r') as f:
        lines = f.readlines()
    return_value = []
    for i, l in enumerate(lines): #split each line into two, where the last pipe is
        last_pipe_idx = l.rfind('|')
        if last_pipe_idx != -1: #if the line is empty just ignore this line, will happen if the transcript has trailing newline characters
            phones = l[last_pipe_idx+1:]
            the_rest = l[:last_pipe_idx+1]
            return_value.append([the_rest, phones])
    return return_value

def phones_from_str(phones_str):
    '''Given phone seq in a string, return a list of phones, ignore <_START_> and <_END_> tokens.'''
    assert phones_str is not None or phones_str is not '' #if error here, make sure input transcript file does not have any trailing new line chars
    phones_str = phones_str.strip('<_START_>').strip('<_END_>\n') #NOTE what if we are looking at the last line in the transcript file, is there a newline char?
    phones = phones_str.split(' ')
    phones = [p for p in phones if p is not '']
    return phones

def save_corrupted(data, transcript_path, out_dir, filename):
    '''take the data, and save to disk'''
    transcript_file_name = os.path.basename(transcript_path)
    out_file = os.path.join(out_dir, filename + '_' + transcript_file_name)
    lines = []
    for the_rest, corrupted in data:
        lines.append(the_rest + corrupted) #join the strings together
    lines = ''.join(lines)
    with open(out_file, 'w') as f:
        f.write(lines)
    print('Successfully saved corrupted transcript to {}. Add it to dctts config file.'.format(out_file))

'''
Corruption methods ###############################################################################
'''
    
def swap_halves(data, corruption_percentage):
    '''swap first and latter halves of sentence'''
    num_to_corrupt = int(corruption_percentage*len(data))
    for i, l in enumerate(data[:num_to_corrupt]):
        phone_str = l[1]
        phones = phones_from_str(phone_str)
        swapped_phones = phones[len(phones)/2:] + phones[:len(phones)/2] #swap
        data[i][1] = '<_START_> ' + ' '.join(swapped_phones) + ' <_END_>\n' #recreate string
        

def swap_phones(data):
    '''swap the first and last phone of every word'''
    pass

def swap_words(data):
    '''swap each pair of words'''
    pass

def delete_phones(data, num_to_remove, corruption_percentage):
    '''delete N number of phones from middle of transcript'''
    if num_to_remove is None:
        raise ValueError("For delete phones must choose set the corruption_num command line argument")
    num_to_corrupt = int(corruption_percentage*len(data))
    for i, l in enumerate(data[:num_to_corrupt]):
        phone_str = l[1]
        phones = phones_from_str(phone_str)
        assert len(phones) > num_to_remove
        removed_phones = []
        print('\nutt', i+1, 'from phone seq', list(enumerate(phones)))
        for _ in range(num_to_remove):
            removed_phone = phones.pop(len(phones)/2)
            removed_phones.append((removed_phone, len(phones)/2))
        print('removed', removed_phones)
        data[i][1] = '<_START_> ' + ' '.join(phones) + ' <_END_>\n' #recreate string

'''
Corruption methods ###############################################################################
'''

def main():
    #create parser
    parser = argparse.ArgumentParser(
        description="Load LJ speech transcript (training or test!) and corrupt it given a chosen method.")

    #add arguments
    parser.add_argument("--transcript_path", action="store", dest="transcript_path", type=str, required=True,
                        help="Path to LJSpeech transcript file.")
    parser.add_argument("--out_dir", action="store", dest="out_dir", type=str, default=None,
                        help="Directory to save corrupted transcript to.")
    parser.add_argument("--corruption_method", action="store", dest="corruption_method", type=str, required=True,
                        help="Corruption method chosen.")
    parser.add_argument("--corruption_percentage", action="store", dest="corruption_percentage", type=float, required=True,
                        help="What percent of the training utts to corrupt: 0.25 means we corrupt 25 percent of the sentences and leave 75 percent untouched.") 
    parser.add_argument("--corruption_num", action="store", dest="corruption_num", type=float, default=None,
                        help="Used if the corruption method accepts an extra numerical parameter.") 

    #parse arguments
    args = parser.parse_args()
    if args.out_dir is None: #if we are not given an out_dir, save the new modified script to where we loaded LJSpeech
        args.out_dir = os.path.dirname(args.transcript_path)

    #load data from file
    data = load_transcript(args.transcript_path)

    #call the appropriate corruption method to corrupt data
    if args.corruption_method == 'swap_halves':
        swap_halves(data, args.corruption_percentage)
        save_corrupted(data, args.transcript_path, args.out_dir, filename=args.corruption_method + '_' + str(args.corruption_percentage))
    elif args.corruption_method == 'swap_phones':
        swap_phones(data)
    elif args.corruption_method == 'swap_words':
        swap_words(data)
    elif args.corruption_method == 'delete_phones':
        delete_phones(data, args.corruption_num, args.corruption_percentage)
        save_corrupted(data, args.transcript_path, args.out_dir, filename=args.corruption_method + '_' + str(args.corruption_num) + '_' + str(args.corruption_percentage))
    else:
        raise ValueError("Unsupported --corruption_method, got {}.".format(args.corruption_method))

if __name__ == "__main__":
    main()

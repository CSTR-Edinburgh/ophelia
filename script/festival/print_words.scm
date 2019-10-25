

; Usage: 
;     cd to directory containing ./utts.data, then with a version of Festival with the correct
;     lexicon installed:

;    FEST=/afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/tool/festival/festival/bin/festival
;    SCRIPT=/afs/inf.ed.ac.uk/user/o/owatts/repos/dc_tts/script/make_rich_phones.scm
;    $FEST -b $SCRIPT | grep ___KEEP___ | sed 's/___KEEP___//' | tee ./transcript.csv



;;; Taken from build_unitsel:
;; Is this the last segment in a word [more complicated than you would think]
(define (seg_word_final seg)
"(seg_word_final seg)
Is this segment word final?"
  (let ((this_seg_word (item.parent (item.relation.parent  seg 'SylStructure)))
    (silence (car (cadr (car (PhoneSet.description '(silences))))))
    next_seg_word)
    (if (item.next seg)
    (set! next_seg_word (item.parent (item.relation.parent (item.next seg) 'SylStructure))))
    (if (or (equal? this_seg_word next_seg_word)
         (string-equal (item.feat seg "name") silence))
    nil
    t)))


(define (following_punc seg)
  ( set! this_seg_word (item.parent (item.relation.parent  seg 'SylStructure)))
  ( set! this_seg_token (item.relation.next  this_seg_word 'Token))
  (if this_seg_token
    ( format t "<%s> " (item.name this_seg_token))
    ( format t "<> " )    %% else
  )
)


(define (print_words_pos utt)
(if (utt.relation.present utt 'Segment)
    (begin
      ( format t "<_START_> " )
      (mapcar
       (lambda (x)
            ( if (not (equal? (item.name x) "#") )   ;; TODO: unhardcode silent symbols
              (format t "%s " (item.name x) )
            )
            (if (seg_word_final x) 
                (following_punc x)
            )            
       )
       (utt.relation.items utt 'Segment))
        ( format t "<_END_>" )
      )
    (format t "Utterance contains no Segments\n"))
  nil)



(define (print_words_pos utt)
(if (utt.relation.present utt 'Token)
    (begin
      (mapcar
       (lambda (x)
            (format t "%s " (item.name x) )            
       )
       (utt.relation.items utt 'Token))
        ;( format t "<_END_>" )
      )
    (format t "Utterance contains no Tokens\n"))
  nil)



;;; Taken from build_unitsel:
;; Do the linguistic side of synthesis.
(define (utt.synth_toSegment_text utt)
  (Initialize utt)
  (Text utt)
  (Token_POS utt)    ;; when utt.synth is called
  (Token utt)
  (POS utt)
  (Phrasify utt)
  (Word utt)
  (Pauses utt)
  (Intonation utt)
  (PostLex utt))

(define (synth_utts utts_data_file)
    (set! uttlist (load utts_data_file t))
    (mapcar
     (lambda (line)
        (set! utt (utt.synth_toSegment_text (eval (list 'Utterance 'Text (car (cdr line))))))   ;;;  Initialize
       (format t "___KEEP___%s||" (car line))
        ;(format t "%s|" (car (cdr line)) )     
        (format t "%s|" (car (cdr line)) )          
        (print_words_pos utt)
        ; (utt.relation.print utt 'Text)
        ( format t "\n" )
       t)
     uttlist)
  )


(if (not (member_string 'unilex-rpx (lex.list)))
    (load (path-append lexdir "unilex/" (string-append 'unilex-rpx ".scm"))))

; (if (not (member_string 'cmudict (lex.list)))
;     (load (path-append lexdir "cmu/" (string-append 'cmudict-0.4 ".scm"))))


;(require 'unilex_phones)
;(lex.select 'unilex-rpx)




(if (not (member_string 'combilex-rpx (lex.list)))
    (load (path-append lexdir "combilex/" (string-append 'combilex-rpx ".scm"))))
(lex.select 'combilex-rpx)
; (require 'postlex)
;   (set! postlex_rules_hooks (list postlex_apos_s_check
;                                   postlex_intervoc_r
;                                   postlex_the_vs_thee
;                                   postlex_a
;                                   ))
  ; (set! postlex_rules_hooks (list 
  ;                                 postlex_intervoc_r                                                                    
  ;                                 ))



; (lex.select 'cmudict)



;(set! utt1 (Utterance Text "Hello there, world, isn't it a nice day?!"))
;(utt.synth utt1)
;(print_phones_punc utt1)

(synth_utts "./utts.data")







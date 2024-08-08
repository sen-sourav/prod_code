import os
import copy
import torch
import statistics
import tqdm
from basic_pitch_torch.inference import predict
from ultimate_accompaniment_transformer import TMIDIX
from ultimate_accompaniment_transformer.midi_to_colab_audio import midi_to_colab_audio
from app.helper import save_audio_to_wav, my_linear_mixing
from pathlib import Path


def model_inference(model, input_file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_file_path).split('.')[0]

    # Step 1: Convert vocal wav to midi
    model_output, midi_data, note_events = predict(input_file_path, model_path=Path('../repos/basic-pitch-torch/basic_pitch_torch/assets/basic_pitch_pytorch_icassp_2022.pth').resolve(), minimum_frequency=60)
    vocals_midi_path = os.path.join(output_dir, f"{base_name}.mid")
    midi_data.write(vocals_midi_path)

    # Load midi
    f = vocals_midi_path

    raw_score = TMIDIX.midi2single_track_ms_score(f)
    
    #===============================================================================
    # Enhanced score notes
    
    escore_notes = TMIDIX.advanced_score_processor(raw_score, return_enhanced_score_notes=True)[0]
    
    escore_notes = [e for e in escore_notes if e[3] != 9]
    
    if len(escore_notes) > 0:
    
        #=======================================================
        # PRE-PROCESSING
    
        #===============================================================================
        # Augmented enhanced score notes
    
        escore_notes = TMIDIX.augment_enhanced_score_notes(escore_notes, timings_divider=32)
    
        cscore = TMIDIX.chordify_score([1000, escore_notes])
    
        melody = TMIDIX.fix_monophonic_score_durations([sorted(e, key=lambda x: x[4], reverse=True)[0] for e in cscore])
    
        #=======================================================
        # FINAL PROCESSING
    
        melody_chords = []
    
        #=======================================================
        # MAIN PROCESSING CYCLE
        #=======================================================
    
        pe = cscore[0][0]
    
        mpe = melody[0]
    
        midx = 1
    
        for i, c in enumerate(cscore):
    
            c.sort(key=lambda x: (x[3], x[4]), reverse=True)
    
            # Next melody note
    
            if midx < len(melody):
    
              # Time
              mtime = melody[midx][1]-mpe[1]
    
              mdur = melody[midx][2]
    
              mdelta_time = max(0, min(127, mtime))
    
              # Durations
              mdur = max(0, min(127, mdur))
    
              # Pitch
              mptc = melody[midx][4]
    
            else:
              mtime = 127-mpe[1]
    
              mdur = mpe[2]
    
              mdelta_time = max(0, min(127, mtime))
    
              # Durations
              mdur = max(0, min(127, mdur))
    
              # Pitch
              mptc = mpe[4]
    
    
            e = melody[i]
    
            #=======================================================
            # Timings...
    
            time = e[1]-pe[1]
    
            dur = e[2]
    
            delta_time = max(0, min(127, time))
    
            # Durations
    
            dur = max(0, min(127, dur))
    
            # Pitches
    
            ptc = max(1, min(127, e[4]))
    
            if ptc < 60:
              ptc = 60 + (ptc % 12)
    
            cha = e[3]
    
            #=======================================================
            # FINAL NOTE SEQ
    
            if midx < len(melody):
              melody_chords.append([delta_time, dur+128, ptc+384, mdelta_time+512, mptc+640])
              mpe = melody[midx]
              midx += 1
            else:
              melody_chords.append([delta_time, dur+128, ptc+384, mdelta_time+512, mptc+640])
    
            pe = e
    
    #=======================================================
    
    song = melody_chords
    song_f = []
    
    
    
    time = 0
    dur = 128
    vel = 90
    pitch = 0
    pat = 40
    channel = 3
    
    patches = [0] * 16
    patches[3] = 40
    patches[0] = 0
    
    for ss in song:
    
    
    
      time += ss[0] * 32
    
      dur = (ss[1]-128) * 32
    
      pitch = (ss[2]-256) % 128
    
      vel = max(40, pitch)
    
      song_f.append(['note', time, dur, channel, pitch, vel, pat])
    
    
    
    detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                              output_signature = 'Ultimate Accompaniment Transformer',
                                                              output_file_name = f"{output_dir}/{base_name}_seed_composition",
                                                              track_name='Project Los Angeles',
                                                              list_of_MIDI_patches=patches
                                                              )
    
    
    melody_MIDI_patch_number = 40 # @param {type:"slider", min:0, max:127, step:1}
    accompaniment_MIDI_patch_number = 0 # @param {type:"slider", min:0, max:127, step:1}
    
    number_of_prime_melody_notes = 17 # @param {type:"slider", min:0, max:64, step:1}
    use_harmonized_melody = True # @param {type:"boolean"}
    harmonized_melody_octave = 4 # @param {type:"slider", min:3, max:5, step:1}
    force_accompaniment_generation_for_each_melody_note = False # @param {type:"boolean"}
    tokens_sampling_rate = 8 # @param {type:"slider", min:1, max:16, step:1}
    number_of_memory_tokens = 3264 # @param {type:"slider", min:32, max:8188, step:16}
    temperature = 1 # @param {type:"slider", min:0.1, max:1, step:0.05}
    
    def generate_acc(input_seq,
                      next_note_time,
                      force_acc = False,
                      num_samples = 2,
                      num_batches = 8,
                      num_memory_tokens = 4096,
                      temperature=0.9
                      ):
    
        input_seq = input_seq[-num_memory_tokens:]
    
        if force_acc:
          x = torch.LongTensor([input_seq+[0]] * num_batches).cuda()
        else:
          x = torch.LongTensor([input_seq] * num_batches).cuda()
    
        cur_time = 0
        ctime = 0
    
        o = 0
    
        while cur_time < next_note_time and o < 384:
    
          samples = []
    
          for _ in range(num_samples):
    
            with ctx:
              out = model.generate(x,
                                  1,
                                  temperature=temperature,
                                  return_prime=True,
                                  verbose=False)
    
              with torch.no_grad():
                test_loss, test_acc = model(out)
    
            samples.append([out[:,-1].tolist(), test_acc.tolist()])
    
          accs = [y[1] for y in samples]
          max_acc = max(accs)
          o = statistics.mode(samples[accs.index(max_acc)][0])
    
          if 0 <= o < 128:
            cur_time += o
    
          if cur_time < next_note_time and o < 384:
    
            ctime = cur_time
    
            out = torch.LongTensor([[o]] * num_batches).cuda()
            x = torch.cat((x, out), dim=1)
    
        return list(statistics.mode([tuple(t) for t in x[:, len(input_seq):].tolist()])), ctime
    
    #===============================================================================
    
    torch.cuda.empty_cache()
    
    #===============================================================================
    
    melody_tones_chords = TMIDIX.harmonize_enhanced_melody_score_notes(melody)
    
    harm_melody_chords =  []
    
    for i, m in enumerate(melody_tones_chords):
      cho = []
      mm = sorted(m, reverse=True)
      for mmm in mm:
        cho.extend([0, max(0, min(127,melody[i][2]))+128, (harmonized_melody_octave*12)+mmm+256])
    
      harm_melody_chords.append(cho)
    
    #===============================================================================
    
    output1 = []
    output2 = []
    
    for i in range(number_of_prime_melody_notes):
    
      output1.extend(melody_chords[i])
    
      if use_harmonized_melody:
        output1.extend(harm_melody_chords[i])
        output2.append(harm_melody_chords[i])
      else:
        output2.append([])
    
    hidx = number_of_prime_melody_notes
    cur_time = 0
    
    for m in tqdm.tqdm(melody_chords[number_of_prime_melody_notes:]):
    
      try:
        mel = copy.deepcopy(m)
        mel[0] = mel[0] - cur_time
        next_note_time = mel[3]-512
    
        if use_harmonized_melody:
          input = mel + harm_melody_chords[hidx]
        else:
          input = mel
    
        output1.extend(input)
    
        out, cur_time = generate_acc(output1,
                                    next_note_time,
                                    force_acc=force_accompaniment_generation_for_each_melody_note,
                                    num_samples=tokens_sampling_rate,
                                    num_batches=tokens_sampling_rate,
                                    num_memory_tokens=number_of_memory_tokens,
                                    temperature=temperature
                                    )
        output1.extend(out)
    
        if use_harmonized_melody:
          output2.append(harm_melody_chords[hidx] + out)
        else:
          output2.append(out)
    
        hidx += 1
    
      except KeyboardInterrupt:
        print('Stopping...')
        print('Stopping generation...')
        break
    
      except Exception as e:
        print('Error:', e)
        break
    
    torch.cuda.empty_cache()
    
    out1 = output2
    
    if len(out1) != 0:
    
        song = out1
        song_f = []
    
        time = 0
        ntime = 0
        ndur = 0
        vel = 90
        npitch = 0
        channel = 0
    
        patches = [0] * 16
        patches[0] = accompaniment_MIDI_patch_number
        patches[3] = melody_MIDI_patch_number
    
        for i, ss in enumerate(song):
    
                ntime += melody_chords[i][0] * 32
                ndur = (melody_chords[i][1]-128) * 32
                nchannel = 1
                npitch = (melody_chords[i][2]-256) % 128
                vel = max(40, npitch)+20
    
                time = ntime
    
                for s in ss:
    
                  if 0 <= s < 128:
    
                      time += s * 32
    
                  if 128 <= s < 256:
    
                      dur = (s-128) * 32
    
                  if 256 <= s < 384:
    
                      pitch = (s-256)
    
                      vel = max(40, pitch)
    
                      song_f.append(['note', time, dur, 0, pitch, vel, accompaniment_MIDI_patch_number])
    
    detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                              output_signature = 'Ultimate Accompaniment Transformer',
                                                              output_file_name = f"{output_dir}/{base_name}_composition",
                                                              track_name='Project Los Angeles',
                                                              list_of_MIDI_patches=patches
                                                              )






    # Step 2: Convert piano midi to wav
    midi_audio = midi_to_colab_audio(f"{output_dir}/{base_name}_composition.mid")
    save_audio_to_wav(midi_audio, f"{output_dir}/{base_name}_composition.wav", sample_rate=16000)

    # Step 3: Mix vocal with piano
    piano = f"{output_dir}/{base_name}_composition.wav"
    vocals = input_file_path
    output_file = f"{output_dir}/{base_name}_final_mix.wav"
    my_linear_mixing(piano, vocals, output_file)

    return output_file

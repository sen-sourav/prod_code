import os
import copy
import torch
import statistics
from basic_pitch_torch.inference import predict
from ultimate_accompaniment_transformer import TMIDIX
from ultimate_accompaniment_transformer.midi_to_colab_audio import midi_to_colab_audio
from app.helper import save_audio_to_wav, my_linear_mixing

def generate_acc(model, input_seq, next_note_time, force_acc=False, num_samples=2, num_batches=8, num_memory_tokens=4096, temperature=0.9):
    input_seq = input_seq[-num_memory_tokens:]
    if force_acc:
        x = torch.LongTensor([input_seq + [0]] * num_batches).cuda()
    else:
        x = torch.LongTensor([input_seq] * num_batches).cuda()
    cur_time = 0
    ctime = 0
    o = 0
    while cur_time < next_note_time and o < 384:
        samples = []
        for _ in range(num_samples):
            with torch.cuda.amp.autocast():
                out = model.generate(x, 1, temperature=temperature, return_prime=True, verbose=False)
                with torch.no_grad():
                    test_loss, test_acc = model(out)
            samples.append([out[:, -1].tolist(), test_acc.tolist()])
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

def model_inference(model, input_file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_file_path).split('.')[0]
    
    # Step 1: Convert vocal wav to midi
    model_output, midi_data, note_events = predict(input_file_path, model_path='basic-pitch-torch/basic_pitch_torch/assets/basic_pitch_pytorch_icassp_2022.pth', minimum_frequency=60)
    vocals_midi_path = os.path.join(output_dir, f"{base_name}.mid")
    midi_data.write(vocals_midi_path)

    # Load midi
    f = vocals_midi_path

    # Process midi file to create song
    raw_score = TMIDIX.midi2single_track_ms_score(f)
    escore_notes = TMIDIX.advanced_score_processor(raw_score, return_enhanced_score_notes=True)[0]
    escore_notes = [e for e in escore_notes if e[3] != 9]

    if len(escore_notes) > 0:
        escore_notes = TMIDIX.augment_enhanced_score_notes(escore_notes, timings_divider=32)
        cscore = TMIDIX.chordify_score([1000, escore_notes])
        melody = TMIDIX.fix_monophonic_score_durations([sorted(e, key=lambda x: x[4], reverse=True)[0] for e in cscore])
        melody_chords = []
        pe = cscore[0][0]
        mpe = melody[0]
        midx = 1

        for i, c in enumerate(cscore):
            c.sort(key=lambda x: (x[3], x[4]), reverse=True)
            if midx < len(melody):
                mtime = melody[midx][1]-mpe[1]
                mdur = melody[midx][2]
                mdelta_time = max(0, min(127, mtime))
                mdur = max(0, min(127, mdur))
                mptc = melody[midx][4]
            else:
                mtime = 127-mpe[1]
                mdur = mpe[2]
                mdelta_time = max(0, min(127, mtime))
                mdur = max(0, min(127, mdur))
                mptc = mpe[4]

            e = melody[i]
            time = e[1]-pe[1]
            dur = e[2]
            delta_time = max(0, min(127, time))
            dur = max(0, min(127, dur))
            ptc = max(1, min(127, e[4]))
            if ptc < 60:
                ptc = 60 + (ptc % 12)
            cha = e[3]

            if midx < len(melody):
                melody_chords.append([delta_time, dur+128, ptc+384, mdelta_time+512, mptc+640])
                mpe = melody[midx]
                midx += 1
            else:
                melody_chords.append([delta_time, dur+128, ptc+384, mdelta_time+512, mptc+640])

            pe = e

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
                                                                output_signature='Ultimate Accompaniment Transformer',
                                                                output_file_name=f"{output_dir}/{base_name}_composition.mid",
                                                                track_name='Project Los Angeles',
                                                                list_of_MIDI_patches=patches)

        # Step 2: Convert piano midi to wav
        midi_audio = midi_to_colab_audio(f"{output_dir}/{base_name}_composition.mid")
        save_audio_to_wav(midi_audio, f"{output_dir}/{base_name}_composition.wav", sample_rate=16000)

        # Step 3: Mix vocal with piano
        piano = f"{output_dir}/{base_name}_composition.wav"
        vocals = input_file_path
        output_file = f"{output_dir}/{base_name}_final_mix.wav"
        my_linear_mixing(piano, vocals, output_file)

        return output_file
    else:
        raise ValueError("No enhanced score notes found in MIDI file.")

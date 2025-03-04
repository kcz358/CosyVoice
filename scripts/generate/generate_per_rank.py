
import argparse
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import os
import jsonlines
from tqdm import tqdm

def process_data(data, role_list):
    total_text = []
    total_file_path = []
    new_data = []
    for da in data:
        idx = da['id']
        messages = da['messages']
        new_messages = []
        for turn, message in enumerate(messages):
            role = message['role']
            content = message['content']
            new_content = []
            for cont_idx, cont in enumerate(content):
                file_path = f"{idx}_{turn}_{role}_{cont_idx}.wav"
                if cont["type"] == "audio_url":
                    # Lets only extract the file path
                    cont["audio_url"]["url"] = cont['audio_url']["url"].split("/")[-1]
                    new_content.append(cont)
                if cont["type"] == "text" and role in role_list:
                    total_text.append(cont["text"])
                    total_file_path.append(file_path)
                    new_content.append({
                        "type": "audio_url",
                        "audio_url": {
                            "url": file_path
                        }
                    })
            new_messages.append({
                "role": role,
                "content": new_content
            })
        new_data.append({
            "id": idx,
            "messages": new_messages
        })
    return new_data, total_text, total_file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, help="input file")
    parser.add_argument("--output", "-o", type=str, help="output folder")
    parser.add_argument("--output-file", "-of", type=str, help="output file")
    parser.add_argument("--role", choices=["user", "assistant", "all"], default="assistant")

    args = parser.parse_args()
    input_file = args.input
    output_folder = args.output
    output_file = args.output_file
    role = args.role
    if role == "all":
        role_list = ["user", "assistant"]
    else:
        role_list = [role]

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "mp_data"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "generated_audio"), exist_ok=True)

    with jsonlines.open(input_file) as reader:
        data = list(reader)

    new_data, total_text, total_file_path = process_data(data, role_list)
    with jsonlines.open(os.path.join(output_folder, "mp_data", output_file), "w") as writer:
        writer.write_all(new_data)

    cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

    pbar = tqdm(total=len(total_text), desc="Generating audio")
    # NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
    # zero_shot usage
    prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
    for text, file_path in zip(total_text, total_file_path):
        for i, j in enumerate(cosyvoice.inference_zero_shot(text, '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
            torchaudio.save(os.path.join(output_folder, "generated_audio", file_path), j['tts_speech'], cosyvoice.sample_rate)
        pbar.update(1)



    
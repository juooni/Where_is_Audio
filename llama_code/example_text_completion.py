# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
from llama.generation import Llama
from typing import List

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_gen_len: int = 20,
    max_batch_size: int = 4,
):


    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    prompt_sentence=[]
    with open('./llama/prompttext.txt','r') as file:
        lines=file.readlines()
        for line in lines:
            prompt_sentence.append(line.strip())
    

    prompts: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        """provide appropriate sound effect of following sentences.
        I heard a thumping sound in the deep night living room => thumping sound;
        man eating the bread on the floor.=> crunching sound;
        there must be heavy rain in this area => sound of rain;
        the church suddenly burst into flames. all of whom perished in the fire=> sound of fire;
        they placed the raised dog into the device and activated it. => dog barking sound;
        He spoke loudly and turned on the faucet => screaming sound;
        """
    ]
    effect_prompt=[]
    original_prompt=prompts[0]
    with open('./llama/prompttext_result.txt','w+') as file:
        for sen in prompt_sentence:
            prompts[0]=original_prompt
            prompts[0]=prompts[0]+sen+' => '
            results = generator.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            for prompt, result in zip(prompts, results):
            #print(prompt)
                effect_prompt.append(result['generation'])
                print(f"> {result['generation']}")
                for char in result['generation']:
                    if(char==';'): break
                    file.write(char)
                file.write('\n')
                #file.write(result['generation'])



if __name__ == "__main__":
    fire.Fire(main)

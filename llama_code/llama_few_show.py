# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

from llama.generation import Llama
from typing import List

def get_prompt(inputtext):
    ckpt_dir='llama-2-7b'
    tokenizer_path='tokenizer.model'
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=512,
        max_batch_size=4,
    )
    prompts: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        """provide appropriate sound effect of following sentences.
        I heard a thumping sound in the deep night living room where the moonlight shines bizarily. => thumping sound;
        Then, it's said that a man who closely resembled himself appeared from behind the chair where he was sitting, eating the bread on the floor.=> crunching sound;
        Once or twice a year, there must be heavy rain in this area, whether their prayers are answered or not. => sound of rain;
        Additionally, emitting smoke and a foul odor, the church suddenly burst into flames. To make matters worse, the door jammed, trapping everyone inside the church, all of whom perished in the fire=> sound of fire;
        Confident of inevitable success, they placed the raised dog into the device and activated it. => dog barking sound;
        He spoke loudly and turned on the faucet => screaming sound;
        """+inputtext+""" =>"""
        
        # Few shot prompt (providing a few examples before asking model to complete more);
    ]
    results = generator.text_completion(
        prompts,
        max_gen_len=20,
        temperature=0.6,
        top_p=0.9,
    )
    print(results['generation'])
    return results['generation']

get_prompt("knock knock")

'''    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")'''


'''if __name__ == "__main__":
    fire.Fire(main)'''

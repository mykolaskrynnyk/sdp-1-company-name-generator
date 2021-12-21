"""
This script utility functions for sequence generation.
"""
# deep learning
import tensorflow as tf

def generate(model, start_with: str = '', EOS: str = '!', temperature: float = 1) -> str:
        """

        """
        states = None
        next_char = tf.constant(['^' + start_with])
        result = [next_char]

        while result[-1] != EOS:
            next_char, states = model.generate_one_step(next_char, states = states, temperature = temperature)
            result.append(next_char)

        result = tf.strings.join(result)
        name = result[0].numpy().decode('utf-8')

        return name

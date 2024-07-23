"""Runs inference with a RT-1 model."""

import time
import copy

from absl import app
from absl import flags

from flax.training import checkpoints
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import rt1

_CHECKPOINT_PATH = flags.DEFINE_string("checkpoint_path", None, "Path to checkpoint.")
flags.mark_flag_as_required("checkpoint_path")


class RT1Policy:
    """Runs inference with a RT-1 policy."""

    def __init__(
        self,
        checkpoint_path=None,
        model=rt1.RT1(),
        variables=None,
        seqlen=15,
        rng=None,
    ):
        """Initializes the policy.

        Args:
          checkpoint_path: A checkpoint point from which to load variables. Either
            this or variables must be provided.
          model: A nn.Module to use for the policy. Must match with the variables
            provided by checkpoint_path or variables.
          variables: If provided, will use variables instead of loading from
            checkpoint_path.
          seqlen: The history length to use for observations.
          rng: a jax.random.PRNGKey to use for the random number generator.
        """
        if not variables and not checkpoint_path:
            raise ValueError("At least one of `variables` or `checkpoint_path` must be defined.")
        self.model = model
        self._checkpoint_path = checkpoint_path
        self.seqlen = seqlen

        self._run_action_inference_jit = jax.jit(self._run_action_inference)

        if rng is None:
            self.rng = jax.random.PRNGKey(0)
        else:
            self.rng = rng

        if variables:
            self.variables = variables
        else:
            state_dict = checkpoints.restore_checkpoint(checkpoint_path, None)
            variables = {
                "params": state_dict["params"],
                "batch_stats": state_dict["batch_stats"],
            }
            self.variables = variables

    def _run_action_inference(self, observation, rng):
        """A jittable function for running inference."""

        # We add zero action tokens so that the shape is (seqlen, 11).
        # Note that in the vanilla RT-1 setup, where
        # `include_prev_timesteps_actions=False`, the network will not use the
        # input tokens and instead uses zero action tokens, thereby not using the
        # action history. We still pass it in for simplicity.
        act_tokens = jnp.zeros((1, 6, 11))

        # Add a batch dim to the observation.
        batch_obs = jax.tree_map(lambda x: jnp.expand_dims(x, 0), observation)

        _, random_rng = jax.random.split(rng)

        output_logits = self.model.apply(
            self.variables,
            batch_obs,
            act=None,
            act_tokens=act_tokens,
            train=False,
            rngs={"random": random_rng},
        )

        time_step_tokens = self.model.num_image_tokens + self.model.num_action_tokens
        output_logits = jnp.reshape(output_logits, (1, self.seqlen, time_step_tokens, -1))
        action_logits = output_logits[:, -1, ...]
        action_logits = action_logits[:, self.model.num_image_tokens - 1 : -1]

        action_logp = jax.nn.softmax(action_logits)
        action_token = jnp.argmax(action_logp, axis=-1)

        # Detokenize the full action sequence.
        detokenized = rt1.detokenize_action(
            action_token, self.model.vocab_size, self.model.world_vector_range
        )

        detokenized = jax.tree_map(lambda x: x[0], detokenized)

        return detokenized

    def action(self, observation):
        """Outputs the action given observation from the env."""
        # Assume obs has no batch dimensions.
        observation = copy.deepcopy(observation)

        # Jax does not support string types, so remove it from the dict if it
        # exists.
        if "natural_language_instruction" in observation:
            del observation["natural_language_instruction"]

        image = observation["image"]
        # Resize using TF image resize to avoid any issues with using different
        # resize implementation, since we also use tf.image.resize in the data
        # pipeline. Also scale image to [0, 1].
        image = tf.image.resize(image, (300, 300)).numpy()
        image /= 255.0
        observation["image"] = image

        self.rng, rng = jax.random.split(self.rng)
        action = self._run_action_inference_jit(observation, rng)
        action = jax.device_get(action)

        # Use the base pose mode if the episode if the network outputs an invalid
        # `terminate_episode` action.
        if np.sum(action["terminate_episode"]) == 0:
            action["terminate_episode"] = np.zeros_like(action["terminate_episode"])
            action["terminate_episode"][-1] = 1
        return action


class USE:
    # Universal Sentence Encoder language model, used to embed natural language instructions for RT-1-X
    def __init__(self):
        self.embedding = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def normalize_task_name(self, task_name):
        replaced = (
            task_name.replace("_", " ")
            .replace("1f", " ")
            .replace("4f", " ")
            .replace("-", " ")
            .replace("50", " ")
            .replace("55", " ")
            .replace("56", " ")
        )
        return replaced.lstrip(" ").rstrip(" ")

    def embed(self, instruction):
        natural_language_embedding_tf = self.embedding([self.normalize_task_name(instruction)])[0]
        natural_language_embedding_jax = jnp.array(natural_language_embedding_tf.numpy())
        return natural_language_embedding_jax

def load_images(start=0):
    images = []
    for i in range(start, start + 15):
        image_tf = tf.image.decode_image(tf.io.read_file(f"colabs/{i}.png"))
        image_jax = jnp.array(image_tf.numpy())
        images.append(image_jax)
    images_jax = jnp.stack(images, axis=0)
    return images_jax



def main(argv):
    del argv
    sequence_length = 15
    num_action_tokens = 11
    layer_size = 256
    vocab_size = 512
    num_image_tokens = 81
    rt1x_model = rt1.RT1(
        num_image_tokens=num_image_tokens,
        num_action_tokens=num_action_tokens,
        layer_size=layer_size,
        vocab_size=vocab_size,
        # Use token learner to reduce tokens per image to 81.
        use_token_learner=True,
        # RT-1-X uses (-2.0, 2.0) instead of (-1.0, 1.0).
        world_vector_range=(-2.0, 2.0),
    )
    policy = RT1Policy(
        checkpoint_path=_CHECKPOINT_PATH.value,
        model=rt1x_model,
        seqlen=sequence_length,
    )
    USE_embedding = USE()
    curr = time.time()

    for i in range(10):
        natural_language_embedding_jax = USE_embedding.embed("Pick up the red block")
        stacked_embeddings = jnp.stack([natural_language_embedding_jax] * 15, axis=0)
        images = load_images(i)
        # This is a batched observation
        obs = {
            "image": images,
            "natural_language_embedding": stacked_embeddings,
        }

        action = policy.action(obs)
        print(action)
        print("Frequency,", 1 / (time.time() - curr))
        curr = time.time()


if __name__ == "__main__":
    app.run(main)

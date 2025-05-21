import json
import pathlib
import time

import torch
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from src.llama_2.model import ModelArgs, Transformer


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(
        checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str
    ) -> "LLaMA":
        """Build the LLaMA model.

        Args:
            checkpoints_dir (str): The directory containing the checkpoints.
            tokenizer_path (str): The path to the tokenizer.
            load_model (bool): Whether to load the model.
            max_seq_len (int): The maximum sequence length.
            max_batch_size (int): The maximum batch size.
            device (str): The device to run the model on.

        Returns:
            LLaMA: The LLaMA model.
        """
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(pathlib.Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, "No checkpoints found"
            checkpoint_path = checkpoints[0]
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f} seconds")
            prev_time = time.time()

        with open(pathlib.Path(checkpoints_dir) / "params.json") as f:
            params = json.load(f)

        model_args = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params,
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        elif device == "mps":
            torch.set_default_tensor_type(torch.mps.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = Transformer(model_args).to(device)

        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded model in {time.time() - prev_time:.2f} seconds")

        return LLaMA(model, tokenizer, model_args)

    def text_completion(
        self, prompts: list[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: int | None = None
    ) -> list[str]:
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len

        # Encode the prompts
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        # Make sure batch size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"Batch size {batch_size} is too large"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.args.max_seq_len, f"Max prompt length {max_prompt_len} is too long"
        total_len = min(self.args.max_seq_len, max_prompt_len + max_gen_len)

        # Create a list that will contain the tokens for each prompt
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.args.device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.args.device)

        eos_reached = torch.Tensor([False] * batch_size, device=self.args.device)
        prompt_tokens_mask = tokens != pad_id

        for cur_pos in tqdm(range(1, total_len), desc="Generating"):
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos - 1 : cur_pos], cur_pos)

            # Temperature applied before softmax
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            # Greedy sampling
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.unsqueeze(-1)
            # Only  replace token if it is padding tokens
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)

            tokens[:, cur_pos] = next_token
            # EOS reached if the token is the eos token
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())
            if eos_reached.all():
                break

        output_tokens = []
        output_text = []
        for _, current_prompt_tokens in enumerate(tokens.tolist()):
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            output_tokens.append(current_prompt_tokens)
            output_text.append(self.tokenizer.decode(current_prompt_tokens))

        return (output_text, output_tokens)

    def _sample_top_p(self, probs: torch.Tensor, top_p: float) -> torch.Tensor:
        """Sample from the top-p distribution.

        Args:
            probs (torch.Tensor): The probabilities.
            top_p (float): The top-p value.
        """
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0
        probs_sort = probs_sort / probs_sort.sum(dim=-1, keepdim=True)
        next_token = torch.multinomial(probs_sort, num_samples=1)
        return probs_idx.gather(dim=-1, index=next_token)


if __name__ == "__main__":
    torch.manual_seed(0)
    allow_cuda = False
    allow_mps = False
    device = (
        "cuda"
        if torch.cuda.is_available() and not allow_cuda
        else "mps"
        if torch.backends.mps.is_available() and not allow_mps
        else "cpu"
    )

    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
        # Few shot promt
        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        # Zero shot prompt
        """Tell me if the following person is actually Doraemon disguised as human:
        Name: Umar Jamil
        Decision:
        """,
    ]

    # Get the directory where the script is located
    curr_dir = pathlib.Path(__file__).parent.absolute()

    model = LLaMA.build(
        checkpoints_dir=str(curr_dir / "llama-2-7b"),
        tokenizer_path=str(curr_dir / "tokenizer.model"),
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device,
    )

    print("All good to go!")

    out_tokens, out_texts = model.text_completion(prompts, max_gen_len=64)
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f"{out_texts[i]}")
        print("-" * 50)

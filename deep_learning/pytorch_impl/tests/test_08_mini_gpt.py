"""
Mini GPT 테스트

실행: pytest deep_learning/pytorch_impl/tests/test_08_mini_gpt.py -v
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "mini_gpt", Path(__file__).parent.parent / "08_mini_gpt.py"
)
gpt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpt_module)

MiniGPT = gpt_module.MiniGPT
MiniGPTConfig = gpt_module.MiniGPTConfig


@pytest.fixture
def small_config():
    """테스트용 작은 설정"""
    return MiniGPTConfig(
        vocab_size=100,
        d_model=32,
        num_heads=4,
        num_layers=2,
        max_len=64,
        dropout=0.0,
    )


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 10


class TestMiniGPT:
    """MiniGPT 모델 테스트"""

    def test_initialization(self, small_config):
        """모델이 올바르게 초기화되는지"""
        model = MiniGPT(small_config)

        assert model.config == small_config

    def test_forward_shape(self, small_config, batch_size, seq_len):
        """forward 출력 shape 확인"""
        model = MiniGPT(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        logits, loss = model(input_ids)

        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)
        assert loss is None  # targets 없으면 loss도 없음

    def test_forward_with_targets(self, small_config, batch_size, seq_len):
        """targets와 함께 forward - loss 계산"""
        model = MiniGPT(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        logits, loss = model(input_ids, targets=targets)

        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)
        assert loss is not None
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0  # loss는 양수

    def test_different_seq_lengths(self, small_config):
        """다양한 시퀀스 길이에서 동작"""
        model = MiniGPT(small_config)

        for seq_len in [1, 5, 20, 50]:
            input_ids = torch.randint(0, small_config.vocab_size, (2, seq_len))
            logits, _ = model(input_ids)
            assert logits.shape == (2, seq_len, small_config.vocab_size)

    def test_causal_property(self, small_config):
        """causal 성질 확인 - 미래 토큰이 현재에 영향 X"""
        model = MiniGPT(small_config)
        model.eval()

        # 같은 prefix, 다른 suffix
        input1 = torch.randint(0, small_config.vocab_size, (1, 10))
        input2 = input1.clone()
        input2[:, 5:] = torch.randint(0, small_config.vocab_size, (1, 5))

        logits1, _ = model(input1)
        logits2, _ = model(input2)

        # 앞 5개 position의 logits는 동일
        assert torch.allclose(logits1[:, :5, :], logits2[:, :5, :], atol=1e-5)


class TestGeneration:
    """텍스트 생성 테스트"""

    def test_generate_shape(self, small_config):
        """generate 출력 shape 확인"""
        model = MiniGPT(small_config)
        model.eval()

        prompt = torch.randint(0, small_config.vocab_size, (1, 5))
        max_new_tokens = 10

        generated = model.generate(prompt, max_new_tokens=max_new_tokens)

        assert generated.shape == (1, 5 + max_new_tokens)

    def test_generate_starts_with_prompt(self, small_config):
        """생성된 시퀀스가 prompt로 시작하는지"""
        model = MiniGPT(small_config)
        model.eval()

        prompt = torch.randint(0, small_config.vocab_size, (1, 5))
        generated = model.generate(prompt, max_new_tokens=10)

        assert torch.equal(generated[:, :5], prompt)

    def test_generate_valid_tokens(self, small_config):
        """생성된 토큰이 vocab 범위 내인지"""
        model = MiniGPT(small_config)
        model.eval()

        prompt = torch.randint(0, small_config.vocab_size, (1, 5))
        generated = model.generate(prompt, max_new_tokens=20)

        assert (generated >= 0).all()
        assert (generated < small_config.vocab_size).all()

    def test_greedy_decoding_deterministic(self, small_config):
        """temperature=0이면 deterministic한지"""
        model = MiniGPT(small_config)
        model.eval()

        prompt = torch.randint(0, small_config.vocab_size, (1, 5))

        gen1 = model.generate(prompt, max_new_tokens=10, temperature=0.0001)
        gen2 = model.generate(prompt, max_new_tokens=10, temperature=0.0001)

        assert torch.equal(gen1, gen2)

    def test_temperature_affects_diversity(self, small_config):
        """temperature가 다양성에 영향을 주는지"""
        model = MiniGPT(small_config)
        model.eval()

        prompt = torch.randint(0, small_config.vocab_size, (1, 3))

        # 높은 temperature로 여러 번 생성
        generations = []
        for _ in range(5):
            gen = model.generate(prompt.clone(), max_new_tokens=10, temperature=1.5)
            generations.append(gen)

        # 모든 생성이 같으면 안 됨 (확률적으로 거의 불가능)
        all_same = all(torch.equal(generations[0], g) for g in generations)
        # 참고: 이 테스트는 확률적이므로 가끔 실패할 수 있음
        # 실패하면 temperature를 더 높이거나 생성 횟수를 늘리세요

    def test_batch_generation(self, small_config):
        """배치 생성이 작동하는지"""
        model = MiniGPT(small_config)
        model.eval()

        batch_size = 3
        prompt = torch.randint(0, small_config.vocab_size, (batch_size, 5))

        generated = model.generate(prompt, max_new_tokens=10)

        assert generated.shape == (batch_size, 15)


class TestGradientFlow:
    """Gradient 흐름 테스트"""

    def test_backward_pass(self, small_config):
        """backward pass가 정상 동작하는지"""
        model = MiniGPT(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 10))
        targets = torch.randint(0, small_config.vocab_size, (2, 10))

        _, loss = model(input_ids, targets=targets)
        loss.backward()

        # 모든 파라미터에 gradient가 있는지
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_no_nan_gradients(self, small_config):
        """gradient에 NaN이 없는지"""
        model = MiniGPT(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 10))
        targets = torch.randint(0, small_config.vocab_size, (2, 10))

        _, loss = model(input_ids, targets=targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


# ============================================
# 힌트
# ============================================
"""
=== HINT LEVEL 1: 구조 ===

MiniGPT = TokenEmbed + PosEmbed + DecoderBlocks + FinalNorm + LMHead

Forward:
    1. x = token_embed(input_ids) + pos_embed(positions)
    2. x = dropout(x)
    3. for block in blocks: x = block(x)
    4. x = final_norm(x)
    5. logits = lm_head(x)

Generate:
    - Autoregressive loop
    - 매 step마다 마지막 position의 logits만 사용
    - Sampling 또는 argmax


=== HINT LEVEL 2: 핵심 코드 ===

__init__:
    self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
    self.position_embedding = nn.Embedding(config.max_len, config.d_model)
    self.dropout = nn.Dropout(config.dropout)

    self.blocks = nn.ModuleList([
        DecoderBlock(config.d_model, config.num_heads, config.d_ff, config.dropout)
        for _ in range(config.num_layers)
    ])

    self.final_norm = nn.LayerNorm(config.d_model)
    self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

forward:
    # Embeddings
    tok_emb = self.token_embedding(input_ids)
    pos = torch.arange(seq_len, device=input_ids.device)
    pos_emb = self.position_embedding(pos)

    x = self.dropout(tok_emb + pos_emb)

    # Blocks
    for block in self.blocks:
        x = block(x)

    x = self.final_norm(x)
    logits = self.lm_head(x)

    # Loss
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

    return logits, loss


=== HINT LEVEL 3: generate 전체 구현 ===

@torch.no_grad()
def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=None, top_p=None):
    self.eval()

    for _ in range(max_new_tokens):
        # 최대 길이 제한
        idx_cond = input_ids if input_ids.size(1) <= self.config.max_len else input_ids[:, -self.config.max_len:]

        # Forward pass
        logits, _ = self(idx_cond)

        # 마지막 position의 logits만 사용
        logits = logits[:, -1, :]  # (batch, vocab_size)

        # Temperature 적용
        if temperature > 0:
            logits = logits / temperature

        # Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        # Top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        # Sampling
        probs = F.softmax(logits, dim=-1)

        if temperature == 0 or temperature < 1e-6:
            # Greedy
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            # Sample
            next_token = torch.multinomial(probs, num_samples=1)

        # Concat
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids
"""

"""Tests for puffer-explore — correctness, interface, and performance."""

import pytest
import torch
import time


DEVICE = "cpu"  # Tests run on CPU for CI; benchmark.py tests GPU


class TestTinyMLP:
    def test_shape(self):
        from puffer_explore.networks import TinyMLP
        net = TinyMLP(input_dim=128, output_dim=64, hidden_dim=128)
        x = torch.randn(32, 128)
        assert net(x).shape == (32, 64)

    def test_param_count(self):
        from puffer_explore.networks import TinyMLP
        net = TinyMLP(input_dim=128, output_dim=64, hidden_dim=128)
        n_params = sum(p.numel() for p in net.parameters())
        # Should be small: ~33K for 128→128→128→64
        assert n_params < 50_000, f"Too many params: {n_params}"

    def test_compile(self):
        from puffer_explore.networks import TinyMLP, compile_network
        net = TinyMLP(input_dim=64, output_dim=32, hidden_dim=64)
        example = torch.randn(16, 64)
        compiled = compile_network(net, example)
        out = compiled(example)
        assert out.shape == (16, 32)


class TestBaseExploration:
    def test_pre_allocated_buffer(self):
        from puffer_explore.methods.rnd import RND
        rnd = RND(obs_dim=64, n_envs=32, rollout_steps=16, device=DEVICE, use_compile=False)
        assert rnd._intrinsic_rewards.shape == (32 * 16,)
        assert rnd._intrinsic_rewards.device.type == DEVICE

    def test_beta_decay(self):
        from puffer_explore.methods.rnd import RND
        rnd = RND(obs_dim=64, n_envs=4, rollout_steps=4, beta=1.0, beta_decay=0.5,
                   device=DEVICE, use_compile=False)
        obs = torch.randn(16, 64)
        next_obs = torch.randn(16, 64)
        actions = torch.randint(0, 4, (16,))
        rewards = torch.zeros(16)

        rnd.augment_rewards(rewards, obs, next_obs, actions)
        assert rnd.beta == pytest.approx(0.5)  # 1.0 * 0.5^1

        rnd.augment_rewards(rewards, obs, next_obs, actions)
        assert rnd.beta == pytest.approx(0.25)  # 1.0 * 0.5^2


class TestRND:
    def test_reward_shape(self):
        from puffer_explore.methods.rnd import RND
        rnd = RND(obs_dim=64, n_envs=32, rollout_steps=16, device=DEVICE, use_compile=False)
        obs = torch.randn(512, 64)
        next_obs = torch.randn(512, 64)
        actions = torch.randint(0, 4, (512,))
        r = rnd.compute_rewards(obs, next_obs, actions)
        assert r.shape == (512,)
        assert (r >= 0).all()

    def test_update_reduces_loss(self):
        from puffer_explore.methods.rnd import RND
        rnd = RND(obs_dim=64, n_envs=4, rollout_steps=4, lr=0.01, device=DEVICE, use_compile=False)
        obs = torch.randn(64, 64)
        next_obs = torch.randn(64, 64)
        actions = torch.randint(0, 4, (64,))

        m1 = rnd.update(obs, next_obs, actions)
        for _ in range(50):
            rnd.update(obs, next_obs, actions)
        m2 = rnd.update(obs, next_obs, actions)
        assert m2["explore/rnd_loss"] < m1["explore/rnd_loss"]

    def test_target_frozen(self):
        from puffer_explore.methods.rnd import RND
        rnd = RND(obs_dim=64, device=DEVICE, use_compile=False)
        for p in rnd.target.parameters():
            assert not p.requires_grad

    def test_augment_modifies_rewards(self):
        from puffer_explore.methods.rnd import RND
        rnd = RND(obs_dim=64, n_envs=4, rollout_steps=4, beta=1.0, device=DEVICE, use_compile=False)
        obs = torch.randn(16, 64)
        next_obs = torch.randn(16, 64)
        actions = torch.randint(0, 4, (16,))
        rewards = torch.zeros(16)

        rnd.augment_rewards(rewards, obs, next_obs, actions)
        assert not torch.allclose(rewards, torch.zeros(16))


class TestCountBased:
    def test_reward_shape(self):
        from puffer_explore.methods.count_based import CountBased
        cb = CountBased(obs_dim=64, n_envs=32, rollout_steps=16, device=DEVICE)
        obs = torch.randn(512, 64)
        r = cb.compute_rewards(obs, obs, torch.zeros(512, dtype=torch.long))
        assert r.shape == (512,)
        assert (r > 0).all()

    def test_revisit_lower_reward(self):
        from puffer_explore.methods.count_based import CountBased
        cb = CountBased(obs_dim=64, n_envs=1, rollout_steps=2, device=DEVICE)
        obs = torch.ones(2, 64) * 0.5
        r1 = cb.compute_rewards(obs[:1], obs[:1], torch.zeros(1, dtype=torch.long))
        r2 = cb.compute_rewards(obs[:1], obs[:1], torch.zeros(1, dtype=torch.long))
        assert r2[0] < r1[0]

    def test_no_trainable_params(self):
        from puffer_explore.methods.count_based import CountBased
        cb = CountBased(obs_dim=64, device=DEVICE)
        metrics = cb.update(torch.randn(8, 64), torch.randn(8, 64), torch.zeros(8, dtype=torch.long))
        assert "explore/unique_buckets" in metrics


class TestNovelD:
    def test_reward_nonnegative(self):
        from puffer_explore.methods.noveld import NovelD
        nd = NovelD(obs_dim=64, n_envs=16, rollout_steps=8, device=DEVICE, use_compile=False)
        obs = torch.randn(128, 64)
        next_obs = torch.randn(128, 64)
        actions = torch.randint(0, 4, (128,))
        r = nd.compute_rewards(obs, next_obs, actions)
        assert (r >= 0).all()

    def test_erir_zeros_revisits(self):
        from puffer_explore.methods.noveld import NovelD
        torch.manual_seed(42)
        nd = NovelD(obs_dim=64, n_envs=1, rollout_steps=4, use_erir=True, alpha=0.0,
                    device=DEVICE, use_compile=False)
        # Use random obs that produce non-trivial novelty
        obs = torch.randn(4, 64)
        # All four next_obs are IDENTICAL — ERIR should keep only first, zero the rest
        repeated_next = torch.randn(1, 64).expand(4, -1).contiguous()
        actions = torch.zeros(4, dtype=torch.long)

        r = nd.compute_rewards(obs, repeated_next, actions)
        # Indices 1,2,3 should be zero (duplicate of index 0's hash)
        assert r[1] == 0.0, f"Duplicate should be zero, got {r[1]}"
        assert r[2] == 0.0, f"Duplicate should be zero, got {r[2]}"
        assert r[3] == 0.0, f"Duplicate should be zero, got {r[3]}"


class TestICM:
    def test_reward_shape(self):
        from puffer_explore.methods.icm import ICM
        icm = ICM(obs_dim=64, n_actions=4, n_envs=16, rollout_steps=8,
                   device=DEVICE, use_compile=False)
        obs = torch.randn(128, 64)
        next_obs = torch.randn(128, 64)
        actions = torch.randint(0, 4, (128,))
        r = icm.compute_rewards(obs, next_obs, actions)
        assert r.shape == (128,)
        assert (r >= 0).all()

    def test_update_returns_losses(self):
        from puffer_explore.methods.icm import ICM
        icm = ICM(obs_dim=64, n_actions=4, device=DEVICE, use_compile=False)
        obs = torch.randn(32, 64)
        next_obs = torch.randn(32, 64)
        actions = torch.randint(0, 4, (32,))
        m = icm.update(obs, next_obs, actions)
        assert "explore/icm_fwd_loss" in m
        assert "explore/icm_inv_loss" in m


class TestEnsemble:
    def test_reward_shape(self):
        from puffer_explore.methods.ensemble import EnsembleDisagreement
        ens = EnsembleDisagreement(obs_dim=64, n_actions=4, n_envs=16, rollout_steps=8,
                                    n_models=3, device=DEVICE)
        obs = torch.randn(128, 64)
        next_obs = torch.randn(128, 64)
        actions = torch.randint(0, 4, (128,))
        r = ens.compute_rewards(obs, next_obs, actions)
        assert r.shape == (128,)


class TestNGU:
    def test_reward_shape(self):
        from puffer_explore.methods.ngu import NGU
        ngu = NGU(obs_dim=64, n_envs=16, rollout_steps=8, device=DEVICE, use_compile=False)
        obs = torch.randn(128, 64)
        next_obs = torch.randn(128, 64)
        actions = torch.randint(0, 4, (128,))
        r = ngu.compute_rewards(obs, next_obs, actions)
        assert r.shape == (128,)
        assert (r >= 0).all()

    def test_episodic_resets(self):
        from puffer_explore.methods.ngu import NGU
        ngu = NGU(obs_dim=64, n_envs=4, rollout_steps=4, device=DEVICE, use_compile=False)
        obs = torch.randn(16, 64)
        next_obs = torch.randn(16, 64)
        actions = torch.zeros(16, dtype=torch.long)
        ngu.compute_rewards(obs, next_obs, actions)
        assert ngu._epi_counts.sum() > 0
        ngu.update(obs, next_obs, actions)
        assert ngu._epi_counts.sum() == 0  # Reset after update


class TestRIDE:
    def test_reward_shape(self):
        from puffer_explore.methods.ride import RIDE
        ride = RIDE(obs_dim=64, n_actions=4, n_envs=16, rollout_steps=8,
                     device=DEVICE, use_compile=False)
        obs = torch.randn(128, 64)
        next_obs = torch.randn(128, 64)
        actions = torch.randint(0, 4, (128,))
        r = ride.compute_rewards(obs, next_obs, actions)
        assert r.shape == (128,)
        assert (r >= 0).all()

    def test_update_returns_losses(self):
        from puffer_explore.methods.ride import RIDE
        ride = RIDE(obs_dim=64, n_actions=4, device=DEVICE, use_compile=False)
        obs = torch.randn(32, 64)
        next_obs = torch.randn(32, 64)
        actions = torch.randint(0, 4, (32,))
        m = ride.update(obs, next_obs, actions)
        assert "explore/ride_fwd_loss" in m
        assert "explore/ride_inv_loss" in m


class TestIntegration:
    def test_create_all_methods(self):
        from puffer_explore.integration import create_exploration, METHODS
        assert len(METHODS) == 7, f"Expected 7 methods, got {len(METHODS)}"
        for name in METHODS:
            kwargs = {"n_actions": 4} if name in ("icm", "ensemble", "ride") else {}
            if name in ("rnd", "noveld", "icm", "ngu", "ride"):
                kwargs["use_compile"] = False
            e = create_exploration(name, obs_dim=64, n_envs=4, rollout_steps=4,
                                   device=DEVICE, **kwargs)
            obs = torch.randn(16, 64)
            r = e.compute_rewards(obs, obs, torch.zeros(16, dtype=torch.long))
            assert r.shape == (16,), f"Failed for {name}"

    def test_standalone_trainer(self):
        """Test the standalone trainer without PufferLib dependency."""
        import gymnasium as gym
        from puffer_explore.integration import StandaloneExploreTrainer
        from puffer_explore.methods.count_based import CountBased

        env = gym.make("CartPole-v1")
        obs_dim = env.observation_space.shape[0]
        # High beta so the augmentation is clearly visible
        exploration = CountBased(obs_dim=obs_dim, n_envs=1, rollout_steps=32,
                                 device=DEVICE, beta=10.0)

        policy = torch.nn.Linear(obs_dim, env.action_space.n)
        trainer = StandaloneExploreTrainer(env, exploration, device=DEVICE)
        rollout = trainer.collect_rollout(policy, rollout_steps=32)

        assert rollout["obs"].shape == (32, obs_dim)
        assert rollout["rewards"].shape == (32,)
        # With beta=10.0, intrinsic reward should visibly change the totals
        diff = (rollout["rewards"] - rollout["raw_rewards"]).abs().sum()
        assert diff > 0.01, f"Augmentation should be visible, diff={diff}"
        env.close()


class TestThroughput:
    """Basic throughput regression tests. Not comprehensive (use benchmark.py for that)."""

    def test_rnd_fast_enough(self):
        """RND compute_rewards should process 100K+ obs in reasonable time on CPU."""
        from puffer_explore.methods.rnd import RND
        rnd = RND(obs_dim=128, n_envs=256, rollout_steps=64, device=DEVICE, use_compile=False)
        n = 256 * 64
        obs = torch.randn(n, 128)

        rnd.compute_rewards(obs, obs, torch.zeros(n, dtype=torch.long))

        t0 = time.perf_counter()
        for _ in range(10):
            rnd.compute_rewards(obs, obs, torch.zeros(n, dtype=torch.long))
        elapsed = (time.perf_counter() - t0) / 10

        sps = n / elapsed
        # CPU target: 50K+ SPS. GPU target (benchmark.py): 10M+ SPS.
        assert sps > 50_000, f"Too slow: {sps:.0f} SPS (need >50K on CPU)"

    def test_count_based_faster_than_rnd(self):
        """Count-based should be faster than RND (no neural network)."""
        from puffer_explore.methods.rnd import RND
        from puffer_explore.methods.count_based import CountBased

        n = 16384
        obs = torch.randn(n, 64)
        actions = torch.zeros(n, dtype=torch.long)

        rnd = RND(obs_dim=64, n_envs=128, rollout_steps=128, device=DEVICE, use_compile=False)
        cb = CountBased(obs_dim=64, n_envs=128, rollout_steps=128, device=DEVICE)

        # Warm up
        rnd.compute_rewards(obs, obs, actions)
        cb.compute_rewards(obs, obs, actions)

        t0 = time.perf_counter()
        for _ in range(20):
            rnd.compute_rewards(obs, obs, actions)
        rnd_time = (time.perf_counter() - t0) / 20

        t0 = time.perf_counter()
        for _ in range(20):
            cb.compute_rewards(obs, obs, actions)
        cb_time = (time.perf_counter() - t0) / 20

        assert cb_time < rnd_time, f"Count-based ({cb_time:.4f}s) should be faster than RND ({rnd_time:.4f}s)"


class TestGoExplore:
    def test_archive_add(self):
        from puffer_explore.methods.go_explore import GoExploreArchive
        import numpy as np
        archive = GoExploreArchive()
        obs = np.random.rand(10).astype(np.float32)
        added = archive.add_or_update(obs, [0, 1, 2], score=1.0)
        assert added
        assert archive.size == 1

    def test_archive_update_better(self):
        from puffer_explore.methods.go_explore import GoExploreArchive
        import numpy as np
        archive = GoExploreArchive()
        obs = np.ones(10, dtype=np.float32) * 0.5
        archive.add_or_update(obs, [0, 1, 2, 3], score=1.0)
        updated = archive.add_or_update(obs, [0, 1], score=2.0)
        assert updated
        cell = list(archive.cells.values())[0]
        assert cell.score == 2.0
        assert len(cell.trajectory) == 2

    def test_archive_select(self):
        from puffer_explore.methods.go_explore import GoExploreArchive
        import numpy as np
        archive = GoExploreArchive()
        for i in range(10):
            obs = np.random.rand(10).astype(np.float32) * i
            archive.add_or_update(obs, list(range(i)), score=float(i))
        cell = archive.select_cell()
        assert cell is not None

    def test_phase1_runs(self):
        import gymnasium as gym
        from puffer_explore.methods.go_explore import GoExplorePhase1
        env = gym.make("CartPole-v1")
        explorer = GoExplorePhase1(env, explore_steps=20)
        stats = explorer.run(total_steps=500, verbose=False)
        assert stats["archive_size"] > 0
        assert stats["total_steps"] >= 500
        env.close()


class TestBetaScheduler:
    def test_constant(self):
        from puffer_explore.config import BetaScheduler
        sched = BetaScheduler(initial_beta=0.01, schedule="constant")
        for _ in range(100):
            b = sched.step()
        assert b == 0.01

    def test_linear_decay(self):
        from puffer_explore.config import BetaScheduler
        sched = BetaScheduler(initial_beta=1.0, schedule="linear", beta_min=0.0, total_steps=100)
        betas = [sched.step() for _ in range(100)]
        assert betas[0] > betas[-1]
        assert betas[-1] == pytest.approx(0.0, abs=0.02)

    def test_cosine(self):
        from puffer_explore.config import BetaScheduler
        sched = BetaScheduler(initial_beta=1.0, schedule="cosine", beta_min=0.0, total_steps=100)
        betas = [sched.step() for _ in range(100)]
        assert betas[0] > betas[50] > betas[-1]

    def test_exponential(self):
        from puffer_explore.config import BetaScheduler
        sched = BetaScheduler(initial_beta=1.0, schedule="exponential", beta_decay=0.99)
        b1 = sched.step()
        for _ in range(98):
            sched.step()
        b100 = sched.step()
        assert b100 < b1


class TestSharedEncoder:
    def test_creates_with_feature_dim(self):
        from puffer_explore.shared_encoder import create_shared_encoder_exploration
        se = create_shared_encoder_exploration(
            "rnd", feature_dim=128, n_envs=4, rollout_steps=4,
            device=DEVICE, use_compile=False,
        )
        features = torch.randn(16, 128)
        r = se.compute_rewards(features, features, torch.zeros(16, dtype=torch.long))
        assert r.shape == (16,)

    def test_augment_rewards(self):
        from puffer_explore.shared_encoder import create_shared_encoder_exploration
        se = create_shared_encoder_exploration(
            "rnd", feature_dim=64, n_envs=4, rollout_steps=4,
            device=DEVICE, beta=10.0, use_compile=False,
        )
        features = torch.randn(16, 64)
        next_features = torch.randn(16, 64)  # Different from features → nonzero RND error
        rewards = torch.zeros(16)
        se.augment_rewards(rewards, features, next_features, torch.zeros(16, dtype=torch.long))
        assert rewards.abs().sum() > 0


class TestConfig:
    def test_load_preset(self):
        from puffer_explore.config import load_explore_config
        cfg = load_explore_config("breakout_rnd")
        assert cfg.method == "rnd"
        assert cfg.beta == 0.005

    def test_write_ini(self):
        from puffer_explore.config import ExploreConfig, write_ini_section
        cfg = ExploreConfig(method="rnd", beta=0.01)
        ini = write_ini_section(cfg)
        assert "[explore]" in ini
        assert "method = rnd" in ini

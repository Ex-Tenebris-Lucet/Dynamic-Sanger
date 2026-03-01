"""
Sanger's Rule: Online PCA via the Generalized Hebbian Algorithm.

Finds the top-k principal components of a data stream, one sample at a time.
Neuron i converges to the i-th eigenvector of the input covariance matrix.
Gram-Schmidt decorrelation emerges naturally from hierarchical subtraction:
each neuron learns the residual after subtracting all higher-variance neurons.

Enhancements beyond vanilla Sanger:

1. EMA centering (sensory adaptation): Subtracts a bias-corrected running mean
   from input before learning, so W discovers variance structure rather than
   mean direction. No zero-mean data requirement.

2. Norm-based adaptive learning rate: Two factors scale lr per neuron, both
   derived from ||W_i||² (weight vector norm squared):

   a) own_factor = min(1, sqrt(|norm_sq - 1|)): Sanger drives converged
      neurons to unit norm. sqrt shape stays adaptive through most of
      convergence, drops sharply near 1.0. Converged neurons freeze.

   b) pred_gate = max(0, 1 - (||W_{i-1}||² - 1)²): neuron i's lr scales
      with predecessor convergence. Parabola: conservative sequential unlock.

3. Input energy normalization: gradient divided by ||x||², making updates
   scale-invariant. Allows a single base_lr across data regimes.

4. Conditional learning: tracks EMA of explained variance (sum(ema_y2) / ema_x2).
   Skips gradient computation when variance is well-explained. Both EMAs use
   the same decay, so bias cancels in the ratio.

5. Dynamic sizing: grows k when the last neuron crosses norm 0.25 (indicating
   real variance found). Input dimension (d) expansion is always handled.
   Converged neurons freeze naturally via own_factor. Spare neurons stay quiet
   at ||W||~0.05 until new variance arrives.

Two entry points:
    sanger_step(W, x, ...)       -> single vector, online update
    sanger_step_batch(W, X, ...) -> N vectors, batch gradient (three matmuls)
"""

import mlx.core as mx


def _compute_lr(W, base_lr):
    """
    Compute per-neuron learning rate from weight norms.

    lr[i] = pred_gate[i] * own_factor[i] * base_lr

    own_factor: min(1, sqrt(|norm_sq - 1|)) -- sqrt stays adaptive through
                most of convergence, drops sharply near target.
    pred_gate:  max(0, 1 - (pred_norm_sq - 1)^2) -- parabola, conservative.
    """
    norm_sq = mx.sum(W * W, axis=1)  # (k,)

    own_factor = mx.minimum(1.0, mx.sqrt(mx.abs(norm_sq - 1.0)))

    pred_norm_sq = norm_sq[:-1]
    pred_convergence = (pred_norm_sq - 1.0) * (pred_norm_sq - 1.0)
    pred_gate = mx.concatenate([
        mx.array([1.0]),
        mx.maximum(0.0, 1.0 - pred_convergence)
    ])

    return pred_gate * own_factor * base_lr, norm_sq


def sanger_step(W, x, ema_y2, ema_input, ema_x2, step,
                base_lr=1.0, eigen_decay=0.99, center_decay=0.9,
                learn_threshold=0.95, dynamic=True, eps=1e-8):
    """
    One step of Sanger's rule with dynamic sizing and conditional learning.

    Args:
        W: Weight matrix (k, d). Row i -> i-th PC direction. Init: randn * 0.01.
        x: Input vector (d,). Raw (uncentered).
        ema_y2: Per-neuron EMA of y^2 (k,). Init: zeros.
        ema_input: Running mean of input (d,). Init: zeros.
        ema_x2: EMA of input energy scalar. Init: 0.
        step: Step counter (scalar mx.array). Init: 0.
        base_lr: Base learning rate (scale-invariant due to ||x||^2 normalization).
        eigen_decay: EMA decay for variance tracking. 0.99 ~ 100-sample window.
        center_decay: EMA decay for input centering. 0.0 to disable.
        learn_threshold: Skip learning when explained variance >= this.
                        None to always learn (no eval overhead).
        dynamic: If True, grow k when last neuron crosses norm 0.25.
        eps: Numerical stability floor.

    Returns:
        W, ema_y2, ema_input, ema_x2, step, y
        Note: W and ema_y2 may be larger than input if growth occurred.
    """
    # d-growth: expand W if input dimension grew
    if x.shape[0] > W.shape[1]:
        n_new = x.shape[0] - W.shape[1]
        W = mx.concatenate([W, mx.random.normal((W.shape[0], n_new)) * 0.01], axis=1)
        ema_input = mx.concatenate([ema_input, mx.zeros((n_new,))])

    # EMA centering with bias correction
    if center_decay > 0:
        step = step + 1
        ema_input = center_decay * ema_input + (1.0 - center_decay) * x
        correction = 1.0 / (1.0 - mx.power(mx.array(center_decay), step) + eps)
        x_centered = x - ema_input * correction
    else:
        x_centered = x

    # Forward pass (always)
    y = W @ x_centered  # (k,)

    # Update tracking EMAs
    ema_y2 = eigen_decay * ema_y2 + (1.0 - eigen_decay) * (y * y)
    x_energy = mx.sum(x_centered * x_centered)
    ema_x2 = eigen_decay * ema_x2 + (1.0 - eigen_decay) * x_energy

    # Pre-compute checks (single eval for both variance and growth)
    need_check = learn_threshold is not None or dynamic
    should_learn = True
    last_norm_sq = None

    if need_check:
        explained = mx.sum(ema_y2) / (ema_x2 + eps)
        if dynamic:
            last_norm_sq = mx.sum(W[-1] * W[-1])
            mx.eval(explained, last_norm_sq)
        else:
            mx.eval(explained)

        if learn_threshold is not None and float(explained) >= learn_threshold:
            should_learn = False

    if not should_learn:
        return W, ema_y2, ema_input, ema_x2, step, y

    # Norm-based adaptive lr
    lr, _ = _compute_lr(W, base_lr)

    # Sanger's rule (vectorized), normalized by input energy
    yyT = mx.outer(y, y)
    gram_schmidt = mx.tril(yyT) @ W
    hebbian = mx.outer(y, x_centered)
    dW = mx.expand_dims(lr, 1) * (hebbian - gram_schmidt) / (x_energy + eps)
    W = W + dW

    # Dynamic k-growth: spawn when last neuron crosses norm 0.25
    if dynamic and last_norm_sq is not None and float(last_norm_sq) > 0.0625:
        W = mx.concatenate([W, mx.random.normal((1, W.shape[1])) * 0.01], axis=0)
        ema_y2 = mx.concatenate([ema_y2, mx.array([0.0])])

    return W, ema_y2, ema_input, ema_x2, step, y


def sanger_step_batch(W, X, ema_y2, ema_input, ema_x2, step,
                      base_lr=1.0, eigen_decay=0.99, center_decay=0.9,
                      learn_threshold=0.95, dynamic=True, eps=1e-8):
    """
    Batch Sanger update from N samples sharing one weight matrix.

    Three matrix multiplies, no Python loops. Designed for spatial weight
    sharing (e.g. V2: 1,764 positions per frame, all sharing one W).

    Args:
        W: Weight matrix (k, d). Row i -> i-th PC direction.
        X: Batch of input vectors (N, d). Raw (uncentered).
        ema_y2: Per-neuron EMA of y^2 (k,). Init: zeros.
        ema_input: Running mean of input (d,). Init: zeros.
        ema_x2: EMA of input energy scalar. Init: 0.
        step: Step counter (scalar mx.array). Init: 0.
        base_lr: Base learning rate.
        eigen_decay: EMA decay for variance tracking.
        center_decay: EMA decay for input centering. 0.0 to disable.
        learn_threshold: Skip learning when explained variance >= this.
                        None to always learn.
        dynamic: If True, grow k when last neuron crosses norm 0.25.
        eps: Numerical stability floor.

    Returns:
        W, ema_y2, ema_input, ema_x2, step, Y
        Note: W and ema_y2 may be larger than input if growth occurred.
              Y shape matches pre-growth k.
    """
    N = X.shape[0]

    # d-growth: expand W if input dimension grew
    if X.shape[1] > W.shape[1]:
        n_new = X.shape[1] - W.shape[1]
        W = mx.concatenate([W, mx.random.normal((W.shape[0], n_new)) * 0.01], axis=1)
        ema_input = mx.concatenate([ema_input, mx.zeros((n_new,))])

    # EMA centering with bias correction
    if center_decay > 0:
        step = step + 1
        batch_mean = mx.mean(X, axis=0)
        ema_input = center_decay * ema_input + (1.0 - center_decay) * batch_mean
        correction = 1.0 / (1.0 - mx.power(mx.array(center_decay), step) + eps)
        X_centered = X - ema_input * correction
    else:
        X_centered = X

    # Forward pass (always)
    Y = X_centered @ W.T  # (N, k)

    # Update tracking EMAs
    mean_y2 = mx.mean(Y * Y, axis=0)
    ema_y2 = eigen_decay * ema_y2 + (1.0 - eigen_decay) * mean_y2
    x_energy = mx.mean(mx.sum(X_centered * X_centered, axis=1))
    ema_x2 = eigen_decay * ema_x2 + (1.0 - eigen_decay) * x_energy

    # Pre-compute checks (single eval)
    need_check = learn_threshold is not None or dynamic
    should_learn = True
    last_norm_sq = None

    if need_check:
        explained = mx.sum(ema_y2) / (ema_x2 + eps)
        if dynamic:
            last_norm_sq = mx.sum(W[-1] * W[-1])
            mx.eval(explained, last_norm_sq)
        else:
            mx.eval(explained)

        if learn_threshold is not None and float(explained) >= learn_threshold:
            should_learn = False

    if not should_learn:
        return W, ema_y2, ema_input, ema_x2, step, Y

    # Norm-based adaptive lr
    lr, _ = _compute_lr(W, base_lr)

    # Batch Sanger gradient, normalized by mean input energy
    hebbian = (Y.T @ X_centered) / N
    gram_schmidt = mx.tril((Y.T @ Y) / N) @ W
    dW = mx.expand_dims(lr, 1) * (hebbian - gram_schmidt) / (x_energy + eps)
    W = W + dW

    # Dynamic k-growth
    if dynamic and last_norm_sq is not None and float(last_norm_sq) > 0.0625:
        W = mx.concatenate([W, mx.random.normal((1, W.shape[1])) * 0.01], axis=0)
        ema_y2 = mx.concatenate([ema_y2, mx.array([0.0])])

    return W, ema_y2, ema_input, ema_x2, step, Y


if __name__ == "__main__":
    """
    Vacuum tests: Sanger's rule with dynamic sizing.

    All tests start with k=2 and grow dynamically.

    1-2: Basic PCA quality (zero-mean, offset data)
    3:   Distribution shift with norm crossing timeline
    4-5: Batch Sanger (zero-mean, offset)
    6:   Batch V2-scale (d=32, grows from k=2)
    7:   Dynamic growth diagnostic

    Setup: eigenvalues [8, 4, 2, 1, 0.5^k...].
    """
    import numpy as np
    import time

    def make_data(d, n_samples, seed=42, mean_offset=None):
        """Generate data with known eigenstructure."""
        np.random.seed(seed)
        eigenvalues = np.array([8.0, 4.0, 2.0, 1.0] +
                               [0.5**i for i in range(d - 4)])
        Q, _ = np.linalg.qr(np.random.randn(d, d))
        L = np.linalg.cholesky(Q @ np.diag(eigenvalues) @ Q.T)
        X = np.random.randn(n_samples, d) @ L.T
        if mean_offset is not None:
            X += mean_offset
        return X, eigenvalues, Q

    def evaluate(W_mx, X):
        """Measure variance captured vs optimal SVD (converged neurons only)."""
        W_np = np.array(W_mx)
        norms = np.linalg.norm(W_np, axis=1)
        mask = norms > 0.5  # only count converged neurons
        W_active = W_np[mask]
        k_conv = W_active.shape[0]
        k_total = len(norms)
        if k_conv == 0:
            return 0.0, 0, k_total
        X_c = X - X.mean(axis=0)
        projected = X_c @ W_active.T
        captured = np.sum(np.var(projected, axis=0))
        _, S, Vt = np.linalg.svd(X_c, full_matrices=False)
        optimal = np.sum(np.var(X_c @ Vt[:k_conv].T, axis=0))
        return captured / optimal, k_conv, k_total

    d, k_init, n_samples = 16, 2, 2048
    print("=== Sanger's Rule Vacuum Test (Dynamic Sizing) ===\n")

    # --- Test 1: Zero-mean data ---
    print(f"Test 1: Zero-mean data ({n_samples} samples, k_init={k_init})")
    X, true_eig, Q = make_data(d, n_samples)

    W1 = mx.array(np.random.RandomState(0).randn(k_init, d) * 0.01)
    ema_y2_1 = mx.zeros((k_init,))
    ema_input1 = mx.zeros((d,))
    ema_x2_1 = mx.array(0.0)
    step1 = mx.array(0)

    t0 = time.time()
    for i in range(len(X)):
        x = mx.array(X[i])
        W1, ema_y2_1, ema_input1, ema_x2_1, step1, y = sanger_step(
            W1, x, ema_y2_1, ema_input1, ema_x2_1, step1)
    mx.eval(W1, ema_y2_1)
    t1 = time.time()

    ratio1, k1_conv, k1_total = evaluate(W1, X)
    norms1 = np.linalg.norm(np.array(W1), axis=1)
    print(f"  Grew to k={k1_total} ({k1_conv} converged)")
    print(f"  Variance ratio: {ratio1*100:.1f}% of optimal top-{k1_conv}")
    print(f"  Norms: {norms1.round(3)}")
    print(f"  Time: {(t1-t0)*1000:.0f}ms")

    # --- Test 2: Non-zero-mean data ---
    print(f"\nTest 2: Non-zero-mean data ({n_samples} samples, k_init={k_init})")
    mean_offset = np.array([10.0, -5.0, 3.0, -8.0] + [2.0] * (d - 4))
    X_offset, _, _ = make_data(d, n_samples, mean_offset=mean_offset)

    W2 = mx.array(np.random.RandomState(0).randn(k_init, d) * 0.01)
    ema_y2_2 = mx.zeros((k_init,))
    ema_input2 = mx.zeros((d,))
    ema_x2_2 = mx.array(0.0)
    step2 = mx.array(0)

    for i in range(len(X_offset)):
        x = mx.array(X_offset[i])
        W2, ema_y2_2, ema_input2, ema_x2_2, step2, y = sanger_step(
            W2, x, ema_y2_2, ema_input2, ema_x2_2, step2)
    mx.eval(W2, ema_y2_2)
    ratio2, k2_conv, k2_total = evaluate(W2, X_offset)
    print(f"  Grew to k={k2_total} ({k2_conv} converged)")
    print(f"  Variance ratio: {ratio2*100:.1f}% of optimal top-{k2_conv}")

    # --- Test 3: Distribution shift ---
    n_half = 512
    print(f"\nTest 3: Distribution shift ({n_half} + {n_half} samples, k_init={k_init})")
    X_shift1, _, _ = make_data(d, n_half, seed=42)
    X_shift2, _, _ = make_data(d, n_half, seed=99)
    X_shift2 *= 2.0
    X_shift = np.concatenate([X_shift1, X_shift2], axis=0)

    W3 = mx.array(np.random.RandomState(0).randn(k_init, d) * 0.01)
    ema_y2_3 = mx.zeros((k_init,))
    ema_input3 = mx.zeros((d,))
    ema_x2_3 = mx.array(0.0)
    step3 = mx.array(0)

    # Track norm crossings per neuron (dynamic k)
    thresholds = [0.25, 0.5, 0.75, 0.9, 0.99]
    crossings = {t: {} for t in thresholds}

    for i in range(n_half * 2):
        x = mx.array(X_shift[i])
        W3, ema_y2_3, ema_input3, ema_x2_3, step3, y = sanger_step(
            W3, x, ema_y2_3, ema_input3, ema_x2_3, step3)
        if (i+1) % 8 == 0:
            mx.eval(W3)
            norms = np.linalg.norm(np.array(W3), axis=1)
            for ni in range(len(norms)):
                for t in thresholds:
                    if ni not in crossings[t] and norms[ni] >= t:
                        crossings[t][ni] = i + 1

    mx.eval(W3, ema_y2_3)
    ratio3, k3_conv, k3_total = evaluate(W3, X_shift)
    norms3 = np.linalg.norm(np.array(W3), axis=1)

    print(f"  Grew to k={k3_total} ({k3_conv} converged)")
    print(f"  Norm crossing timeline:")
    print(f"  {'Neuron':>8s}", end="")
    for t in thresholds:
        print(f"  {t:>6.2f}", end="")
    print()
    all_neurons = sorted(set().union(*[crossings[t].keys() for t in thresholds]))
    for ni in all_neurons:
        if crossings[0.25].get(ni) is not None:
            print(f"  {ni:>8d}", end="")
            for t in thresholds:
                val = crossings[t].get(ni)
                if val is not None:
                    marker = " *" if val > n_half else ""
                    print(f"  {str(val) + marker:>6s}", end="")
                else:
                    print(f"  {'—':>6s}", end="")
            print()

    print(f"  (* = after distribution shift at sample {n_half})")
    print(f"  Final norms: {norms3.round(3)}")
    print(f"  Variance ratio: {ratio3*100:.1f}% of optimal top-{k3_conv}")

    # --- Test 4: Batch Sanger, zero-mean ---
    n_batch_samples = 2048 * 64
    batch_size = 64
    n_batches = n_batch_samples // batch_size
    print(f"\nTest 4: Batch Sanger, zero-mean ({n_batches} batches x {batch_size}, k_init={k_init})")
    X_batch, _, _ = make_data(d, n_batch_samples, seed=42)

    W4 = mx.array(np.random.RandomState(0).randn(k_init, d) * 0.01)
    ema_y2_4 = mx.zeros((k_init,))
    ema_input4 = mx.zeros((d,))
    ema_x2_4 = mx.array(0.0)
    step4 = mx.array(0)

    for i in range(0, n_batch_samples, batch_size):
        batch = mx.array(X_batch[i:i+batch_size])
        W4, ema_y2_4, ema_input4, ema_x2_4, step4, Y = sanger_step_batch(
            W4, batch, ema_y2_4, ema_input4, ema_x2_4, step4)
    mx.eval(W4, ema_y2_4)
    ratio4, k4_conv, k4_total = evaluate(W4, X_batch)
    print(f"  Grew to k={k4_total} ({k4_conv} converged)")
    print(f"  Variance ratio: {ratio4*100:.1f}% of optimal top-{k4_conv}")
    print(f"  Eigenvalue estimates: {np.array(ema_y2_4[:k4_conv]).round(2)}")

    # --- Test 5: Batch Sanger, offset ---
    print(f"\nTest 5: Batch Sanger, offset ({n_batches} batches x {batch_size}, k_init={k_init})")
    X_batch_off, _, _ = make_data(d, n_batch_samples, mean_offset=mean_offset)

    W5 = mx.array(np.random.RandomState(0).randn(k_init, d) * 0.01)
    ema_y2_5 = mx.zeros((k_init,))
    ema_input5 = mx.zeros((d,))
    ema_x2_5 = mx.array(0.0)
    step5 = mx.array(0)

    for i in range(0, n_batch_samples, batch_size):
        batch = mx.array(X_batch_off[i:i+batch_size])
        W5, ema_y2_5, ema_input5, ema_x2_5, step5, Y = sanger_step_batch(
            W5, batch, ema_y2_5, ema_input5, ema_x2_5, step5)
    mx.eval(W5, ema_y2_5)
    ratio5, k5_conv, k5_total = evaluate(W5, X_batch_off)
    print(f"  Grew to k={k5_total} ({k5_conv} converged)")
    print(f"  Variance ratio: {ratio5*100:.1f}% of optimal top-{k5_conv}")

    # --- Test 6: Batch V2-scale ---
    d6, n_frames = 32, 2000
    print(f"\nTest 6: Batch V2-scale ({n_frames} frames x 1764 x {d6}D, k_init={k_init})")
    X6_all, eig6, _ = make_data(d6, 1764 * n_frames, seed=77)

    W6 = mx.array(np.random.RandomState(0).randn(k_init, d6) * 0.01)
    ema_y2_6 = mx.zeros((k_init,))
    ema_input6 = mx.zeros((d6,))
    ema_x2_6 = mx.array(0.0)
    step6 = mx.array(0)

    for f in range(n_frames):
        batch = mx.array(X6_all[f*1764:(f+1)*1764])
        W6, ema_y2_6, ema_input6, ema_x2_6, step6, Y = sanger_step_batch(
            W6, batch, ema_y2_6, ema_input6, ema_x2_6, step6)
    mx.eval(W6, ema_y2_6)
    ratio6, k6_conv, k6_total = evaluate(W6, X6_all[:1764*50])
    norms6 = np.linalg.norm(np.array(W6), axis=1)
    print(f"  Grew to k={k6_total} ({k6_conv} converged)")
    print(f"  Variance ratio: {ratio6*100:.1f}% of optimal top-{k6_conv}")
    print(f"  Norms: {norms6.round(3)}")
    print(f"  Total samples: {1764*n_frames:,}")

    # --- Test 7: Dynamic growth diagnostic ---
    print(f"\nTest 7: Dynamic growth diagnostic (k_init={k_init}, 2048 samples)")
    W7 = mx.array(np.random.RandomState(0).randn(k_init, d) * 0.01)
    ema_y2_7 = mx.zeros((k_init,))
    ema_input7 = mx.zeros((d,))
    ema_x2_7 = mx.array(0.0)
    step7 = mx.array(0)

    X7, _, _ = make_data(d, 2048, seed=42)
    growth_log = []  # (sample, new_k)
    for i in range(len(X7)):
        k_before = W7.shape[0]
        x = mx.array(X7[i])
        W7, ema_y2_7, ema_input7, ema_x2_7, step7, y = sanger_step(
            W7, x, ema_y2_7, ema_input7, ema_x2_7, step7)
        if W7.shape[0] > k_before:
            growth_log.append((i+1, W7.shape[0]))
    mx.eval(W7, ema_y2_7)
    final_norms7 = np.linalg.norm(np.array(W7), axis=1)
    ratio7, k7_conv, k7_total = evaluate(W7, X7)

    print(f"  Growth events:")
    for sample, new_k in growth_log:
        print(f"    Sample {sample:>5d}: k -> {new_k}")
    print(f"  Final: k={k7_total} ({k7_conv} converged)")
    print(f"  Norms: {final_norms7.round(3)}")
    print(f"  Variance ratio: {ratio7*100:.1f}% of optimal top-{k7_conv}")
    converged_norms = final_norms7[final_norms7 > 0.5]
    norm_deviation = np.max(np.abs(converged_norms - 1.0)) if len(converged_norms) > 0 else 1.0

    # --- Speed benchmark (fixed-size, no dynamic, for raw speed) ---
    print("\n--- Speed Benchmark (dynamic=False, learn_threshold=None) ---")
    k_bench = 4
    W_bench = mx.array(np.random.randn(k_bench, d).astype(np.float32) * 0.01)
    ema_b = mx.zeros((k_bench,))
    ema_ib = mx.zeros((d,))
    ema_xb = mx.array(0.0)
    step_b = mx.array(0)
    x_bench = mx.array(np.random.randn(d).astype(np.float32))

    for _ in range(100):
        W_bench, ema_b, ema_ib, ema_xb, step_b, _ = sanger_step(
            W_bench, x_bench, ema_b, ema_ib, ema_xb, step_b,
            dynamic=False, learn_threshold=None)
    mx.eval(W_bench, ema_b, ema_ib)

    n_bench = 10000
    t0 = time.time()
    for _ in range(n_bench):
        W_bench, ema_b, ema_ib, ema_xb, step_b, _ = sanger_step(
            W_bench, x_bench, ema_b, ema_ib, ema_xb, step_b,
            dynamic=False, learn_threshold=None)
    mx.eval(W_bench, ema_b, ema_ib)
    t_online = time.time() - t0

    d_bb, k_bb = 32, 8
    W_bb = mx.array(np.random.randn(k_bb, d_bb).astype(np.float32) * 0.01)
    ema_bb = mx.zeros((k_bb,))
    ema_ibb = mx.zeros((d_bb,))
    ema_xbb = mx.array(0.0)
    step_bb = mx.array(0)
    X_bb = mx.array(np.random.randn(1764, d_bb).astype(np.float32))

    for _ in range(10):
        W_bb, ema_bb, ema_ibb, ema_xbb, step_bb, _ = sanger_step_batch(
            W_bb, X_bb, ema_bb, ema_ibb, ema_xbb, step_bb,
            dynamic=False, learn_threshold=None)
    mx.eval(W_bb, ema_bb, ema_ibb)

    n_batch_bench = 2000
    t0 = time.time()
    for _ in range(n_batch_bench):
        W_bb, ema_bb, ema_ibb, ema_xbb, step_bb, _ = sanger_step_batch(
            W_bb, X_bb, ema_bb, ema_ibb, ema_xbb, step_bb,
            dynamic=False, learn_threshold=None)
    mx.eval(W_bb, ema_bb, ema_ibb)
    t_batch = time.time() - t0

    print(f"  Online (16D->4):       {n_bench} steps in {t_online*1000:.0f}ms "
          f"({n_bench/t_online:.0f} steps/sec)")
    print(f"  Batch (1764x32->8):    {n_batch_bench} frames in {t_batch*1000:.0f}ms "
          f"({n_batch_bench/t_batch:.0f} frames/sec)")

    # --- Verdict ---
    print(f"\n{'='*60}")
    pass1 = ratio1 > 0.98
    pass2 = ratio2 > 0.98
    pass3 = ratio3 > 0.95
    pass4 = ratio4 > 0.99
    pass5 = ratio5 > 0.99
    pass6 = ratio6 > 0.98
    pass7 = norm_deviation < 0.05 and k7_conv >= 4
    print(f"  Test 1 (online, zero-mean):   {'PASS' if pass1 else 'FAIL'} — {ratio1*100:.1f}% (k: {k_init}->{k1_total}, {k1_conv} conv)")
    print(f"  Test 2 (online, offset):      {'PASS' if pass2 else 'FAIL'} — {ratio2*100:.1f}% (k: {k_init}->{k2_total}, {k2_conv} conv)")
    print(f"  Test 3 (online, shift):       {'PASS' if pass3 else 'FAIL'} — {ratio3*100:.1f}% (k: {k_init}->{k3_total}, {k3_conv} conv)")
    print(f"  Test 4 (batch, zero-mean):    {'PASS' if pass4 else 'FAIL'} — {ratio4*100:.1f}% (k: {k_init}->{k4_total}, {k4_conv} conv)")
    print(f"  Test 5 (batch, offset):       {'PASS' if pass5 else 'FAIL'} — {ratio5*100:.1f}% (k: {k_init}->{k5_total}, {k5_conv} conv)")
    print(f"  Test 6 (batch, V2-scale):     {'PASS' if pass6 else 'FAIL'} — {ratio6*100:.1f}% (k: {k_init}->{k6_total}, {k6_conv} conv)")
    print(f"  Test 7 (growth + norms):      {'PASS' if pass7 else 'FAIL'} — {k7_conv} converged, max norm dev {norm_deviation:.4f}")
    all_pass = pass1 and pass2 and pass3 and pass4 and pass5 and pass6 and pass7
    if all_pass:
        print(f"  ALL PASSED")
    print(f"{'='*60}")

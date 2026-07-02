# Outcome-Trained Advice and Holdout-Max Advice Distillation

## 0. One-line idea

Parallel sampling gives diverse solution traces. A verifier lets us identify which traces worked best. Instead of only keeping the best trace, we train an **advisor** to produce language guidance that makes the next round of parallel generation more likely to contain an even better trace.

The research program has two stages:

1. **First prove advice is an optimizable control surface.** Train an advice policy directly from the outcomes its advice causes.
2. **Then make advice optimization more sample-efficient.** Use within-batch contrasts and hidden local maxima to distill cheap supervised advice updates.

This keeps the core story simple:

```text
parallel samples -> verified rewards -> advice -> next parallel samples -> higher max
```

---

## 1. Problem setting

We have a problem `x`, a generator policy, an advisor policy, and a cheap verifier.

```text
Generator:  π_gen(y | x, c)
Advisor:    π_adv(c | x, A)
Verifier:   R(x, y) -> [0, 1]
Archive:    A = {(y_i, r_i)}
```

Here:

- `y` is a reasoning/solution trace.
- `c` is advice or a compact improvement memo.
- `A` is the current pool of generated traces and rewards.
- `R` is assumed cheap enough that we can score many candidates.

The objective is not average reward. The objective is best-of-`G` reward:

\[
J_G = \mathbb{E}\left[\max_{i \leq G} R(x, y_i)\right].
\]

The central question is:

> Can we train an advisor whose advice increases the next batch maximum, compared with simply sampling more from the generator?

---

## 2. Why parallel sampling?

Sequential reasoning can collapse into local solution modes. Parallel sampling preserves diverse attempts, which is important for discovery. This aligns with recent work arguing that parallel sampling can outperform sequential sampling under the same number of solutions because sequential sampling can reduce exploration.

The point is not merely to sample many times forever. The point is to ask:

> Once parallel sampling gives us a diverse batch, how do we convert that batch into a better next batch?

That conversion is the role of the advisor.

---

## 3. Related-work map

### Parallel sampling

Parallel sampling gives diversity and improves best-of-`G` by brute force. But it does not itself learn from the contrast between failed and successful traces.

Relevant work:

- [Understanding Performance Gap Between Parallel and Sequential Sampling in Large Reasoning Models](https://arxiv.org/abs/2604.05868)

### PDR: language-space improvement

Parallel-Distill-Refine uses the pattern:

```text
generate diverse drafts -> distill compact workspace -> refine conditioned on workspace
```

This supports the idea that a compact language artifact can act as an improvement operator for future generations.

Relevant work:

- [Rethinking Thinking Tokens: LLMs as Improvement Operators](https://arxiv.org/abs/2510.01123)

### Evolutionary/state-reuse methods

Evolutionary systems and state-reuse methods start from good states and mutate/evolve them. They use the verifier to keep better artifacts.

Relevant work:

- [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131)

### TTT-Discover

TTT-Discover is close in spirit because discovery cares about finding one great solution to the current problem, not maximizing average policy quality. It uses test-time training, state reuse, and an entropic/max-seeking objective to focus learning pressure on high-reward discoveries.

Relevant work:

- [Learning to Discover at Test Time](https://arxiv.org/abs/2601.16175)

### MAML / TTT-E2E analogy

The eventual version is meta-learning-like: train an initialization so that a small inner-loop update produces useful problem-specific improvement. But this should come after the simple advisor experiments work.

Relevant work:

- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)
- [End-to-End Test-Time Training for Long Context](https://arxiv.org/abs/2512.23675)

---

## 4. Core decomposition: generator and advisor

Instead of directly fine-tuning the generator at first, separate the system into two roles.

### Generator

The generator produces solution traces:

\[
y \sim \pi_\text{gen}(\cdot \mid x, c).
\]

The generator may be fixed in the early experiments.

### Advisor

The advisor produces advice:

\[
c \sim \pi_\text{adv}(\cdot \mid x, A).
\]

The advice is a compact language object that conditions the generator’s next batch.

The advice should be operational, not generic. A useful format:

```text
Current bottleneck:
...

What the stronger attempts are doing right:
...

Promising next direction:
...

Avoid:
...
```

The advisor is the main trainable object.

---

## 5. Stage ordering

The order matters. First establish that advice can be optimized by outcome. Only then use holdout-max distillation as a sample-efficiency improvement.

| Stage | Name | Question | Mechanism | Success criterion |
|---:|---|---|---|---|
| 0 | Parallel baseline | How strong is raw best-of-`G`? | No advice | Establish max-vs-budget curve |
| 1 | Advice interface | Does advice help at all? | Fixed/generated advice suffixes | Advice beats no-advice at equal rollout budget |
| 2 | Outcome-trained advice | Can we reinforce advice that causes high max reward? | Entropic/max objective over advice outcomes | Trained advisor improves best-of-`G` |
| 3 | Holdout-max distillation | Can within-batch contrasts make advice learning cheaper? | SFT on hidden-max advice data | Same improvement with fewer verifier calls |
| 4 | Distill + outcome train | Are the two signals complementary? | SFT warm start, then outcome training | Beats either alone |
| 5 | Recursive loop | Does it compound across rounds? | Repeat sample → advise → sample → update | Frontier improves faster than raw sampling |
| 6 | Meta-learned inner update | Can the advisor learn to adapt quickly? | MAML-like outer objective | Few-step inner updates reliably improve max |

---

# 6. Stage 0 — Parallel baseline

Purpose: establish the raw discovery curve.

```python
def parallel_baseline(problem, generator, verifier, G):
    traces = [generator.sample(problem, advice=None) for _ in range(G)]
    rewards = [verifier(problem, y) for y in traces]
    return max(rewards), list(zip(traces, rewards))
```

Report:

- best reward as a function of rollout budget,
- reward distribution,
- diversity of generated traces,
- frequency of new records.

This baseline is critical. The method must beat simply spending the same number of verifier calls on raw parallel sampling.

---

# 7. Stage 1 — Advice interface search

Before training the advisor, test whether advice is a useful lever at all.

Candidate interfaces:

1. **Single suffix:** one advice memo conditions all next rollouts.
2. **Multi-suffix:** sample `M` advice memos; each memo gets `G/M` rollouts.
3. **Critique-then-regenerate:** advisor critiques current pool; generator solves again.
4. **Agentic interaction:** generator may call advisor during solving.

Start simple. The strongest early candidates are probably:

```text
single suffix      = simplest
multi-suffix       = preserves more diversity
```

Pseudocode:

```python
def test_advice_interface(problem, generator, advisor, verifier, G, M):
    # initial pool
    A = []
    for _ in range(G):
        y = generator.sample(problem, advice=None)
        A.append((y, verifier(problem, y)))

    # produce advice
    memos = [advisor.sample(problem, archive=A) for _ in range(M)]

    # advised next rollouts
    advised = []
    for c in memos:
        for _ in range(G // M):
            y = generator.sample(problem, advice=c)
            advised.append((y, verifier(problem, y)))

    # control next rollouts
    control = []
    for _ in range(G):
        y = generator.sample(problem, advice=None)
        control.append((y, verifier(problem, y)))

    return max(r for _, r in advised), max(r for _, r in control)
```

Question:

\[
\max R(\text{advised rollouts}) > \max R(\text{control rollouts})?
\]

If advice cannot move the next-batch maximum, do not proceed to more complex training.

---

# 8. Stage 2 — Outcome-trained advice

This should come before holdout-max distillation.

The advisor samples advice candidates:

\[
c_1, \dots, c_M \sim \pi_\text{adv}(\cdot \mid x, A).
\]

Each advice candidate gets `n` generator rollouts:

\[
y_{j,1}, \dots, y_{j,n} \sim \pi_\text{gen}(\cdot \mid x, c_j).
\]

Define an advice score using an entropic/max-like objective:

\[
S(c_j) = \frac{1}{\beta}\log\sum_{\ell=1}^{n} \exp(\beta R(x,y_{j,\ell})).
\]

For large `β`, this approximates:

\[
S(c_j) \approx \max_{\ell} R(x,y_{j,\ell}).
\]

Then update the advisor toward high-scoring advice:

\[
\max_\phi \sum_j w_j \log \pi_\phi(c_j \mid x,A)
\]

where:

\[
w_j = \operatorname{softmax}(\alpha S(c_j)).
\]

Pseudocode:

```python
def outcome_train_advisor(problem, generator, advisor, verifier, archive, M, n, beta):
    advice_candidates = [advisor.sample(problem, archive) for _ in range(M)]
    scored_advice = []

    for c in advice_candidates:
        child_rewards = []
        for _ in range(n):
            y = generator.sample(problem, advice=c)
            r = verifier(problem, y)
            child_rewards.append(r)
            archive.append((y, r))

        S = entropic_score(child_rewards, beta=beta)
        scored_advice.append((c, S))

    advisor.update_weighted_sft(scored_advice)
    return advisor, archive
```

This stage tests the simplest optimization question:

> Can advice be trained directly by the best outcomes it causes?

If yes, advice is a real control surface.

---

# 9. Stage 3 — Holdout-Max Advice Distillation

Now introduce the sample-efficiency idea.

Outcome-trained advice can be expensive because every advice candidate must be tested with rollouts. Holdout-Max Advice Distillation tries to create cheaper supervised data from a single generated pool.

From archive `A`, sample a mini-batch of `g` traces:

\[
B = \{(y_1,r_1),\dots,(y_g,r_g)\}.
\]

Sort by reward:

\[
r_{(1)} \leq \dots \leq r_{(g)}.
\]

Visible set:

\[
V = \{y_{(1)},\dots,y_{(g-1)}\}.
\]

Hidden local max:

\[
h = y_{(g)}.
\]

The teacher sees both `V` and `h`, but writes advice for how to beat `V` without revealing `h`.

Training example:

\[
(x,V,R(V)) \rightarrow c.
\]

Pseudocode:

```python
def build_hmad_data(problem, archive, teacher, K, g):
    D = []

    for _ in range(K):
        batch = sample_minibatch(archive, size=g)
        if all_same_reward(batch):
            continue

        batch = sorted(batch, key=lambda z: z[1])
        visible = batch[:-1]
        hidden_best = batch[-1]

        memo = teacher.sample(
            prompt=hmad_teacher_prompt(
                problem=problem,
                visible=visible,
                hidden_best=hidden_best,
            )
        )

        D.append((hmad_student_input(problem, visible), memo))

    return D
```

Teacher instruction:

```text
You are given visible attempts and a hidden stronger attempt.
Write compact advice for how future attempts can beat the visible attempts.
Do not reveal, quote, or copy the hidden attempt.
Focus on failure modes, useful principles, and promising next directions.
```

Then train the advisor:

```python
D = build_hmad_data(problem, archive, teacher, K, g)
advisor_inner = sft(advisor_base, D, steps=inner_steps)
```

Deploy it through the best advice interface found in Stage 1:

```python
memos = [advisor_inner.sample(problem, archive) for _ in range(M)]
new_traces = generate_with_memos(generator, problem, memos, G)
```

This stage asks:

> Can hidden-max synthetic advice approximate useful outcome-trained advice with fewer verifier calls?

---

# 10. Stage 4 — Distillation + outcome training

If both Stage 2 and Stage 3 work, combine them.

Simple recipe:

```text
1. Build HMAD data from current archive.
2. SFT advisor on HMAD data.
3. Sample advice from the adapted advisor.
4. Test advice with generator rollouts.
5. Reinforce advice that caused high entropic/max child reward.
```

This gives two signals:

- **HMAD SFT:** cheap, dense, synthetic supervision.
- **Outcome training:** expensive, sparse, ground-truth correction.

The division of labor is clean:

```text
Distillation proposes better advice.
Outcome training selects advice that actually works.
```

---

# 11. Stage 5 — Recursive loop

Run the whole process repeatedly.

```python
def recursive_advised_discovery(problem, generator, advisor, teacher, verifier, T, G, M):
    archive = []

    # initial parallel pool
    for _ in range(G):
        y = generator.sample(problem, advice=None)
        archive.append((y, verifier(problem, y)))

    frontier = [max(r for _, r in archive)]

    for t in range(T):
        # optional direct outcome training
        advisor, archive = outcome_train_advisor(
            problem, generator, advisor, verifier, archive,
            M=M, n=G // M, beta=beta
        )

        # optional HMAD inner update
        D = build_hmad_data(problem, archive, teacher, K=K, g=g)
        advisor_inner = sft(advisor, D, steps=inner_steps)

        # next advised generation
        memos = [advisor_inner.sample(problem, archive) for _ in range(M)]
        for c in memos:
            for _ in range(G // M):
                y = generator.sample(problem, advice=c)
                archive.append((y, verifier(problem, y)))

        frontier.append(max(r for _, r in archive))

    return archive, frontier
```

Evaluation:

\[
\text{frontier}_t = \max_{(y,r) \in A_t} r.
\]

Compare against equal total verifier budget spent on raw parallel sampling.

---

# 12. Stage 6 — Holy grail: meta-learned inner update

Only after the above works.

Inner loop:

\[
\phi'_x = \operatorname{SFT}(\phi; \mathcal{D}_\text{HMAD}(x)).
\]

Outer loop:

\[
\max_\phi \mathbb{E}_{x}\left[
\frac{1}{\beta}\log
\mathbb{E}_{c \sim \pi_{\phi'_x}}
\exp\left(\beta S(c)\right)
\right]
\]

where `S(c)` is the entropic/max reward of generator rollouts conditioned on advice `c`.

Interpretation:

> Train the advisor initialization so that one small HMAD inner-loop update produces advice that improves the next batch maximum.

This is the MAML-like / TTT-E2E-like version. It is the eventual version, not the first implementation.

---

# 13. Metrics

Use metrics aligned with discovery, not average policy quality.

Primary:

```text
best reward at fixed verifier budget
```

Secondary:

```text
probability of improving over current best
expected improvement when improvement occurs
top-k reward average
reward distribution shift
trace diversity
advice diversity
verifier calls per new record
```

For continuous rewards:

\[
\Delta_t = \max R(A_{t+1}) - \max R(A_t).
\]

For binary rewards, use pass@budget, but continuous or graded rewards are preferable for studying recursive improvement.

---

# 14. Testbed desiderata

The best testbed should have:

1. **Cheap verification:** scoring a solution is much cheaper than generating it.
2. **Reward in `[0,1]`:** ideally continuous or at least graded.
3. **Many local improvements:** not just solved/unsolved.
4. **Diverse valid strategies:** parallel sampling should expose different approaches.
5. **Low reward hacking risk:** verifier should be hard to exploit superficially.
6. **Meaningful frontier:** there should be room to beat the current best repeatedly.

Good candidate families:

```text
algorithmic optimization
program synthesis with partial tests
code performance optimization
heuristic search problems
constraint satisfaction with graded scores
scientific design toy tasks
```

---

# 15. Failure modes and simple safeguards

## Generic advice

Failure:

```text
The advisor learns vague advice like "check your work" or "try a different approach."
```

Safeguard:

Compare against generic-advice baselines. Force memos to state bottleneck, preserved insight, next direction, and avoid list.

## Diversity collapse

Failure:

```text
One advice memo makes all next traces too similar.
```

Safeguard:

Use `M > 1` memos and split rollout budget across them.

## Local optimum

Failure:

```text
Advice reinforces a shallow trick that improves early but saturates.
```

Safeguard:

Keep some plain parallel exploration, or require multiple advice memos with distinct strategies.

## Rationalization

Failure:

```text
Teacher invents plausible but non-causal explanations for why the hidden max is better.
```

Safeguard:

Outcome training checks whether advice actually causes better child samples.

## Noisy outcome scores

Failure:

```text
A good memo looks bad because rollout sampling missed the tail.
```

Safeguard:

Do not hard-reject advice from one comparison. Use outcome rewards for soft weighting or later training data, not brittle binary decisions.

---

# 16. Clean thesis statement

Parallel sampling is useful because it preserves diverse attempts. But a discovery system should not only select the best attempt; it should learn how to make the next batch better.

The simplest path is:

```text
1. Treat advice as an action.
2. Score advice by the best solution it induces.
3. Train the advisor toward advice that raises the next-batch maximum.
4. Once this works, use holdout-max distillation to get cheaper supervised advice updates from within-batch contrasts.
```

The final system is:

```text
parallel generation
-> verification
-> outcome-trained advice
-> holdout-max advice distillation
-> advised generation
-> recursive frontier improvement
```

The key experimental claim is:

\[
\boxed{\text{Advice trained on max-seeking outcomes improves best-of-budget discovery over raw parallel sampling.}}
\]

The sample-efficiency claim is:

\[
\boxed{\text{Holdout-max distillation reduces the number of verifier rollouts needed to learn useful advice.}}
\]

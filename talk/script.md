# FlyAda — WRC SARA 2026 oral presentation script

**Paper:** WRC26_0024 · *FlyAda: Belief-State Adaptation for Diffusion Policies under Partial Observation*  
**Speaker:** Lingxue Lyu, University of Pennsylvania  
**Slot:** 9 min presentation + 3 min Q&A · video channel: 8–10 min  
**Deck:** `paper/FlyAda_WRC_SARA_2026_oral.pptx` — 11 content slides + Thank You (12 pages)

> **Delivery notes.** One question, one failure, one method, three results — 11 content slides, ≈ 9:08 at 130 words per minute. Cues below are derived from the word counts, not hand-written. Five slides carry video (3, 5, 8, 9, 10) and every one autoplays and loops on slide entry: **keep talking over it, do not wait for it to finish**. Bullets and takeaway bars fade in automatically — nothing needs clicking mid-slide.

> **There are no backup slides.** The Q&A material — anticipated questions, implementation detail, the Task B and probe numbers, and the full caveat list — is in the **Thank You slide's speaker notes**, where presenter view keeps it in front of you.

---

### Slide 1 — Title

*[0:00 – 0:14]   ·   31 words*

Good morning. I am Lingxue Lyu, from the University of Pennsylvania.

This paper is about when a small, online-updated latent actually helps an
action-chunk diffusion policy — and when it does not.

---

### Slide 2 — 1 · The question

*[0:14 – 0:59]   ·   96 words*

Let me start with the question, because the answer surprised us.

Action-chunk diffusion policies are a strong imitation-learning recipe for
continuous control. The common assumption is that they should execute poorly
when test-time dynamics shift, because the denoiser conditions on a single
observation.

Our prior work found that a small online-updated latent closed exactly that kind
of gap, and we set out to port the recipe to diffusion.

The first set of experiments did not show the gap we were trying to close. So
the question became sharper: when does a latent like this actually help?

---

### Slide 3 — 2 · The task

*[0:59 – 1:50]   ·   111 words   ·   **video, autoplays + loops***

[VIDEO left, autoplays and loops]

The task, in one slide. A UAV flying to a goal at fifty hertz.

On the left, the policy running. Each re-plan it commits to a chunk of eight
actions — the dashed line — executes only the first four, then re-plans from
wherever it actually ended up.

Two things to hold on to. First, we perturb the dynamics: mass, drag, wind and
control delay, and the test sweep pushes well past the training range. Second,
and this is the one that matters — in the partial-observation setting the three
velocity dimensions are zeroed in what the policy sees. The simulator still
integrates the true velocity. Only the policy's view of it is gone.

---

### Slide 4 — 3 · Full observation: already robust

*[1:50 – 2:30]   ·   87 words*

First result, and it is the negative one.

Under full observation, vanilla diffusion is already robust. One point zero on
every cell of the conventional sweep — and still around ninety-five percent an
order of magnitude past the training range, at four times mass, ten times drag,
eight steps of delay.

Adding the adaptation latent on top gained nothing, and on the most extreme
cells it actually hurt, because the latent went off-distribution.

So the failure is not caused by dynamics shift itself. We had to look
elsewhere.

---

### Slide 5 — 4 · Hide velocity and it breaks

*[2:30 – 3:27]   ·   124 words   ·   **video, autoplays + loops***

[VIDEO left, autoplays and loops]

Here is where it breaks. Same seed, same backbone, same demonstrations. The only
difference between these two is whether velocity is in the observation.

For the first fifty steps they are indistinguishable — both accelerate toward
the goal correctly.

Now watch the speed panel. Blue, which can see velocity, brakes and stops inside
the capture ball. Orange, which cannot, actually gets closer in position — zero
point four metres against zero point five — but it arrives at one point four
five metres per second, through a one metre per second stop tolerance. It sails
straight through, and then oscillates for the rest of the episode.

It accelerates correctly. It just cannot tell when to brake.

Across the twelve-condition sweep that is zero point three percent success.

---

### Slide 6 — 5 · Why more history is not the answer

*[3:27 – 4:18]   ·   109 words*

The obvious fix is more observation history. We trained a three-frame stacking
baseline — same backbone, same demonstrations, same budget. It reaches one point
one percent.

Here is why. The history does contain the velocity — a heuristic numerical estimator can recover it. But look at the graph: two
demonstrations reach the same position at different speeds and they are followed by
the same demonstrated chunk. One output reduces the loss for both.

Frame stacking increases the histories the model can represent. It does not
change the supervision signal, so the velocity-discriminating direction is never
used.

The information is there; the objective never asks for it. That is the gap to
close.

---

### Slide 7 — 6 · FlyAda

*[4:18 – 5:25]   ·   146 words*

That tells us what to add: a per-step supervisory signal on a learned channel of
the conditioning vector.

FlyAda keeps the single-frame observation and the same diffusion backbone. The
only addition is the orange path — a small observer head, about seventy-three
thousand parameters, reading the previous state, the action, and the current
state. 


Its output updates a latent by an exponential moving average at every
environment step, and that latent is concatenated to the observation as extra
conditioning.

And the piece that closes the gap is the yellow box: a linear head decoding the
latent to velocity, trained against the true velocity from the demonstrations.
That target is privileged — available at training time, never used at
deployment. Without it, everything else identical, nominal success reaches about
three percent.

Three ingredients, nothing new on their own. The claim is that all three have to be
present together.

---

### Slide 8 — 7 · Main result

*[5:25 – 6:29]   ·   138 words   ·   **video, autoplays + loops***

[VIDEO right, autoplays and loops]

The main result. Across the partial-observation sweep FlyAda is at one point
zero on the entire training-range block, where both baselines sit at the floor.


It stays there extrapolating fifty to one hundred percent on mass and two
hundred percent on drag.

It degrades where you would expect: zero point nine five at two metres per
second of wind, zero at three. That is a budget limit, not a belief limit — a
five metre per second squared acceleration cap cannot hold station against a
three metre per second wind.

And it is not specific to a single goal. On the right, the same checkpoints on a
three-waypoint chain with no re-training — the baselines loop past the first
waypoint and never recover, FlyAda passes all three. 

One point zero nominal,
zero point nine three hard, against zero for both.

---

### Slide 9 — 8 · It transfers to a 6-DoF body

*[6:29 – 7:07]   ·   84 words   ·   **video, autoplays + loops***

[VIDEO left, autoplays and loops]

One more transfer check, rendered in MuJoCo: the same checkpoints on a six-DoF
quadrotor with dynamics and a five-hundred-hertz attitude controller under
a fifty-hertz policy. We re-tuned nothing.

And the same failure returns, now with gravity attached. Vanilla sinks, touches the ground, and drifts away. That happens on
eleven of twelve seeds. FlyAda parks on the goal in under eighty steps and holds
station.

Forty percent nominal, eighty-five percent hard, against zero for both baselines
on both.

---

### Slide 10 — 9 · What is doing the work

*[7:07 – 8:24]   ·   165 words   ·   **video, autoplays + loops***

[VIDEO top right, autoplays and loops]

So what is actually doing the work?

On the left, we freeze the FlyAda weights and toggle only the test-time update
rule. Pin the latent at zero: five percent. Freeze it after ten steps: two and a
half. Keep updating continuously: ten percent, and the mean final distance drops
from over three metres to zero point six eight. Same weights throughout —
continuous online updating is the load-bearing part.

On the right, what the latent contains. Black is the true velocity the policy
never sees; orange is what we decode from the latent, live. A linear probe
recovers velocity at R-squared zero point nine nine two.

Underneath, the same latents on their top two principal components, coloured by
speed — slow near the origin, fast on the periphery. The structure is velocity.

And a classifier from that latent to which of five perturbations it is flying
under reaches twenty-four percent, against twenty percent chance.

The latent is not a dynamics-regime label. It is the missing state channel.

---

### Slide 11 — 10 · Caveats and conclusion

*[8:24 – 9:08]   ·   96 words*

Two honest caveats. Everything here is simulation-only. And the auxiliary loss
needs a true-velocity target at training time — mild for velocity, but it does
not obviously extend to quantities you cannot instrument.

To conclude.

Diffusion policies are robust when they can see the state. They fail when a
state channel is hidden, and more history does not fix it, because the imitation
loss never supervises extracting that channel.

FlyAda fixes it by making a small latent explicitly learn the missing channel,
and by keeping that latent updated online.

Thank you — I am happy to take questions.

---

### Slide 12 — Thank you  ·  Q&A prep

Thank you — I am happy to take questions.

Everything below is Q&A material. It is in the notes rather than on slides so the
deck stays at ten content pages; presenter view keeps it in front of you.

════════════════════════════════════════════════════════════════════════
ANTICIPATED QUESTIONS
════════════════════════════════════════════════════════════════════════
ANTICIPATED QUESTIONS

Q: Isn't the observer just a Kalman filter / finite-difference estimator?
   Functionally close, and that is part of the point — a hand-coded
   finite-difference estimator does recover velocity here, so this is not an
   information problem. The finding is that the diffusion MSE objective gives the
   network no gradient that asks for it, so you have to supply one.

Q: Why EMA rather than additive accumulation, and why alpha = 0.3?
   Additive accumulation was our first attempt: ||z_t|| grew unboundedly over the
   80-step rollouts, the conditioning went off-distribution, and nominal success
   stalled around 7% even at 25 epochs. EMA bounds z_t to f_phi's tanh output
   range. Alpha 0.1 and 0.5 also worked, but less well. Not exhaustively swept.

Q: Why does success jump so abruptly during training?
   Rollout success goes 0 -> 93% once the velocity-decoding loss falls below
   about 0.17, around epoch 14, and locks in at 100% by epoch 19. The transition
   is sharp; we did not look further into why.

Q: Why is FlyAda worse than vanilla on the full-observation extrapolation cells?
   The latent goes off-distribution there. Under full observation the policy has
   no missing channel to recover, so the latent contributes noise rather than
   information, and at 4x mass or 3 m/s wind that costs something.

Q: Isn't 0.100 vs 0.050 in the ablation within noise at 40 seeds?
   On its own, yes — Wilson half-widths near p=0.1 at n=40 are wide. The paper
   flags this in-text and reads Table III as a relative ordering; the mean final
   distance (0.68 m vs 1.60 and 3.61) separates the variants far more cleanly
   than the success rate does, and points the same way.

Q: Any real hardware?
   No — simulation-only, and we flag it as a caveat.

════════════════════════════════════════════════════════════════════════
IMPLEMENTATION DETAIL
════════════════════════════════════════════════════════════════════════
Denoiser: 1D temporal UNet, 3.98 M params, FiLM conditioning on a sinusoidal
timestep embedding plus the (state, latent) embedding. Hidden 128. DDPM training
with T=50, DDIM sampling with 20 steps. Horizon H=8, execute K=4.

Observer f_phi: two hidden layers of 128, Mish, tanh output, latent dim 32,
about 73 K params. EMA alpha 0.3. Auxiliary loss weight lambda 5.

Data: a Stable-Baselines3 PPO teacher on full state (100% on a 60-seed
deterministic eval) provides 50,067 nominal transitions and 50,023 under
perturbations sampled from the training-range sweep. All diffusion-based
policies in the paper share this pool.

Training cost: vanilla and frame-stack, 30 epochs at batch 128, about 3 min on a
single RTX 5080. FlyAda, 25 epochs at batch 32, about 8 min, because each step
rolls the latent along 80-step trajectories.

Inference: the observer is one forward pass of 73 K params per env step against
a 3.98 M denoiser sampled 20 times. The sampler is the bottleneck, not the
observer.

════════════════════════════════════════════════════════════════════════
LONG HORIZON — TASK B, THE NUMBERS BEHIND THE VIDEO
════════════════════════════════════════════════════════════════════════
Task B: three sequential waypoints, 2.5-4.5 m apart, 800-step budget. No
retraining — g_rel and d_goal in the observation simply advance to the next
waypoint when the current one is reached, so a single-goal policy transfers by
construction.

Task success (all three waypoints): FlyAda 1.00 nominal, 0.93 hard; vanilla and
frame-stack 0.00 on both, plateauing at 2-11% waypoint completion.

On the Extreme condition all three are at floor on strict success, so the paper
separates them by waypoint fraction and tracking error instead: FlyAda 0.12 WP
fraction and 1.07/1.31/3.76 m mean/RMSE/max tracking error, against 0.00 and
3.0-3.3/5.0-5.1 m for both baselines.

════════════════════════════════════════════════════════════════════════
WHAT z ENCODES — THE PROBE IN FULL
════════════════════════════════════════════════════════════════════════
Rolled FlyAda on 25 seeds each across nominal plus four single-axis
perturbations, recording (z_t, v_true) at every env step — about 10.5 K pairs.

Linear probe z -> v on a 70/30 split: R^2 = 0.992 on the held-out half
(0.99/0.99/0.99 per axis). The decoder head itself sits at R^2 = 0.987,
RMSE 0.20 m/s. So z_t is, to a good approximation, an affine map of the true
velocity — unsurprising, since dec_phi is a single linear layer and the joint
training optimises for exactly that.

Classifier z -> which of the five conditions: 24.2% against 20% chance. The
latent specialises to its supervisory target and gives up almost no capacity to
identifying the dynamics regime.

Smoothness: mean ||z_{t+1} - z_t|| is 0.12 and sits in [0.10, 0.12] across all
five conditions — perturbed conditions do not make z_t noisier, consistent with
the EMA bound being the dominant regulariser.

PCA on z_t, top two components, 52.6% of variance (29.3 + 23.3).

Left: coloured by ||v||. Low-speed points cluster near the origin, high-speed
points sit on the periphery, with the radial direction roughly tracking velocity
magnitude.

Right: per-condition centroid and 1-sigma ellipse over the same projection. The
five conditions superpose almost exactly — which is the same story the 24.2%
classifier tells, in a picture.

════════════════════════════════════════════════════════════════════════
FULL CAVEATS — beyond the two on slide 11
════════════════════════════════════════════════════════════════════════
1. Privileged target. L_vel needs true velocity at training time. Mild for
   velocity — free to log even when withheld at deployment — but it does not
   obviously extend to quantities that are hard to instrument even in
   simulation, like drag or mass, where hiding a *parameter* needs a different
   supervisory target than hiding a *state channel*.

2. Simulation only. A physical deployment needs the 20 DDIM steps to clear the
   80 ms re-plan window (K=4 at 50 Hz) on embedded compute, and would expose the
   observer's raw-position input to unmodelled sensor noise.

3. No recurrent baseline. Recurrent encoders are the standard alternative and
   would have been a reasonable choice. We used the explicit observer because it
   appends to the existing conditioning with no architectural change, and
   because an explicit latent makes the same-weights ablation and the linear
   probe straightforward design toggles rather than probes on an opaque hidden
   state. We did not run the head-to-head; it is a fair follow-up.

4. Generality untested. Whether this extends to higher-DoF dynamics, or to
   parameters less directly observable than velocity, we did not test.

---

## Timing budget

| Slide | Section | Words | Start | ≈ sec |
|---:|---|---:|---:|---:|
| 1 | Title | 31 | 0:00 | 14 |
| 2 | 1 · The question | 96 | 0:14 | 44 |
| 3 | 2 · The task | 111 | 0:59 | 51 |
| 4 | 3 · Full observation: already robust | 87 | 1:50 | 40 |
| 5 | 4 · Hide velocity and it breaks | 124 | 2:30 | 57 |
| 6 | 5 · Why more history is not the answer | 109 | 3:27 | 50 |
| 7 | 6 · FlyAda | 146 | 4:18 | 67 |
| 8 | 7 · Main result | 138 | 5:25 | 64 |
| 9 | 8 · It transfers to a 6-DoF body | 84 | 6:29 | 39 |
| 10 | 9 · What is doing the work | 165 | 7:07 | 76 |
| 11 | 10 · Caveats and conclusion | 96 | 8:24 | 44 |

**Total: 1187 words ≈ 9:08 at 130 wpm.**

Faster delivery (150 wpm) lands near 7:55; slower (115 wpm) near 10:19. The 9-minute slot and the 8–10 minute video window are both comfortable at the scripted pace — there is roughly a minute of headroom for pauses on the videos and for the questions that always take longer than expected.

The Thank You slide's notes add 1013 words of prepared Q&A answers that are not spoken unless asked.

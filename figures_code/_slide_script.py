"""Speaker script for the WRC SARA 2026 oral presentation (paper WRC26_0024).

One entry per slide. These strings are written into the PowerPoint speaker-notes
pane by paper/_make_slide_deck.py, and are exported to Markdown with a timing
table by paper/_make_slide_script_md.py.

Structure: a 10-slide main line — one question, one failure, one method, three
results — followed by a Thank You slide and a backup section that is never shown
unless a question calls for it. Backup slides carry their answer in the notes
rather than narration.

Timing target: about 9:00 of speech. The session allots 9 min + 3 min Q&A, and
the video channel asks for 8-10 min, so one script serves both. Written at
roughly 130 words per minute — an unhurried, clearly-articulated pace.

Do NOT hand-write time cues into the text. `timings()` below derives them from
the word counts, so editing a sentence re-times the rest of the talk instead of
leaving a stale marker behind. Backup slides are excluded from the clock.
"""
import re

WPM = 130.0

# (script key, slide number in the deck, title, is_backup)
ORDER = [
    ("title",      1,  "Title", False),
    ("question",   2,  "1 · The question", False),
    ("setup",      3,  "2 · The task", False),
    ("fullobs",    4,  "3 · Full observation: already robust", False),
    ("partial",    5,  "4 · Hide velocity and it breaks", False),
    ("why",        6,  "5 · Why more history is not the answer", False),
    ("method",     7,  "6 · FlyAda", False),
    ("results",    8,  "7 · Main result", False),
    ("mujoco",     9,  "8 · It transfers to a 6-DoF body", False),
    ("mechanism",  10, "9 · What is doing the work", False),
    ("conclusion", 11, "10 · Caveats and conclusion", False),
    ("thanks",     12, "Thank you  ·  Q&A prep in the notes", False),
]


def spoken_words(text: str) -> int:
    """Words a presenter would actually say — stage cues and Q&A prep excluded."""
    body = []
    for line in text.strip().split("\n"):
        s = line.strip()
        if not s or s.startswith("[") or s.startswith("---") or s.startswith("Q:"):
            continue
        body.append(s)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-]*", " ".join(body)))


def mmss(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def timings(wpm: float = WPM):
    """[(key, slide_no, title, words, start_s, end_s, is_backup)].

    Only the main line advances the clock; the Thank You slide and everything in
    backup are outside the 9 minutes.
    """
    out, clock = [], 0.0
    for key, num, title, backup in ORDER:
        w = spoken_words(SCRIPT[key])
        on_clock = not backup and key != "thanks"
        dur = w / wpm * 60.0 if on_clock else 0.0
        out.append((key, num, title, w, clock, clock + dur, backup))
        clock += dur
    return out


SCRIPT = {

# ================================================================ MAIN LINE ===

"title": """
Good morning. I am Lingxue Lyu, from the University of Pennsylvania.

This paper is about when a small, online-updated latent actually helps an
action-chunk diffusion policy — and when it does not.
""",

"question": """
Let me start with the question, because the answer surprised us.

Action-chunk diffusion policies are a strong imitation-learning recipe for
continuous control. The common assumption is that they should execute poorly
when test-time dynamics shift, because the denoiser conditions on a single
observation.

Our prior work found that a small online-updated latent closed exactly that kind
of gap, and we set out to port the recipe to diffusion.

The first set of experiments did not show the gap we were trying to close. So
the question became sharper: when does a latent like this actually help?
""",

"setup": """[VIDEO left, autoplays and loops]

The task, in one slide. A UAV flying to a goal at fifty hertz.

On the left, the policy running. Each re-plan it commits to a chunk of eight
actions — the dashed line — executes only the first four, then re-plans from
wherever it actually ended up.

Two things to hold on to. First, we perturb the dynamics: mass, drag, wind and
control delay, and the test sweep pushes well past the training range. Second,
and this is the one that matters — in the partial-observation setting the three
velocity dimensions are zeroed in what the policy sees. The simulator still
integrates the true velocity. Only the policy's view of it is gone.
""",

"fullobs": """
First result, and it is the negative one.

Under full observation, vanilla diffusion is already robust. One point zero on
every cell of the conventional sweep — and still around ninety-five percent an
order of magnitude past the training range, at four times mass, ten times drag,
eight steps of delay.

Adding the adaptation latent on top gained nothing, and on the most extreme
cells it actually hurt, because the latent went off-distribution.

So the failure is not caused by dynamics shift itself. We had to look
elsewhere.
""",

"partial": """[VIDEO left, autoplays and loops]

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
""",

"why": """
The obvious fix is more observation history. We trained a three-frame stacking
baseline — same backbone, same demonstrations, same budget. It reaches one point
one percent.

Here is why. The history does contain the velocity — a hand-coded
finite-difference estimator recovers it fine. But look at the diagram: two
demonstrations reach the same position at different speeds and are followed by
the same demonstrated chunk. One output minimises the loss for both.

Frame stacking enlarges the histories the model can represent. It does not
change the supervisory signal, so the velocity-discriminating direction is never
exercised.

The information is there; the objective never asks for it. That is the gap to
close.
""",

"method": """
That tells us what to add: a per-step supervisory signal on a learned channel of
the conditioning vector.

FlyAda keeps the single-frame observation and the same diffusion backbone. The
only addition is the orange path — a small observer head, about seventy-three
thousand parameters, reading the previous state, the action, and the current
state. Its output updates a latent by an exponential moving average at every
environment step, and that latent is concatenated to the observation as extra
conditioning.

And the piece that closes the gap is the yellow box: a linear head decoding the
latent to velocity, trained against the true velocity from the demonstrations.
That target is privileged — available at training time, never used at
deployment. Without it, everything else identical, nominal success reaches about
three percent.

Three ingredients, none new on its own. The claim is that all three have to be
present together.
""",

"results": """[VIDEO right, autoplays and loops]

The main result. Across the partial-observation sweep FlyAda is at one point
zero on the entire training-range block, where both baselines sit at the floor.
It stays there extrapolating fifty to one hundred percent on mass and two
hundred percent on drag.

It degrades where you would expect: zero point nine five at two metres per
second of wind, zero at three. That is a budget limit, not a belief limit — a
five metre per second squared acceleration cap cannot hold station against a
three metre per second wind.

And it is not specific to a single goal. On the right, the same checkpoints on a
three-waypoint chain with no retraining — the baselines loop past the first
waypoint and never recover, FlyAda threads all three. One point zero nominal,
zero point nine three hard, against zero for both.
""",

"mujoco": """[VIDEO left, autoplays and loops]

One more transfer check, rendered in MuJoCo: the same checkpoints on a six-DoF
quadrotor with rotor dynamics and a five-hundred-hertz attitude controller under
a fifty-hertz policy. We retuned nothing.

And the same failure returns, now with gravity attached. Vanilla cannot arrest
its own descent — it sinks, touches the ground, and drifts away. That happens on
eleven of twelve seeds. FlyAda parks on the goal in under eighty steps and holds
station.

Forty percent nominal, eighty-five percent hard, against zero for both baselines
on both.
""",

"mechanism": """[VIDEO top right, autoplays and loops]

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
""",

"conclusion": """
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
""",

"thanks": """Thank you — I am happy to take questions.

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
""",
}

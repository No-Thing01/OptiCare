# Hackathon solution architecture and MVP planning

**Created:** 9/5/2026 23:13:56  
**Updated:** 9/5/2026 23:27:32  
**Exported:** 9/5/2026 23:49:38  
**Link:** [https://claude.ai/chat/1b406aa0-3a02-4f0b-ac56-f3f45364f0d6](https://claude.ai/chat/1b406aa0-3a02-4f0b-ac56-f3f45364f0d6)  

## Assistant:

> 9/5/2026 23:27:32

> Finalizing tables, naming choices, and disclaimer placement.
>
> - Reading docx skill to know how to properly read the uploaded research document
> - Convert the uploaded docx research document to markdown for reading
> - Weighing OpenCV.js against custom C++ for faster delivery.
> - Smart India Hackathon 2026 PS 26038 diabetic retinopathy
> - Smart India Hackathon 2026 problem statement 26038 MATLAB Simulink
> - Researching MathWorks' involvement in SIH problem statements.
> - "26038" SIH 2026 problem statement
> - Searching for details on a specific problem statement.
> - sih.gov.in 2026 problem statements list AICTE MathWorks retinopathy
> - Building a table to map the tech stack clearly.
> - Finalizing tables, naming choices, and disclaimer placement.
> - **Done**

I read through the whole document. It's genuinely strong research — but it's scoped like a funded clinical R&D roadmap (U-Net segmentation, ResNet-101 training from scratch, SimEvents queueing simulation for 100,000 patients/year), not a 24–48 hour build. So my first job as your mentor is triage: pull out the two or three ideas in there that are *both* genuinely novel *and* buildable solo/small-team in your stack, and ruthlessly defer the rest.

Before architecting, I checked two things that change the plan: what SIH 2026 actually looks like right now, and who else is already in this space.

> Researching details about a healthcare AI platform's features.
>
> - MadhuNetrAI offline explainable AI confidence retinal screening
> - Researching details about a healthcare AI platform's features.
> - **Done**

> [Interactive Widget: dual_verification_dr_pipeline]

## 1. The Core Pitch

**NetraSetu** ("eye-bridge" — rename it, it's just a placeholder) cross-checks every AI diagnosis against an independent, training-free mathematical lesion detector *before* a doctor ever sees it, and only calls a case "AI-verified" when a trained neural network and classical eigenvalue math agree on where the disease actually is. The quality-and-trust check runs entirely offline in the browser, so a rural health worker gets instant, actionable feedback before the image ever needs signal to leave the village.

## 2. The Problem-Solver

Your doc's own argument is that the bottleneck has moved from *capturing* images to *interpreting* them, and interpretation fails in three specific, named ways. Each one maps directly to something NetraSetu does — not a generic "we added a heatmap" gesture:

| Documented drawback | How NetraSetu answers it |
|---|---|
| High ungradeable-image rate in field conditions | Client-side quality gate gives *actionable* feedback ("too dark — clean the lens," not just "rejected"), before upload — catches the problem at the point of capture, not after a wasted round trip |
| Uncalibrated overconfidence (ECE > 0.15 is common) | Real temperature scaling, *plus* a second, independent safety net: the agreement check below |
| Black-box mistrust / "explanation drift" (heatmap lighting up a lens flare instead of a real lesion) | Grad-CAM is cross-checked against a classical Hessian-eigenvalue blob detector run on the same image. If the AI's attention doesn't land where real lesion-like structures actually are, the case is auto-flagged instead of silently trusted |
| Sub-pixel microaneurysms destroyed by CNN downsampling | The Hessian detector runs on the full-resolution green channel, untouched by any pooling layer |

Two honesty checks worth doing before you commit to a pitch:

**Don't compete on raw accuracy.** A 2024–25 real-world study of Remidio's offline Medios-AI, run at NIO Super Specialty Hospital in Pune, found 84.6% sensitivity and 99.0% specificity for DR specifically, and 99.3% sensitivity for detecting any retinal disease overall. A hackathon fine-tune on a few thousand images cannot out-perform a validated clinical model, and a judge who knows this space will spot an overclaim instantly. Your honest, winnable edge is *hardware-agnostic + self-auditing*, not *more accurate*.

**Know the real competitor your doc doesn't mention.** MadhuNetrAI, built by Wadhwani AI with AIIMS Delhi and integrated into eSanjeevani, is already deployed across dozens of facilities and has screened tens of thousands of images. It performs image assessment, standardized grading, and referral triage for frontline workers. Nothing in its public description mentions an offline capture-side gate, calibrated confidence, or independent lesion verification — that's your gap. If a judge asks "how is this different from what the government's already piloting," you now have a sharp, specific answer instead of a shrug.

## 3. Tech Architecture

Three layers, mapped to where each skill is actually strongest — not spread thin everywhere:

| Layer | Component | Stack | Role |
|---|---|---|---|
| Frontend | Capture app (ASHA-facing) | HTML/CSS/JS + OpenCV.js | Instant offline quality check, plain-language recapture prompts (do this in Punjabi + English — cheap to add, given the doc's own reference site is a Punjab village, and it's a genuine empathy point with judges) |
| Frontend | Doctor dashboard | HTML/CSS/JS + Canvas | Grade, calibrated confidence, Grad-CAM + Hessian overlay, trust badge |
| Core engine A (your signature IP) | CLAHE + Hessian-eigenvalue lesion detector | C/C++ (OpenCV) | Classical, training-free sub-pixel candidate detection — no dataset needed, pure math |
| Core engine B | CNN grading + calibration + Grad-CAM | Python (TensorFlow/PyTorch) + Flask/FastAPI | Transfer-learned binary classifier, temperature scaling, explainability, orchestration |
| Verification (optional) | Validation script | MATLAB (Image Processing Toolbox) | See compliance note below |

Here's the actual data flow:

The OpenCV.js gate uses the *prebuilt* WASM binary — no compilation toolchain risk. Your hand-written C++ is the real differentiator, and the safest way to ship it under time pressure is as a small **native binary** the Python backend calls via `subprocess`, not a WASM cross-compile you're debugging at 3am. Two sketches to prove this is real, not hand-wavy:

```cpp
// Hessian eigenvalue blob score — candidate microaneurysm/lesion detector
// input: single-channel green plane after CLAHE, CV_32F
cv::GaussianBlur(green, green, cv::Size(0, 0), sigma);   // scale-space smoothing
cv::Sobel(green, gxx, CV_32F, 2, 0, 3);
cv::Sobel(green, gyy, CV_32F, 0, 2, 3);
cv::Sobel(green, gxy, CV_32F, 1, 1, 3);

for (int y = 0; y < green.rows; y++) for (int x = 0; x < green.cols; x++) {
    float a = gxx.at<float>(y,x), b = gxy.at<float>(y,x), c = gyy.at<float>(y,x);
    float tr = a + c, det = a*c - b*b;
    float disc = std::sqrt(std::max(0.f, tr*tr - 4*det));
    float l1 = (tr + disc) / 2.f, l2 = (tr - disc) / 2.f;      // closed-form eigenvalues
    if (l1 < 0 && l2 < 0)                                       // isotropic dark blob
        blobScore.at<float>(y,x) = std::min(std::abs(l1), std::abs(l2));
}
```

```python
# Temperature scaling — fit ONCE on a held-out validation split, ~15 lines
T = torch.nn.Parameter(torch.ones(1) * 1.5)
optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
def closure():
    optimizer.zero_grad()
    loss = F.cross_entropy(val_logits / T, val_labels)
    loss.backward()
    return loss
optimizer.step(closure)
# inference: calibrated_probs = softmax(logits / T)
```

**Datasets** (both are exactly the ones your own doc flags as strategically relevant): fine-tune the CNN (EfficientNet-B0 or MobileNetV2, transfer learning, frozen base) on **APTOS 2019** for binary Referable-vs-Not classification — simpler and more robust on a small hackathon fine-tune than the full 5-class scale, and it's literally the clinical threshold your doc names as the critical target. Use **IDRiD**'s pixel-level lesion masks — the Indian-demographic dataset — to *validate* the Hessian detector's recall against real ground truth. That gives you a genuine, defensible number ("our classical detector recalls X% of annotated microaneurysms") instead of an unverifiable claim.

**MATLAB compliance check, before you lock the stack:** MathWorks has partnered with SIH since 2019, and in past editions has sponsored a specific numbered problem statement that requires the winning build to actually use MATLAB, Simulink, and Simscape. Your research doc leans unusually hard on named MATLAB Toolbox functions (`adapthisteq`, `gradCAM`, `trainnet`, `SimEvents`) — that's a strong signal PS 26038 might be this year's MathWorks track, where MATLAB could be a judged requirement, not a nice-to-have. I couldn't pull the exact PS text to confirm this — check your PS page on sih.gov.in before finalizing. If it *is* mandatory, don't try to run your live demo through MATLAB; instead, write a small standalone MATLAB script that reproduces the CLAHE + Hessian math and cross-validates it against your C++ output on the IDRiD set. That satisfies "we used MATLAB meaningfully" honestly, without betting your live demo on the tool you're least fluent in.

## 4. Hackathon Execution Plan

SIH 2026 launched August 21 with 226 problem statements, requires teams of exactly six, and the actual grand finale is a 36-hour non-stop window in December — so this plan fits comfortably inside either that or the 24–48h you mentioned, with slack to spare. Assign 2 people per stack layer and run all three in parallel from hour one.

**Hours 0–3 — Setup.** Download APTOS 2019 + IDRiD (check your PS rules, but pre-fetching data is normally fine before the clock starts). Scaffold repo, wireframe both UIs, confirm the MATLAB question above.

**Hours 3–10 — Parallel builds.** ML track: kick off EfficientNet-B0 transfer-learning fine-tune on APTOS (Colab T4, a couple hours in the background). C++ track: build and threshold-tune the quality metrics on real good/blurry/dark samples, then the Hessian detector, validated against a handful of IDRiD masks. Web track: capture UI with the OpenCV.js gate wired up, and a dashboard skeleton with placeholder data.

**Hours 10–20 — Integration.** Wrap the trained model + temperature scaling + Grad-CAM into a Flask endpoint. Compile the C++ detector, call it via `subprocess`, compute the agreement score. Wire frontend → backend end to end.

**Hours 20–28 — Robustness pass.** Throw messy/rotated/dark test images at the quality gate until thresholds feel right. Add the Punjabi/Hindi recapture strings. Style the dashboard for a "this looks like real medical software" feel, not a hackathon toy.

**Hours 28–34 — Numbers and deck.** Report sensitivity/specificity/F1 on your held-out split — *not* raw accuracy, echoing your own doc's point about the accuracy fallacy in class-imbalanced screening — and be upfront in the deck that this is a proof-of-concept pipeline, not a clinically validated device. That intellectual honesty is exactly what a judge who's read the same literature you have will respect.

**What NOT to build:** full 5-class ICDR grading, U-Net anatomical segmentation, and the SimEvents 100,000-patient queue simulation. All three are real and valuable, but each is days of work for near-zero demo payoff relative to the trust-layer story — put them on a "roadmap" slide, don't burn hours on them.

**Hours 34+ — Rehearse, and record a backup video of the full demo.** Hackathon wifi fails; a recorded fallback has saved more finals than any feature.

## 5. The Winning Edge

Don't lead the demo with an accuracy number — lead with a **live disagreement**. Stage two cases side by side: one where Grad-CAM and the Hessian blobs land on the same spot (green "Verified" badge, sub-15-second doctor review), and one where you deliberately feed an image where the CNN's attention drifts to an artifact with no real blob underneath it — and show the system catch its *own* AI being wrong in real time, auto-routing to "Needs review" instead of silently trusting the heatmap. That ten-second moment visually proves calibration, explainability, and trust all at once — which is a far stronger demo than any single benchmark number, and it's the exact clinical-trust problem your entire research document is built around.

---

Once you've locked the MATLAB question and picked Flask vs. FastAPI, I can help scaffold the actual code — the Hessian detector or the Flask endpoint are good places to start.

> Used tool
>
> - Generating recommend_claude_apps...
> - **Done**

Good luck with the build — this is a genuinely strong problem to tackle, and the trust-gate angle gives you something real to say when judges ask "why should we believe this."

---

**Sources:**

- [Creating a retinal image database to develop an automated screening tool for diabetic retinopathy in India](https://digitalcommons.isical.ac.in/journal-articles/5297)
- [DrRetina — AI Diabetic Retinopathy Agent](https://lablab.ai/submissions/zbnb0q61v5mrt68l16cxwbtp)
- [India launches AI-driven community screening for diabetic retinopathy](https://ianslive.in/india-launches-ai-driven-community-screening-for-diabetic-retinopathy--20251217112431)
- [Wadhwani AI and AIIMS’ MadhuNetrAI wins top recognition at India AI Impact Summit 2026](https://biovoicenews.com/?p=65320)
- [NHA Organizes Federated Intelligence Hackathon for Healthcare at IIT Kanpur | 27 Jan 2026](https://www.drishtiias.com/state-pcs-current-affairs/nha-organizes-federated-intelligence-hackathon-for-healthcare-at-iit-kanpur/print_manually)
- [wadhwani ai and aiims win recognition for ai powered diabetic retinopathy screening solution at india ai impact summit 2026](https://biospectrumindia.com/news/16/27354/wadhwani-ai-and-aiims-win-recognition-for-ai-powered-diabetic-retinopathy-screening-solution-at-india-ai-impact-summit-2026.html)
- [Sunday, April 12, 2026](https://biovoicenews.com/tag/retinopathy/)
- [teenage whiz kid invents an ai system to diagnose her grandfathers eye disease](https://spectrum.ieee.org/the-human-os/biomedical/diagnostics/teenage-whiz-kid-invents-an-ai-system-to-diagnose-her-grandfathers-eye-disease)
- [www.biospectrumindia.com](https://www.biospectrumindia.com/article/pdf/27354)
- [blogs.reskilll.com](https://blogs.reskilll.com/?p=236)
- [Innovation Meets Excellence: Solar Masters’ Winning Journey at Smart India Hackathon 2024](https://blogs.mathworks.com/student-lounge/?p=12588)
- [student lounge](https://blogs.mathworks.com/student-lounge/?p=13393)
- [blogs.mathworks.com](https://blogs.mathworks.com/student-lounge/wp-json/wp/v2/posts/12588)
- [blogs.mathworks.com](https://blogs.mathworks.com/student-lounge/2026/?from=jp)
- [MUCLecture 2023 101514867](https://uomus.edu.iq/img/lectures21/MUCLecture_2023_101514867.pdf)
- [www.cet.ac.in](https://www.cet.ac.in/?p=14821)
- [value added programme exploring matlab simulink 0](https://vit.ac.in/school-electrical-engineering-select/value-added-programme-exploring-matlab-simulink-0)
- [COURSE MODULE  MATLAB 2026](https://mail.eee.mait.ac.in/images/pdf/Module_development/COURSE_MODULE__MATLAB_2026.pdf)
- [www.sierraic.com](https://www.sierraic.com/26)
- [Épp megvették: 85492](https://index.hu/tech/numbers/?rnd=1348)
- [www.suse.com](https://www.suse.com/de-de/security/cve/index.html)
- [www.suse.com](https://www.suse.com/ja-jp/security/cve/index.html)
- [www.sierraic.com](https://www.sierraic.com/260P)
- [HearingsAndOrders\HearingFiles\2026\2026 02\Applications\Docket 115 to 200 2026](https://bogfiles.dnrc.mt.gov//HearingsAndOrders\HearingFiles\2026\2026_02\Applications\Docket 115 to 200-2026.pdf)
- [2026-01-29 15:02:36 -05:00](https://upd.dev/anomalyco/opencode/blame/branch/complete-client-dev/script/beta.ts)
- [2026-02-10 13:06:18 -08:00](https://upd.dev/DKB/zig/blame/branch/restricted/lib/c/math.zig)
- [2026-07-03 00:19:11 -05:00](https://upd.dev/anomalyco/opencode/blame/branch/dev/packages/codemode/test/stdlib.test.ts)
- [0% found this document useful (0 votes)](https://www.scribd.com/document/912761306/SIH-2025-Problem-Statements)
- [SIH 2026 Problem Statements: How to Choose the Right One for Your Team](https://blogs.reskilll.com/sih-2026-problem-statements-how-to-choose-right-one-team/)
- [SIH 2026 Problem Statements: How to Choose the Right One for Your Team](https://blogs.reskilll.com/?p=237)
- [How to Select and Validate Your SIH 2026 Problem Statement: A Decision Framework](https://blogs.reskilll.com/?p=442)
- [SIH 2026 Problem Statements Released: Categories, Themes, and Early Analysis](https://blogs.reskilll.com/?p=444)
- [SIH 2026 Problem Statements Released: Categories, Themes, and Early Analysis](https://blogs.reskilll.com/sih-2026-problem-statements-released-categories-themes-early-analysis/)
- [0% found this document useful (0 votes)](https://www.scribd.com/document/972931324/Problem-Statement)
- [aicte.gov.in](https://aicte.gov.in/sites/default/files/sih2020.pdf)
- [Uncertainty-inspired Open Set Learning for Retinal Anomaly   Identification](https://arxiv.org/pdf/2304.03981)
- [Sunday, May 10, 2026](https://biovoicenews.com/tag/madhunetrai/)
- [offline ai retinal screening smartphone accuracy 20 Aug 26 08 08](https://www.omnicuris.com/medshots/daily_updates/offline-ai-retinal-screening-smartphone-accuracy_20-Aug-26_08-08)
- [Explainable AI for Diabetic Retinopathy Screening: Enhancing Clinician Trust in Deep Learning Predictions](https://archives.christuniversity.in/items/show/25772)
- [offline smartphone ai accurately detects dr glaucoma and amd in real world study](https://www.ophthalmologytimes.com/view/offline-smartphone-ai-accurately-detects-dr-glaucoma-and-amd-in-real-world-study)
- [Optomed - Terveysteknologiayhtiö](https://forum.inderes.com/t/optomed-terveysteknologiayhtio/2389/4874)

Powered by Claude Exporter (https://www.ai-chat-exporter.net)

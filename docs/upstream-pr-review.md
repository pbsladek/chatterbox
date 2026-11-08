# Upstream open PR review

Reviewed the 92 open PRs in [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox/pulls)
on 2026-11-07 (America/Los_Angeles), against this repository's merged maintenance
baseline, `5043fba`. Screening covered every PR's title and changed-file
inventory. Candidate descriptions, patches, and overlapping fixes received a
code review; the decisions below do not imply line-by-line validation of every
large feature branch. The regression suite exercises the selected adaptations.

## Changes brought into this repository

Eight PRs supplied focused fixes. These are adaptations onto the maintained
code, preserving the local model/mirror loading APIs and dependency lock.
The original contributors and reviewed head revisions are recorded below.

| Upstream PR | Contributor | Reviewed head | Adaptation |
| --- | --- | --- | --- |
| [#580](https://github.com/resemble-ai/chatterbox/pull/580) | [JonRC](https://github.com/JonRC) | `d8783fc2edf6f1e1632813b733b84b1e49e65a07` | Initialize the Chinese converter on the first Chinese input and reuse it. |
| [#575](https://github.com/resemble-ai/chatterbox/pull/575) | [Notaloop763](https://github.com/Notaloop763) | `632cdd75bdefda8d1e16d6c14c5c3397dfc1d722` | Keep floating-point WAV output in examples and README snippets, avoiding PCM clipping. |
| [#572](https://github.com/resemble-ai/chatterbox/pull/572) | [ANDROIDHASSAN](https://github.com/ANDROIDHASSAN) | `b8121fb51d36a2da6ded273bf48d525a4ede3e62` | Respect Hindi and Arabic sentence punctuation, including the Arabic full stop. |
| [#554](https://github.com/resemble-ai/chatterbox/pull/554) | [li-lizhe](https://github.com/li-lizhe) | `3e54fd10f78777ea34bc990fb8f78e591fe4aae9` | Normalize device objects/indexed strings and stage checkpoints on CPU before explicit placement. |
| [#509](https://github.com/resemble-ai/chatterbox/pull/509) | [JuvGut](https://github.com/JuvGut) | `75c0bbf29340fac7776b3f53f97e5891d1238a84` | Cast audio and mel tensors to float32 before device transfer. |
| [#500](https://github.com/resemble-ai/chatterbox/pull/500) | [felixgwilliams](https://github.com/felixgwilliams) | `646ef7b309e074e7396737d0dcb652dda17290c2` | Use a Python float for loudness gain so NumPy 2 preserves float32 audio. |
| [#431](https://github.com/resemble-ai/chatterbox/pull/431) | [akx](https://github.com/akx) | `582829e3a1b87b763bdc7790d0251663e329c7ae` | Select an available device in both Turbo and Nano examples. |
| [#278](https://github.com/resemble-ai/chatterbox/pull/278) | [xenova](https://github.com/xenova) | `f0b477a74e70d38aad059f1a2e175c5c61bd60f2` | Use one decoding batch when CFG is zero; fix both English and multilingual callers. |

The device adaptation stages CUDA checkpoints on CPU as well, so the saved
GPU index cannot override the caller's requested placement. The zero-guidance
fix also updates the multilingual caller; applying the old T3-only patch
would leave that caller with a mismatched batch. Existing Chinese mapping and
segmenter files still load locally, with initialization delayed until Chinese
text is actually encoded.

Regression coverage exercises all four loading APIs with device objects,
indexed MPS fallback, checkpoint staging, actual tiny-backbone KV-cache
sampling with and without guidance, both high-level CFG callers, lazy Chinese
initialization, multilingual punctuation, and float32 conditioning under
NumPy 1 and 2. Float WAV export was checked with samples beyond full scale.
Real checkpoint synthesis checks cover default-guidance parity, zero-guidance
English/French, and Turbo voice conditioning with NumPy 2. CUDA and MPS
hardware performance is outside this validation.

## Follow-up candidates

The most promising feature work is configurable generation limits (#570),
long-text Turbo generation (#568), the public CLI (#538), and fixing URL
reference prompts in the demo (#569). Keep the existing default generation
budget until a replacement is checked for truncation across languages.
A public CLI should expose the mirror revision and standalone local loading
options already supported by this repository.

Alignment (#563), streaming (#528), CUDA graphs (#545), and MPS cache
flushing (#577) need dedicated accuracy, audio, or hardware measurements.
The model snapshots' vocabulary has `[en]` and `[zh]` but lacks `[te]` and
`[ml]`; adding a supported-language label alone is insufficient to establish
those languages' synthesis quality. The dummy-watermarker fallback (#469)
would change the watermarking behavior and is not included.

## Complete triage inventory

"Covered" means the behavior is already represented in this repository or
in the selected fix. "Defer" identifies possible later work, not approval
of an untested patch. Head revisions above identify the code actually used;
open PRs may change after this review.

| PR | Decision | Reason |
| --- | --- | --- |
| [#580](https://github.com/resemble-ai/chatterbox/pull/580) | Adapted here | Initialize the Chinese converter on the first Chinese input and reuse it. |
| [#577](https://github.com/resemble-ai/chatterbox/pull/577) | Defer: hardware validation | MPS cache flushing needs measurements on supported Macs before enabling it by default. |
| [#575](https://github.com/resemble-ai/chatterbox/pull/575) | Adapted here | Keep floating-point WAV output in examples and README snippets, avoiding PCM clipping. |
| [#573](https://github.com/resemble-ai/chatterbox/pull/573) | Already covered | Package version lookup already has a source-tree fallback. |
| [#572](https://github.com/resemble-ai/chatterbox/pull/572) | Adapted here | Respect Hindi and Arabic sentence punctuation, including the Arabic full stop. |
| [#570](https://github.com/resemble-ai/chatterbox/pull/570) | Defer: generation policy | The text-length heuristic changes default output limits; validate truncation across languages first. |
| [#569](https://github.com/resemble-ai/chatterbox/pull/569) | Useful follow-up | Fix URL prompts in a dedicated demo change with bounded downloads, cache handling, and app tests. |
| [#568](https://github.com/resemble-ai/chatterbox/pull/568) | Useful follow-up | Opt-in long-text chunking needs prosody/boundary fixtures and token-budget policy review. |
| [#565](https://github.com/resemble-ai/chatterbox/pull/565) | Defer: installation profile | CUDA/Blackwell/Windows instructions need a validated matching Torch profile rather than bypassing dependency checks. |
| [#563](https://github.com/resemble-ai/chatterbox/pull/563) | Useful follow-up | Optional alignment needs accuracy benchmarks and clear Turbo/Nano compatibility boundaries. |
| [#562](https://github.com/resemble-ai/chatterbox/pull/562) | Defer: separate application | API servers, databases, queues, and orchestration exceed the inference-library maintenance scope. |
| [#554](https://github.com/resemble-ai/chatterbox/pull/554) | Adapted here | Normalize device objects/indexed strings and stage checkpoints on CPU before explicit placement. |
| [#547](https://github.com/resemble-ai/chatterbox/pull/547) | Defer: packaging choice | Pinned spacy-pkuseg 1.0.2 already works in Python 3.14 CI; optional Chinese support needs separate coverage. |
| [#545](https://github.com/resemble-ai/chatterbox/pull/545) | Defer: hardware validation | CUDA graph/static-cache changes need a GPU compatibility and memory benchmark. |
| [#538](https://github.com/resemble-ai/chatterbox/pull/538) | Useful follow-up | A public CLI should support this repository's mirror revisions, local directories, and Nano. |
| [#528](https://github.com/resemble-ai/chatterbox/pull/528) | Defer: audio API | Streaming needs chunk-boundary, latency, and watermarking validation with real audio. |
| [#520](https://github.com/resemble-ai/chatterbox/pull/520) | Do not claim support | The archived multilingual vocabulary lacks [te]; transliteration and a registry entry do not establish Telugu support. |
| [#510](https://github.com/resemble-ai/chatterbox/pull/510) | Superseded | The current source no longer contains the alignment stream analyzer these patches repair. |
| [#509](https://github.com/resemble-ai/chatterbox/pull/509) | Adapted here | Cast audio and mel tensors to float32 before device transfer. |
| [#505](https://github.com/resemble-ai/chatterbox/pull/505) | Superseded | The current source no longer contains the alignment stream analyzer these patches repair. |
| [#500](https://github.com/resemble-ai/chatterbox/pull/500) | Adapted here | Use a Python float for loudness gain so NumPy 2 preserves float32 audio. |
| [#489](https://github.com/resemble-ai/chatterbox/pull/489) | Do not cherry-pick | Mixed historical platform/dependency changes; selected dtype and device fixes cover the reproducible defects. |
| [#486](https://github.com/resemble-ai/chatterbox/pull/486) | Do not cherry-pick | Broad dependency unpinning would replace the validated lock; review upgrades as tested version changes. |
| [#484](https://github.com/resemble-ai/chatterbox/pull/484) | Superseded | A Python-3.11-only requirement conflicts with the validated Python 3.10–3.14 baseline. |
| [#471](https://github.com/resemble-ai/chatterbox/pull/471) | Already covered | Cangjie mapping already comes from the local model snapshot. |
| [#469](https://github.com/resemble-ai/chatterbox/pull/469) | Do not adopt | The proposed fallback silently substitutes dummy watermarking for the tested real watermarker. |
| [#466](https://github.com/resemble-ai/chatterbox/pull/466) | Already covered | Current dependency/API baseline already uses Transformers 5.2 and Gradio 6.8. |
| [#465](https://github.com/resemble-ai/chatterbox/pull/465) | Already covered | Current dependency/API baseline already uses Transformers 5.2 and Gradio 6.8. |
| [#461](https://github.com/resemble-ai/chatterbox/pull/461) | Covered by selected fix | Duplicate CPU checkpoint-loading work; the normalized-device fix covers these paths. |
| [#460](https://github.com/resemble-ai/chatterbox/pull/460) | Defer: editorial | README spelling/example/formatting edits can be consolidated against the current fork guide. |
| [#459](https://github.com/resemble-ai/chatterbox/pull/459) | Already covered | SPDX metadata, license files, builds, and GitHub release CI already exist. |
| [#457](https://github.com/resemble-ai/chatterbox/pull/457) | Do not cherry-pick | The long-text fix is bundled with unrelated ASR, language, UI, and dependency changes. |
| [#455](https://github.com/resemble-ai/chatterbox/pull/455) | Superseded | The current source no longer contains the alignment stream analyzer these patches repair. |
| [#454](https://github.com/resemble-ai/chatterbox/pull/454) | Do not cherry-pick | Mixed historical platform/dependency changes; selected dtype and device fixes cover the reproducible defects. |
| [#432](https://github.com/resemble-ai/chatterbox/pull/432) | Do not cherry-pick | Mixed historical platform/dependency changes; selected dtype and device fixes cover the reproducible defects. |
| [#431](https://github.com/resemble-ai/chatterbox/pull/431) | Adapted here | Select an available device in both Turbo and Nano examples. |
| [#430](https://github.com/resemble-ai/chatterbox/pull/430) | Covered by selected fix | Duplicate CPU checkpoint-loading work; the normalized-device fix covers these paths. |
| [#429](https://github.com/resemble-ai/chatterbox/pull/429) | Do not cherry-pick | Broad dependency unpinning would replace the validated lock; review upgrades as tested version changes. |
| [#426](https://github.com/resemble-ai/chatterbox/pull/426) | Already covered | All variants already expose from_local for standalone checkpoint directories. |
| [#420](https://github.com/resemble-ai/chatterbox/pull/420) | Defer: documentation | Useful docstring expansion, but lower priority than reproduced runtime defects. |
| [#418](https://github.com/resemble-ai/chatterbox/pull/418) | Covered by selected fix | Duplicate CPU checkpoint-loading work; the normalized-device fix covers these paths. |
| [#417](https://github.com/resemble-ai/chatterbox/pull/417) | Defer: editorial | README spelling/example/formatting edits can be consolidated against the current fork guide. |
| [#414](https://github.com/resemble-ai/chatterbox/pull/414) | Defer: editorial | README spelling/example/formatting edits can be consolidated against the current fork guide. |
| [#409](https://github.com/resemble-ai/chatterbox/pull/409) | Defer: separate engine/assets | Malayalam fallback/ASR integration introduces different model assets and dependencies. |
| [#407](https://github.com/resemble-ai/chatterbox/pull/407) | Covered by selected fix | Duplicate CPU checkpoint-loading work; the normalized-device fix covers these paths. |
| [#404](https://github.com/resemble-ai/chatterbox/pull/404) | Superseded | Current NumPy markers and the Python 3.10–3.14 build matrix cover installation compatibility. |
| [#403](https://github.com/resemble-ai/chatterbox/pull/403) | Defer: extra model dependency | Hebrew diacritization needs separate Dicta model/dependency preservation and offline tests. |
| [#395](https://github.com/resemble-ai/chatterbox/pull/395) | Defer: installation profile | CUDA/Blackwell/Windows instructions need a validated matching Torch profile rather than bypassing dependency checks. |
| [#393](https://github.com/resemble-ai/chatterbox/pull/393) | Covered by selected fix | Duplicate CPU checkpoint-loading work; the normalized-device fix covers these paths. |
| [#391](https://github.com/resemble-ai/chatterbox/pull/391) | Superseded | Historical dependency pins/conflicts do not apply to the current locked graph. |
| [#387](https://github.com/resemble-ai/chatterbox/pull/387) | Defer: documentation | A language-extension guide should distinguish tokenizer edits from validated matching checkpoints. |
| [#382](https://github.com/resemble-ai/chatterbox/pull/382) | Defer: editorial | README spelling/example/formatting edits can be consolidated against the current fork guide. |
| [#379](https://github.com/resemble-ai/chatterbox/pull/379) | Covered by selected fix | Duplicate CPU checkpoint-loading work; the normalized-device fix covers these paths. |
| [#378](https://github.com/resemble-ai/chatterbox/pull/378) | Defer: demo hardware coverage | Gradio MPS selection needs an app smoke test on a supported Mac. |
| [#375](https://github.com/resemble-ai/chatterbox/pull/375) | Skip | Unrelated file or branding-only changes do not improve the maintained runtime. |
| [#364](https://github.com/resemble-ai/chatterbox/pull/364) | Covered by selected fix | Duplicate CPU checkpoint-loading work; the normalized-device fix covers these paths. |
| [#353](https://github.com/resemble-ai/chatterbox/pull/353) | Defer: broad model changes | Quantization and inference rewrites need quality, memory, and hardware benchmarks. |
| [#337](https://github.com/resemble-ai/chatterbox/pull/337) | Superseded | Historical dependency pins/conflicts do not apply to the current locked graph. |
| [#334](https://github.com/resemble-ai/chatterbox/pull/334) | Defer: editorial | README spelling/example/formatting edits can be consolidated against the current fork guide. |
| [#330](https://github.com/resemble-ai/chatterbox/pull/330) | Defer: editorial | README spelling/example/formatting edits can be consolidated against the current fork guide. |
| [#326](https://github.com/resemble-ai/chatterbox/pull/326) | Superseded | Current NumPy markers and the Python 3.10–3.14 build matrix cover installation compatibility. |
| [#315](https://github.com/resemble-ai/chatterbox/pull/315) | Defer: editorial | README spelling/example/formatting edits can be consolidated against the current fork guide. |
| [#285](https://github.com/resemble-ai/chatterbox/pull/285) | Defer: separate application | API servers, databases, queues, and orchestration exceed the inference-library maintenance scope. |
| [#283](https://github.com/resemble-ai/chatterbox/pull/283) | Defer: deployment target | A Docker image should use the maintained locks and explicit CPU/CUDA profiles. |
| [#279](https://github.com/resemble-ai/chatterbox/pull/279) | Defer: editorial | README spelling/example/formatting edits can be consolidated against the current fork guide. |
| [#278](https://github.com/resemble-ai/chatterbox/pull/278) | Adapted here | Use one decoding batch when CFG is zero; fix both English and multilingual callers. |
| [#276](https://github.com/resemble-ai/chatterbox/pull/276) | Covered by selected fix | Duplicate CPU checkpoint-loading work; the normalized-device fix covers these paths. |
| [#266](https://github.com/resemble-ai/chatterbox/pull/266) | Defer: installation profile | CUDA/Blackwell/Windows instructions need a validated matching Torch profile rather than bypassing dependency checks. |
| [#263](https://github.com/resemble-ai/chatterbox/pull/263) | Already covered | Cangjie mapping already comes from the local model snapshot. |
| [#257](https://github.com/resemble-ai/chatterbox/pull/257) | Covered by selected fix | Duplicate CPU checkpoint-loading work; the normalized-device fix covers these paths. |
| [#224](https://github.com/resemble-ai/chatterbox/pull/224) | Defer: editorial | README spelling/example/formatting edits can be consolidated against the current fork guide. |
| [#223](https://github.com/resemble-ai/chatterbox/pull/223) | Defer: editorial | README spelling/example/formatting edits can be consolidated against the current fork guide. |
| [#222](https://github.com/resemble-ai/chatterbox/pull/222) | Defer: editorial | README spelling/example/formatting edits can be consolidated against the current fork guide. |
| [#206](https://github.com/resemble-ai/chatterbox/pull/206) | Defer: editorial | README spelling/example/formatting edits can be consolidated against the current fork guide. |
| [#204](https://github.com/resemble-ai/chatterbox/pull/204) | Defer: broad API rewrite | Batch/multi-sequence generation changes dozens of model files and needs its own correctness/quality suite. |
| [#200](https://github.com/resemble-ai/chatterbox/pull/200) | Defer: deployment target | A hosted Colab notebook is a separate install/runtime target from these wheel releases. |
| [#195](https://github.com/resemble-ai/chatterbox/pull/195) | Superseded | Historical dependency pins/conflicts do not apply to the current locked graph. |
| [#185](https://github.com/resemble-ai/chatterbox/pull/185) | Skip | Unrelated file or branding-only changes do not improve the maintained runtime. |
| [#182](https://github.com/resemble-ai/chatterbox/pull/182) | Defer: editorial | README spelling/example/formatting edits can be consolidated against the current fork guide. |
| [#180](https://github.com/resemble-ai/chatterbox/pull/180) | Defer: generation/editing feature | Long-text, pause-tag, and waveform editing policies need explicit behavior and audio quality fixtures. |
| [#178](https://github.com/resemble-ai/chatterbox/pull/178) | Defer: generation/editing feature | Long-text, pause-tag, and waveform editing policies need explicit behavior and audio quality fixtures. |
| [#169](https://github.com/resemble-ai/chatterbox/pull/169) | Do not cherry-pick | Broad dependency unpinning would replace the validated lock; review upgrades as tested version changes. |
| [#168](https://github.com/resemble-ai/chatterbox/pull/168) | Defer: editorial | README spelling/example/formatting edits can be consolidated against the current fork guide. |
| [#167](https://github.com/resemble-ai/chatterbox/pull/167) | Defer: editorial | README spelling/example/formatting edits can be consolidated against the current fork guide. |
| [#164](https://github.com/resemble-ai/chatterbox/pull/164) | Defer: generation/editing feature | Long-text, pause-tag, and waveform editing policies need explicit behavior and audio quality fixtures. |
| [#157](https://github.com/resemble-ai/chatterbox/pull/157) | Defer: editorial | README spelling/example/formatting edits can be consolidated against the current fork guide. |
| [#156](https://github.com/resemble-ai/chatterbox/pull/156) | Defer: editorial | README spelling/example/formatting edits can be consolidated against the current fork guide. |
| [#153](https://github.com/resemble-ai/chatterbox/pull/153) | Defer: editorial | README spelling/example/formatting edits can be consolidated against the current fork guide. |
| [#152](https://github.com/resemble-ai/chatterbox/pull/152) | Defer: editorial | README spelling/example/formatting edits can be consolidated against the current fork guide. |
| [#149](https://github.com/resemble-ai/chatterbox/pull/149) | Defer: editorial | README spelling/example/formatting edits can be consolidated against the current fork guide. |
| [#144](https://github.com/resemble-ai/chatterbox/pull/144) | Defer: UI redesign | The Gradio layout rewrite needs a separate application review. |
| [#103](https://github.com/resemble-ai/chatterbox/pull/103) | Do not cherry-pick | Mixed historical platform/dependency changes; selected dtype and device fixes cover the reproducible defects. |

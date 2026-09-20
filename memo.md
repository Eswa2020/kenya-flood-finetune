# Memo: Kenya Urban Flood Early-Warning Assistant

**To:** County Disaster Management Director
**From:** Esther Warimu Kamau
**Re:** Recommendation on the fine-tuned flood early-warning assistant

## Why now

The Kenya Meteorological Service Authority has forecast above-average rainfall across more than 80% of the country for the October–December 2026 short rains, driven by a very strong El Niño combined with a positive Indian Ocean Dipole. Over 2 million people across 46 of Kenya's 47 counties face elevated flood, landslide, and displacement risk this season. Government messaging has been explicit that "early warning must translate into early action." This assistant was built to support exactly that window.

## What we built

We adapted a general-purpose AI language model on 180 examples drawn from Kenya Meteorological Department (KMD) bulletins, documented 2026 flood events, and international flood-preparedness practice, so it can explain flood risk advisories in plain language, give practical mitigation steps by county and terrain type, and — critically — always point residents to the correct official channel (KMD, NDMA, or their county disaster management committee) rather than guessing at real-time risk itself.

## What improved, in plain numbers

Tested against 18 questions the model had never seen, compared to the unmodified base model:

- **Match to our specific, sourced guidance nearly doubled** (a standard text-similarity score rose 72%, from 0.15 to 0.26).
- **Independent quality scoring rose from an already-solid 4.26 to 4.47 out of 5** — general flood-safety reasoning was already reasonable in the base model; fine-tuning sharpened it further.
- **Factual reliability rose from 3.78 to 4.17 out of 5, and every single one of the 18 test answers cleared our safety threshold** — no ungrounded or invented guidance was produced in testing.
- The single largest improvement was on a Kenya-specific fact the base model had no way of knowing: what rainfall threshold KMD flags as a warning sign. Judged quality on that question rose from 1.8 to 3.8 out of 5 after fine-tuning — this is exactly where local, sourced training data adds value a generic model cannot supply on its own.

## Compute cost

Training took under two minutes of rented cloud GPU time, at a cost of well under $1. As with any fine-tuning project, the larger investment was in dataset curation and verification against authoritative sources — the compute itself is not the constraint.

## Two recommended next actions

1. **Pilot the assistant now, ahead of the October–December short rains**, for informational and preparedness guidance only — not as a replacement for official real-time alerts. Given the season's forecast severity, a working preparedness tool now has more value than a perfected one later.
2. **Expand the dataset before the next training round** to cover hyper-local guidance for informal settlements and Swahili/Sheng-language queries, both currently thin in coverage, and add verified county-level emergency contact details once confirmed by each county's disaster management office.

## The main risk

**Confident but ungrounded guidance**, the same risk any fine-tuned assistant carries. Testing found no groundedness failures in this run, but the sample is small (18 questions) and does not cover every county or scenario. **Mitigation**: treat this as a preparedness and education tool, not a real-time alert source; keep the system prompt's instruction to always defer specific risk calls to official KMD/NDMA channels; and monitor real user queries after deployment to catch any topic where the model's confidence outpaces its actual training coverage.

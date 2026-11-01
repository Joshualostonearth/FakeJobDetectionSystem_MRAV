# Real-world fake job / internship samples (21)

Real fake-job and fake-internship messages collected on 2026-11-01. Each one was
published by a university security office, a security vendor or a news outlet
and confirmed there to be a scam. The message text is copied word for word from
the source page, or transcribed from the source's screenshot for RW_FAKE_016 (image variant),
RW_FAKE_020 and RW_FAKE_021. Nothing here is synthetic.

| File | Contents |
|---|---|
| `fake_jobs.json` | Source of truth: text, channel, scam type, red flags, images, source URL |
| `fake_jobs.csv` | Same data in `test_dataset.csv` column names (`doc_content` = text, `is_fraudulent=1`, `true_label=0`) |
| `texts/RW_FAKE_0xx.txt` | Plain text per sample (for the text pipeline) |
| `images/` | 13 screenshots/posters/flyers for 10 of the samples (for the OCR/image pipeline) |
| `build_csv.py` | Regenerates CSV + texts from the JSON |

Mix: 15 emails (university-targeted check, mule, ID-harvesting and equipment scams),
1 PDF flyer, 1 email poster, 1 MMS "We are hiring" poster, 3 SMS recruiter/task scams,
1 Indian pay-to-join internship offer letter (Rs 1,594 fee).

Caveats
- Most samples are US university scams (UMBC, UCLA, UH, Montclair), because those
  offices publish full text. Indian WhatsApp/Telegram task-scam reports usually describe
  the scam without quoting it, so only RW_FAKE_020 is Indian.
- Phone numbers and emails are the scammers' own, as published by the sources. Some
  sources redacted parts of them (RW_FAKE_009 and RW_FAKE_021).
- RW_FAKE_001–008 share one source (UMBC). Expect near-duplicate wording in 002/005
  and 003/004/006.
- The "real" company names in these samples (UCLA, Indeed, Onin Staffing, etc.) were
  impersonated. Those organisations are victims, not the senders.

---
license: mit
extra_gated_prompt: "You agree to not use the dataset to conduct experiments that cause harm to human subjects."
extra_gated_fields:
  Company: text
  Country: country
  Specific date: date_picker
  I want to use this dataset for:
    type: select
    options: 
      - Research
      - label: Other
        value: other
  I agree to use this dataset for research use ONLY: checkbox
---

## Dataset Details

This dataset contains attack prompts generated from GCG, AutoDAN, PAIR, and DeepInception for **research use ONLY**.

## Dataset Sources

<!-- Provide the basic links for the dataset. -->

- **Repository:** [https://github.com/uw-nsl/SafeDecoding](https://github.com/uw-nsl/SafeDecoding)
- **Paper:** [https://arxiv.org/abs/2402.08983](https://arxiv.org/abs/2402.08983)

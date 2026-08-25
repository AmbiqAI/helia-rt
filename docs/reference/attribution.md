<!--
  DRAFT — LEGAL-ADJACENT, COUNSEL-OWNED. The "License Boundaries" section below
  is owned by counsel: engineering sign-off cannot absorb it, and approval from
  Adam alone is not sufficient to publish it. The section places the product
  position recorded in https://github.com/AmbiqAI/helia-rt/issues/200 ("Non-Ambiq
  hosts/targets are allowed for development, testing, validation, and CI/CD;
  production MCU deployment remains Ambiq-only") beside the governing text in
  LICENSE, which does not currently carve those activities out. Nothing here
  interprets, narrows, or extends LICENSE — where the two differ, LICENSE
  governs and this page must be corrected, not the other way round.
-->

# Attribution and Trademarks

heliaRT is developed and maintained by Ambiq. It is designed to be compatible with LiteRT for Micro APIs and includes code derived from or compatible with upstream TensorFlow Lite for Microcontrollers / LiteRT for Micro components where applicable.

## License Scope

heliaRT is released under the [Ambiq Apollo SDK License](https://github.com/AmbiqAI/helia-rt/blob/main/LICENSE). Free use, modification, and redistribution are permitted solely for execution on Ambiq-manufactured CPUs. See the repository license for the full terms, and [License Boundaries](#license-boundaries) below for how Ambiq's product position relates to that restriction.

<!-- TODO(adam): README.md:86 and CONTRIBUTING.md:39 carry this same "solely for
     execution on Ambiq-manufactured CPUs" summary with no pointer to the
     boundaries discussion. If License Boundaries publishes, decide whether those
     two need the same forward reference. Deliberately not edited in this PR. -->

Some files in this repository are derived from upstream TensorFlow Lite for Microcontrollers / LiteRT for Micro or other third-party projects. Those files retain their original copyright and license notices in source headers or adjacent license files. Review the repository [third-party notices](https://github.com/AmbiqAI/helia-rt/blob/main/THIRD_PARTY_NOTICES.md) before redistributing source or binary artifacts.

## License Boundaries

A common question is whether the Ambiq-only field-of-use restriction blocks
everyday engineering work on a laptop, an x86 CI runner, or a non-Ambiq
evaluation board. Ambiq's product position, recorded in
[#200](https://github.com/AmbiqAI/helia-rt/issues/200), is that it should not —
but LICENSE governs and does not currently carve these activities out; see the
open question below.

> Non-Ambiq hosts/targets are allowed for development, testing, validation, and
> CI/CD; production MCU deployment remains Ambiq-only.

| Activity | Ambiq's product position on non-Ambiq hosts/targets |
|---|---|
| Development | Allowed |
| Testing | Allowed |
| Validation | Allowed |
| CI/CD | Allowed |
| Production MCU deployment | Ambiq-only |
| Production deployment on any other non-Ambiq CPU | Open — see LICENSE |

This table is illustrative rather than exhaustive, and it records a product
position, not a grant of rights. The governing text is the
[LICENSE](https://github.com/AmbiqAI/helia-rt/blob/main/LICENSE) itself:

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the “Software”), to use,
> copy, modify, and distribute the Software solely for incorporation into,
> and execution on, computing platforms that include an Ambiq-manufactured CPU
> (including but not limited to the Apollo series). Use, reproduction,
> modification, or distribution of the Software on any CPU not manufactured by
> Ambiq is strictly prohibited, subject to the following conditions:
>
> *[redistribution and endorsement conditions omitted — see LICENSE]*
>
> Field-of-Use Restriction. The Software and any derivative works
> thereof (in source or binary form) may only be executed on a CPU
> manufactured by Ambiq. Execution on any non-Ambiq CPU, even as part of a
> multi-CPU system, is expressly forbidden.

!!! warning "LICENSE governs"
    The table above summarizes Ambiq's product position for planning purposes.
    It is not a modification of the license, it does not grant rights the
    license withholds, and it does not replace legal advice. Where the position
    and the LICENSE text differ, the LICENSE text controls —
    [contact Ambiq AITG](mailto:support.aitg@ambiq.com).

<!-- TODO(adam): the product position in #200 and the LICENSE text are not
     reconcilable as written — #200 carves out development/testing/validation/
     CI-CD, while LICENSE names no activities and prohibits execution on any
     non-Ambiq CPU outright. Counsel must decide whether LICENSE itself is
     amended to carry the carve-out. Until then this section states intent, not
     authorization, and must not be cited as permission. -->

<!-- TODO(adam): #200 says "production MCU deployment"; the quoted Field-of-Use
     Restriction says CPU and expressly reaches multi-CPU systems. Confirm which
     boundary is intended, and what the "any other non-Ambiq CPU" row should say
     once counsel has ruled. -->

<!-- TODO(adam): for counsel — whether running heliaRT under an instruction-set
     simulator, an FVP, or virtual hardware on an x86 host sits on the
     development/test side. On the LICENSE text as quoted, that is the Software
     executing on a non-Ambiq CPU. #200 does not address it. This page must not
     answer it. -->

For which *releases* Ambiq supports and for how long, see the
[Support Policy](support-policy.md).

## Non-affiliation

LiteRT, TensorFlow, Google, and related marks are trademarks of Google LLC. heliaRT is not affiliated with, endorsed by, sponsored by, or otherwise associated with Google LLC.

References to LiteRT, TensorFlow Lite for Microcontrollers, or TFLM describe API compatibility, upstream lineage, or migration context.

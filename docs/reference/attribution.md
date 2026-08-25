<!--
  Wording approved by the portfolio authority (Dr. Adam Page) on 2026-08-25.

  The "License Boundaries" section quotes LICENSE as amended by
  https://github.com/AmbiqAI/helia-rt/pull/220, which must merge before this
  page. The block quote is verbatim from that branch, not a paraphrase — if
  #220's wording changes in review, re-quote it here before merging.
-->

# Attribution and Trademarks

heliaRT is developed and maintained by Ambiq. It is designed to be compatible with LiteRT for Micro APIs and includes code derived from or compatible with upstream TensorFlow Lite for Microcontrollers / LiteRT for Micro components where applicable.

## License Scope

heliaRT is released under the [Ambiq Apollo SDK License](https://github.com/AmbiqAI/helia-rt/blob/main/LICENSE). Use, modification, and redistribution are free; production or commercial deployment is licensed solely for Ambiq-manufactured CPUs, while development, testing, and simulation may run on any CPU. See the repository license for the full terms, and [License Boundaries](#license-boundaries) below for what that means in practice.

Some files in this repository are derived from upstream TensorFlow Lite for Microcontrollers / LiteRT for Micro or other third-party projects. Those files retain their original copyright and license notices in source headers or adjacent license files. Review the repository [third-party notices](https://github.com/AmbiqAI/helia-rt/blob/main/THIRD_PARTY_NOTICES.md) before redistributing source or binary artifacts.

## License Boundaries

A common question is whether the Ambiq field-of-use restriction blocks everyday
engineering work on a laptop, an x86 CI runner, or a non-Ambiq evaluation board.
It does not. The license restricts **production or commercial** deployment to
Ambiq-manufactured CPUs, and expressly permits development, testing, validation,
benchmarking, continuous integration, emulation, and simulation on non-Ambiq
CPUs.

| Activity on non-Ambiq CPUs | Status |
|---|:---:|
| Development | ✓ Permitted |
| Testing | ✓ Permitted |
| Validation | ✓ Permitted |
| Benchmarking | ✓ Permitted |
| Continuous integration | ✓ Permitted |
| Emulation and simulation — desktop and server CPUs, instruction-set simulators, virtual or emulated hardware | ✓ Permitted |
| Deployment or execution in a production or commercial environment | — Not permitted |

Running heliaRT under an instruction-set simulator, an FVP, or virtual hardware
on a desktop or server CPU is development and simulation, and is permitted.

The restriction is not MCU-specific: it applies to deployment or execution in a
production or commercial environment on any CPU not manufactured by Ambiq,
including as part of a multi-CPU system.

The governing text is the
[LICENSE](https://github.com/AmbiqAI/helia-rt/blob/main/LICENSE) itself:

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the “Software”), to use,
> copy, modify, and distribute the Software solely for incorporation into,
> and execution on, computing platforms that include an Ambiq-manufactured CPU
> (including but not limited to the Apollo series), except as expressly
> permitted below, subject to the following conditions:
>
> *[redistribution and endorsement conditions omitted — see LICENSE]*
>
> Deployment or execution of the Software in a production or commercial
> environment on any CPU not manufactured by Ambiq is prohibited.
>
> Development, testing, validation, benchmarking, continuous integration,
> emulation, and simulation of the Software on non-Ambiq CPUs — including
> desktop and server CPUs, instruction-set simulators, and virtual or emulated
> hardware — are permitted.
>
> Field-of-Use Restriction. The Software and any derivative works
> thereof (in source or binary form) may only be deployed or executed in a
> production or commercial environment on a CPU manufactured by Ambiq.
> Production or commercial execution on any non-Ambiq CPU, even as part of a
> multi-CPU system, is expressly forbidden. Development, testing, validation,
> benchmarking, continuous integration, emulation, and simulation on non-Ambiq
> CPUs remain permitted as set out above.

!!! warning "LICENSE governs"
    The table above summarizes the license for planning purposes. It is not a
    modification of the license and does not replace legal advice. Where the
    summary and the LICENSE text differ, the LICENSE text controls —
    [contact Ambiq AITG](mailto:support.aitg@ambiq.com).

<!-- TODO(adam): the LICENSE does not define "production or commercial
     environment", and this page deliberately does not either — it uses the
     license's own words. If a customer asks for a boundary test, that
     definition belongs in LICENSE first, not here. -->

For which *releases* Ambiq supports and for how long, see the
[Support Policy](support-policy.md).

## Non-affiliation

LiteRT, TensorFlow, Google, and related marks are trademarks of Google LLC. heliaRT is not affiliated with, endorsed by, sponsored by, or otherwise associated with Google LLC.

References to LiteRT, TensorFlow Lite for Microcontrollers, or TFLM describe API compatibility, upstream lineage, or migration context.

<!--
  Wording approved by the portfolio authority (Dr. Adam Page) on 2026-08-25.

  The "License Boundaries" section quotes LICENSE as amended by
  https://github.com/AmbiqAI/helia-rt/pull/220, which must merge before this
  page. The LICENSE lines in the block quote are verbatim excerpts from that
  branch at 12dbac41, with omissions explicitly marked — if #220 changes again
  before merge, re-quote it here.
-->

# Attribution and Trademarks

heliaRT is developed and maintained by Ambiq. It is designed to be compatible with LiteRT for Micro APIs and includes code derived from or compatible with upstream TensorFlow Lite for Microcontrollers / LiteRT for Micro components where applicable.

## License Scope

heliaRT is released under the [Ambiq Apollo SDK License](https://github.com/AmbiqAI/helia-rt/blob/main/LICENSE). Free use, modification, and redistribution for Ambiq-manufactured CPUs; production or commercial deployment on non-Ambiq CPUs is not licensed, while development, testing, validation, benchmarking, CI, emulation, and simulation may run on any CPU. See the repository license for the full terms, and [License Boundaries](#license-boundaries) below for what that means in practice.

Some files in this repository are derived from upstream TensorFlow Lite for Microcontrollers / LiteRT for Micro or other third-party projects. Those files retain their original copyright and license notices in source headers or adjacent license files. Review the repository [third-party notices](https://github.com/AmbiqAI/helia-rt/blob/main/THIRD_PARTY_NOTICES.md) before redistributing source or binary artifacts.

## License Boundaries

A common question is whether the Ambiq field-of-use restriction blocks everyday
engineering work on a laptop, an x86 CI runner, or a non-Ambiq evaluation board.
It does not. The license restricts **production or commercial** deployment to
Ambiq-manufactured CPUs, and permits use, reproduction, modification, and
distribution of the Software and derivative works as reasonably necessary for
development, testing, validation, benchmarking, continuous integration,
emulation, and simulation on non-Ambiq CPUs.

| Activity on non-Ambiq CPUs | Status |
|---|:---:|
| Development | ✓ Permitted |
| Testing | ✓ Permitted |
| Validation | ✓ Permitted |
| Benchmarking | ✓ Permitted |
| Continuous integration | ✓ Permitted |
| Emulation and simulation — desktop and server CPUs, instruction-set simulators, virtual or emulated hardware | ✓ Permitted |
| Internal development, evaluation, and demonstration by the licensee | ✓ Permitted |
| Deployment or execution in a production or commercial environment | — Not permitted |

Three consequences worth spelling out:

- **Derivative works are covered.** The permission reaches the Software *and
  derivative works*, so a modified or ported build for a development or CI
  target is inside it, not outside.
- **Build and artifact flow is licensed, not just execution.** Use,
  reproduction, modification, and distribution are permitted as reasonably
  necessary for those activities, which is what moving a host-built binary
  between developers and CI runners requires.
- **Internal evaluation and demonstration are not production.** The license
  defines "production or commercial environment" and expressly excludes
  internal development, evaluation, and demonstration by the licensee, so an
  internal demo on a laptop is not a production deployment.

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
> Field-of-Use Restriction. The Software and any derivative works thereof
> (in source or binary form) may only be deployed or executed in a production
> or commercial environment on a CPU manufactured by Ambiq. Use,
> reproduction, modification, or distribution of the Software or any
> derivative work for deployment or execution in a production or commercial
> environment on any CPU not manufactured by Ambiq is strictly prohibited,
> even as part of a multi-CPU system. Use, reproduction, modification, and
> distribution of the Software and derivative works as reasonably necessary
> for development, testing, validation, benchmarking, continuous integration,
> emulation, and simulation on non-Ambiq CPUs (including desktop and server
> CPUs, instruction-set simulators, and virtual or emulated hardware) are
> permitted, subject to the conditions above.
>
> For purposes of this License, a “production or commercial environment”
> means any deployment of the Software to, or operation of the Software for,
> third parties or end users, or any use of the Software in the delivery of a
> product or service for which consideration is received; internal
> development, evaluation, and demonstration by the licensee is not a
> production or commercial environment.

!!! warning "LICENSE governs"
    The table above summarizes the license for planning purposes. It is not a
    modification of the license and does not replace legal advice. Where the
    summary and the LICENSE text differ, the LICENSE text controls —
    [contact Ambiq AITG](mailto:support.aitg@ambiq.com).

<!-- The LICENSE now defines "production or commercial environment" itself, so
     this page quotes that definition rather than offering one of its own. Keep
     it that way: any refinement of the boundary belongs in LICENSE first. -->

For which *releases* Ambiq supports and for how long, see the
[Support Policy](support-policy.md).

## Non-affiliation

LiteRT, TensorFlow, Google, and related marks are trademarks of Google LLC. heliaRT is not affiliated with, endorsed by, sponsored by, or otherwise associated with Google LLC.

References to LiteRT, TensorFlow Lite for Microcontrollers, or TFLM describe API compatibility, upstream lineage, or migration context.

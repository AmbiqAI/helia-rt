<!--
  DRAFT — LEGAL-ADJACENT. The "License Boundaries" section below requires Adam's
  explicit sign-off before it publishes. It restates the product decision
  recorded in https://github.com/AmbiqAI/helia-rt/issues/200 ("Non-Ambiq
  hosts/targets are allowed for development, testing, validation, and CI/CD;
  production MCU deployment remains Ambiq-only") alongside the governing text in
  LICENSE. Nothing here interprets, narrows, or extends LICENSE — where the two
  could be read differently, LICENSE governs and this page must be corrected,
  not the other way round.
-->

# Attribution and Trademarks

heliaRT is developed and maintained by Ambiq. It is designed to be compatible with LiteRT for Micro APIs and includes code derived from or compatible with upstream TensorFlow Lite for Microcontrollers / LiteRT for Micro components where applicable.

## License Scope

heliaRT is released under the [Ambiq Apollo SDK License](https://github.com/AmbiqAI/helia-rt/blob/main/LICENSE). Free use, modification, and redistribution are permitted solely for execution on Ambiq-manufactured CPUs. See the repository license for the full terms.

Some files in this repository are derived from upstream TensorFlow Lite for Microcontrollers / LiteRT for Micro or other third-party projects. Those files retain their original copyright and license notices in source headers or adjacent license files. Review the repository [third-party notices](https://github.com/AmbiqAI/helia-rt/blob/main/THIRD_PARTY_NOTICES.md) before redistributing source or binary artifacts.

## License Boundaries

A common question is whether the Ambiq-only field-of-use restriction blocks
everyday engineering work on a laptop, an x86 CI runner, or a non-Ambiq
evaluation board. It does not. Ambiq's product position is:

> Non-Ambiq hosts/targets are allowed for development, testing, validation, and
> CI/CD; production MCU deployment remains Ambiq-only.

| Activity | Allowed on non-Ambiq hosts/targets |
|---|:---:|
| Development | ✓ |
| Testing | ✓ |
| Validation | ✓ |
| CI/CD | ✓ |
| Production MCU deployment | — Ambiq-only |

The governing text is the [LICENSE](https://github.com/AmbiqAI/helia-rt/blob/main/LICENSE)
itself, which grants permission to "use, copy, modify, and distribute the
Software solely for incorporation into, and execution on, computing platforms
that include an Ambiq-manufactured CPU (including but not limited to the Apollo
series)," and adds a Field-of-Use Restriction stating that the Software and its
derivative works "may only be executed on a CPU manufactured by Ambiq."

!!! warning "LICENSE governs"
    The table above summarizes Ambiq's product position for planning purposes.
    It is not a modification of the license, and it does not replace legal
    advice. If your reading of the LICENSE text conflicts with this summary,
    the LICENSE text controls — [contact Ambiq AITG](mailto:support.aitg@ambiq.com)
    and we will resolve the discrepancy.

<!-- TODO(adam): the product decision in #200 and the Field-of-Use Restriction in
     LICENSE are stated at different levels of generality — #200 carves out
     development/test/validation/CI-CD, LICENSE does not name those activities.
     Confirm whether LICENSE itself should be amended to carry the carve-out, or
     whether this page is the intended (and sufficient) place for it. Until that
     is settled, this section should not be treated as an authorization. -->

<!-- TODO(adam): #200 says "production MCU deployment". Confirm whether the
     intended boundary is "MCU" specifically or "production deployment on any
     CPU", since LICENSE says CPU and explicitly reaches multi-CPU systems. -->

Simulators, instruction-set models, and virtual hardware used in CI are covered
by the development/testing/validation/CI-CD side of this boundary.

<!-- TODO(adam): confirm the sentence above. #200 does not name simulators or
     virtual hardware explicitly; it is being read as development/test tooling.
     Delete it if that reading is not intended. -->

For which *releases* Ambiq supports and for how long, see the
[Support Policy](support-policy.md).

## Non-affiliation

LiteRT, TensorFlow, Google, and related marks are trademarks of Google LLC. heliaRT is not affiliated with, endorsed by, sponsored by, or otherwise associated with Google LLC.

References to LiteRT, TensorFlow Lite for Microcontrollers, or TFLM describe API compatibility, upstream lineage, or migration context.

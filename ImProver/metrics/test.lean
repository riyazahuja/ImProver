import ImProver.metrics.tagger



@[improver_example test, version unoptimized]
theorem myTaggedTheorem : True := by
  sorry


@[improver_example test, version optimized]
theorem myTaggedTheorem2 : True := by
  trivial

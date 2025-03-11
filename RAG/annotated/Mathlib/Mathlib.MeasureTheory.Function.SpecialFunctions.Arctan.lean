@[measurability]
theorem measurable_arctan : Measurable arctan :=
  continuous_arctan.measurable


@[measurability]
theorem Measurable.arctan (hf : Measurable f) : Measurable fun x => arctan (f x) :=
  measurable_arctan.comp hf



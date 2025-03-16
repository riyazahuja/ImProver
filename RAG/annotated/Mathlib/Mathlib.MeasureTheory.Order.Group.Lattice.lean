@[to_additive (attr := measurability)]
theorem measurable_oneLePart [MeasurableSup α] : Measurable (oneLePart : α → α) :=
  measurable_sup_const _


@[to_additive (attr := measurability)]
protected theorem Measurable.oneLePart [MeasurableSup α] (hf : Measurable f) :
    Measurable fun x ↦ oneLePart (f x) :=
  measurable_oneLePart.comp hf


@[to_additive (attr := measurability)]
theorem measurable_leOnePart [MeasurableSup α] : Measurable (leOnePart : α → α) :=
  (measurable_sup_const _).comp measurable_inv


@[to_additive (attr := measurability)]
protected theorem Measurable.leOnePart [MeasurableSup α] (hf : Measurable f) :
    Measurable fun x ↦ leOnePart (f x) :=
  measurable_leOnePart.comp hf


@[to_additive (attr := measurability)]
theorem measurable_mabs : Measurable (mabs : α → α) :=
  measurable_id'.sup measurable_inv


@[to_additive (attr := measurability)]
protected theorem Measurable.mabs (hf : Measurable f) : Measurable fun x ↦ mabs (f x) :=
  measurable_mabs.comp hf


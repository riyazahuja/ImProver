/-- The submonoid of positive elements. -/
@[simps] def pos : Submonoid α where
  carrier := Set.Ioi 0
  one_mem' := zero_lt_one
  mul_mem' := mul_pos


@[simp] lemma mem_pos : a ∈ pos α ↔ 0 < a := Iff.rfl



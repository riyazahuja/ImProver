@[to_additive]
lemma ciSup_mul (hf : BddAbove (range f)) (a : G) : (⨆ i, f i) * a = ⨆ i, f i * a :=
  (OrderIso.mulRight a).map_ciSup hf


@[to_additive]
lemma ciSup_div (hf : BddAbove (range f)) (a : G) : (⨆ i, f i) / a = ⨆ i, f i / a := by
  /-
    ι : Type u_1
    G : Type u_2
    inst✝³ : Group G
    inst✝² : ConditionallyCompleteLattice G
    inst✝¹ : Nonempty ι
    f : ι → G
    inst✝ : MulRightMono G
    hf : BddAbove (Set.range f)
    a : G
    ⊢ Eq (HDiv.hDiv (iSup fun i => f i) a) (iSup fun i => HDiv.hDiv (f i) a)
  -/
  simp only [div_eq_mul_inv, ciSup_mul hf]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma ciInf_mul (hf : BddBelow (range f)) (a : G) : (⨅ i, f i) * a = ⨅ i, f i * a :=
  (OrderIso.mulRight a).map_ciInf hf


@[to_additive]
lemma ciInf_div (hf : BddBelow (range f)) (a : G) : (⨅ i, f i) / a = ⨅ i, f i / a := by
  /-
    ι : Type u_1
    G : Type u_2
    inst✝³ : Group G
    inst✝² : ConditionallyCompleteLattice G
    inst✝¹ : Nonempty ι
    f : ι → G
    inst✝ : MulRightMono G
    hf : BddBelow (Set.range f)
    a : G
    ⊢ Eq (HDiv.hDiv (iInf fun i => f i) a) (iInf fun i => HDiv.hDiv (f i) a)
  -/
  simp only [div_eq_mul_inv, ciInf_mul hf]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma mul_ciSup (hf : BddAbove (range f)) (a : G) : (a * ⨆ i, f i) = ⨆ i, a * f i :=
  (OrderIso.mulLeft a).map_ciSup hf


@[to_additive]
lemma mul_ciInf (hf : BddBelow (range f)) (a : G) : (a * ⨅ i, f i) = ⨅ i, a * f i :=
  (OrderIso.mulLeft a).map_ciInf hf



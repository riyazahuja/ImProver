@[deprecated isBoundedUnder_of (since := "2024-06-07")]
lemma Filter.isBounded_le_map_of_bounded_range {ι : Type*} (F : Filter ι) {f : ι → ℝ}
    (h : Bornology.IsBounded (Set.range f)) :
    (F.map f).IsBounded (· ≤ ·) := by
  /-
    ι : Type u_1
    F : Filter ι
    f : ι → Real
    h : Bornology.IsBounded (Set.range f)
    ⊢ Filter.IsBounded (fun x1 x2 => LE.le x1 x2) (Filter.map f F)
  -/
  obtain ⟨c, hc⟩ := h.bddAbove
  /-
    case intro
    ι : Type u_1
    F : Filter ι
    f : ι → Real
    h : Bornology.IsBounded (Set.range f)
    c : Real
    hc : Membership.mem (upperBounds (Set.range f)) c
    ⊢ Filter.IsBounded (fun x1 x2 => LE.le x1 x2) (Filter.map f F)
  -/
  exact isBoundedUnder_of ⟨c, by simpa [mem_upperBounds] using hc⟩
  /-
    🎉 no goals
  -/


@[deprecated isBoundedUnder_of (since := "2024-06-07")]
lemma Filter.isBounded_ge_map_of_bounded_range {ι : Type*} (F : Filter ι) {f : ι → ℝ}
    (h : Bornology.IsBounded (Set.range f)) :
    (F.map f).IsBounded (· ≥ ·) := by
  /-
    ι : Type u_1
    F : Filter ι
    f : ι → Real
    h : Bornology.IsBounded (Set.range f)
    ⊢ Filter.IsBounded (fun x1 x2 => GE.ge x1 x2) (Filter.map f F)
  -/
  obtain ⟨c, hc⟩ := h.bddBelow
  /-
    case intro
    ι : Type u_1
    F : Filter ι
    f : ι → Real
    h : Bornology.IsBounded (Set.range f)
    c : Real
    hc : Membership.mem (lowerBounds (Set.range f)) c
    ⊢ Filter.IsBounded (fun x1 x2 => GE.ge x1 x2) (Filter.map f F)
  -/
  apply isBoundedUnder_of ⟨c, by simpa [mem_lowerBounds] using hc⟩
  /-
    🎉 no goals
  -/



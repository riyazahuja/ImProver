theorem Int.measurable_floor [OpensMeasurableSpace R] : Measurable (Int.floor : R → ℤ) :=
  measurable_to_countable fun x => by
    /-
      R : Type u_2
      inst✝⁵ : LinearOrderedRing R
      inst✝⁴ : FloorRing R
      inst✝³ : TopologicalSpace R
      inst✝² : OrderTopology R
      inst✝¹ : MeasurableSpace R
      inst✝ : OpensMeasurableSpace R
      x : R
      ⊢ MeasurableSet (Set.preimage Int.floor (Singleton.singleton (Int.floor x)))
    -/
    simpa only [Int.preimage_floor_singleton] using measurableSet_Ico
    /-
      🎉 no goals
    -/


@[measurability]
theorem Measurable.floor [OpensMeasurableSpace R] {f : α → R} (hf : Measurable f) :
    Measurable fun x => ⌊f x⌋ :=
  Int.measurable_floor.comp hf


theorem Int.measurable_ceil [OpensMeasurableSpace R] : Measurable (Int.ceil : R → ℤ) :=
  measurable_to_countable fun x => by
    /-
      R : Type u_2
      inst✝⁵ : LinearOrderedRing R
      inst✝⁴ : FloorRing R
      inst✝³ : TopologicalSpace R
      inst✝² : OrderTopology R
      inst✝¹ : MeasurableSpace R
      inst✝ : OpensMeasurableSpace R
      x : R
      ⊢ MeasurableSet (Set.preimage Int.ceil (Singleton.singleton (Int.ceil x)))
    -/
    simpa only [Int.preimage_ceil_singleton] using measurableSet_Ioc
    /-
      🎉 no goals
    -/


@[measurability]
theorem Measurable.ceil [OpensMeasurableSpace R] {f : α → R} (hf : Measurable f) :
    Measurable fun x => ⌈f x⌉ :=
  Int.measurable_ceil.comp hf


theorem measurable_fract [BorelSpace R] : Measurable (Int.fract : R → R) := by
  /-
    R : Type u_2
    inst✝⁵ : LinearOrderedRing R
    inst✝⁴ : FloorRing R
    inst✝³ : TopologicalSpace R
    inst✝² : OrderTopology R
    inst✝¹ : MeasurableSpace R
    inst✝ : BorelSpace R
    ⊢ Measurable Int.fract
  -/
  intro s hs
  /-
    R : Type u_2
    inst✝⁵ : LinearOrderedRing R
    inst✝⁴ : FloorRing R
    inst✝³ : TopologicalSpace R
    inst✝² : OrderTopology R
    inst✝¹ : MeasurableSpace R
    inst✝ : BorelSpace R
    s : Set R
    hs : MeasurableSet s
    ⊢ MeasurableSet (Set.preimage Int.fract s)
  -/
  rw [Int.preimage_fract]
  /-
    R : Type u_2
    inst✝⁵ : LinearOrderedRing R
    inst✝⁴ : FloorRing R
    inst✝³ : TopologicalSpace R
    inst✝² : OrderTopology R
    inst✝¹ : MeasurableSpace R
    inst✝ : BorelSpace R
    s : Set R
    hs : MeasurableSet s
    ⊢ MeasurableSet (Set.iUnion fun m => Set.preimage (fun x => HSub.hSub x ↑m) (I …
  -/
  exact MeasurableSet.iUnion fun z => measurable_id.sub_const _ (hs.inter measurableSet_Ico)
  /-
    🎉 no goals
  -/


@[measurability]
theorem Measurable.fract [BorelSpace R] {f : α → R} (hf : Measurable f) :
    Measurable fun x => Int.fract (f x) :=
  measurable_fract.comp hf


theorem MeasurableSet.image_fract [BorelSpace R] {s : Set R} (hs : MeasurableSet s) :
    MeasurableSet (Int.fract '' s) := by
  /-
    R : Type u_2
    inst✝⁵ : LinearOrderedRing R
    inst✝⁴ : FloorRing R
    inst✝³ : TopologicalSpace R
    inst✝² : OrderTopology R
    inst✝¹ : MeasurableSpace R
    inst✝ : BorelSpace R
    s : Set R
    hs : MeasurableSet s
    ⊢ MeasurableSet (Set.image Int.fract s)
  -/
  simp only [Int.image_fract, sub_eq_add_neg, image_add_right']
  /-
    R : Type u_2
    inst✝⁵ : LinearOrderedRing R
    inst✝⁴ : FloorRing R
    inst✝³ : TopologicalSpace R
    inst✝² : OrderTopology R
    inst✝¹ : MeasurableSpace R
    inst✝ : BorelSpace R
    s : Set R
    hs : MeasurableSet s
    ⊢ MeasurableSet (Set.iUnion fun m => Inter.inter (Set.preimage (fun x => HAdd. …
  -/
  exact MeasurableSet.iUnion fun m => (measurable_add_const _ hs).inter measurableSet_Ico
  /-
    🎉 no goals
  -/


theorem Nat.measurable_floor : Measurable (Nat.floor : R → ℕ) :=
  measurable_to_countable fun n => by
    /-
      R : Type u_2
      inst✝⁵ : LinearOrderedSemiring R
      inst✝⁴ : FloorSemiring R
      inst✝³ : TopologicalSpace R
      inst✝² : OrderTopology R
      inst✝¹ : MeasurableSpace R
      inst✝ : OpensMeasurableSpace R
      n : R
      ⊢ MeasurableSet (Set.preimage Nat.floor (Singleton.singleton (Nat.floor n)))
    -/
                                          /-
                                            🎉 no goals
                                          -/
    rcases eq_or_ne ⌊n⌋₊ 0 with h | h <;> simp [h, Nat.preimage_floor_of_ne_zero, -floor_eq_zero]
                                          /-
                                            🎉 no goals
                                          -/


@[measurability]
theorem Measurable.nat_floor (hf : Measurable f) : Measurable fun x => ⌊f x⌋₊ :=
  Nat.measurable_floor.comp hf


theorem Nat.measurable_ceil : Measurable (Nat.ceil : R → ℕ) :=
  measurable_to_countable fun n => by
    /-
      R : Type u_2
      inst✝⁵ : LinearOrderedSemiring R
      inst✝⁴ : FloorSemiring R
      inst✝³ : TopologicalSpace R
      inst✝² : OrderTopology R
      inst✝¹ : MeasurableSpace R
      inst✝ : OpensMeasurableSpace R
      n : R
      ⊢ MeasurableSet (Set.preimage Nat.ceil (Singleton.singleton (Nat.ceil n)))
    -/
                                          /-
                                            🎉 no goals
                                          -/
    rcases eq_or_ne ⌈n⌉₊ 0 with h | h <;> simp_all [h, Nat.preimage_ceil_of_ne_zero, -ceil_eq_zero]
                                          /-
                                            🎉 no goals
                                          -/


@[measurability]
theorem Measurable.nat_ceil (hf : Measurable f) : Measurable fun x => ⌈f x⌉₊ :=
  Nat.measurable_ceil.comp hf



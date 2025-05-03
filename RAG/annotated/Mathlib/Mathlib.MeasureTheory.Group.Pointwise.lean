@[to_additive]
theorem MeasurableSet.const_smul {G α : Type*} [Group G] [MulAction G α] [MeasurableSpace G]
    [MeasurableSpace α] [MeasurableSMul G α] {s : Set α} (hs : MeasurableSet s) (a : G) :
    MeasurableSet (a • s) := by
  /-
    G : Type u_1
    α : Type u_2
    inst✝⁴ : Group G
    inst✝³ : MulAction G α
    inst✝² : MeasurableSpace G
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSMul G α
    s : Set α
    hs : MeasurableSet s
    a : G
    ⊢ MeasurableSet (HSMul.hSMul a s)
  -/
  rw [← preimage_smul_inv]
  /-
    G : Type u_1
    α : Type u_2
    inst✝⁴ : Group G
    inst✝³ : MulAction G α
    inst✝² : MeasurableSpace G
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSMul G α
    s : Set α
    hs : MeasurableSet s
    a : G
    ⊢ MeasurableSet (Set.preimage (fun x => HSMul.hSMul (Inv.inv a) x) s)
  -/
  exact measurable_const_smul _ hs
  /-
    🎉 no goals
  -/


theorem MeasurableSet.const_smul_of_ne_zero {G₀ α : Type*} [GroupWithZero G₀] [MulAction G₀ α]
    [MeasurableSpace G₀] [MeasurableSpace α] [MeasurableSMul G₀ α] {s : Set α}
    (hs : MeasurableSet s) {a : G₀} (ha : a ≠ 0) : MeasurableSet (a • s) := by
  /-
    G₀ : Type u_1
    α : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : MulAction G₀ α
    inst✝² : MeasurableSpace G₀
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSMul G₀ α
    s : Set α
    hs : MeasurableSet s
    a : G₀
    ha : Ne a 0
    ⊢ MeasurableSet (HSMul.hSMul a s)
  -/
  rw [← preimage_smul_inv₀ ha]
  /-
    G₀ : Type u_1
    α : Type u_2
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : MulAction G₀ α
    inst✝² : MeasurableSpace G₀
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSMul G₀ α
    s : Set α
    hs : MeasurableSet s
    a : G₀
    ha : Ne a 0
    ⊢ MeasurableSet (Set.preimage (fun x => HSMul.hSMul (Inv.inv a) x) s)
  -/
  exact measurable_const_smul _ hs
  /-
    🎉 no goals
  -/


theorem MeasurableSet.const_smul₀ {G₀ α : Type*} [GroupWithZero G₀] [Zero α]
    [MulActionWithZero G₀ α] [MeasurableSpace G₀] [MeasurableSpace α] [MeasurableSMul G₀ α]
    [MeasurableSingletonClass α] {s : Set α} (hs : MeasurableSet s) (a : G₀) :
    MeasurableSet (a • s) := by
  /-
    G₀ : Type u_1
    α : Type u_2
    inst✝⁶ : GroupWithZero G₀
    inst✝⁵ : Zero α
    inst✝⁴ : MulActionWithZero G₀ α
    inst✝³ : MeasurableSpace G₀
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSMul G₀ α
    inst✝ : MeasurableSingletonClass α
    s : Set α
    hs : MeasurableSet s
    a : G₀
    ⊢ MeasurableSet (HSMul.hSMul a s)
  -/
  rcases eq_or_ne a 0 with (rfl | ha)
  /-
    case inl
    G₀ : Type u_1
    α : Type u_2
    inst✝⁶ : GroupWithZero G₀
    inst✝⁵ : Zero α
    inst✝⁴ : MulActionWithZero G₀ α
    inst✝³ : MeasurableSpace G₀
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSMul G₀ α
    inst✝ : MeasurableSingletonClass α
    s : Set α
    hs : MeasurableSet s
    ⊢ MeasurableSet (HSMul.hSMul 0 s)
  -/
  exacts [(subsingleton_zero_smul_set s).measurableSet, hs.const_smul_of_ne_zero ha]
  /-
    🎉 no goals
  -/


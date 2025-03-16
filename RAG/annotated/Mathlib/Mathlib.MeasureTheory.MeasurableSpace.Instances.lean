instance Empty.instMeasurableSpace : MeasurableSpace Empty := ⊤


instance PUnit.instMeasurableSpace : MeasurableSpace PUnit := ⊤


instance Bool.instMeasurableSpace : MeasurableSpace Bool := ⊤


instance Prop.instMeasurableSpace : MeasurableSpace Prop := ⊤


instance Nat.instMeasurableSpace : MeasurableSpace ℕ := ⊤


instance ENat.instMeasurableSpace : MeasurableSpace ℕ∞ := ⊤


instance Fin.instMeasurableSpace (n : ℕ) : MeasurableSpace (Fin n) := ⊤


instance ZMod.instMeasurableSpace (n : ℕ) : MeasurableSpace (ZMod n) := ⊤


instance Int.instMeasurableSpace : MeasurableSpace ℤ := ⊤


instance Rat.instMeasurableSpace : MeasurableSpace ℚ := ⊤


@[to_additive]
instance IterateMulAct.instMeasurableSpace {α : Type*} {f : α → α} :
    MeasurableSpace (IterateMulAct f) := ⊤


@[to_additive]
instance IterateMulAct.instDiscreteMeasurableSpace {α : Type*} {f : α → α} :
    DiscreteMeasurableSpace (IterateMulAct f) := inferInstance


instance (priority := 100) Subsingleton.measurableSingletonClass
    {α} [MeasurableSpace α] [Subsingleton α] : MeasurableSingletonClass α := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : Subsingleton α
    ⊢ MeasurableSingletonClass α
  -/
  refine ⟨fun i => ?_⟩
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : Subsingleton α
    i : α
    ⊢ MeasurableSet (Singleton.singleton i)
  -/
  convert MeasurableSet.univ
  /-
    case h.e'_3
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : Subsingleton α
    i : α
    ⊢ Eq (Singleton.singleton i) Set.univ
  -/
  simp [Set.eq_univ_iff_forall, eq_iff_true_of_subsingleton]
  /-
    🎉 no goals
  -/


instance Bool.instMeasurableSingletonClass : MeasurableSingletonClass Bool := ⟨fun _ => trivial⟩


instance Prop.instMeasurableSingletonClass : MeasurableSingletonClass Prop := ⟨fun _ => trivial⟩


instance Nat.instMeasurableSingletonClass : MeasurableSingletonClass ℕ := ⟨fun _ => trivial⟩


instance ENat.instDiscreteMeasurableSpace : DiscreteMeasurableSpace ℕ∞ := ⟨fun _ ↦ trivial⟩


instance ENat.instMeasurableSingletonClass : MeasurableSingletonClass ℕ∞ := inferInstance


instance Fin.instMeasurableSingletonClass (n : ℕ) : MeasurableSingletonClass (Fin n) :=
  ⟨fun _ => trivial⟩


instance ZMod.instMeasurableSingletonClass (n : ℕ) : MeasurableSingletonClass (ZMod n) :=
  ⟨fun _ => trivial⟩


instance Int.instMeasurableSingletonClass : MeasurableSingletonClass ℤ := ⟨fun _ => trivial⟩


instance Rat.instMeasurableSingletonClass : MeasurableSingletonClass ℚ := ⟨fun _ => trivial⟩


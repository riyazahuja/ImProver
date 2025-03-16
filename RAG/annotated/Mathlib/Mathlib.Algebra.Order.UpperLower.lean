@[to_additive]
theorem IsUpperSet.smul_subset (hs : IsUpperSet s) (hx : 1 ≤ x) : x • s ⊆ s :=
  smul_set_subset_iff.2 fun _ ↦ hs <| le_mul_of_one_le_left' hx


@[to_additive]
theorem IsLowerSet.smul_subset (hs : IsLowerSet s) (hx : x ≤ 1) : x • s ⊆ s :=
  smul_set_subset_iff.2 fun _ ↦ hs <| mul_le_of_le_one_left' hx


@[to_additive]
theorem IsUpperSet.smul (hs : IsUpperSet s) : IsUpperSet (a • s) := hs.image <| OrderIso.mulLeft _


@[to_additive]
theorem IsLowerSet.smul (hs : IsLowerSet s) : IsLowerSet (a • s) := hs.image <| OrderIso.mulLeft _


@[to_additive]
theorem Set.OrdConnected.smul (hs : s.OrdConnected) : (a • s).OrdConnected := by
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    s : Set α
    a : α
    hs : s.OrdConnected
    ⊢ (HSMul.hSMul a s).OrdConnected
  -/
  rw [← hs.upperClosure_inter_lowerClosure, smul_set_inter]
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    s : Set α
    a : α
    hs : s.OrdConnected
    ⊢ (Inter.inter (HSMul.hSMul a ↑(upperClosure s)) (HSMul.hSMul a ↑(lowerClosure …
  -/
  exact (upperClosure _).upper.smul.ordConnected.inter (lowerClosure _).lower.smul.ordConnected
  /-
    🎉 no goals
  -/


@[to_additive]
theorem IsUpperSet.mul_left (ht : IsUpperSet t) : IsUpperSet (s * t) := by
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    s t : Set α
    ht : IsUpperSet t
    ⊢ IsUpperSet (HMul.hMul s t)
  -/
  rw [← smul_eq_mul, ← Set.iUnion_smul_set]
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    s t : Set α
    ht : IsUpperSet t
    ⊢ IsUpperSet (Set.iUnion fun a => Set.iUnion fun h => HSMul.hSMul a t)
  -/
  exact isUpperSet_iUnion₂ fun x _ ↦ ht.smul
  /-
    🎉 no goals
  -/


@[to_additive]
theorem IsUpperSet.mul_right (hs : IsUpperSet s) : IsUpperSet (s * t) := by
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    s t : Set α
    hs : IsUpperSet s
    ⊢ IsUpperSet (HMul.hMul s t)
  -/
  rw [mul_comm]
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    s t : Set α
    hs : IsUpperSet s
    ⊢ IsUpperSet (HMul.hMul t s)
  -/
  exact hs.mul_left
  /-
    🎉 no goals
  -/


@[to_additive]
theorem IsLowerSet.mul_left (ht : IsLowerSet t) : IsLowerSet (s * t) := ht.toDual.mul_left


@[to_additive]
theorem IsLowerSet.mul_right (hs : IsLowerSet s) : IsLowerSet (s * t) := hs.toDual.mul_right


@[to_additive]
theorem IsUpperSet.inv (hs : IsUpperSet s) : IsLowerSet s⁻¹ := fun _ _ h ↦ hs <| inv_le_inv' h


@[to_additive]
theorem IsLowerSet.inv (hs : IsLowerSet s) : IsUpperSet s⁻¹ := fun _ _ h ↦ hs <| inv_le_inv' h


@[to_additive]
theorem IsUpperSet.div_left (ht : IsUpperSet t) : IsLowerSet (s / t) := by
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    s t : Set α
    ht : IsUpperSet t
    ⊢ IsLowerSet (HDiv.hDiv s t)
  -/
  rw [div_eq_mul_inv]
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    s t : Set α
    ht : IsUpperSet t
    ⊢ IsLowerSet (HMul.hMul s (Inv.inv t))
  -/
  exact ht.inv.mul_left
  /-
    🎉 no goals
  -/


@[to_additive]
theorem IsUpperSet.div_right (hs : IsUpperSet s) : IsUpperSet (s / t) := by
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    s t : Set α
    hs : IsUpperSet s
    ⊢ IsUpperSet (HDiv.hDiv s t)
  -/
  rw [div_eq_mul_inv]
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    s t : Set α
    hs : IsUpperSet s
    ⊢ IsUpperSet (HMul.hMul s (Inv.inv t))
  -/
  exact hs.mul_right
  /-
    🎉 no goals
  -/


@[to_additive]
theorem IsLowerSet.div_left (ht : IsLowerSet t) : IsUpperSet (s / t) := ht.toDual.div_left


@[to_additive]
theorem IsLowerSet.div_right (hs : IsLowerSet s) : IsLowerSet (s / t) := hs.toDual.div_right


@[to_additive]
instance : One (UpperSet α) :=
  ⟨Ici 1⟩


@[to_additive]
instance : Mul (UpperSet α) :=
  ⟨fun s t ↦ ⟨image2 (· * ·) s t, s.2.mul_right⟩⟩


@[to_additive]
instance : Div (UpperSet α) :=
  ⟨fun s t ↦ ⟨image2 (· / ·) s t, s.2.div_right⟩⟩


@[to_additive]
instance : SMul α (UpperSet α) :=
  ⟨fun a s ↦ ⟨(a • ·) '' s, s.2.smul⟩⟩


@[to_additive (attr := simp,norm_cast)]
theorem coe_one : ((1 : UpperSet α) : Set α) = Set.Ici 1 :=
  rfl


@[to_additive (attr := simp,norm_cast)]
theorem coe_mul (s t : UpperSet α) : (↑(s * t) : Set α) = s * t :=
  rfl


@[to_additive (attr := simp,norm_cast)]
theorem coe_div (s t : UpperSet α) : (↑(s / t) : Set α) = s / t :=
  rfl


@[to_additive (attr := simp)]
theorem Ici_one : Ici (1 : α) = 1 :=
  rfl


@[to_additive]
instance : MulAction α (UpperSet α) :=
  SetLike.coe_injective.mulAction _ (fun _ _ => rfl)


@[to_additive]
instance commSemigroup : CommSemigroup (UpperSet α) :=
  { (SetLike.coe_injective.commSemigroup _ coe_mul : CommSemigroup (UpperSet α)) with }


@[to_additive]
private theorem one_mul (s : UpperSet α) : 1 * s = s :=
  SetLike.coe_injective <|
    (subset_mul_right _ left_mem_Ici).antisymm' <| by
      /-
        α : Type u_1
        inst✝ : OrderedCommGroup α
        s : UpperSet α
        ⊢ HasSubset.Subset (HMul.hMul (Set.Ici 1) ↑s) ↑s
      -/
      rw [← smul_eq_mul, ← Set.iUnion_smul_set]
      /-
        α : Type u_1
        inst✝ : OrderedCommGroup α
        s : UpperSet α
        ⊢ HasSubset.Subset (Set.iUnion fun a => Set.iUnion fun h => HSMul.hSMul a ↑s) ↑s
      -/
      exact Set.iUnion₂_subset fun _ ↦ s.upper.smul_subset
      /-
        🎉 no goals
      -/


@[to_additive]
instance : CommMonoid (UpperSet α) :=
  { UpperSet.commSemigroup with
    one := 1
    one_mul := one_mul
    mul_one := fun s ↦ by
      /-
        α : Type u_1
        inst✝ : OrderedCommGroup α
        s✝ t : Set α
        a : α
        s : UpperSet α
        ⊢ Eq (HMul.hMul s 1) s
      -/
      rw [mul_comm]
      /-
        α : Type u_1
        inst✝ : OrderedCommGroup α
        s✝ t : Set α
        a : α
        s : UpperSet α
        ⊢ Eq (HMul.hMul 1 s) s
      -/
      exact one_mul _ }
      /-
        🎉 no goals
      -/


@[to_additive]
instance : One (LowerSet α) :=
  ⟨Iic 1⟩


@[to_additive]
instance : Mul (LowerSet α) :=
  ⟨fun s t ↦ ⟨image2 (· * ·) s t, s.2.mul_right⟩⟩


@[to_additive]
instance : Div (LowerSet α) :=
  ⟨fun s t ↦ ⟨image2 (· / ·) s t, s.2.div_right⟩⟩


@[to_additive]
instance : SMul α (LowerSet α) :=
  ⟨fun a s ↦ ⟨(a • ·) '' s, s.2.smul⟩⟩


@[to_additive (attr := simp,norm_cast)]
theorem coe_mul (s t : LowerSet α) : (↑(s * t) : Set α) = s * t :=
  rfl


@[to_additive (attr := simp,norm_cast)]
theorem coe_div (s t : LowerSet α) : (↑(s / t) : Set α) = s / t :=
  rfl


@[to_additive (attr := simp)]
theorem Iic_one : Iic (1 : α) = 1 :=
  rfl


@[to_additive]
instance : MulAction α (LowerSet α) :=
  SetLike.coe_injective.mulAction _ (fun _ _ => rfl)


@[to_additive]
instance commSemigroup : CommSemigroup (LowerSet α) :=
  { (SetLike.coe_injective.commSemigroup _ coe_mul : CommSemigroup (LowerSet α)) with }


@[to_additive]
private theorem one_mul (s : LowerSet α) : 1 * s = s :=
  SetLike.coe_injective <|
    (subset_mul_right _ right_mem_Iic).antisymm' <| by
      /-
        α : Type u_1
        inst✝ : OrderedCommGroup α
        s : LowerSet α
        ⊢ HasSubset.Subset (HMul.hMul (Set.Iic 1) ↑s) ↑s
      -/
      rw [← smul_eq_mul, ← Set.iUnion_smul_set]
      /-
        α : Type u_1
        inst✝ : OrderedCommGroup α
        s : LowerSet α
        ⊢ HasSubset.Subset (Set.iUnion fun a => Set.iUnion fun h => HSMul.hSMul a ↑s) ↑s
      -/
      exact Set.iUnion₂_subset fun _ ↦ s.lower.smul_subset
      /-
        🎉 no goals
      -/


@[to_additive]
instance : CommMonoid (LowerSet α) :=
  { LowerSet.commSemigroup with
    one := 1
    one_mul := one_mul
    mul_one := fun s ↦ by
      /-
        α : Type u_1
        inst✝ : OrderedCommGroup α
        s✝ t : Set α
        a : α
        s : LowerSet α
        ⊢ Eq (HMul.hMul s 1) s
      -/
      rw [mul_comm]
      /-
        α : Type u_1
        inst✝ : OrderedCommGroup α
        s✝ t : Set α
        a : α
        s : LowerSet α
        ⊢ Eq (HMul.hMul 1 s) s
      -/
      exact one_mul _ }
      /-
        🎉 no goals
      -/


@[to_additive (attr := simp)]
theorem upperClosure_one : upperClosure (1 : Set α) = 1 :=
  upperClosure_singleton _


@[to_additive (attr := simp)]
theorem lowerClosure_one : lowerClosure (1 : Set α) = 1 :=
  lowerClosure_singleton _


@[to_additive (attr := simp)]
theorem upperClosure_smul : upperClosure (a • s) = a • upperClosure s :=
  upperClosure_image <| OrderIso.mulLeft a


@[to_additive (attr := simp)]
theorem lowerClosure_smul : lowerClosure (a • s) = a • lowerClosure s :=
  lowerClosure_image <| OrderIso.mulLeft a


@[to_additive]
theorem mul_upperClosure : s * upperClosure t = upperClosure (s * t) := by
  simp_rw [← smul_eq_mul, ← Set.iUnion_smul_set, upperClosure_iUnion, upperClosure_smul,
    UpperSet.coe_iInf₂]
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    s t : Set α
    ⊢ Eq (Set.iUnion fun a => Set.iUnion fun h => HSMul.hSMul a ↑(upperClosure t)) …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mul_lowerClosure : s * lowerClosure t = lowerClosure (s * t) := by
  simp_rw [← smul_eq_mul, ← Set.iUnion_smul_set, lowerClosure_iUnion, lowerClosure_smul,
    LowerSet.coe_iSup₂]
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    s t : Set α
    ⊢ Eq (Set.iUnion fun a => Set.iUnion fun h => HSMul.hSMul a ↑(lowerClosure t)) …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive]
theorem upperClosure_mul : ↑(upperClosure s) * t = upperClosure (s * t) := by
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    s t : Set α
    ⊢ Eq (HMul.hMul (↑(upperClosure s)) t) ↑(upperClosure (HMul.hMul s t))
  -/
  simp_rw [mul_comm _ t]
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    s t : Set α
    ⊢ Eq (HMul.hMul t ↑(upperClosure s)) ↑(upperClosure (HMul.hMul t s))
  -/
  exact mul_upperClosure _ _
  /-
    🎉 no goals
  -/


@[to_additive]
theorem lowerClosure_mul : ↑(lowerClosure s) * t = lowerClosure (s * t) := by
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    s t : Set α
    ⊢ Eq (HMul.hMul (↑(lowerClosure s)) t) ↑(lowerClosure (HMul.hMul s t))
  -/
  simp_rw [mul_comm _ t]
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    s t : Set α
    ⊢ Eq (HMul.hMul t ↑(lowerClosure s)) ↑(lowerClosure (HMul.hMul t s))
  -/
  exact mul_lowerClosure _ _
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem upperClosure_mul_distrib : upperClosure (s * t) = upperClosure s * upperClosure t :=
  SetLike.coe_injective <| by
    /-
      α : Type u_1
      inst✝ : OrderedCommGroup α
      s t : Set α
      ⊢ Eq ↑(upperClosure (HMul.hMul s t)) ↑(HMul.hMul (upperClosure s) (upperClosur …
    -/
    rw [UpperSet.coe_mul, mul_upperClosure, upperClosure_mul, UpperSet.upperClosure]
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
theorem lowerClosure_mul_distrib : lowerClosure (s * t) = lowerClosure s * lowerClosure t :=
  SetLike.coe_injective <| by
    /-
      α : Type u_1
      inst✝ : OrderedCommGroup α
      s t : Set α
      ⊢ Eq ↑(lowerClosure (HMul.hMul s t)) ↑(HMul.hMul (lowerClosure s) (lowerClosur …
    -/
    rw [LowerSet.coe_mul, mul_lowerClosure, lowerClosure_mul, LowerSet.lowerClosure]
    /-
      🎉 no goals
    -/



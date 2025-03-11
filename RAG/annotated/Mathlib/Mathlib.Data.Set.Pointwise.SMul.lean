@[to_additive] lemma smul_set_subset_mul : a ∈ s → a • t ⊆ s * t := image_subset_image2_right


open scoped RightActions in
@[to_additive] lemma op_smul_set_subset_mul : a ∈ t → s <• a ⊆ s * t := image_subset_image2_left


@[to_additive]
theorem image_op_smul : (op '' s) • t = t * s := by
  /-
    α : Type u_2
    inst✝ : Mul α
    s t : Set α
    ⊢ Eq (HSMul.hSMul (Set.image MulOpposite.op s) t) (HMul.hMul t s)
  -/
  rw [← image2_smul, ← image2_mul, image2_image_left, image2_swap]
  /-
    α : Type u_2
    inst✝ : Mul α
    s t : Set α
    ⊢ Eq (Set.image2 (fun a b => SMul.smul (MulOpposite.op b) a) t s) (Set.image2  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem iUnion_op_smul_set (s t : Set α) : ⋃ a ∈ t, MulOpposite.op a • s = s * t :=
  iUnion_image_right _


@[to_additive]
theorem mul_subset_iff_left : s * t ⊆ u ↔ ∀ a ∈ s, a • t ⊆ u :=
  image2_subset_iff_left


@[to_additive]
theorem mul_subset_iff_right : s * t ⊆ u ↔ ∀ b ∈ t, op b • s ⊆ u :=
  image2_subset_iff_right


@[to_additive] lemma pair_mul (a b : α) (s : Set α) : {a, b} * s = a • s ∪ b • s := by
  /-
    α : Type u_2
    inst✝ : Mul α
    a b : α
    s : Set α
    ⊢ Eq (HMul.hMul (Insert.insert a (Singleton.singleton b)) s) (Union.union (HSM …
  -/
  rw [insert_eq, union_mul, singleton_mul, singleton_mul]; rfl
                                                           /-
                                                             🎉 no goals
                                                           -/


@[to_additive] lemma mul_pair (s : Set α) (a b : α) : s * {a, b} = s <• a ∪ s <• b := by
  /-
    α : Type u_2
    inst✝ : Mul α
    s : Set α
    a b : α
    ⊢ Eq (HMul.hMul s (Insert.insert a (Singleton.singleton b))) (Union.union (HSM …
  -/
  rw [insert_eq, mul_union, mul_singleton, mul_singleton]; rfl
                                                           /-
                                                             🎉 no goals
                                                           -/


@[to_additive] lemma range_mul [Mul α] {ι : Sort*} (a : α) (f : ι → α) :
    range (fun i ↦ a * f i) = a • range f := range_smul a f


@[to_additive]
instance smulCommClass_set [SMul α γ] [SMul β γ] [SMulCommClass α β γ] :
    SMulCommClass α β (Set γ) :=
  ⟨fun _ _ ↦ Commute.set_image <| smul_comm _ _⟩


@[to_additive]
instance smulCommClass_set' [SMul α γ] [SMul β γ] [SMulCommClass α β γ] :
    SMulCommClass α (Set β) (Set γ) :=
  ⟨fun _ _ _ ↦ image_image2_distrib_right <| smul_comm _⟩


@[to_additive]
instance smulCommClass_set'' [SMul α γ] [SMul β γ] [SMulCommClass α β γ] :
    SMulCommClass (Set α) β (Set γ) :=
  haveI := SMulCommClass.symm α β γ
  SMulCommClass.symm _ _ _


@[to_additive]
instance smulCommClass [SMul α γ] [SMul β γ] [SMulCommClass α β γ] :
    SMulCommClass (Set α) (Set β) (Set γ) :=
  ⟨fun _ _ _ ↦ image2_left_comm smul_comm⟩


@[to_additive vaddAssocClass]
instance isScalarTower [SMul α β] [SMul α γ] [SMul β γ] [IsScalarTower α β γ] :
    IsScalarTower α β (Set γ) where
                         /-
                           F : Type u_1
                           α : Type u_2
                           β : Type u_3
                           γ : Type u_4
                           s s₁ s₂ : Set α
                           t t₁ t₂ : Set β
                           a✝ : α
                           b✝ : β
                           inst✝³ : SMul α β
                           inst✝² : SMul α γ
                           inst✝¹ : SMul β γ
                           inst✝ : IsScalarTower α β γ
                           a : α
                           b : β
                           T : Set γ
                           ⊢ Eq (HSMul.hSMul (HSMul.hSMul a b) T) (HSMul.hSMul a (HSMul.hSMul b T))
                         -/
  smul_assoc a b T := by simp only [← image_smul, image_image, smul_assoc]
                         /-
                           🎉 no goals
                         -/


@[to_additive vaddAssocClass']
instance isScalarTower' [SMul α β] [SMul α γ] [SMul β γ] [IsScalarTower α β γ] :
    IsScalarTower α (Set β) (Set γ) :=
  ⟨fun _ _ _ ↦ image2_image_left_comm <| smul_assoc _⟩


@[to_additive vaddAssocClass'']
instance isScalarTower'' [SMul α β] [SMul α γ] [SMul β γ] [IsScalarTower α β γ] :
    IsScalarTower (Set α) (Set β) (Set γ) where
  smul_assoc _ _ _ := image2_assoc smul_assoc


@[to_additive]
instance isCentralScalar [SMul α β] [SMul αᵐᵒᵖ β] [IsCentralScalar α β] :
    IsCentralScalar α (Set β) :=
  ⟨fun _ S ↦ (congr_arg fun f ↦ f '' S) <| funext fun _ ↦ op_smul_eq_smul _ _⟩


/-- A multiplicative action of a monoid `α` on a type `β` gives a multiplicative action of `Set α`
on `Set β`. -/
@[to_additive
      "An additive action of an additive monoid `α` on a type `β` gives an additive action of
      `Set α` on `Set β`"]
protected def mulAction [Monoid α] [MulAction α β] : MulAction (Set α) (Set β) where
  mul_smul _ _ _ := image2_assoc mul_smul
                                                  /-
                                                    F : Type u_1
                                                    α : Type u_2
                                                    β : Type u_3
                                                    γ : Type u_4
                                                    s✝ s₁ s₂ : Set α
                                                    t t₁ t₂ : Set β
                                                    a : α
                                                    b : β
                                                    inst✝¹ : Monoid α
                                                    inst✝ : MulAction α β
                                                    s : Set β
                                                    ⊢ Eq (Set.image (fun x2 => HSMul.hSMul 1 x2) s) s
                                                  -/
  one_smul s := image2_singleton_left.trans <| by simp_rw [one_smul, image_id']
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- A multiplicative action of a monoid on a type `β` gives a multiplicative action on `Set β`. -/
@[to_additive
      "An additive action of an additive monoid on a type `β` gives an additive action on `Set β`."]
protected def mulActionSet [Monoid α] [MulAction α β] : MulAction α (Set β) where
                       /-
                         F : Type u_1
                         α : Type u_2
                         β : Type u_3
                         γ : Type u_4
                         s s₁ s₂ : Set α
                         t t₁ t₂ : Set β
                         a : α
                         b : β
                         inst✝¹ : Monoid α
                         inst✝ : MulAction α β
                         x✝² x✝¹ : α
                         x✝ : Set β
                         ⊢ Eq (HSMul.hSMul (HMul.hMul x✝² x✝¹) x✝) (HSMul.hSMul x✝² (HSMul.hSMul x✝¹ x✝))
                       -/
                   /-
                     F : Type u_1
                     α : Type u_2
                     β : Type u_3
                     γ : Type u_4
                     s s₁ s₂ : Set α
                     t t₁ t₂ : Set β
                     a : α
                     b : β
                     inst✝¹ : Monoid α
                     inst✝ : MulAction α β
                     x✝ : Set β
                     ⊢ Eq (HSMul.hSMul 1 x✝) x✝
                   -/
  mul_smul _ _ _ := by simp only [← image_smul, image_image, ← mul_smul]
                   /-
                     🎉 no goals
                   -/
                       /-
                         🎉 no goals
                       -/
  one_smul _ := by simp only [← image_smul, one_smul, image_id']


/-- If scalar multiplication by elements of `α` sends `(0 : β)` to zero,
then the same is true for `(0 : Set β)`. -/
protected def smulZeroClassSet [Zero β] [SMulZeroClass α β] :
    SMulZeroClass α (Set β) where
                                             /-
                                               F : Type u_1
                                               α : Type u_2
                                               β : Type u_3
                                               γ : Type u_4
                                               s s₁ s₂ : Set α
                                               t t₁ t₂ : Set β
                                               a : α
                                               b : β
                                               inst✝¹ : Zero β
                                               inst✝ : SMulZeroClass α β
                                               x✝ : α
                                               ⊢ Eq (Singleton.singleton (HSMul.hSMul x✝ 0)) 0
                                             -/
  smul_zero _ := image_singleton.trans <| by rw [smul_zero, singleton_zero]
                                             /-
                                               🎉 no goals
                                             -/


/-- If the scalar multiplication `(· • ·) : α → β → β` is distributive,
then so is `(· • ·) : α → Set β → Set β`. -/
protected def distribSMulSet [AddZeroClass β] [DistribSMul α β] :
    DistribSMul α (Set β) where
  smul_add _ _ _ := image_image2_distrib <| smul_add _


/-- A distributive multiplicative action of a monoid on an additive monoid `β` gives a distributive
multiplicative action on `Set β`. -/
protected def distribMulActionSet [Monoid α] [AddMonoid β] [DistribMulAction α β] :
    DistribMulAction α (Set β) where
  smul_add := smul_add
  smul_zero := smul_zero


/-- A multiplicative action of a monoid on a monoid `β` gives a multiplicative action on `Set β`. -/
protected def mulDistribMulActionSet [Monoid α] [Monoid β] [MulDistribMulAction α β] :
    MulDistribMulAction α (Set β) where
  smul_mul _ _ _ := image_image2_distrib <| smul_mul' _
                                            /-
                                              F : Type u_1
                                              α : Type u_2
                                              β : Type u_3
                                              γ : Type u_4
                                              s s₁ s₂ : Set α
                                              t t₁ t₂ : Set β
                                              a : α
                                              b : β
                                              inst✝² : Monoid α
                                              inst✝¹ : Monoid β
                                              inst✝ : MulDistribMulAction α β
                                              x✝ : α
                                              ⊢ Eq (Singleton.singleton (HSMul.hSMul x✝ 1)) 1
                                            -/
  smul_one _ := image_singleton.trans <| by rw [smul_one, singleton_one]
                                            /-
                                              🎉 no goals
                                            -/


instance [Zero α] [Zero β] [SMul α β] [NoZeroSMulDivisors α β] :
    NoZeroSMulDivisors (Set α) (Set β) :=
  ⟨fun {s t} h ↦ by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      s✝ s₁ s₂ : Set α
      t✝ t₁ t₂ : Set β
      a : α
      b : β
      inst✝³ : Zero α
      inst✝² : Zero β
      inst✝¹ : SMul α β
      inst✝ : NoZeroSMulDivisors α β
      s : Set α
      t : Set β
      h : Eq (HSMul.hSMul s t) 0
      ⊢ Or (Eq s 0) (Eq t 0)
    -/
    by_contra! H
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      s✝ s₁ s₂ : Set α
      t✝ t₁ t₂ : Set β
      a : α
      b : β
      inst✝³ : Zero α
      inst✝² : Zero β
      inst✝¹ : SMul α β
      inst✝ : NoZeroSMulDivisors α β
      s : Set α
      t : Set β
      h : Eq (HSMul.hSMul s t) 0
      H : And (Ne s 0) (Ne t 0)
      ⊢ False
    -/
    have hst : (s • t).Nonempty := h.symm.subst zero_nonempty
    rw [Ne, ← hst.of_smul_left.subset_zero_iff, Ne,
      ← hst.of_smul_right.subset_zero_iff] at H
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      s✝ s₁ s₂ : Set α
      t✝ t₁ t₂ : Set β
      a : α
      b : β
      inst✝³ : Zero α
      inst✝² : Zero β
      inst✝¹ : SMul α β
      inst✝ : NoZeroSMulDivisors α β
      s : Set α
      t : Set β
      h : Eq (HSMul.hSMul s t) 0
      H : And (Not (HasSubset.Subset s 0)) (Not (HasSubset.Subset t 0))
      hst : (HSMul.hSMul s t).Nonempty
      ⊢ False
    -/
    simp only [not_subset, mem_zero] at H
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      s✝ s₁ s₂ : Set α
      t✝ t₁ t₂ : Set β
      a : α
      b : β
      inst✝³ : Zero α
      inst✝² : Zero β
      inst✝¹ : SMul α β
      inst✝ : NoZeroSMulDivisors α β
      s : Set α
      t : Set β
      h : Eq (HSMul.hSMul s t) 0
      hst : (HSMul.hSMul s t).Nonempty
      H : And (Exists fun a => And (Membership.mem s a) (Not (Eq a 0))) (Exists fun  …
      ⊢ False
    -/
    obtain ⟨⟨a, hs, ha⟩, b, ht, hb⟩ := H
    /-
      case intro.intro.intro.intro.intro
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      s✝ s₁ s₂ : Set α
      t✝ t₁ t₂ : Set β
      a✝ : α
      b✝ : β
      inst✝³ : Zero α
      inst✝² : Zero β
      inst✝¹ : SMul α β
      inst✝ : NoZeroSMulDivisors α β
      s : Set α
      t : Set β
      h : Eq (HSMul.hSMul s t) 0
      hst : (HSMul.hSMul s t).Nonempty
      a : α
      hs : Membership.mem s a
      ha : Not (Eq a 0)
      b : β
      ht : Membership.mem t b
      hb : Not (Eq b 0)
      ⊢ False
    -/
    exact (eq_zero_or_eq_zero_of_smul_eq_zero <| h.subset <| smul_mem_smul hs ht).elim ha hb⟩
    /-
      🎉 no goals
    -/


instance noZeroSMulDivisors_set [Zero α] [Zero β] [SMul α β] [NoZeroSMulDivisors α β] :
    NoZeroSMulDivisors α (Set β) :=
  ⟨fun {a s} h ↦ by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      s✝ s₁ s₂ : Set α
      t t₁ t₂ : Set β
      a✝ : α
      b : β
      inst✝³ : Zero α
      inst✝² : Zero β
      inst✝¹ : SMul α β
      inst✝ : NoZeroSMulDivisors α β
      a : α
      s : Set β
      h : Eq (HSMul.hSMul a s) 0
      ⊢ Or (Eq a 0) (Eq s 0)
    -/
    by_contra! H
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      s✝ s₁ s₂ : Set α
      t t₁ t₂ : Set β
      a✝ : α
      b : β
      inst✝³ : Zero α
      inst✝² : Zero β
      inst✝¹ : SMul α β
      inst✝ : NoZeroSMulDivisors α β
      a : α
      s : Set β
      h : Eq (HSMul.hSMul a s) 0
      H : And (Ne a 0) (Ne s 0)
      ⊢ False
    -/
    have hst : (a • s).Nonempty := h.symm.subst zero_nonempty
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      s✝ s₁ s₂ : Set α
      t t₁ t₂ : Set β
      a✝ : α
      b : β
      inst✝³ : Zero α
      inst✝² : Zero β
      inst✝¹ : SMul α β
      inst✝ : NoZeroSMulDivisors α β
      a : α
      s : Set β
      h : Eq (HSMul.hSMul a s) 0
      H : And (Ne a 0) (Ne s 0)
      hst : (HSMul.hSMul a s).Nonempty
      ⊢ False
    -/
    rw [Ne, Ne, ← hst.of_image.subset_zero_iff, not_subset] at H
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      s✝ s₁ s₂ : Set α
      t t₁ t₂ : Set β
      a✝ : α
      b : β
      inst✝³ : Zero α
      inst✝² : Zero β
      inst✝¹ : SMul α β
      inst✝ : NoZeroSMulDivisors α β
      a : α
      s : Set β
      h : Eq (HSMul.hSMul a s) 0
      H : And (Not (Eq a 0)) (Exists fun a => And (Membership.mem s a) (Not (Members …
      hst : (HSMul.hSMul a s).Nonempty
      ⊢ False
    -/
    obtain ⟨ha, b, ht, hb⟩ := H
    /-
      case intro.intro.intro
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      s✝ s₁ s₂ : Set α
      t t₁ t₂ : Set β
      a✝ : α
      b✝ : β
      inst✝³ : Zero α
      inst✝² : Zero β
      inst✝¹ : SMul α β
      inst✝ : NoZeroSMulDivisors α β
      a : α
      s : Set β
      h : Eq (HSMul.hSMul a s) 0
      hst : (HSMul.hSMul a s).Nonempty
      ha : Not (Eq a 0)
      b : β
      ht : Membership.mem s b
      hb : Not (Membership.mem 0 b)
      ⊢ False
    -/
    exact (eq_zero_or_eq_zero_of_smul_eq_zero <| h.subset <| smul_mem_smul_set ht).elim ha hb⟩
    /-
      🎉 no goals
    -/


instance [Zero α] [Mul α] [NoZeroDivisors α] : NoZeroDivisors (Set α) :=
  ⟨fun h ↦ eq_zero_or_eq_zero_of_smul_eq_zero h⟩


@[to_additive]
theorem image_smul_distrib [MulOneClass α] [MulOneClass β] [FunLike F α β] [MonoidHomClass F α β]
    (f : F) (a : α) (s : Set α) :
    f '' (a • s) = f a • f '' s :=
  image_comm <| map_mul _ _


@[to_additive]
theorem op_smul_set_smul_eq_smul_smul_set (a : α) (s : Set β) (t : Set γ)
    (h : ∀ (a : α) (b : β) (c : γ), (op a • b) • c = b • a • c) : (op a • s) • t = s • a • t := by
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝² : SMul (MulOpposite α) β
    inst✝¹ : SMul β γ
    inst✝ : SMul α γ
    a : α
    s : Set β
    t : Set γ
    h : ∀ (a : α) (b : β) (c : γ), Eq (HSMul.hSMul (HSMul.hSMul (MulOpposite.op a) …
    ⊢ Eq (HSMul.hSMul (HSMul.hSMul (MulOpposite.op a) s) t) (HSMul.hSMul s (HSMul. …
  -/
  ext
  /-
    case h
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝² : SMul (MulOpposite α) β
    inst✝¹ : SMul β γ
    inst✝ : SMul α γ
    a : α
    s : Set β
    t : Set γ
    h : ∀ (a : α) (b : β) (c : γ), Eq (HSMul.hSMul (HSMul.hSMul (MulOpposite.op a) …
    x✝ : γ
    ⊢ Iff (Membership.mem (HSMul.hSMul (HSMul.hSMul (MulOpposite.op a) s) t) x✝) ( …
  -/
  simp [mem_smul, mem_smul_set, h]
  /-
    🎉 no goals
  -/


                                                                 /-
                                                                   α : Type u_2
                                                                   β : Type u_3
                                                                   inst✝¹ : Zero β
                                                                   inst✝ : SMulZeroClass α β
                                                                   s : Set α
                                                                   ⊢ HasSubset.Subset (HSMul.hSMul s 0) 0
                                                                 -/
theorem smul_zero_subset (s : Set α) : s • (0 : Set β) ⊆ 0 := by simp [subset_def, mem_smul]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem Nonempty.smul_zero (hs : s.Nonempty) : s • (0 : Set β) = 0 :=
                                    /-
                                      α : Type u_2
                                      β : Type u_3
                                      inst✝¹ : Zero β
                                      inst✝ : SMulZeroClass α β
                                      s : Set α
                                      hs : s.Nonempty
                                      ⊢ HasSubset.Subset 0 (HSMul.hSMul s 0)
                                    -/
  s.smul_zero_subset.antisymm <| by simpa [mem_smul] using hs
                                    /-
                                      🎉 no goals
                                    -/


theorem zero_mem_smul_set (h : (0 : β) ∈ t) : (0 : β) ∈ a • t := ⟨0, h, smul_zero _⟩


theorem zero_mem_smul_set_iff (ha : a ≠ 0) : (0 : β) ∈ a • t ↔ (0 : β) ∈ t := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝³ : Zero β
    inst✝² : SMulZeroClass α β
    t : Set β
    a : α
    inst✝¹ : Zero α
    inst✝ : NoZeroSMulDivisors α β
    ha : Ne a 0
    ⊢ Iff (Membership.mem (HSMul.hSMul a t) 0) (Membership.mem t 0)
  -/
  refine ⟨?_, zero_mem_smul_set⟩
  /-
    α : Type u_2
    β : Type u_3
    inst✝³ : Zero β
    inst✝² : SMulZeroClass α β
    t : Set β
    a : α
    inst✝¹ : Zero α
    inst✝ : NoZeroSMulDivisors α β
    ha : Ne a 0
    ⊢ Membership.mem (HSMul.hSMul a t) 0 → Membership.mem t 0
  -/
  rintro ⟨b, hb, h⟩
  /-
    case intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : Zero β
    inst✝² : SMulZeroClass α β
    t : Set β
    a : α
    inst✝¹ : Zero α
    inst✝ : NoZeroSMulDivisors α β
    ha : Ne a 0
    b : β
    hb : Membership.mem t b
    h : Eq ((fun x => HSMul.hSMul a x) b) 0
    ⊢ Membership.mem t 0
  -/
  rwa [(eq_zero_or_eq_zero_of_smul_eq_zero h).resolve_left ha] at hb
  /-
    🎉 no goals
  -/


                                                                 /-
                                                                   α : Type u_2
                                                                   β : Type u_3
                                                                   inst✝² : Zero α
                                                                   inst✝¹ : Zero β
                                                                   inst✝ : SMulWithZero α β
                                                                   t : Set β
                                                                   ⊢ HasSubset.Subset (HSMul.hSMul 0 t) 0
                                                                 -/
theorem zero_smul_subset (t : Set β) : (0 : Set α) • t ⊆ 0 := by simp [subset_def, mem_smul]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem Nonempty.zero_smul (ht : t.Nonempty) : (0 : Set α) • t = 0 :=
                                    /-
                                      α : Type u_2
                                      β : Type u_3
                                      inst✝² : Zero α
                                      inst✝¹ : Zero β
                                      inst✝ : SMulWithZero α β
                                      t : Set β
                                      ht : t.Nonempty
                                      ⊢ HasSubset.Subset 0 (HSMul.hSMul 0 t)
                                    -/
  t.zero_smul_subset.antisymm <| by simpa [mem_smul] using ht
                                    /-
                                      🎉 no goals
                                    -/


/-- A nonempty set is scaled by zero to the singleton set containing 0. -/
@[simp] theorem zero_smul_set {s : Set β} (h : s.Nonempty) : (0 : α) • s = (0 : Set β) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : Zero α
    inst✝¹ : Zero β
    inst✝ : SMulWithZero α β
    s : Set β
    h : s.Nonempty
    ⊢ Eq (HSMul.hSMul 0 s) 0
  -/
  simp only [← image_smul, image_eta, zero_smul, h.image_const, singleton_zero]
  /-
    🎉 no goals
  -/


theorem zero_smul_set_subset (s : Set β) : (0 : α) • s ⊆ 0 :=
  image_subset_iff.2 fun x _ ↦ zero_smul α x


theorem subsingleton_zero_smul_set (s : Set β) : ((0 : α) • s).Subsingleton :=
  subsingleton_singleton.anti <| zero_smul_set_subset s


theorem zero_mem_smul_iff :
    (0 : β) ∈ s • t ↔ (0 : α) ∈ s ∧ t.Nonempty ∨ (0 : β) ∈ t ∧ s.Nonempty := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝³ : Zero α
    inst✝² : Zero β
    inst✝¹ : SMulWithZero α β
    s : Set α
    t : Set β
    inst✝ : NoZeroSMulDivisors α β
    ⊢ Iff (Membership.mem (HSMul.hSMul s t) 0) (Or (And (Membership.mem s 0) t.Non …
  -/
  constructor
    /-
      case mp
      α : Type u_2
      β : Type u_3
      inst✝³ : Zero α
      inst✝² : Zero β
      inst✝¹ : SMulWithZero α β
      s : Set α
      t : Set β
      inst✝ : NoZeroSMulDivisors α β
      ⊢ Membership.mem (HSMul.hSMul s t) 0 → Or (And (Membership.mem s 0) t.Nonempty …
    -/
  · rintro ⟨a, ha, b, hb, h⟩
    /-
      case mp.intro.intro.intro.intro
      α : Type u_2
      β : Type u_3
      inst✝³ : Zero α
      inst✝² : Zero β
      inst✝¹ : SMulWithZero α β
      s : Set α
      t : Set β
      inst✝ : NoZeroSMulDivisors α β
      a : α
      ha : Membership.mem s a
      b : β
      hb : Membership.mem t b
      h : Eq ((fun x1 x2 => HSMul.hSMul x1 x2) a b) 0
      ⊢ Or (And (Membership.mem s 0) t.Nonempty) (And (Membership.mem t 0) s.Nonempty)
    -/
    obtain rfl | rfl := eq_zero_or_eq_zero_of_smul_eq_zero h
      /-
        case mp.intro.intro.intro.intro.inl
        α : Type u_2
        β : Type u_3
        inst✝³ : Zero α
        inst✝² : Zero β
        inst✝¹ : SMulWithZero α β
        s : Set α
        t : Set β
        inst✝ : NoZeroSMulDivisors α β
        b : β
        hb : Membership.mem t b
        ha : Membership.mem s 0
        h : Eq ((fun x1 x2 => HSMul.hSMul x1 x2) 0 b) 0
        ⊢ Or (And (Membership.mem s 0) t.Nonempty) (And (Membership.mem t 0) s.Nonempty)
      -/
    · exact Or.inl ⟨ha, b, hb⟩
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.intro.inr
        α : Type u_2
        β : Type u_3
        inst✝³ : Zero α
        inst✝² : Zero β
        inst✝¹ : SMulWithZero α β
        s : Set α
        t : Set β
        inst✝ : NoZeroSMulDivisors α β
        a : α
        ha : Membership.mem s a
        hb : Membership.mem t 0
        h : Eq ((fun x1 x2 => HSMul.hSMul x1 x2) a 0) 0
        ⊢ Or (And (Membership.mem s 0) t.Nonempty) (And (Membership.mem t 0) s.Nonempty)
      -/
    · exact Or.inr ⟨hb, a, ha⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u_2
      β : Type u_3
      inst✝³ : Zero α
      inst✝² : Zero β
      inst✝¹ : SMulWithZero α β
      s : Set α
      t : Set β
      inst✝ : NoZeroSMulDivisors α β
      ⊢ Or (And (Membership.mem s 0) t.Nonempty) (And (Membership.mem t 0) s.Nonempt …
    -/
  · rintro (⟨hs, b, hb⟩ | ⟨ht, a, ha⟩)
      /-
        case mpr.inl.intro.intro
        α : Type u_2
        β : Type u_3
        inst✝³ : Zero α
        inst✝² : Zero β
        inst✝¹ : SMulWithZero α β
        s : Set α
        t : Set β
        inst✝ : NoZeroSMulDivisors α β
        hs : Membership.mem s 0
        b : β
        hb : Membership.mem t b
        ⊢ Membership.mem (HSMul.hSMul s t) 0
      -/
    · exact ⟨0, hs, b, hb, zero_smul _ _⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr.intro.intro
        α : Type u_2
        β : Type u_3
        inst✝³ : Zero α
        inst✝² : Zero β
        inst✝¹ : SMulWithZero α β
        s : Set α
        t : Set β
        inst✝ : NoZeroSMulDivisors α β
        ht : Membership.mem t 0
        a : α
        ha : Membership.mem s a
        ⊢ Membership.mem (HSMul.hSMul s t) 0
      -/
    · exact ⟨a, ha, 0, ht, smul_zero _⟩
      /-
        🎉 no goals
      -/


@[to_additive]
theorem op_smul_set_mul_eq_mul_smul_set (a : α) (s : Set α) (t : Set α) :
    op a • s * t = s * a • t :=
  op_smul_set_smul_eq_smul_smul_set _ _ _ fun _ _ _ => mul_assoc _ _ _


@[to_additive]
theorem pairwiseDisjoint_smul_iff :
    s.PairwiseDisjoint (· • t) ↔ (s ×ˢ t).InjOn fun p ↦ p.1 * p.2 :=
  pairwiseDisjoint_image_right_iff fun _ _ ↦ mul_right_injective _


@[to_additive (attr := simp)]
theorem smul_mem_smul_set_iff : a • x ∈ a • s ↔ x ∈ s :=
  (MulAction.injective _).mem_set_image


@[to_additive]
theorem mem_smul_set_iff_inv_smul_mem : x ∈ a • A ↔ a⁻¹ • x ∈ A :=
  show x ∈ MulAction.toPerm a '' A ↔ _ from mem_image_equiv


@[to_additive]
theorem mem_inv_smul_set_iff : x ∈ a⁻¹ • A ↔ a • x ∈ A := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : Group α
    inst✝ : MulAction α β
    A : Set β
    a : α
    x : β
    ⊢ Iff (Membership.mem (HSMul.hSMul (Inv.inv a) A) x) (Membership.mem A (HSMul. …
  -/
  simp only [← image_smul, mem_image, inv_smul_eq_iff, exists_eq_right]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma mem_smul_set_inv {s : Set α} : a ∈ b • s⁻¹ ↔ b ∈ a • s := by
  /-
    α : Type u_2
    inst✝ : Group α
    a b : α
    s : Set α
    ⊢ Iff (Membership.mem (HSMul.hSMul b (Inv.inv s)) a) (Membership.mem (HSMul.hS …
  -/
  simp [mem_smul_set_iff_inv_smul_mem]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem preimage_smul (a : α) (t : Set β) : (fun x ↦ a • x) ⁻¹' t = a⁻¹ • t :=
  ((MulAction.toPerm a).symm.image_eq_preimage _).symm


@[to_additive]
theorem preimage_smul_inv (a : α) (t : Set β) : (fun x ↦ a⁻¹ • x) ⁻¹' t = a • t :=
  preimage_smul (toUnits a)⁻¹ t


@[to_additive (attr := simp)]
theorem set_smul_subset_set_smul_iff : a • A ⊆ a • B ↔ A ⊆ B :=
  image_subset_image_iff <| MulAction.injective _


@[to_additive]
theorem set_smul_subset_iff : a • A ⊆ B ↔ A ⊆ a⁻¹ • B :=
  image_subset_iff.trans <|
    iff_of_eq <| congr_arg _ <| preimage_equiv_eq_image_symm _ <| MulAction.toPerm _


@[to_additive]
theorem subset_set_smul_iff : A ⊆ a • B ↔ a⁻¹ • A ⊆ B :=
  Iff.symm <|
    image_subset_iff.trans <|
      Iff.symm <| iff_of_eq <| congr_arg _ <| image_equiv_eq_preimage_symm _ <| MulAction.toPerm _


@[to_additive]
theorem smul_set_inter : a • (s ∩ t) = a • s ∩ a • t :=
  image_inter <| MulAction.injective a


@[to_additive]
theorem smul_set_iInter {ι : Sort*}
    (a : α) (t : ι → Set β) : (a • ⋂ i, t i) = ⋂ i, a • t i :=
  image_iInter (MulAction.bijective a) t


@[to_additive]
theorem smul_set_sdiff : a • (s \ t) = a • s \ a • t :=
  image_diff (MulAction.injective a) _ _


open scoped symmDiff in
@[to_additive]
theorem smul_set_symmDiff : a • s ∆ t = (a • s) ∆ (a • t) :=
  image_symmDiff (MulAction.injective a) _ _


@[to_additive (attr := simp)]
theorem smul_set_univ : a • (univ : Set β) = univ :=
  image_univ_of_surjective <| MulAction.surjective a


@[to_additive (attr := simp)]
theorem smul_univ {s : Set α} (hs : s.Nonempty) : s • (univ : Set β) = univ :=
  let ⟨a, ha⟩ := hs
  eq_univ_of_forall fun b ↦ ⟨a, ha, a⁻¹ • b, trivial, smul_inv_smul _ _⟩


@[to_additive]
theorem smul_set_compl : a • sᶜ = (a • s)ᶜ := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : Group α
    inst✝ : MulAction α β
    s : Set β
    a : α
    ⊢ Eq (HSMul.hSMul a (HasCompl.compl s)) (HasCompl.compl (HSMul.hSMul a s))
  -/
  simp_rw [Set.compl_eq_univ_diff, smul_set_sdiff, smul_set_univ]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem smul_inter_ne_empty_iff {s t : Set α} {x : α} :
    x • s ∩ t ≠ ∅ ↔ ∃ a b, (a ∈ t ∧ b ∈ s) ∧ a * b⁻¹ = x := by
  /-
    α : Type u_2
    inst✝ : Group α
    s t : Set α
    x : α
    ⊢ Iff (Ne (Inter.inter (HSMul.hSMul x s) t) EmptyCollection.emptyCollection) ( …
  -/
  rw [← nonempty_iff_ne_empty]
  /-
    α : Type u_2
    inst✝ : Group α
    s t : Set α
    x : α
    ⊢ Iff (Inter.inter (HSMul.hSMul x s) t).Nonempty (Exists fun a => Exists fun b …
  -/
  constructor
    /-
      case mp
      α : Type u_2
      inst✝ : Group α
      s t : Set α
      x : α
      ⊢ (Inter.inter (HSMul.hSMul x s) t).Nonempty → Exists fun a => Exists fun b => …
    -/
  · rintro ⟨a, h, ha⟩
    /-
      case mp.intro.intro
      α : Type u_2
      inst✝ : Group α
      s t : Set α
      x a : α
      h : Membership.mem (HSMul.hSMul x s) a
      ha : Membership.mem t a
      ⊢ Exists fun a => Exists fun b => And (And (Membership.mem t a) (Membership.me …
    -/
    obtain ⟨b, hb, rfl⟩ := mem_smul_set.mp h
    /-
      case mp.intro.intro.intro.intro
      α : Type u_2
      inst✝ : Group α
      s t : Set α
      x b : α
      hb : Membership.mem s b
      h : Membership.mem (HSMul.hSMul x s) (HSMul.hSMul x b)
      ha : Membership.mem t (HSMul.hSMul x b)
      ⊢ Exists fun a => Exists fun b => And (And (Membership.mem t a) (Membership.me …
    -/
    exact ⟨x • b, b, ⟨ha, hb⟩, by simp⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_2
      inst✝ : Group α
      s t : Set α
      x : α
      ⊢ (Exists fun a => Exists fun b => And (And (Membership.mem t a) (Membership.m …
    -/
  · rintro ⟨a, b, ⟨ha, hb⟩, rfl⟩
    /-
      case mpr.intro.intro.intro.intro
      α : Type u_2
      inst✝ : Group α
      s t : Set α
      a b : α
      ha : Membership.mem t a
      hb : Membership.mem s b
      ⊢ (Inter.inter (HSMul.hSMul (HMul.hMul a (Inv.inv b)) s) t).Nonempty
    -/
    exact ⟨a, mem_inter (mem_smul_set.mpr ⟨b, hb, by simp⟩) ha⟩
    /-
      🎉 no goals
    -/


@[to_additive]
theorem smul_inter_ne_empty_iff' {s t : Set α} {x : α} :
    x • s ∩ t ≠ ∅ ↔ ∃ a b, (a ∈ t ∧ b ∈ s) ∧ a / b = x := by
  /-
    α : Type u_2
    inst✝ : Group α
    s t : Set α
    x : α
    ⊢ Iff (Ne (Inter.inter (HSMul.hSMul x s) t) EmptyCollection.emptyCollection) ( …
  -/
  simp_rw [smul_inter_ne_empty_iff, div_eq_mul_inv]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem op_smul_inter_ne_empty_iff {s t : Set α} {x : αᵐᵒᵖ} :
    x • s ∩ t ≠ ∅ ↔ ∃ a b, (a ∈ s ∧ b ∈ t) ∧ a⁻¹ * b = MulOpposite.unop x := by
  /-
    α : Type u_2
    inst✝ : Group α
    s t : Set α
    x : MulOpposite α
    ⊢ Iff (Ne (Inter.inter (HSMul.hSMul x s) t) EmptyCollection.emptyCollection) ( …
  -/
  rw [← nonempty_iff_ne_empty]
  /-
    α : Type u_2
    inst✝ : Group α
    s t : Set α
    x : MulOpposite α
    ⊢ Iff (Inter.inter (HSMul.hSMul x s) t).Nonempty (Exists fun a => Exists fun b …
  -/
  constructor
    /-
      case mp
      α : Type u_2
      inst✝ : Group α
      s t : Set α
      x : MulOpposite α
      ⊢ (Inter.inter (HSMul.hSMul x s) t).Nonempty → Exists fun a => Exists fun b => …
    -/
  · rintro ⟨a, h, ha⟩
    /-
      case mp.intro.intro
      α : Type u_2
      inst✝ : Group α
      s t : Set α
      x : MulOpposite α
      a : α
      h : Membership.mem (HSMul.hSMul x s) a
      ha : Membership.mem t a
      ⊢ Exists fun a => Exists fun b => And (And (Membership.mem s a) (Membership.me …
    -/
    obtain ⟨b, hb, rfl⟩ := mem_smul_set.mp h
    /-
      case mp.intro.intro.intro.intro
      α : Type u_2
      inst✝ : Group α
      s t : Set α
      x : MulOpposite α
      b : α
      hb : Membership.mem s b
      h : Membership.mem (HSMul.hSMul x s) (HSMul.hSMul x b)
      ha : Membership.mem t (HSMul.hSMul x b)
      ⊢ Exists fun a => Exists fun b => And (And (Membership.mem s a) (Membership.me …
    -/
    exact ⟨b, x • b, ⟨hb, ha⟩, by simp⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_2
      inst✝ : Group α
      s t : Set α
      x : MulOpposite α
      ⊢ (Exists fun a => Exists fun b => And (And (Membership.mem s a) (Membership.m …
    -/
  · rintro ⟨a, b, ⟨ha, hb⟩, H⟩
    /-
      case mpr.intro.intro.intro.intro
      α : Type u_2
      inst✝ : Group α
      s t : Set α
      x : MulOpposite α
      a b : α
      H : Eq (HMul.hMul (Inv.inv a) b) (MulOpposite.unop x)
      ha : Membership.mem s a
      hb : Membership.mem t b
      ⊢ (Inter.inter (HSMul.hSMul x s) t).Nonempty
    -/
    have : MulOpposite.op (a⁻¹ * b) = x := congr_arg MulOpposite.op H
    /-
      case mpr.intro.intro.intro.intro
      α : Type u_2
      inst✝ : Group α
      s t : Set α
      x : MulOpposite α
      a b : α
      H : Eq (HMul.hMul (Inv.inv a) b) (MulOpposite.unop x)
      ha : Membership.mem s a
      hb : Membership.mem t b
      this : Eq (MulOpposite.op (HMul.hMul (Inv.inv a) b)) x
      ⊢ (Inter.inter (HSMul.hSMul x s) t).Nonempty
    -/
    exact ⟨b, mem_inter (mem_smul_set.mpr ⟨a, ha, by simp [← this]⟩) hb⟩
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
theorem iUnion_inv_smul : ⋃ g : α, g⁻¹ • s = ⋃ g : α, g • s :=
  (Function.Surjective.iSup_congr _ inv_surjective) fun _ ↦ rfl


@[to_additive]
theorem iUnion_smul_eq_setOf_exists {s : Set β} : ⋃ g : α, g • s = { a | ∃ g : α, g • a ∈ s } := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : Group α
    inst✝ : MulAction α β
    s : Set β
    ⊢ Eq (Set.iUnion fun g => HSMul.hSMul g s) (setOf fun a => Exists fun g => Mem …
  -/
  simp_rw [← iUnion_setOf, ← iUnion_inv_smul, ← preimage_smul, preimage]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma inv_smul_set_distrib (a : α) (s : Set α) : (a • s)⁻¹ = op a⁻¹ • s⁻¹ := by
  /-
    α : Type u_2
    inst✝ : Group α
    a : α
    s : Set α
    ⊢ Eq (Inv.inv (HSMul.hSMul a s)) (HSMul.hSMul (MulOpposite.op (Inv.inv a)) (In …
  -/
  ext; simp [mem_smul_set_iff_inv_smul_mem]
       /-
         🎉 no goals
       -/


@[to_additive (attr := simp)]
lemma inv_op_smul_set_distrib (a : α) (s : Set α) : (op a • s)⁻¹ = a⁻¹ • s⁻¹ := by
  /-
    α : Type u_2
    inst✝ : Group α
    a : α
    s : Set α
    ⊢ Eq (Inv.inv (HSMul.hSMul (MulOpposite.op a) s)) (HSMul.hSMul (Inv.inv a) (In …
  -/
  ext; simp [mem_smul_set_iff_inv_smul_mem]
       /-
         🎉 no goals
       -/


@[to_additive (attr := simp)]
lemma disjoint_smul_set : Disjoint (a • s) (a • t) ↔ Disjoint s t :=
  disjoint_image_iff <| MulAction.injective _


@[to_additive]
lemma disjoint_smul_set_left : Disjoint (a • s) t ↔ Disjoint s (a⁻¹ • t) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : Group α
    inst✝ : MulAction α β
    s t : Set β
    a : α
    ⊢ Iff (Disjoint (HSMul.hSMul a s) t) (Disjoint s (HSMul.hSMul (Inv.inv a) t))
  -/
  simpa using disjoint_smul_set (a := a) (t := a⁻¹ • t)
  /-
    🎉 no goals
  -/


@[to_additive]
lemma disjoint_smul_set_right : Disjoint s (a • t) ↔ Disjoint (a⁻¹ • s) t := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : Group α
    inst✝ : MulAction α β
    s t : Set β
    a : α
    ⊢ Iff (Disjoint s (HSMul.hSMul a t)) (Disjoint (HSMul.hSMul (Inv.inv a) s) t)
  -/
  simpa using disjoint_smul_set (a := a) (s := a⁻¹ • s)
  /-
    🎉 no goals
  -/


@[to_additive] alias smul_set_disjoint_iff := disjoint_smul_set

-- `alias` doesn't add the deprecation suggestion to the `to_additive` version
-- see https://github.com/leanprover-community/mathlib4/issues/19424

/-- Any intersection of translates of two sets `s` and `t` can be covered by a single translate of
`(s⁻¹ * s) ∩ (t⁻¹ * t)`.

This is useful to show that the intersection of approximate subgroups is an approximate subgroup. -/
@[to_additive
"Any intersection of translates of two sets `s` and `t` can be covered by a single translate of
`(-s + s) ∩ (-t + t)`.

This is useful to show that the intersection of approximate subgroups is an approximate subgroup."]
lemma exists_smul_inter_smul_subset_smul_inv_mul_inter_inv_mul (s t : Set α) (a b : α) :
    ∃ z : α, a • s ∩ b • t ⊆ z • ((s⁻¹ * s) ∩ (t⁻¹ * t)) := by
  /-
    α : Type u_2
    inst✝ : Group α
    s t : Set α
    a b : α
    ⊢ Exists fun z => HasSubset.Subset (Inter.inter (HSMul.hSMul a s) (HSMul.hSMul …
  -/
  obtain hAB | ⟨z, hzA, hzB⟩ := (a • s ∩ b • t).eq_empty_or_nonempty
    /-
      case inl
      α : Type u_2
      inst✝ : Group α
      s t : Set α
      a b : α
      hAB : Eq (Inter.inter (HSMul.hSMul a s) (HSMul.hSMul b t)) EmptyCollection.emp …
      ⊢ Exists fun z => HasSubset.Subset (Inter.inter (HSMul.hSMul a s) (HSMul.hSMul …
    -/
  · exact ⟨1, by simp [hAB]⟩
    /-
      🎉 no goals
    -/
  /-
    case inr.intro.intro
    α : Type u_2
    inst✝ : Group α
    s t : Set α
    a b z : α
    hzA : Membership.mem (HSMul.hSMul a s) z
    hzB : Membership.mem (HSMul.hSMul b t) z
    ⊢ Exists fun z => HasSubset.Subset (Inter.inter (HSMul.hSMul a s) (HSMul.hSMul …
  -/
  refine ⟨z, ?_⟩
  calc
    a • s ∩ b • t ⊆ (z • s⁻¹) * s ∩ ((z • t⁻¹) * t) := by
      gcongr <;> apply smul_set_subset_mul <;> simpa
    _ = z • ((s⁻¹ * s) ∩ (t⁻¹ * t)) := by simp_rw [Set.smul_set_inter, smul_mul_assoc]


@[simp] lemma mem_invOf_smul_set [Invertible a] : b ∈ ⅟a • s ↔ a • b ∈ s :=
  mem_inv_smul_set_iff (a := unitOfInvertible a)


@[to_additive]
lemma smul_graphOn (x : α × β) (s : Set α) (f : F) :
    x • s.graphOn f = (x.1 • s).graphOn fun a ↦ x.2 / f x.1 * f a := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : Group α
    inst✝² : CommGroup β
    inst✝¹ : FunLike F α β
    inst✝ : MonoidHomClass F α β
    x : Prod α β
    s : Set α
    f : F
    ⊢ Eq (HSMul.hSMul x (Set.graphOn (⇑f) s)) (Set.graphOn (fun a => HMul.hMul (HD …
  -/
  ext ⟨a, b⟩
  simp [mem_smul_set_iff_inv_smul_mem, Prod.ext_iff, and_comm (a := _ = a), inv_mul_eq_iff_eq_mul,
    mul_left_comm _ _⁻¹, eq_inv_mul_iff_mul_eq, ← mul_div_right_comm, div_eq_iff_eq_mul, mul_comm b]


@[to_additive]
lemma smul_graphOn_univ (x : α × β) (f : F) :
                                                                      /-
                                                                        F : Type u_1
                                                                        α : Type u_2
                                                                        β : Type u_3
                                                                        inst✝³ : Group α
                                                                        inst✝² : CommGroup β
                                                                        inst✝¹ : FunLike F α β
                                                                        inst✝ : MonoidHomClass F α β
                                                                        x : Prod α β
                                                                        f : F
                                                                        ⊢ Eq (HSMul.hSMul x (Set.graphOn (⇑f) Set.univ)) (Set.graphOn (fun a => HMul.h …
                                                                      -/
    x • univ.graphOn f = univ.graphOn fun a ↦ x.2 / f x.1 * f a := by simp [smul_graphOn]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[to_additive] lemma smul_div_smul_comm (a : α) (s : Set α) (b : α) (t : Set α) :
    a • s / b • t = (a / b) • (s / t) := by
  simp_rw [← image_smul, smul_eq_mul, ← singleton_mul, mul_div_mul_comm _ s,
    singleton_div_singleton]


@[simp]
theorem smul_mem_smul_set_iff₀ (ha : a ≠ 0) (A : Set β) (x : β) : a • x ∈ a • A ↔ x ∈ A :=
  show Units.mk0 a ha • _ ∈ _ ↔ _ from smul_mem_smul_set_iff


theorem mem_smul_set_iff_inv_smul_mem₀ (ha : a ≠ 0) (A : Set β) (x : β) : x ∈ a • A ↔ a⁻¹ • x ∈ A :=
  show _ ∈ Units.mk0 a ha • _ ↔ _ from mem_smul_set_iff_inv_smul_mem


theorem mem_inv_smul_set_iff₀ (ha : a ≠ 0) (A : Set β) (x : β) : x ∈ a⁻¹ • A ↔ a • x ∈ A :=
  show _ ∈ (Units.mk0 a ha)⁻¹ • _ ↔ _ from mem_inv_smul_set_iff


theorem preimage_smul₀ (ha : a ≠ 0) (t : Set β) : (fun x ↦ a • x) ⁻¹' t = a⁻¹ • t :=
  preimage_smul (Units.mk0 a ha) t


theorem preimage_smul_inv₀ (ha : a ≠ 0) (t : Set β) : (fun x ↦ a⁻¹ • x) ⁻¹' t = a • t :=
  preimage_smul (Units.mk0 a ha)⁻¹ t


@[simp]
theorem set_smul_subset_set_smul_iff₀ (ha : a ≠ 0) {A B : Set β} : a • A ⊆ a • B ↔ A ⊆ B :=
  show Units.mk0 a ha • _ ⊆ _ ↔ _ from set_smul_subset_set_smul_iff


theorem set_smul_subset_iff₀ (ha : a ≠ 0) {A B : Set β} : a • A ⊆ B ↔ A ⊆ a⁻¹ • B :=
  show Units.mk0 a ha • _ ⊆ _ ↔ _ from set_smul_subset_iff


theorem subset_set_smul_iff₀ (ha : a ≠ 0) {A B : Set β} : A ⊆ a • B ↔ a⁻¹ • A ⊆ B :=
  show _ ⊆ Units.mk0 a ha • _ ↔ _ from subset_set_smul_iff


theorem smul_set_inter₀ (ha : a ≠ 0) : a • (s ∩ t) = a • s ∩ a • t :=
  show Units.mk0 a ha • _ = _ from smul_set_inter


theorem smul_set_sdiff₀ (ha : a ≠ 0) : a • (s \ t) = a • s \ a • t :=
  image_diff (MulAction.injective₀ ha) _ _


open scoped symmDiff in
theorem smul_set_symmDiff₀ (ha : a ≠ 0) : a • s ∆ t = (a • s) ∆ (a • t) :=
  image_symmDiff (MulAction.injective₀ ha) _ _


theorem smul_set_univ₀ (ha : a ≠ 0) : a • (univ : Set β) = univ :=
  image_univ_of_surjective <| MulAction.surjective₀ ha


theorem smul_univ₀ {s : Set α} (hs : ¬s ⊆ 0) : s • (univ : Set β) = univ :=
  let ⟨a, ha, ha₀⟩ := not_subset.1 hs
  eq_univ_of_forall fun b ↦ ⟨a, ha, a⁻¹ • b, trivial, smul_inv_smul₀ ha₀ _⟩


theorem smul_univ₀' {s : Set α} (hs : s.Nontrivial) : s • (univ : Set β) = univ :=
  smul_univ₀ hs.not_subset_singleton


                                                           /-
                                                             α : Type u_2
                                                             inst✝ : GroupWithZero α
                                                             ⊢ Eq (Inv.inv 0) 0
                                                           -/
@[simp] protected lemma inv_zero : (0 : Set α)⁻¹ = 0 := by ext; simp
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp] lemma inv_smul_set_distrib₀ (a : α) (s : Set α) : (a • s)⁻¹ = op a⁻¹ • s⁻¹ := by
  /-
    α : Type u_2
    inst✝ : GroupWithZero α
    a : α
    s : Set α
    ⊢ Eq (Inv.inv (HSMul.hSMul a s)) (HSMul.hSMul (MulOpposite.op (Inv.inv a)) (In …
  -/
  obtain rfl | ha := eq_or_ne a 0
    /-
      case inl
      α : Type u_2
      inst✝ : GroupWithZero α
      s : Set α
      ⊢ Eq (Inv.inv (HSMul.hSMul 0 s)) (HSMul.hSMul (MulOpposite.op (Inv.inv 0)) (In …
    -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  · obtain rfl | hs := s.eq_empty_or_nonempty <;> simp [*]
                                                  /-
                                                    🎉 no goals
                                                  -/
    /-
      case inr
      α : Type u_2
      inst✝ : GroupWithZero α
      a : α
      s : Set α
      ha : Ne a 0
      ⊢ Eq (Inv.inv (HSMul.hSMul a s)) (HSMul.hSMul (MulOpposite.op (Inv.inv a)) (In …
    -/
  · ext; simp [mem_smul_set_iff_inv_smul_mem₀, *]
         /-
           🎉 no goals
         -/


@[simp] lemma inv_op_smul_set_distrib₀ (a : α) (s : Set α) : (op a • s)⁻¹ = a⁻¹ • s⁻¹ := by
  /-
    α : Type u_2
    inst✝ : GroupWithZero α
    a : α
    s : Set α
    ⊢ Eq (Inv.inv (HSMul.hSMul (MulOpposite.op a) s)) (HSMul.hSMul (Inv.inv a) (In …
  -/
  obtain rfl | ha := eq_or_ne a 0
    /-
      case inl
      α : Type u_2
      inst✝ : GroupWithZero α
      s : Set α
      ⊢ Eq (Inv.inv (HSMul.hSMul (MulOpposite.op 0) s)) (HSMul.hSMul (Inv.inv 0) (In …
    -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  · obtain rfl | hs := s.eq_empty_or_nonempty <;> simp [*]
                                                  /-
                                                    🎉 no goals
                                                  -/
    /-
      case inr
      α : Type u_2
      inst✝ : GroupWithZero α
      a : α
      s : Set α
      ha : Ne a 0
      ⊢ Eq (Inv.inv (HSMul.hSMul (MulOpposite.op a) s)) (HSMul.hSMul (Inv.inv a) (In …
    -/
  · ext; simp [mem_smul_set_iff_inv_smul_mem₀, *]
         /-
           🎉 no goals
         -/


@[simp]
theorem smul_set_neg : a • -t = -(a • t) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : Monoid α
    inst✝¹ : AddGroup β
    inst✝ : DistribMulAction α β
    a : α
    t : Set β
    ⊢ Eq (HSMul.hSMul a (Neg.neg t)) (Neg.neg (HSMul.hSMul a t))
  -/
  simp_rw [← image_smul, ← image_neg_eq_neg, image_image, smul_neg]
  /-
    🎉 no goals
  -/


@[simp]
protected theorem smul_neg : s • -t = -(s • t) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : Monoid α
    inst✝¹ : AddGroup β
    inst✝ : DistribMulAction α β
    s : Set α
    t : Set β
    ⊢ Eq (HSMul.hSMul s (Neg.neg t)) (Neg.neg (HSMul.hSMul s t))
  -/
  simp_rw [← image_neg_eq_neg]
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : Monoid α
    inst✝¹ : AddGroup β
    inst✝ : DistribMulAction α β
    s : Set α
    t : Set β
    ⊢ Eq (HSMul.hSMul s (Set.image (fun x => Neg.neg x) t)) (Set.image (fun x => N …
  -/
  exact image_image2_right_comm smul_neg
  /-
    🎉 no goals
  -/


theorem add_smul_subset (a b : α) (s : Set β) : (a + b) • s ⊆ a • s + b • s := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : Semiring α
    inst✝¹ : AddCommMonoid β
    inst✝ : Module α β
    a b : α
    s : Set β
    ⊢ HasSubset.Subset (HSMul.hSMul (HAdd.hAdd a b) s) (HAdd.hAdd (HSMul.hSMul a s …
  -/
  rintro _ ⟨x, hx, rfl⟩
  /-
    case intro.intro
    α : Type u_2
    β : Type u_3
    inst✝² : Semiring α
    inst✝¹ : AddCommMonoid β
    inst✝ : Module α β
    a b : α
    s : Set β
    x : β
    hx : Membership.mem s x
    ⊢ Membership.mem (HAdd.hAdd (HSMul.hSMul a s) (HSMul.hSMul b s)) ((fun x => HS …
  -/
  simpa only [add_smul] using add_mem_add (smul_mem_smul_set hx) (smul_mem_smul_set hx)
  /-
    🎉 no goals
  -/


@[simp]
theorem neg_smul_set : -a • t = -(a • t) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : Ring α
    inst✝¹ : AddCommGroup β
    inst✝ : Module α β
    a : α
    t : Set β
    ⊢ Eq (HSMul.hSMul (Neg.neg a) t) (Neg.neg (HSMul.hSMul a t))
  -/
  simp_rw [← image_smul, ← image_neg_eq_neg, image_image, neg_smul]
  /-
    🎉 no goals
  -/


@[simp]
protected theorem neg_smul : -s • t = -(s • t) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : Ring α
    inst✝¹ : AddCommGroup β
    inst✝ : Module α β
    s : Set α
    t : Set β
    ⊢ Eq (HSMul.hSMul (Neg.neg s) t) (Neg.neg (HSMul.hSMul s t))
  -/
  simp_rw [← image_neg_eq_neg]
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : Ring α
    inst✝¹ : AddCommGroup β
    inst✝ : Module α β
    s : Set α
    t : Set β
    ⊢ Eq (HSMul.hSMul (Set.image (fun x => Neg.neg x) s) t) (Set.image (fun x => N …
  -/
  exact image2_image_left_comm neg_smul
  /-
    🎉 no goals
  -/



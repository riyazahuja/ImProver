/-- If scalar multiplication by elements of `α` sends `(0 : β)` to zero,
then the same is true for `(0 : Finset β)`. -/
protected def smulZeroClass [Zero β] [SMulZeroClass α β] : SMulZeroClass α (Finset β) :=
  coe_injective.smulZeroClass ⟨toSet, coe_zero⟩ coe_smul_finset


/-- If the scalar multiplication `(· • ·) : α → β → β` is distributive,
then so is `(· • ·) : α → Finset β → Finset β`. -/
protected def distribSMul [AddZeroClass β] [DistribSMul α β] : DistribSMul α (Finset β) :=
  coe_injective.distribSMul coeAddMonoidHom coe_smul_finset


/-- A distributive multiplicative action of a monoid on an additive monoid `β` gives a distributive
multiplicative action on `Finset β`. -/
protected def distribMulAction [Monoid α] [AddMonoid β] [DistribMulAction α β] :
    DistribMulAction α (Finset β) :=
  coe_injective.distribMulAction coeAddMonoidHom coe_smul_finset


/-- A multiplicative action of a monoid on a monoid `β` gives a multiplicative action on `Set β`. -/
protected def mulDistribMulAction [Monoid α] [Monoid β] [MulDistribMulAction α β] :
    MulDistribMulAction α (Finset β) :=
  coe_injective.mulDistribMulAction coeMonoidHom coe_smul_finset


instance [DecidableEq α] [Zero α] [Mul α] [NoZeroDivisors α] : NoZeroDivisors (Finset α) :=
  Function.Injective.noZeroDivisors toSet coe_injective coe_zero coe_mul


instance noZeroSMulDivisors [Zero α] [Zero β] [SMul α β] [NoZeroSMulDivisors α β] :
    NoZeroSMulDivisors (Finset α) (Finset β) where
  eq_zero_or_eq_zero_of_smul_eq_zero {s t} := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : DecidableEq β
      inst✝³ : Zero α
      inst✝² : Zero β
      inst✝¹ : SMul α β
      inst✝ : NoZeroSMulDivisors α β
      s : Finset α
      t : Finset β
      ⊢ Eq (HSMul.hSMul s t) 0 → Or (Eq s 0) (Eq t 0)
    -/
    exact_mod_cast eq_zero_or_eq_zero_of_smul_eq_zero (c := s.toSet) (x := t.toSet)
    /-
      🎉 no goals
    -/


instance noZeroSMulDivisors_finset [Zero α] [Zero β] [SMul α β] [NoZeroSMulDivisors α β] :
    NoZeroSMulDivisors α (Finset β) :=
  Function.Injective.noZeroSMulDivisors toSet coe_injective coe_zero coe_smul_finset


                                                                     /-
                                                                       α : Type u_1
                                                                       β : Type u_2
                                                                       inst✝² : DecidableEq β
                                                                       inst✝¹ : Zero β
                                                                       inst✝ : SMulZeroClass α β
                                                                       s : Finset α
                                                                       ⊢ HasSubset.Subset (HSMul.hSMul s 0) 0
                                                                     -/
lemma smul_zero_subset (s : Finset α) : s • (0 : Finset β) ⊆ 0 := by simp [subset_iff, mem_smul]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


lemma Nonempty.smul_zero (hs : s.Nonempty) : s • (0 : Finset β) = 0 :=
                                    /-
                                      α : Type u_1
                                      β : Type u_2
                                      inst✝² : DecidableEq β
                                      inst✝¹ : Zero β
                                      inst✝ : SMulZeroClass α β
                                      s : Finset α
                                      hs : s.Nonempty
                                      ⊢ HasSubset.Subset 0 (HSMul.hSMul s 0)
                                    -/
  s.smul_zero_subset.antisymm <| by simpa [mem_smul] using hs
                                    /-
                                      🎉 no goals
                                    -/


lemma zero_mem_smul_finset (h : (0 : β) ∈ t) : (0 : β) ∈ a • t :=
  mem_smul_finset.2 ⟨0, h, smul_zero _⟩


lemma zero_mem_smul_finset_iff (ha : a ≠ 0) : (0 : β) ∈ a • t ↔ (0 : β) ∈ t := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : DecidableEq β
    inst✝³ : Zero β
    inst✝² : SMulZeroClass α β
    t : Finset β
    a : α
    inst✝¹ : Zero α
    inst✝ : NoZeroSMulDivisors α β
    ha : Ne a 0
    ⊢ Iff (Membership.mem (HSMul.hSMul a t) 0) (Membership.mem t 0)
  -/
  rw [← mem_coe, coe_smul_finset, Set.zero_mem_smul_set_iff ha, mem_coe]
  /-
    🎉 no goals
  -/


                                                                     /-
                                                                       α : Type u_1
                                                                       β : Type u_2
                                                                       inst✝³ : DecidableEq β
                                                                       inst✝² : Zero α
                                                                       inst✝¹ : Zero β
                                                                       inst✝ : SMulWithZero α β
                                                                       t : Finset β
                                                                       ⊢ HasSubset.Subset (HSMul.hSMul 0 t) 0
                                                                     -/
lemma zero_smul_subset (t : Finset β) : (0 : Finset α) • t ⊆ 0 := by simp [subset_iff, mem_smul]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


lemma Nonempty.zero_smul (ht : t.Nonempty) : (0 : Finset α) • t = 0 :=
                                    /-
                                      α : Type u_1
                                      β : Type u_2
                                      inst✝³ : DecidableEq β
                                      inst✝² : Zero α
                                      inst✝¹ : Zero β
                                      inst✝ : SMulWithZero α β
                                      t : Finset β
                                      ht : t.Nonempty
                                      ⊢ HasSubset.Subset 0 (HSMul.hSMul 0 t)
                                    -/
  t.zero_smul_subset.antisymm <| by simpa [mem_smul] using ht
                                    /-
                                      🎉 no goals
                                    -/


/-- A nonempty set is scaled by zero to the singleton set containing zero. -/
@[simp] lemma zero_smul_finset {s : Finset β} (h : s.Nonempty) : (0 : α) • s = (0 : Finset β) :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        inst✝³ : DecidableEq β
                        inst✝² : Zero α
                        inst✝¹ : Zero β
                        inst✝ : SMulWithZero α β
                        s : Finset β
                        h : s.Nonempty
                        ⊢ Eq ↑(HSMul.hSMul 0 s) ↑0
                      -/
  coe_injective <| by simpa using @Set.zero_smul_set α _ _ _ _ _ h
                      /-
                        🎉 no goals
                      -/


lemma zero_smul_finset_subset (s : Finset β) : (0 : α) • s ⊆ 0 :=
  image_subset_iff.2 fun x _ ↦ mem_zero.2 <| zero_smul α x


lemma zero_mem_smul_iff :
    (0 : β) ∈ s • t ↔ (0 : α) ∈ s ∧ t.Nonempty ∨ (0 : β) ∈ t ∧ s.Nonempty := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : DecidableEq β
    inst✝³ : Zero α
    inst✝² : Zero β
    inst✝¹ : SMulWithZero α β
    s : Finset α
    t : Finset β
    inst✝ : NoZeroSMulDivisors α β
    ⊢ Iff (Membership.mem (HSMul.hSMul s t) 0) (Or (And (Membership.mem s 0) t.Non …
  -/
  rw [← mem_coe, coe_smul, Set.zero_mem_smul_iff]; rfl
                                                   /-
                                                     🎉 no goals
                                                   -/


                                                       /-
                                                         α : Type u_1
                                                         inst✝¹ : DecidableEq α
                                                         inst✝ : MulZeroClass α
                                                         s : Finset α
                                                         ⊢ HasSubset.Subset (HMul.hMul s 0) 0
                                                       -/
lemma mul_zero_subset (s : Finset α) : s * 0 ⊆ 0 := by simp [subset_iff, mem_mul]
                                                       /-
                                                         🎉 no goals
                                                       -/

                                                       /-
                                                         α : Type u_1
                                                         inst✝¹ : DecidableEq α
                                                         inst✝ : MulZeroClass α
                                                         s : Finset α
                                                         ⊢ HasSubset.Subset (HMul.hMul 0 s) 0
                                                       -/
lemma zero_mul_subset (s : Finset α) : 0 * s ⊆ 0 := by simp [subset_iff, mem_mul]
                                                       /-
                                                         🎉 no goals
                                                       -/


lemma Nonempty.mul_zero (hs : s.Nonempty) : s * 0 = 0 :=
                                   /-
                                     α : Type u_1
                                     inst✝¹ : DecidableEq α
                                     inst✝ : MulZeroClass α
                                     s : Finset α
                                     hs : s.Nonempty
                                     ⊢ HasSubset.Subset 0 (HMul.hMul s 0)
                                   -/
  s.mul_zero_subset.antisymm <| by simpa [mem_mul] using hs
                                   /-
                                     🎉 no goals
                                   -/


lemma Nonempty.zero_mul (hs : s.Nonempty) : 0 * s = 0 :=
                                   /-
                                     α : Type u_1
                                     inst✝¹ : DecidableEq α
                                     inst✝ : MulZeroClass α
                                     s : Finset α
                                     hs : s.Nonempty
                                     ⊢ HasSubset.Subset 0 (HMul.hMul 0 s)
                                   -/
  s.zero_mul_subset.antisymm <| by simpa [mem_mul] using hs
                                   /-
                                     🎉 no goals
                                   -/


@[simp] lemma smul_mem_smul_finset_iff₀ (ha : a ≠ 0) : a • b ∈ a • s ↔ b ∈ s :=
  smul_mem_smul_finset_iff (Units.mk0 a ha)


lemma inv_smul_mem_iff₀ (ha : a ≠ 0) : a⁻¹ • b ∈ s ↔ b ∈ a • s :=
  show _ ↔ _ ∈ Units.mk0 a ha • _ from inv_smul_mem_iff


lemma mem_inv_smul_finset_iff₀ (ha : a ≠ 0) : b ∈ a⁻¹ • s ↔ a • b ∈ s :=
  show _ ∈ (Units.mk0 a ha)⁻¹ • _ ↔ _ from mem_inv_smul_finset_iff


@[simp]
lemma smul_finset_subset_smul_finset_iff₀ (ha : a ≠ 0) : a • s ⊆ a • t ↔ s ⊆ t :=
  show Units.mk0 a ha • _ ⊆ _ ↔ _ from smul_finset_subset_smul_finset_iff


lemma smul_finset_subset_iff₀ (ha : a ≠ 0) : a • s ⊆ t ↔ s ⊆ a⁻¹ • t :=
  show Units.mk0 a ha • _ ⊆ _ ↔ _ from smul_finset_subset_iff


lemma subset_smul_finset_iff₀ (ha : a ≠ 0) : s ⊆ a • t ↔ a⁻¹ • s ⊆ t :=
  show _ ⊆ Units.mk0 a ha • _ ↔ _ from subset_smul_finset_iff


lemma smul_finset_inter₀ (ha : a ≠ 0) : a • (s ∩ t) = a • s ∩ a • t :=
  image_inter _ _ <| MulAction.injective₀ ha


lemma smul_finset_sdiff₀ (ha : a ≠ 0) : a • (s \ t) = a • s \ a • t :=
  image_sdiff _ _ <| MulAction.injective₀ ha


open scoped symmDiff in
lemma smul_finset_symmDiff₀ (ha : a ≠ 0) : a • s ∆ t = (a • s) ∆ (a • t) :=
  image_symmDiff _ _ <| MulAction.injective₀ ha


lemma smul_finset_univ₀ [Fintype β] (ha : a ≠ 0) : a • (univ : Finset β) = univ :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        inst✝³ : DecidableEq β
                        inst✝² : GroupWithZero α
                        inst✝¹ : MulAction α β
                        a : α
                        inst✝ : Fintype β
                        ha : Ne a 0
                        ⊢ Eq ↑(HSMul.hSMul a Finset.univ) ↑Finset.univ
                      -/
  coe_injective <| by push_cast; exact Set.smul_set_univ₀ ha
                                 /-
                                   🎉 no goals
                                 -/


lemma smul_univ₀ [Fintype β] {s : Finset α} (hs : ¬s ⊆ 0) : s • (univ : Finset β) = univ :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : DecidableEq β
      inst✝² : GroupWithZero α
      inst✝¹ : MulAction α β
      inst✝ : Fintype β
      s : Finset α
      hs : Not (HasSubset.Subset s 0)
      ⊢ Eq ↑(HSMul.hSMul s Finset.univ) ↑Finset.univ
    -/
    rw [← coe_subset] at hs
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : DecidableEq β
      inst✝² : GroupWithZero α
      inst✝¹ : MulAction α β
      inst✝ : Fintype β
      s : Finset α
      hs : Not (HasSubset.Subset ↑s ↑0)
      ⊢ Eq ↑(HSMul.hSMul s Finset.univ) ↑Finset.univ
    -/
    push_cast at hs ⊢
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : DecidableEq β
      inst✝² : GroupWithZero α
      inst✝¹ : MulAction α β
      inst✝ : Fintype β
      s : Finset α
      hs : Not (HasSubset.Subset (↑s) 0)
      ⊢ Eq (HSMul.hSMul (↑s) Set.univ) Set.univ
    -/
    exact Set.smul_univ₀ hs
    /-
      🎉 no goals
    -/


lemma smul_univ₀' [Fintype β] {s : Finset α} (hs : s.Nontrivial) : s • (univ : Finset β) = univ :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        inst✝³ : DecidableEq β
                        inst✝² : GroupWithZero α
                        inst✝¹ : MulAction α β
                        inst✝ : Fintype β
                        s : Finset α
                        hs : s.Nontrivial
                        ⊢ Eq ↑(HSMul.hSMul s Finset.univ) ↑Finset.univ
                      -/
  coe_injective <| by push_cast; exact Set.smul_univ₀' hs
                                 /-
                                   🎉 no goals
                                 -/


                                                       /-
                                                         α : Type u_1
                                                         inst✝¹ : GroupWithZero α
                                                         inst✝ : DecidableEq α
                                                         s : Finset α
                                                         ⊢ HasSubset.Subset (HDiv.hDiv s 0) 0
                                                       -/
lemma div_zero_subset (s : Finset α) : s / 0 ⊆ 0 := by simp [subset_iff, mem_div]
                                                       /-
                                                         🎉 no goals
                                                       -/


                                                       /-
                                                         α : Type u_1
                                                         inst✝¹ : GroupWithZero α
                                                         inst✝ : DecidableEq α
                                                         s : Finset α
                                                         ⊢ HasSubset.Subset (HDiv.hDiv 0 s) 0
                                                       -/
lemma zero_div_subset (s : Finset α) : 0 / s ⊆ 0 := by simp [subset_iff, mem_div]
                                                       /-
                                                         🎉 no goals
                                                       -/


lemma Nonempty.div_zero (hs : s.Nonempty) : s / 0 = 0 :=
                                   /-
                                     α : Type u_1
                                     inst✝¹ : GroupWithZero α
                                     inst✝ : DecidableEq α
                                     s : Finset α
                                     hs : s.Nonempty
                                     ⊢ HasSubset.Subset 0 (HDiv.hDiv s 0)
                                   -/
  s.div_zero_subset.antisymm <| by simpa [mem_div] using hs
                                   /-
                                     🎉 no goals
                                   -/


lemma Nonempty.zero_div (hs : s.Nonempty) : 0 / s = 0 :=
                                   /-
                                     α : Type u_1
                                     inst✝¹ : GroupWithZero α
                                     inst✝ : DecidableEq α
                                     s : Finset α
                                     hs : s.Nonempty
                                     ⊢ HasSubset.Subset 0 (HDiv.hDiv 0 s)
                                   -/
  s.zero_div_subset.antisymm <| by simpa [mem_div] using hs
                                   /-
                                     🎉 no goals
                                   -/


                                                              /-
                                                                α : Type u_1
                                                                inst✝¹ : GroupWithZero α
                                                                inst✝ : DecidableEq α
                                                                ⊢ Eq (Inv.inv 0) 0
                                                              -/
@[simp] protected lemma inv_zero : (0 : Finset α)⁻¹ = 0 := by ext; simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp] lemma inv_smul_finset_distrib₀ (a : α) (s : Finset α) : (a • s)⁻¹ = s⁻¹ <• a⁻¹ := by
  /-
    α : Type u_1
    inst✝¹ : GroupWithZero α
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    ⊢ Eq (Inv.inv (HSMul.hSMul a s)) (HSMul.hSMul (MulOpposite.op (Inv.inv a)) (In …
  -/
  obtain rfl | ha := eq_or_ne a 0
    /-
      case inl
      α : Type u_1
      inst✝¹ : GroupWithZero α
      inst✝ : DecidableEq α
      s : Finset α
      ⊢ Eq (Inv.inv (HSMul.hSMul 0 s)) (HSMul.hSMul (MulOpposite.op (Inv.inv 0)) (In …
    -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  · obtain rfl | hs := s.eq_empty_or_nonempty <;> simp [*]
                                                  /-
                                                    🎉 no goals
                                                  -/
  -- was `simp` and very slow (https://github.com/leanprover-community/mathlib4/issues/19751)
    /-
      case inr
      α : Type u_1
      inst✝¹ : GroupWithZero α
      inst✝ : DecidableEq α
      a : α
      s : Finset α
      ha : Ne a 0
      ⊢ Eq (Inv.inv (HSMul.hSMul a s)) (HSMul.hSMul (MulOpposite.op (Inv.inv a)) (In …
    -/
  · ext; simp only [mem_inv', ne_eq, not_false_eq_true, ← inv_smul_mem_iff₀, smul_eq_mul,
      MulOpposite.op_inv, inv_eq_zero, MulOpposite.op_eq_zero_iff, inv_inv,
      MulOpposite.smul_eq_mul_unop, MulOpposite.unop_op, mul_inv_rev, ha]


@[simp] lemma inv_op_smul_finset_distrib₀ (a : α) (s : Finset α) : (s <• a)⁻¹ = a⁻¹ • s⁻¹ := by
  /-
    α : Type u_1
    inst✝¹ : GroupWithZero α
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    ⊢ Eq (Inv.inv (HSMul.hSMul (MulOpposite.op a) s)) (HSMul.hSMul (Inv.inv a) (In …
  -/
  obtain rfl | ha := eq_or_ne a 0
    /-
      case inl
      α : Type u_1
      inst✝¹ : GroupWithZero α
      inst✝ : DecidableEq α
      s : Finset α
      ⊢ Eq (Inv.inv (HSMul.hSMul (MulOpposite.op 0) s)) (HSMul.hSMul (Inv.inv 0) (In …
    -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  · obtain rfl | hs := s.eq_empty_or_nonempty <;> simp [*]
                                                  /-
                                                    🎉 no goals
                                                  -/
  -- was `simp` and very slow (https://github.com/leanprover-community/mathlib4/issues/19751)
    /-
      case inr
      α : Type u_1
      inst✝¹ : GroupWithZero α
      inst✝ : DecidableEq α
      a : α
      s : Finset α
      ha : Ne a 0
      ⊢ Eq (Inv.inv (HSMul.hSMul (MulOpposite.op a) s)) (HSMul.hSMul (Inv.inv a) (In …
    -/
  · ext; simp only [mem_inv', ne_eq, MulOpposite.op_eq_zero_iff, not_false_eq_true, ←
      inv_smul_mem_iff₀, MulOpposite.smul_eq_mul_unop, MulOpposite.unop_inv, MulOpposite.unop_op,
      inv_eq_zero, inv_inv, smul_eq_mul, mul_inv_rev, ha]


@[simp]
lemma smul_finset_neg (a : α) (t : Finset β) : a • -t = -(a • t) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : DecidableEq β
    inst✝² : Monoid α
    inst✝¹ : AddGroup β
    inst✝ : DistribMulAction α β
    a : α
    t : Finset β
    ⊢ Eq (HSMul.hSMul a (Neg.neg t)) (Neg.neg (HSMul.hSMul a t))
  -/
  simp only [← image_smul, ← image_neg_eq_neg, Function.comp_def, image_image, smul_neg]
  /-
    🎉 no goals
  -/


@[simp]
protected lemma smul_neg (s : Finset α) (t : Finset β) : s • -t = -(s • t) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : DecidableEq β
    inst✝² : Monoid α
    inst✝¹ : AddGroup β
    inst✝ : DistribMulAction α β
    s : Finset α
    t : Finset β
    ⊢ Eq (HSMul.hSMul s (Neg.neg t)) (Neg.neg (HSMul.hSMul s t))
  -/
  simp_rw [← image_neg_eq_neg]; exact image_image₂_right_comm smul_neg
                                /-
                                  🎉 no goals
                                -/



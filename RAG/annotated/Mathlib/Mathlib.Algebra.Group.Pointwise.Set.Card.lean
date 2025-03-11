@[to_additive]
lemma _root_.Cardinal.mk_mul_le : #(s * t) ≤ #s * #t := by
  /-
    M : Type u_2
    inst✝ : Mul M
    s t : Set M
    ⊢ LE.le (Cardinal.mk ↑(HMul.hMul s t)) (HMul.hMul (Cardinal.mk ↑s) (Cardinal.m …
  -/
  rw [← image2_mul]; exact Cardinal.mk_image2_le
                     /-
                       🎉 no goals
                     -/


@[to_additive]
lemma natCard_mul_le : Nat.card (s * t) ≤ Nat.card s * Nat.card t := by
  /-
    M : Type u_2
    inst✝¹ : Mul M
    s t : Set M
    inst✝ : IsCancelMul M
    ⊢ LE.le (Nat.card ↑(HMul.hMul s t)) (HMul.hMul (Nat.card ↑s) (Nat.card ↑t))
  -/
  obtain h | h := (s * t).infinite_or_finite
    /-
      case inl
      M : Type u_2
      inst✝¹ : Mul M
      s t : Set M
      inst✝ : IsCancelMul M
      h : (HMul.hMul s t).Infinite
      ⊢ LE.le (Nat.card ↑(HMul.hMul s t)) (HMul.hMul (Nat.card ↑s) (Nat.card ↑t))
    -/
  · simp [Set.Infinite.card_eq_zero h]
    /-
      🎉 no goals
    -/
  /-
    case inr
    M : Type u_2
    inst✝¹ : Mul M
    s t : Set M
    inst✝ : IsCancelMul M
    h : (HMul.hMul s t).Finite
    ⊢ LE.le (Nat.card ↑(HMul.hMul s t)) (HMul.hMul (Nat.card ↑s) (Nat.card ↑t))
  -/
  simp only [Nat.card, ← Cardinal.toNat_mul]
  /-
    case inr
    M : Type u_2
    inst✝¹ : Mul M
    s t : Set M
    inst✝ : IsCancelMul M
    h : (HMul.hMul s t).Finite
    ⊢ LE.le (Cardinal.toNat (Cardinal.mk ↑(HMul.hMul s t))) (Cardinal.toNat (HMul. …
  -/
  refine Cardinal.toNat_le_toNat Cardinal.mk_mul_le ?_
  /-
    case inr
    M : Type u_2
    inst✝¹ : Mul M
    s t : Set M
    inst✝ : IsCancelMul M
    h : (HMul.hMul s t).Finite
    ⊢ LT.lt (HMul.hMul (Cardinal.mk ↑s) (Cardinal.mk ↑t)) Cardinal.aleph0
  -/
  aesop (add simp [Cardinal.mul_lt_aleph0_iff, finite_mul])
  /-
    🎉 no goals
  -/


@[to_additive] alias card_mul_le := natCard_mul_le

-- `alias` doesn't add the deprecation suggestion to the `to_additive` version
-- see https://github.com/leanprover-community/mathlib4/issues/19424

@[to_additive (attr := simp)]
lemma _root_.Cardinal.mk_inv (s : Set G) : #↥(s⁻¹) = #s := by
  /-
    G : Type u_1
    inst✝ : InvolutiveInv G
    s : Set G
    ⊢ Eq (Cardinal.mk ↑(Inv.inv s)) (Cardinal.mk ↑s)
  -/
  rw [← image_inv_eq_inv, Cardinal.mk_image_eq_of_injOn _ _ inv_injective.injOn]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma natCard_inv (s : Set G) : Nat.card ↥(s⁻¹) = Nat.card s := by
  /-
    G : Type u_1
    inst✝ : InvolutiveInv G
    s : Set G
    ⊢ Eq (Nat.card ↑(Inv.inv s)) (Nat.card ↑s)
  -/
  rw [← image_inv_eq_inv, Nat.card_image_of_injective inv_injective]
  /-
    🎉 no goals
  -/


@[to_additive] alias card_inv := natCard_inv

-- `alias` doesn't add the deprecation suggestion to the `to_additive` version
-- see https://github.com/leanprover-community/mathlib4/issues/19424

@[to_additive (attr := simp)]
                                                           /-
                                                             G : Type u_1
                                                             inst✝ : InvolutiveInv G
                                                             s : Set G
                                                             ⊢ Eq (Inv.inv s).encard s.encard
                                                           -/
lemma encard_inv (s : Set G) : s⁻¹.encard = s.encard := by simp [encard, ENat.card]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[to_additive (attr := simp)]
                                                        /-
                                                          G : Type u_1
                                                          inst✝ : InvolutiveInv G
                                                          s : Set G
                                                          ⊢ Eq (Inv.inv s).ncard s.ncard
                                                        -/
lemma ncard_inv (s : Set G) : s⁻¹.ncard = s.ncard := by simp [ncard]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[to_additive]
lemma _root_.Cardinal.mk_div_le : #(s / t) ≤ #s * #t := by
  /-
    M : Type u_2
    inst✝ : DivInvMonoid M
    s t : Set M
    ⊢ LE.le (Cardinal.mk ↑(HDiv.hDiv s t)) (HMul.hMul (Cardinal.mk ↑s) (Cardinal.m …
  -/
  rw [← image2_div]; exact Cardinal.mk_image2_le
                     /-
                       🎉 no goals
                     -/


@[to_additive]
lemma natCard_div_le : Nat.card (s / t) ≤ Nat.card s * Nat.card t := by
  /-
    G : Type u_1
    inst✝ : Group G
    s t : Set G
    ⊢ LE.le (Nat.card ↑(HDiv.hDiv s t)) (HMul.hMul (Nat.card ↑s) (Nat.card ↑t))
  -/
  rw [div_eq_mul_inv, ← natCard_inv t]; exact natCard_mul_le
                                        /-
                                          🎉 no goals
                                        -/


@[to_additive] alias card_div_le := natCard_div_le

-- `alias` doesn't add the deprecation suggestion to the `to_additive` version
-- see https://github.com/leanprover-community/mathlib4/issues/19424

@[to_additive (attr := simp)]
lemma _root_.Cardinal.mk_smul_set (a : G) (s : Set α) : #↥(a • s) = #s :=
  Cardinal.mk_image_eq_of_injOn _ _ (MulAction.injective a).injOn


@[to_additive (attr := simp)]
lemma natCard_smul_set (a : G) (s : Set α) : Nat.card ↥(a • s) = Nat.card s :=
  Nat.card_image_of_injective (MulAction.injective a) _


@[to_additive]
alias card_smul_set := Cardinal.mk_smul_set

-- `alias` doesn't add the deprecation suggestion to the `to_additive` version
-- see https://github.com/leanprover-community/mathlib4/issues/19424

@[to_additive (attr := simp)]
lemma encard_smul_set (a : G) (s : Set α) : (a • s).encard = s.encard := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝¹ : Group G
    inst✝ : MulAction G α
    a : G
    s : Set α
    ⊢ Eq (HSMul.hSMul a s).encard s.encard
  -/
  simp [encard, ENat.card]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
                                                                         /-
                                                                           G : Type u_1
                                                                           α : Type u_3
                                                                           inst✝¹ : Group G
                                                                           inst✝ : MulAction G α
                                                                           a : G
                                                                           s : Set α
                                                                           ⊢ Eq (HSMul.hSMul a s).ncard s.ncard
                                                                         -/
lemma ncard_smul_set (a : G) (s : Set α) : (a • s).ncard = s.ncard := by simp [ncard]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/



/-- An element `p` is said to be idempotent if `p * p = p`
-/
def IsIdempotentElem (p : M) : Prop :=
  p * p = p


theorem of_isIdempotent [Std.IdempotentOp (α := M) (· * ·)] (a : M) : IsIdempotentElem a :=
  Std.IdempotentOp.idempotent a


theorem eq {p : M} (h : IsIdempotentElem p) : p * p = p :=
  h


theorem mul_of_commute {p q : S} (h : Commute p q) (h₁ : IsIdempotentElem p)
    (h₂ : IsIdempotentElem q) : IsIdempotentElem (p * q) := by
  /-
    S : Type u_3
    inst✝ : Semigroup S
    p q : S
    h : Commute p q
    h₁ : IsIdempotentElem p
    h₂ : IsIdempotentElem q
    ⊢ IsIdempotentElem (HMul.hMul p q)
  -/
  rw [IsIdempotentElem, mul_assoc, ← mul_assoc q, ← h.eq, mul_assoc p, h₂.eq, ← mul_assoc, h₁.eq]
  /-
    🎉 no goals
  -/


lemma mul {M} [CommSemigroup M] {e₁ e₂ : M}
    (he₁ : IsIdempotentElem e₁) (he₂ : IsIdempotentElem e₂) : IsIdempotentElem (e₁ * e₂) :=
  he₁.mul_of_commute (.all e₁ e₂) he₂


theorem zero : IsIdempotentElem (0 : M₀) :=
  mul_zero _


theorem one : IsIdempotentElem (1 : M₁) :=
  mul_one _


theorem one_sub {p : R} (h : IsIdempotentElem p) : IsIdempotentElem (1 - p) := by
  /-
    R : Type u_6
    inst✝ : NonAssocRing R
    p : R
    h : IsIdempotentElem p
    ⊢ IsIdempotentElem (HSub.hSub 1 p)
  -/
  rw [IsIdempotentElem, mul_sub, mul_one, sub_mul, one_mul, h.eq, sub_self, sub_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem one_sub_iff {p : R} : IsIdempotentElem (1 - p) ↔ IsIdempotentElem p :=
  ⟨fun h => sub_sub_cancel 1 p ▸ h.one_sub, IsIdempotentElem.one_sub⟩


theorem add_sub_mul_of_commute {R} [Ring R] {p q : R} (h : Commute p q)
    (hp : IsIdempotentElem p) (hq : IsIdempotentElem q) :
    IsIdempotentElem (p + q - p * q) := by
  /-
    R : Type u_9
    inst✝ : Ring R
    p q : R
    h : Commute p q
    hp : IsIdempotentElem p
    hq : IsIdempotentElem q
    ⊢ IsIdempotentElem (HSub.hSub (HAdd.hAdd p q) (HMul.hMul p q))
  -/
  convert (hp.one_sub.mul_of_commute ?_ hq.one_sub).one_sub using 1
    /-
      case h.e'_3
      R : Type u_9
      inst✝ : Ring R
      p q : R
      h : Commute p q
      hp : IsIdempotentElem p
      hq : IsIdempotentElem q
      ⊢ Eq (HSub.hSub (HAdd.hAdd p q) (HMul.hMul p q)) (HSub.hSub 1 (HMul.hMul (HSub …
    -/
  · simp_rw [sub_mul, mul_sub, one_mul, mul_one, sub_sub, sub_sub_cancel, add_sub, add_comm]
    /-
      🎉 no goals
    -/
    /-
      R : Type u_9
      inst✝ : Ring R
      p q : R
      h : Commute p q
      hp : IsIdempotentElem p
      hq : IsIdempotentElem q
      ⊢ Commute (HSub.hSub 1 p) (HSub.hSub 1 q)
    -/
  · simp_rw [commute_iff_eq, sub_mul, mul_sub, one_mul, mul_one, sub_sub, add_sub, add_comm, h.eq]
    /-
      🎉 no goals
    -/


theorem add_sub_mul {R} [CommRing R] {p q : R} (hp : IsIdempotentElem p) (hq : IsIdempotentElem q) :
    IsIdempotentElem (p + q - p * q) :=
  add_sub_mul_of_commute (mul_comm p q) hp hq


theorem pow {p : N} (n : ℕ) (h : IsIdempotentElem p) : IsIdempotentElem (p ^ n) :=
  Nat.recOn n ((pow_zero p).symm ▸ one) fun n _ =>
    show p ^ n.succ * p ^ n.succ = p ^ n.succ by
      /-
        N : Type u_2
        inst✝ : Monoid N
        p : N
        n✝ : Nat
        h : IsIdempotentElem p
        n : Nat
        x✝ : IsIdempotentElem (HPow.hPow p n)
        ⊢ Eq (HMul.hMul (HPow.hPow p n.succ) (HPow.hPow p n.succ)) (HPow.hPow p n.succ)
      -/
      conv_rhs => rw [← h.eq] -- Porting note: was `nth_rw 3 [← h.eq]`
      /-
        N : Type u_2
        inst✝ : Monoid N
        p : N
        n✝ : Nat
        h : IsIdempotentElem p
        n : Nat
        x✝ : IsIdempotentElem (HPow.hPow p n)
        ⊢ Eq (HMul.hMul (HPow.hPow p n.succ) (HPow.hPow p n.succ)) (HPow.hPow (HMul.hM …
      -/
      rw [← sq, ← sq, ← pow_mul, ← pow_mul']
      /-
        🎉 no goals
      -/


theorem pow_succ_eq {p : N} (n : ℕ) (h : IsIdempotentElem p) : p ^ (n + 1) = p :=
                                                                 /-
                                                                   N : Type u_2
                                                                   inst✝ : Monoid N
                                                                   p : N
                                                                   n✝ : Nat
                                                                   h : IsIdempotentElem p
                                                                   n : Nat
                                                                   ih : Eq (HPow.hPow p (HAdd.hAdd n 1)) p
                                                                   ⊢ Eq (HPow.hPow p (HAdd.hAdd n.succ 1)) p
                                                                 -/
  Nat.recOn n ((Nat.zero_add 1).symm ▸ pow_one p) fun n ih => by rw [pow_succ, ih, h.eq]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
theorem iff_eq_one {p : G} : IsIdempotentElem p ↔ p = 1 :=
  Iff.intro (fun h => mul_left_cancel ((mul_one p).symm ▸ h.eq : p * p = p * 1)) fun h =>
    h.symm ▸ one


@[simp]
theorem iff_eq_zero_or_one {p : G₀} : IsIdempotentElem p ↔ p = 0 ∨ p = 1 := by
  refine
    Iff.intro (fun h => or_iff_not_imp_left.mpr fun hp => ?_) fun h =>
      h.elim (fun hp => hp.symm ▸ zero) fun hp => hp.symm ▸ one
  /-
    G₀ : Type u_8
    inst✝ : CancelMonoidWithZero G₀
    p : G₀
    h : IsIdempotentElem p
    hp : Not (Eq p 0)
    ⊢ Eq p 1
  -/
  exact mul_left_cancel₀ hp (h.trans (mul_one p).symm)
  /-
    🎉 no goals
  -/


lemma map {M N F} [Mul M] [Mul N] [FunLike F M N] [MulHomClass F M N] {e : M}
    (he : IsIdempotentElem e) (f : F) : IsIdempotentElem (f e) := by
  /-
    M : Type u_9
    N : Type u_10
    F : Type u_11
    inst✝³ : Mul M
    inst✝² : Mul N
    inst✝¹ : FunLike F M N
    inst✝ : MulHomClass F M N
    e : M
    he : IsIdempotentElem e
    f : F
    ⊢ IsIdempotentElem (f e)
  -/
  rw [IsIdempotentElem, ← map_mul, he.eq]
  /-
    🎉 no goals
  -/


instance : Zero { p : M₀ // IsIdempotentElem p } where zero := ⟨0, zero⟩


@[simp]
theorem coe_zero : ↑(0 : { p : M₀ // IsIdempotentElem p }) = (0 : M₀) :=
  rfl


instance : One { p : M₁ // IsIdempotentElem p } where one := ⟨1, one⟩


@[simp]
theorem coe_one : ↑(1 : { p : M₁ // IsIdempotentElem p }) = (1 : M₁) :=
  rfl


instance : HasCompl { p : R // IsIdempotentElem p } :=
  ⟨fun p => ⟨1 - p, p.prop.one_sub⟩⟩


@[simp]
theorem coe_compl (p : { p : R // IsIdempotentElem p }) : ↑pᶜ = (1 : R) - ↑p :=
  rfl


@[simp]
theorem compl_compl (p : { p : R // IsIdempotentElem p }) : pᶜᶜ = p :=
  Subtype.ext <| sub_sub_cancel _ _


@[simp]
theorem zero_compl : (0 : { p : R // IsIdempotentElem p })ᶜ = 1 :=
  Subtype.ext <| sub_zero _


@[simp]
theorem one_compl : (1 : { p : R // IsIdempotentElem p })ᶜ = 0 :=
  Subtype.ext <| sub_self _



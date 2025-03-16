local notation "𝕎" => WittVector p -- type as `\bbW`


/--
`WittVector.verschiebung` translates the entries of a Witt vector upward, inserting 0s in the gaps.
`WittVector.shift` does the opposite, removing the first entries.
This is mainly useful as an auxiliary construction for `WittVector.verschiebung_nonzero`.
-/
def shift (x : 𝕎 R) (n : ℕ) : 𝕎 R :=
  @mk' p R fun i => x.coeff (n + i)


theorem shift_coeff (x : 𝕎 R) (n k : ℕ) : (x.shift n).coeff k = x.coeff (n + k) :=
  rfl


theorem verschiebung_shift (x : 𝕎 R) (k : ℕ) (h : ∀ i < k + 1, x.coeff i = 0) :
    verschiebung (x.shift k.succ) = x.shift k := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝ : CommRing R
    x : WittVector p R
    k : Nat
    h : ∀ (i : Nat), LT.lt i (HAdd.hAdd k 1) → Eq (x.coeff i) 0
    ⊢ Eq (WittVector.verschiebung (x.shift k.succ)) (x.shift k)
  -/
  ext ⟨j⟩
    /-
      case h.zero
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      x : WittVector p R
      k : Nat
      h : ∀ (i : Nat), LT.lt i (HAdd.hAdd k 1) → Eq (x.coeff i) 0
      ⊢ Eq ((WittVector.verschiebung (x.shift k.succ)).coeff 0) ((x.shift k).coeff 0)
    -/
  · rw [verschiebung_coeff_zero, shift_coeff, h]
    /-
      case h.zero.a
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      x : WittVector p R
      k : Nat
      h : ∀ (i : Nat), LT.lt i (HAdd.hAdd k 1) → Eq (x.coeff i) 0
      ⊢ LT.lt (HAdd.hAdd k 0) (HAdd.hAdd k 1)
    -/
    apply Nat.lt_succ_self
    /-
      🎉 no goals
    -/
    /-
      case h.succ
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      x : WittVector p R
      k : Nat
      h : ∀ (i : Nat), LT.lt i (HAdd.hAdd k 1) → Eq (x.coeff i) 0
      n✝ : Nat
      ⊢ Eq ((WittVector.verschiebung (x.shift k.succ)).coeff (HAdd.hAdd n✝ 1)) ((x.s …
    -/
  · simp only [verschiebung_coeff_succ, shift]
    /-
      case h.succ
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      x : WittVector p R
      k : Nat
      h : ∀ (i : Nat), LT.lt i (HAdd.hAdd k 1) → Eq (x.coeff i) 0
      n✝ : Nat
      ⊢ Eq (x.coeff (HAdd.hAdd k.succ n✝)) (x.coeff (HAdd.hAdd k (HAdd.hAdd n✝ 1)))
    -/
    congr 1
    /-
      case h.succ.e_a
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      x : WittVector p R
      k : Nat
      h : ∀ (i : Nat), LT.lt i (HAdd.hAdd k 1) → Eq (x.coeff i) 0
      n✝ : Nat
      ⊢ Eq (HAdd.hAdd k.succ n✝) (HAdd.hAdd k (HAdd.hAdd n✝ 1))
    -/
    rw [Nat.add_succ, add_comm, Nat.add_succ, add_comm]
    /-
      🎉 no goals
    -/


theorem eq_iterate_verschiebung {x : 𝕎 R} {n : ℕ} (h : ∀ i < n, x.coeff i = 0) :
    x = verschiebung^[n] (x.shift n) := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝ : CommRing R
    x : WittVector p R
    n : Nat
    h : ∀ (i : Nat), LT.lt i n → Eq (x.coeff i) 0
    ⊢ Eq x (Nat.iterate (⇑WittVector.verschiebung) n (x.shift n))
  -/
  induction' n with k ih
    /-
      case zero
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      x : WittVector p R
      h : ∀ (i : Nat), LT.lt i 0 → Eq (x.coeff i) 0
      ⊢ Eq x (Nat.iterate (⇑WittVector.verschiebung) 0 (x.shift 0))
    -/
  · cases x; simp [shift]
             /-
               🎉 no goals
             -/
    /-
      case succ
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      x : WittVector p R
      k : Nat
      ih : (∀ (i : Nat), LT.lt i k → Eq (x.coeff i) 0) → Eq x (Nat.iterate (⇑WittVec …
      h : ∀ (i : Nat), LT.lt i (HAdd.hAdd k 1) → Eq (x.coeff i) 0
      ⊢ Eq x (Nat.iterate (⇑WittVector.verschiebung) (HAdd.hAdd k 1) (x.shift (HAdd. …
    -/
  · dsimp; rw [verschiebung_shift]
      /-
        case succ
        p : Nat
        R : Type u_1
        hp : Fact (Nat.Prime p)
        inst✝ : CommRing R
        x : WittVector p R
        k : Nat
        ih : (∀ (i : Nat), LT.lt i k → Eq (x.coeff i) 0) → Eq x (Nat.iterate (⇑WittVec …
        h : ∀ (i : Nat), LT.lt i (HAdd.hAdd k 1) → Eq (x.coeff i) 0
        ⊢ Eq x (Nat.iterate (⇑WittVector.verschiebung) k (x.shift k))
      -/
    · exact ih fun i hi => h _ (hi.trans (Nat.lt_succ_self _))
      /-
        🎉 no goals
      -/
      /-
        case succ.h
        p : Nat
        R : Type u_1
        hp : Fact (Nat.Prime p)
        inst✝ : CommRing R
        x : WittVector p R
        k : Nat
        ih : (∀ (i : Nat), LT.lt i k → Eq (x.coeff i) 0) → Eq x (Nat.iterate (⇑WittVec …
        h : ∀ (i : Nat), LT.lt i (HAdd.hAdd k 1) → Eq (x.coeff i) 0
        ⊢ ∀ (i : Nat), LT.lt i (HAdd.hAdd k 1) → Eq (x.coeff i) 0
      -/
    · exact h
      /-
        🎉 no goals
      -/


theorem verschiebung_nonzero {x : 𝕎 R} (hx : x ≠ 0) :
    ∃ n : ℕ, ∃ x' : 𝕎 R, x'.coeff 0 ≠ 0 ∧ x = verschiebung^[n] x' := by
  classical
  have hex : ∃ k : ℕ, x.coeff k ≠ 0 := by
    by_contra! hall
    apply hx
    ext i
    simp only [hall, zero_coeff]
  let n := Nat.find hex
  use n, x.shift n
  refine ⟨Nat.find_spec hex, eq_iterate_verschiebung fun i hi => not_not.mp ?_⟩
  exact Nat.find_min hex hi


instance [CharP R p] [NoZeroDivisors R] : NoZeroDivisors (𝕎 R) :=
  ⟨fun {x y} => by
    /-
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝² : CommRing R
      inst✝¹ : CharP R p
      inst✝ : NoZeroDivisors R
      x y : WittVector p R
      ⊢ Eq (HMul.hMul x y) 0 → Or (Eq x 0) (Eq y 0)
    -/
    contrapose!
    /-
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝² : CommRing R
      inst✝¹ : CharP R p
      inst✝ : NoZeroDivisors R
      x y : WittVector p R
      ⊢ And (Ne x 0) (Ne y 0) → Ne (HMul.hMul x y) 0
    -/
    rintro ⟨ha, hb⟩
    /-
      case intro
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝² : CommRing R
      inst✝¹ : CharP R p
      inst✝ : NoZeroDivisors R
      x y : WittVector p R
      ha : Ne x 0
      hb : Ne y 0
      ⊢ Ne (HMul.hMul x y) 0
    -/
    rcases verschiebung_nonzero ha with ⟨na, wa, hwa0, rfl⟩
    /-
      case intro.intro.intro.intro
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝² : CommRing R
      inst✝¹ : CharP R p
      inst✝ : NoZeroDivisors R
      y : WittVector p R
      hb : Ne y 0
      na : Nat
      wa : WittVector p R
      hwa0 : Ne (wa.coeff 0) 0
      ha : Ne (Nat.iterate (⇑WittVector.verschiebung) na wa) 0
      ⊢ Ne (HMul.hMul (Nat.iterate (⇑WittVector.verschiebung) na wa) y) 0
    -/
    rcases verschiebung_nonzero hb with ⟨nb, wb, hwb0, rfl⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝² : CommRing R
      inst✝¹ : CharP R p
      inst✝ : NoZeroDivisors R
      na : Nat
      wa : WittVector p R
      hwa0 : Ne (wa.coeff 0) 0
      ha : Ne (Nat.iterate (⇑WittVector.verschiebung) na wa) 0
      nb : Nat
      wb : WittVector p R
      hwb0 : Ne (wb.coeff 0) 0
      hb : Ne (Nat.iterate (⇑WittVector.verschiebung) nb wb) 0
      ⊢ Ne (HMul.hMul (Nat.iterate (⇑WittVector.verschiebung) na wa) (Nat.iterate (⇑ …
    -/
    refine ne_of_apply_ne (fun x => x.coeff (na + nb)) ?_
    /-
      case intro.intro.intro.intro.intro.intro.intro
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝² : CommRing R
      inst✝¹ : CharP R p
      inst✝ : NoZeroDivisors R
      na : Nat
      wa : WittVector p R
      hwa0 : Ne (wa.coeff 0) 0
      ha : Ne (Nat.iterate (⇑WittVector.verschiebung) na wa) 0
      nb : Nat
      wb : WittVector p R
      hwb0 : Ne (wb.coeff 0) 0
      hb : Ne (Nat.iterate (⇑WittVector.verschiebung) nb wb) 0
      ⊢ Ne ((fun x => x.coeff (HAdd.hAdd na nb)) (HMul.hMul (Nat.iterate (⇑WittVecto …
    -/
    dsimp only
    /-
      case intro.intro.intro.intro.intro.intro.intro
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝² : CommRing R
      inst✝¹ : CharP R p
      inst✝ : NoZeroDivisors R
      na : Nat
      wa : WittVector p R
      hwa0 : Ne (wa.coeff 0) 0
      ha : Ne (Nat.iterate (⇑WittVector.verschiebung) na wa) 0
      nb : Nat
      wb : WittVector p R
      hwb0 : Ne (wb.coeff 0) 0
      hb : Ne (Nat.iterate (⇑WittVector.verschiebung) nb wb) 0
      ⊢ Ne ((HMul.hMul (Nat.iterate (⇑WittVector.verschiebung) na wa) (Nat.iterate ( …
    -/
    rw [iterate_verschiebung_mul_coeff, zero_coeff]
    /-
      case intro.intro.intro.intro.intro.intro.intro
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝² : CommRing R
      inst✝¹ : CharP R p
      inst✝ : NoZeroDivisors R
      na : Nat
      wa : WittVector p R
      hwa0 : Ne (wa.coeff 0) 0
      ha : Ne (Nat.iterate (⇑WittVector.verschiebung) na wa) 0
      nb : Nat
      wb : WittVector p R
      hwb0 : Ne (wb.coeff 0) 0
      hb : Ne (Nat.iterate (⇑WittVector.verschiebung) nb wb) 0
      ⊢ Ne (HMul.hMul (HPow.hPow (wa.coeff 0) (HPow.hPow p nb)) (HPow.hPow (wb.coeff …
    -/
    exact mul_ne_zero (pow_ne_zero _ hwa0) (pow_ne_zero _ hwb0)⟩
    /-
      🎉 no goals
    -/


instance instIsDomain [CharP R p] [IsDomain R] : IsDomain (𝕎 R) :=
  NoZeroDivisors.to_isDomain _



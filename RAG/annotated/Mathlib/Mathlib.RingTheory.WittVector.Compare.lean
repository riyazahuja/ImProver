local notation "𝕎" => WittVector p


theorem eq_of_le_of_cast_pow_eq_zero [CharP R p] (i : ℕ) (hin : i ≤ n)
    (hpi : (p : TruncatedWittVector p n R) ^ i = 0) : i = n := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CharP R p
    i : Nat
    hin : LE.le i n
    hpi : Eq (HPow.hPow (↑p) i) 0
    ⊢ Eq i n
  -/
  contrapose! hpi
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CharP R p
    i : Nat
    hin : LE.le i n
    hpi : Ne i n
    ⊢ Ne (HPow.hPow (↑p) i) 0
  -/
  replace hin := lt_of_le_of_ne hin hpi; clear hpi
  have : (p : TruncatedWittVector p n R) ^ i = WittVector.truncate n ((p : 𝕎 R) ^ i) := by
    rw [RingHom.map_pow, map_natCast]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CharP R p
    i : Nat
    hin : LT.lt i n
    this : Eq (HPow.hPow (↑p) i) ((WittVector.truncate n) (HPow.hPow (↑p) i))
    ⊢ Ne (HPow.hPow (↑p) i) 0
  -/
  rw [this, ne_eq, TruncatedWittVector.ext_iff, not_forall]; clear this
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CharP R p
    i : Nat
    hin : LT.lt i n
    ⊢ Exists fun x => Not (Eq (TruncatedWittVector.coeff x ((WittVector.truncate n …
  -/
  use ⟨i, hin⟩
  /-
    case h
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CharP R p
    i : Nat
    hin : LT.lt i n
    ⊢ Not (Eq (TruncatedWittVector.coeff ⟨i, hin⟩ ((WittVector.truncate n) (HPow.h …
  -/
  rw [WittVector.coeff_truncate, coeff_zero, Fin.val_mk, WittVector.coeff_p_pow]
  /-
    case h
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CharP R p
    i : Nat
    hin : LT.lt i n
    ⊢ Not (Eq 1 0)
  -/
  haveI : Nontrivial R := CharP.nontrivial_of_char_ne_one hp.1.ne_one
  /-
    case h
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CharP R p
    i : Nat
    hin : LT.lt i n
    this : Nontrivial R
    ⊢ Not (Eq 1 0)
  -/
  exact one_ne_zero
  /-
    🎉 no goals
  -/


theorem card_zmod : Fintype.card (TruncatedWittVector p n (ZMod p)) = p ^ n := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (Fintype.card (TruncatedWittVector p n (ZMod p))) (HPow.hPow p n)
  -/
  rw [card, ZMod.card]
  /-
    🎉 no goals
  -/


theorem charP_zmod : CharP (TruncatedWittVector p n (ZMod p)) (p ^ n) :=
  charP_of_prime_pow_injective _ _ _ (card_zmod _ _) (eq_of_le_of_cast_pow_eq_zero p n (ZMod p))


/-- The unique isomorphism between `ZMod p^n` and `TruncatedWittVector p n (ZMod p)`.

This isomorphism exists, because `TruncatedWittVector p n (ZMod p)` is a finite ring
with characteristic and cardinality `p^n`.
-/
def zmodEquivTrunc : ZMod (p ^ n) ≃+* TruncatedWittVector p n (ZMod p) :=
  ZMod.ringEquiv (TruncatedWittVector p n (ZMod p)) (card_zmod _ _)


theorem zmodEquivTrunc_apply {x : ZMod (p ^ n)} :
    zmodEquivTrunc p n x =
                                    /-
                                      p : Nat
                                      hp : Fact (Nat.Prime p)
                                      n : Nat
                                      R : Type u_1
                                      inst✝ : CommRing R
                                      x : ZMod (HPow.hPow p n)
                                      ⊢ Dvd.dvd (HPow.hPow p n) (HPow.hPow p n)
                                    -/
      ZMod.castHom (m := p ^ n) (by rfl) (TruncatedWittVector p n (ZMod p)) x :=
                                    /-
                                      🎉 no goals
                                    -/
  rfl


/-- The following diagram commutes:
```text
          ZMod (p^n) ----------------------------> ZMod (p^m)
            |                                        |
            |                                        |
            v                                        v
TruncatedWittVector p n (ZMod p) ----> TruncatedWittVector p m (ZMod p)
```
Here the vertical arrows are `TruncatedWittVector.zmodEquivTrunc`,
the horizontal arrow at the top is `ZMod.castHom`,
and the horizontal arrow at the bottom is `TruncatedWittVector.truncate`.
-/
theorem commutes {m : ℕ} (hm : n ≤ m) :
    (truncate hm).comp (zmodEquivTrunc p m).toRingHom =
      (zmodEquivTrunc p n).toRingHom.comp (ZMod.castHom (pow_dvd_pow p hm) _) :=
  RingHom.ext_zmod _ _


theorem commutes' {m : ℕ} (hm : n ≤ m) (x : ZMod (p ^ m)) :
    truncate hm (zmodEquivTrunc p m x) = zmodEquivTrunc p n (ZMod.castHom (pow_dvd_pow p hm) _ x) :=
                                                                  /-
                                                                    p : Nat
                                                                    hp : Fact (Nat.Prime p)
                                                                    n m : Nat
                                                                    hm : LE.le n m
                                                                    x : ZMod (HPow.hPow p m)
                                                                    ⊢ Eq (((TruncatedWittVector.truncate hm).comp (TruncatedWittVector.zmodEquivTr …
                                                                  -/
  show (truncate hm).comp (zmodEquivTrunc p m).toRingHom x = _ by rw [commutes _ _ hm]; rfl
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


theorem commutes_symm' {m : ℕ} (hm : n ≤ m) (x : TruncatedWittVector p m (ZMod p)) :
    (zmodEquivTrunc p n).symm (truncate hm x) =
      ZMod.castHom (pow_dvd_pow p hm) _ ((zmodEquivTrunc p m).symm x) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n m : Nat
    hm : LE.le n m
    x : TruncatedWittVector p m (ZMod p)
    ⊢ Eq ((TruncatedWittVector.zmodEquivTrunc p n).symm ((TruncatedWittVector.trun …
  -/
  apply (zmodEquivTrunc p n).injective
  /-
    case a
    p : Nat
    hp : Fact (Nat.Prime p)
    n m : Nat
    hm : LE.le n m
    x : TruncatedWittVector p m (ZMod p)
    ⊢ Eq ((TruncatedWittVector.zmodEquivTrunc p n) ((TruncatedWittVector.zmodEquiv …
  -/
  rw [← commutes' _ _ hm]
  /-
    case a
    p : Nat
    hp : Fact (Nat.Prime p)
    n m : Nat
    hm : LE.le n m
    x : TruncatedWittVector p m (ZMod p)
    ⊢ Eq ((TruncatedWittVector.zmodEquivTrunc p n) ((TruncatedWittVector.zmodEquiv …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The following diagram commutes:
```text
TruncatedWittVector p n (ZMod p) ----> TruncatedWittVector p m (ZMod p)
            |                                        |
            |                                        |
            v                                        v
          ZMod (p^n) ----------------------------> ZMod (p^m)
```
Here the vertical arrows are `(TruncatedWittVector.zmodEquivTrunc p _).symm`,
the horizontal arrow at the top is `ZMod.castHom`,
and the horizontal arrow at the bottom is `TruncatedWittVector.truncate`.
-/
theorem commutes_symm {m : ℕ} (hm : n ≤ m) :
    (zmodEquivTrunc p n).symm.toRingHom.comp (truncate hm) =
      (ZMod.castHom (pow_dvd_pow p hm) _).comp (zmodEquivTrunc p m).symm.toRingHom := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n m : Nat
    hm : LE.le n m
    ⊢ Eq ((TruncatedWittVector.zmodEquivTrunc p n).symm.toRingHom.comp (TruncatedW …
  -/
  ext; apply commutes_symm'
       /-
         🎉 no goals
       -/


/-- `toZModPow` is a family of compatible ring homs. We get this family by composing
`TruncatedWittVector.zmodEquivTrunc` (in right-to-left direction) with `WittVector.truncate`. -/
def toZModPow (k : ℕ) : 𝕎 (ZMod p) →+* ZMod (p ^ k) :=
  (zmodEquivTrunc p k).symm.toRingHom.comp (truncate k)


theorem toZModPow_compat (m n : ℕ) (h : m ≤ n) :
    (ZMod.castHom (pow_dvd_pow p h) (ZMod (p ^ m))).comp (toZModPow p n) = toZModPow p m :=
  calc
    (ZMod.castHom _ (ZMod (p ^ m))).comp ((zmodEquivTrunc p n).symm.toRingHom.comp (truncate n))
    _ = ((zmodEquivTrunc p m).symm.toRingHom.comp (TruncatedWittVector.truncate h)).comp
          (truncate n) := by
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        m n : Nat
        h : LE.le m n
        ⊢ Eq ((ZMod.castHom ⋯ (ZMod (HPow.hPow p m))).comp ((TruncatedWittVector.zmodE …
      -/
      rw [commutes_symm, RingHom.comp_assoc]
      /-
        🎉 no goals
      -/
    _ = (zmodEquivTrunc p m).symm.toRingHom.comp (truncate m) := by
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        m n : Nat
        h : LE.le m n
        ⊢ Eq (((TruncatedWittVector.zmodEquivTrunc p m).symm.toRingHom.comp (Truncated …
      -/
      rw [RingHom.comp_assoc, truncate_comp_wittVector_truncate]
      /-
        🎉 no goals
      -/


/-- `toPadicInt` lifts `toZModPow : 𝕎 (ZMod p) →+* ZMod (p ^ k)` to a ring hom to `ℤ_[p]`
using `PadicInt.lift`, the universal property of `ℤ_[p]`.
-/
def toPadicInt : 𝕎 (ZMod p) →+* ℤ_[p] :=
  PadicInt.lift <| toZModPow_compat p


theorem zmodEquivTrunc_compat (k₁ k₂ : ℕ) (hk : k₁ ≤ k₂) :
    (TruncatedWittVector.truncate hk).comp
        ((zmodEquivTrunc p k₂).toRingHom.comp (PadicInt.toZModPow k₂)) =
      (zmodEquivTrunc p k₁).toRingHom.comp (PadicInt.toZModPow k₁) := by
  rw [← RingHom.comp_assoc, commutes, RingHom.comp_assoc,
    PadicInt.zmod_cast_comp_toZModPow _ _ hk]


/-- `fromPadicInt` uses `WittVector.lift` to lift `TruncatedWittVector.zmodEquivTrunc`
composed with `PadicInt.toZModPow` to a ring hom `ℤ_[p] →+* 𝕎 (ZMod p)`.
-/
def fromPadicInt : ℤ_[p] →+* 𝕎 (ZMod p) :=
  (WittVector.lift fun k => (zmodEquivTrunc p k).toRingHom.comp (PadicInt.toZModPow k)) <|
    zmodEquivTrunc_compat _


theorem toPadicInt_comp_fromPadicInt : (toPadicInt p).comp (fromPadicInt p) = RingHom.id ℤ_[p] := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ Eq ((WittVector.toPadicInt p).comp (WittVector.fromPadicInt p)) (RingHom.id  …
  -/
  rw [← PadicInt.toZModPow_eq_iff_ext]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ ∀ (n : Nat), Eq ((PadicInt.toZModPow n).comp ((WittVector.toPadicInt p).comp …
  -/
  intro n
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq ((PadicInt.toZModPow n).comp ((WittVector.toPadicInt p).comp (WittVector. …
  -/
  rw [← RingHom.comp_assoc, toPadicInt, PadicInt.lift_spec]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq ((WittVector.toZModPow p n).comp (WittVector.fromPadicInt p)) ((PadicInt. …
  -/
  simp only [fromPadicInt, toZModPow, RingHom.comp_id]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (((TruncatedWittVector.zmodEquivTrunc p n).symm.toRingHom.comp (WittVecto …
  -/
  rw [RingHom.comp_assoc, truncate_comp_lift, ← RingHom.comp_assoc]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (((TruncatedWittVector.zmodEquivTrunc p n).symm.toRingHom.comp (Truncated …
  -/
  simp only [RingEquiv.symm_toRingHom_comp_toRingHom, RingHom.id_comp]
  /-
    🎉 no goals
  -/


theorem toPadicInt_comp_fromPadicInt_ext (x) :
    (toPadicInt p).comp (fromPadicInt p) x = RingHom.id ℤ_[p] x := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : PadicInt p
    ⊢ Eq (((WittVector.toPadicInt p).comp (WittVector.fromPadicInt p)) x) ((RingHo …
  -/
  rw [toPadicInt_comp_fromPadicInt]
  /-
    🎉 no goals
  -/


theorem fromPadicInt_comp_toPadicInt :
    (fromPadicInt p).comp (toPadicInt p) = RingHom.id (𝕎 (ZMod p)) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ Eq ((WittVector.fromPadicInt p).comp (WittVector.toPadicInt p)) (RingHom.id  …
  -/
  apply WittVector.hom_ext
  /-
    case h
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ ∀ (k : Nat), Eq ((WittVector.truncate k).comp ((WittVector.fromPadicInt p).c …
  -/
  intro n
  /-
    case h
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq ((WittVector.truncate n).comp ((WittVector.fromPadicInt p).comp (WittVect …
  -/
  rw [fromPadicInt, ← RingHom.comp_assoc, truncate_comp_lift, RingHom.comp_assoc]
  simp only [toPadicInt, toZModPow, RingHom.comp_id, PadicInt.lift_spec, RingHom.id_comp, ←
    RingHom.comp_assoc, RingEquiv.toRingHom_comp_symm_toRingHom]


theorem fromPadicInt_comp_toPadicInt_ext (x) :
    (fromPadicInt p).comp (toPadicInt p) x = RingHom.id (𝕎 (ZMod p)) x := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : WittVector p (ZMod p)
    ⊢ Eq (((WittVector.fromPadicInt p).comp (WittVector.toPadicInt p)) x) ((RingHo …
  -/
  rw [fromPadicInt_comp_toPadicInt]
  /-
    🎉 no goals
  -/


/-- The ring of Witt vectors over `ZMod p` is isomorphic to the ring of `p`-adic integers. This
equivalence is witnessed by `WittVector.toPadicInt` with inverse `WittVector.fromPadicInt`.
-/
def equiv : 𝕎 (ZMod p) ≃+* ℤ_[p] where
  toFun := toPadicInt p
  invFun := fromPadicInt p
  left_inv := fromPadicInt_comp_toPadicInt_ext _
  right_inv := toPadicInt_comp_fromPadicInt_ext _
  map_mul' := RingHom.map_mul _
  map_add' := RingHom.map_add _



theorem toNat_injective {n : Nat} : Function.Injective (BitVec.toNat : BitVec n → _)
  | ⟨_, _⟩, ⟨_, _⟩, rfl => rfl


theorem toFin_injective {n : Nat} : Function.Injective (toFin : BitVec n → _)
  | ⟨_, _⟩, ⟨_, _⟩, rfl => rfl


instance : SMul ℕ (BitVec w) := ⟨fun x y => ofFin <| x • y.toFin⟩

instance : SMul ℤ (BitVec w) := ⟨fun x y => ofFin <| x • y.toFin⟩

instance : Pow (BitVec w) ℕ  := ⟨fun x n => ofFin <| x.toFin ^ n⟩


lemma toFin_nsmul (n : ℕ) (x : BitVec w)  : toFin (n • x) = n • x.toFin := rfl

lemma toFin_zsmul (z : ℤ) (x : BitVec w)  : toFin (z • x) = z • x.toFin := rfl

lemma toFin_pow (x : BitVec w) (n : ℕ)    : toFin (x ^ n) = x.toFin ^ n := rfl


instance : CommSemiring (BitVec w) :=
  toFin_injective.commSemiring _
    rfl /- toFin_zero -/
    rfl /- toFin_one -/
    toFin_add
    toFin_mul
    toFin_nsmul
    toFin_pow
    (fun _ => rfl) /- toFin_natCast -/
-- The statement in the new API would be: `n#(k.succ) = ((n / 2)#k).concat (n % 2 != 0)`


@[simp] lemma ofFin_neg {x : Fin (2 ^ w)} : ofFin (-x) = -(ofFin x) := by
  /-
    w : Nat
    x : Fin (HPow.hPow 2 w)
    ⊢ Eq { toFin := Neg.neg x } (Neg.neg { toFin := x })
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp] lemma ofFin_natCast (n : ℕ) : ofFin (n : Fin (2^w)) = n := by
  /-
    w n : Nat
    ⊢ Eq { toFin := ↑n } ↑n
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma toFin_natCast (n : ℕ) : toFin (n : BitVec w) = n := by
  /-
    w n : Nat
    ⊢ Eq (↑n).toFin ↑n
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem ofFin_intCast (z : ℤ) : ofFin (z : Fin (2^w)) = ↑z := by
  /-
    w : Nat
    z : Int
    ⊢ Eq { toFin := ↑z } ↑z
  -/
  cases w
  case zero =>
    simp only [eq_nil]
  case succ w =>
    simp only [Int.cast, IntCast.intCast]
    unfold Int.castDef
    cases' z with z z
    · rfl
    · rw [ofInt_negSucc_eq_not_ofNat]
      simp only [Nat.cast_add, Nat.cast_one, neg_add_rev]
      rw [← add_ofFin, ofFin_neg, ofFin_ofNat, ofNat_eq_ofNat, ofFin_neg, ofFin_natCast,
        natCast_eq_ofNat, negOne_eq_allOnes, ← sub_toAdd, allOnes_sub_eq_not]


theorem toFin_intCast (z : ℤ) : toFin (z : BitVec w) = z := by
  /-
    w : Nat
    z : Int
    ⊢ Eq (↑z).toFin ↑z
  -/
  apply toFin_inj.mpr <| (ofFin_intCast z).symm
  /-
    🎉 no goals
  -/


instance : CommRing (BitVec w) :=
  toFin_injective.commRing _
    toFin_zero toFin_one toFin_add toFin_mul toFin_neg toFin_sub
    toFin_nsmul toFin_zsmul toFin_pow toFin_natCast toFin_intCast


/-- The ring `BitVec m` is isomorphic to `Fin (2 ^ m)`. -/
@[simps]
def equivFin {m : ℕ} : BitVec m ≃+* Fin (2 ^ m) where
  toFun a := a.toFin
  invFun a := ofFin a
  left_inv _ := rfl
  right_inv _ := rfl
  map_mul' _ _ := rfl
  map_add' _ _ := rfl



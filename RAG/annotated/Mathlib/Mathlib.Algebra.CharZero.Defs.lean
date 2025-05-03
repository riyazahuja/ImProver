/-- Typeclass for monoids with characteristic zero.
  (This is usually stated on fields but it makes sense for any additive monoid with 1.)

*Warning*: for a semiring `R`, `CharZero R` and `CharP R 0` need not coincide.
* `CharZero R` requires an injection `ℕ ↪ R`;
* `CharP R 0` asks that only `0 : ℕ` maps to `0 : R` under the map `ℕ → R`.
For instance, endowing `{0, 1}` with addition given by `max` (i.e. `1` is absorbing), shows that
`CharZero {0, 1}` does not hold and yet `CharP {0, 1} 0` does.
This example is formalized in `Counterexamples/CharPZeroNeCharZero.lean`.
-/
class CharZero (R) [AddMonoidWithOne R] : Prop where
  /-- An additive monoid with one has characteristic zero if the canonical map `ℕ → R` is
  injective. -/
  cast_injective : Function.Injective (Nat.cast : ℕ → R)


theorem charZero_of_inj_zero [AddGroupWithOne R] (H : ∀ n : ℕ, (n : R) = 0 → n = 0) :
    CharZero R :=
  ⟨@fun m n h => by
    induction m generalizing n with
    | zero => rw [H n]; rw [← h, Nat.cast_zero]
    | succ m ih =>
      cases n
      · apply H; rw [h, Nat.cast_zero]
      · simp only [Nat.cast_succ, add_right_cancel_iff] at h; rwa [ih]⟩


theorem cast_injective : Function.Injective (Nat.cast : ℕ → R) :=
  CharZero.cast_injective


@[simp, norm_cast]
theorem cast_inj {m n : ℕ} : (m : R) = n ↔ m = n :=
  cast_injective.eq_iff


@[simp, norm_cast]
                                                         /-
                                                           R : Type u_1
                                                           inst✝¹ : AddMonoidWithOne R
                                                           inst✝ : CharZero R
                                                           n : Nat
                                                           ⊢ Iff (Eq (↑n) 0) (Eq n 0)
                                                         -/
theorem cast_eq_zero {n : ℕ} : (n : R) = 0 ↔ n = 0 := by rw [← cast_zero, cast_inj]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[norm_cast]
theorem cast_ne_zero {n : ℕ} : (n : R) ≠ 0 ↔ n ≠ 0 :=
  not_congr cast_eq_zero


theorem cast_add_one_ne_zero (n : ℕ) : (n + 1 : R) ≠ 0 :=
  mod_cast n.succ_ne_zero


@[simp, norm_cast]
                                                        /-
                                                          R : Type u_1
                                                          inst✝¹ : AddMonoidWithOne R
                                                          inst✝ : CharZero R
                                                          n : Nat
                                                          ⊢ Iff (Eq (↑n) 1) (Eq n 1)
                                                        -/
theorem cast_eq_one {n : ℕ} : (n : R) = 1 ↔ n = 1 := by rw [← cast_one, cast_inj]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[norm_cast]
theorem cast_ne_one {n : ℕ} : (n : R) ≠ 1 ↔ n ≠ 1 :=
  cast_eq_one.not


instance (priority := 100) AtLeastTwo.toNeZero (n : ℕ) [n.AtLeastTwo] : NeZero n :=
  ⟨Nat.ne_of_gt (Nat.le_of_lt one_lt)⟩


@[simp] lemma ofNat_ne_zero (n : ℕ) [n.AtLeastTwo] : (no_index (ofNat n) : R) ≠ 0 :=
  Nat.cast_ne_zero.2 (NeZero.ne n)


@[simp] lemma zero_ne_ofNat (n : ℕ) [n.AtLeastTwo] : 0 ≠ (no_index (ofNat n) : R) :=
  (ofNat_ne_zero n).symm


@[simp] lemma ofNat_ne_one (n : ℕ) [n.AtLeastTwo] : (no_index (ofNat n) : R) ≠ 1 :=
  Nat.cast_ne_one.2 (Nat.AtLeastTwo.ne_one)


@[simp] lemma one_ne_ofNat (n : ℕ) [n.AtLeastTwo] : (1 : R) ≠ no_index (ofNat n) :=
  (ofNat_ne_one n).symm


@[simp] lemma ofNat_eq_ofNat {m n : ℕ} [m.AtLeastTwo] [n.AtLeastTwo] :
    (no_index (ofNat m) : R) = no_index (ofNat n) ↔ (ofNat m : ℕ) = ofNat n :=
  Nat.cast_inj


instance charZero {M} {n : ℕ} [NeZero n] [AddMonoidWithOne M] [CharZero M] : NeZero (n : M) :=
  ⟨Nat.cast_ne_zero.mpr out⟩


instance charZero_one {M} [AddMonoidWithOne M] [CharZero M] : NeZero (1 : M) where
  out := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝¹ : AddMonoidWithOne M
      inst✝ : CharZero M
      ⊢ Ne 1 0
    -/
    rw [← Nat.cast_one, Nat.cast_ne_zero]
    /-
      R : Type u_1
      M : Type u_2
      inst✝¹ : AddMonoidWithOne M
      inst✝ : CharZero M
      ⊢ Ne 1 0
    -/
    trivial
    /-
      🎉 no goals
    -/


instance charZero_ofNat {M} {n : ℕ} [n.AtLeastTwo] [AddMonoidWithOne M] [CharZero M] :
    NeZero (OfNat.ofNat n : M) :=
  ⟨OfNat.ofNat_ne_zero n⟩



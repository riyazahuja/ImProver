/-- The generator of the kernel of the unique homomorphism ℕ → R for a semiring R.

*Warning*: for a semiring `R`, `CharP R 0` and `CharZero R` need not coincide.
* `CharP R 0` asks that only `0 : ℕ` maps to `0 : R` under the map `ℕ → R`;
* `CharZero R` requires an injection `ℕ ↪ R`.

For instance, endowing `{0, 1}` with addition given by `max` (i.e. `1` is absorbing), shows that
`CharZero {0, 1}` does not hold and yet `CharP {0, 1} 0` does.
This example is formalized in `Counterexamples/CharPZeroNeCharZero.lean`.
-/
@[mk_iff]
class _root_.CharP : Prop where
  cast_eq_zero_iff' : ∀ x : ℕ, (x : R) = 0 ↔ p ∣ x


lemma cast_eq_zero_iff (a : ℕ) : (a : R) = 0 ↔ p ∣ a := cast_eq_zero_iff' a


variable {R} in
lemma congr {q : ℕ} (h : p = q) : CharP R q := h ▸ ‹CharP R p›


lemma natCast_eq_natCast' (h : a ≡ b [MOD p]) : (a : R) = b := by
  /-
    R : Type u_1
    inst✝¹ : AddMonoidWithOne R
    p : Nat
    inst✝ : CharP R p
    a b : Nat
    h : p.ModEq a b
    ⊢ Eq ↑a ↑b
  -/
  wlog hle : a ≤ b
    /-
      case inr
      R : Type u_1
      inst✝¹ : AddMonoidWithOne R
      p : Nat
      inst✝ : CharP R p
      a b : Nat
      h : p.ModEq a b
      this : ∀ (R : Type u_1) [inst : AddMonoidWithOne R] (p : Nat) [inst_1 : CharP  …
      hle : Not (LE.le a b)
      ⊢ Eq ↑a ↑b
    -/
  · exact (this R p h.symm (le_of_not_le hle)).symm
    /-
      🎉 no goals
    -/
  /-
    R✝ : Type u_1
    inst✝² : AddMonoidWithOne R✝
    p✝ a✝ b✝ : Nat
    R : Type u_1
    inst✝¹ : AddMonoidWithOne R
    p : Nat
    inst✝ : CharP R p
    a b : Nat
    h : p.ModEq a b
    hle : LE.le a b
    ⊢ Eq ↑a ↑b
  -/
  rw [Nat.modEq_iff_dvd' hle] at h
  /-
    R✝ : Type u_1
    inst✝² : AddMonoidWithOne R✝
    p✝ a✝ b✝ : Nat
    R : Type u_1
    inst✝¹ : AddMonoidWithOne R
    p : Nat
    inst✝ : CharP R p
    a b : Nat
    h : Dvd.dvd p (HSub.hSub b a)
    hle : LE.le a b
    ⊢ Eq ↑a ↑b
  -/
  rw [← Nat.sub_add_cancel hle, Nat.cast_add, (cast_eq_zero_iff R p _).mpr h, zero_add]
  /-
    🎉 no goals
  -/


@[simp] lemma cast_eq_zero : (p : R) = 0 := (cast_eq_zero_iff R p p).2 dvd_rfl

-- TODO: This lemma needs to be `@[simp]` for confluence in the presence of `CharP.cast_eq_zero` and
-- `Nat.cast_ofNat`, but with `no_index` on its entire LHS, it matches literally every expression so
-- is too expensive. If https://github.com/leanprover/lean4/issues/2867 is fixed in a performant way, this can be made `@[simp]`.
--
-- @[simp]

lemma ofNat_eq_zero [p.AtLeastTwo] : (ofNat(p) : R) = 0 := cast_eq_zero R p


lemma natCast_eq_natCast_mod (a : ℕ) : (a : R) = a % p :=
  natCast_eq_natCast' R p (Nat.mod_modEq a p).symm


lemma eq {p q : ℕ} (_hp : CharP R p) (_hq : CharP R q) : p = q :=
  Nat.dvd_antisymm ((cast_eq_zero_iff R p q).1 (cast_eq_zero _ _))
    ((cast_eq_zero_iff R q p).1 (cast_eq_zero _ _))


instance ofCharZero [CharZero R] : CharP R 0 where
                            /-
                              R : Type u_1
                              inst✝² : AddMonoidWithOne R
                              p : Nat
                              inst✝¹ : CharP R p
                              a b : Nat
                              inst✝ : CharZero R
                              x : Nat
                              ⊢ Iff (Eq (↑x) 0) (Dvd.dvd 0 x)
                            -/
  cast_eq_zero_iff' x := by rw [zero_dvd_iff, ← Nat.cast_zero, Nat.cast_inj]
                            /-
                              🎉 no goals
                            -/


lemma natCast_eq_natCast : (a : R) = b ↔ a ≡ b [MOD p] := by
  /-
    R : Type u_1
    inst✝² : AddMonoidWithOne R
    p : Nat
    inst✝¹ : CharP R p
    a b : Nat
    inst✝ : IsRightCancelAdd R
    ⊢ Iff (Eq ↑a ↑b) (p.ModEq a b)
  -/
  wlog hle : a ≤ b
    /-
      case inr
      R : Type u_1
      inst✝² : AddMonoidWithOne R
      p : Nat
      inst✝¹ : CharP R p
      a b : Nat
      inst✝ : IsRightCancelAdd R
      this : ∀ (R : Type u_1) [inst : AddMonoidWithOne R] (p : Nat) [inst_1 : CharP  …
      hle : Not (LE.le a b)
      ⊢ Iff (Eq ↑a ↑b) (p.ModEq a b)
    -/
  · rw [eq_comm, this R p (le_of_not_le hle), Nat.ModEq.comm]
    /-
      🎉 no goals
    -/
  rw [Nat.modEq_iff_dvd' hle, ← cast_eq_zero_iff R p (b - a),
    ← add_right_cancel_iff (G := R) (a := a) (b := b - a), zero_add, ← Nat.cast_add,
    Nat.sub_add_cancel hle, eq_comm]


lemma intCast_eq_zero_iff (a : ℤ) : (a : R) = 0 ↔ (p : ℤ) ∣ a := by
  /-
    R : Type u_1
    inst✝¹ : AddGroupWithOne R
    p : Nat
    inst✝ : CharP R p
    a : Int
    ⊢ Iff (Eq (↑a) 0) (Dvd.dvd (↑p) a)
  -/
  rcases lt_trichotomy a 0 with (h | rfl | h)
    /-
      case inl
      R : Type u_1
      inst✝¹ : AddGroupWithOne R
      p : Nat
      inst✝ : CharP R p
      a : Int
      h : LT.lt a 0
      ⊢ Iff (Eq (↑a) 0) (Dvd.dvd (↑p) a)
    -/
  · rw [← neg_eq_zero, ← Int.cast_neg, ← Int.dvd_neg]
    /-
      case inl
      R : Type u_1
      inst✝¹ : AddGroupWithOne R
      p : Nat
      inst✝ : CharP R p
      a : Int
      h : LT.lt a 0
      ⊢ Iff (Eq (↑(Neg.neg a)) 0) (Dvd.dvd (↑p) (Neg.neg a))
    -/
    lift -a to ℕ using neg_nonneg.mpr (le_of_lt h) with b
    /-
      case inl.intro
      R : Type u_1
      inst✝¹ : AddGroupWithOne R
      p : Nat
      inst✝ : CharP R p
      a : Int
      h : LT.lt a 0
      b : Nat
      ⊢ Iff (Eq (↑↑b) 0) (Dvd.dvd ↑p ↑b)
    -/
    rw [Int.cast_natCast, CharP.cast_eq_zero_iff R p, Int.natCast_dvd_natCast]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      R : Type u_1
      inst✝¹ : AddGroupWithOne R
      p : Nat
      inst✝ : CharP R p
      ⊢ Iff (Eq (↑0) 0) (Dvd.dvd (↑p) 0)
    -/
  · simp only [Int.cast_zero, eq_self_iff_true, Int.dvd_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      R : Type u_1
      inst✝¹ : AddGroupWithOne R
      p : Nat
      inst✝ : CharP R p
      a : Int
      h : LT.lt 0 a
      ⊢ Iff (Eq (↑a) 0) (Dvd.dvd (↑p) a)
    -/
  · lift a to ℕ using le_of_lt h with b
    /-
      case inr.inr.intro
      R : Type u_1
      inst✝¹ : AddGroupWithOne R
      p : Nat
      inst✝ : CharP R p
      a : Int
      b : Nat
      h✝ h : LT.lt 0 ↑b
      ⊢ Iff (Eq (↑↑b) 0) (Dvd.dvd ↑p ↑b)
    -/
    rw [Int.cast_natCast, CharP.cast_eq_zero_iff R p, Int.natCast_dvd_natCast]
    /-
      🎉 no goals
    -/


lemma intCast_eq_intCast : (a : R) = b ↔ a ≡ b [ZMOD p] := by
  /-
    R : Type u_1
    inst✝¹ : AddGroupWithOne R
    p : Nat
    inst✝ : CharP R p
    a b : Int
    ⊢ Iff (Eq ↑a ↑b) ((↑p).ModEq a b)
  -/
  rw [eq_comm, ← sub_eq_zero, ← Int.cast_sub, CharP.intCast_eq_zero_iff R p, Int.modEq_iff_dvd]
  /-
    🎉 no goals
  -/


lemma intCast_eq_intCast_mod : (a : R) = a % (p : ℤ) :=
  (CharP.intCast_eq_intCast R p).mpr (Int.mod_modEq a p).symm


lemma charP_to_charZero [CharP R 0] : CharZero R :=
  charZero_of_inj_zero fun n h0 => eq_zero_of_zero_dvd ((cast_eq_zero_iff R 0 n).mp h0)


lemma charP_zero_iff_charZero : CharP R 0 ↔ CharZero R :=
  ⟨fun _ ↦ charP_to_charZero R, fun _ ↦ ofCharZero R⟩


lemma «exists» : ∃ p, CharP R p :=
  letI := Classical.decEq R
  by_cases
    (fun H : ∀ p : ℕ, (p : R) = 0 → p = 0 =>
                       /-
                         R : Type u_1
                         inst✝ : NonAssocSemiring R
                         this : DecidableEq R := Classical.decEq R
                         H : ∀ (p : Nat), Eq (↑p) 0 → Eq p 0
                         x : Nat
                         ⊢ Iff (Eq (↑x) 0) (Dvd.dvd 0 x)
                       -/
      ⟨0, ⟨fun x => by rw [zero_dvd_iff]; exact ⟨H x, by rintro rfl; simp⟩⟩⟩)
                                          /-
                                            🎉 no goals
                                          -/
    fun H =>
    ⟨Nat.find (not_forall.1 H),
      ⟨fun x =>
        ⟨fun H1 =>
          Nat.dvd_of_mod_eq_zero
            (by_contradiction fun H2 =>
              Nat.find_min (not_forall.1 H)
                (Nat.mod_lt x <|
                  Nat.pos_of_ne_zero <| not_of_not_imp <| Nat.find_spec (not_forall.1 H))
                (not_imp_of_and_not
                  ⟨by
                    rwa [← Nat.mod_add_div x (Nat.find (not_forall.1 H)), Nat.cast_add,
                      Nat.cast_mul,
                      of_not_not (not_not_of_not_imp <| Nat.find_spec (not_forall.1 H)),
                      zero_mul, add_zero] at H1,
                    H2⟩)),
          fun H1 => by
          rw [← Nat.mul_div_cancel' H1, Nat.cast_mul,
            of_not_not (not_not_of_not_imp <| Nat.find_spec (not_forall.1 H)),
            zero_mul]⟩⟩⟩


lemma existsUnique : ∃! p, CharP R p :=
  let ⟨c, H⟩ := CharP.exists R
  ⟨c, H, fun _y H2 => CharP.eq R H2 H⟩


@[deprecated (since := "2024-12-17")] alias exists_unique := existsUnique


/-- Noncomputable function that outputs the unique characteristic of a semiring. -/
noncomputable def ringChar [NonAssocSemiring R] : ℕ := Classical.choose (CharP.existsUnique R)


lemma spec : ∀ x : ℕ, (x : R) = 0 ↔ ringChar R ∣ x := by
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    ⊢ ∀ (x : Nat), Iff (Eq (↑x) 0) (Dvd.dvd (ringChar R) x)
  -/
  letI : CharP R (ringChar R) := (Classical.choose_spec (CharP.existsUnique R)).1
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    this : CharP R (ringChar R) := (Classical.choose_spec (CharP.existsUnique R)). …
    ⊢ ∀ (x : Nat), Iff (Eq (↑x) 0) (Dvd.dvd (ringChar R) x)
  -/
  exact CharP.cast_eq_zero_iff R (ringChar R)
  /-
    🎉 no goals
  -/


lemma eq (p : ℕ) [C : CharP R p] : ringChar R = p :=
  ((Classical.choose_spec (CharP.existsUnique R)).2 p C).symm


instance charP : CharP R (ringChar R) :=
  ⟨spec R⟩


lemma of_eq {p : ℕ} (h : ringChar R = p) : CharP R p :=
  CharP.congr (ringChar R) h


lemma eq_iff {p : ℕ} : ringChar R = p ↔ CharP R p :=
  ⟨of_eq, @eq R _ p⟩


lemma dvd {x : ℕ} (hx : (x : R) = 0) : ringChar R ∣ x :=
  (spec R x).1 hx


@[simp]
lemma eq_zero [CharZero R] : ringChar R = 0 :=
  eq R 0


                                                     /-
                                                       R : Type u_1
                                                       inst✝ : NonAssocSemiring R
                                                       ⊢ Eq (↑(ringChar R)) 0
                                                     -/
lemma Nat.cast_ringChar : (ringChar R : R) = 0 := by rw [ringChar.spec]
                                                     /-
                                                       🎉 no goals
                                                     -/


lemma CharP.neg_one_ne_one [Ring R] (p : ℕ) [CharP R p] [Fact (2 < p)] : (-1 : R) ≠ (1 : R) := by
  suffices (2 : R) ≠ 0 by
    intro h
    symm at h
    rw [← sub_eq_zero, sub_neg_eq_add] at h
    norm_num at h
    exact this h
    -- Porting note: this could probably be golfed
  /-
    R : Type u_1
    inst✝² : Ring R
    p : Nat
    inst✝¹ : CharP R p
    inst✝ : Fact (LT.lt 2 p)
    ⊢ Ne 2 0
  -/
  intro h
  /-
    R : Type u_1
    inst✝² : Ring R
    p : Nat
    inst✝¹ : CharP R p
    inst✝ : Fact (LT.lt 2 p)
    h : Eq 2 0
    ⊢ False
  -/
  rw [show (2 : R) = (2 : ℕ) by norm_cast] at h
  /-
    R : Type u_1
    inst✝² : Ring R
    p : Nat
    inst✝¹ : CharP R p
    inst✝ : Fact (LT.lt 2 p)
    h : Eq (↑2) 0
    ⊢ False
  -/
  have := (CharP.cast_eq_zero_iff R p 2).mp h
  /-
    R : Type u_1
    inst✝² : Ring R
    p : Nat
    inst✝¹ : CharP R p
    inst✝ : Fact (LT.lt 2 p)
    h : Eq (↑2) 0
    this : Dvd.dvd p 2
    ⊢ False
  -/
  have := Nat.le_of_dvd (by decide) this
  /-
    R : Type u_1
    inst✝² : Ring R
    p : Nat
    inst✝¹ : CharP R p
    inst✝ : Fact (LT.lt 2 p)
    h : Eq (↑2) 0
    this✝ : Dvd.dvd p 2
    this : LE.le p 2
    ⊢ False
  -/
  rw [fact_iff] at *
  /-
    R : Type u_1
    inst✝² : Ring R
    p : Nat
    inst✝¹ : CharP R p
    inst✝ : LT.lt 2 p
    h : Eq (↑2) 0
    this✝ : Dvd.dvd p 2
    this : LE.le p 2
    ⊢ False
  -/
  omega
  /-
    🎉 no goals
  -/


lemma cast_eq_mod (p : ℕ) [CharP R p] (k : ℕ) : (k : R) = (k % p : ℕ) :=
  calc
                                           /-
                                             R : Type u_1
                                             inst✝¹ : NonAssocRing R
                                             p : Nat
                                             inst✝ : CharP R p
                                             k : Nat
                                             ⊢ Eq ↑k ↑(HAdd.hAdd (HMod.hMod k p) (HMul.hMul p (HDiv.hDiv k p)))
                                           -/
    (k : R) = ↑(k % p + p * (k / p)) := by rw [Nat.mod_add_div]
                                           /-
                                             🎉 no goals
                                           -/
                       /-
                         R : Type u_1
                         inst✝¹ : NonAssocRing R
                         p : Nat
                         inst✝ : CharP R p
                         k : Nat
                         ⊢ Eq ↑(HAdd.hAdd (HMod.hMod k p) (HMul.hMul p (HDiv.hDiv k p))) ↑(HMod.hMod k p)
                       -/
    _ = ↑(k % p) := by simp [cast_eq_zero]
                       /-
                         🎉 no goals
                       -/


lemma ringChar_zero_iff_CharZero : ringChar R = 0 ↔ CharZero R := by
  /-
    R : Type u_1
    inst✝ : NonAssocRing R
    ⊢ Iff (Eq (ringChar R) 0) (CharZero R)
  -/
  rw [ringChar.eq_iff, charP_zero_iff_charZero]
  /-
    🎉 no goals
  -/


lemma char_ne_one [Nontrivial R] (p : ℕ) [hc : CharP R p] : p ≠ 1 := fun hp : p = 1 =>
                           /-
                             R : Type u_1
                             inst✝¹ : NonAssocSemiring R
                             inst✝ : Nontrivial R
                             p : Nat
                             hc : CharP R p
                             hp : Eq p 1
                             ⊢ Eq 1 0
                           -/
  have : (1 : R) = 0 := by simpa using (cast_eq_zero_iff R p 1).mpr (hp ▸ dvd_refl p)
                           /-
                             🎉 no goals
                           -/
  absurd this one_ne_zero


lemma char_is_prime_of_two_le (p : ℕ) [CharP R p] (hp : 2 ≤ p) : Nat.Prime p :=
  suffices ∀ (d) (_ : d ∣ p), d = 1 ∨ d = p from Nat.prime_def.mpr ⟨hp, this⟩
  fun (d : ℕ) (hdvd : ∃ e, p = d * e) =>
  let ⟨e, hmul⟩ := hdvd
  have : (p : R) = 0 := (cast_eq_zero_iff R p p).mpr (dvd_refl p)
  have : (d : R) * e = 0 := @Nat.cast_mul R _ d e ▸ hmul ▸ this
  Or.elim (eq_zero_or_eq_zero_of_mul_eq_zero this)
    (fun hd : (d : R) = 0 =>
      have : p ∣ d := (cast_eq_zero_iff R p d).mp hd
      show d = 1 ∨ d = p from Or.inr (this.antisymm' ⟨e, hmul⟩))
    fun he : (e : R) = 0 =>
    have : p ∣ e := (cast_eq_zero_iff R p e).mp he
    have : e ∣ p := dvd_of_mul_left_eq d (Eq.symm hmul)
    have : e = p := ‹e ∣ p›.antisymm ‹p ∣ e›
                          /-
                            R : Type u_1
                            inst✝² : NonAssocSemiring R
                            inst✝¹ : NoZeroDivisors R
                            p : Nat
                            inst✝ : CharP R p
                            hp : LE.le 2 p
                            d : Nat
                            hdvd : Exists fun e => Eq p (HMul.hMul d e)
                            e : Nat
                            hmul : Eq p (HMul.hMul d e)
                            this✝³ : Eq (↑p) 0
                            this✝² : Eq (HMul.hMul ↑d ↑e) 0
                            he : Eq (↑e) 0
                            this✝¹ : Dvd.dvd p e
                            this✝ : Dvd.dvd e p
                            this : Eq e p
                            ⊢ LT.lt 0 p
                          -/
    have h₀ : 0 < p := by omega
                          /-
                            🎉 no goals
                          -/
                               /-
                                 R : Type u_1
                                 inst✝² : NonAssocSemiring R
                                 inst✝¹ : NoZeroDivisors R
                                 p : Nat
                                 inst✝ : CharP R p
                                 hp : LE.le 2 p
                                 d : Nat
                                 hdvd : Exists fun e => Eq p (HMul.hMul d e)
                                 e : Nat
                                 hmul : Eq p (HMul.hMul d e)
                                 this✝³ : Eq (↑p) 0
                                 this✝² : Eq (HMul.hMul ↑d ↑e) 0
                                 he : Eq (↑e) 0
                                 this✝¹ : Dvd.dvd p e
                                 this✝ : Dvd.dvd e p
                                 this : Eq e p
                                 h₀ : LT.lt 0 p
                                 ⊢ Eq (HMul.hMul d p) (HMul.hMul 1 p)
                               -/
    have : d * p = 1 * p := by rw [‹e = p›] at hmul; rw [one_mul]; exact Eq.symm hmul
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
    show d = 1 ∨ d = p from Or.inl (mul_right_cancel₀ h₀.ne' this)


lemma char_is_prime_or_zero (p : ℕ) [hc : CharP R p] : Nat.Prime p ∨ p = 0 :=
  match p, hc with
  | 0, _ => Or.inr rfl
  | 1, hc => absurd (Eq.refl (1 : ℕ)) (@char_ne_one R _ _ (1 : ℕ) hc)
  | m + 2, hc => Or.inl (@char_is_prime_of_two_le R _ _ (m + 2) hc (Nat.le_add_left 2 m))


/-- The characteristic is prime if it is non-zero. -/
lemma char_prime_of_ne_zero {p : ℕ} [CharP R p] (hp : p ≠ 0) : p.Prime :=
  (CharP.char_is_prime_or_zero R p).resolve_right hp


lemma exists' (R : Type*) [NonAssocRing R] [NoZeroDivisors R] [Nontrivial R] :
    CharZero R ∨ ∃ p : ℕ, Fact p.Prime ∧ CharP R p := by
  /-
    R : Type u_2
    inst✝² : NonAssocRing R
    inst✝¹ : NoZeroDivisors R
    inst✝ : Nontrivial R
    ⊢ Or (CharZero R) (Exists fun p => And (Fact (Nat.Prime p)) (CharP R p))
  -/
  obtain ⟨p, hchar⟩ := CharP.exists R
  /-
    case intro
    R : Type u_2
    inst✝² : NonAssocRing R
    inst✝¹ : NoZeroDivisors R
    inst✝ : Nontrivial R
    p : Nat
    hchar : CharP R p
    ⊢ Or (CharZero R) (Exists fun p => And (Fact (Nat.Prime p)) (CharP R p))
  -/
  rcases char_is_prime_or_zero R p with h | rfl
  /-
    case intro.inl
    R : Type u_2
    inst✝² : NonAssocRing R
    inst✝¹ : NoZeroDivisors R
    inst✝ : Nontrivial R
    p : Nat
    hchar : CharP R p
    h : Nat.Prime p
    ⊢ Or (CharZero R) (Exists fun p => And (Fact (Nat.Prime p)) (CharP R p))
  -/
  exacts [Or.inr ⟨p, Fact.mk h, hchar⟩, Or.inl (charP_to_charZero R)]
  /-
    🎉 no goals
  -/


lemma char_is_prime_of_pos (p : ℕ) [NeZero p] [CharP R p] : Fact p.Prime :=
  ⟨(CharP.char_is_prime_or_zero R _).resolve_right <| NeZero.ne p⟩


lemma CharOne.subsingleton [CharP R 1] : Subsingleton R :=
  Subsingleton.intro <|
                                                          /-
                                                            R : Type u_1
                                                            inst✝¹ : NonAssocSemiring R
                                                            inst✝ : CharP R 1
                                                            this : ∀ (r : R), Eq r 0
                                                            a b : R
                                                            ⊢ Eq a b
                                                          -/
    suffices ∀ r : R, r = 0 from fun a b => show a = b by rw [this a, this b]
                                                          /-
                                                            🎉 no goals
                                                          -/
                      /-
                        R : Type u_1
                        inst✝¹ : NonAssocSemiring R
                        inst✝ : CharP R 1
                        r : R
                        ⊢ Eq r (HMul.hMul 1 r)
                      -/
    fun r =>
                      /-
                        🎉 no goals
                      -/
                            /-
                              R : Type u_1
                              inst✝¹ : NonAssocSemiring R
                              inst✝ : CharP R 1
                              r : R
                              ⊢ Eq (HMul.hMul 1 r) (HMul.hMul (↑1) r)
                            -/
    calc
                            /-
                              🎉 no goals
                            -/
                      /-
                        R : Type u_1
                        inst✝¹ : NonAssocSemiring R
                        inst✝ : CharP R 1
                        r : R
                        ⊢ Eq (HMul.hMul (↑1) r) (HMul.hMul 0 r)
                      -/
      r = 1 * r := by rw [one_mul]
                      /-
                        🎉 no goals
                      -/
                  /-
                    R : Type u_1
                    inst✝¹ : NonAssocSemiring R
                    inst✝ : CharP R 1
                    r : R
                    ⊢ Eq (HMul.hMul 0 r) 0
                  -/
      _ = (1 : ℕ) * r := by rw [Nat.cast_one]
                  /-
                    🎉 no goals
                  -/
      _ = 0 * r := by rw [CharP.cast_eq_zero]
      _ = 0 := by rw [zero_mul]


lemma false_of_nontrivial_of_char_one [Nontrivial R] [CharP R 1] : False := by
  /-
    R : Type u_1
    inst✝² : NonAssocSemiring R
    inst✝¹ : Nontrivial R
    inst✝ : CharP R 1
    ⊢ False
  -/
  have : Subsingleton R := CharOne.subsingleton
  /-
    R : Type u_1
    inst✝² : NonAssocSemiring R
    inst✝¹ : Nontrivial R
    inst✝ : CharP R 1
    this : Subsingleton R
    ⊢ False
  -/
  exact false_of_nontrivial_of_subsingleton R
  /-
    🎉 no goals
  -/


lemma ringChar_ne_one [Nontrivial R] : ringChar R ≠ 1 := by
  /-
    R : Type u_1
    inst✝¹ : NonAssocSemiring R
    inst✝ : Nontrivial R
    ⊢ Ne (ringChar R) 1
  -/
  intro h
  /-
    R : Type u_1
    inst✝¹ : NonAssocSemiring R
    inst✝ : Nontrivial R
    h : Eq (ringChar R) 1
    ⊢ False
  -/
  apply zero_ne_one' R
  /-
    R : Type u_1
    inst✝¹ : NonAssocSemiring R
    inst✝ : Nontrivial R
    h : Eq (ringChar R) 1
    ⊢ Eq 0 1
  -/
  symm
  /-
    R : Type u_1
    inst✝¹ : NonAssocSemiring R
    inst✝ : Nontrivial R
    h : Eq (ringChar R) 1
    ⊢ Eq 1 0
  -/
  rw [← Nat.cast_one, ringChar.spec, h]
  /-
    🎉 no goals
  -/


lemma nontrivial_of_char_ne_one {v : ℕ} (hv : v ≠ 1) [hr : CharP R v] : Nontrivial R :=
  ⟨⟨(1 : ℕ), 0, fun h =>
               /-
                 R : Type u_1
                 inst✝ : NonAssocSemiring R
                 v : Nat
                 hv : Ne v 1
                 hr : CharP R v
                 h : Eq (↑1) 0
                 ⊢ Eq v 1
               -/
      hv <| by rwa [CharP.cast_eq_zero_iff _ v, Nat.dvd_one] at h⟩⟩
               /-
                 🎉 no goals
               -/


lemma of_not_dvd [CharP R p] (h : ¬p ∣ n) : NeZero (n : R) :=
  ⟨(CharP.cast_eq_zero_iff R p n).not.mpr h⟩


lemma not_char_dvd (p : ℕ) [CharP R p] (k : ℕ) [h : NeZero (k : R)] : ¬p ∣ k := by
  /-
    R : Type u_1
    inst✝¹ : AddMonoidWithOne R
    p : Nat
    inst✝ : CharP R p
    k : Nat
    h : NeZero ↑k
    ⊢ Not (Dvd.dvd p k)
  -/
  rwa [← CharP.cast_eq_zero_iff R p k, ← Ne, ← neZero_iff]
  /-
    🎉 no goals
  -/


/-- The definition of the exponential characteristic of a semiring. -/
class inductive ExpChar : ℕ → Prop
  | zero [CharZero R] : ExpChar 1
  | prime {q : ℕ} (hprime : q.Prime) [hchar : CharP R q] : ExpChar q


instance expChar_prime (p) [CharP R p] [Fact p.Prime] : ExpChar R p := ExpChar.prime Fact.out

instance expChar_one [CharZero R] : ExpChar R 1 := ExpChar.zero


lemma expChar_ne_zero (p : ℕ) [hR : ExpChar R p] : p ≠ 0 := by
  /-
    R : Type u_1
    inst✝ : AddMonoidWithOne R
    p : Nat
    hR : ExpChar R p
    ⊢ Ne p 0
  -/
  cases hR
    /-
      case zero
      R : Type u_1
      inst✝¹ : AddMonoidWithOne R
      inst✝ : CharZero R
      ⊢ Ne 1 0
    -/
  · exact one_ne_zero
    /-
      🎉 no goals
    -/
    /-
      case prime
      R : Type u_1
      inst✝ : AddMonoidWithOne R
      p : Nat
      hprime✝ : Nat.Prime p
      hchar✝ : CharP R p
      ⊢ Ne p 0
    -/
  · exact ‹p.Prime›.ne_zero
    /-
      🎉 no goals
    -/


variable {R} in
/-- The exponential characteristic is unique. -/
lemma ExpChar.eq {p q : ℕ} (hp : ExpChar R p) (hq : ExpChar R q) : p = q := by
  /-
    R : Type u_1
    inst✝ : AddMonoidWithOne R
    p q : Nat
    hp : ExpChar R p
    hq : ExpChar R q
    ⊢ Eq p q
  -/
  rcases hp with ⟨hp⟩ | ⟨hp'⟩
    /-
      case zero
      R : Type u_1
      inst✝¹ : AddMonoidWithOne R
      q : Nat
      hq : ExpChar R q
      inst✝ : CharZero R
      ⊢ Eq 1 q
    -/
  · rcases hq with hq | hq'
    /-
      case zero.zero
      R : Type u_1
      inst✝² : AddMonoidWithOne R
      inst✝¹ inst✝ : CharZero R
      ⊢ Eq 1 1
    -/
    exacts [rfl, False.elim (Nat.not_prime_zero (CharP.eq R ‹_› (CharP.ofCharZero R) ▸ hq'))]
    /-
      🎉 no goals
    -/
    /-
      case prime
      R : Type u_1
      inst✝ : AddMonoidWithOne R
      p q : Nat
      hq : ExpChar R q
      hp' : Nat.Prime p
      hchar✝ : CharP R p
      ⊢ Eq p q
    -/
  · rcases hq with hq | hq'
    exacts [False.elim (Nat.not_prime_zero (CharP.eq R ‹_› (CharP.ofCharZero R) ▸ hp')),
      CharP.eq R ‹_› ‹_›]


lemma ExpChar.congr {p : ℕ} (q : ℕ) [hq : ExpChar R q] (h : q = p) : ExpChar R p := h ▸ hq


/-- The exponential characteristic is one if the characteristic is zero. -/
lemma expChar_one_of_char_zero (q : ℕ) [hp : CharP R 0] [hq : ExpChar R q] : q = 1 := by
  /-
    R : Type u_1
    inst✝ : AddMonoidWithOne R
    q : Nat
    hp : CharP R 0
    hq : ExpChar R q
    ⊢ Eq q 1
  -/
  rcases hq with q | hq_prime
    /-
      case zero
      R : Type u_1
      inst✝¹ : AddMonoidWithOne R
      hp : CharP R 0
      inst✝ : CharZero R
      ⊢ Eq 1 1
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case prime
      R : Type u_1
      inst✝ : AddMonoidWithOne R
      q : Nat
      hp : CharP R 0
      hq_prime : Nat.Prime q
      hchar✝ : CharP R q
      ⊢ Eq q 1
    -/
  · exact False.elim <| hq_prime.ne_zero <| ‹CharP R q›.eq R hp
    /-
      🎉 no goals
    -/


/-- The characteristic equals the exponential characteristic iff the former is prime. -/
lemma char_eq_expChar_iff (p q : ℕ) [hp : CharP R p] [hq : ExpChar R q] : p = q ↔ p.Prime := by
  /-
    R : Type u_1
    inst✝ : AddMonoidWithOne R
    p q : Nat
    hp : CharP R p
    hq : ExpChar R q
    ⊢ Iff (Eq p q) (Nat.Prime p)
  -/
  rcases hq with q | hq_prime
    /-
      case zero
      R : Type u_1
      inst✝¹ : AddMonoidWithOne R
      p : Nat
      hp : CharP R p
      inst✝ : CharZero R
      ⊢ Iff (Eq p 1) (Nat.Prime p)
    -/
  · rw [(CharP.eq R hp inferInstance : p = 0)]
    /-
      case zero
      R : Type u_1
      inst✝¹ : AddMonoidWithOne R
      p : Nat
      hp : CharP R p
      inst✝ : CharZero R
      ⊢ Iff (Eq 0 1) (Nat.Prime 0)
    -/
    decide
    /-
      🎉 no goals
    -/
    /-
      case prime
      R : Type u_1
      inst✝ : AddMonoidWithOne R
      p q : Nat
      hp : CharP R p
      hq_prime : Nat.Prime q
      hchar✝ : CharP R q
      ⊢ Iff (Eq p q) (Nat.Prime p)
    -/
  · exact ⟨fun hpq => hpq.symm ▸ hq_prime, fun _ => CharP.eq R hp ‹CharP R q›⟩
    /-
      🎉 no goals
    -/


/-- The exponential characteristic is a prime number or one.
See also `CharP.char_is_prime_or_zero`. -/
lemma expChar_is_prime_or_one (q : ℕ) [hq : ExpChar R q] : Nat.Prime q ∨ q = 1 := by
  cases hq with
  | zero => exact .inr rfl
  | prime hp => exact .inl hp


/-- The exponential characteristic is positive. -/
lemma expChar_pos (q : ℕ) [ExpChar R q] : 0 < q := by
  /-
    R : Type u_1
    inst✝¹ : AddMonoidWithOne R
    q : Nat
    inst✝ : ExpChar R q
    ⊢ LT.lt 0 q
  -/
  rcases expChar_is_prime_or_one R q with h | rfl
  /-
    case inl
    R : Type u_1
    inst✝¹ : AddMonoidWithOne R
    q : Nat
    inst✝ : ExpChar R q
    h : Nat.Prime q
    ⊢ LT.lt 0 q
  -/
  exacts [Nat.Prime.pos h, Nat.one_pos]
  /-
    🎉 no goals
  -/


/-- Any power of the exponential characteristic is positive. -/
lemma expChar_pow_pos (q : ℕ) [ExpChar R q] (n : ℕ) : 0 < q ^ n :=
  Nat.pos_pow_of_pos n (expChar_pos R q)


/-- Noncomputable function that outputs the unique exponential characteristic of a semiring. -/
noncomputable def ringExpChar : ℕ := max (ringChar R) 1


lemma ringExpChar.eq (q : ℕ) [h : ExpChar R q] : ringExpChar R = q := by
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    q : Nat
    h : ExpChar R q
    ⊢ Eq (ringExpChar R) q
  -/
  rcases h with _ | h
    /-
      case zero
      R : Type u_1
      inst✝¹ : NonAssocSemiring R
      inst✝ : CharZero R
      ⊢ Eq (ringExpChar R) 1
    -/
  · haveI := CharP.ofCharZero R
    /-
      case zero
      R : Type u_1
      inst✝¹ : NonAssocSemiring R
      inst✝ : CharZero R
      this : CharP R 0
      ⊢ Eq (ringExpChar R) 1
    -/
    rw [ringExpChar, ringChar.eq R 0]; rfl
                                       /-
                                         🎉 no goals
                                       -/
  /-
    case prime
    R : Type u_1
    inst✝ : NonAssocSemiring R
    q : Nat
    h : Nat.Prime q
    hchar✝ : CharP R q
    ⊢ Eq (ringExpChar R) q
  -/
  rw [ringExpChar, ringChar.eq R q]
  /-
    case prime
    R : Type u_1
    inst✝ : NonAssocSemiring R
    q : Nat
    h : Nat.Prime q
    hchar✝ : CharP R q
    ⊢ Eq (Max.max q 1) q
  -/
  exact Nat.max_eq_left h.one_lt.le
  /-
    🎉 no goals
  -/


@[simp] lemma ringExpChar.eq_one [CharZero R] : ringExpChar R = 1 := by
  /-
    R : Type u_1
    inst✝¹ : NonAssocSemiring R
    inst✝ : CharZero R
    ⊢ Eq (ringExpChar R) 1
  -/
  rw [ringExpChar, ringChar.eq_zero, max_eq_right (Nat.zero_le _)]
  /-
    🎉 no goals
  -/


/-- The exponential characteristic is one if the characteristic is zero. -/
lemma char_zero_of_expChar_one (p : ℕ) [hp : CharP R p] [hq : ExpChar R 1] : p = 0 := by
  /-
    R : Type u_1
    inst✝¹ : NonAssocSemiring R
    inst✝ : Nontrivial R
    p : Nat
    hp : CharP R p
    hq : ExpChar R 1
    ⊢ Eq p 0
  -/
  cases hq
    /-
      case zero
      R : Type u_1
      inst✝² : NonAssocSemiring R
      inst✝¹ : Nontrivial R
      p : Nat
      hp : CharP R p
      inst✝ : CharZero R
      ⊢ Eq p 0
    -/
  · exact CharP.eq R hp inferInstance
    /-
      🎉 no goals
    -/
    /-
      case prime
      R : Type u_1
      inst✝¹ : NonAssocSemiring R
      inst✝ : Nontrivial R
      p : Nat
      hp : CharP R p
      hprime✝ : Nat.Prime 1
      hchar✝ : CharP R 1
      ⊢ Eq p 0
    -/
  · exact False.elim (CharP.char_ne_one R 1 rfl)
    /-
      🎉 no goals
    -/

-- This could be an instance, but there are no `ExpChar R 1` instances in mathlib.

/-- The characteristic is zero if the exponential characteristic is one. -/
lemma charZero_of_expChar_one' [hq : ExpChar R 1] : CharZero R := by
  /-
    R : Type u_1
    inst✝¹ : NonAssocSemiring R
    inst✝ : Nontrivial R
    hq : ExpChar R 1
    ⊢ CharZero R
  -/
  cases hq
    /-
      case zero
      R : Type u_1
      inst✝² : NonAssocSemiring R
      inst✝¹ : Nontrivial R
      inst✝ : CharZero R
      ⊢ CharZero R
    -/
  · assumption
    /-
      🎉 no goals
    -/
    /-
      case prime
      R : Type u_1
      inst✝¹ : NonAssocSemiring R
      inst✝ : Nontrivial R
      hprime✝ : Nat.Prime 1
      hchar✝ : CharP R 1
      ⊢ CharZero R
    -/
  · exact False.elim (CharP.char_ne_one R 1 rfl)
    /-
      🎉 no goals
    -/


/-- The exponential characteristic is one iff the characteristic is zero. -/
lemma expChar_one_iff_char_zero (p q : ℕ) [CharP R p] [ExpChar R q] : q = 1 ↔ p = 0 := by
  /-
    R : Type u_1
    inst✝³ : NonAssocSemiring R
    inst✝² : Nontrivial R
    p q : Nat
    inst✝¹ : CharP R p
    inst✝ : ExpChar R q
    ⊢ Iff (Eq q 1) (Eq p 0)
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝³ : NonAssocSemiring R
      inst✝² : Nontrivial R
      p q : Nat
      inst✝¹ : CharP R p
      inst✝ : ExpChar R q
      ⊢ Eq q 1 → Eq p 0
    -/
  · rintro rfl
    /-
      case mp
      R : Type u_1
      inst✝³ : NonAssocSemiring R
      inst✝² : Nontrivial R
      p : Nat
      inst✝¹ : CharP R p
      inst✝ : ExpChar R 1
      ⊢ Eq p 0
    -/
    exact char_zero_of_expChar_one R p
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝³ : NonAssocSemiring R
      inst✝² : Nontrivial R
      p q : Nat
      inst✝¹ : CharP R p
      inst✝ : ExpChar R q
      ⊢ Eq p 0 → Eq q 1
    -/
  · rintro rfl
    /-
      case mpr
      R : Type u_1
      inst✝³ : NonAssocSemiring R
      inst✝² : Nontrivial R
      q : Nat
      inst✝¹ : ExpChar R q
      inst✝ : CharP R 0
      ⊢ Eq q 1
    -/
    exact expChar_one_of_char_zero R q
    /-
      🎉 no goals
    -/


lemma ExpChar.exists [Ring R] [IsDomain R] : ∃ q, ExpChar R q := by
  /-
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : IsDomain R
    ⊢ Exists fun q => ExpChar R q
  -/
  obtain _ | ⟨p, ⟨hp⟩, _⟩ := CharP.exists' R
  /-
    case inl
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : IsDomain R
    h✝ : CharZero R
    ⊢ Exists fun q => ExpChar R q
  -/
  exacts [⟨1, .zero⟩, ⟨p, .prime hp⟩]
  /-
    🎉 no goals
  -/


lemma ExpChar.exists_unique [Ring R] [IsDomain R] : ∃! q, ExpChar R q :=
  let ⟨q, H⟩ := ExpChar.exists R
  ⟨q, H, fun _ H2 ↦ ExpChar.eq H2 H⟩


instance ringExpChar.expChar [Ring R] [IsDomain R] : ExpChar R (ringExpChar R) := by
  /-
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : IsDomain R
    ⊢ ExpChar R (ringExpChar R)
  -/
  obtain ⟨q, _⟩ := ExpChar.exists R
  /-
    case intro
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : IsDomain R
    q : Nat
    h✝ : ExpChar R q
    ⊢ ExpChar R (ringExpChar R)
  -/
  rwa [ringExpChar.eq R q]
  /-
    🎉 no goals
  -/


variable {R} in
lemma ringExpChar.of_eq [Ring R] [IsDomain R] {q : ℕ} (h : ringExpChar R = q) : ExpChar R q :=
  h ▸ ringExpChar.expChar R


variable {R} in
lemma ringExpChar.eq_iff [Ring R] [IsDomain R] {q : ℕ} : ringExpChar R = q ↔ ExpChar R q :=
  ⟨ringExpChar.of_eq, fun _ ↦ ringExpChar.eq R q⟩


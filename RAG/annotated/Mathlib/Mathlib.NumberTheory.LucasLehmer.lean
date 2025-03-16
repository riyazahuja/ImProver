/-- The Mersenne numbers, 2^p - 1. -/
def mersenne (p : ℕ) : ℕ :=
  2 ^ p - 1


theorem strictMono_mersenne : StrictMono mersenne := fun m n h ↦
                                                                   /-
                                                                     m n : Nat
                                                                     h : LT.lt m n
                                                                     ⊢ LT.lt (HPow.hPow 2 m) (HPow.hPow 2 n)
                                                                   -/
  (Nat.sub_lt_sub_iff_right <| Nat.one_le_pow _ _ two_pos).2 <| by gcongr; norm_num1
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp]
theorem mersenne_lt_mersenne {p q : ℕ} : mersenne p < mersenne q ↔ p < q :=
  strictMono_mersenne.lt_iff_lt


@[gcongr] protected alias ⟨_, GCongr.mersenne_lt_mersenne⟩ := mersenne_lt_mersenne


@[simp]
theorem mersenne_le_mersenne {p q : ℕ} : mersenne p ≤ mersenne q ↔ p ≤ q :=
  strictMono_mersenne.le_iff_le


@[gcongr] protected alias ⟨_, GCongr.mersenne_le_mersenne⟩ := mersenne_le_mersenne


@[simp] theorem mersenne_zero : mersenne 0 = 0 := rfl


@[simp] lemma mersenne_odd : ∀ {p : ℕ}, Odd (mersenne p) ↔ p ≠ 0
            /-
              ⊢ Iff (Odd (mersenne 0)) (Ne 0 0)
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
  | p + 1 => by
    simpa using Nat.Even.sub_odd (one_le_pow₀ one_le_two)
      (even_two.pow_of_ne_zero p.succ_ne_zero) odd_one


@[simp] theorem mersenne_pos {p : ℕ} : 0 < mersenne p ↔ 0 < p := mersenne_lt_mersenne (p := 0)


alias ⟨_, mersenne_pos_of_pos⟩ := mersenne_pos


/-- Extension for the `positivity` tactic: `mersenne`. -/
@[positivity mersenne _]
def evalMersenne : PositivityExt where eval {u α} _zα _pα e := do
  match u, α, e with
  | 0, ~q(ℕ), ~q(mersenne $a) =>
    let ra ← core q(inferInstance) q(inferInstance) a
    assertInstancesCommute
    match ra with
    | .positive pa => pure (.positive q(mersenne_pos_of_pos $pa))
    | _ => pure (.nonnegative q(Nat.zero_le (mersenne $a)))
  | _, _, _ => throwError "not mersenne"


@[simp]
theorem one_lt_mersenne {p : ℕ} : 1 < mersenne p ↔ 1 < p :=
  mersenne_lt_mersenne (p := 1)


@[simp]
theorem succ_mersenne (k : ℕ) : mersenne k + 1 = 2 ^ k := by
  /-
    k : Nat
    ⊢ Eq (HAdd.hAdd (mersenne k) 1) (HPow.hPow 2 k)
  -/
  rw [mersenne, tsub_add_cancel_of_le]
  /-
    k : Nat
    ⊢ LE.le 1 (HPow.hPow 2 k)
  -/
  exact one_le_pow₀ (by norm_num)
  /-
    🎉 no goals
  -/


/-- The recurrence `s (i+1) = (s i)^2 - 2` in `ℤ`. -/
def s : ℕ → ℤ
  | 0 => 4
  | i + 1 => s i ^ 2 - 2


/-- The recurrence `s (i+1) = (s i)^2 - 2` in `ZMod (2^p - 1)`. -/
def sZMod (p : ℕ) : ℕ → ZMod (2 ^ p - 1)
  | 0 => 4
  | i + 1 => sZMod p i ^ 2 - 2


/-- The recurrence `s (i+1) = ((s i)^2 - 2) % (2^p - 1)` in `ℤ`. -/
def sMod (p : ℕ) : ℕ → ℤ
  | 0 => 4 % (2 ^ p - 1)
  | i + 1 => (sMod p i ^ 2 - 2) % (2 ^ p - 1)


theorem mersenne_int_pos {p : ℕ} (hp : p ≠ 0) : (0 : ℤ) < 2 ^ p - 1 :=
  sub_pos.2 <| mod_cast Nat.one_lt_two_pow hp


theorem mersenne_int_ne_zero (p : ℕ) (hp : p ≠ 0) : (2 ^ p - 1 : ℤ) ≠ 0 :=
  (mersenne_int_pos hp).ne'


theorem sMod_nonneg (p : ℕ) (hp : p ≠ 0) (i : ℕ) : 0 ≤ sMod p i := by
  /-
    p : Nat
    hp : Ne p 0
    i : Nat
    ⊢ LE.le 0 (LucasLehmer.sMod p i)
  -/
  cases i <;> dsimp [sMod]
    /-
      case zero
      p : Nat
      hp : Ne p 0
      ⊢ LE.le 0 (HMod.hMod 4 (HSub.hSub (HPow.hPow 2 p) 1))
    -/
  · exact sup_eq_right.mp rfl
    /-
      🎉 no goals
    -/
    /-
      case succ
      p : Nat
      hp : Ne p 0
      n✝ : Nat
      ⊢ LE.le 0 (HMod.hMod (HSub.hSub (HPow.hPow (LucasLehmer.sMod p n✝) 2) 2) (HSub …
    -/
  · apply Int.emod_nonneg
    /-
      case succ.a
      p : Nat
      hp : Ne p 0
      n✝ : Nat
      ⊢ Ne (HSub.hSub (HPow.hPow 2 p) 1) 0
    -/
    exact mersenne_int_ne_zero p hp
    /-
      🎉 no goals
    -/


                                                                     /-
                                                                       p i : Nat
                                                                       ⊢ Eq (HMod.hMod (LucasLehmer.sMod p i) (HSub.hSub (HPow.hPow 2 p) 1)) (LucasLe …
                                                                     -/
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
theorem sMod_mod (p i : ℕ) : sMod p i % (2 ^ p - 1) = sMod p i := by cases i <;> simp [sMod]
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


theorem sMod_lt (p : ℕ) (hp : p ≠ 0) (i : ℕ) : sMod p i < 2 ^ p - 1 := by
  /-
    p : Nat
    hp : Ne p 0
    i : Nat
    ⊢ LT.lt (LucasLehmer.sMod p i) (HSub.hSub (HPow.hPow 2 p) 1)
  -/
  rw [← sMod_mod]
  /-
    p : Nat
    hp : Ne p 0
    i : Nat
    ⊢ LT.lt (HMod.hMod (LucasLehmer.sMod p i) (HSub.hSub (HPow.hPow 2 p) 1)) (HSub …
  -/
  refine (Int.emod_lt _ (mersenne_int_ne_zero p hp)).trans_eq ?_
  /-
    p : Nat
    hp : Ne p 0
    i : Nat
    ⊢ Eq (abs (HSub.hSub (HPow.hPow 2 p) 1)) (HSub.hSub (HPow.hPow 2 p) 1)
  -/
  exact abs_of_nonneg (mersenne_int_pos hp).le
  /-
    🎉 no goals
  -/


theorem sZMod_eq_s (p' : ℕ) (i : ℕ) : sZMod (p' + 2) i = (s i : ZMod (2 ^ (p' + 2) - 1)) := by
  /-
    p' i : Nat
    ⊢ Eq (LucasLehmer.sZMod (HAdd.hAdd p' 2) i) ↑(LucasLehmer.s i)
  -/
  induction' i with i ih
    /-
      case zero
      p' : Nat
      ⊢ Eq (LucasLehmer.sZMod (HAdd.hAdd p' 2) 0) ↑(LucasLehmer.s 0)
    -/
  · dsimp [s, sZMod]
    /-
      case zero
      p' : Nat
      ⊢ Eq 4 ↑4
    -/
    norm_num
    /-
      🎉 no goals
    -/
    /-
      case succ
      p' i : Nat
      ih : Eq (LucasLehmer.sZMod (HAdd.hAdd p' 2) i) ↑(LucasLehmer.s i)
      ⊢ Eq (LucasLehmer.sZMod (HAdd.hAdd p' 2) (HAdd.hAdd i 1)) ↑(LucasLehmer.s (HAd …
    -/
  · push_cast [s, sZMod, ih]; rfl
                              /-
                                🎉 no goals
                              -/

-- These next two don't make good `norm_cast` lemmas.

theorem Int.natCast_pow_pred (b p : ℕ) (w : 0 < b) : ((b ^ p - 1 : ℕ) : ℤ) = (b : ℤ) ^ p - 1 := by
  /-
    b p : Nat
    w : LT.lt 0 b
    ⊢ Eq (↑(HSub.hSub (HPow.hPow b p) 1)) (HSub.hSub (HPow.hPow (↑b) p) 1)
  -/
  have : 1 ≤ b ^ p := Nat.one_le_pow p b w
  /-
    b p : Nat
    w : LT.lt 0 b
    this : LE.le 1 (HPow.hPow b p)
    ⊢ Eq (↑(HSub.hSub (HPow.hPow b p) 1)) (HSub.hSub (HPow.hPow (↑b) p) 1)
  -/
  norm_cast
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-05-25")] alias Int.coe_nat_pow_pred := Int.natCast_pow_pred


theorem Int.coe_nat_two_pow_pred (p : ℕ) : ((2 ^ p - 1 : ℕ) : ℤ) = (2 ^ p - 1 : ℤ) :=
                               /-
                                 p : Nat
                                 ⊢ LT.lt 0 2
                               -/
  Int.natCast_pow_pred 2 p (by decide)
                               /-
                                 🎉 no goals
                               -/


theorem sZMod_eq_sMod (p : ℕ) (i : ℕ) : sZMod p i = (sMod p i : ZMod (2 ^ p - 1)) := by
  /-
    p i : Nat
    ⊢ Eq (LucasLehmer.sZMod p i) ↑(LucasLehmer.sMod p i)
  -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
  induction i <;> push_cast [← Int.coe_nat_two_pow_pred p, sMod, sZMod, *] <;> rfl
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- The Lucas-Lehmer residue is `s p (p-2)` in `ZMod (2^p - 1)`. -/
def lucasLehmerResidue (p : ℕ) : ZMod (2 ^ p - 1) :=
  sZMod p (p - 2)


theorem residue_eq_zero_iff_sMod_eq_zero (p : ℕ) (w : 1 < p) :
    lucasLehmerResidue p = 0 ↔ sMod p (p - 2) = 0 := by
  /-
    p : Nat
    w : LT.lt 1 p
    ⊢ Iff (Eq (LucasLehmer.lucasLehmerResidue p) 0) (Eq (LucasLehmer.sMod p (HSub. …
  -/
  dsimp [lucasLehmerResidue]
  /-
    p : Nat
    w : LT.lt 1 p
    ⊢ Iff (Eq (LucasLehmer.sZMod p (HSub.hSub p 2)) 0) (Eq (LucasLehmer.sMod p (HS …
  -/
  rw [sZMod_eq_sMod p]
  /-
    p : Nat
    w : LT.lt 1 p
    ⊢ Iff (Eq (↑(LucasLehmer.sMod p (HSub.hSub p 2))) 0) (Eq (LucasLehmer.sMod p ( …
  -/
  constructor
  · -- We want to use that fact that `0 ≤ s_mod p (p-2) < 2^p - 1`
    -- and `lucas_lehmer_residue p = 0 → 2^p - 1 ∣ s_mod p (p-2)`.
    /-
      case mp
      p : Nat
      w : LT.lt 1 p
      ⊢ Eq (↑(LucasLehmer.sMod p (HSub.hSub p 2))) 0 → Eq (LucasLehmer.sMod p (HSub. …
    -/
    intro h
    simp? [ZMod.intCast_zmod_eq_zero_iff_dvd] at h says
      simp only [ZMod.intCast_zmod_eq_zero_iff_dvd, ofNat_pos, pow_pos, cast_pred,
        cast_pow, cast_ofNat] at h
    /-
      case mp
      p : Nat
      w : LT.lt 1 p
      h : Dvd.dvd (HSub.hSub (HPow.hPow 2 p) 1) (LucasLehmer.sMod p (HSub.hSub p 2))
      ⊢ Eq (LucasLehmer.sMod p (HSub.hSub p 2)) 0
    -/
    apply Int.eq_zero_of_dvd_of_nonneg_of_lt _ _ h <;> clear h
      /-
        p : Nat
        w : LT.lt 1 p
        ⊢ LE.le 0 (LucasLehmer.sMod p (HSub.hSub p 2))
      -/
    · exact sMod_nonneg _ (by positivity) _
      /-
        🎉 no goals
      -/
      /-
        p : Nat
        w : LT.lt 1 p
        ⊢ LT.lt (LucasLehmer.sMod p (HSub.hSub p 2)) (HSub.hSub (HPow.hPow 2 p) 1)
      -/
    · exact sMod_lt _ (by positivity) _
      /-
        🎉 no goals
      -/
    /-
      case mpr
      p : Nat
      w : LT.lt 1 p
      ⊢ Eq (LucasLehmer.sMod p (HSub.hSub p 2)) 0 → Eq (↑(LucasLehmer.sMod p (HSub.h …
    -/
  · intro h
    /-
      case mpr
      p : Nat
      w : LT.lt 1 p
      h : Eq (LucasLehmer.sMod p (HSub.hSub p 2)) 0
      ⊢ Eq (↑(LucasLehmer.sMod p (HSub.hSub p 2))) 0
    -/
    rw [h]
    /-
      case mpr
      p : Nat
      w : LT.lt 1 p
      h : Eq (LucasLehmer.sMod p (HSub.hSub p 2)) 0
      ⊢ Eq (↑0) 0
    -/
    simp
    /-
      🎉 no goals
    -/


/-- **Lucas-Lehmer Test**: a Mersenne number `2^p-1` is prime if and only if
the Lucas-Lehmer residue `s p (p-2) % (2^p - 1)` is zero.
-/
def LucasLehmerTest (p : ℕ) : Prop :=
  lucasLehmerResidue p = 0

-- Porting note: We have a fast `norm_num` extension, and we would rather use that than accidentally
-- have `simp` use `decide`!
/-
instance : DecidablePred LucasLehmerTest :=
  inferInstanceAs (DecidablePred (lucasLehmerResidue · = 0))
-/


/-- `q` is defined as the minimum factor of `mersenne p`, bundled as an `ℕ+`. -/
def q (p : ℕ) : ℕ+ :=
  ⟨Nat.minFac (mersenne p), Nat.minFac_pos (mersenne p)⟩

-- It would be nice to define this as (ℤ/qℤ)[x] / (x^2 - 3),
-- obtaining the ring structure for free,
-- but that seems to be more trouble than it's worth;
-- if it were easy to make the definition,
-- cardinality calculations would be somewhat more involved, too.

/-- We construct the ring `X q` as ℤ/qℤ + √3 ℤ/qℤ. -/
def X (q : ℕ+) : Type :=
  ZMod q × ZMod q


instance : Inhabited (X q) := inferInstanceAs (Inhabited (ZMod q × ZMod q))

instance : Fintype (X q) := inferInstanceAs (Fintype (ZMod q × ZMod q))

instance : DecidableEq (X q) := inferInstanceAs (DecidableEq (ZMod q × ZMod q))

instance : AddCommGroup (X q) := inferInstanceAs (AddCommGroup (ZMod q × ZMod q))


@[ext]
theorem ext {x y : X q} (h₁ : x.1 = y.1) (h₂ : x.2 = y.2) : x = y := by
  /-
    q : PNat
    x y : LucasLehmer.X q
    h₁ : Eq x.1 y.1
    h₂ : Eq x.2 y.2
    ⊢ Eq x y
  -/
  cases x; cases y; congr
                    /-
                      🎉 no goals
                    -/


@[simp] theorem zero_fst : (0 : X q).1 = 0 := rfl

@[simp] theorem zero_snd : (0 : X q).2 = 0 := rfl


@[simp]
theorem add_fst (x y : X q) : (x + y).1 = x.1 + y.1 :=
  rfl


@[simp]
theorem add_snd (x y : X q) : (x + y).2 = x.2 + y.2 :=
  rfl


@[simp]
theorem neg_fst (x : X q) : (-x).1 = -x.1 :=
  rfl


@[simp]
theorem neg_snd (x : X q) : (-x).2 = -x.2 :=
  rfl


instance : Mul (X q) where mul x y := (x.1 * y.1 + 3 * x.2 * y.2, x.1 * y.2 + x.2 * y.1)


@[simp]
theorem mul_fst (x y : X q) : (x * y).1 = x.1 * y.1 + 3 * x.2 * y.2 :=
  rfl


@[simp]
theorem mul_snd (x y : X q) : (x * y).2 = x.1 * y.2 + x.2 * y.1 :=
  rfl


instance : One (X q) where one := ⟨1, 0⟩


@[simp]
theorem one_fst : (1 : X q).1 = 1 :=
  rfl


@[simp]
theorem one_snd : (1 : X q).2 = 0 :=
  rfl


instance : Monoid (X q) :=
  { inferInstanceAs (Mul (X q)), inferInstanceAs (One (X q)) with
                                 /-
                                   q : PNat
                                   x y z : LucasLehmer.X q
                                   ⊢ Eq (HMul.hMul (HMul.hMul x y) z) (HMul.hMul x (HMul.hMul y z))
                                 -/
                                                   /-
                                                     🎉 no goals
                                                   -/
    mul_assoc := fun x y z => by ext <;> dsimp <;> ring
                                                   /-
                                                     🎉 no goals
                                                   -/
                           /-
                             q : PNat
                             x : LucasLehmer.X q
                             ⊢ Eq (HMul.hMul 1 x) x
                           -/
                                   /-
                                     🎉 no goals
                                   -/
    one_mul := fun x => by ext <;> simp
                                   /-
                                     🎉 no goals
                                   -/
                           /-
                             q : PNat
                             x : LucasLehmer.X q
                             ⊢ Eq (HMul.hMul x 1) x
                           -/
                                   /-
                                     🎉 no goals
                                   -/
    mul_one := fun x => by ext <;> simp }
                                   /-
                                     🎉 no goals
                                   -/


instance : NatCast (X q) where
    natCast := fun n => ⟨n, 0⟩


@[simp] theorem fst_natCast (n : ℕ) : (n : X q).fst = (n : ZMod q) := rfl


@[simp] theorem snd_natCast (n : ℕ) : (n : X q).snd = (0 : ZMod q) := rfl

-- See note [no_index around OfNat.ofNat]

@[simp] theorem ofNat_fst (n : ℕ) [n.AtLeastTwo] :
    (no_index (OfNat.ofNat n) : X q).fst = OfNat.ofNat n :=
  rfl

-- See note [no_index around OfNat.ofNat]

@[simp] theorem ofNat_snd (n : ℕ) [n.AtLeastTwo] :
    (no_index (OfNat.ofNat n) : X q).snd = 0 :=
  rfl


instance : AddGroupWithOne (X q) :=
  { inferInstanceAs (Monoid (X q)), inferInstanceAs (AddCommGroup (X q)),
      inferInstanceAs (NatCast (X q)) with
                       /-
                         q : PNat
                         ⊢ Eq (NatCast.natCast 0) 0
                       -/
                               /-
                                 🎉 no goals
                               -/
    natCast_zero := by ext <;> simp
                               /-
                                 🎉 no goals
                               -/
                               /-
                                 q : PNat
                                 x✝ : Nat
                                 ⊢ Eq (NatCast.natCast (HAdd.hAdd x✝ 1)) (HAdd.hAdd (NatCast.natCast x✝) 1)
                               -/
                                       /-
                                         🎉 no goals
                                       -/
    natCast_succ := fun _ ↦ by ext <;> simp
                                       /-
                                         🎉 no goals
                                       -/
    intCast := fun n => ⟨n, 0⟩
                                 /-
                                   q : PNat
                                   n : Nat
                                   ⊢ Eq (IntCast.intCast ↑n) ↑n
                                 -/
                                         /-
                                           🎉 no goals
                                         -/
    intCast_ofNat := fun n => by ext <;> simp
                                         /-
                                           🎉 no goals
                                         -/
                                   /-
                                     q : PNat
                                     n : Nat
                                     ⊢ Eq (IntCast.intCast (Int.negSucc n)) (Neg.neg ↑(HAdd.hAdd n 1))
                                   -/
                                           /-
                                             🎉 no goals
                                           -/
    intCast_negSucc := fun n => by ext <;> simp }
                                           /-
                                             🎉 no goals
                                           -/


theorem left_distrib (x y z : X q) : x * (y + z) = x * y + x * z := by
  /-
    q : PNat
    x y z : LucasLehmer.X q
    ⊢ Eq (HMul.hMul x (HAdd.hAdd y z)) (HAdd.hAdd (HMul.hMul x y) (HMul.hMul x z))
  -/
                    /-
                      🎉 no goals
                    -/
  ext <;> dsimp <;> ring
                    /-
                      🎉 no goals
                    -/


theorem right_distrib (x y z : X q) : (x + y) * z = x * z + y * z := by
  /-
    q : PNat
    x y z : LucasLehmer.X q
    ⊢ Eq (HMul.hMul (HAdd.hAdd x y) z) (HAdd.hAdd (HMul.hMul x z) (HMul.hMul y z))
  -/
                    /-
                      🎉 no goals
                    -/
  ext <;> dsimp <;> ring
                    /-
                      🎉 no goals
                    -/


instance : Ring (X q) :=
  { inferInstanceAs (AddGroupWithOne (X q)), inferInstanceAs (AddCommGroup (X q)),
      inferInstanceAs (Monoid (X q)) with
    left_distrib := left_distrib
    right_distrib := right_distrib
                           /-
                             q : PNat
                             x✝ : LucasLehmer.X q
                             ⊢ Eq (HMul.hMul x✝ 0) 0
                           -/
                           /-
                             q : PNat
                             x✝ : LucasLehmer.X q
                             ⊢ Eq (HMul.hMul 0 x✝) 0
                           -/
                                   /-
                                     🎉 no goals
                                   -/
                                   /-
                                     🎉 no goals
                                   -/
                                   /-
                                     🎉 no goals
                                   -/
    mul_zero := fun _ ↦ by ext <;> simp
                                   /-
                                     🎉 no goals
                                   -/
    zero_mul := fun _ ↦ by ext <;> simp }


instance : CommRing (X q) :=
  { inferInstanceAs (Ring (X q)) with
                             /-
                               q : PNat
                               x✝¹ x✝ : LucasLehmer.X q
                               ⊢ Eq (HMul.hMul x✝¹ x✝) (HMul.hMul x✝ x✝¹)
                             -/
                                               /-
                                                 🎉 no goals
                                               -/
    mul_comm := fun _ _ ↦ by ext <;> dsimp <;> ring }
                                               /-
                                                 🎉 no goals
                                               -/


instance [Fact (1 < (q : ℕ))] : Nontrivial (X q) :=
  ⟨⟨0, 1, ne_of_apply_ne Prod.fst zero_ne_one⟩⟩


@[simp]
theorem fst_intCast (n : ℤ) : (n : X q).fst = (n : ZMod q) :=
  rfl


@[simp]
theorem snd_intCast (n : ℤ) : (n : X q).snd = (0 : ZMod q) :=
  rfl


@[deprecated (since := "2024-05-25")] alias nat_coe_fst := fst_natCast

@[deprecated (since := "2024-05-25")] alias nat_coe_snd := snd_natCast

@[deprecated (since := "2024-05-25")] alias int_coe_fst := fst_intCast

@[deprecated (since := "2024-05-25")] alias int_coe_snd := snd_intCast


@[norm_cast]
                                                                              /-
                                                                                q : PNat
                                                                                n m : Int
                                                                                ⊢ Eq (↑(HMul.hMul n m)) (HMul.hMul ↑n ↑m)
                                                                              -/
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
theorem coe_mul (n m : ℤ) : ((n * m : ℤ) : X q) = (n : X q) * (m : X q) := by ext <;> simp
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


@[norm_cast]
                                                                /-
                                                                  q : PNat
                                                                  n : Nat
                                                                  ⊢ Eq ↑↑n ↑n
                                                                -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
theorem coe_natCast (n : ℕ) : ((n : ℤ) : X q) = (n : X q) := by ext <;> simp
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[deprecated (since := "2024-04-05")] alias coe_nat := coe_natCast


/-- The cardinality of `X` is `q^2`. -/
theorem card_eq : Fintype.card (X q) = q ^ 2 := by
  /-
    q : PNat
    ⊢ Eq (Fintype.card (LucasLehmer.X q)) (HPow.hPow (↑q) 2)
  -/
  dsimp [X]
  /-
    q : PNat
    ⊢ Eq (Fintype.card (Prod (ZMod ↑q) (ZMod ↑q))) (HPow.hPow (↑q) 2)
  -/
  rw [Fintype.card_prod, ZMod.card q, sq]
  /-
    🎉 no goals
  -/


/-- There are strictly fewer than `q^2` units, since `0` is not a unit. -/
nonrec theorem card_units_lt (w : 1 < q) : Fintype.card (X q)ˣ < q ^ 2 := by
  /-
    q : PNat
    w : LT.lt 1 q
    ⊢ LT.lt (Fintype.card (Units (LucasLehmer.X q))) (HPow.hPow (↑q) 2)
  -/
  have : Fact (1 < (q : ℕ)) := ⟨w⟩
  /-
    q : PNat
    w : LT.lt 1 q
    this : Fact (LT.lt 1 ↑q)
    ⊢ LT.lt (Fintype.card (Units (LucasLehmer.X q))) (HPow.hPow (↑q) 2)
  -/
  convert card_units_lt (X q)
  /-
    case h.e'_4
    q : PNat
    w : LT.lt 1 q
    this : Fact (LT.lt 1 ↑q)
    ⊢ Eq (HPow.hPow (↑q) 2) (Fintype.card (LucasLehmer.X q))
  -/
  rw [card_eq]
  /-
    🎉 no goals
  -/


/-- We define `ω = 2 + √3`. -/
def ω : X q := (2, 1)


/-- We define `ωb = 2 - √3`, which is the inverse of `ω`. -/
def ωb : X q := (2, -1)


theorem ω_mul_ωb (q : ℕ+) : (ω : X q) * ωb = 1 := by
  /-
    q : PNat
    ⊢ Eq (HMul.hMul LucasLehmer.X.ω LucasLehmer.X.ωb) 1
  -/
  dsimp [ω, ωb]
  /-
    q : PNat
    ⊢ Eq (HMul.hMul { fst := 2, snd := 1 } { fst := 2, snd := -1 }) 1
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp; ring
                /-
                  🎉 no goals
                -/


theorem ωb_mul_ω (q : ℕ+) : (ωb : X q) * ω = 1 := by
  /-
    q : PNat
    ⊢ Eq (HMul.hMul LucasLehmer.X.ωb LucasLehmer.X.ω) 1
  -/
  rw [mul_comm, ω_mul_ωb]
  /-
    🎉 no goals
  -/


/-- A closed form for the recurrence relation. -/
theorem closed_form (i : ℕ) : (s i : X q) = (ω : X q) ^ 2 ^ i + (ωb : X q) ^ 2 ^ i := by
  /-
    q : PNat
    i : Nat
    ⊢ Eq (↑(LucasLehmer.s i)) (HAdd.hAdd (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 i …
  -/
  induction' i with i ih
    /-
      case zero
      q : PNat
      ⊢ Eq (↑(LucasLehmer.s 0)) (HAdd.hAdd (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 0 …
    -/
  · dsimp [s, ω, ωb]
    /-
      case zero
      q : PNat
      ⊢ Eq (↑4) (HAdd.hAdd (HPow.hPow { fst := 2, snd := 1 } 1) (HPow.hPow { fst :=  …
    -/
            /-
              🎉 no goals
            -/
    ext <;> norm_num
            /-
              🎉 no goals
            -/
  · calc
      (s (i + 1) : X q) = (s i ^ 2 - 2 : ℤ) := rfl
      _ = (s i : X q) ^ 2 - 2 := by push_cast; rfl
      _ = (ω ^ 2 ^ i + ωb ^ 2 ^ i) ^ 2 - 2 := by rw [ih]
      _ = (ω ^ 2 ^ i) ^ 2 + (ωb ^ 2 ^ i) ^ 2 + 2 * (ωb ^ 2 ^ i * ω ^ 2 ^ i) - 2 := by ring
      _ = (ω ^ 2 ^ i) ^ 2 + (ωb ^ 2 ^ i) ^ 2 := by
        rw [← mul_pow ωb ω, ωb_mul_ω, one_pow, mul_one, add_sub_cancel_right]
      _ = ω ^ 2 ^ (i + 1) + ωb ^ 2 ^ (i + 1) := by rw [← pow_mul, ← pow_mul, _root_.pow_succ]


/-- If `1 < p`, then `q p`, the smallest prime factor of `mersenne p`, is more than 2. -/
theorem two_lt_q (p' : ℕ) : 2 < q (p' + 2) := by
  /-
    p' : Nat
    ⊢ LT.lt 2 (LucasLehmer.q (HAdd.hAdd p' 2))
  -/
  refine (minFac_prime (one_lt_mersenne.2 ?_).ne').two_le.lt_of_ne' ?_
    /-
      case refine_1
      p' : Nat
      ⊢ LT.lt 1 (HAdd.hAdd p' 2)
    -/
  · exact le_add_left _ _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      p' : Nat
      ⊢ Ne (mersenne (HAdd.hAdd p' 2)).minFac 2
    -/
  · rw [Ne, minFac_eq_two_iff, mersenne, Nat.pow_succ']
    /-
      case refine_2
      p' : Nat
      ⊢ Not (Dvd.dvd 2 (HSub.hSub (HMul.hMul 2 (HPow.hPow 2 (HAdd.hAdd p' 1))) 1))
    -/
    exact Nat.two_not_dvd_two_mul_sub_one Nat.one_le_two_pow
    /-
      🎉 no goals
    -/


theorem ω_pow_formula (p' : ℕ) (h : lucasLehmerResidue (p' + 2) = 0) :
    ∃ k : ℤ,
      (ω : X (q (p' + 2))) ^ 2 ^ (p' + 1) =
        k * mersenne (p' + 2) * (ω : X (q (p' + 2))) ^ 2 ^ p' - 1 := by
  /-
    p' : Nat
    h : Eq (LucasLehmer.lucasLehmerResidue (HAdd.hAdd p' 2)) 0
    ⊢ Exists fun k => Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1)) …
  -/
  dsimp [lucasLehmerResidue] at h
  /-
    p' : Nat
    h : Eq (LucasLehmer.sZMod (HAdd.hAdd p' 2) (HSub.hSub (HAdd.hAdd p' 2) 2)) 0
    ⊢ Exists fun k => Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1)) …
  -/
  rw [sZMod_eq_s p'] at h
  simp? [ZMod.intCast_zmod_eq_zero_iff_dvd] at h says
    simp only [add_tsub_cancel_right, ZMod.intCast_zmod_eq_zero_iff_dvd, ofNat_pos,
      pow_pos, cast_pred, cast_pow, cast_ofNat] at h
  /-
    p' : Nat
    h : Dvd.dvd (HSub.hSub (HPow.hPow 2 (HAdd.hAdd p' 2)) 1) (LucasLehmer.s p')
    ⊢ Exists fun k => Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1)) …
  -/
  cases' h with k h
  /-
    case intro
    p' : Nat
    k : Int
    h : Eq (LucasLehmer.s p') (HMul.hMul (HSub.hSub (HPow.hPow 2 (HAdd.hAdd p' 2)) …
    ⊢ Exists fun k => Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1)) …
  -/
  use k
  /-
    case h
    p' : Nat
    k : Int
    h : Eq (LucasLehmer.s p') (HMul.hMul (HSub.hSub (HPow.hPow 2 (HAdd.hAdd p' 2)) …
    ⊢ Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) (HSub.hSub (HM …
  -/
  replace h := congr_arg (fun n : ℤ => (n : X (q (p' + 2)))) h
  -- coercion from ℤ to X q
  /-
    case h
    p' : Nat
    k : Int
    h : Eq ((fun n => ↑n) (LucasLehmer.s p')) ((fun n => ↑n) (HMul.hMul (HSub.hSub …
    ⊢ Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) (HSub.hSub (HM …
  -/
  dsimp at h
  /-
    case h
    p' : Nat
    k : Int
    h : Eq ↑(LucasLehmer.s p') ↑(HMul.hMul (HSub.hSub (HPow.hPow 2 (HAdd.hAdd p' 2 …
    ⊢ Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) (HSub.hSub (HM …
  -/
  rw [closed_form] at h
  /-
    case h
    p' : Nat
    k : Int
    h : Eq (HAdd.hAdd (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 p')) (HPow.hPow Luca …
    ⊢ Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) (HSub.hSub (HM …
  -/
  replace h := congr_arg (fun x => ω ^ 2 ^ p' * x) h
  /-
    case h
    p' : Nat
    k : Int
    h : Eq ((fun x => HMul.hMul (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 p')) x) (H …
    ⊢ Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) (HSub.hSub (HM …
  -/
  dsimp at h
  /-
    case h
    p' : Nat
    k : Int
    h : Eq (HMul.hMul (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 p')) (HAdd.hAdd (HPo …
    ⊢ Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) (HSub.hSub (HM …
  -/
  have t : 2 ^ p' + 2 ^ p' = 2 ^ (p' + 1) := by ring
  /-
    case h
    p' : Nat
    k : Int
    h : Eq (HMul.hMul (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 p')) (HAdd.hAdd (HPo …
    t : Eq (HAdd.hAdd (HPow.hPow 2 p') (HPow.hPow 2 p')) (HPow.hPow 2 (HAdd.hAdd p …
    ⊢ Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) (HSub.hSub (HM …
  -/
  rw [mul_add, ← pow_add ω, t, ← mul_pow ω ωb (2 ^ p'), ω_mul_ωb, one_pow] at h
  /-
    case h
    p' : Nat
    k : Int
    h : Eq (HAdd.hAdd (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) 1 …
    t : Eq (HAdd.hAdd (HPow.hPow 2 p') (HPow.hPow 2 p')) (HPow.hPow 2 (HAdd.hAdd p …
    ⊢ Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) (HSub.hSub (HM …
  -/
  rw [mul_comm, coe_mul] at h
  /-
    case h
    p' : Nat
    k : Int
    h : Eq (HAdd.hAdd (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) 1 …
    t : Eq (HAdd.hAdd (HPow.hPow 2 p') (HPow.hPow 2 p')) (HPow.hPow 2 (HAdd.hAdd p …
    ⊢ Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) (HSub.hSub (HM …
  -/
  rw [mul_comm _ (k : X (q (p' + 2)))] at h
  /-
    case h
    p' : Nat
    k : Int
    h : Eq (HAdd.hAdd (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) 1 …
    t : Eq (HAdd.hAdd (HPow.hPow 2 p') (HPow.hPow 2 p')) (HPow.hPow 2 (HAdd.hAdd p …
    ⊢ Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) (HSub.hSub (HM …
  -/
  replace h := eq_sub_of_add_eq h
  /-
    case h
    p' : Nat
    k : Int
    t : Eq (HAdd.hAdd (HPow.hPow 2 p') (HPow.hPow 2 p')) (HPow.hPow 2 (HAdd.hAdd p …
    h : Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) (HSub.hSub ( …
    ⊢ Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) (HSub.hSub (HM …
  -/
  have : 1 ≤ 2 ^ (p' + 2) := Nat.one_le_pow _ _ (by decide)
  /-
    case h
    p' : Nat
    k : Int
    t : Eq (HAdd.hAdd (HPow.hPow 2 p') (HPow.hPow 2 p')) (HPow.hPow 2 (HAdd.hAdd p …
    h : Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) (HSub.hSub ( …
    this : LE.le 1 (HPow.hPow 2 (HAdd.hAdd p' 2))
    ⊢ Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) (HSub.hSub (HM …
  -/
  exact mod_cast h
  /-
    🎉 no goals
  -/


/-- `q` is the minimum factor of `mersenne p`, so `M p = 0` in `X q`. -/
theorem mersenne_coe_X (p : ℕ) : (mersenne p : X (q p)) = 0 := by
  /-
    p : Nat
    ⊢ Eq (↑(mersenne p)) 0
  -/
  ext <;> simp [mersenne, q, ZMod.natCast_zmod_eq_zero_iff_dvd, -pow_pos]
          /-
            🎉 no goals
          -/
  /-
    case h₁
    p : Nat
    ⊢ Dvd.dvd (HSub.hSub (HPow.hPow 2 p) 1).minFac (HSub.hSub (HPow.hPow 2 p) 1)
  -/
  apply Nat.minFac_dvd
  /-
    🎉 no goals
  -/


theorem ω_pow_eq_neg_one (p' : ℕ) (h : lucasLehmerResidue (p' + 2) = 0) :
    (ω : X (q (p' + 2))) ^ 2 ^ (p' + 1) = -1 := by
  /-
    p' : Nat
    h : Eq (LucasLehmer.lucasLehmerResidue (HAdd.hAdd p' 2)) 0
    ⊢ Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) (-1)
  -/
  cases' ω_pow_formula p' h with k w
  /-
    case intro
    p' : Nat
    h : Eq (LucasLehmer.lucasLehmerResidue (HAdd.hAdd p' 2)) 0
    k : Int
    w : Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) (HSub.hSub ( …
    ⊢ Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) (-1)
  -/
  rw [mersenne_coe_X] at w
  /-
    case intro
    p' : Nat
    h : Eq (LucasLehmer.lucasLehmerResidue (HAdd.hAdd p' 2)) 0
    k : Int
    w : Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) (HSub.hSub ( …
    ⊢ Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) (-1)
  -/
  simpa using w
  /-
    🎉 no goals
  -/


theorem ω_pow_eq_one (p' : ℕ) (h : lucasLehmerResidue (p' + 2) = 0) :
    (ω : X (q (p' + 2))) ^ 2 ^ (p' + 2) = 1 :=
  calc
    (ω : X (q (p' + 2))) ^ 2 ^ (p' + 2) = (ω ^ 2 ^ (p' + 1)) ^ 2 := by
      /-
        p' : Nat
        h : Eq (LucasLehmer.lucasLehmerResidue (HAdd.hAdd p' 2)) 0
        ⊢ Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 2))) (HPow.hPow (HP …
      -/
      rw [← pow_mul, ← Nat.pow_succ]
      /-
        🎉 no goals
      -/
                       /-
                         p' : Nat
                         h : Eq (LucasLehmer.lucasLehmerResidue (HAdd.hAdd p' 2)) 0
                         ⊢ Eq (HPow.hPow (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) 2)  …
                       -/
    _ = (-1) ^ 2 := by rw [ω_pow_eq_neg_one p' h]
                       /-
                         🎉 no goals
                       -/
                /-
                  p' : Nat
                  h : Eq (LucasLehmer.lucasLehmerResidue (HAdd.hAdd p' 2)) 0
                  ⊢ Eq (HPow.hPow (-1) 2) 1
                -/
    _ = 1 := by simp
                /-
                  🎉 no goals
                -/


/-- `ω` as an element of the group of units. -/
def ωUnit (p : ℕ) : Units (X (q p)) where
  val := ω
  inv := ωb
  val_inv := ω_mul_ωb _
  inv_val := ωb_mul_ω _


@[simp]
theorem ωUnit_coe (p : ℕ) : (ωUnit p : X (q p)) = ω :=
  rfl


/-- The order of `ω` in the unit group is exactly `2^p`. -/
theorem order_ω (p' : ℕ) (h : lucasLehmerResidue (p' + 2) = 0) :
    orderOf (ωUnit (p' + 2)) = 2 ^ (p' + 2) := by
  /-
    p' : Nat
    h : Eq (LucasLehmer.lucasLehmerResidue (HAdd.hAdd p' 2)) 0
    ⊢ Eq (orderOf (LucasLehmer.ωUnit (HAdd.hAdd p' 2))) (HPow.hPow 2 (HAdd.hAdd p' …
  -/
  apply Nat.eq_prime_pow_of_dvd_least_prime_pow
  -- the order of ω divides 2^p
    /-
      case pp
      p' : Nat
      h : Eq (LucasLehmer.lucasLehmerResidue (HAdd.hAdd p' 2)) 0
      ⊢ Nat.Prime 2
    -/
  · exact Nat.prime_two
    /-
      🎉 no goals
    -/
    /-
      case h₁
      p' : Nat
      h : Eq (LucasLehmer.lucasLehmerResidue (HAdd.hAdd p' 2)) 0
      ⊢ Not (Dvd.dvd (orderOf (LucasLehmer.ωUnit (HAdd.hAdd p' 2))) (HPow.hPow 2 (HA …
    -/
  · intro o
    /-
      case h₁
      p' : Nat
      h : Eq (LucasLehmer.lucasLehmerResidue (HAdd.hAdd p' 2)) 0
      o : Dvd.dvd (orderOf (LucasLehmer.ωUnit (HAdd.hAdd p' 2))) (HPow.hPow 2 (HAdd. …
      ⊢ False
    -/
    have ω_pow := orderOf_dvd_iff_pow_eq_one.1 o
    replace ω_pow :=
      congr_arg (Units.coeHom (X (q (p' + 2))) : Units (X (q (p' + 2))) → X (q (p' + 2))) ω_pow
    simp? at ω_pow says
      simp only [Units.coeHom_apply, Units.val_pow_eq_pow_val, ωUnit_coe, Units.val_one] at ω_pow
    have h : (1 : ZMod (q (p' + 2))) = -1 :=
      congr_arg Prod.fst (ω_pow.symm.trans (ω_pow_eq_neg_one p' h))
    /-
      case h₁
      p' : Nat
      h✝ : Eq (LucasLehmer.lucasLehmerResidue (HAdd.hAdd p' 2)) 0
      o : Dvd.dvd (orderOf (LucasLehmer.ωUnit (HAdd.hAdd p' 2))) (HPow.hPow 2 (HAdd. …
      ω_pow : Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) 1
      h : Eq 1 (-1)
      ⊢ False
    -/
    haveI : Fact (2 < (q (p' + 2) : ℕ)) := ⟨two_lt_q _⟩
    /-
      case h₁
      p' : Nat
      h✝ : Eq (LucasLehmer.lucasLehmerResidue (HAdd.hAdd p' 2)) 0
      o : Dvd.dvd (orderOf (LucasLehmer.ωUnit (HAdd.hAdd p' 2))) (HPow.hPow 2 (HAdd. …
      ω_pow : Eq (HPow.hPow LucasLehmer.X.ω (HPow.hPow 2 (HAdd.hAdd p' 1))) 1
      h : Eq 1 (-1)
      this : Fact (LT.lt 2 ↑(LucasLehmer.q (HAdd.hAdd p' 2)))
      ⊢ False
    -/
    apply ZMod.neg_one_ne_one h.symm
    /-
      🎉 no goals
    -/
    /-
      case h₂
      p' : Nat
      h : Eq (LucasLehmer.lucasLehmerResidue (HAdd.hAdd p' 2)) 0
      ⊢ Dvd.dvd (orderOf (LucasLehmer.ωUnit (HAdd.hAdd p' 2))) (HPow.hPow 2 (HAdd.hA …
    -/
  · apply orderOf_dvd_iff_pow_eq_one.2
    /-
      case h₂
      p' : Nat
      h : Eq (LucasLehmer.lucasLehmerResidue (HAdd.hAdd p' 2)) 0
      ⊢ Eq (HPow.hPow (LucasLehmer.ωUnit (HAdd.hAdd p' 2)) (HPow.hPow 2 (HAdd.hAdd ( …
    -/
    apply Units.ext
    /-
      case h₂.a
      p' : Nat
      h : Eq (LucasLehmer.lucasLehmerResidue (HAdd.hAdd p' 2)) 0
      ⊢ Eq ↑(HPow.hPow (LucasLehmer.ωUnit (HAdd.hAdd p' 2)) (HPow.hPow 2 (HAdd.hAdd  …
    -/
    push_cast
    /-
      case h₂.a
      p' : Nat
      h : Eq (LucasLehmer.lucasLehmerResidue (HAdd.hAdd p' 2)) 0
      ⊢ Eq (HPow.hPow (↑(LucasLehmer.ωUnit (HAdd.hAdd p' 2))) (HPow.hPow 2 (HAdd.hAd …
    -/
    exact ω_pow_eq_one p' h
    /-
      🎉 no goals
    -/


theorem order_ineq (p' : ℕ) (h : lucasLehmerResidue (p' + 2) = 0) :
    2 ^ (p' + 2) < (q (p' + 2) : ℕ) ^ 2 :=
  calc
    2 ^ (p' + 2) = orderOf (ωUnit (p' + 2)) := (order_ω p' h).symm
    _ ≤ Fintype.card (X (q (p' + 2)))ˣ := orderOf_le_card_univ
    _ < (q (p' + 2) : ℕ) ^ 2 := card_units_lt (Nat.lt_of_succ_lt (two_lt_q _))


theorem lucas_lehmer_sufficiency (p : ℕ) (w : 1 < p) : LucasLehmerTest p → (mersenne p).Prime := by
  /-
    p : Nat
    w : LT.lt 1 p
    ⊢ LucasLehmer.LucasLehmerTest p → Nat.Prime (mersenne p)
  -/
  let p' := p - 2
  /-
    p : Nat
    w : LT.lt 1 p
    p' : Nat := HSub.hSub p 2
    ⊢ LucasLehmer.LucasLehmerTest p → Nat.Prime (mersenne p)
  -/
  have z : p = p' + 2 := (tsub_eq_iff_eq_add_of_le w.nat_succ_le).mp rfl
  /-
    p : Nat
    w : LT.lt 1 p
    p' : Nat := HSub.hSub p 2
    z : Eq p (HAdd.hAdd p' 2)
    ⊢ LucasLehmer.LucasLehmerTest p → Nat.Prime (mersenne p)
  -/
  have w : 1 < p' + 2 := Nat.lt_of_sub_eq_succ rfl
  /-
    p : Nat
    w✝ : LT.lt 1 p
    p' : Nat := HSub.hSub p 2
    z : Eq p (HAdd.hAdd p' 2)
    w : LT.lt 1 (HAdd.hAdd p' 2)
    ⊢ LucasLehmer.LucasLehmerTest p → Nat.Prime (mersenne p)
  -/
  contrapose
  /-
    p : Nat
    w✝ : LT.lt 1 p
    p' : Nat := HSub.hSub p 2
    z : Eq p (HAdd.hAdd p' 2)
    w : LT.lt 1 (HAdd.hAdd p' 2)
    ⊢ Not (Nat.Prime (mersenne p)) → Not (LucasLehmer.LucasLehmerTest p)
  -/
  intro a t
  /-
    p : Nat
    w✝ : LT.lt 1 p
    p' : Nat := HSub.hSub p 2
    z : Eq p (HAdd.hAdd p' 2)
    w : LT.lt 1 (HAdd.hAdd p' 2)
    a : Not (Nat.Prime (mersenne p))
    t : LucasLehmer.LucasLehmerTest p
    ⊢ False
  -/
  rw [z] at a
  /-
    p : Nat
    w✝ : LT.lt 1 p
    p' : Nat := HSub.hSub p 2
    z : Eq p (HAdd.hAdd p' 2)
    w : LT.lt 1 (HAdd.hAdd p' 2)
    a : Not (Nat.Prime (mersenne (HAdd.hAdd p' 2)))
    t : LucasLehmer.LucasLehmerTest p
    ⊢ False
  -/
  rw [z] at t
  /-
    p : Nat
    w✝ : LT.lt 1 p
    p' : Nat := HSub.hSub p 2
    z : Eq p (HAdd.hAdd p' 2)
    w : LT.lt 1 (HAdd.hAdd p' 2)
    a : Not (Nat.Prime (mersenne (HAdd.hAdd p' 2)))
    t : LucasLehmer.LucasLehmerTest (HAdd.hAdd p' 2)
    ⊢ False
  -/
  have h₁ := order_ineq p' t
  /-
    p : Nat
    w✝ : LT.lt 1 p
    p' : Nat := HSub.hSub p 2
    z : Eq p (HAdd.hAdd p' 2)
    w : LT.lt 1 (HAdd.hAdd p' 2)
    a : Not (Nat.Prime (mersenne (HAdd.hAdd p' 2)))
    t : LucasLehmer.LucasLehmerTest (HAdd.hAdd p' 2)
    h₁ : LT.lt (HPow.hPow 2 (HAdd.hAdd p' 2)) (HPow.hPow (↑(LucasLehmer.q (HAdd.hA …
    ⊢ False
  -/
  have h₂ := Nat.minFac_sq_le_self (mersenne_pos.2 (Nat.lt_of_succ_lt w)) a
  /-
    p : Nat
    w✝ : LT.lt 1 p
    p' : Nat := HSub.hSub p 2
    z : Eq p (HAdd.hAdd p' 2)
    w : LT.lt 1 (HAdd.hAdd p' 2)
    a : Not (Nat.Prime (mersenne (HAdd.hAdd p' 2)))
    t : LucasLehmer.LucasLehmerTest (HAdd.hAdd p' 2)
    h₁ : LT.lt (HPow.hPow 2 (HAdd.hAdd p' 2)) (HPow.hPow (↑(LucasLehmer.q (HAdd.hA …
    h₂ : LE.le (HPow.hPow (mersenne (HAdd.hAdd p' 2)).minFac 2) (mersenne (HAdd.hA …
    ⊢ False
  -/
  have h := lt_of_lt_of_le h₁ h₂
  /-
    p : Nat
    w✝ : LT.lt 1 p
    p' : Nat := HSub.hSub p 2
    z : Eq p (HAdd.hAdd p' 2)
    w : LT.lt 1 (HAdd.hAdd p' 2)
    a : Not (Nat.Prime (mersenne (HAdd.hAdd p' 2)))
    t : LucasLehmer.LucasLehmerTest (HAdd.hAdd p' 2)
    h₁ : LT.lt (HPow.hPow 2 (HAdd.hAdd p' 2)) (HPow.hPow (↑(LucasLehmer.q (HAdd.hA …
    h₂ : LE.le (HPow.hPow (mersenne (HAdd.hAdd p' 2)).minFac 2) (mersenne (HAdd.hA …
    h : LT.lt (HPow.hPow 2 (HAdd.hAdd p' 2)) (mersenne (HAdd.hAdd p' 2))
    ⊢ False
  -/
  exact not_lt_of_ge (Nat.sub_le _ _) h
  /-
    🎉 no goals
  -/


/-- Version of `sMod` that is `ℕ`-valued. One should have `q = 2 ^ p - 1`.
This can be reduced by the kernel. -/
def sModNat (q : ℕ) : ℕ → ℕ
  | 0 => 4 % q
  | i + 1 => (sModNat q i ^ 2 + (q - 2)) % q


theorem sModNat_eq_sMod (p k : ℕ) (hp : 2 ≤ p) : (sModNat (2 ^ p - 1) k : ℤ) = sMod p k := by
  have h1 := calc
    4 = 2 ^ 2 := by norm_num
    _ ≤ 2 ^ p := Nat.pow_le_pow_of_le_right (by norm_num) hp
  /-
    p k : Nat
    hp : LE.le 2 p
    h1 : LE.le 4 (HPow.hPow 2 p)
    ⊢ Eq (↑(LucasLehmer.norm_num_ext.sModNat (HSub.hSub (HPow.hPow 2 p) 1) k)) (Lu …
  -/
  have h2 : 1 ≤ 2 ^ p := by omega
  induction k with
  | zero =>
    rw [sModNat, sMod, Int.ofNat_emod]
    simp [h2]
  | succ k ih =>
    rw [sModNat, sMod, ← ih]
    have h3 : 2 ≤ 2 ^ p - 1 := by
      zify [h2]
      calc
        (2 : Int) ≤ 4 - 1 := by norm_num
        _         ≤ 2 ^ p - 1 := by zify at h1; exact Int.sub_le_sub_right h1 _
    zify [h2, h3]
    rw [← add_sub_assoc, sub_eq_add_neg, add_assoc, add_comm _ (-2), ← add_assoc,
      Int.add_emod_self, ← sub_eq_add_neg]


/-- Tail-recursive version of `sModNat`. -/
def sModNatTR (q : ℕ) (k : Nat) : ℕ :=
  go k (4 % q)
where
  /-- Helper function for `sMod''`. -/
  go : ℕ → ℕ → ℕ
  | 0, acc => acc
  | n + 1, acc => go n ((acc ^ 2 + (q - 2)) % q)


/--
Generalization of `sModNat` with arbitrary base case,
useful for proving `sModNatTR` and `sModNat` agree.
-/
def sModNat_aux (b : ℕ) (q : ℕ) : ℕ → ℕ
  | 0 => b
  | i + 1 => (sModNat_aux b q i ^ 2 + (q - 2)) % q


theorem sModNat_aux_eq (q k : ℕ) : sModNat_aux (4 % q) q k = sModNat q k := by
  induction k with
  | zero => rfl
  | succ k ih => rw [sModNat_aux, ih, sModNat, ← ih]


theorem sModNatTR_eq_sModNat (q : ℕ) (i : ℕ) : sModNatTR q i = sModNat q i := by
  /-
    q i : Nat
    ⊢ Eq (LucasLehmer.norm_num_ext.sModNatTR q i) (LucasLehmer.norm_num_ext.sModNa …
  -/
  rw [sModNatTR, helper, sModNat_aux_eq]
  /-
    🎉 no goals
  -/
where
  helper b q k : sModNatTR.go q k b = sModNat_aux b q k := by
    induction k generalizing b with
    | zero => rfl
    | succ k ih =>
      rw [sModNatTR.go, ih, sModNat_aux]
      clear ih
      induction k with
      | zero => rfl
      | succ k ih =>
        rw [sModNat_aux, ih, sModNat_aux]


lemma testTrueHelper (p : ℕ) (hp : Nat.blt 1 p = true) (h : sModNatTR (2 ^ p - 1) (p - 2) = 0) :
    LucasLehmerTest p := by
  /-
    p : Nat
    hp : Eq (Nat.blt 1 p) Bool.true
    h : Eq (LucasLehmer.norm_num_ext.sModNatTR (HSub.hSub (HPow.hPow 2 p) 1) (HSub …
    ⊢ LucasLehmer.LucasLehmerTest p
  -/
  rw [Nat.blt_eq] at hp
  rw [LucasLehmerTest, LucasLehmer.residue_eq_zero_iff_sMod_eq_zero p hp, ← sModNat_eq_sMod p _ hp,
    ← sModNatTR_eq_sModNat, h]
  /-
    p : Nat
    hp : LT.lt 1 p
    h : Eq (LucasLehmer.norm_num_ext.sModNatTR (HSub.hSub (HPow.hPow 2 p) 1) (HSub …
    ⊢ Eq (↑0) 0
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma testFalseHelper (p : ℕ) (hp : Nat.blt 1 p = true)
    (h : Nat.ble 1 (sModNatTR (2 ^ p - 1) (p - 2))) : ¬ LucasLehmerTest p := by
  /-
    p : Nat
    hp : Eq (Nat.blt 1 p) Bool.true
    h : Eq (Nat.ble 1 (LucasLehmer.norm_num_ext.sModNatTR (HSub.hSub (HPow.hPow 2  …
    ⊢ Not (LucasLehmer.LucasLehmerTest p)
  -/
  rw [Nat.blt_eq] at hp
  /-
    p : Nat
    hp : LT.lt 1 p
    h : Eq (Nat.ble 1 (LucasLehmer.norm_num_ext.sModNatTR (HSub.hSub (HPow.hPow 2  …
    ⊢ Not (LucasLehmer.LucasLehmerTest p)
  -/
  rw [Nat.ble_eq, Nat.succ_le, Nat.pos_iff_ne_zero] at h
  rw [LucasLehmerTest, LucasLehmer.residue_eq_zero_iff_sMod_eq_zero p hp, ← sModNat_eq_sMod p _ hp,
    ← sModNatTR_eq_sModNat]
  /-
    p : Nat
    hp : LT.lt 1 p
    h : Ne (LucasLehmer.norm_num_ext.sModNatTR (HSub.hSub (HPow.hPow 2 p) 1) (HSub …
    ⊢ Not (Eq (↑(LucasLehmer.norm_num_ext.sModNatTR (HSub.hSub (HPow.hPow 2 p) 1)  …
  -/
  simpa using h
  /-
    🎉 no goals
  -/


theorem isNat_lucasLehmerTest : {p np : ℕ} →
    IsNat p np → LucasLehmerTest np → LucasLehmerTest p
  | _, _, ⟨rfl⟩, h => h


theorem isNat_not_lucasLehmerTest : {p np : ℕ} →
    IsNat p np → ¬ LucasLehmerTest np → ¬ LucasLehmerTest p
  | _, _, ⟨rfl⟩, h => h


/-- Calculate `LucasLehmer.LucasLehmerTest p` for `2 ≤ p` by using kernel reduction for the
`sMod'` function. -/
@[norm_num LucasLehmer.LucasLehmerTest (_ : ℕ)]
def evalLucasLehmerTest : NormNumExt where eval {_ _} e := do
  let .app _ (p : Q(ℕ)) ← Meta.whnfR e | failure
  let ⟨ep, hp⟩ ← deriveNat p _
  let np := ep.natLit!
  unless 1 < np do
    failure
  haveI' h1ltp : Nat.blt 1 $ep =Q true := ⟨⟩
  if sModNatTR (2 ^ np - 1) (np - 2) = 0 then
    haveI' hs : sModNatTR (2 ^ $ep - 1) ($ep - 2) =Q 0 := ⟨⟩
    have pf : Q(LucasLehmerTest $ep) := q(testTrueHelper $ep $h1ltp $hs)
    have pf' : Q(LucasLehmerTest $p) := q(isNat_lucasLehmerTest $hp $pf)
    return .isTrue pf'
  else
    haveI' hs : Nat.ble 1 (sModNatTR (2 ^ $ep - 1) ($ep - 2)) =Q true := ⟨⟩
    have pf : Q(¬ LucasLehmerTest $ep) := q(testFalseHelper $ep $h1ltp $hs)
    have pf' : Q(¬ LucasLehmerTest $p) := q(isNat_not_lucasLehmerTest $hp $pf)
    return .isFalse pf'


theorem modEq_mersenne (n k : ℕ) : k ≡ k / 2 ^ n + k % 2 ^ n [MOD 2 ^ n - 1] :=
  -- See https://leanprover.zulipchat.com/#narrow/stream/113489-new-members/topic/help.20finding.20a.20lemma/near/177698446
  calc
    k = 2 ^ n * (k / 2 ^ n) + k % 2 ^ n := (Nat.div_add_mod k (2 ^ n)).symm
    _ ≡ 1 * (k / 2 ^ n) + k % 2 ^ n [MOD 2 ^ n - 1] :=
      ((Nat.modEq_sub <| Nat.succ_le_of_lt <| pow_pos zero_lt_two _).mul_right _).add_right _
                                    /-
                                      n k : Nat
                                      ⊢ Eq (HAdd.hAdd (HMul.hMul 1 (HDiv.hDiv k (HPow.hPow 2 n))) (HMod.hMod k (HPow …
                                    -/
    _ = k / 2 ^ n + k % 2 ^ n := by rw [one_mul]
                                    /-
                                      🎉 no goals
                                    -/

-- It's hard to know what the limiting factor for large Mersenne primes would be.
-- In the purely computational world, I think it's the squaring operation in `s`.


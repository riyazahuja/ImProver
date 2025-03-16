/-- The ring of integers adjoined with a square root of `d`.
  These have the form `a + b √d` where `a b : ℤ`. The components
  are called `re` and `im` by analogy to the negative `d` case. -/
@[ext]
structure Zsqrtd (d : ℤ) where
  re : ℤ
  im : ℤ
  deriving DecidableEq


@[inherit_doc] prefix:100 "ℤ√" => Zsqrtd


/-- Convert an integer to a `ℤ√d` -/
def ofInt (n : ℤ) : ℤ√d :=
  ⟨n, 0⟩


theorem ofInt_re (n : ℤ) : (ofInt n : ℤ√d).re = n :=
  rfl


theorem ofInt_im (n : ℤ) : (ofInt n : ℤ√d).im = 0 :=
  rfl


/-- The zero of the ring -/
instance : Zero (ℤ√d) :=
  ⟨ofInt 0⟩


@[simp]
theorem zero_re : (0 : ℤ√d).re = 0 :=
  rfl


@[simp]
theorem zero_im : (0 : ℤ√d).im = 0 :=
  rfl


instance : Inhabited (ℤ√d) :=
  ⟨0⟩


/-- The one of the ring -/
instance : One (ℤ√d) :=
  ⟨ofInt 1⟩


@[simp]
theorem one_re : (1 : ℤ√d).re = 1 :=
  rfl


@[simp]
theorem one_im : (1 : ℤ√d).im = 0 :=
  rfl


/-- The representative of `√d` in the ring -/
def sqrtd : ℤ√d :=
  ⟨0, 1⟩


@[simp]
theorem sqrtd_re : (sqrtd : ℤ√d).re = 0 :=
  rfl


@[simp]
theorem sqrtd_im : (sqrtd : ℤ√d).im = 1 :=
  rfl


/-- Addition of elements of `ℤ√d` -/
instance : Add (ℤ√d) :=
  ⟨fun z w => ⟨z.1 + w.1, z.2 + w.2⟩⟩


@[simp]
theorem add_def (x y x' y' : ℤ) : (⟨x, y⟩ + ⟨x', y'⟩ : ℤ√d) = ⟨x + x', y + y'⟩ :=
  rfl


@[simp]
theorem add_re (z w : ℤ√d) : (z + w).re = z.re + w.re :=
  rfl


@[simp]
theorem add_im (z w : ℤ√d) : (z + w).im = z.im + w.im :=
  rfl


/-- Negation in `ℤ√d` -/
instance : Neg (ℤ√d) :=
  ⟨fun z => ⟨-z.1, -z.2⟩⟩


@[simp]
theorem neg_re (z : ℤ√d) : (-z).re = -z.re :=
  rfl


@[simp]
theorem neg_im (z : ℤ√d) : (-z).im = -z.im :=
  rfl


/-- Multiplication in `ℤ√d` -/
instance : Mul (ℤ√d) :=
  ⟨fun z w => ⟨z.1 * w.1 + d * z.2 * w.2, z.1 * w.2 + z.2 * w.1⟩⟩


@[simp]
theorem mul_re (z w : ℤ√d) : (z * w).re = z.re * w.re + d * z.im * w.im :=
  rfl


@[simp]
theorem mul_im (z w : ℤ√d) : (z * w).im = z.re * w.im + z.im * w.re :=
  rfl


instance addCommGroup : AddCommGroup (ℤ√d) := by
  refine
  { add := (· + ·)
    zero := (0 : ℤ√d)
    sub := fun a b => a + -b
    neg := Neg.neg
    nsmul := @nsmulRec (ℤ√d) ⟨0⟩ ⟨(· + ·)⟩
    zsmul := @zsmulRec (ℤ√d) ⟨0⟩ ⟨(· + ·)⟩ ⟨Neg.neg⟩ (@nsmulRec (ℤ√d) ⟨0⟩ ⟨(· + ·)⟩)
    add_assoc := ?_
    zero_add := ?_
    add_zero := ?_
    neg_add_cancel := ?_
    add_comm := ?_ } <;>
  /-
    case refine_1
    d : Int
    ⊢ ∀ (a b c : Zsqrtd d), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (HAdd.hA …
  -/
  intros <;>
  /-
    case refine_1
    d : Int
    a✝ b✝ c✝ : Zsqrtd d
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd a✝ b✝) c✝) (HAdd.hAdd a✝ (HAdd.hAdd b✝ c✝))
  -/
  ext <;>
  /-
    case refine_1.re
    d : Int
    a✝ b✝ c✝ : Zsqrtd d
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd a✝ b✝) c✝).re (HAdd.hAdd a✝ (HAdd.hAdd b✝ c✝)).re
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
  /-
    🎉 no goals
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
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  simp [add_comm, add_left_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem sub_re (z w : ℤ√d) : (z - w).re = z.re - w.re :=
  rfl


@[simp]
theorem sub_im (z w : ℤ√d) : (z - w).im = z.im - w.im :=
  rfl


instance addGroupWithOne : AddGroupWithOne (ℤ√d) :=
  { Zsqrtd.addCommGroup with
    natCast := fun n => ofInt n
    intCast := ofInt
    one := 1 }


instance commRing : CommRing (ℤ√d) := by
  refine
  { Zsqrtd.addGroupWithOne with
    mul := (· * ·)
    npow := @npowRec (ℤ√d) ⟨1⟩ ⟨(· * ·)⟩,
    add_comm := ?_
    left_distrib := ?_
    right_distrib := ?_
    zero_mul := ?_
    mul_zero := ?_
    mul_assoc := ?_
    one_mul := ?_
    mul_one := ?_
    mul_comm := ?_ } <;>
  /-
    case refine_1
    d : Int
    ⊢ ∀ (a b : Zsqrtd d), Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
  -/
  intros <;>
  /-
    case refine_1
    d : Int
    a✝ b✝ : Zsqrtd d
    ⊢ Eq (HAdd.hAdd a✝ b✝) (HAdd.hAdd b✝ a✝)
  -/
  ext <;>
  /-
    case refine_1.re
    d : Int
    a✝ b✝ : Zsqrtd d
    ⊢ Eq (HAdd.hAdd a✝ b✝).re (HAdd.hAdd b✝ a✝).re
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
  /-
    🎉 no goals
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
  /-
    🎉 no goals
  -/
  simp <;>
  /-
    case refine_1.re
    d : Int
    a✝ b✝ : Zsqrtd d
    ⊢ Eq (HAdd.hAdd a✝.re b✝.re) (HAdd.hAdd b✝.re a✝.re)
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
  /-
    🎉 no goals
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
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  ring
  /-
    🎉 no goals
  -/


                                 /-
                                   d : Int
                                   ⊢ AddMonoid (Zsqrtd d)
                                 -/
instance : AddMonoid (ℤ√d) := by infer_instance
                                 /-
                                   🎉 no goals
                                 -/


                              /-
                                d : Int
                                ⊢ Monoid (Zsqrtd d)
                              -/
instance : Monoid (ℤ√d) := by infer_instance
                              /-
                                🎉 no goals
                              -/


                                  /-
                                    d : Int
                                    ⊢ CommMonoid (Zsqrtd d)
                                  -/
instance : CommMonoid (ℤ√d) := by infer_instance
                                  /-
                                    🎉 no goals
                                  -/


                                     /-
                                       d : Int
                                       ⊢ CommSemigroup (Zsqrtd d)
                                     -/
instance : CommSemigroup (ℤ√d) := by infer_instance
                                     /-
                                       🎉 no goals
                                     -/


                                 /-
                                   d : Int
                                   ⊢ Semigroup (Zsqrtd d)
                                 -/
instance : Semigroup (ℤ√d) := by infer_instance
                                 /-
                                   🎉 no goals
                                 -/


                                        /-
                                          d : Int
                                          ⊢ AddCommSemigroup (Zsqrtd d)
                                        -/
instance : AddCommSemigroup (ℤ√d) := by infer_instance
                                        /-
                                          🎉 no goals
                                        -/


                                    /-
                                      d : Int
                                      ⊢ AddSemigroup (Zsqrtd d)
                                    -/
instance : AddSemigroup (ℤ√d) := by infer_instance
                                    /-
                                      🎉 no goals
                                    -/


                                    /-
                                      d : Int
                                      ⊢ CommSemiring (Zsqrtd d)
                                    -/
instance : CommSemiring (ℤ√d) := by infer_instance
                                    /-
                                      🎉 no goals
                                    -/


                                /-
                                  d : Int
                                  ⊢ Semiring (Zsqrtd d)
                                -/
instance : Semiring (ℤ√d) := by infer_instance
                                /-
                                  🎉 no goals
                                -/


                            /-
                              d : Int
                              ⊢ Ring (Zsqrtd d)
                            -/
instance : Ring (ℤ√d) := by infer_instance
                            /-
                              🎉 no goals
                            -/


                               /-
                                 d : Int
                                 ⊢ Distrib (Zsqrtd d)
                               -/
instance : Distrib (ℤ√d) := by infer_instance
                               /-
                                 🎉 no goals
                               -/


/-- Conjugation in `ℤ√d`. The conjugate of `a + b √d` is `a - b √d`. -/
instance : Star (ℤ√d) where
  star z := ⟨z.1, -z.2⟩


@[simp]
theorem star_mk (x y : ℤ) : star (⟨x, y⟩ : ℤ√d) = ⟨x, -y⟩ :=
  rfl


@[simp]
theorem star_re (z : ℤ√d) : (star z).re = z.re :=
  rfl


@[simp]
theorem star_im (z : ℤ√d) : (star z).im = -z.im :=
  rfl


instance : StarRing (ℤ√d) where
  star_involutive _ := Zsqrtd.ext rfl (neg_neg _)
                     /-
                       d : Int
                       a b : Zsqrtd d
                       ⊢ Eq (Star.star (HMul.hMul a b)) (HMul.hMul (Star.star b) (Star.star a))
                     -/
                                      /-
                                        🎉 no goals
                                      -/
  star_mul a b := by ext <;> simp <;> ring
                                      /-
                                        🎉 no goals
                                      -/
  star_add _ _ := Zsqrtd.ext rfl (neg_add _ _)

-- Porting note: proof was `by decide`

instance nontrivial : Nontrivial (ℤ√d) :=
                                     /-
                                       d : Int
                                       ⊢ Not (And (Eq (Zsqrtd.re 0) (Zsqrtd.re 1)) (Eq (Zsqrtd.im 0) (Zsqrtd.im 1)))
                                     -/
  ⟨⟨0, 1, Zsqrtd.ext_iff.not.mpr (by simp)⟩⟩
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem natCast_re (n : ℕ) : (n : ℤ√d).re = n :=
  rfl


@[simp]
theorem ofNat_re (n : ℕ) [n.AtLeastTwo] : (no_index (OfNat.ofNat n) : ℤ√d).re = n :=
  rfl


@[simp]
theorem natCast_im (n : ℕ) : (n : ℤ√d).im = 0 :=
  rfl


@[simp]
theorem ofNat_im (n : ℕ) [n.AtLeastTwo] : (no_index (OfNat.ofNat n) : ℤ√d).im = 0 :=
  rfl


theorem natCast_val (n : ℕ) : (n : ℤ√d) = ⟨n, 0⟩ :=
  rfl


@[simp]
                                                    /-
                                                      d n : Int
                                                      ⊢ Eq (↑n).re n
                                                    -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
theorem intCast_re (n : ℤ) : (n : ℤ√d).re = n := by cases n <;> rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp]
                                                    /-
                                                      d n : Int
                                                      ⊢ Eq (↑n).im 0
                                                    -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
theorem intCast_im (n : ℤ) : (n : ℤ√d).im = 0 := by cases n <;> rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/


                                                       /-
                                                         d n : Int
                                                         ⊢ Eq ↑n { re := n, im := 0 }
                                                       -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
theorem intCast_val (n : ℤ) : (n : ℤ√d) = ⟨n, 0⟩ := by ext <;> simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


                                                         /-
                                                           d : Int
                                                           m n : Nat
                                                           ⊢ Eq ↑m ↑n → Eq m n
                                                         -/
instance : CharZero (ℤ√d) where cast_injective m n := by simp [Zsqrtd.ext_iff]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
                                                             /-
                                                               d n : Int
                                                               ⊢ Eq (Zsqrtd.ofInt n) ↑n
                                                             -/
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
theorem ofInt_eq_intCast (n : ℤ) : (ofInt n : ℤ√d) = n := by ext <;> simp [ofInt_re, ofInt_im]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[deprecated (since := "2024-04-05")] alias coe_nat_re := natCast_re

@[deprecated (since := "2024-04-05")] alias coe_nat_im := natCast_im

@[deprecated (since := "2024-04-05")] alias coe_nat_val := natCast_val

@[deprecated (since := "2024-04-05")] alias coe_int_re := intCast_re

@[deprecated (since := "2024-04-05")] alias coe_int_im := intCast_im

@[deprecated (since := "2024-04-05")] alias coe_int_val := intCast_val

@[deprecated (since := "2024-04-05")] alias ofInt_eq_coe := ofInt_eq_intCast


@[simp]
                                                                         /-
                                                                           d n x y : Int
                                                                           ⊢ Eq (HMul.hMul ↑n { re := x, im := y }) { re := HMul.hMul n x, im := HMul.hMu …
                                                                         -/
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
theorem smul_val (n x y : ℤ) : (n : ℤ√d) * ⟨x, y⟩ = ⟨n * x, n * y⟩ := by ext <;> simp
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


                                                                 /-
                                                                   d a : Int
                                                                   b : Zsqrtd d
                                                                   ⊢ Eq (HMul.hMul (↑a) b).re (HMul.hMul a b.re)
                                                                 -/
theorem smul_re (a : ℤ) (b : ℤ√d) : (↑a * b).re = a * b.re := by simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


                                                                 /-
                                                                   d a : Int
                                                                   b : Zsqrtd d
                                                                   ⊢ Eq (HMul.hMul (↑a) b).im (HMul.hMul a b.im)
                                                                 -/
theorem smul_im (a : ℤ) (b : ℤ√d) : (↑a * b).im = a * b.im := by simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
                                                                        /-
                                                                          d x y : Int
                                                                          ⊢ Eq (HMul.hMul Zsqrtd.sqrtd { re := x, im := y }) { re := HMul.hMul d y, im : …
                                                                        -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
theorem muld_val (x y : ℤ) : sqrtd (d := d) * ⟨x, y⟩ = ⟨d * y, x⟩ := by ext <;> simp
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


@[simp]
                                                          /-
                                                            d : Int
                                                            ⊢ Eq (HMul.hMul Zsqrtd.sqrtd Zsqrtd.sqrtd) ↑d
                                                          -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
theorem dmuld : sqrtd (d := d) * sqrtd (d := d) = d := by ext <;> simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
                                                                                      /-
                                                                                        d n x y : Int
                                                                                        ⊢ Eq (HMul.hMul (HMul.hMul Zsqrtd.sqrtd ↑n) { re := x, im := y }) { re := HMul …
                                                                                      -/
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/
theorem smuld_val (n x y : ℤ) : sqrtd * (n : ℤ√d) * ⟨x, y⟩ = ⟨d * n * y, n * x⟩ := by ext <;> simp
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/


                                                                            /-
                                                                              d x y : Int
                                                                              ⊢ Eq { re := x, im := y } (HAdd.hAdd (↑x) (HMul.hMul Zsqrtd.sqrtd ↑y))
                                                                            -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
theorem decompose {x y : ℤ} : (⟨x, y⟩ : ℤ√d) = x + sqrtd (d := d) * y := by ext <;> simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


theorem mul_star {x y : ℤ} : (⟨x, y⟩ * star ⟨x, y⟩ : ℤ√d) = x * x - d * y * y := by
  /-
    d x y : Int
    ⊢ Eq (HMul.hMul { re := x, im := y } (Star.star { re := x, im := y })) (HSub.h …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp [sub_eq_add_neg, mul_comm]
          /-
            🎉 no goals
          -/


@[deprecated (since := "2024-05-25")] alias coe_int_add := Int.cast_add

@[deprecated (since := "2024-05-25")] alias coe_int_sub := Int.cast_sub

@[deprecated (since := "2024-05-25")] alias coe_int_mul := Int.cast_mul

@[deprecated (since := "2024-05-25")] alias coe_int_inj := Int.cast_inj


theorem intCast_dvd (z : ℤ) (a : ℤ√d) : ↑z ∣ a ↔ z ∣ a.re ∧ z ∣ a.im := by
  /-
    d z : Int
    a : Zsqrtd d
    ⊢ Iff (Dvd.dvd (↑z) a) (And (Dvd.dvd z a.re) (Dvd.dvd z a.im))
  -/
  constructor
    /-
      case mp
      d z : Int
      a : Zsqrtd d
      ⊢ Dvd.dvd (↑z) a → And (Dvd.dvd z a.re) (Dvd.dvd z a.im)
    -/
  · rintro ⟨x, rfl⟩
    simp only [add_zero, intCast_re, zero_mul, mul_im, dvd_mul_right, and_self_iff,
      mul_re, mul_zero, intCast_im]
    /-
      case mpr
      d z : Int
      a : Zsqrtd d
      ⊢ And (Dvd.dvd z a.re) (Dvd.dvd z a.im) → Dvd.dvd (↑z) a
    -/
  · rintro ⟨⟨r, hr⟩, ⟨i, hi⟩⟩
    /-
      case mpr.intro.intro.intro
      d z : Int
      a : Zsqrtd d
      r : Int
      hr : Eq a.re (HMul.hMul z r)
      i : Int
      hi : Eq a.im (HMul.hMul z i)
      ⊢ Dvd.dvd (↑z) a
    -/
    use ⟨r, i⟩
    /-
      case h
      d z : Int
      a : Zsqrtd d
      r : Int
      hr : Eq a.re (HMul.hMul z r)
      i : Int
      hi : Eq a.im (HMul.hMul z i)
      ⊢ Eq a (HMul.hMul ↑z { re := r, im := i })
    -/
    rw [smul_val, Zsqrtd.ext_iff]
    /-
      case h
      d z : Int
      a : Zsqrtd d
      r : Int
      hr : Eq a.re (HMul.hMul z r)
      i : Int
      hi : Eq a.im (HMul.hMul z i)
      ⊢ And (Eq a.re { re := HMul.hMul z r, im := HMul.hMul z i }.re) (Eq a.im { re  …
    -/
    exact ⟨hr, hi⟩
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem intCast_dvd_intCast (a b : ℤ) : (a : ℤ√d) ∣ b ↔ a ∣ b := by
  /-
    d a b : Int
    ⊢ Iff (Dvd.dvd ↑a ↑b) (Dvd.dvd a b)
  -/
  rw [intCast_dvd]
  /-
    d a b : Int
    ⊢ Iff (And (Dvd.dvd a (↑b).re) (Dvd.dvd a (↑b).im)) (Dvd.dvd a b)
  -/
  constructor
    /-
      case mp
      d a b : Int
      ⊢ And (Dvd.dvd a (↑b).re) (Dvd.dvd a (↑b).im) → Dvd.dvd a b
    -/
  · rintro ⟨hre, -⟩
    /-
      case mp.intro
      d a b : Int
      hre : Dvd.dvd a (↑b).re
      ⊢ Dvd.dvd a b
    -/
    rwa [intCast_re] at hre
    /-
      🎉 no goals
    -/
    /-
      case mpr
      d a b : Int
      ⊢ Dvd.dvd a b → And (Dvd.dvd a (↑b).re) (Dvd.dvd a (↑b).im)
    -/
  · rw [intCast_re, intCast_im]
    /-
      case mpr
      d a b : Int
      ⊢ Dvd.dvd a b → And (Dvd.dvd a b) (Dvd.dvd a 0)
    -/
    exact fun hc => ⟨hc, dvd_zero a⟩
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-05-25")] alias coe_int_dvd_iff := intCast_dvd

@[deprecated (since := "2024-05-25")] alias coe_int_dvd_coe_int := intCast_dvd_intCast


protected theorem eq_of_smul_eq_smul_left {a : ℤ} {b c : ℤ√d} (ha : a ≠ 0) (h : ↑a * b = a * c) :
    b = c := by
  /-
    d a : Int
    b c : Zsqrtd d
    ha : Ne a 0
    h : Eq (HMul.hMul (↑a) b) (HMul.hMul (↑a) c)
    ⊢ Eq b c
  -/
  rw [Zsqrtd.ext_iff] at h ⊢
  /-
    d a : Int
    b c : Zsqrtd d
    ha : Ne a 0
    h : And (Eq (HMul.hMul (↑a) b).re (HMul.hMul (↑a) c).re) (Eq (HMul.hMul (↑a) b …
    ⊢ And (Eq b.re c.re) (Eq b.im c.im)
  -/
                          /-
                            🎉 no goals
                          -/
  apply And.imp _ _ h <;> simpa only [smul_re, smul_im] using mul_left_cancel₀ ha
                          /-
                            🎉 no goals
                          -/


theorem gcd_eq_zero_iff (a : ℤ√d) : Int.gcd a.re a.im = 0 ↔ a = 0 := by
  /-
    d : Int
    a : Zsqrtd d
    ⊢ Iff (Eq (a.re.gcd a.im) 0) (Eq a 0)
  -/
  simp only [Int.gcd_eq_zero_iff, Zsqrtd.ext_iff, eq_self_iff_true, zero_im, zero_re]
  /-
    🎉 no goals
  -/


theorem gcd_pos_iff (a : ℤ√d) : 0 < Int.gcd a.re a.im ↔ a ≠ 0 :=
  pos_iff_ne_zero.trans <| not_congr a.gcd_eq_zero_iff


theorem coprime_of_dvd_coprime {a b : ℤ√d} (hcoprime : IsCoprime a.re a.im) (hdvd : b ∣ a) :
    IsCoprime b.re b.im := by
  /-
    d : Int
    a b : Zsqrtd d
    hcoprime : IsCoprime a.re a.im
    hdvd : Dvd.dvd b a
    ⊢ IsCoprime b.re b.im
  -/
  apply isCoprime_of_dvd
    /-
      case nonzero
      d : Int
      a b : Zsqrtd d
      hcoprime : IsCoprime a.re a.im
      hdvd : Dvd.dvd b a
      ⊢ Not (And (Eq b.re 0) (Eq b.im 0))
    -/
  · rintro ⟨hre, him⟩
    /-
      case nonzero.intro
      d : Int
      a b : Zsqrtd d
      hcoprime : IsCoprime a.re a.im
      hdvd : Dvd.dvd b a
      hre : Eq b.re 0
      him : Eq b.im 0
      ⊢ False
    -/
    obtain rfl : b = 0 := Zsqrtd.ext hre him
    /-
      case nonzero.intro
      d : Int
      a : Zsqrtd d
      hcoprime : IsCoprime a.re a.im
      hdvd : Dvd.dvd 0 a
      hre : Eq (Zsqrtd.re 0) 0
      him : Eq (Zsqrtd.im 0) 0
      ⊢ False
    -/
    rw [zero_dvd_iff] at hdvd
    /-
      case nonzero.intro
      d : Int
      a : Zsqrtd d
      hcoprime : IsCoprime a.re a.im
      hdvd : Eq a 0
      hre : Eq (Zsqrtd.re 0) 0
      him : Eq (Zsqrtd.im 0) 0
      ⊢ False
    -/
    simp [hdvd, zero_im, zero_re, not_isCoprime_zero_zero] at hcoprime
    /-
      🎉 no goals
    -/
    /-
      case H
      d : Int
      a b : Zsqrtd d
      hcoprime : IsCoprime a.re a.im
      hdvd : Dvd.dvd b a
      ⊢ ∀ (z : Int), Membership.mem (nonunits Int) z → Ne z 0 → Dvd.dvd z b.re → Not …
    -/
  · rintro z hz - hzdvdu hzdvdv
    /-
      case H
      d : Int
      a b : Zsqrtd d
      hcoprime : IsCoprime a.re a.im
      hdvd : Dvd.dvd b a
      z : Int
      hz : Membership.mem (nonunits Int) z
      hzdvdu : Dvd.dvd z b.re
      hzdvdv : Dvd.dvd z b.im
      ⊢ False
    -/
    apply hz
    obtain ⟨ha, hb⟩ : z ∣ a.re ∧ z ∣ a.im := by
      rw [← intCast_dvd]
      apply dvd_trans _ hdvd
      rw [intCast_dvd]
      exact ⟨hzdvdu, hzdvdv⟩
    /-
      case H.intro
      d : Int
      a b : Zsqrtd d
      hcoprime : IsCoprime a.re a.im
      hdvd : Dvd.dvd b a
      z : Int
      hz : Membership.mem (nonunits Int) z
      hzdvdu : Dvd.dvd z b.re
      hzdvdv : Dvd.dvd z b.im
      ha : Dvd.dvd z a.re
      hb : Dvd.dvd z a.im
      ⊢ IsUnit z
    -/
    exact hcoprime.isUnit_of_dvd' ha hb
    /-
      🎉 no goals
    -/


theorem exists_coprime_of_gcd_pos {a : ℤ√d} (hgcd : 0 < Int.gcd a.re a.im) :
    ∃ b : ℤ√d, a = ((Int.gcd a.re a.im : ℤ) : ℤ√d) * b ∧ IsCoprime b.re b.im := by
  /-
    d : Int
    a : Zsqrtd d
    hgcd : LT.lt 0 (a.re.gcd a.im)
    ⊢ Exists fun b => And (Eq a (HMul.hMul (↑↑(a.re.gcd a.im)) b)) (IsCoprime b.re …
  -/
  obtain ⟨re, im, H1, Hre, Him⟩ := Int.exists_gcd_one hgcd
  /-
    case intro.intro.intro.intro
    d : Int
    a : Zsqrtd d
    hgcd : LT.lt 0 (a.re.gcd a.im)
    re im : Int
    H1 : Eq (re.gcd im) 1
    Hre : Eq a.re (HMul.hMul re ↑(a.re.gcd a.im))
    Him : Eq a.im (HMul.hMul im ↑(a.re.gcd a.im))
    ⊢ Exists fun b => And (Eq a (HMul.hMul (↑↑(a.re.gcd a.im)) b)) (IsCoprime b.re …
  -/
  rw [mul_comm] at Hre Him
  /-
    case intro.intro.intro.intro
    d : Int
    a : Zsqrtd d
    hgcd : LT.lt 0 (a.re.gcd a.im)
    re im : Int
    H1 : Eq (re.gcd im) 1
    Hre : Eq a.re (HMul.hMul (↑(a.re.gcd a.im)) re)
    Him : Eq a.im (HMul.hMul (↑(a.re.gcd a.im)) im)
    ⊢ Exists fun b => And (Eq a (HMul.hMul (↑↑(a.re.gcd a.im)) b)) (IsCoprime b.re …
  -/
  refine ⟨⟨re, im⟩, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.refine_1
      d : Int
      a : Zsqrtd d
      hgcd : LT.lt 0 (a.re.gcd a.im)
      re im : Int
      H1 : Eq (re.gcd im) 1
      Hre : Eq a.re (HMul.hMul (↑(a.re.gcd a.im)) re)
      Him : Eq a.im (HMul.hMul (↑(a.re.gcd a.im)) im)
      ⊢ Eq a (HMul.hMul ↑↑(a.re.gcd a.im) { re := re, im := im })
    -/
  · rw [smul_val, ← Hre, ← Him]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_2
      d : Int
      a : Zsqrtd d
      hgcd : LT.lt 0 (a.re.gcd a.im)
      re im : Int
      H1 : Eq (re.gcd im) 1
      Hre : Eq a.re (HMul.hMul (↑(a.re.gcd a.im)) re)
      Him : Eq a.im (HMul.hMul (↑(a.re.gcd a.im)) im)
      ⊢ IsCoprime { re := re, im := im }.re { re := re, im := im }.im
    -/
  · rw [← Int.gcd_eq_one_iff_coprime, H1]
    /-
      🎉 no goals
    -/


/-- Read `SqLe a c b d` as `a √c ≤ b √d` -/
def SqLe (a c b d : ℕ) : Prop :=
  c * a * a ≤ d * b * b


theorem sqLe_of_le {c d x y z w : ℕ} (xz : z ≤ x) (yw : y ≤ w) (xy : SqLe x c y d) : SqLe z c w d :=
  le_trans (mul_le_mul (Nat.mul_le_mul_left _ xz) xz (Nat.zero_le _) (Nat.zero_le _)) <|
    le_trans xy (mul_le_mul (Nat.mul_le_mul_left _ yw) yw (Nat.zero_le _) (Nat.zero_le _))


theorem sqLe_add_mixed {c d x y z w : ℕ} (xy : SqLe x c y d) (zw : SqLe z c w d) :
    c * (x * z) ≤ d * (y * w) :=
  Nat.mul_self_le_mul_self_iff.1 <| by
    /-
      c d x y z w : Nat
      xy : Zsqrtd.SqLe x c y d
      zw : Zsqrtd.SqLe z c w d
      ⊢ LE.le (HMul.hMul (HMul.hMul c (HMul.hMul x z)) (HMul.hMul c (HMul.hMul x z)) …
    -/
    simpa [mul_comm, mul_left_comm] using mul_le_mul xy zw (Nat.zero_le _) (Nat.zero_le _)
    /-
      🎉 no goals
    -/


theorem sqLe_add {c d x y z w : ℕ} (xy : SqLe x c y d) (zw : SqLe z c w d) :
    SqLe (x + z) c (y + w) d := by
  /-
    c d x y z w : Nat
    xy : Zsqrtd.SqLe x c y d
    zw : Zsqrtd.SqLe z c w d
    ⊢ Zsqrtd.SqLe (HAdd.hAdd x z) c (HAdd.hAdd y w) d
  -/
  have xz := sqLe_add_mixed xy zw
  /-
    c d x y z w : Nat
    xy : Zsqrtd.SqLe x c y d
    zw : Zsqrtd.SqLe z c w d
    xz : LE.le (HMul.hMul c (HMul.hMul x z)) (HMul.hMul d (HMul.hMul y w))
    ⊢ Zsqrtd.SqLe (HAdd.hAdd x z) c (HAdd.hAdd y w) d
  -/
  simp? [SqLe, mul_assoc] at xy zw says simp only [SqLe, mul_assoc] at xy zw
  /-
    c d x y z w : Nat
    xz : LE.le (HMul.hMul c (HMul.hMul x z)) (HMul.hMul d (HMul.hMul y w))
    xy : LE.le (HMul.hMul c (HMul.hMul x x)) (HMul.hMul d (HMul.hMul y y))
    zw : LE.le (HMul.hMul c (HMul.hMul z z)) (HMul.hMul d (HMul.hMul w w))
    ⊢ Zsqrtd.SqLe (HAdd.hAdd x z) c (HAdd.hAdd y w) d
  -/
  simp [SqLe, mul_add, mul_comm, mul_left_comm, add_le_add, *]
  /-
    🎉 no goals
  -/


theorem sqLe_cancel {c d x y z w : ℕ} (zw : SqLe y d x c) (h : SqLe (x + z) c (y + w) d) :
    SqLe z c w d := by
  /-
    c d x y z w : Nat
    zw : Zsqrtd.SqLe y d x c
    h : Zsqrtd.SqLe (HAdd.hAdd x z) c (HAdd.hAdd y w) d
    ⊢ Zsqrtd.SqLe z c w d
  -/
  apply le_of_not_gt
  /-
    case a
    c d x y z w : Nat
    zw : Zsqrtd.SqLe y d x c
    h : Zsqrtd.SqLe (HAdd.hAdd x z) c (HAdd.hAdd y w) d
    ⊢ Not (GT.gt (HMul.hMul (HMul.hMul c z) z) (HMul.hMul (HMul.hMul d w) w))
  -/
  intro l
  /-
    case a
    c d x y z w : Nat
    zw : Zsqrtd.SqLe y d x c
    h : Zsqrtd.SqLe (HAdd.hAdd x z) c (HAdd.hAdd y w) d
    l : GT.gt (HMul.hMul (HMul.hMul c z) z) (HMul.hMul (HMul.hMul d w) w)
    ⊢ False
  -/
  refine not_le_of_gt ?_ h
  /-
    case a
    c d x y z w : Nat
    zw : Zsqrtd.SqLe y d x c
    h : Zsqrtd.SqLe (HAdd.hAdd x z) c (HAdd.hAdd y w) d
    l : GT.gt (HMul.hMul (HMul.hMul c z) z) (HMul.hMul (HMul.hMul d w) w)
    ⊢ GT.gt (HMul.hMul (HMul.hMul c (HAdd.hAdd x z)) (HAdd.hAdd x z)) (HMul.hMul ( …
  -/
  simp only [SqLe, mul_add, mul_comm, mul_left_comm, add_assoc, gt_iff_lt]
  /-
    case a
    c d x y z w : Nat
    zw : Zsqrtd.SqLe y d x c
    h : Zsqrtd.SqLe (HAdd.hAdd x z) c (HAdd.hAdd y w) d
    l : GT.gt (HMul.hMul (HMul.hMul c z) z) (HMul.hMul (HMul.hMul d w) w)
    ⊢ LT.lt (HAdd.hAdd (HMul.hMul d (HMul.hMul y y)) (HAdd.hAdd (HMul.hMul d (HMul …
  -/
  have hm := sqLe_add_mixed zw (le_of_lt l)
  /-
    case a
    c d x y z w : Nat
    zw : Zsqrtd.SqLe y d x c
    h : Zsqrtd.SqLe (HAdd.hAdd x z) c (HAdd.hAdd y w) d
    l : GT.gt (HMul.hMul (HMul.hMul c z) z) (HMul.hMul (HMul.hMul d w) w)
    hm : LE.le (HMul.hMul d (HMul.hMul y w)) (HMul.hMul c (HMul.hMul x z))
    ⊢ LT.lt (HAdd.hAdd (HMul.hMul d (HMul.hMul y y)) (HAdd.hAdd (HMul.hMul d (HMul …
  -/
  simp only [SqLe, mul_assoc, gt_iff_lt] at l zw
  exact
    lt_of_le_of_lt (add_le_add_right zw _)
      (add_lt_add_left (add_lt_add_of_le_of_lt hm (add_lt_add_of_le_of_lt hm l)) _)


theorem sqLe_smul {c d x y : ℕ} (n : ℕ) (xy : SqLe x c y d) : SqLe (n * x) c (n * y) d := by
  /-
    c d x y n : Nat
    xy : Zsqrtd.SqLe x c y d
    ⊢ Zsqrtd.SqLe (HMul.hMul n x) c (HMul.hMul n y) d
  -/
  simpa [SqLe, mul_left_comm, mul_assoc] using Nat.mul_le_mul_left (n * n) xy
  /-
    🎉 no goals
  -/


theorem sqLe_mul {d x y z w : ℕ} :
    (SqLe x 1 y d → SqLe z 1 w d → SqLe (x * w + y * z) d (x * z + d * y * w) 1) ∧
      (SqLe x 1 y d → SqLe w d z 1 → SqLe (x * z + d * y * w) 1 (x * w + y * z) d) ∧
        (SqLe y d x 1 → SqLe z 1 w d → SqLe (x * z + d * y * w) 1 (x * w + y * z) d) ∧
          (SqLe y d x 1 → SqLe w d z 1 → SqLe (x * w + y * z) d (x * z + d * y * w) 1) := by
  /-
    d x y z w : Nat
    ⊢ And (Zsqrtd.SqLe x 1 y d → Zsqrtd.SqLe z 1 w d → Zsqrtd.SqLe (HAdd.hAdd (HMu …
  -/
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
      /-
        case refine_1
        d x y z w : Nat
        ⊢ Zsqrtd.SqLe x 1 y d → Zsqrtd.SqLe z 1 w d → Zsqrtd.SqLe (HAdd.hAdd (HMul.hMu …
      -/
    · intro xy zw
      have :=
        Int.mul_nonneg (sub_nonneg_of_le (Int.ofNat_le_ofNat_of_le xy))
      /-
        case refine_1
        d x y z w : Nat
        xy : Zsqrtd.SqLe x 1 y d
        zw : Zsqrtd.SqLe z 1 w d
        this : LE.le 0 (HMul.hMul (HSub.hSub ↑(HMul.hMul (HMul.hMul d y) y) ↑(HMul.hMu …
        ⊢ Zsqrtd.SqLe (HAdd.hAdd (HMul.hMul x w) (HMul.hMul y z)) d (HAdd.hAdd (HMul.h …
      -/
      /-
        case refine_1
        d x y z w : Nat
        xy : Zsqrtd.SqLe x 1 y d
        zw : Zsqrtd.SqLe z 1 w d
        this : LE.le 0 (HMul.hMul (HSub.hSub ↑(HMul.hMul (HMul.hMul d y) y) ↑(HMul.hMu …
        ⊢ LE.le 0 (HSub.hSub ↑(HMul.hMul (HMul.hMul 1 (HAdd.hAdd (HMul.hMul x z) (HMul …
      -/
      /-
        case h.e'_4
        d x y z w : Nat
        xy : Zsqrtd.SqLe x 1 y d
        zw : Zsqrtd.SqLe z 1 w d
        this : LE.le 0 (HMul.hMul (HSub.hSub ↑(HMul.hMul (HMul.hMul d y) y) ↑(HMul.hMu …
        ⊢ Eq (HSub.hSub ↑(HMul.hMul (HMul.hMul 1 (HAdd.hAdd (HMul.hMul x z) (HMul.hMul …
      -/
      /-
        case h.e'_4
        d x y z w : Nat
        xy : Zsqrtd.SqLe x 1 y d
        zw : Zsqrtd.SqLe z 1 w d
        this : LE.le 0 (HMul.hMul (HSub.hSub ↑(HMul.hMul (HMul.hMul d y) y) ↑(HMul.hMu …
        ⊢ Eq (HSub.hSub (HMul.hMul (HAdd.hAdd (HMul.hMul ↑x ↑z) (HMul.hMul (HMul.hMul  …
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
      convert this using 1
      /-
        case h.e'_4
        d x y z w : Nat
        xy : Zsqrtd.SqLe y d x 1
        zw : Zsqrtd.SqLe w d z 1
        this : LE.le 0 (HMul.hMul (HSub.hSub ↑(HMul.hMul (HMul.hMul 1 x) x) ↑(HMul.hMu …
        ⊢ Eq (HSub.hSub ↑(HMul.hMul (HMul.hMul 1 (HAdd.hAdd (HMul.hMul x z) (HMul.hMul …
      -/
      simp only [one_mul, Int.ofNat_add, Int.ofNat_mul]
      /-
        case h.e'_4
        d x y z w : Nat
        xy : Zsqrtd.SqLe y d x 1
        zw : Zsqrtd.SqLe w d z 1
        this : LE.le 0 (HMul.hMul (HSub.hSub ↑(HMul.hMul (HMul.hMul 1 x) x) ↑(HMul.hMu …
        ⊢ Eq (HSub.hSub (HMul.hMul (HAdd.hAdd (HMul.hMul ↑x ↑z) (HMul.hMul (HMul.hMul  …
      -/
      ring
      /-
        🎉 no goals
      -/


open Int in
/-- "Generalized" `nonneg`. `nonnegg c d x y` means `a √c + b √d ≥ 0`;
  we are interested in the case `c = 1` but this is more symmetric -/
def Nonnegg (c d : ℕ) : ℤ → ℤ → Prop
  | (a : ℕ), (b : ℕ) => True
  | (a : ℕ), -[b+1] => SqLe (b + 1) c a d
  | -[a+1], (b : ℕ) => SqLe (a + 1) d b c
  | -[_+1], -[_+1] => False


theorem nonnegg_comm {c d : ℕ} {x y : ℤ} : Nonnegg c d x y = Nonnegg d c y x := by
  /-
    c d : Nat
    x y : Int
    ⊢ Eq (Zsqrtd.Nonnegg c d x y) (Zsqrtd.Nonnegg d c y x)
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
  induction x <;> induction y <;> rfl
                                  /-
                                    🎉 no goals
                                  -/


theorem nonnegg_neg_pos {c d} : ∀ {a b : ℕ}, Nonnegg c d (-a) b ↔ SqLe a d b c
                /-
                  c d b : Nat
                  ⊢ Zsqrtd.Nonnegg c d (Neg.neg ↑0) ↑b → Zsqrtd.SqLe 0 d b c
                -/
  | 0, b => ⟨by simp [SqLe, Nat.zero_le], fun _ => trivial⟩
                /-
                  🎉 no goals
                -/
                   /-
                     c d a b : Nat
                     ⊢ Iff (Zsqrtd.Nonnegg c d (Neg.neg ↑(HAdd.hAdd a 1)) ↑b) (Zsqrtd.SqLe (HAdd.hA …
                   -/
  | a + 1, b => by rw [← Int.negSucc_coe]; rfl
                                           /-
                                             🎉 no goals
                                           -/


theorem nonnegg_pos_neg {c d} {a b : ℕ} : Nonnegg c d a (-b) ↔ SqLe b c a d := by
  /-
    c d a b : Nat
    ⊢ Iff (Zsqrtd.Nonnegg c d (↑a) (Neg.neg ↑b)) (Zsqrtd.SqLe b c a d)
  -/
  rw [nonnegg_comm]; exact nonnegg_neg_pos
                     /-
                       🎉 no goals
                     -/


open Int in
theorem nonnegg_cases_right {c d} {a : ℕ} :
    ∀ {b : ℤ}, (∀ x : ℕ, b = -x → SqLe x c a d) → Nonnegg c d a b
  | (b : Nat), _ => trivial
  | -[b+1], h => h (b + 1) rfl


theorem nonnegg_cases_left {c d} {b : ℕ} {a : ℤ} (h : ∀ x : ℕ, a = -x → SqLe x d b c) :
    Nonnegg c d a b :=
  cast nonnegg_comm (nonnegg_cases_right h)


/-- The norm of an element of `ℤ[√d]`. -/
def norm (n : ℤ√d) : ℤ :=
  n.re * n.re - d * n.im * n.im


theorem norm_def (n : ℤ√d) : n.norm = n.re * n.re - d * n.im * n.im :=
  rfl


@[simp]
                                             /-
                                               d : Int
                                               ⊢ Eq (Zsqrtd.norm 0) 0
                                             -/
theorem norm_zero : norm (0 : ℤ√d) = 0 := by simp [norm]
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
                                            /-
                                              d : Int
                                              ⊢ Eq (Zsqrtd.norm 1) 1
                                            -/
theorem norm_one : norm (1 : ℤ√d) = 1 := by simp [norm]
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
                                                            /-
                                                              d n : Int
                                                              ⊢ Eq (↑n).norm (HMul.hMul n n)
                                                            -/
theorem norm_intCast (n : ℤ) : norm (n : ℤ√d) = n * n := by simp [norm]
                                                            /-
                                                              🎉 no goals
                                                            -/


@[deprecated (since := "2024-04-17")]
alias norm_int_cast := norm_intCast


@[simp]
theorem norm_natCast (n : ℕ) : norm (n : ℤ√d) = n * n :=
  norm_intCast n


@[deprecated (since := "2024-04-17")]
alias norm_nat_cast := norm_natCast


@[simp]
theorem norm_mul (n m : ℤ√d) : norm (n * m) = norm n * norm m := by
  /-
    d : Int
    n m : Zsqrtd d
    ⊢ Eq (HMul.hMul n m).norm (HMul.hMul n.norm m.norm)
  -/
  simp only [norm, mul_im, mul_re]
  /-
    d : Int
    n m : Zsqrtd d
    ⊢ Eq (HSub.hSub (HMul.hMul (HAdd.hAdd (HMul.hMul n.re m.re) (HMul.hMul (HMul.h …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- `norm` as a `MonoidHom`. -/
def normMonoidHom : ℤ√d →* ℤ where
  toFun := norm
  map_mul' := norm_mul
  map_one' := norm_one


theorem norm_eq_mul_conj (n : ℤ√d) : (norm n : ℤ√d) = n * star n := by
  /-
    d : Int
    n : Zsqrtd d
    ⊢ Eq (↑n.norm) (HMul.hMul n (Star.star n))
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp [norm, star, mul_comm, sub_eq_add_neg]
          /-
            🎉 no goals
          -/


@[simp]
theorem norm_neg (x : ℤ√d) : (-x).norm = x.norm :=
  -- Porting note: replaced `simp` with `rw`
  -- See https://github.com/leanprover-community/mathlib4/issues/5026
                                    /-
                                      d : Int
                                      x : Zsqrtd d
                                      ⊢ Eq ↑(Neg.neg x).norm ↑x.norm
                                    -/
  (Int.cast_inj (α := ℤ√d)).1 <| by rw [norm_eq_mul_conj, star_neg, neg_mul_neg, norm_eq_mul_conj]
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem norm_conj (x : ℤ√d) : (star x).norm = x.norm :=
  -- Porting note: replaced `simp` with `rw`
  -- See https://github.com/leanprover-community/mathlib4/issues/5026
                                    /-
                                      d : Int
                                      x : Zsqrtd d
                                      ⊢ Eq ↑(Star.star x).norm ↑x.norm
                                    -/
  (Int.cast_inj (α := ℤ√d)).1 <| by rw [norm_eq_mul_conj, star_star, mul_comm, norm_eq_mul_conj]
                                    /-
                                      🎉 no goals
                                    -/


theorem norm_nonneg (hd : d ≤ 0) (n : ℤ√d) : 0 ≤ n.norm :=
  add_nonneg (mul_self_nonneg _)
    (by
      /-
        d : Int
        hd : LE.le d 0
        n : Zsqrtd d
        ⊢ LE.le 0 (Neg.neg (HMul.hMul (HMul.hMul d n.im) n.im))
      -/
      rw [mul_assoc, neg_mul_eq_neg_mul]
      /-
        d : Int
        hd : LE.le d 0
        n : Zsqrtd d
        ⊢ LE.le 0 (HMul.hMul (Neg.neg d) (HMul.hMul n.im n.im))
      -/
      exact mul_nonneg (neg_nonneg.2 hd) (mul_self_nonneg _))
      /-
        🎉 no goals
      -/


theorem norm_eq_one_iff {x : ℤ√d} : x.norm.natAbs = 1 ↔ IsUnit x :=
  ⟨fun h =>
    isUnit_iff_dvd_one.2 <|
      (le_total 0 (norm x)).casesOn
        (fun hx =>
          ⟨star x, by
            rwa [← Int.natCast_inj, Int.natAbs_of_nonneg hx, ← @Int.cast_inj (ℤ√d) _ _,
              norm_eq_mul_conj, eq_comm] at h⟩)
        fun hx =>
          ⟨-star x, by
            rwa [← Int.natCast_inj, Int.ofNat_natAbs_of_nonpos hx, ← @Int.cast_inj (ℤ√d) _ _,
              Int.cast_neg, norm_eq_mul_conj, neg_mul_eq_mul_neg, eq_comm] at h⟩,
    fun h => by
    /-
      d : Int
      x : Zsqrtd d
      h : IsUnit x
      ⊢ Eq x.norm.natAbs 1
    -/
    let ⟨y, hy⟩ := isUnit_iff_dvd_one.1 h
    /-
      d : Int
      x : Zsqrtd d
      h : IsUnit x
      y : Zsqrtd d
      hy : Eq 1 (HMul.hMul x y)
      ⊢ Eq x.norm.natAbs 1
    -/
    have := congr_arg (Int.natAbs ∘ norm) hy
    rw [Function.comp_apply, Function.comp_apply, norm_mul, Int.natAbs_mul, norm_one,
      Int.natAbs_one, eq_comm, mul_eq_one] at this
    /-
      d : Int
      x : Zsqrtd d
      h : IsUnit x
      y : Zsqrtd d
      hy : Eq 1 (HMul.hMul x y)
      this : And (Eq x.norm.natAbs 1) (Eq y.norm.natAbs 1)
      ⊢ Eq x.norm.natAbs 1
    -/
    exact this.1⟩
    /-
      🎉 no goals
    -/


theorem isUnit_iff_norm_isUnit {d : ℤ} (z : ℤ√d) : IsUnit z ↔ IsUnit z.norm := by
  /-
    d : Int
    z : Zsqrtd d
    ⊢ Iff (IsUnit z) (IsUnit z.norm)
  -/
  rw [Int.isUnit_iff_natAbs_eq, norm_eq_one_iff]
  /-
    🎉 no goals
  -/


theorem norm_eq_one_iff' {d : ℤ} (hd : d ≤ 0) (z : ℤ√d) : z.norm = 1 ↔ IsUnit z := by
  /-
    d : Int
    hd : LE.le d 0
    z : Zsqrtd d
    ⊢ Iff (Eq z.norm 1) (IsUnit z)
  -/
  rw [← norm_eq_one_iff, ← Int.natCast_inj, Int.natAbs_of_nonneg (norm_nonneg hd z), Int.ofNat_one]
  /-
    🎉 no goals
  -/


theorem norm_eq_zero_iff {d : ℤ} (hd : d < 0) (z : ℤ√d) : z.norm = 0 ↔ z = 0 := by
  /-
    d : Int
    hd : LT.lt d 0
    z : Zsqrtd d
    ⊢ Iff (Eq z.norm 0) (Eq z 0)
  -/
  constructor
    /-
      case mp
      d : Int
      hd : LT.lt d 0
      z : Zsqrtd d
      ⊢ Eq z.norm 0 → Eq z 0
    -/
  · intro h
    /-
      case mp
      d : Int
      hd : LT.lt d 0
      z : Zsqrtd d
      h : Eq z.norm 0
      ⊢ Eq z 0
    -/
    rw [norm_def, sub_eq_add_neg, mul_assoc] at h
    /-
      case mp
      d : Int
      hd : LT.lt d 0
      z : Zsqrtd d
      h : Eq (HAdd.hAdd (HMul.hMul z.re z.re) (Neg.neg (HMul.hMul d (HMul.hMul z.im  …
      ⊢ Eq z 0
    -/
    have left := mul_self_nonneg z.re
    /-
      case mp
      d : Int
      hd : LT.lt d 0
      z : Zsqrtd d
      h : Eq (HAdd.hAdd (HMul.hMul z.re z.re) (Neg.neg (HMul.hMul d (HMul.hMul z.im  …
      left : LE.le 0 (HMul.hMul z.re z.re)
      ⊢ Eq z 0
    -/
    have right := neg_nonneg.mpr (mul_nonpos_of_nonpos_of_nonneg hd.le (mul_self_nonneg z.im))
    /-
      case mp
      d : Int
      hd : LT.lt d 0
      z : Zsqrtd d
      h : Eq (HAdd.hAdd (HMul.hMul z.re z.re) (Neg.neg (HMul.hMul d (HMul.hMul z.im  …
      left : LE.le 0 (HMul.hMul z.re z.re)
      right : LE.le 0 (Neg.neg (HMul.hMul d (HMul.hMul z.im z.im)))
      ⊢ Eq z 0
    -/
    obtain ⟨ha, hb⟩ := (add_eq_zero_iff_of_nonneg left right).mp h
    /-
      case mp.intro
      d : Int
      hd : LT.lt d 0
      z : Zsqrtd d
      h : Eq (HAdd.hAdd (HMul.hMul z.re z.re) (Neg.neg (HMul.hMul d (HMul.hMul z.im  …
      left : LE.le 0 (HMul.hMul z.re z.re)
      right : LE.le 0 (Neg.neg (HMul.hMul d (HMul.hMul z.im z.im)))
      ha : Eq (HMul.hMul z.re z.re) 0
      hb : Eq (Neg.neg (HMul.hMul d (HMul.hMul z.im z.im))) 0
      ⊢ Eq z 0
    -/
    ext <;> apply eq_zero_of_mul_self_eq_zero
      /-
        case mp.intro.re.h
        d : Int
        hd : LT.lt d 0
        z : Zsqrtd d
        h : Eq (HAdd.hAdd (HMul.hMul z.re z.re) (Neg.neg (HMul.hMul d (HMul.hMul z.im  …
        left : LE.le 0 (HMul.hMul z.re z.re)
        right : LE.le 0 (Neg.neg (HMul.hMul d (HMul.hMul z.im z.im)))
        ha : Eq (HMul.hMul z.re z.re) 0
        hb : Eq (Neg.neg (HMul.hMul d (HMul.hMul z.im z.im))) 0
        ⊢ Eq (HMul.hMul z.re z.re) 0
      -/
    · exact ha
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.im.h
        d : Int
        hd : LT.lt d 0
        z : Zsqrtd d
        h : Eq (HAdd.hAdd (HMul.hMul z.re z.re) (Neg.neg (HMul.hMul d (HMul.hMul z.im  …
        left : LE.le 0 (HMul.hMul z.re z.re)
        right : LE.le 0 (Neg.neg (HMul.hMul d (HMul.hMul z.im z.im)))
        ha : Eq (HMul.hMul z.re z.re) 0
        hb : Eq (Neg.neg (HMul.hMul d (HMul.hMul z.im z.im))) 0
        ⊢ Eq (HMul.hMul z.im z.im) 0
      -/
    · rw [neg_eq_zero, mul_eq_zero] at hb
      /-
        case mp.intro.im.h
        d : Int
        hd : LT.lt d 0
        z : Zsqrtd d
        h : Eq (HAdd.hAdd (HMul.hMul z.re z.re) (Neg.neg (HMul.hMul d (HMul.hMul z.im  …
        left : LE.le 0 (HMul.hMul z.re z.re)
        right : LE.le 0 (Neg.neg (HMul.hMul d (HMul.hMul z.im z.im)))
        ha : Eq (HMul.hMul z.re z.re) 0
        hb : Or (Eq d 0) (Eq (HMul.hMul z.im z.im) 0)
        ⊢ Eq (HMul.hMul z.im z.im) 0
      -/
      exact hb.resolve_left hd.ne
      /-
        🎉 no goals
      -/
    /-
      case mpr
      d : Int
      hd : LT.lt d 0
      z : Zsqrtd d
      ⊢ Eq z 0 → Eq z.norm 0
    -/
  · rintro rfl
    /-
      case mpr
      d : Int
      hd : LT.lt d 0
      ⊢ Eq (Zsqrtd.norm 0) 0
    -/
    exact norm_zero
    /-
      🎉 no goals
    -/


theorem norm_eq_of_associated {d : ℤ} (hd : d ≤ 0) {x y : ℤ√d} (h : Associated x y) :
    x.norm = y.norm := by
  /-
    d : Int
    hd : LE.le d 0
    x y : Zsqrtd d
    h : Associated x y
    ⊢ Eq x.norm y.norm
  -/
  obtain ⟨u, rfl⟩ := h
  /-
    case intro
    d : Int
    hd : LE.le d 0
    x : Zsqrtd d
    u : Units (Zsqrtd d)
    ⊢ Eq x.norm (HMul.hMul x ↑u).norm
  -/
  rw [norm_mul, (norm_eq_one_iff' hd _).mpr u.isUnit, mul_one]
  /-
    🎉 no goals
  -/


/-- Nonnegativity of an element of `ℤ√d`. -/
def Nonneg : ℤ√d → Prop
  | ⟨a, b⟩ => Nonnegg d 1 a b


instance : LE (ℤ√d) :=
  ⟨fun a b => Nonneg (b - a)⟩


instance : LT (ℤ√d) :=
  ⟨fun a b => ¬b ≤ a⟩


instance decidableNonnegg (c d a b) : Decidable (Nonnegg c d a b) := by
  /-
    d✝ c d : Nat
    a b : Int
    ⊢ Decidable (Zsqrtd.Nonnegg c d a b)
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
  cases a <;> cases b <;> unfold Nonnegg SqLe <;> infer_instance
                                                  /-
                                                    🎉 no goals
                                                  -/


instance decidableNonneg : ∀ a : ℤ√d, Decidable (Nonneg a)
  | ⟨_, _⟩ => Zsqrtd.decidableNonnegg _ _ _ _


instance decidableLE : DecidableRel (α := ℤ√d) (· ≤ ·) := fun _ _ => decidableNonneg _


open Int in
theorem nonneg_cases : ∀ {a : ℤ√d}, Nonneg a → ∃ x y : ℕ, a = ⟨x, y⟩ ∨ a = ⟨x, -y⟩ ∨ a = ⟨-x, y⟩
  | ⟨(x : ℕ), (y : ℕ)⟩, _ => ⟨x, y, Or.inl rfl⟩
  | ⟨(x : ℕ), -[y+1]⟩, _ => ⟨x, y + 1, Or.inr <| Or.inl rfl⟩
  | ⟨-[x+1], (y : ℕ)⟩, _ => ⟨x + 1, y, Or.inr <| Or.inr rfl⟩
  | ⟨-[_+1], -[_+1]⟩, h => False.elim h


open Int in
theorem nonneg_add_lem {x y z w : ℕ} (xy : Nonneg (⟨x, -y⟩ : ℤ√d)) (zw : Nonneg (⟨-z, w⟩ : ℤ√d)) :
    Nonneg (⟨x, -y⟩ + ⟨-z, w⟩ : ℤ√d) := by
  have : Nonneg ⟨Int.subNatNat x z, Int.subNatNat w y⟩ :=
    Int.subNatNat_elim x z
      (fun m n i => SqLe y d m 1 → SqLe n 1 w d → Nonneg ⟨i, Int.subNatNat w y⟩)
      (fun j k =>
        Int.subNatNat_elim w y
          (fun m n i => SqLe n d (k + j) 1 → SqLe k 1 m d → Nonneg ⟨Int.ofNat j, i⟩)
          (fun _ _ _ _ => trivial) fun m n xy zw => sqLe_cancel zw xy)
      (fun j k =>
        Int.subNatNat_elim w y
          (fun m n i => SqLe n d k 1 → SqLe (k + j + 1) 1 m d → Nonneg ⟨-[j+1], i⟩)
          (fun m n xy zw => sqLe_cancel xy zw) fun m n xy zw =>
          let t := Nat.le_trans zw (sqLe_of_le (Nat.le_add_right n (m + 1)) le_rfl xy)
          have : k + j + 1 ≤ k :=
            Nat.mul_self_le_mul_self_iff.1 (by simpa [one_mul] using t)
          absurd this (not_le_of_gt <| Nat.succ_le_succ <| Nat.le_add_right _ _))
      (nonnegg_pos_neg.1 xy) (nonnegg_neg_pos.1 zw)
  /-
    d x y z w : Nat
    xy : { re := ↑x, im := Neg.neg ↑y }.Nonneg
    zw : { re := Neg.neg ↑z, im := ↑w }.Nonneg
    this : { re := Int.subNatNat x z, im := Int.subNatNat w y }.Nonneg
    ⊢ (HAdd.hAdd { re := ↑x, im := Neg.neg ↑y } { re := Neg.neg ↑z, im := ↑w }).No …
  -/
  rw [add_def, neg_add_eq_sub]
  /-
    d x y z w : Nat
    xy : { re := ↑x, im := Neg.neg ↑y }.Nonneg
    zw : { re := Neg.neg ↑z, im := ↑w }.Nonneg
    this : { re := Int.subNatNat x z, im := Int.subNatNat w y }.Nonneg
    ⊢ { re := HAdd.hAdd (↑x) (Neg.neg ↑z), im := HSub.hSub ↑w ↑y }.Nonneg
  -/
  rwa [Int.subNatNat_eq_coe, Int.subNatNat_eq_coe] at this
  /-
    🎉 no goals
  -/


theorem Nonneg.add {a b : ℤ√d} (ha : Nonneg a) (hb : Nonneg b) : Nonneg (a + b) := by
  /-
    d : Nat
    a b : Zsqrtd ↑d
    ha : a.Nonneg
    hb : b.Nonneg
    ⊢ (HAdd.hAdd a b).Nonneg
  -/
  rcases nonneg_cases ha with ⟨x, y, rfl | rfl | rfl⟩ <;>
    /-
      case intro.intro.inl
      d : Nat
      b : Zsqrtd ↑d
      hb : b.Nonneg
      x y : Nat
      ha : { re := ↑x, im := ↑y }.Nonneg
      ⊢ (HAdd.hAdd { re := ↑x, im := ↑y } b).Nonneg
    -/
    rcases nonneg_cases hb with ⟨z, w, rfl | rfl | rfl⟩
    /-
      case intro.intro.inl.intro.intro.inl
      d x y : Nat
      ha : { re := ↑x, im := ↑y }.Nonneg
      z w : Nat
      hb : { re := ↑z, im := ↑w }.Nonneg
      ⊢ (HAdd.hAdd { re := ↑x, im := ↑y } { re := ↑z, im := ↑w }).Nonneg
    -/
  · trivial
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inl.intro.intro.inr.inl
      d x y : Nat
      ha : { re := ↑x, im := ↑y }.Nonneg
      z w : Nat
      hb : { re := ↑z, im := Neg.neg ↑w }.Nonneg
      ⊢ (HAdd.hAdd { re := ↑x, im := ↑y } { re := ↑z, im := Neg.neg ↑w }).Nonneg
    -/
  · refine nonnegg_cases_right fun i h => sqLe_of_le ?_ ?_ (nonnegg_pos_neg.1 hb)
      /-
        case intro.intro.inl.intro.intro.inr.inl.refine_1
        d x y : Nat
        ha : { re := ↑x, im := ↑y }.Nonneg
        z w : Nat
        hb : { re := ↑z, im := Neg.neg ↑w }.Nonneg
        i : Nat
        h : Eq (HAdd.hAdd { re := ↑x, im := ↑y }.im { re := ↑z, im := Neg.neg ↑w }.im) …
        ⊢ LE.le i w
      -/
    · dsimp only at h
      /-
        case intro.intro.inl.intro.intro.inr.inl.refine_1
        d x y : Nat
        ha : { re := ↑x, im := ↑y }.Nonneg
        z w : Nat
        hb : { re := ↑z, im := Neg.neg ↑w }.Nonneg
        i : Nat
        h : Eq (HAdd.hAdd (↑y) (Neg.neg ↑w)) (Neg.neg ↑i)
        ⊢ LE.le i w
      -/
      exact Int.ofNat_le.1 (le_of_neg_le_neg (Int.le.intro y (by simp [add_comm, *])))
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.inl.intro.intro.inr.inl.refine_2
        d x y : Nat
        ha : { re := ↑x, im := ↑y }.Nonneg
        z w : Nat
        hb : { re := ↑z, im := Neg.neg ↑w }.Nonneg
        i : Nat
        h : Eq (HAdd.hAdd { re := ↑x, im := ↑y }.im { re := ↑z, im := Neg.neg ↑w }.im) …
        ⊢ LE.le z (HAdd.hAdd x z)
      -/
    · apply Nat.le_add_left
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.inl.intro.intro.inr.inr
      d x y : Nat
      ha : { re := ↑x, im := ↑y }.Nonneg
      z w : Nat
      hb : { re := Neg.neg ↑z, im := ↑w }.Nonneg
      ⊢ (HAdd.hAdd { re := ↑x, im := ↑y } { re := Neg.neg ↑z, im := ↑w }).Nonneg
    -/
  · refine nonnegg_cases_left fun i h => sqLe_of_le ?_ ?_ (nonnegg_neg_pos.1 hb)
      /-
        case intro.intro.inl.intro.intro.inr.inr.refine_1
        d x y : Nat
        ha : { re := ↑x, im := ↑y }.Nonneg
        z w : Nat
        hb : { re := Neg.neg ↑z, im := ↑w }.Nonneg
        i : Nat
        h : Eq (HAdd.hAdd { re := ↑x, im := ↑y }.re { re := Neg.neg ↑z, im := ↑w }.re) …
        ⊢ LE.le i z
      -/
    · dsimp only at h
      /-
        case intro.intro.inl.intro.intro.inr.inr.refine_1
        d x y : Nat
        ha : { re := ↑x, im := ↑y }.Nonneg
        z w : Nat
        hb : { re := Neg.neg ↑z, im := ↑w }.Nonneg
        i : Nat
        h : Eq (HAdd.hAdd (↑x) (Neg.neg ↑z)) (Neg.neg ↑i)
        ⊢ LE.le i z
      -/
      exact Int.ofNat_le.1 (le_of_neg_le_neg (Int.le.intro x (by simp [add_comm, *])))
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.inl.intro.intro.inr.inr.refine_2
        d x y : Nat
        ha : { re := ↑x, im := ↑y }.Nonneg
        z w : Nat
        hb : { re := Neg.neg ↑z, im := ↑w }.Nonneg
        i : Nat
        h : Eq (HAdd.hAdd { re := ↑x, im := ↑y }.re { re := Neg.neg ↑z, im := ↑w }.re) …
        ⊢ LE.le w (HAdd.hAdd y w)
      -/
    · apply Nat.le_add_left
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.inr.inl.intro.intro.inl
      d x y : Nat
      ha : { re := ↑x, im := Neg.neg ↑y }.Nonneg
      z w : Nat
      hb : { re := ↑z, im := ↑w }.Nonneg
      ⊢ (HAdd.hAdd { re := ↑x, im := Neg.neg ↑y } { re := ↑z, im := ↑w }).Nonneg
    -/
  · refine nonnegg_cases_right fun i h => sqLe_of_le ?_ ?_ (nonnegg_pos_neg.1 ha)
      /-
        case intro.intro.inr.inl.intro.intro.inl.refine_1
        d x y : Nat
        ha : { re := ↑x, im := Neg.neg ↑y }.Nonneg
        z w : Nat
        hb : { re := ↑z, im := ↑w }.Nonneg
        i : Nat
        h : Eq (HAdd.hAdd { re := ↑x, im := Neg.neg ↑y }.im { re := ↑z, im := ↑w }.im) …
        ⊢ LE.le i y
      -/
    · dsimp only at h
      /-
        case intro.intro.inr.inl.intro.intro.inl.refine_1
        d x y : Nat
        ha : { re := ↑x, im := Neg.neg ↑y }.Nonneg
        z w : Nat
        hb : { re := ↑z, im := ↑w }.Nonneg
        i : Nat
        h : Eq (HAdd.hAdd (Neg.neg ↑y) ↑w) (Neg.neg ↑i)
        ⊢ LE.le i y
      -/
      exact Int.ofNat_le.1 (le_of_neg_le_neg (Int.le.intro w (by simp [*])))
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.inr.inl.intro.intro.inl.refine_2
        d x y : Nat
        ha : { re := ↑x, im := Neg.neg ↑y }.Nonneg
        z w : Nat
        hb : { re := ↑z, im := ↑w }.Nonneg
        i : Nat
        h : Eq (HAdd.hAdd { re := ↑x, im := Neg.neg ↑y }.im { re := ↑z, im := ↑w }.im) …
        ⊢ LE.le x (HAdd.hAdd x z)
      -/
    · apply Nat.le_add_right
      /-
        🎉 no goals
      -/
  · have : Nonneg ⟨_, _⟩ :=
      nonnegg_pos_neg.2 (sqLe_add (nonnegg_pos_neg.1 ha) (nonnegg_pos_neg.1 hb))
    /-
      case intro.intro.inr.inl.intro.intro.inr.inl
      d x y : Nat
      ha : { re := ↑x, im := Neg.neg ↑y }.Nonneg
      z w : Nat
      hb : { re := ↑z, im := Neg.neg ↑w }.Nonneg
      this : { re := ↑(HAdd.hAdd x z), im := Neg.neg ↑(HAdd.hAdd y w) }.Nonneg
      ⊢ (HAdd.hAdd { re := ↑x, im := Neg.neg ↑y } { re := ↑z, im := Neg.neg ↑w }).No …
    -/
    rw [Nat.cast_add, Nat.cast_add, neg_add] at this
    /-
      case intro.intro.inr.inl.intro.intro.inr.inl
      d x y : Nat
      ha : { re := ↑x, im := Neg.neg ↑y }.Nonneg
      z w : Nat
      hb : { re := ↑z, im := Neg.neg ↑w }.Nonneg
      this : { re := HAdd.hAdd ↑x ↑z, im := HAdd.hAdd (Neg.neg ↑y) (Neg.neg ↑w) }.No …
      ⊢ (HAdd.hAdd { re := ↑x, im := Neg.neg ↑y } { re := ↑z, im := Neg.neg ↑w }).No …
    -/
    rwa [add_def]
    /-
      🎉 no goals
    -/
    -- Porting note: was
    -- simpa [add_comm] using
    --   nonnegg_pos_neg.2 (sqLe_add (nonnegg_pos_neg.1 ha) (nonnegg_pos_neg.1 hb))
    /-
      case intro.intro.inr.inl.intro.intro.inr.inr
      d x y : Nat
      ha : { re := ↑x, im := Neg.neg ↑y }.Nonneg
      z w : Nat
      hb : { re := Neg.neg ↑z, im := ↑w }.Nonneg
      ⊢ (HAdd.hAdd { re := ↑x, im := Neg.neg ↑y } { re := Neg.neg ↑z, im := ↑w }).No …
    -/
  · exact nonneg_add_lem ha hb
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr.inr.intro.intro.inl
      d x y : Nat
      ha : { re := Neg.neg ↑x, im := ↑y }.Nonneg
      z w : Nat
      hb : { re := ↑z, im := ↑w }.Nonneg
      ⊢ (HAdd.hAdd { re := Neg.neg ↑x, im := ↑y } { re := ↑z, im := ↑w }).Nonneg
    -/
  · refine nonnegg_cases_left fun i h => sqLe_of_le ?_ ?_ (nonnegg_neg_pos.1 ha)
      /-
        case intro.intro.inr.inr.intro.intro.inl.refine_1
        d x y : Nat
        ha : { re := Neg.neg ↑x, im := ↑y }.Nonneg
        z w : Nat
        hb : { re := ↑z, im := ↑w }.Nonneg
        i : Nat
        h : Eq (HAdd.hAdd { re := Neg.neg ↑x, im := ↑y }.re { re := ↑z, im := ↑w }.re) …
        ⊢ LE.le i x
      -/
    · dsimp only at h
      /-
        case intro.intro.inr.inr.intro.intro.inl.refine_1
        d x y : Nat
        ha : { re := Neg.neg ↑x, im := ↑y }.Nonneg
        z w : Nat
        hb : { re := ↑z, im := ↑w }.Nonneg
        i : Nat
        h : Eq (HAdd.hAdd (Neg.neg ↑x) ↑z) (Neg.neg ↑i)
        ⊢ LE.le i x
      -/
      exact Int.ofNat_le.1 (le_of_neg_le_neg (Int.le.intro _ h))
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.inr.inr.intro.intro.inl.refine_2
        d x y : Nat
        ha : { re := Neg.neg ↑x, im := ↑y }.Nonneg
        z w : Nat
        hb : { re := ↑z, im := ↑w }.Nonneg
        i : Nat
        h : Eq (HAdd.hAdd { re := Neg.neg ↑x, im := ↑y }.re { re := ↑z, im := ↑w }.re) …
        ⊢ LE.le y (HAdd.hAdd y w)
      -/
    · apply Nat.le_add_right
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.inr.inr.intro.intro.inr.inl
      d x y : Nat
      ha : { re := Neg.neg ↑x, im := ↑y }.Nonneg
      z w : Nat
      hb : { re := ↑z, im := Neg.neg ↑w }.Nonneg
      ⊢ (HAdd.hAdd { re := Neg.neg ↑x, im := ↑y } { re := ↑z, im := Neg.neg ↑w }).No …
    -/
  · dsimp
    /-
      case intro.intro.inr.inr.intro.intro.inr.inl
      d x y : Nat
      ha : { re := Neg.neg ↑x, im := ↑y }.Nonneg
      z w : Nat
      hb : { re := ↑z, im := Neg.neg ↑w }.Nonneg
      ⊢ { re := HAdd.hAdd (Neg.neg ↑x) ↑z, im := HAdd.hAdd (↑y) (Neg.neg ↑w) }.Nonneg
    -/
    rw [add_comm, add_comm (y : ℤ)]
    /-
      case intro.intro.inr.inr.intro.intro.inr.inl
      d x y : Nat
      ha : { re := Neg.neg ↑x, im := ↑y }.Nonneg
      z w : Nat
      hb : { re := ↑z, im := Neg.neg ↑w }.Nonneg
      ⊢ { re := HAdd.hAdd (↑z) (Neg.neg ↑x), im := HAdd.hAdd (Neg.neg ↑w) ↑y }.Nonneg
    -/
    exact nonneg_add_lem hb ha
    /-
      🎉 no goals
    -/
  · have : Nonneg ⟨_, _⟩ :=
      nonnegg_neg_pos.2 (sqLe_add (nonnegg_neg_pos.1 ha) (nonnegg_neg_pos.1 hb))
    /-
      case intro.intro.inr.inr.intro.intro.inr.inr
      d x y : Nat
      ha : { re := Neg.neg ↑x, im := ↑y }.Nonneg
      z w : Nat
      hb : { re := Neg.neg ↑z, im := ↑w }.Nonneg
      this : { re := Neg.neg ↑(HAdd.hAdd x z), im := ↑(HAdd.hAdd y w) }.Nonneg
      ⊢ (HAdd.hAdd { re := Neg.neg ↑x, im := ↑y } { re := Neg.neg ↑z, im := ↑w }).No …
    -/
    rw [Nat.cast_add, Nat.cast_add, neg_add] at this
    /-
      case intro.intro.inr.inr.intro.intro.inr.inr
      d x y : Nat
      ha : { re := Neg.neg ↑x, im := ↑y }.Nonneg
      z w : Nat
      hb : { re := Neg.neg ↑z, im := ↑w }.Nonneg
      this : { re := HAdd.hAdd (Neg.neg ↑x) (Neg.neg ↑z), im := HAdd.hAdd ↑y ↑w }.No …
      ⊢ (HAdd.hAdd { re := Neg.neg ↑x, im := ↑y } { re := Neg.neg ↑z, im := ↑w }).No …
    -/
    rwa [add_def]
    /-
      🎉 no goals
    -/
    -- Porting note: was
    -- simpa [add_comm] using
    --   nonnegg_neg_pos.2 (sqLe_add (nonnegg_neg_pos.1 ha) (nonnegg_neg_pos.1 hb))


theorem nonneg_iff_zero_le {a : ℤ√d} : Nonneg a ↔ 0 ≤ a :=
                       /-
                         d : Nat
                         a : Zsqrtd ↑d
                         ⊢ Iff a.Nonneg (HSub.hSub a 0).Nonneg
                       -/
  show _ ↔ Nonneg _ by simp
                       /-
                         🎉 no goals
                       -/


theorem le_of_le_le {x y z w : ℤ} (xz : x ≤ z) (yw : y ≤ w) : (⟨x, y⟩ : ℤ√d) ≤ ⟨z, w⟩ :=
  show Nonneg ⟨z - x, w - y⟩ from
    match z - x, w - y, Int.le.dest_sub xz, Int.le.dest_sub yw with
    | _, _, ⟨_, rfl⟩, ⟨_, rfl⟩ => trivial


open Int in
protected theorem nonneg_total : ∀ a : ℤ√d, Nonneg a ∨ Nonneg (-a)
  | ⟨(x : ℕ), (y : ℕ)⟩ => Or.inl trivial
  | ⟨-[_+1], -[_+1]⟩ => Or.inr trivial
  | ⟨0, -[_+1]⟩ => Or.inr trivial
  | ⟨-[_+1], 0⟩ => Or.inr trivial
  | ⟨(_ + 1 : ℕ), -[_+1]⟩ => Nat.le_total _ _
  | ⟨-[_+1], (_ + 1 : ℕ)⟩ => Nat.le_total _ _


protected theorem le_total (a b : ℤ√d) : a ≤ b ∨ b ≤ a := by
  /-
    d : Nat
    a b : Zsqrtd ↑d
    ⊢ Or (LE.le a b) (LE.le b a)
  -/
  have t := (b - a).nonneg_total
  /-
    d : Nat
    a b : Zsqrtd ↑d
    t : Or (HSub.hSub b a).Nonneg (Neg.neg (HSub.hSub b a)).Nonneg
    ⊢ Or (LE.le a b) (LE.le b a)
  -/
  rwa [neg_sub] at t
  /-
    🎉 no goals
  -/


instance preorder : Preorder (ℤ√d) where
  le := (· ≤ ·)
                                      /-
                                        d : Nat
                                        a : Zsqrtd ↑d
                                        ⊢ (HSub.hSub a a).Nonneg
                                      -/
  le_refl a := show Nonneg (a - a) by simp only [sub_self]; trivial
                                                            /-
                                                              🎉 no goals
                                                            -/
                               /-
                                 d : Nat
                                 a b c : Zsqrtd ↑d
                                 hab : LE.le a b
                                 hbc : LE.le b c
                                 ⊢ LE.le a c
                               -/
  le_trans a b c hab hbc := by simpa [sub_add_sub_cancel'] using hab.add hbc
                               /-
                                 🎉 no goals
                               -/
  lt := (· < ·)
  lt_iff_le_not_le _ _ := (and_iff_right_of_imp (Zsqrtd.le_total _ _).resolve_left).symm


open Int in
theorem le_arch (a : ℤ√d) : ∃ n : ℕ, a ≤ n := by
  obtain ⟨x, y, (h : a ≤ ⟨x, y⟩)⟩ : ∃ x y : ℕ, Nonneg (⟨x, y⟩ + -a) :=
    match -a with
    | ⟨Int.ofNat x, Int.ofNat y⟩ => ⟨0, 0, by trivial⟩
    | ⟨Int.ofNat x, -[y+1]⟩ => ⟨0, y + 1, by simp [add_def, Int.negSucc_coe, add_assoc]; trivial⟩
    | ⟨-[x+1], Int.ofNat y⟩ => ⟨x + 1, 0, by simp [Int.negSucc_coe, add_assoc]; trivial⟩
    | ⟨-[x+1], -[y+1]⟩ => ⟨x + 1, y + 1, by simp [Int.negSucc_coe, add_assoc]; trivial⟩
  /-
    case intro.intro
    d : Nat
    a : Zsqrtd ↑d
    x y : Nat
    h : LE.le a { re := ↑x, im := ↑y }
    ⊢ Exists fun n => LE.le a ↑n
  -/
  refine ⟨x + d * y, h.trans ?_⟩
  /-
    case intro.intro
    d : Nat
    a : Zsqrtd ↑d
    x y : Nat
    h : LE.le a { re := ↑x, im := ↑y }
    ⊢ LE.le { re := ↑x, im := ↑y } ↑(HAdd.hAdd x (HMul.hMul d y))
  -/
  change Nonneg ⟨↑x + d * y - ↑x, 0 - ↑y⟩
  /-
    case intro.intro
    d : Nat
    a : Zsqrtd ↑d
    x y : Nat
    h : LE.le a { re := ↑x, im := ↑y }
    ⊢ { re := HSub.hSub (HAdd.hAdd (↑x) (HMul.hMul ↑d ↑y)) ↑x, im := HSub.hSub 0 ↑ …
  -/
  cases' y with y
    /-
      case intro.intro.zero
      d : Nat
      a : Zsqrtd ↑d
      x : Nat
      h : LE.le a { re := ↑x, im := ↑0 }
      ⊢ { re := HSub.hSub (HAdd.hAdd (↑x) (HMul.hMul ↑d ↑0)) ↑x, im := HSub.hSub 0 ↑ …
    -/
  · simp
    /-
      case intro.intro.zero
      d : Nat
      a : Zsqrtd ↑d
      x : Nat
      h : LE.le a { re := ↑x, im := ↑0 }
      ⊢ { re := 0, im := 0 }.Nonneg
    -/
    trivial
    /-
      🎉 no goals
    -/
  have h : ∀ y, SqLe y d (d * y) 1 := fun y => by
    simpa [SqLe, mul_comm, mul_left_comm] using Nat.mul_le_mul_right (y * y) (Nat.le_mul_self d)
  /-
    case intro.intro.succ
    d : Nat
    a : Zsqrtd ↑d
    x y : Nat
    h✝ : LE.le a { re := ↑x, im := ↑(HAdd.hAdd y 1) }
    h : ∀ (y : Nat), Zsqrtd.SqLe y d (HMul.hMul d y) 1
    ⊢ { re := HSub.hSub (HAdd.hAdd (↑x) (HMul.hMul ↑d ↑(HAdd.hAdd y 1))) ↑x, im := …
  -/
  rw [show (x : ℤ) + d * Nat.succ y - x = d * Nat.succ y by simp]
  /-
    case intro.intro.succ
    d : Nat
    a : Zsqrtd ↑d
    x y : Nat
    h✝ : LE.le a { re := ↑x, im := ↑(HAdd.hAdd y 1) }
    h : ∀ (y : Nat), Zsqrtd.SqLe y d (HMul.hMul d y) 1
    ⊢ { re := HMul.hMul ↑d ↑y.succ, im := HSub.hSub 0 ↑(HAdd.hAdd y 1) }.Nonneg
  -/
  exact h (y + 1)
  /-
    🎉 no goals
  -/


protected theorem add_le_add_left (a b : ℤ√d) (ab : a ≤ b) (c : ℤ√d) : c + a ≤ c + b :=
                   /-
                     d : Nat
                     a b : Zsqrtd ↑d
                     ab : LE.le a b
                     c : Zsqrtd ↑d
                     ⊢ (HSub.hSub (HAdd.hAdd c b) (HAdd.hAdd c a)).Nonneg
                   -/
  show Nonneg _ by rw [add_sub_add_left_eq_sub]; exact ab
                                                 /-
                                                   🎉 no goals
                                                 -/


protected theorem le_of_add_le_add_left (a b c : ℤ√d) (h : c + a ≤ c + b) : a ≤ b := by
  /-
    d : Nat
    a b c : Zsqrtd ↑d
    h : LE.le (HAdd.hAdd c a) (HAdd.hAdd c b)
    ⊢ LE.le a b
  -/
  simpa using Zsqrtd.add_le_add_left _ _ h (-c)
  /-
    🎉 no goals
  -/


protected theorem add_lt_add_left (a b : ℤ√d) (h : a < b) (c) : c + a < c + b := fun h' =>
  h (Zsqrtd.le_of_add_le_add_left _ _ _ h')


theorem nonneg_smul {a : ℤ√d} {n : ℕ} (ha : Nonneg a) : Nonneg ((n : ℤ√d) * a) := by
  /-
    d : Nat
    a : Zsqrtd ↑d
    n : Nat
    ha : a.Nonneg
    ⊢ (HMul.hMul (↑n) a).Nonneg
  -/
  rw [← Int.cast_natCast n]
  exact
    match a, nonneg_cases ha, ha with
    | _, ⟨x, y, Or.inl rfl⟩, _ => by rw [smul_val]; trivial
    | _, ⟨x, y, Or.inr <| Or.inl rfl⟩, ha => by
      rw [smul_val]; simpa using nonnegg_pos_neg.2 (sqLe_smul n <| nonnegg_pos_neg.1 ha)
    | _, ⟨x, y, Or.inr <| Or.inr rfl⟩, ha => by
      rw [smul_val]; simpa using nonnegg_neg_pos.2 (sqLe_smul n <| nonnegg_neg_pos.1 ha)


theorem nonneg_muld {a : ℤ√d} (ha : Nonneg a) : Nonneg (sqrtd * a) :=
  match a, nonneg_cases ha, ha with
  | _, ⟨_, _, Or.inl rfl⟩, _ => trivial
  | _, ⟨x, y, Or.inr <| Or.inl rfl⟩, ha => by
    /-
      d : Nat
      a : Zsqrtd ↑d
      ha✝ : a.Nonneg
      x y : Nat
      ha : { re := ↑x, im := Neg.neg ↑y }.Nonneg
      ⊢ (HMul.hMul Zsqrtd.sqrtd { re := ↑x, im := Neg.neg ↑y }).Nonneg
    -/
    simp only [muld_val, mul_neg]
    /-
      d : Nat
      a : Zsqrtd ↑d
      ha✝ : a.Nonneg
      x y : Nat
      ha : { re := ↑x, im := Neg.neg ↑y }.Nonneg
      ⊢ { re := Neg.neg (HMul.hMul ↑d ↑y), im := ↑x }.Nonneg
    -/
    apply nonnegg_neg_pos.2
    /-
      d : Nat
      a : Zsqrtd ↑d
      ha✝ : a.Nonneg
      x y : Nat
      ha : { re := ↑x, im := Neg.neg ↑y }.Nonneg
      ⊢ Zsqrtd.SqLe (HMul.hMul d y) 1 x d
    -/
    simpa [SqLe, mul_comm, mul_left_comm] using Nat.mul_le_mul_left d (nonnegg_pos_neg.1 ha)
    /-
      🎉 no goals
    -/
  | _, ⟨x, y, Or.inr <| Or.inr rfl⟩, ha => by
    /-
      d : Nat
      a : Zsqrtd ↑d
      ha✝ : a.Nonneg
      x y : Nat
      ha : { re := Neg.neg ↑x, im := ↑y }.Nonneg
      ⊢ (HMul.hMul Zsqrtd.sqrtd { re := Neg.neg ↑x, im := ↑y }).Nonneg
    -/
    simp only [muld_val]
    /-
      d : Nat
      a : Zsqrtd ↑d
      ha✝ : a.Nonneg
      x y : Nat
      ha : { re := Neg.neg ↑x, im := ↑y }.Nonneg
      ⊢ { re := HMul.hMul ↑d ↑y, im := Neg.neg ↑x }.Nonneg
    -/
    apply nonnegg_pos_neg.2
    /-
      d : Nat
      a : Zsqrtd ↑d
      ha✝ : a.Nonneg
      x y : Nat
      ha : { re := Neg.neg ↑x, im := ↑y }.Nonneg
      ⊢ Zsqrtd.SqLe x d (HMul.hMul d y) 1
    -/
    simpa [SqLe, mul_comm, mul_left_comm] using Nat.mul_le_mul_left d (nonnegg_neg_pos.1 ha)
    /-
      🎉 no goals
    -/


theorem nonneg_mul_lem {x y : ℕ} {a : ℤ√d} (ha : Nonneg a) : Nonneg (⟨x, y⟩ * a) := by
  have : (⟨x, y⟩ * a : ℤ√d) = (x : ℤ√d) * a + sqrtd * ((y : ℤ√d) * a) := by
    rw [decompose, right_distrib, mul_assoc, Int.cast_natCast, Int.cast_natCast]
  /-
    d x y : Nat
    a : Zsqrtd ↑d
    ha : a.Nonneg
    this : Eq (HMul.hMul { re := ↑x, im := ↑y } a) (HAdd.hAdd (HMul.hMul (↑x) a) ( …
    ⊢ (HMul.hMul { re := ↑x, im := ↑y } a).Nonneg
  -/
  rw [this]
  /-
    d x y : Nat
    a : Zsqrtd ↑d
    ha : a.Nonneg
    this : Eq (HMul.hMul { re := ↑x, im := ↑y } a) (HAdd.hAdd (HMul.hMul (↑x) a) ( …
    ⊢ (HAdd.hAdd (HMul.hMul (↑x) a) (HMul.hMul Zsqrtd.sqrtd (HMul.hMul (↑y) a))).N …
  -/
  exact (nonneg_smul ha).add (nonneg_muld <| nonneg_smul ha)
  /-
    🎉 no goals
  -/


theorem nonneg_mul {a b : ℤ√d} (ha : Nonneg a) (hb : Nonneg b) : Nonneg (a * b) :=
  match a, b, nonneg_cases ha, nonneg_cases hb, ha, hb with
  | _, _, ⟨_, _, Or.inl rfl⟩, ⟨_, _, Or.inl rfl⟩, _, _ => trivial
  | _, _, ⟨x, y, Or.inl rfl⟩, ⟨z, w, Or.inr <| Or.inr rfl⟩, _, hb => nonneg_mul_lem hb
  | _, _, ⟨x, y, Or.inl rfl⟩, ⟨z, w, Or.inr <| Or.inl rfl⟩, _, hb => nonneg_mul_lem hb
  | _, _, ⟨x, y, Or.inr <| Or.inr rfl⟩, ⟨z, w, Or.inl rfl⟩, ha, _ => by
    /-
      d : Nat
      a b : Zsqrtd ↑d
      ha✝ : a.Nonneg
      hb : b.Nonneg
      x y z w : Nat
      ha : { re := Neg.neg ↑x, im := ↑y }.Nonneg
      x✝ : { re := ↑z, im := ↑w }.Nonneg
      ⊢ (HMul.hMul { re := Neg.neg ↑x, im := ↑y } { re := ↑z, im := ↑w }).Nonneg
    -/
    rw [mul_comm]; exact nonneg_mul_lem ha
                   /-
                     🎉 no goals
                   -/
  | _, _, ⟨x, y, Or.inr <| Or.inl rfl⟩, ⟨z, w, Or.inl rfl⟩, ha, _ => by
    /-
      d : Nat
      a b : Zsqrtd ↑d
      ha✝ : a.Nonneg
      hb : b.Nonneg
      x y z w : Nat
      ha : { re := ↑x, im := Neg.neg ↑y }.Nonneg
      x✝ : { re := ↑z, im := ↑w }.Nonneg
      ⊢ (HMul.hMul { re := ↑x, im := Neg.neg ↑y } { re := ↑z, im := ↑w }).Nonneg
    -/
    rw [mul_comm]; exact nonneg_mul_lem ha
                   /-
                     🎉 no goals
                   -/
  | _, _, ⟨x, y, Or.inr <| Or.inr rfl⟩, ⟨z, w, Or.inr <| Or.inr rfl⟩, ha, hb => by
    rw [calc
          (⟨-x, y⟩ * ⟨-z, w⟩ : ℤ√d) = ⟨_, _⟩ := rfl
          _ = ⟨x * z + d * y * w, -(x * w + y * z)⟩ := by simp [add_comm]
          ]
    /-
      d : Nat
      a b : Zsqrtd ↑d
      ha✝ : a.Nonneg
      hb✝ : b.Nonneg
      x y z w : Nat
      ha : { re := Neg.neg ↑x, im := ↑y }.Nonneg
      hb : { re := Neg.neg ↑z, im := ↑w }.Nonneg
      ⊢ { re := HAdd.hAdd (HMul.hMul ↑x ↑z) (HMul.hMul (HMul.hMul ↑d ↑y) ↑w), im :=  …
    -/
    exact nonnegg_pos_neg.2 (sqLe_mul.left (nonnegg_neg_pos.1 ha) (nonnegg_neg_pos.1 hb))
    /-
      🎉 no goals
    -/
  | _, _, ⟨x, y, Or.inr <| Or.inr rfl⟩, ⟨z, w, Or.inr <| Or.inl rfl⟩, ha, hb => by
    rw [calc
          (⟨-x, y⟩ * ⟨z, -w⟩ : ℤ√d) = ⟨_, _⟩ := rfl
          _ = ⟨-(x * z + d * y * w), x * w + y * z⟩ := by simp [add_comm]
          ]
    /-
      d : Nat
      a b : Zsqrtd ↑d
      ha✝ : a.Nonneg
      hb✝ : b.Nonneg
      x y z w : Nat
      ha : { re := Neg.neg ↑x, im := ↑y }.Nonneg
      hb : { re := ↑z, im := Neg.neg ↑w }.Nonneg
      ⊢ { re := Neg.neg (HAdd.hAdd (HMul.hMul ↑x ↑z) (HMul.hMul (HMul.hMul ↑d ↑y) ↑w …
    -/
    exact nonnegg_neg_pos.2 (sqLe_mul.right.left (nonnegg_neg_pos.1 ha) (nonnegg_pos_neg.1 hb))
    /-
      🎉 no goals
    -/
  | _, _, ⟨x, y, Or.inr <| Or.inl rfl⟩, ⟨z, w, Or.inr <| Or.inr rfl⟩, ha, hb => by
    rw [calc
          (⟨x, -y⟩ * ⟨-z, w⟩ : ℤ√d) = ⟨_, _⟩ := rfl
          _ = ⟨-(x * z + d * y * w), x * w + y * z⟩ := by simp [add_comm]
          ]
    exact
        nonnegg_neg_pos.2 (sqLe_mul.right.right.left (nonnegg_pos_neg.1 ha) (nonnegg_neg_pos.1 hb))
  | _, _, ⟨x, y, Or.inr <| Or.inl rfl⟩, ⟨z, w, Or.inr <| Or.inl rfl⟩, ha, hb => by
    rw [calc
          (⟨x, -y⟩ * ⟨z, -w⟩ : ℤ√d) = ⟨_, _⟩ := rfl
          _ = ⟨x * z + d * y * w, -(x * w + y * z)⟩ := by simp [add_comm]
          ]
    exact
        nonnegg_pos_neg.2
          (sqLe_mul.right.right.right (nonnegg_pos_neg.1 ha) (nonnegg_pos_neg.1 hb))


protected theorem mul_nonneg (a b : ℤ√d) : 0 ≤ a → 0 ≤ b → 0 ≤ a * b := by
  /-
    d : Nat
    a b : Zsqrtd ↑d
    ⊢ LE.le 0 a → LE.le 0 b → LE.le 0 (HMul.hMul a b)
  -/
  simp_rw [← nonneg_iff_zero_le]
  /-
    d : Nat
    a b : Zsqrtd ↑d
    ⊢ a.Nonneg → b.Nonneg → (HMul.hMul a b).Nonneg
  -/
  exact nonneg_mul
  /-
    🎉 no goals
  -/


theorem not_sqLe_succ (c d y) (h : 0 < c) : ¬SqLe (y + 1) c 0 d :=
  not_le_of_gt <| mul_pos (mul_pos h <| Nat.succ_pos _) <| Nat.succ_pos _

-- Porting note: renamed field and added theorem to make `x` explicit

/-- A nonsquare is a natural number that is not equal to the square of an
  integer. This is implemented as a typeclass because it's a necessary condition
  for much of the Pell equation theory. -/
class Nonsquare (x : ℕ) : Prop where
  ns' : ∀ n : ℕ, x ≠ n * n


theorem Nonsquare.ns (x : ℕ) [Nonsquare x] : ∀ n : ℕ, x ≠ n * n := ns'


theorem d_pos : 0 < d :=
  lt_of_le_of_ne (Nat.zero_le _) <| Ne.symm <| Nonsquare.ns d 0


theorem divides_sq_eq_zero {x y} (h : x * x = d * y * y) : x = 0 ∧ y = 0 :=
  let g := x.gcd y
  Or.elim g.eq_zero_or_pos
    (fun H => ⟨Nat.eq_zero_of_gcd_eq_zero_left H, Nat.eq_zero_of_gcd_eq_zero_right H⟩) fun gpos =>
    False.elim <| by
      /-
        d : Nat
        dnsq : Zsqrtd.Nonsquare d
        x y : Nat
        h : Eq (HMul.hMul x x) (HMul.hMul (HMul.hMul d y) y)
        g : Nat := x.gcd y
        gpos : GT.gt g 0
        ⊢ False
      -/
      let ⟨m, n, co, (hx : x = m * g), (hy : y = n * g)⟩ := Nat.exists_coprime _ _
      /-
        d : Nat
        dnsq : Zsqrtd.Nonsquare d
        x y : Nat
        h : Eq (HMul.hMul x x) (HMul.hMul (HMul.hMul d y) y)
        g : Nat := x.gcd y
        gpos : GT.gt g 0
        m n : Nat
        co : m.Coprime n
        hx : Eq x (HMul.hMul m g)
        hy : Eq y (HMul.hMul n g)
        ⊢ False
      -/
      rw [hx, hy] at h
      have : m * m = d * (n * n) := by
        refine mul_left_cancel₀ (mul_pos gpos gpos).ne' ?_
        -- Porting note: was `simpa [mul_comm, mul_left_comm] using h`
        calc
          g * g * (m * m)
          _ = m * g * (m * g) := by ring
          _ = d * (n * g) * (n * g) := h
          _ = g * g * (d * (n * n)) := by ring
      have co2 :=
        let co1 := co.mul_right co
        co1.mul co1
      exact
        Nonsquare.ns d m
          (Nat.dvd_antisymm (by rw [this]; apply dvd_mul_right) <|
            co2.dvd_of_dvd_mul_right <| by simp [this])


theorem divides_sq_eq_zero_z {x y : ℤ} (h : x * x = d * y * y) : x = 0 ∧ y = 0 := by
  /-
    d : Nat
    dnsq : Zsqrtd.Nonsquare d
    x y : Int
    h : Eq (HMul.hMul x x) (HMul.hMul (HMul.hMul (↑d) y) y)
    ⊢ And (Eq x 0) (Eq y 0)
  -/
  rw [mul_assoc, ← Int.natAbs_mul_self, ← Int.natAbs_mul_self, ← Int.ofNat_mul, ← mul_assoc] at h
  exact
    let ⟨h1, h2⟩ := divides_sq_eq_zero (Int.ofNat.inj h)
    ⟨Int.natAbs_eq_zero.mp h1, Int.natAbs_eq_zero.mp h2⟩


theorem not_divides_sq (x y) : (x + 1) * (x + 1) ≠ d * (y + 1) * (y + 1) := fun e => by
  /-
    d : Nat
    dnsq : Zsqrtd.Nonsquare d
    x y : Nat
    e : Eq (HMul.hMul (HAdd.hAdd x 1) (HAdd.hAdd x 1)) (HMul.hMul (HMul.hMul d (HA …
    ⊢ False
  -/
  have t := (divides_sq_eq_zero e).left
  /-
    d : Nat
    dnsq : Zsqrtd.Nonsquare d
    x y : Nat
    e : Eq (HMul.hMul (HAdd.hAdd x 1) (HAdd.hAdd x 1)) (HMul.hMul (HMul.hMul d (HA …
    t : Eq (HAdd.hAdd x 1) 0
    ⊢ False
  -/
  contradiction
  /-
    🎉 no goals
  -/


open Int in
theorem nonneg_antisymm : ∀ {a : ℤ√d}, Nonneg a → Nonneg (-a) → a = 0
  | ⟨0, 0⟩, _, _ => rfl
  | ⟨-[_+1], -[_+1]⟩, xy, _ => False.elim xy
  | ⟨(_ + 1 : Nat), (_ + 1 : Nat)⟩, _, yx => False.elim yx
                                                             /-
                                                               d : Nat
                                                               dnsq : Zsqrtd.Nonsquare d
                                                               a✝ : Nat
                                                               xy : { re := Int.negSucc a✝, im := 0 }.Nonneg
                                                               x✝ : (Neg.neg { re := Int.negSucc a✝, im := 0 }).Nonneg
                                                               ⊢ LT.lt 0 1
                                                             -/
  | ⟨-[_+1], 0⟩, xy, _ => absurd xy (not_sqLe_succ _ _ _ (by decide))
                                                             /-
                                                               🎉 no goals
                                                             -/
                                                                    /-
                                                                      d : Nat
                                                                      dnsq : Zsqrtd.Nonsquare d
                                                                      n✝ : Nat
                                                                      x✝ : { re := ↑(HAdd.hAdd n✝ 1), im := 0 }.Nonneg
                                                                      yx : (Neg.neg { re := ↑(HAdd.hAdd n✝ 1), im := 0 }).Nonneg
                                                                      ⊢ LT.lt 0 1
                                                                    -/
  | ⟨(_ + 1 : Nat), 0⟩, _, yx => absurd yx (not_sqLe_succ _ _ _ (by decide))
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  | ⟨0, -[_+1]⟩, xy, _ => absurd xy (not_sqLe_succ _ _ _ d_pos)
  | ⟨0, (_ + 1 : Nat)⟩, _, yx => absurd yx (not_sqLe_succ _ _ _ d_pos)
  | ⟨(x + 1 : Nat), -[y+1]⟩, (xy : SqLe _ _ _ _), (yx : SqLe _ _ _ _) => by
    /-
      d : Nat
      dnsq : Zsqrtd.Nonsquare d
      x y : Nat
      xy : Zsqrtd.SqLe (HAdd.hAdd y 1) d (HAdd.hAdd x 1) 1
      yx : Zsqrtd.SqLe (HAdd.hAdd x 1) 1 y.succ d
      ⊢ Eq { re := ↑(HAdd.hAdd x 1), im := Int.negSucc y } 0
    -/
    let t := le_antisymm yx xy
    /-
      d : Nat
      dnsq : Zsqrtd.Nonsquare d
      x y : Nat
      xy : Zsqrtd.SqLe (HAdd.hAdd y 1) d (HAdd.hAdd x 1) 1
      yx : Zsqrtd.SqLe (HAdd.hAdd x 1) 1 y.succ d
      t : Eq (HMul.hMul (HMul.hMul 1 (HAdd.hAdd x 1)) (HAdd.hAdd x 1)) (HMul.hMul (H …
      ⊢ Eq { re := ↑(HAdd.hAdd x 1), im := Int.negSucc y } 0
    -/
    rw [one_mul] at t
    /-
      d : Nat
      dnsq : Zsqrtd.Nonsquare d
      x y : Nat
      xy : Zsqrtd.SqLe (HAdd.hAdd y 1) d (HAdd.hAdd x 1) 1
      yx : Zsqrtd.SqLe (HAdd.hAdd x 1) 1 y.succ d
      t : Eq (HMul.hMul (HAdd.hAdd x 1) (HAdd.hAdd x 1)) (HMul.hMul (HMul.hMul d y.s …
      ⊢ Eq { re := ↑(HAdd.hAdd x 1), im := Int.negSucc y } 0
    -/
    exact absurd t (not_divides_sq _ _)
    /-
      🎉 no goals
    -/
  | ⟨-[x+1], (y + 1 : Nat)⟩, (xy : SqLe _ _ _ _), (yx : SqLe _ _ _ _) => by
    /-
      d : Nat
      dnsq : Zsqrtd.Nonsquare d
      x y : Nat
      xy : Zsqrtd.SqLe (HAdd.hAdd x 1) 1 (HAdd.hAdd y 1) d
      yx : Zsqrtd.SqLe (HAdd.hAdd y 1) d x.succ 1
      ⊢ Eq { re := Int.negSucc x, im := ↑(HAdd.hAdd y 1) } 0
    -/
    let t := le_antisymm xy yx
    /-
      d : Nat
      dnsq : Zsqrtd.Nonsquare d
      x y : Nat
      xy : Zsqrtd.SqLe (HAdd.hAdd x 1) 1 (HAdd.hAdd y 1) d
      yx : Zsqrtd.SqLe (HAdd.hAdd y 1) d x.succ 1
      t : Eq (HMul.hMul (HMul.hMul 1 (HAdd.hAdd x 1)) (HAdd.hAdd x 1)) (HMul.hMul (H …
      ⊢ Eq { re := Int.negSucc x, im := ↑(HAdd.hAdd y 1) } 0
    -/
    rw [one_mul] at t
    /-
      d : Nat
      dnsq : Zsqrtd.Nonsquare d
      x y : Nat
      xy : Zsqrtd.SqLe (HAdd.hAdd x 1) 1 (HAdd.hAdd y 1) d
      yx : Zsqrtd.SqLe (HAdd.hAdd y 1) d x.succ 1
      t : Eq (HMul.hMul (HAdd.hAdd x 1) (HAdd.hAdd x 1)) (HMul.hMul (HMul.hMul d (HA …
      ⊢ Eq { re := Int.negSucc x, im := ↑(HAdd.hAdd y 1) } 0
    -/
    exact absurd t (not_divides_sq _ _)
    /-
      🎉 no goals
    -/


theorem le_antisymm {a b : ℤ√d} (ab : a ≤ b) (ba : b ≤ a) : a = b :=
                                              /-
                                                d : Nat
                                                dnsq : Zsqrtd.Nonsquare d
                                                a b : Zsqrtd ↑d
                                                ab : LE.le a b
                                                ba : LE.le b a
                                                ⊢ (Neg.neg (HSub.hSub a b)).Nonneg
                                              -/
  eq_of_sub_eq_zero <| nonneg_antisymm ba (by rwa [neg_sub])
                                              /-
                                                🎉 no goals
                                              -/


instance linearOrder : LinearOrder (ℤ√d) :=
  { Zsqrtd.preorder with
    le_antisymm := fun _ _ => Zsqrtd.le_antisymm
    le_total := Zsqrtd.le_total
    decidableLE := Zsqrtd.decidableLE }


protected theorem eq_zero_or_eq_zero_of_mul_eq_zero : ∀ {a b : ℤ√d}, a * b = 0 → a = 0 ∨ b = 0
  | ⟨x, y⟩, ⟨z, w⟩, h => by
    /-
      d : Nat
      dnsq : Zsqrtd.Nonsquare d
      x y z w : Int
      h : Eq (HMul.hMul { re := x, im := y } { re := z, im := w }) 0
      ⊢ Or (Eq { re := x, im := y } 0) (Eq { re := z, im := w } 0)
    -/
    injection h with h1 h2
    /-
      d : Nat
      dnsq : Zsqrtd.Nonsquare d
      x y z w : Int
      h1 : Eq (HAdd.hAdd (HMul.hMul { re := x, im := y }.re { re := z, im := w }.re) …
      h2 : Eq (HAdd.hAdd (HMul.hMul { re := x, im := y }.re { re := z, im := w }.im) …
      ⊢ Or (Eq { re := x, im := y } 0) (Eq { re := z, im := w } 0)
    -/
    have h1 : x * z = -(d * y * w) := eq_neg_of_add_eq_zero_left h1
    /-
      d : Nat
      dnsq : Zsqrtd.Nonsquare d
      x y z w : Int
      h1✝ : Eq (HAdd.hAdd (HMul.hMul { re := x, im := y }.re { re := z, im := w }.re …
      h2 : Eq (HAdd.hAdd (HMul.hMul { re := x, im := y }.re { re := z, im := w }.im) …
      h1 : Eq (HMul.hMul x z) (Neg.neg (HMul.hMul (HMul.hMul (↑d) y) w))
      ⊢ Or (Eq { re := x, im := y } 0) (Eq { re := z, im := w } 0)
    -/
    have h2 : x * w = -(y * z) := eq_neg_of_add_eq_zero_left h2
    have fin : x * x = d * y * y → (⟨x, y⟩ : ℤ√d) = 0 := fun e =>
      match x, y, divides_sq_eq_zero_z e with
      | _, _, ⟨rfl, rfl⟩ => rfl
    exact
      if z0 : z = 0 then
        if w0 : w = 0 then
          Or.inr
            (match z, w, z0, w0 with
            | _, _, rfl, rfl => rfl)
        else
          Or.inl <|
            fin <|
              mul_right_cancel₀ w0 <|
                calc
                  x * x * w = -y * (x * z) := by simp [h2, mul_assoc, mul_left_comm]
                  _ = d * y * y * w := by simp [h1, mul_assoc, mul_left_comm]
      else
        Or.inl <|
          fin <|
            mul_right_cancel₀ z0 <|
              calc
                x * x * z = d * -y * (x * w) := by simp [h1, mul_assoc, mul_left_comm]
                _ = d * y * y * z := by simp [h2, mul_assoc, mul_left_comm]


instance : NoZeroDivisors (ℤ√d) where
  eq_zero_or_eq_zero_of_mul_eq_zero := Zsqrtd.eq_zero_or_eq_zero_of_mul_eq_zero


instance : IsDomain (ℤ√d) :=
  NoZeroDivisors.to_isDomain _


protected theorem mul_pos (a b : ℤ√d) (a0 : 0 < a) (b0 : 0 < b) : 0 < a * b := fun ab =>
  Or.elim
    (eq_zero_or_eq_zero_of_mul_eq_zero
      (le_antisymm ab (Zsqrtd.mul_nonneg _ _ (le_of_lt a0) (le_of_lt b0))))
    (fun e => ne_of_gt a0 e) fun e => ne_of_gt b0 e


instance : LinearOrderedCommRing (ℤ√d) :=
  { Zsqrtd.commRing, Zsqrtd.linearOrder, Zsqrtd.nontrivial with
    add_le_add_left := Zsqrtd.add_le_add_left
    mul_pos := Zsqrtd.mul_pos
                      /-
                        d : Nat
                        dnsq : Zsqrtd.Nonsquare d
                        ⊢ LE.le 0 1
                      -/
    zero_le_one := by trivial }
                      /-
                        🎉 no goals
                      -/


                                         /-
                                           d : Nat
                                           dnsq : Zsqrtd.Nonsquare d
                                           ⊢ LinearOrderedRing (Zsqrtd ↑d)
                                         -/
instance : LinearOrderedRing (ℤ√d) := by infer_instance
                                         /-
                                           🎉 no goals
                                         -/


                                   /-
                                     d : Nat
                                     dnsq : Zsqrtd.Nonsquare d
                                     ⊢ OrderedRing (Zsqrtd ↑d)
                                   -/
instance : OrderedRing (ℤ√d) := by infer_instance
                                   /-
                                     🎉 no goals
                                   -/


theorem norm_eq_zero {d : ℤ} (h_nonsquare : ∀ n : ℤ, d ≠ n * n) (a : ℤ√d) : norm a = 0 ↔ a = 0 := by
  /-
    d : Int
    h_nonsquare : ∀ (n : Int), Ne d (HMul.hMul n n)
    a : Zsqrtd d
    ⊢ Iff (Eq a.norm 0) (Eq a 0)
  -/
  refine ⟨fun ha => Zsqrtd.ext_iff.mpr ?_, fun h => by rw [h, norm_zero]⟩
  /-
    d : Int
    h_nonsquare : ∀ (n : Int), Ne d (HMul.hMul n n)
    a : Zsqrtd d
    ha : Eq a.norm 0
    ⊢ And (Eq a.re (Zsqrtd.re 0)) (Eq a.im (Zsqrtd.im 0))
  -/
  dsimp only [norm] at ha
  /-
    d : Int
    h_nonsquare : ∀ (n : Int), Ne d (HMul.hMul n n)
    a : Zsqrtd d
    ha : Eq (HSub.hSub (HMul.hMul a.re a.re) (HMul.hMul (HMul.hMul d a.im) a.im)) 0
    ⊢ And (Eq a.re (Zsqrtd.re 0)) (Eq a.im (Zsqrtd.im 0))
  -/
  rw [sub_eq_zero] at ha
  /-
    d : Int
    h_nonsquare : ∀ (n : Int), Ne d (HMul.hMul n n)
    a : Zsqrtd d
    ha : Eq (HMul.hMul a.re a.re) (HMul.hMul (HMul.hMul d a.im) a.im)
    ⊢ And (Eq a.re (Zsqrtd.re 0)) (Eq a.im (Zsqrtd.im 0))
  -/
  by_cases h : 0 ≤ d
    /-
      case pos
      d : Int
      h_nonsquare : ∀ (n : Int), Ne d (HMul.hMul n n)
      a : Zsqrtd d
      ha : Eq (HMul.hMul a.re a.re) (HMul.hMul (HMul.hMul d a.im) a.im)
      h : LE.le 0 d
      ⊢ And (Eq a.re (Zsqrtd.re 0)) (Eq a.im (Zsqrtd.im 0))
    -/
  · obtain ⟨d', rfl⟩ := Int.eq_ofNat_of_zero_le h
    /-
      case pos.intro
      d' : Nat
      h_nonsquare : ∀ (n : Int), Ne (↑d') (HMul.hMul n n)
      a : Zsqrtd ↑d'
      ha : Eq (HMul.hMul a.re a.re) (HMul.hMul (HMul.hMul (↑d') a.im) a.im)
      h : LE.le 0 ↑d'
      ⊢ And (Eq a.re (Zsqrtd.re 0)) (Eq a.im (Zsqrtd.im 0))
    -/
    haveI : Nonsquare d' := ⟨fun n h => h_nonsquare n <| mod_cast h⟩
    /-
      case pos.intro
      d' : Nat
      h_nonsquare : ∀ (n : Int), Ne (↑d') (HMul.hMul n n)
      a : Zsqrtd ↑d'
      ha : Eq (HMul.hMul a.re a.re) (HMul.hMul (HMul.hMul (↑d') a.im) a.im)
      h : LE.le 0 ↑d'
      this : Zsqrtd.Nonsquare d'
      ⊢ And (Eq a.re (Zsqrtd.re 0)) (Eq a.im (Zsqrtd.im 0))
    -/
    exact divides_sq_eq_zero_z ha
    /-
      🎉 no goals
    -/
    /-
      case neg
      d : Int
      h_nonsquare : ∀ (n : Int), Ne d (HMul.hMul n n)
      a : Zsqrtd d
      ha : Eq (HMul.hMul a.re a.re) (HMul.hMul (HMul.hMul d a.im) a.im)
      h : Not (LE.le 0 d)
      ⊢ And (Eq a.re (Zsqrtd.re 0)) (Eq a.im (Zsqrtd.im 0))
    -/
  · push_neg at h
    suffices a.re * a.re = 0 by
      rw [eq_zero_of_mul_self_eq_zero this] at ha ⊢
      simpa only [true_and, or_self_right, zero_re, zero_im, eq_self_iff_true, zero_eq_mul,
        mul_zero, mul_eq_zero, h.ne, false_or, or_self_iff] using ha
    /-
      case neg
      d : Int
      h_nonsquare : ∀ (n : Int), Ne d (HMul.hMul n n)
      a : Zsqrtd d
      ha : Eq (HMul.hMul a.re a.re) (HMul.hMul (HMul.hMul d a.im) a.im)
      h : LT.lt d 0
      ⊢ Eq (HMul.hMul a.re a.re) 0
    -/
    apply _root_.le_antisymm _ (mul_self_nonneg _)
    /-
      d : Int
      h_nonsquare : ∀ (n : Int), Ne d (HMul.hMul n n)
      a : Zsqrtd d
      ha : Eq (HMul.hMul a.re a.re) (HMul.hMul (HMul.hMul d a.im) a.im)
      h : LT.lt d 0
      ⊢ LE.le (HMul.hMul a.re a.re) 0
    -/
    rw [ha, mul_assoc]
    /-
      d : Int
      h_nonsquare : ∀ (n : Int), Ne d (HMul.hMul n n)
      a : Zsqrtd d
      ha : Eq (HMul.hMul a.re a.re) (HMul.hMul (HMul.hMul d a.im) a.im)
      h : LT.lt d 0
      ⊢ LE.le (HMul.hMul d (HMul.hMul a.im a.im)) 0
    -/
    exact mul_nonpos_of_nonpos_of_nonneg h.le (mul_self_nonneg _)
    /-
      🎉 no goals
    -/


@[ext]
theorem hom_ext [Ring R] {d : ℤ} (f g : ℤ√d →+* R) (h : f sqrtd = g sqrtd) : f = g := by
  /-
    R : Type
    inst✝ : Ring R
    d : Int
    f g : RingHom (Zsqrtd d) R
    h : Eq (f Zsqrtd.sqrtd) (g Zsqrtd.sqrtd)
    ⊢ Eq f g
  -/
  ext ⟨x_re, x_im⟩
  /-
    case a.mk
    R : Type
    inst✝ : Ring R
    d : Int
    f g : RingHom (Zsqrtd d) R
    h : Eq (f Zsqrtd.sqrtd) (g Zsqrtd.sqrtd)
    x_re x_im : Int
    ⊢ Eq (f { re := x_re, im := x_im }) (g { re := x_re, im := x_im })
  -/
  simp [decompose, h]
  /-
    🎉 no goals
  -/


/-- The unique `RingHom` from `ℤ√d` to a ring `R`, constructed by replacing `√d` with the provided
root. Conversely, this associates to every mapping `ℤ√d →+* R` a value of `√d` in `R`. -/
@[simps]
def lift {d : ℤ} : { r : R // r * r = ↑d } ≃ (ℤ√d →+* R) where
  toFun r :=
    { toFun := fun a => a.1 + a.2 * (r : R)
                      /-
                        R : Type
                        inst✝ : CommRing R
                        d : Int
                        r : Subtype fun r => Eq (HMul.hMul r r) ↑d
                        ⊢ Eq ((↑{ toFun := fun a => HAdd.hAdd (↑a.re) (HMul.hMul ↑a.im ↑r), map_one' : …
                      -/
      map_zero' := by simp
                      /-
                        🎉 no goals
                      -/
      map_add' := fun a b => by
                     /-
                       R : Type
                       inst✝ : CommRing R
                       d : Int
                       r : Subtype fun r => Eq (HMul.hMul r r) ↑d
                       ⊢ Eq ((fun a => HAdd.hAdd (↑a.re) (HMul.hMul ↑a.im ↑r)) 1) 1
                     -/
        /-
          R : Type
          inst✝ : CommRing R
          d : Int
          r : Subtype fun r => Eq (HMul.hMul r r) ↑d
          a b : Zsqrtd d
          ⊢ Eq ((↑{ toFun := fun a => HAdd.hAdd (↑a.re) (HMul.hMul ↑a.im ↑r), map_one' : …
        -/
                     /-
                       🎉 no goals
                     -/
        simp only [add_re, Int.cast_add, add_im]
        /-
          R : Type
          inst✝ : CommRing R
          d : Int
          r : Subtype fun r => Eq (HMul.hMul r r) ↑d
          a b : Zsqrtd d
          ⊢ Eq (HAdd.hAdd (HAdd.hAdd ↑a.re ↑b.re) (HMul.hMul (HAdd.hAdd ↑a.im ↑b.im) ↑r) …
        -/
        ring
        /-
          🎉 no goals
        -/
      map_one' := by simp
        /-
          R : Type
          inst✝ : CommRing R
          d : Int
          r : Subtype fun r => Eq (HMul.hMul r r) ↑d
          a b : Zsqrtd d
          this : Eq (HMul.hMul (HAdd.hAdd (↑a.re) (HMul.hMul ↑a.im ↑r)) (HAdd.hAdd (↑b.r …
          ⊢ Eq ({ toFun := fun a => HAdd.hAdd (↑a.re) (HMul.hMul ↑a.im ↑r), map_one' :=  …
        -/
      map_mul' := fun a b => by
        /-
          R : Type
          inst✝ : CommRing R
          d : Int
          r : Subtype fun r => Eq (HMul.hMul r r) ↑d
          a b : Zsqrtd d
          this : Eq (HMul.hMul (HAdd.hAdd (↑a.re) (HMul.hMul ↑a.im ↑r)) (HAdd.hAdd (↑b.r …
          ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul ↑a.re ↑b.re) (HMul.hMul (HMul.hMul ↑d ↑a …
        -/
        have :
        /-
          🎉 no goals
        -/
          (a.re + a.im * r : R) * (b.re + b.im * r) =
            a.re * b.re + (a.re * b.im + a.im * b.re) * r + a.im * b.im * (r * r) := by
          ring
        simp only [mul_re, Int.cast_add, Int.cast_mul, mul_im, this, r.prop]
        ring }
                           /-
                             R : Type
                             inst✝ : CommRing R
                             d : Int
                             f : RingHom (Zsqrtd d) R
                             ⊢ Eq (HMul.hMul (f Zsqrtd.sqrtd) (f Zsqrtd.sqrtd)) ↑d
                           -/
  invFun f := ⟨f sqrtd, by rw [← f.map_mul, dmuld, map_intCast]⟩
                           /-
                             🎉 no goals
                           -/
  left_inv r := by
    /-
      R : Type
      inst✝ : CommRing R
      d : Int
      r : Subtype fun r => Eq (HMul.hMul r r) ↑d
      ⊢ Eq ((fun f => ⟨f Zsqrtd.sqrtd, ⋯⟩) ((fun r => { toFun := fun a => HAdd.hAdd  …
    -/
    ext
    /-
      case a
      R : Type
      inst✝ : CommRing R
      d : Int
      r : Subtype fun r => Eq (HMul.hMul r r) ↑d
      ⊢ Eq ↑((fun f => ⟨f Zsqrtd.sqrtd, ⋯⟩) ((fun r => { toFun := fun a => HAdd.hAdd …
    -/
    simp
    /-
      🎉 no goals
    -/
  right_inv f := by
    /-
      R : Type
      inst✝ : CommRing R
      d : Int
      f : RingHom (Zsqrtd d) R
      ⊢ Eq ((fun r => { toFun := fun a => HAdd.hAdd (↑a.re) (HMul.hMul ↑a.im ↑r), ma …
    -/
    ext
    /-
      case h
      R : Type
      inst✝ : CommRing R
      d : Int
      f : RingHom (Zsqrtd d) R
      ⊢ Eq (((fun r => { toFun := fun a => HAdd.hAdd (↑a.re) (HMul.hMul ↑a.im ↑r), m …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- `lift r` is injective if `d` is non-square, and R has characteristic zero (that is, the map from
`ℤ` into `R` is injective). -/
theorem lift_injective [CharZero R] {d : ℤ} (r : { r : R // r * r = ↑d })
    (hd : ∀ n : ℤ, d ≠ n * n) : Function.Injective (lift r) :=
  (injective_iff_map_eq_zero (lift r)).mpr fun a ha => by
    /-
      R : Type
      inst✝¹ : CommRing R
      inst✝ : CharZero R
      d : Int
      r : Subtype fun r => Eq (HMul.hMul r r) ↑d
      hd : ∀ (n : Int), Ne d (HMul.hMul n n)
      a : Zsqrtd d
      ha : Eq ((Zsqrtd.lift r) a) 0
      ⊢ Eq a 0
    -/
    have h_inj : Function.Injective ((↑) : ℤ → R) := Int.cast_injective
    suffices lift r a.norm = 0 by
      simp only [intCast_re, add_zero, lift_apply_apply, intCast_im, Int.cast_zero,
        zero_mul] at this
      rwa [← Int.cast_zero, h_inj.eq_iff, norm_eq_zero hd] at this
    /-
      R : Type
      inst✝¹ : CommRing R
      inst✝ : CharZero R
      d : Int
      r : Subtype fun r => Eq (HMul.hMul r r) ↑d
      hd : ∀ (n : Int), Ne d (HMul.hMul n n)
      a : Zsqrtd d
      ha : Eq ((Zsqrtd.lift r) a) 0
      h_inj : Function.Injective Int.cast
      ⊢ Eq ((Zsqrtd.lift r) ↑a.norm) 0
    -/
    rw [norm_eq_mul_conj, RingHom.map_mul, ha, zero_mul]
    /-
      🎉 no goals
    -/


/-- An element of `ℤ√d` has norm equal to `1` if and only if it is contained in the submonoid
of unitary elements. -/
theorem norm_eq_one_iff_mem_unitary {d : ℤ} {a : ℤ√d} : a.norm = 1 ↔ a ∈ unitary (ℤ√d) := by
  /-
    d : Int
    a : Zsqrtd d
    ⊢ Iff (Eq a.norm 1) (Membership.mem (unitary (Zsqrtd d)) a)
  -/
  rw [unitary.mem_iff_self_mul_star, ← norm_eq_mul_conj]
  /-
    d : Int
    a : Zsqrtd d
    ⊢ Iff (Eq a.norm 1) (Eq (↑a.norm) 1)
  -/
  norm_cast
  /-
    🎉 no goals
  -/


/-- The kernel of the norm map on `ℤ√d` equals the submonoid of unitary elements. -/
theorem mker_norm_eq_unitary {d : ℤ} : MonoidHom.mker (@normMonoidHom d) = unitary (ℤ√d) :=
  Submonoid.ext fun _ => norm_eq_one_iff_mem_unitary



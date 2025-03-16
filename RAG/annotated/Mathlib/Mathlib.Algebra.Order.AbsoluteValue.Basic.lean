/-- `AbsoluteValue R S` is the type of absolute values on `R` mapping to `S`:
the maps that preserve `*`, are nonnegative, positive definite and satisfy the triangle equality. -/
structure AbsoluteValue (R S : Type*) [Semiring R] [OrderedSemiring S] extends R →ₙ* S where
  /-- The absolute value is nonnegative -/
  nonneg' : ∀ x, 0 ≤ toFun x
  /-- The absolute value is positive definitive -/
  eq_zero' : ∀ x, toFun x = 0 ↔ x = 0
  /-- The absolute value satisfies the triangle inequality -/
  add_le' : ∀ x y, toFun (x + y) ≤ toFun x + toFun y


instance funLike : FunLike (AbsoluteValue R S) R S where
  coe f := f.toFun
                             /-
                               ι : Type u_1
                               α : Type u_2
                               R✝ : Type u_3
                               S✝ : Type u_4
                               R : Type u_5
                               S : Type u_6
                               inst✝¹ : Semiring R
                               inst✝ : OrderedSemiring S
                               abv f g : AbsoluteValue R S
                               h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by obtain ⟨⟨_, _⟩, _⟩ := f; obtain ⟨⟨_, _⟩, _⟩ := g; congr
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


instance zeroHomClass : ZeroHomClass (AbsoluteValue R S) R S where
  map_zero f := (f.eq_zero' _).2 rfl


instance mulHomClass : MulHomClass (AbsoluteValue R S) R S :=
  { AbsoluteValue.zeroHomClass (R := R) (S := S) with map_mul := fun f => f.map_mul' }


instance nonnegHomClass : NonnegHomClass (AbsoluteValue R S) R S :=
  { AbsoluteValue.zeroHomClass (R := R) (S := S) with apply_nonneg := fun f => f.nonneg' }


instance subadditiveHomClass : SubadditiveHomClass (AbsoluteValue R S) R S :=
  { AbsoluteValue.zeroHomClass (R := R) (S := S) with map_add_le_add := fun f => f.add_le' }


@[simp]
theorem coe_mk (f : R →ₙ* S) {h₁ h₂ h₃} : (AbsoluteValue.mk f h₁ h₂ h₃ : R → S) = f :=
  rfl


@[ext]
theorem ext ⦃f g : AbsoluteValue R S⦄ : (∀ x, f x = g x) → f = g :=
  DFunLike.ext _ _


/-- See Note [custom simps projection]. -/
def Simps.apply (f : AbsoluteValue R S) : R → S :=
  f


@[simp]
theorem coe_toMulHom : ⇑abv.toMulHom = abv :=
  rfl


@[bound]
protected theorem nonneg (x : R) : 0 ≤ abv x :=
  abv.nonneg' x


@[simp]
protected theorem eq_zero {x : R} : abv x = 0 ↔ x = 0 :=
  abv.eq_zero' x


@[bound]
protected theorem add_le (x y : R) : abv (x + y) ≤ abv x + abv y :=
  abv.add_le' x y


/-- The triangle inequality for an `AbsoluteValue` applied to a list. -/
lemma listSum_le (l : List R) : abv l.sum ≤ (l.map abv).sum := by
  induction l with
  | nil => simp
  | cons head tail ih => exact (abv.add_le ..).trans <| add_le_add_left ih (abv head)


@[simp]
protected theorem map_mul (x y : R) : abv (x * y) = abv x * abv y :=
  abv.map_mul' x y


protected theorem ne_zero_iff {x : R} : abv x ≠ 0 ↔ x ≠ 0 :=
  abv.eq_zero.not


protected theorem pos {x : R} (hx : x ≠ 0) : 0 < abv x :=
  lt_of_le_of_ne (abv.nonneg x) (Ne.symm <| mt abv.eq_zero.mp hx)


@[simp]
protected theorem pos_iff {x : R} : 0 < abv x ↔ x ≠ 0 :=
  ⟨fun h₁ => mt abv.eq_zero.mpr h₁.ne', abv.pos⟩


protected theorem ne_zero {x : R} (hx : x ≠ 0) : abv x ≠ 0 :=
  (abv.pos hx).ne'


theorem map_one_of_isLeftRegular (h : IsLeftRegular (abv 1)) : abv 1 = 1 :=
          /-
            R : Type u_5
            S : Type u_6
            inst✝¹ : Semiring R
            inst✝ : OrderedSemiring S
            abv : AbsoluteValue R S
            h : IsLeftRegular (abv 1)
            ⊢ Eq ((fun x => HMul.hMul (abv 1) x) (abv 1)) ((fun x => HMul.hMul (abv 1) x) 1)
          -/
  h <| by simp [← abv.map_mul]
          /-
            🎉 no goals
          -/


@[simp]
protected theorem map_zero : abv 0 = 0 :=
  abv.eq_zero.2 rfl


protected theorem sub_le (a b c : R) : abv (a - c) ≤ abv (a - b) + abv (b - c) := by
  /-
    R : Type u_5
    S : Type u_6
    inst✝¹ : Ring R
    inst✝ : OrderedSemiring S
    abv : AbsoluteValue R S
    a b c : R
    ⊢ LE.le (abv (HSub.hSub a c)) (HAdd.hAdd (abv (HSub.hSub a b)) (abv (HSub.hSub …
  -/
  simpa [sub_eq_add_neg, add_assoc] using abv.add_le (a - b) (b - c)
  /-
    🎉 no goals
  -/


@[simp high] -- Porting note: added `high` to apply it before `AbsoluteValue.eq_zero`
theorem map_sub_eq_zero_iff (a b : R) : abv (a - b) = 0 ↔ a = b :=
  abv.eq_zero.trans sub_eq_zero


@[simp]
protected theorem map_one : abv 1 = 1 :=
  abv.map_one_of_isLeftRegular (isRegular_of_ne_zero <| abv.ne_zero one_ne_zero).left


instance monoidWithZeroHomClass : MonoidWithZeroHomClass (AbsoluteValue R S) R S :=
  { AbsoluteValue.mulHomClass with
    map_zero := fun f => f.map_zero
    map_one := fun f => f.map_one }


/-- Absolute values from a nontrivial `R` to a linear ordered ring preserve `*`, `0` and `1`. -/
def toMonoidWithZeroHom : R →*₀ S :=
  abv


@[simp]
theorem coe_toMonoidWithZeroHom : ⇑abv.toMonoidWithZeroHom = abv :=
  rfl


/-- Absolute values from a nontrivial `R` to a linear ordered ring preserve `*` and `1`. -/
def toMonoidHom : R →* S :=
  abv


@[simp]
theorem coe_toMonoidHom : ⇑abv.toMonoidHom = abv :=
  rfl


@[simp]
protected theorem map_pow (a : R) (n : ℕ) : abv (a ^ n) = abv a ^ n :=
  abv.toMonoidHom.map_pow a n


omit [Nontrivial R] in
/-- An absolute value satisfies `f (n : R) ≤ n` for every `n : ℕ`. -/
lemma apply_nat_le_self (n : ℕ) : abv n ≤ n := by
  /-
    R : Type u_5
    S : Type u_6
    inst✝² : Semiring R
    inst✝¹ : OrderedRing S
    abv : AbsoluteValue R S
    inst✝ : IsDomain S
    n : Nat
    ⊢ LE.le (abv ↑n) ↑n
  -/
  cases subsingleton_or_nontrivial R
    /-
      case inl
      R : Type u_5
      S : Type u_6
      inst✝² : Semiring R
      inst✝¹ : OrderedRing S
      abv : AbsoluteValue R S
      inst✝ : IsDomain S
      n : Nat
      h✝ : Subsingleton R
      ⊢ LE.le (abv ↑n) ↑n
    -/
  · simp [Subsingleton.eq_zero (n : R)]
    /-
      🎉 no goals
    -/
  induction n with
  | zero => simp
  | succ n hn =>
    simp only [Nat.cast_succ]
    calc
      abv (n + 1) ≤ abv n + abv 1 := abv.add_le ..
      _ = abv n + 1 := congrArg (abv n + ·) abv.map_one
      _ ≤ n + 1 := add_le_add_right hn 1


@[bound]
protected theorem le_sub (a b : R) : abv a - abv b ≤ abv (a - b) :=
                            /-
                              R : Type u_5
                              S : Type u_6
                              inst✝¹ : Ring R
                              inst✝ : OrderedRing S
                              abv : AbsoluteValue R S
                              a b : R
                              ⊢ LE.le (abv a) (HAdd.hAdd (abv (HSub.hSub a b)) (abv b))
                            -/
  sub_le_iff_le_add.2 <| by simpa using abv.add_le (a - b) b
                            /-
                              🎉 no goals
                            -/


@[simp]
protected theorem map_neg (a : R) : abv (-a) = abv a := by
  /-
    R : Type u_3
    S : Type u_4
    inst✝² : OrderedCommRing S
    inst✝¹ : Ring R
    abv : AbsoluteValue R S
    inst✝ : NoZeroDivisors S
    a : R
    ⊢ Eq (abv (Neg.neg a)) (abv a)
  -/
  by_cases ha : a = 0; · simp [ha]
                         /-
                           🎉 no goals
                         -/
  refine
    (mul_self_eq_mul_self_iff.mp (by rw [← abv.map_mul, neg_mul_neg, abv.map_mul])).resolve_right ?_
  /-
    case neg
    R : Type u_3
    S : Type u_4
    inst✝² : OrderedCommRing S
    inst✝¹ : Ring R
    abv : AbsoluteValue R S
    inst✝ : NoZeroDivisors S
    a : R
    ha : Not (Eq a 0)
    ⊢ Not (Eq (abv (Neg.neg a)) (Neg.neg (abv a)))
  -/
  exact ((neg_lt_zero.mpr (abv.pos ha)).trans (abv.pos (neg_ne_zero.mpr ha))).ne'
  /-
    🎉 no goals
  -/


                                                                      /-
                                                                        R : Type u_3
                                                                        S : Type u_4
                                                                        inst✝² : OrderedCommRing S
                                                                        inst✝¹ : Ring R
                                                                        abv : AbsoluteValue R S
                                                                        inst✝ : NoZeroDivisors S
                                                                        a b : R
                                                                        ⊢ Eq (abv (HSub.hSub a b)) (abv (HSub.hSub b a))
                                                                      -/
protected theorem map_sub (a b : R) : abv (a - b) = abv (b - a) := by rw [← neg_sub, abv.map_neg]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


/-- Bound `abv (a + b)` from below -/
@[bound]
protected theorem le_add (a b : R) : abv a - abv b ≤ abv (a + b) := by
  /-
    R : Type u_3
    S : Type u_4
    inst✝² : OrderedCommRing S
    inst✝¹ : Ring R
    abv : AbsoluteValue R S
    inst✝ : NoZeroDivisors S
    a b : R
    ⊢ LE.le (HSub.hSub (abv a) (abv b)) (abv (HAdd.hAdd a b))
  -/
  simpa only [tsub_le_iff_right, add_neg_cancel_right, abv.map_neg] using abv.add_le (a + b) (-b)
  /-
    🎉 no goals
  -/


/-- Bound `abv (a - b)` from above -/
@[bound]
lemma sub_le_add (a b : R) : abv (a - b) ≤ abv a + abv b := by
  /-
    R : Type u_3
    S : Type u_4
    inst✝² : OrderedCommRing S
    inst✝¹ : Ring R
    abv : AbsoluteValue R S
    inst✝ : NoZeroDivisors S
    a b : R
    ⊢ LE.le (abv (HSub.hSub a b)) (HAdd.hAdd (abv a) (abv b))
  -/
  simpa only [← sub_eq_add_neg, AbsoluteValue.map_neg] using abv.add_le a (-b)
  /-
    🎉 no goals
  -/


instance [Nontrivial R] [IsDomain S] : MulRingNormClass (AbsoluteValue R S) R S :=
  { AbsoluteValue.subadditiveHomClass,
    AbsoluteValue.monoidWithZeroHomClass with
    map_neg_eq_map := fun f => f.map_neg
    eq_zero_of_map_eq_zero := fun f _ => f.eq_zero.1 }


open Int in
lemma apply_natAbs_eq (x : ℤ) : abv (natAbs x) = abv x := by
  /-
    R : Type u_3
    S : Type u_4
    inst✝² : OrderedCommRing S
    inst✝¹ : Ring R
    abv : AbsoluteValue R S
    inst✝ : NoZeroDivisors S
    x : Int
    ⊢ Eq (abv ↑x.natAbs) (abv ↑x)
  -/
                                               /-
                                                 🎉 no goals
                                               -/
  obtain ⟨_, rfl | rfl⟩ := eq_nat_or_neg x <;> simp
                                               /-
                                                 🎉 no goals
                                               -/


open Int in
/-- Values of an absolute value coincide on the image of `ℕ` in `R`
if and only if they coincide on the image of `ℤ` in `R`. -/
lemma eq_on_nat_iff_eq_on_int {f g : AbsoluteValue R S} :
    (∀ n : ℕ , f n = g n) ↔ ∀ n : ℤ , f n = g n := by
  /-
    R : Type u_3
    S : Type u_4
    inst✝² : OrderedCommRing S
    inst✝¹ : Ring R
    inst✝ : NoZeroDivisors S
    f g : AbsoluteValue R S
    ⊢ Iff (∀ (n : Nat), Eq (f ↑n) (g ↑n)) (∀ (n : Int), Eq (f ↑n) (g ↑n))
  -/
  refine ⟨fun h z ↦ ?_, fun a n ↦ mod_cast a n⟩
  /-
    R : Type u_3
    S : Type u_4
    inst✝² : OrderedCommRing S
    inst✝¹ : Ring R
    inst✝ : NoZeroDivisors S
    f g : AbsoluteValue R S
    h : ∀ (n : Nat), Eq (f ↑n) (g ↑n)
    z : Int
    ⊢ Eq (f ↑z) (g ↑z)
  -/
                                                /-
                                                  🎉 no goals
                                                -/
  obtain ⟨n , rfl | rfl⟩ := eq_nat_or_neg z <;> simp [h n]
                                                /-
                                                  🎉 no goals
                                                -/


/-- `AbsoluteValue.abs` is `abs` as a bundled `AbsoluteValue`. -/
@[simps]
protected def abs : AbsoluteValue S S where
  toFun := abs
  nonneg' := abs_nonneg
  eq_zero' _ := abs_eq_zero
  add_le' := abs_add
  map_mul' := abs_mul


instance : Inhabited (AbsoluteValue S S) :=
  ⟨AbsoluteValue.abs⟩


@[bound]
theorem abs_abv_sub_le_abv_sub (a b : R) : abs (abv a - abv b) ≤ abv (a - b) :=
                                       /-
                                         R : Type u_5
                                         S : Type u_6
                                         inst✝¹ : Ring R
                                         inst✝ : LinearOrderedCommRing S
                                         abv : AbsoluteValue R S
                                         a b : R
                                         ⊢ LE.le (HSub.hSub (abv b) (abv a)) (abv (HSub.hSub a b))
                                       -/
  abs_sub_le_iff.2 ⟨abv.le_sub _ _, by rw [abv.map_sub]; apply abv.le_sub⟩
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- The *trivial* absolute value takes the value `1` on all nonzero elements. -/
protected
def trivial: AbsoluteValue R S where
  toFun x := if x = 0 then 0 else 1
  map_mul' x y := by
    /-
      ι : Type u_1
      α : Type u_2
      R✝ : Type u_3
      S✝ : Type u_4
      R : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : DecidablePred fun x => Eq x 0
      inst✝² : NoZeroDivisors R
      S : Type u_6
      inst✝¹ : OrderedSemiring S
      inst✝ : Nontrivial S
      x y : R
      ⊢ Eq ((fun x => ite (Eq x 0) 0 1) (HMul.hMul x y)) (HMul.hMul ((fun x => ite ( …
    -/
    rcases eq_or_ne x 0 with rfl | hx
      /-
        case inl
        ι : Type u_1
        α : Type u_2
        R✝ : Type u_3
        S✝ : Type u_4
        R : Type u_5
        inst✝⁴ : Semiring R
        inst✝³ : DecidablePred fun x => Eq x 0
        inst✝² : NoZeroDivisors R
        S : Type u_6
        inst✝¹ : OrderedSemiring S
        inst✝ : Nontrivial S
        y : R
        ⊢ Eq ((fun x => ite (Eq x 0) 0 1) (HMul.hMul 0 y)) (HMul.hMul ((fun x => ite ( …
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case inr
      ι : Type u_1
      α : Type u_2
      R✝ : Type u_3
      S✝ : Type u_4
      R : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : DecidablePred fun x => Eq x 0
      inst✝² : NoZeroDivisors R
      S : Type u_6
      inst✝¹ : OrderedSemiring S
      inst✝ : Nontrivial S
      x y : R
      hx : Ne x 0
      ⊢ Eq ((fun x => ite (Eq x 0) 0 1) (HMul.hMul x y)) (HMul.hMul ((fun x => ite ( …
    -/
    rcases eq_or_ne y 0 with rfl | hy
      /-
        case inr.inl
        ι : Type u_1
        α : Type u_2
        R✝ : Type u_3
        S✝ : Type u_4
        R : Type u_5
        inst✝⁴ : Semiring R
        inst✝³ : DecidablePred fun x => Eq x 0
        inst✝² : NoZeroDivisors R
        S : Type u_6
        inst✝¹ : OrderedSemiring S
        inst✝ : Nontrivial S
        x : R
        hx : Ne x 0
        ⊢ Eq ((fun x => ite (Eq x 0) 0 1) (HMul.hMul x 0)) (HMul.hMul ((fun x => ite ( …
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case inr.inr
      ι : Type u_1
      α : Type u_2
      R✝ : Type u_3
      S✝ : Type u_4
      R : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : DecidablePred fun x => Eq x 0
      inst✝² : NoZeroDivisors R
      S : Type u_6
      inst✝¹ : OrderedSemiring S
      inst✝ : Nontrivial S
      x y : R
      hx : Ne x 0
      hy : Ne y 0
      ⊢ Eq ((fun x => ite (Eq x 0) 0 1) (HMul.hMul x y)) (HMul.hMul ((fun x => ite ( …
    -/
    simp [hx, hy]
    /-
      🎉 no goals
    -/
                  /-
                    ι : Type u_1
                    α : Type u_2
                    R✝ : Type u_3
                    S✝ : Type u_4
                    R : Type u_5
                    inst✝⁴ : Semiring R
                    inst✝³ : DecidablePred fun x => Eq x 0
                    inst✝² : NoZeroDivisors R
                    S : Type u_6
                    inst✝¹ : OrderedSemiring S
                    inst✝ : Nontrivial S
                    x : R
                    ⊢ LE.le 0 ({ toFun := fun x => ite (Eq x 0) 0 1, map_mul' := ⋯ }.toFun x)
                  -/
                                                       /-
                                                         🎉 no goals
                                                       -/
  nonneg' x := by rcases eq_or_ne x 0 with hx | hx <;> simp [hx]
                                                       /-
                                                         🎉 no goals
                                                       -/
                   /-
                     ι : Type u_1
                     α : Type u_2
                     R✝ : Type u_3
                     S✝ : Type u_4
                     R : Type u_5
                     inst✝⁴ : Semiring R
                     inst✝³ : DecidablePred fun x => Eq x 0
                     inst✝² : NoZeroDivisors R
                     S : Type u_6
                     inst✝¹ : OrderedSemiring S
                     inst✝ : Nontrivial S
                     x : R
                     ⊢ Iff (Eq ({ toFun := fun x => ite (Eq x 0) 0 1, map_mul' := ⋯ }.toFun x) 0) ( …
                   -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  eq_zero' x := by rcases eq_or_ne x 0 with hx | hx <;> simp [hx]
                                                        /-
                                                          🎉 no goals
                                                        -/
  add_le' x y := by
    /-
      ι : Type u_1
      α : Type u_2
      R✝ : Type u_3
      S✝ : Type u_4
      R : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : DecidablePred fun x => Eq x 0
      inst✝² : NoZeroDivisors R
      S : Type u_6
      inst✝¹ : OrderedSemiring S
      inst✝ : Nontrivial S
      x y : R
      ⊢ LE.le ({ toFun := fun x => ite (Eq x 0) 0 1, map_mul' := ⋯ }.toFun (HAdd.hAd …
    -/
    rcases eq_or_ne x 0 with rfl | hx
      /-
        case inl
        ι : Type u_1
        α : Type u_2
        R✝ : Type u_3
        S✝ : Type u_4
        R : Type u_5
        inst✝⁴ : Semiring R
        inst✝³ : DecidablePred fun x => Eq x 0
        inst✝² : NoZeroDivisors R
        S : Type u_6
        inst✝¹ : OrderedSemiring S
        inst✝ : Nontrivial S
        y : R
        ⊢ LE.le ({ toFun := fun x => ite (Eq x 0) 0 1, map_mul' := ⋯ }.toFun (HAdd.hAd …
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case inr
      ι : Type u_1
      α : Type u_2
      R✝ : Type u_3
      S✝ : Type u_4
      R : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : DecidablePred fun x => Eq x 0
      inst✝² : NoZeroDivisors R
      S : Type u_6
      inst✝¹ : OrderedSemiring S
      inst✝ : Nontrivial S
      x y : R
      hx : Ne x 0
      ⊢ LE.le ({ toFun := fun x => ite (Eq x 0) 0 1, map_mul' := ⋯ }.toFun (HAdd.hAd …
    -/
    rcases eq_or_ne y 0 with rfl | hy
      /-
        case inr.inl
        ι : Type u_1
        α : Type u_2
        R✝ : Type u_3
        S✝ : Type u_4
        R : Type u_5
        inst✝⁴ : Semiring R
        inst✝³ : DecidablePred fun x => Eq x 0
        inst✝² : NoZeroDivisors R
        S : Type u_6
        inst✝¹ : OrderedSemiring S
        inst✝ : Nontrivial S
        x : R
        hx : Ne x 0
        ⊢ LE.le ({ toFun := fun x => ite (Eq x 0) 0 1, map_mul' := ⋯ }.toFun (HAdd.hAd …
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case inr.inr
      ι : Type u_1
      α : Type u_2
      R✝ : Type u_3
      S✝ : Type u_4
      R : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : DecidablePred fun x => Eq x 0
      inst✝² : NoZeroDivisors R
      S : Type u_6
      inst✝¹ : OrderedSemiring S
      inst✝ : Nontrivial S
      x y : R
      hx : Ne x 0
      hy : Ne y 0
      ⊢ LE.le ({ toFun := fun x => ite (Eq x 0) 0 1, map_mul' := ⋯ }.toFun (HAdd.hAd …
    -/
    simp only [hx, ↓reduceIte, hy, one_add_one_eq_two]
    /-
      case inr.inr
      ι : Type u_1
      α : Type u_2
      R✝ : Type u_3
      S✝ : Type u_4
      R : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : DecidablePred fun x => Eq x 0
      inst✝² : NoZeroDivisors R
      S : Type u_6
      inst✝¹ : OrderedSemiring S
      inst✝ : Nontrivial S
      x y : R
      hx : Ne x 0
      hy : Ne y 0
      ⊢ LE.le (ite (Eq (HAdd.hAdd x y) 0) 0 1) 2
    -/
                                                 /-
                                                   🎉 no goals
                                                 -/
    rcases eq_or_ne (x + y) 0 with hxy | hxy <;> simp [hxy, one_le_two]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
lemma trivial_apply {x : R} (hx : x ≠ 0) : AbsoluteValue.trivial (S := S) x = 1 :=
  if_neg hx


/-- A function `f` is an absolute value if it is nonnegative, zero only at 0, additive, and
multiplicative.

See also the type `AbsoluteValue` which represents a bundled version of absolute values.
-/
class IsAbsoluteValue {S} [OrderedSemiring S] {R} [Semiring R] (f : R → S) : Prop where
  /-- The absolute value is nonnegative -/
  abv_nonneg' : ∀ x, 0 ≤ f x
  /-- The absolute value is positive definitive -/
  abv_eq_zero' : ∀ {x}, f x = 0 ↔ x = 0
  /-- The absolute value satisfies the triangle inequality -/
  abv_add' : ∀ x y, f (x + y) ≤ f x + f y
  /-- The absolute value is multiplicative -/
  abv_mul' : ∀ x y, f (x * y) = f x * f y


lemma abv_nonneg (x) : 0 ≤ abv x := abv_nonneg' x


open Lean Meta Mathlib Meta Positivity Qq in
/-- The `positivity` extension which identifies expressions of the form `abv a`. -/
@[positivity _]
def Mathlib.Meta.Positivity.evalAbv : PositivityExt where eval {_ _α} _zα _pα e := do
  let (.app f a) ← whnfR e | throwError "not abv ·"
  let pa' ← mkAppM ``abv_nonneg #[f, a]
  pure (.nonnegative pa')


lemma abv_eq_zero {x} : abv x = 0 ↔ x = 0 := abv_eq_zero'


lemma abv_add (x y) : abv (x + y) ≤ abv x + abv y := abv_add' x y


lemma abv_mul (x y) : abv (x * y) = abv x * abv y := abv_mul' x y


/-- A bundled absolute value is an absolute value. -/
instance _root_.AbsoluteValue.isAbsoluteValue (abv : AbsoluteValue R S) : IsAbsoluteValue abv where
  abv_nonneg' := abv.nonneg
  abv_eq_zero' := abv.eq_zero
  abv_add' := abv.add_le
  abv_mul' := abv.map_mul


/-- Convert an unbundled `IsAbsoluteValue` to a bundled `AbsoluteValue`. -/
@[simps]
def toAbsoluteValue : AbsoluteValue R S where
  toFun := abv
  add_le' := abv_add'
  eq_zero' _ := abv_eq_zero'
  nonneg' := abv_nonneg'
  map_mul' := abv_mul'


theorem abv_zero : abv 0 = 0 :=
  (toAbsoluteValue abv).map_zero


theorem abv_pos {a : R} : 0 < abv a ↔ a ≠ 0 :=
  (toAbsoluteValue abv).pos_iff


instance abs_isAbsoluteValue : IsAbsoluteValue (abs : S → S) :=
  AbsoluteValue.abs.isAbsoluteValue


theorem abv_one [Nontrivial R] : abv 1 = 1 :=
  (toAbsoluteValue abv).map_one


/-- `abv` as a `MonoidWithZeroHom`. -/
def abvHom [Nontrivial R] : R →*₀ S :=
  (toAbsoluteValue abv).toMonoidWithZeroHom


theorem abv_pow [Nontrivial R] (abv : R → S) [IsAbsoluteValue abv] (a : R) (n : ℕ) :
    abv (a ^ n) = abv a ^ n :=
  (toAbsoluteValue abv).map_pow a n


theorem abv_sub_le (a b c : R) : abv (a - c) ≤ abv (a - b) + abv (b - c) := by
  /-
    S : Type u_5
    inst✝² : OrderedRing S
    R : Type u_6
    inst✝¹ : Ring R
    abv : R → S
    inst✝ : IsAbsoluteValue abv
    a b c : R
    ⊢ LE.le (abv (HSub.hSub a c)) (HAdd.hAdd (abv (HSub.hSub a b)) (abv (HSub.hSub …
  -/
  simpa [sub_eq_add_neg, add_assoc] using abv_add abv (a - b) (b - c)
  /-
    🎉 no goals
  -/


theorem sub_abv_le_abv_sub (a b : R) : abv a - abv b ≤ abv (a - b) :=
  (toAbsoluteValue abv).le_sub a b


theorem abv_neg (a : R) : abv (-a) = abv a :=
  (toAbsoluteValue abv).map_neg a


theorem abv_sub (a b : R) : abv (a - b) = abv (b - a) :=
  (toAbsoluteValue abv).map_sub a b


theorem abs_abv_sub_le_abv_sub (a b : R) : abs (abv a - abv b) ≤ abv (a - b) :=
  (toAbsoluteValue abv).abs_abv_sub_le_abv_sub a b


theorem abv_one' : abv 1 = 1 :=
  (toAbsoluteValue abv).map_one_of_isLeftRegular <|
    (isRegular_of_ne_zero <| (toAbsoluteValue abv).ne_zero one_ne_zero).left


/-- An absolute value as a monoid with zero homomorphism, assuming the target is a semifield. -/
def abvHom' : R →*₀ S where
  toFun := abv; map_zero' := abv_zero abv; map_one' := abv_one' abv; map_mul' := abv_mul abv


theorem abv_inv (a : R) : abv a⁻¹ = (abv a)⁻¹ :=
  map_inv₀ (abvHom' abv) a


theorem abv_div (a b : R) : abv (a / b) = abv a / abv b :=
  map_div₀ (abvHom' abv) a b



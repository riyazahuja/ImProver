/-- A seminorm on a ring `R` is a function `f : R → ℝ` that preserves zero, takes nonnegative
  values, is subadditive and submultiplicative and such that `f (-x) = f x` for all `x ∈ R`. -/
structure RingSeminorm (R : Type*) [NonUnitalNonAssocRing R] extends AddGroupSeminorm R where
  /-- The property of a `RingSeminorm` that for all `x` and `y` in the ring, the norm of `x * y` is
    less than the norm of `x` times the norm of `y`. -/
  mul_le' : ∀ x y : R, toFun (x * y) ≤ toFun x * toFun y


/-- A function `f : R → ℝ` is a norm on a (nonunital) ring if it is a seminorm and `f x = 0`
  implies `x = 0`. -/
structure RingNorm (R : Type*) [NonUnitalNonAssocRing R] extends RingSeminorm R, AddGroupNorm R


/-- A multiplicative seminorm on a ring `R` is a function `f : R → ℝ` that preserves zero and
multiplication, takes nonnegative values, is subadditive and such that `f (-x) = f x` for all `x`.
-/
structure MulRingSeminorm (R : Type*) [NonAssocRing R] extends AddGroupSeminorm R,
  MonoidWithZeroHom R ℝ


/-- A multiplicative norm on a ring `R` is a multiplicative ring seminorm such that `f x = 0`
implies `x = 0`. -/
structure MulRingNorm (R : Type*) [NonAssocRing R] extends MulRingSeminorm R, AddGroupNorm R


instance funLike : FunLike (RingSeminorm R) R ℝ where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      R : Type u_1
      inst✝ : NonUnitalRing R
      f g : RingSeminorm R
      h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      ⊢ Eq f g
    -/
    cases f
    /-
      case mk
      R : Type u_1
      inst✝ : NonUnitalRing R
      g : RingSeminorm R
      toAddGroupSeminorm✝ : AddGroupSeminorm R
      mul_le'✝ : ∀ (x y : R), LE.le (toAddGroupSeminorm✝.toFun (HMul.hMul x y)) (HMu …
      h : Eq ((fun f => f.toFun) { toAddGroupSeminorm := toAddGroupSeminorm✝, mul_le …
      ⊢ Eq { toAddGroupSeminorm := toAddGroupSeminorm✝, mul_le' := mul_le'✝ } g
    -/
    cases g
    /-
      case mk.mk
      R : Type u_1
      inst✝ : NonUnitalRing R
      toAddGroupSeminorm✝¹ : AddGroupSeminorm R
      mul_le'✝¹ : ∀ (x y : R), LE.le (toAddGroupSeminorm✝¹.toFun (HMul.hMul x y)) (H …
      toAddGroupSeminorm✝ : AddGroupSeminorm R
      mul_le'✝ : ∀ (x y : R), LE.le (toAddGroupSeminorm✝.toFun (HMul.hMul x y)) (HMu …
      h : Eq ((fun f => f.toFun) { toAddGroupSeminorm := toAddGroupSeminorm✝¹, mul_l …
      ⊢ Eq { toAddGroupSeminorm := toAddGroupSeminorm✝¹, mul_le' := mul_le'✝¹ } { to …
    -/
    congr
    /-
      case mk.mk.e_toAddGroupSeminorm
      R : Type u_1
      inst✝ : NonUnitalRing R
      toAddGroupSeminorm✝¹ : AddGroupSeminorm R
      mul_le'✝¹ : ∀ (x y : R), LE.le (toAddGroupSeminorm✝¹.toFun (HMul.hMul x y)) (H …
      toAddGroupSeminorm✝ : AddGroupSeminorm R
      mul_le'✝ : ∀ (x y : R), LE.le (toAddGroupSeminorm✝.toFun (HMul.hMul x y)) (HMu …
      h : Eq ((fun f => f.toFun) { toAddGroupSeminorm := toAddGroupSeminorm✝¹, mul_l …
      ⊢ Eq toAddGroupSeminorm✝¹ toAddGroupSeminorm✝
    -/
    ext x
    /-
      case mk.mk.e_toAddGroupSeminorm.a
      R : Type u_1
      inst✝ : NonUnitalRing R
      toAddGroupSeminorm✝¹ : AddGroupSeminorm R
      mul_le'✝¹ : ∀ (x y : R), LE.le (toAddGroupSeminorm✝¹.toFun (HMul.hMul x y)) (H …
      toAddGroupSeminorm✝ : AddGroupSeminorm R
      mul_le'✝ : ∀ (x y : R), LE.le (toAddGroupSeminorm✝.toFun (HMul.hMul x y)) (HMu …
      h : Eq ((fun f => f.toFun) { toAddGroupSeminorm := toAddGroupSeminorm✝¹, mul_l …
      x : R
      ⊢ Eq (toAddGroupSeminorm✝¹ x) (toAddGroupSeminorm✝ x)
    -/
    exact congr_fun h x
    /-
      🎉 no goals
    -/


instance ringSeminormClass : RingSeminormClass (RingSeminorm R) R ℝ where
  map_zero f := f.map_zero'
  map_add_le_add f := f.add_le'
  map_mul_le_mul f := f.mul_le'
  map_neg_eq_map f := f.neg'


@[simp]
theorem toFun_eq_coe (p : RingSeminorm R) : (p.toAddGroupSeminorm : R → ℝ) = p :=
  rfl


@[ext]
theorem ext {p q : RingSeminorm R} : (∀ x, p x = q x) → p = q :=
  DFunLike.ext p q


instance : Zero (RingSeminorm R) :=
  ⟨{ AddGroupSeminorm.instZeroAddGroupSeminorm.zero with mul_le' :=
    fun _ _ => (zero_mul _).ge }⟩


theorem eq_zero_iff {p : RingSeminorm R} : p = 0 ↔ ∀ x, p x = 0 :=
  DFunLike.ext_iff


                                                                      /-
                                                                        R : Type u_1
                                                                        inst✝ : NonUnitalRing R
                                                                        p : RingSeminorm R
                                                                        ⊢ Iff (Ne p 0) (Exists fun x => Ne (p x) 0)
                                                                      -/
theorem ne_zero_iff {p : RingSeminorm R} : p ≠ 0 ↔ ∃ x, p x ≠ 0 := by simp [eq_zero_iff]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


instance : Inhabited (RingSeminorm R) :=
  ⟨0⟩


/-- The trivial seminorm on a ring `R` is the `RingSeminorm` taking value `0` at `0` and `1` at
every other element. -/
instance [DecidableEq R] : One (RingSeminorm R) :=
  ⟨{ (1 : AddGroupSeminorm R) with
      mul_le' := fun x y => by
        /-
          R : Type u_1
          inst✝¹ : NonUnitalRing R
          inst✝ : DecidableEq R
          x y : R
          ⊢ LE.le (__src✝.toFun (HMul.hMul x y)) (HMul.hMul (__src✝.toFun x) (__src✝.toF …
        -/
        by_cases h : x * y = 0
          /-
            case pos
            R : Type u_1
            inst✝¹ : NonUnitalRing R
            inst✝ : DecidableEq R
            x y : R
            h : Eq (HMul.hMul x y) 0
            ⊢ LE.le (__src✝.toFun (HMul.hMul x y)) (HMul.hMul (__src✝.toFun x) (__src✝.toF …
          -/
        · refine (if_pos h).trans_le (mul_nonneg ?_ ?_) <;>
              /-
                case pos.refine_1
                R : Type u_1
                inst✝¹ : NonUnitalRing R
                inst✝ : DecidableEq R
                x y : R
                h : Eq (HMul.hMul x y) 0
                ⊢ LE.le 0 (__src✝.toFun x)
              -/
              /-
                case pos.refine_1
                R : Type u_1
                inst✝¹ : NonUnitalRing R
                inst✝ : DecidableEq R
                x y : R
                h : Eq (HMul.hMul x y) 0
                ⊢ LE.le 0 (ite (Eq x 0) 0 1)
              -/
              /-
                case pos
                R : Type u_1
                inst✝¹ : NonUnitalRing R
                inst✝ : DecidableEq R
                x y : R
                h : Eq (HMul.hMul x y) 0
                h✝ : Eq x 0
                ⊢ LE.le 0 0
              -/
              /-
                🎉 no goals
              -/
              split_ifs
              /-
                case pos
                R : Type u_1
                inst✝¹ : NonUnitalRing R
                inst✝ : DecidableEq R
                x y : R
                h : Eq (HMul.hMul x y) 0
                h✝ : Eq y 0
                ⊢ LE.le 0 0
              -/
              exacts [le_rfl, zero_le_one]
              /-
                🎉 no goals
              -/
          /-
            case neg
            R : Type u_1
            inst✝¹ : NonUnitalRing R
            inst✝ : DecidableEq R
            x y : R
            h : Not (Eq (HMul.hMul x y) 0)
            ⊢ LE.le (__src✝.toFun (HMul.hMul x y)) (HMul.hMul (__src✝.toFun x) (__src✝.toF …
          -/
        · change ite _ _ _ ≤ ite _ _ _ * ite _ _ _
          simp only [if_false, h, left_ne_zero_of_mul h, right_ne_zero_of_mul h, mul_one,
            le_refl] }⟩


@[simp]
theorem apply_one [DecidableEq R] (x : R) : (1 : RingSeminorm R) x = if x = 0 then 0 else 1 :=
  rfl


theorem seminorm_one_eq_one_iff_ne_zero (hp : p 1 ≤ 1) : p 1 = 1 ↔ p ≠ 0 := by
  refine
    ⟨fun h => ne_zero_iff.mpr ⟨1, by rw [h]; exact one_ne_zero⟩,
      fun h => ?_⟩
  /-
    R : Type u_1
    inst✝ : Ring R
    p : RingSeminorm R
    hp : LE.le (p 1) 1
    h : Ne p 0
    ⊢ Eq (p 1) 1
  -/
  obtain hp0 | hp0 := (apply_nonneg p (1 : R)).eq_or_gt
    /-
      case inl
      R : Type u_1
      inst✝ : Ring R
      p : RingSeminorm R
      hp : LE.le (p 1) 1
      h : Ne p 0
      hp0 : Eq (p 1) 0
      ⊢ Eq (p 1) 1
    -/
  · exfalso
    /-
      case inl
      R : Type u_1
      inst✝ : Ring R
      p : RingSeminorm R
      hp : LE.le (p 1) 1
      h : Ne p 0
      hp0 : Eq (p 1) 0
      ⊢ False
    -/
    refine h (ext fun x => (apply_nonneg _ _).antisymm' ?_)
    /-
      case inl
      R : Type u_1
      inst✝ : Ring R
      p : RingSeminorm R
      hp : LE.le (p 1) 1
      h : Ne p 0
      hp0 : Eq (p 1) 0
      x : R
      ⊢ LE.le (p x) 0
    -/
    simpa only [hp0, mul_one, mul_zero] using map_mul_le_mul p x 1
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝ : Ring R
      p : RingSeminorm R
      hp : LE.le (p 1) 1
      h : Ne p 0
      hp0 : LT.lt 0 (p 1)
      ⊢ Eq (p 1) 1
    -/
  · refine hp.antisymm ((le_mul_iff_one_le_left hp0).1 ?_)
    /-
      case inr
      R : Type u_1
      inst✝ : Ring R
      p : RingSeminorm R
      hp : LE.le (p 1) 1
      h : Ne p 0
      hp0 : LT.lt 0 (p 1)
      ⊢ LE.le (p 1) (HMul.hMul (p 1) (p 1))
    -/
    simpa only [one_mul] using map_mul_le_mul p (1 : R) _
    /-
      🎉 no goals
    -/


theorem exists_index_pow_le (hna : IsNonarchimedean p) (x y : R) (n : ℕ) :
    ∃ (m : ℕ), m < n + 1 ∧ p ((x + y) ^ (n : ℕ)) ^ (1 / (n : ℝ)) ≤
      (p (x ^ m) * p (y ^ (n - m : ℕ))) ^ (1 / (n : ℝ)) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    p : RingSeminorm R
    hna : IsNonarchimedean ⇑p
    x y : R
    n : Nat
    ⊢ Exists fun m => And (LT.lt m (HAdd.hAdd n 1)) (LE.le (HPow.hPow (p (HPow.hPo …
  -/
  obtain ⟨m, hm_lt, hm⟩ := IsNonarchimedean.add_pow_le hna n x y
  /-
    case intro.intro
    R : Type u_1
    inst✝ : CommRing R
    p : RingSeminorm R
    hna : IsNonarchimedean ⇑p
    x y : R
    n m : Nat
    hm_lt : LT.lt m (HAdd.hAdd n 1)
    hm : LE.le (p (HPow.hPow (HAdd.hAdd x y) n)) (HMul.hMul (p (HPow.hPow x m)) (p …
    ⊢ Exists fun m => And (LT.lt m (HAdd.hAdd n 1)) (LE.le (HPow.hPow (p (HPow.hPo …
  -/
  exact ⟨m, hm_lt, Real.rpow_le_rpow (apply_nonneg p _) hm (one_div_nonneg.mpr n.cast_nonneg')⟩
  /-
    🎉 no goals
  -/


/-- If `f` is a ring seminorm on `a`, then `∀ {n : ℕ}, n ≠ 0 → f (a ^ n) ≤ f a ^ n`. -/
theorem map_pow_le_pow {F α : Type*} [Ring α] [FunLike F α ℝ] [RingSeminormClass F α ℝ] (f : F)
    (a : α) : ∀ {n : ℕ}, n ≠ 0 → f (a ^ n) ≤ f a ^ n
  | 0, h => absurd rfl h
               /-
                 F : Type u_2
                 α : Type u_3
                 inst✝² : Ring α
                 inst✝¹ : FunLike F α Real
                 inst✝ : RingSeminormClass F α Real
                 f : F
                 a : α
                 x✝ : Ne 1 0
                 ⊢ LE.le (f (HPow.hPow a 1)) (HPow.hPow (f a) 1)
               -/
  | 1, _ => by simp only [pow_one, le_refl]
               /-
                 🎉 no goals
               -/
  | n + 2, _ => by
    /-
      F : Type u_2
      α : Type u_3
      inst✝² : Ring α
      inst✝¹ : FunLike F α Real
      inst✝ : RingSeminormClass F α Real
      f : F
      a : α
      n : Nat
      x✝ : Ne (HAdd.hAdd n 2) 0
      ⊢ LE.le (f (HPow.hPow a (HAdd.hAdd n 2))) (HPow.hPow (f a) (HAdd.hAdd n 2))
    -/
    simp only [pow_succ _ (n + 1)]
    exact
      le_trans (map_mul_le_mul f _ a)
        (mul_le_mul_of_nonneg_right (map_pow_le_pow _ _ n.succ_ne_zero) (apply_nonneg f a))


/-- If `f` is a ring seminorm on `a` with `f 1 ≤ 1`, then `∀ (n : ℕ), f (a ^ n) ≤ f a ^ n`. -/
theorem map_pow_le_pow' {F α : Type*} [Ring α] [FunLike F α ℝ] [RingSeminormClass F α ℝ] {f : F}
    (hf1 : f 1 ≤ 1) (a : α) : ∀ n : ℕ, f (a ^ n) ≤ f a ^ n
            /-
              F : Type u_2
              α : Type u_3
              inst✝² : Ring α
              inst✝¹ : FunLike F α Real
              inst✝ : RingSeminormClass F α Real
              f : F
              hf1 : LE.le (f 1) 1
              a : α
              ⊢ LE.le (f (HPow.hPow a 0)) (HPow.hPow (f a) 0)
            -/
  | 0 => by simp only [pow_zero, hf1]
            /-
              🎉 no goals
            -/
  | n + 1 => by
    /-
      F : Type u_2
      α : Type u_3
      inst✝² : Ring α
      inst✝¹ : FunLike F α Real
      inst✝ : RingSeminormClass F α Real
      f : F
      hf1 : LE.le (f 1) 1
      a : α
      n : Nat
      ⊢ LE.le (f (HPow.hPow a (HAdd.hAdd n 1))) (HPow.hPow (f a) (HAdd.hAdd n 1))
    -/
    simp only [pow_succ _ n]
    exact le_trans (map_mul_le_mul f _ a)
      (mul_le_mul_of_nonneg_right (map_pow_le_pow' hf1 _ n) (apply_nonneg f a))


/-- The norm of a `NonUnitalSeminormedRing` as a `RingSeminorm`. -/
def normRingSeminorm (R : Type*) [NonUnitalSeminormedRing R] : RingSeminorm R :=
  { normAddGroupSeminorm R with
    toFun := norm
    mul_le' := norm_mul_le }


/-- If `f` is a ring seminorm on `R` with `f 1 ≤ 1` and `s : ℕ → ℕ` is bounded by `n`, then
  `f (x ^ s (ψ n)) ^ (1 / (ψ n : ℝ))` is eventually bounded. -/
theorem isBoundedUnder (hp : p 1 ≤ 1) {s : ℕ → ℕ} (hs_le : ∀ n : ℕ, s n ≤ n) {x : R} (ψ : ℕ → ℕ) :
    IsBoundedUnder LE.le atTop fun n : ℕ => p (x ^ s (ψ n)) ^ (1 / (ψ n : ℝ)) := by
  have h_le : ∀ m : ℕ, p (x ^ s (ψ m)) ^ (1 / (ψ m : ℝ)) ≤ p x ^ ((s (ψ m) : ℝ) / (ψ m : ℝ)) := by
    intro m
    rw [← mul_one_div (s (ψ m) : ℝ), rpow_mul (apply_nonneg p x), rpow_natCast]
    exact rpow_le_rpow (apply_nonneg _ _) (map_pow_le_pow' hp x _)
      (one_div_nonneg.mpr (cast_nonneg _))
  /-
    R : Type u_1
    inst✝ : Ring R
    p : RingSeminorm R
    hp : LE.le (p 1) 1
    s : Nat → Nat
    hs_le : ∀ (n : Nat), LE.le (s n) n
    x : R
    ψ : Nat → Nat
    h_le : ∀ (m : Nat), LE.le (HPow.hPow (p (HPow.hPow x (s (ψ m)))) (HDiv.hDiv 1  …
    ⊢ Filter.IsBoundedUnder LE.le Filter.atTop fun n => HPow.hPow (p (HPow.hPow x  …
  -/
  apply isBoundedUnder_of
  /-
    case a
    R : Type u_1
    inst✝ : Ring R
    p : RingSeminorm R
    hp : LE.le (p 1) 1
    s : Nat → Nat
    hs_le : ∀ (n : Nat), LE.le (s n) n
    x : R
    ψ : Nat → Nat
    h_le : ∀ (m : Nat), LE.le (HPow.hPow (p (HPow.hPow x (s (ψ m)))) (HDiv.hDiv 1  …
    ⊢ Exists fun b => ∀ (x_1 : Nat), LE.le (HPow.hPow (p (HPow.hPow x (s (ψ x_1))) …
  -/
  by_cases hfx : p x ≤ 1
  · use 1, fun m => le_trans (h_le m)
      (rpow_le_one (apply_nonneg _ _) hfx (div_nonneg (cast_nonneg _) (cast_nonneg _)))
    /-
      case neg
      R : Type u_1
      inst✝ : Ring R
      p : RingSeminorm R
      hp : LE.le (p 1) 1
      s : Nat → Nat
      hs_le : ∀ (n : Nat), LE.le (s n) n
      x : R
      ψ : Nat → Nat
      h_le : ∀ (m : Nat), LE.le (HPow.hPow (p (HPow.hPow x (s (ψ m)))) (HDiv.hDiv 1  …
      hfx : Not (LE.le (p x) 1)
      ⊢ Exists fun b => ∀ (x_1 : Nat), LE.le (HPow.hPow (p (HPow.hPow x (s (ψ x_1))) …
    -/
  · use p x
    /-
      case h
      R : Type u_1
      inst✝ : Ring R
      p : RingSeminorm R
      hp : LE.le (p 1) 1
      s : Nat → Nat
      hs_le : ∀ (n : Nat), LE.le (s n) n
      x : R
      ψ : Nat → Nat
      h_le : ∀ (m : Nat), LE.le (HPow.hPow (p (HPow.hPow x (s (ψ m)))) (HDiv.hDiv 1  …
      hfx : Not (LE.le (p x) 1)
      ⊢ ∀ (x_1 : Nat), LE.le (HPow.hPow (p (HPow.hPow x (s (ψ x_1)))) (HDiv.hDiv 1 ↑ …
    -/
    intro m
    /-
      case h
      R : Type u_1
      inst✝ : Ring R
      p : RingSeminorm R
      hp : LE.le (p 1) 1
      s : Nat → Nat
      hs_le : ∀ (n : Nat), LE.le (s n) n
      x : R
      ψ : Nat → Nat
      h_le : ∀ (m : Nat), LE.le (HPow.hPow (p (HPow.hPow x (s (ψ m)))) (HDiv.hDiv 1  …
      hfx : Not (LE.le (p x) 1)
      m : Nat
      ⊢ LE.le (HPow.hPow (p (HPow.hPow x (s (ψ m)))) (HDiv.hDiv 1 ↑(ψ m))) (p x)
    -/
    apply le_trans (h_le m)
    /-
      case h
      R : Type u_1
      inst✝ : Ring R
      p : RingSeminorm R
      hp : LE.le (p 1) 1
      s : Nat → Nat
      hs_le : ∀ (n : Nat), LE.le (s n) n
      x : R
      ψ : Nat → Nat
      h_le : ∀ (m : Nat), LE.le (HPow.hPow (p (HPow.hPow x (s (ψ m)))) (HDiv.hDiv 1  …
      hfx : Not (LE.le (p x) 1)
      m : Nat
      ⊢ LE.le (HPow.hPow (p x) (HDiv.hDiv ↑(s (ψ m)) ↑(ψ m))) (p x)
    -/
    conv_rhs => rw [← rpow_one (p x)]
    exact rpow_le_rpow_of_exponent_le (le_of_lt (not_le.mp hfx))
      (div_le_one_of_le₀ (cast_le.mpr (hs_le _)) (cast_nonneg _))


instance funLike : FunLike (RingNorm R) R ℝ where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      R : Type u_1
      inst✝ : NonUnitalRing R
      f g : RingNorm R
      h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      ⊢ Eq f g
    -/
    cases f
    /-
      case mk
      R : Type u_1
      inst✝ : NonUnitalRing R
      g : RingNorm R
      toRingSeminorm✝ : RingSeminorm R
      eq_zero_of_map_eq_zero'✝ : ∀ (x : R), Eq (toRingSeminorm✝.toFun x) 0 → Eq x 0
      h : Eq ((fun f => f.toFun) { toRingSeminorm := toRingSeminorm✝, eq_zero_of_map …
      ⊢ Eq { toRingSeminorm := toRingSeminorm✝, eq_zero_of_map_eq_zero' := eq_zero_o …
    -/
    cases g
    /-
      case mk.mk
      R : Type u_1
      inst✝ : NonUnitalRing R
      toRingSeminorm✝¹ : RingSeminorm R
      eq_zero_of_map_eq_zero'✝¹ : ∀ (x : R), Eq (toRingSeminorm✝¹.toFun x) 0 → Eq x 0
      toRingSeminorm✝ : RingSeminorm R
      eq_zero_of_map_eq_zero'✝ : ∀ (x : R), Eq (toRingSeminorm✝.toFun x) 0 → Eq x 0
      h : Eq ((fun f => f.toFun) { toRingSeminorm := toRingSeminorm✝¹, eq_zero_of_ma …
      ⊢ Eq { toRingSeminorm := toRingSeminorm✝¹, eq_zero_of_map_eq_zero' := eq_zero_ …
    -/
    congr
    /-
      case mk.mk.e_toRingSeminorm
      R : Type u_1
      inst✝ : NonUnitalRing R
      toRingSeminorm✝¹ : RingSeminorm R
      eq_zero_of_map_eq_zero'✝¹ : ∀ (x : R), Eq (toRingSeminorm✝¹.toFun x) 0 → Eq x 0
      toRingSeminorm✝ : RingSeminorm R
      eq_zero_of_map_eq_zero'✝ : ∀ (x : R), Eq (toRingSeminorm✝.toFun x) 0 → Eq x 0
      h : Eq ((fun f => f.toFun) { toRingSeminorm := toRingSeminorm✝¹, eq_zero_of_ma …
      ⊢ Eq toRingSeminorm✝¹ toRingSeminorm✝
    -/
    ext x
    /-
      case mk.mk.e_toRingSeminorm.a
      R : Type u_1
      inst✝ : NonUnitalRing R
      toRingSeminorm✝¹ : RingSeminorm R
      eq_zero_of_map_eq_zero'✝¹ : ∀ (x : R), Eq (toRingSeminorm✝¹.toFun x) 0 → Eq x 0
      toRingSeminorm✝ : RingSeminorm R
      eq_zero_of_map_eq_zero'✝ : ∀ (x : R), Eq (toRingSeminorm✝.toFun x) 0 → Eq x 0
      h : Eq ((fun f => f.toFun) { toRingSeminorm := toRingSeminorm✝¹, eq_zero_of_ma …
      x : R
      ⊢ Eq (toRingSeminorm✝¹ x) (toRingSeminorm✝ x)
    -/
    exact congr_fun h x
    /-
      🎉 no goals
    -/


instance ringNormClass : RingNormClass (RingNorm R) R ℝ where
  map_zero f := f.map_zero'
  map_add_le_add f := f.add_le'
  map_mul_le_mul f := f.mul_le'
  map_neg_eq_map f := f.neg'
  eq_zero_of_map_eq_zero f := f.eq_zero_of_map_eq_zero' _

-- Porting note: This is no longer `@[simp]` in Lean 4

theorem toFun_eq_coe (p : RingNorm R) : p.toFun = p := rfl


@[ext]
theorem ext {p q : RingNorm R} : (∀ x, p x = q x) → p = q :=
  DFunLike.ext p q


/-- The trivial norm on a ring `R` is the `RingNorm` taking value `0` at `0` and `1` at every
  other element. -/
instance [DecidableEq R] : One (RingNorm R) :=
  ⟨{ (1 : RingSeminorm R), (1 : AddGroupNorm R) with }⟩


@[simp]
theorem apply_one [DecidableEq R] (x : R) : (1 : RingNorm R) x = if x = 0 then 0 else 1 :=
  rfl


instance [DecidableEq R] : Inhabited (RingNorm R) :=
  ⟨1⟩


instance funLike : FunLike (MulRingSeminorm R) R ℝ where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      R : Type u_1
      inst✝ : NonAssocRing R
      f g : MulRingSeminorm R
      h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      ⊢ Eq f g
    -/
    cases f
    /-
      case mk
      R : Type u_1
      inst✝ : NonAssocRing R
      g : MulRingSeminorm R
      toAddGroupSeminorm✝ : AddGroupSeminorm R
      map_one'✝ : Eq (toAddGroupSeminorm✝.toFun 1) 1
      map_mul'✝ : ∀ (x y : R), Eq (toAddGroupSeminorm✝.toFun (HMul.hMul x y)) (HMul. …
      h : Eq ((fun f => f.toFun) { toAddGroupSeminorm := toAddGroupSeminorm✝, map_on …
      ⊢ Eq { toAddGroupSeminorm := toAddGroupSeminorm✝, map_one' := map_one'✝, map_m …
    -/
    cases g
    /-
      case mk.mk
      R : Type u_1
      inst✝ : NonAssocRing R
      toAddGroupSeminorm✝¹ : AddGroupSeminorm R
      map_one'✝¹ : Eq (toAddGroupSeminorm✝¹.toFun 1) 1
      map_mul'✝¹ : ∀ (x y : R), Eq (toAddGroupSeminorm✝¹.toFun (HMul.hMul x y)) (HMu …
      toAddGroupSeminorm✝ : AddGroupSeminorm R
      map_one'✝ : Eq (toAddGroupSeminorm✝.toFun 1) 1
      map_mul'✝ : ∀ (x y : R), Eq (toAddGroupSeminorm✝.toFun (HMul.hMul x y)) (HMul. …
      h : Eq ((fun f => f.toFun) { toAddGroupSeminorm := toAddGroupSeminorm✝¹, map_o …
      ⊢ Eq { toAddGroupSeminorm := toAddGroupSeminorm✝¹, map_one' := map_one'✝¹, map …
    -/
    congr
    /-
      case mk.mk.e_toAddGroupSeminorm
      R : Type u_1
      inst✝ : NonAssocRing R
      toAddGroupSeminorm✝¹ : AddGroupSeminorm R
      map_one'✝¹ : Eq (toAddGroupSeminorm✝¹.toFun 1) 1
      map_mul'✝¹ : ∀ (x y : R), Eq (toAddGroupSeminorm✝¹.toFun (HMul.hMul x y)) (HMu …
      toAddGroupSeminorm✝ : AddGroupSeminorm R
      map_one'✝ : Eq (toAddGroupSeminorm✝.toFun 1) 1
      map_mul'✝ : ∀ (x y : R), Eq (toAddGroupSeminorm✝.toFun (HMul.hMul x y)) (HMul. …
      h : Eq ((fun f => f.toFun) { toAddGroupSeminorm := toAddGroupSeminorm✝¹, map_o …
      ⊢ Eq toAddGroupSeminorm✝¹ toAddGroupSeminorm✝
    -/
    ext x
    /-
      case mk.mk.e_toAddGroupSeminorm.a
      R : Type u_1
      inst✝ : NonAssocRing R
      toAddGroupSeminorm✝¹ : AddGroupSeminorm R
      map_one'✝¹ : Eq (toAddGroupSeminorm✝¹.toFun 1) 1
      map_mul'✝¹ : ∀ (x y : R), Eq (toAddGroupSeminorm✝¹.toFun (HMul.hMul x y)) (HMu …
      toAddGroupSeminorm✝ : AddGroupSeminorm R
      map_one'✝ : Eq (toAddGroupSeminorm✝.toFun 1) 1
      map_mul'✝ : ∀ (x y : R), Eq (toAddGroupSeminorm✝.toFun (HMul.hMul x y)) (HMul. …
      h : Eq ((fun f => f.toFun) { toAddGroupSeminorm := toAddGroupSeminorm✝¹, map_o …
      x : R
      ⊢ Eq (toAddGroupSeminorm✝¹ x) (toAddGroupSeminorm✝ x)
    -/
    exact congr_fun h x
    /-
      🎉 no goals
    -/


instance mulRingSeminormClass : MulRingSeminormClass (MulRingSeminorm R) R ℝ where
  map_zero f := f.map_zero'
  map_one f := f.map_one'
  map_add_le_add f := f.add_le'
  map_mul f := f.map_mul'
  map_neg_eq_map f := f.neg'


@[simp]
theorem toFun_eq_coe (p : MulRingSeminorm R) : (p.toAddGroupSeminorm : R → ℝ) = p :=
  rfl


@[ext]
theorem ext {p q : MulRingSeminorm R} : (∀ x, p x = q x) → p = q :=
  DFunLike.ext p q


/-- The trivial seminorm on a ring `R` is the `MulRingSeminorm` taking value `0` at `0` and `1` at
every other element. -/
instance : One (MulRingSeminorm R) :=
  ⟨{ (1 : AddGroupSeminorm R) with
      map_one' := if_neg one_ne_zero
      map_mul' := fun x y => by
        /-
          R : Type u_1
          inst✝³ : NonAssocRing R
          inst✝² : DecidableEq R
          inst✝¹ : NoZeroDivisors R
          inst✝ : Nontrivial R
          x y : R
          ⊢ Eq (__src✝.toFun (HMul.hMul x y)) (HMul.hMul (__src✝.toFun x) (__src✝.toFun  …
        -/
        obtain rfl | hx := eq_or_ne x 0
          /-
            case inl
            R : Type u_1
            inst✝³ : NonAssocRing R
            inst✝² : DecidableEq R
            inst✝¹ : NoZeroDivisors R
            inst✝ : Nontrivial R
            y : R
            ⊢ Eq (__src✝.toFun (HMul.hMul 0 y)) (HMul.hMul (__src✝.toFun 0) (__src✝.toFun  …
          -/
        · simp
          /-
            🎉 no goals
          -/
        /-
          case inr
          R : Type u_1
          inst✝³ : NonAssocRing R
          inst✝² : DecidableEq R
          inst✝¹ : NoZeroDivisors R
          inst✝ : Nontrivial R
          x y : R
          hx : Ne x 0
          ⊢ Eq (__src✝.toFun (HMul.hMul x y)) (HMul.hMul (__src✝.toFun x) (__src✝.toFun  …
        -/
        obtain rfl | hy := eq_or_ne y 0
          /-
            case inr.inl
            R : Type u_1
            inst✝³ : NonAssocRing R
            inst✝² : DecidableEq R
            inst✝¹ : NoZeroDivisors R
            inst✝ : Nontrivial R
            x : R
            hx : Ne x 0
            ⊢ Eq (__src✝.toFun (HMul.hMul x 0)) (HMul.hMul (__src✝.toFun x) (__src✝.toFun  …
          -/
        · simp
          /-
            🎉 no goals
          -/
          /-
            case inr.inr
            R : Type u_1
            inst✝³ : NonAssocRing R
            inst✝² : DecidableEq R
            inst✝¹ : NoZeroDivisors R
            inst✝ : Nontrivial R
            x y : R
            hx : Ne x 0
            hy : Ne y 0
            ⊢ Eq (__src✝.toFun (HMul.hMul x y)) (HMul.hMul (__src✝.toFun x) (__src✝.toFun  …
          -/
        · simp [hx, hy] }⟩
          /-
            🎉 no goals
          -/


@[simp]
theorem apply_one (x : R) : (1 : MulRingSeminorm R) x = if x = 0 then 0 else 1 :=
  rfl


instance : Inhabited (MulRingSeminorm R) :=
  ⟨1⟩


instance funLike : FunLike (MulRingNorm R) R ℝ where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      R : Type u_1
      inst✝ : NonAssocRing R
      f g : MulRingNorm R
      h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      ⊢ Eq f g
    -/
    cases f
    /-
      case mk
      R : Type u_1
      inst✝ : NonAssocRing R
      g : MulRingNorm R
      toMulRingSeminorm✝ : MulRingSeminorm R
      eq_zero_of_map_eq_zero'✝ : ∀ (x : R), Eq (toMulRingSeminorm✝.toFun x) 0 → Eq x 0
      h : Eq ((fun f => f.toFun) { toMulRingSeminorm := toMulRingSeminorm✝, eq_zero_ …
      ⊢ Eq { toMulRingSeminorm := toMulRingSeminorm✝, eq_zero_of_map_eq_zero' := eq_ …
    -/
    cases g
    /-
      case mk.mk
      R : Type u_1
      inst✝ : NonAssocRing R
      toMulRingSeminorm✝¹ : MulRingSeminorm R
      eq_zero_of_map_eq_zero'✝¹ : ∀ (x : R), Eq (toMulRingSeminorm✝¹.toFun x) 0 → Eq …
      toMulRingSeminorm✝ : MulRingSeminorm R
      eq_zero_of_map_eq_zero'✝ : ∀ (x : R), Eq (toMulRingSeminorm✝.toFun x) 0 → Eq x 0
      h : Eq ((fun f => f.toFun) { toMulRingSeminorm := toMulRingSeminorm✝¹, eq_zero …
      ⊢ Eq { toMulRingSeminorm := toMulRingSeminorm✝¹, eq_zero_of_map_eq_zero' := eq …
    -/
    congr
    /-
      case mk.mk.e_toMulRingSeminorm
      R : Type u_1
      inst✝ : NonAssocRing R
      toMulRingSeminorm✝¹ : MulRingSeminorm R
      eq_zero_of_map_eq_zero'✝¹ : ∀ (x : R), Eq (toMulRingSeminorm✝¹.toFun x) 0 → Eq …
      toMulRingSeminorm✝ : MulRingSeminorm R
      eq_zero_of_map_eq_zero'✝ : ∀ (x : R), Eq (toMulRingSeminorm✝.toFun x) 0 → Eq x 0
      h : Eq ((fun f => f.toFun) { toMulRingSeminorm := toMulRingSeminorm✝¹, eq_zero …
      ⊢ Eq toMulRingSeminorm✝¹ toMulRingSeminorm✝
    -/
    ext x
    /-
      case mk.mk.e_toMulRingSeminorm.a
      R : Type u_1
      inst✝ : NonAssocRing R
      toMulRingSeminorm✝¹ : MulRingSeminorm R
      eq_zero_of_map_eq_zero'✝¹ : ∀ (x : R), Eq (toMulRingSeminorm✝¹.toFun x) 0 → Eq …
      toMulRingSeminorm✝ : MulRingSeminorm R
      eq_zero_of_map_eq_zero'✝ : ∀ (x : R), Eq (toMulRingSeminorm✝.toFun x) 0 → Eq x 0
      h : Eq ((fun f => f.toFun) { toMulRingSeminorm := toMulRingSeminorm✝¹, eq_zero …
      x : R
      ⊢ Eq (toMulRingSeminorm✝¹ x) (toMulRingSeminorm✝ x)
    -/
    exact congr_fun h x
    /-
      🎉 no goals
    -/


instance mulRingNormClass : MulRingNormClass (MulRingNorm R) R ℝ where
  map_zero f := f.map_zero'
  map_one f := f.map_one'
  map_add_le_add f := f.add_le'
  map_mul f := f.map_mul'
  map_neg_eq_map f := f.neg'
  eq_zero_of_map_eq_zero f := f.eq_zero_of_map_eq_zero' _

-- Porting note: This no longer in `@[simp]`-normal form in Lean 4

theorem toFun_eq_coe (p : MulRingNorm R) : p.toFun = p := rfl


@[ext]
theorem ext {p q : MulRingNorm R} : (∀ x, p x = q x) → p = q :=
  DFunLike.ext p q


/-- The trivial norm on a ring `R` is the `MulRingNorm` taking value `0` at `0` and `1` at every
other element. -/
instance : One (MulRingNorm R) :=
  ⟨{ (1 : MulRingSeminorm R), (1 : AddGroupNorm R) with }⟩


@[simp]
theorem apply_one (x : R) : (1 : MulRingNorm R) x = if x = 0 then 0 else 1 :=
  rfl


instance : Inhabited (MulRingNorm R) :=
  ⟨1⟩



/-- Two multiplicative ring norms `f, g` on `R` are equivalent if there exists a positive constant
  `c` such that for all `x ∈ R`, `(f x)^c = g x`. -/

def equiv (f : MulRingNorm R) (g : MulRingNorm R) :=
  ∃ c : ℝ, 0 < c ∧ (fun x => (f x) ^ c) = g


/-- Equivalence of multiplicative ring norms is reflexive. -/
lemma equiv_refl (f : MulRingNorm R) : equiv f f := by
    /-
      R : Type u_2
      inst✝ : Ring R
      f : MulRingNorm R
      ⊢ f.equiv f
    -/
    exact ⟨1, Real.zero_lt_one, by simp only [Real.rpow_one]⟩
    /-
      🎉 no goals
    -/


/-- Equivalence of multiplicative ring norms is symmetric. -/
lemma equiv_symm {f g : MulRingNorm R} (hfg : equiv f g) : equiv g f := by
  /-
    R : Type u_2
    inst✝ : Ring R
    f g : MulRingNorm R
    hfg : f.equiv g
    ⊢ g.equiv f
  -/
  rcases hfg with ⟨c, hcpos, h⟩
  /-
    case intro.intro
    R : Type u_2
    inst✝ : Ring R
    f g : MulRingNorm R
    c : Real
    hcpos : LT.lt 0 c
    h : Eq (fun x => HPow.hPow (f x) c) ⇑g
    ⊢ g.equiv f
  -/
  use 1/c
  /-
    case h
    R : Type u_2
    inst✝ : Ring R
    f g : MulRingNorm R
    c : Real
    hcpos : LT.lt 0 c
    h : Eq (fun x => HPow.hPow (f x) c) ⇑g
    ⊢ And (LT.lt 0 (HDiv.hDiv 1 c)) (Eq (fun x => HPow.hPow (g x) (HDiv.hDiv 1 c)) …
  -/
  constructor
    /-
      case h.left
      R : Type u_2
      inst✝ : Ring R
      f g : MulRingNorm R
      c : Real
      hcpos : LT.lt 0 c
      h : Eq (fun x => HPow.hPow (f x) c) ⇑g
      ⊢ LT.lt 0 (HDiv.hDiv 1 c)
    -/
  · simp only [one_div, inv_pos, hcpos]
    /-
      🎉 no goals
    -/
  /-
    case h.right
    R : Type u_2
    inst✝ : Ring R
    f g : MulRingNorm R
    c : Real
    hcpos : LT.lt 0 c
    h : Eq (fun x => HPow.hPow (f x) c) ⇑g
    ⊢ Eq (fun x => HPow.hPow (g x) (HDiv.hDiv 1 c)) ⇑f
  -/
  ext x
  /-
    case h.right.h
    R : Type u_2
    inst✝ : Ring R
    f g : MulRingNorm R
    c : Real
    hcpos : LT.lt 0 c
    h : Eq (fun x => HPow.hPow (f x) c) ⇑g
    x : R
    ⊢ Eq (HPow.hPow (g x) (HDiv.hDiv 1 c)) (f x)
  -/
  simpa [← congr_fun h x] using Real.rpow_rpow_inv (apply_nonneg f x) (ne_of_lt hcpos).symm
  /-
    🎉 no goals
  -/


/-- Equivalence of multiplicative ring norms is transitive. -/
lemma equiv_trans {f g k : MulRingNorm R} (hfg : equiv f g) (hgk : equiv g k) :
    equiv f k := by
  /-
    R : Type u_2
    inst✝ : Ring R
    f g k : MulRingNorm R
    hfg : f.equiv g
    hgk : g.equiv k
    ⊢ f.equiv k
  -/
  rcases hfg with ⟨c, hcPos, hfg⟩
  /-
    case intro.intro
    R : Type u_2
    inst✝ : Ring R
    f g k : MulRingNorm R
    hgk : g.equiv k
    c : Real
    hcPos : LT.lt 0 c
    hfg : Eq (fun x => HPow.hPow (f x) c) ⇑g
    ⊢ f.equiv k
  -/
  rcases hgk with ⟨d, hdPos, hgk⟩
  /-
    case intro.intro.intro.intro
    R : Type u_2
    inst✝ : Ring R
    f g k : MulRingNorm R
    c : Real
    hcPos : LT.lt 0 c
    hfg : Eq (fun x => HPow.hPow (f x) c) ⇑g
    d : Real
    hdPos : LT.lt 0 d
    hgk : Eq (fun x => HPow.hPow (g x) d) ⇑k
    ⊢ f.equiv k
  -/
  refine ⟨c*d, (mul_pos_iff_of_pos_left hcPos).mpr hdPos, ?_⟩
  /-
    case intro.intro.intro.intro
    R : Type u_2
    inst✝ : Ring R
    f g k : MulRingNorm R
    c : Real
    hcPos : LT.lt 0 c
    hfg : Eq (fun x => HPow.hPow (f x) c) ⇑g
    d : Real
    hdPos : LT.lt 0 d
    hgk : Eq (fun x => HPow.hPow (g x) d) ⇑k
    ⊢ Eq (fun x => HPow.hPow (f x) (HMul.hMul c d)) ⇑k
  -/
  ext x
  /-
    case intro.intro.intro.intro.h
    R : Type u_2
    inst✝ : Ring R
    f g k : MulRingNorm R
    c : Real
    hcPos : LT.lt 0 c
    hfg : Eq (fun x => HPow.hPow (f x) c) ⇑g
    d : Real
    hdPos : LT.lt 0 d
    hgk : Eq (fun x => HPow.hPow (g x) d) ⇑k
    x : R
    ⊢ Eq (HPow.hPow (f x) (HMul.hMul c d)) (k x)
  -/
  rw [Real.rpow_mul (apply_nonneg f x), congr_fun hfg x, congr_fun hgk x]
  /-
    🎉 no goals
  -/


/-- A nonzero ring seminorm on a field `K` is a ring norm. -/
def RingSeminorm.toRingNorm {K : Type*} [Field K] (f : RingSeminorm K) (hnt : f ≠ 0) :
    RingNorm K :=
  { f with
    eq_zero_of_map_eq_zero' := fun x hx => by
      /-
        R : Type u_1
        K : Type u_2
        inst✝ : Field K
        f : RingSeminorm K
        hnt : Ne f 0
        x : K
        hx : Eq (f.toFun x) 0
        ⊢ Eq x 0
      -/
      obtain ⟨c, hc⟩ := RingSeminorm.ne_zero_iff.mp hnt
      /-
        case intro
        R : Type u_1
        K : Type u_2
        inst✝ : Field K
        f : RingSeminorm K
        hnt : Ne f 0
        x : K
        hx : Eq (f.toFun x) 0
        c : K
        hc : Ne (f c) 0
        ⊢ Eq x 0
      -/
      by_contra hn0
      have hc0 : f c = 0 := by
        rw [← mul_one c, ← mul_inv_cancel₀ hn0, ← mul_assoc, mul_comm c, mul_assoc]
        exact
          le_antisymm
            (le_trans (map_mul_le_mul f _ _)
              (by rw [← RingSeminorm.toFun_eq_coe, ← AddGroupSeminorm.toFun_eq_coe, hx,
                zero_mul]))
            (apply_nonneg f _)
      /-
        case intro
        R : Type u_1
        K : Type u_2
        inst✝ : Field K
        f : RingSeminorm K
        hnt : Ne f 0
        x : K
        hx : Eq (f.toFun x) 0
        c : K
        hc : Ne (f c) 0
        hn0 : Not (Eq x 0)
        hc0 : Eq (f c) 0
        ⊢ False
      -/
      exact hc hc0 }
      /-
        🎉 no goals
      -/


/-- The norm of a `NonUnitalNormedRing` as a `RingNorm`. -/
@[simps!]
def normRingNorm (R : Type*) [NonUnitalNormedRing R] : RingNorm R :=
  { normAddGroupNorm R, normRingSeminorm R with }



/-- A multiplicative ring norm satisfies `f n ≤ n` for every `n : ℕ`. -/
lemma MulRingNorm_nat_le_nat {R : Type*} [Ring R] (n : ℕ) (f : MulRingNorm R) : f n ≤ n := by
  induction n with
  | zero => simp only [Nat.cast_zero, map_zero, le_refl]
  | succ n hn =>
    simp only [Nat.cast_succ]
    calc
      f (n + 1) ≤ f (n) + f 1 := f.add_le' ↑n 1
      _ = f (n) + 1 := by rw [map_one]
      _ ≤ n + 1 := add_le_add_right hn 1


/-- A multiplicative norm composed with the absolute value on integers equals the norm itself. -/
lemma MulRingNorm.apply_natAbs_eq {R : Type*} [Ring R] (x : ℤ) (f : MulRingNorm R) : f (natAbs x) =
    f x := by
  /-
    R : Type u_2
    inst✝ : Ring R
    x : Int
    f : MulRingNorm R
    ⊢ Eq (f ↑x.natAbs) (f ↑x)
  -/
  obtain ⟨n, rfl | rfl⟩ := eq_nat_or_neg x <;>
  /-
    case intro.inl
    R : Type u_2
    inst✝ : Ring R
    f : MulRingNorm R
    n : Nat
    ⊢ Eq (f ↑(↑n).natAbs) (f ↑↑n)
  -/
  /-
    🎉 no goals
  -/
  simp only [natAbs_neg, natAbs_ofNat, cast_neg, cast_natCast, map_neg_eq_map]
  /-
    🎉 no goals
  -/


/-- The seminorm on a `SeminormedRing`, as a `RingSeminorm`. -/
def SeminormedRing.toRingSeminorm (R : Type*) [SeminormedRing R] : RingSeminorm R where
  toFun     := norm
  map_zero' := norm_zero
  add_le'   := norm_add_le
  mul_le'   := norm_mul_le
  neg'      := norm_neg


/-- The norm on a `NormedRing`, as a `RingNorm`. -/
@[simps]
def NormedRing.toRingNorm (R : Type*) [NormedRing R] : RingNorm R where
  toFun     := norm
  map_zero' := norm_zero
  add_le'   := norm_add_le
  mul_le'   := norm_mul_le
  neg'      := norm_neg
                                     /-
                                       R✝ : Type u_1
                                       R : Type u_2
                                       inst✝ : NormedRing R
                                       x : R
                                       hx : Eq ({ toFun := Norm.norm, map_zero' := ⋯, add_le' := ⋯, neg' := ⋯, mul_le …
                                       ⊢ Eq x 0
                                     -/
  eq_zero_of_map_eq_zero' x hx := by rw [← norm_eq_zero]; exact hx
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
theorem NormedRing.toRingNorm_apply (R : Type*) [NormedRing R] (x : R) :
    (NormedRing.toRingNorm R) x = ‖x‖ :=
  rfl


/-- The norm on a `NormedField`, as a `MulRingNorm`. -/
def NormedField.toMulRingNorm (R : Type*) [NormedField R] : MulRingNorm R where
  toFun     := norm
  map_zero' := norm_zero
  map_one'  := norm_one
  add_le'   := norm_add_le
  map_mul'  := norm_mul
  neg'      := norm_neg
                                     /-
                                       R✝ : Type u_1
                                       R : Type u_2
                                       inst✝ : NormedField R
                                       x : R
                                       hx : Eq ({ toFun := Norm.norm, map_zero' := ⋯, add_le' := ⋯, neg' := ⋯, map_on …
                                       ⊢ Eq x 0
                                     -/
  eq_zero_of_map_eq_zero' x hx := by rw [← norm_eq_zero]; exact hx
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- Triangle inequality for `MulRingNorm` applied to a list. -/
lemma mulRingNorm_sum_le_sum_mulRingNorm {R : Type*} [NonAssocRing R] (l : List R)
    (f : MulRingNorm R) : f l.sum ≤ (l.map f).sum := by
  induction l with
  | nil => simp only [List.sum_nil, map_zero, List.map_nil, le_refl]
  | cons head tail ih =>
    simp only [List.sum_cons, List.map_cons]
    calc f (head + List.sum tail) ≤ f head + f (List.sum tail) := by apply f.add_le'
      _ ≤ f head + List.sum (List.map f tail) := by simp only [add_le_add_iff_left, ih]


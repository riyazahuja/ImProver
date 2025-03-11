/-- The nonnegative real power function `x^y`, defined for `x : ℝ≥0` and `y : ℝ` as the
restriction of the real power function. For `x > 0`, it is equal to `exp (y log x)`. For `x = 0`,
one sets `0 ^ 0 = 1` and `0 ^ y = 0` for `y ≠ 0`. -/
noncomputable def rpow (x : ℝ≥0) (y : ℝ) : ℝ≥0 :=
  ⟨(x : ℝ) ^ y, Real.rpow_nonneg x.2 y⟩


noncomputable instance : Pow ℝ≥0 ℝ :=
  ⟨rpow⟩


@[simp]
theorem rpow_eq_pow (x : ℝ≥0) (y : ℝ) : rpow x y = x ^ y :=
  rfl


@[simp, norm_cast]
theorem coe_rpow (x : ℝ≥0) (y : ℝ) : ((x ^ y : ℝ≥0) : ℝ) = (x : ℝ) ^ y :=
  rfl


@[simp]
theorem rpow_zero (x : ℝ≥0) : x ^ (0 : ℝ) = 1 :=
  NNReal.eq <| Real.rpow_zero _


@[simp]
theorem rpow_eq_zero_iff {x : ℝ≥0} {y : ℝ} : x ^ y = 0 ↔ x = 0 ∧ y ≠ 0 := by
  /-
    x : NNReal
    y : Real
    ⊢ Iff (Eq (HPow.hPow x y) 0) (And (Eq x 0) (Ne y 0))
  -/
  rw [← NNReal.coe_inj, coe_rpow, ← NNReal.coe_eq_zero]
  /-
    x : NNReal
    y : Real
    ⊢ Iff (Eq (HPow.hPow (↑x) y) ↑0) (And (Eq (↑x) 0) (Ne y 0))
  -/
  exact Real.rpow_eq_zero_iff_of_nonneg x.2
  /-
    🎉 no goals
  -/


                                                          /-
                                                            x : NNReal
                                                            y : Real
                                                            hy : Ne y 0
                                                            ⊢ Iff (Eq (HPow.hPow x y) 0) (Eq x 0)
                                                          -/
lemma rpow_eq_zero (hy : y ≠ 0) : x ^ y = 0 ↔ x = 0 := by simp [hy]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
theorem zero_rpow {x : ℝ} (h : x ≠ 0) : (0 : ℝ≥0) ^ x = 0 :=
  NNReal.eq <| Real.zero_rpow h


@[simp]
theorem rpow_one (x : ℝ≥0) : x ^ (1 : ℝ) = x :=
  NNReal.eq <| Real.rpow_one _


lemma rpow_neg (x : ℝ≥0) (y : ℝ) : x ^ (-y) = (x ^ y)⁻¹ :=
  NNReal.eq <| Real.rpow_neg x.2 _


@[simp, norm_cast]
lemma rpow_natCast (x : ℝ≥0) (n : ℕ) : x ^ (n : ℝ) = x ^ n :=
                  /-
                    x : NNReal
                    n : Nat
                    ⊢ Eq ↑(HPow.hPow x ↑n) ↑(HPow.hPow x n)
                  -/
  NNReal.eq <| by simpa only [coe_rpow, coe_pow] using Real.rpow_natCast x n
                  /-
                    🎉 no goals
                  -/


@[simp, norm_cast]
lemma rpow_intCast (x : ℝ≥0) (n : ℤ) : x ^ (n : ℝ) = x ^ n := by
  /-
    x : NNReal
    n : Int
    ⊢ Eq (HPow.hPow x ↑n) (HPow.hPow x n)
  -/
  cases n <;> simp only [Int.ofNat_eq_coe, Int.cast_natCast, rpow_natCast, zpow_natCast,
    Int.cast_negSucc, rpow_neg, zpow_negSucc]


@[simp]
theorem one_rpow (x : ℝ) : (1 : ℝ≥0) ^ x = 1 :=
  NNReal.eq <| Real.one_rpow _


theorem rpow_add {x : ℝ≥0} (hx : x ≠ 0) (y z : ℝ) : x ^ (y + z) = x ^ y * x ^ z :=
  NNReal.eq <| Real.rpow_add ((NNReal.coe_pos.trans pos_iff_ne_zero).mpr hx) _ _


theorem rpow_add' (h : y + z ≠ 0) (x : ℝ≥0) : x ^ (y + z) = x ^ y * x ^ z :=
  NNReal.eq <| Real.rpow_add' x.2 h


lemma rpow_add_intCast (hx : x ≠ 0) (y : ℝ) (n : ℤ) : x ^ (y + n) = x ^ y * x ^ n := by
  /-
    x : NNReal
    hx : Ne x 0
    y : Real
    n : Int
    ⊢ Eq (HPow.hPow x (HAdd.hAdd y ↑n)) (HMul.hMul (HPow.hPow x y) (HPow.hPow x n))
  -/
  ext; exact Real.rpow_add_intCast (mod_cast hx) _ _
       /-
         🎉 no goals
       -/


lemma rpow_add_natCast (hx : x ≠ 0) (y : ℝ) (n : ℕ) : x ^ (y + n) = x ^ y * x ^ n := by
  /-
    x : NNReal
    hx : Ne x 0
    y : Real
    n : Nat
    ⊢ Eq (HPow.hPow x (HAdd.hAdd y ↑n)) (HMul.hMul (HPow.hPow x y) (HPow.hPow x n))
  -/
  ext; exact Real.rpow_add_natCast (mod_cast hx) _ _
       /-
         🎉 no goals
       -/


lemma rpow_sub_intCast (hx : x ≠ 0) (y : ℝ) (n : ℕ) : x ^ (y - n) = x ^ y / x ^ n := by
  /-
    x : NNReal
    hx : Ne x 0
    y : Real
    n : Nat
    ⊢ Eq (HPow.hPow x (HSub.hSub y ↑n)) (HDiv.hDiv (HPow.hPow x y) (HPow.hPow x n))
  -/
  ext; exact Real.rpow_sub_intCast (mod_cast hx) _ _
       /-
         🎉 no goals
       -/


lemma rpow_sub_natCast (hx : x ≠ 0) (y : ℝ) (n : ℕ) : x ^ (y - n) = x ^ y / x ^ n := by
  /-
    x : NNReal
    hx : Ne x 0
    y : Real
    n : Nat
    ⊢ Eq (HPow.hPow x (HSub.hSub y ↑n)) (HDiv.hDiv (HPow.hPow x y) (HPow.hPow x n))
  -/
  ext; exact Real.rpow_sub_natCast (mod_cast hx) _ _
       /-
         🎉 no goals
       -/


lemma rpow_add_intCast' {n : ℤ} (h : y + n ≠ 0) (x : ℝ≥0) : x ^ (y + n) = x ^ y * x ^ n := by
  /-
    y : Real
    n : Int
    h : Ne (HAdd.hAdd y ↑n) 0
    x : NNReal
    ⊢ Eq (HPow.hPow x (HAdd.hAdd y ↑n)) (HMul.hMul (HPow.hPow x y) (HPow.hPow x n))
  -/
  ext; exact Real.rpow_add_intCast' (mod_cast x.2) h
       /-
         🎉 no goals
       -/


lemma rpow_add_natCast' {n : ℕ} (h : y + n ≠ 0) (x : ℝ≥0) : x ^ (y + n) = x ^ y * x ^ n := by
  /-
    y : Real
    n : Nat
    h : Ne (HAdd.hAdd y ↑n) 0
    x : NNReal
    ⊢ Eq (HPow.hPow x (HAdd.hAdd y ↑n)) (HMul.hMul (HPow.hPow x y) (HPow.hPow x n))
  -/
  ext; exact Real.rpow_add_natCast' (mod_cast x.2) h
       /-
         🎉 no goals
       -/


lemma rpow_sub_intCast' {n : ℤ} (h : y - n ≠ 0) (x : ℝ≥0) : x ^ (y - n) = x ^ y / x ^ n := by
  /-
    y : Real
    n : Int
    h : Ne (HSub.hSub y ↑n) 0
    x : NNReal
    ⊢ Eq (HPow.hPow x (HSub.hSub y ↑n)) (HDiv.hDiv (HPow.hPow x y) (HPow.hPow x n))
  -/
  ext; exact Real.rpow_sub_intCast' (mod_cast x.2) h
       /-
         🎉 no goals
       -/


lemma rpow_sub_natCast' {n : ℕ} (h : y - n ≠ 0) (x : ℝ≥0) : x ^ (y - n) = x ^ y / x ^ n := by
  /-
    y : Real
    n : Nat
    h : Ne (HSub.hSub y ↑n) 0
    x : NNReal
    ⊢ Eq (HPow.hPow x (HSub.hSub y ↑n)) (HDiv.hDiv (HPow.hPow x y) (HPow.hPow x n))
  -/
  ext; exact Real.rpow_sub_natCast' (mod_cast x.2) h
       /-
         🎉 no goals
       -/


lemma rpow_add_one (hx : x ≠ 0) (y : ℝ) : x ^ (y + 1) = x ^ y * x := by
  /-
    x : NNReal
    hx : Ne x 0
    y : Real
    ⊢ Eq (HPow.hPow x (HAdd.hAdd y 1)) (HMul.hMul (HPow.hPow x y) x)
  -/
  simpa using rpow_add_natCast hx y 1
  /-
    🎉 no goals
  -/


lemma rpow_sub_one (hx : x ≠ 0) (y : ℝ) : x ^ (y - 1) = x ^ y / x := by
  /-
    x : NNReal
    hx : Ne x 0
    y : Real
    ⊢ Eq (HPow.hPow x (HSub.hSub y 1)) (HDiv.hDiv (HPow.hPow x y) x)
  -/
  simpa using rpow_sub_natCast hx y 1
  /-
    🎉 no goals
  -/


lemma rpow_add_one' (h : y + 1 ≠ 0) (x : ℝ≥0) : x ^ (y + 1) = x ^ y * x := by
  /-
    y : Real
    h : Ne (HAdd.hAdd y 1) 0
    x : NNReal
    ⊢ Eq (HPow.hPow x (HAdd.hAdd y 1)) (HMul.hMul (HPow.hPow x y) x)
  -/
  rw [rpow_add' h, rpow_one]
  /-
    🎉 no goals
  -/


lemma rpow_one_add' (h : 1 + y ≠ 0) (x : ℝ≥0) : x ^ (1 + y) = x * x ^ y := by
  /-
    y : Real
    h : Ne (HAdd.hAdd 1 y) 0
    x : NNReal
    ⊢ Eq (HPow.hPow x (HAdd.hAdd 1 y)) (HMul.hMul x (HPow.hPow x y))
  -/
  rw [rpow_add' h, rpow_one]
  /-
    🎉 no goals
  -/


theorem rpow_add_of_nonneg (x : ℝ≥0) {y z : ℝ} (hy : 0 ≤ y) (hz : 0 ≤ z) :
    x ^ (y + z) = x ^ y * x ^ z := by
  /-
    x : NNReal
    y z : Real
    hy : LE.le 0 y
    hz : LE.le 0 z
    ⊢ Eq (HPow.hPow x (HAdd.hAdd y z)) (HMul.hMul (HPow.hPow x y) (HPow.hPow x z))
  -/
  ext; exact Real.rpow_add_of_nonneg x.2 hy hz
       /-
         🎉 no goals
       -/


/-- Variant of `NNReal.rpow_add'` that avoids having to prove `y + z = w` twice. -/
lemma rpow_of_add_eq (x : ℝ≥0) (hw : w ≠ 0) (h : y + z = w) : x ^ w = x ^ y * x ^ z := by
  /-
    w y z : Real
    x : NNReal
    hw : Ne w 0
    h : Eq (HAdd.hAdd y z) w
    ⊢ Eq (HPow.hPow x w) (HMul.hMul (HPow.hPow x y) (HPow.hPow x z))
  -/
  rw [← h, rpow_add']; rwa [h]
                       /-
                         🎉 no goals
                       -/


theorem rpow_mul (x : ℝ≥0) (y z : ℝ) : x ^ (y * z) = (x ^ y) ^ z :=
  NNReal.eq <| Real.rpow_mul x.2 y z


lemma rpow_natCast_mul (x : ℝ≥0) (n : ℕ) (z : ℝ) : x ^ (n * z) = (x ^ n) ^ z := by
  /-
    x : NNReal
    n : Nat
    z : Real
    ⊢ Eq (HPow.hPow x (HMul.hMul (↑n) z)) (HPow.hPow (HPow.hPow x n) z)
  -/
  rw [rpow_mul, rpow_natCast]
  /-
    🎉 no goals
  -/


lemma rpow_mul_natCast (x : ℝ≥0) (y : ℝ) (n : ℕ) : x ^ (y * n) = (x ^ y) ^ n := by
  /-
    x : NNReal
    y : Real
    n : Nat
    ⊢ Eq (HPow.hPow x (HMul.hMul y ↑n)) (HPow.hPow (HPow.hPow x y) n)
  -/
  rw [rpow_mul, rpow_natCast]
  /-
    🎉 no goals
  -/


lemma rpow_intCast_mul (x : ℝ≥0) (n : ℤ) (z : ℝ) : x ^ (n * z) = (x ^ n) ^ z := by
  /-
    x : NNReal
    n : Int
    z : Real
    ⊢ Eq (HPow.hPow x (HMul.hMul (↑n) z)) (HPow.hPow (HPow.hPow x n) z)
  -/
  rw [rpow_mul, rpow_intCast]
  /-
    🎉 no goals
  -/


lemma rpow_mul_intCast (x : ℝ≥0) (y : ℝ) (n : ℤ) : x ^ (y * n) = (x ^ y) ^ n := by
  /-
    x : NNReal
    y : Real
    n : Int
    ⊢ Eq (HPow.hPow x (HMul.hMul y ↑n)) (HPow.hPow (HPow.hPow x y) n)
  -/
  rw [rpow_mul, rpow_intCast]
  /-
    🎉 no goals
  -/


                                                          /-
                                                            x : NNReal
                                                            ⊢ Eq (HPow.hPow x (-1)) (Inv.inv x)
                                                          -/
theorem rpow_neg_one (x : ℝ≥0) : x ^ (-1 : ℝ) = x⁻¹ := by simp [rpow_neg]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem rpow_sub {x : ℝ≥0} (hx : x ≠ 0) (y z : ℝ) : x ^ (y - z) = x ^ y / x ^ z :=
  NNReal.eq <| Real.rpow_sub ((NNReal.coe_pos.trans pos_iff_ne_zero).mpr hx) y z


theorem rpow_sub' (h : y - z ≠ 0) (x : ℝ≥0) : x ^ (y - z) = x ^ y / x ^ z :=
  NNReal.eq <| Real.rpow_sub' x.2 h


lemma rpow_sub_one' (h : y - 1 ≠ 0) (x : ℝ≥0) : x ^ (y - 1) = x ^ y / x := by
  /-
    y : Real
    h : Ne (HSub.hSub y 1) 0
    x : NNReal
    ⊢ Eq (HPow.hPow x (HSub.hSub y 1)) (HDiv.hDiv (HPow.hPow x y) x)
  -/
  rw [rpow_sub' h, rpow_one]
  /-
    🎉 no goals
  -/


lemma rpow_one_sub' (h : 1 - y ≠ 0) (x : ℝ≥0) : x ^ (1 - y) = x / x ^ y := by
  /-
    y : Real
    h : Ne (HSub.hSub 1 y) 0
    x : NNReal
    ⊢ Eq (HPow.hPow x (HSub.hSub 1 y)) (HDiv.hDiv x (HPow.hPow x y))
  -/
  rw [rpow_sub' h, rpow_one]
  /-
    🎉 no goals
  -/


theorem rpow_inv_rpow_self {y : ℝ} (hy : y ≠ 0) (x : ℝ≥0) : (x ^ y) ^ (1 / y) = x := by
  /-
    y : Real
    hy : Ne y 0
    x : NNReal
    ⊢ Eq (HPow.hPow (HPow.hPow x y) (HDiv.hDiv 1 y)) x
  -/
  field_simp [← rpow_mul]
  /-
    🎉 no goals
  -/


theorem rpow_self_rpow_inv {y : ℝ} (hy : y ≠ 0) (x : ℝ≥0) : (x ^ (1 / y)) ^ y = x := by
  /-
    y : Real
    hy : Ne y 0
    x : NNReal
    ⊢ Eq (HPow.hPow (HPow.hPow x (HDiv.hDiv 1 y)) y) x
  -/
  field_simp [← rpow_mul]
  /-
    🎉 no goals
  -/


theorem inv_rpow (x : ℝ≥0) (y : ℝ) : x⁻¹ ^ y = (x ^ y)⁻¹ :=
  NNReal.eq <| Real.inv_rpow x.2 y


theorem div_rpow (x y : ℝ≥0) (z : ℝ) : (x / y) ^ z = x ^ z / y ^ z :=
  NNReal.eq <| Real.div_rpow x.2 y.2 z


theorem sqrt_eq_rpow (x : ℝ≥0) : sqrt x = x ^ (1 / (2 : ℝ)) := by
  /-
    x : NNReal
    ⊢ Eq (NNReal.sqrt x) (HPow.hPow x (1 / 2))
  -/
  refine NNReal.eq ?_
  /-
    x : NNReal
    ⊢ Eq ↑(NNReal.sqrt x) ↑(HPow.hPow x (1 / 2))
  -/
  push_cast
  /-
    x : NNReal
    ⊢ Eq (↑x).sqrt (HPow.hPow (↑x) (1 / 2))
  -/
  exact Real.sqrt_eq_rpow x.1
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias rpow_nat_cast := rpow_natCast


@[simp]
lemma rpow_ofNat (x : ℝ≥0) (n : ℕ) [n.AtLeastTwo] :
    x ^ (no_index (OfNat.ofNat n) : ℝ) = x ^ (OfNat.ofNat n : ℕ) :=
  rpow_natCast x n


theorem rpow_two (x : ℝ≥0) : x ^ (2 : ℝ) = x ^ 2 := rpow_ofNat x 2


theorem mul_rpow {x y : ℝ≥0} {z : ℝ} : (x * y) ^ z = x ^ z * y ^ z :=
  NNReal.eq <| Real.mul_rpow x.2 y.2


/-- `rpow` as a `MonoidHom`-/
@[simps]
def rpowMonoidHom (r : ℝ) : ℝ≥0 →* ℝ≥0 where
  toFun := (· ^ r)
  map_one' := one_rpow _
  map_mul' _x _y := mul_rpow


/-- `rpow` variant of `List.prod_map_pow` for `ℝ≥0`-/
theorem list_prod_map_rpow (l : List ℝ≥0) (r : ℝ) :
    (l.map (· ^ r)).prod = l.prod ^ r :=
  l.prod_hom (rpowMonoidHom r)


theorem list_prod_map_rpow' {ι} (l : List ι) (f : ι → ℝ≥0) (r : ℝ) :
    (l.map (f · ^ r)).prod = (l.map f).prod ^ r := by
  /-
    ι : Type u_1
    l : List ι
    f : ι → NNReal
    r : Real
    ⊢ Eq (List.map (fun x => HPow.hPow (f x) r) l).prod (HPow.hPow (List.map f l). …
  -/
  rw [← list_prod_map_rpow, List.map_map]; rfl
                                           /-
                                             🎉 no goals
                                           -/


/-- `rpow` version of `Multiset.prod_map_pow` for `ℝ≥0`. -/
lemma multiset_prod_map_rpow {ι} (s : Multiset ι) (f : ι → ℝ≥0) (r : ℝ) :
    (s.map (f · ^ r)).prod = (s.map f).prod ^ r :=
  s.prod_hom' (rpowMonoidHom r) _


/-- `rpow` version of `Finset.prod_pow` for `ℝ≥0`. -/
lemma finset_prod_rpow {ι} (s : Finset ι) (f : ι → ℝ≥0) (r : ℝ) :
    (∏ i ∈ s, f i ^ r) = (∏ i ∈ s, f i) ^ r :=
  multiset_prod_map_rpow _ _ _

-- note: these don't really belong here, but they're much easier to prove in terms of the above


/-- `rpow` version of `List.prod_map_pow` for `Real`. -/
theorem _root_.Real.list_prod_map_rpow (l : List ℝ) (hl : ∀ x ∈ l, (0 : ℝ) ≤ x) (r : ℝ) :
    (l.map (· ^ r)).prod = l.prod ^ r := by
  /-
    l : List Real
    hl : ∀ (x : Real), Membership.mem l x → LE.le 0 x
    r : Real
    ⊢ Eq (List.map (fun x => HPow.hPow x r) l).prod (HPow.hPow l.prod r)
  -/
  lift l to List ℝ≥0 using hl
  /-
    case intro
    r : Real
    l : List NNReal
    ⊢ Eq (List.map (fun x => HPow.hPow x r) (List.map NNReal.toReal l)).prod (HPow …
  -/
  have := congr_arg ((↑) : ℝ≥0 → ℝ) (NNReal.list_prod_map_rpow l r)
  /-
    case intro
    r : Real
    l : List NNReal
    this : Eq ↑(List.map (fun x => HPow.hPow x r) l).prod ↑(HPow.hPow l.prod r)
    ⊢ Eq (List.map (fun x => HPow.hPow x r) (List.map NNReal.toReal l)).prod (HPow …
  -/
  push_cast at this
  /-
    case intro
    r : Real
    l : List NNReal
    this : Eq (List.map NNReal.toReal (List.map (fun x => HPow.hPow x r) l)).prod  …
    ⊢ Eq (List.map (fun x => HPow.hPow x r) (List.map NNReal.toReal l)).prod (HPow …
  -/
  rw [List.map_map] at this ⊢
  /-
    case intro
    r : Real
    l : List NNReal
    this : Eq (List.map (Function.comp NNReal.toReal fun x => HPow.hPow x r) l).pr …
    ⊢ Eq (List.map (Function.comp (fun x => HPow.hPow x r) NNReal.toReal) l).prod  …
  -/
  exact mod_cast this
  /-
    🎉 no goals
  -/


theorem _root_.Real.list_prod_map_rpow' {ι} (l : List ι) (f : ι → ℝ)
    (hl : ∀ i ∈ l, (0 : ℝ) ≤ f i) (r : ℝ) :
    (l.map (f · ^ r)).prod = (l.map f).prod ^ r := by
  /-
    ι : Type u_1
    l : List ι
    f : ι → Real
    hl : ∀ (i : ι), Membership.mem l i → LE.le 0 (f i)
    r : Real
    ⊢ Eq (List.map (fun x => HPow.hPow (f x) r) l).prod (HPow.hPow (List.map f l). …
  -/
  rw [← Real.list_prod_map_rpow (l.map f) _ r, List.map_map]
    /-
      ι : Type u_1
      l : List ι
      f : ι → Real
      hl : ∀ (i : ι), Membership.mem l i → LE.le 0 (f i)
      r : Real
      ⊢ Eq (List.map (fun x => HPow.hPow (f x) r) l).prod (List.map (Function.comp ( …
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    ι : Type u_1
    l : List ι
    f : ι → Real
    hl : ∀ (i : ι), Membership.mem l i → LE.le 0 (f i)
    r : Real
    ⊢ ∀ (x : Real), Membership.mem (List.map f l) x → LE.le 0 x
  -/
  simpa using hl
  /-
    🎉 no goals
  -/


/-- `rpow` version of `Multiset.prod_map_pow`. -/
theorem _root_.Real.multiset_prod_map_rpow {ι} (s : Multiset ι) (f : ι → ℝ)
    (hs : ∀ i ∈ s, (0 : ℝ) ≤ f i) (r : ℝ) :
    (s.map (f · ^ r)).prod = (s.map f).prod ^ r := by
  /-
    ι : Type u_1
    s : Multiset ι
    f : ι → Real
    hs : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
    r : Real
    ⊢ Eq (Multiset.map (fun x => HPow.hPow (f x) r) s).prod (HPow.hPow (Multiset.m …
  -/
  induction' s using Quotient.inductionOn with l
  /-
    case h
    ι : Type u_1
    f : ι → Real
    r : Real
    l : List ι
    hs : ∀ (i : ι), Membership.mem (Quotient.mk (List.isSetoid ι) l) i → LE.le 0 ( …
    ⊢ Eq (Multiset.map (fun x => HPow.hPow (f x) r) (Quotient.mk (List.isSetoid ι) …
  -/
  simpa using Real.list_prod_map_rpow' l f hs r
  /-
    🎉 no goals
  -/


/-- `rpow` version of `Finset.prod_pow`. -/
theorem _root_.Real.finset_prod_rpow
    {ι} (s : Finset ι) (f : ι → ℝ) (hs : ∀ i ∈ s, 0 ≤ f i) (r : ℝ) :
    (∏ i ∈ s, f i ^ r) = (∏ i ∈ s, f i) ^ r :=
  Real.multiset_prod_map_rpow s.val f hs r


@[gcongr] theorem rpow_le_rpow {x y : ℝ≥0} {z : ℝ} (h₁ : x ≤ y) (h₂ : 0 ≤ z) : x ^ z ≤ y ^ z :=
  Real.rpow_le_rpow x.2 h₁ h₂


@[gcongr] theorem rpow_lt_rpow {x y : ℝ≥0} {z : ℝ} (h₁ : x < y) (h₂ : 0 < z) : x ^ z < y ^ z :=
  Real.rpow_lt_rpow x.2 h₁ h₂


theorem rpow_lt_rpow_iff {x y : ℝ≥0} {z : ℝ} (hz : 0 < z) : x ^ z < y ^ z ↔ x < y :=
  Real.rpow_lt_rpow_iff x.2 y.2 hz


theorem rpow_le_rpow_iff {x y : ℝ≥0} {z : ℝ} (hz : 0 < z) : x ^ z ≤ y ^ z ↔ x ≤ y :=
  Real.rpow_le_rpow_iff x.2 y.2 hz


theorem le_rpow_inv_iff {x y : ℝ≥0} {z : ℝ} (hz : 0 < z) : x ≤ y ^ z⁻¹ ↔ x ^ z ≤ y := by
  /-
    x y : NNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LE.le x (HPow.hPow y (Inv.inv z))) (LE.le (HPow.hPow x z) y)
  -/
  rw [← rpow_le_rpow_iff hz, ← one_div, rpow_self_rpow_inv hz.ne']
  /-
    🎉 no goals
  -/


@[deprecated le_rpow_inv_iff (since := "2024-07-10")]
theorem le_rpow_one_div_iff {x y : ℝ≥0} {z : ℝ} (hz : 0 < z) : x ≤ y ^ (1 / z) ↔ x ^ z ≤ y := by
  /-
    x y : NNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LE.le x (HPow.hPow y (HDiv.hDiv 1 z))) (LE.le (HPow.hPow x z) y)
  -/
  rw [← rpow_le_rpow_iff hz, rpow_self_rpow_inv hz.ne']
  /-
    🎉 no goals
  -/


theorem rpow_inv_le_iff {x y : ℝ≥0} {z : ℝ} (hz : 0 < z) : x ^ z⁻¹ ≤ y ↔ x ≤ y ^ z := by
  /-
    x y : NNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LE.le (HPow.hPow x (Inv.inv z)) y) (LE.le x (HPow.hPow y z))
  -/
  rw [← rpow_le_rpow_iff hz, ← one_div, rpow_self_rpow_inv hz.ne']
  /-
    🎉 no goals
  -/


@[deprecated rpow_inv_le_iff (since := "2024-07-10")]
theorem rpow_one_div_le_iff {x y : ℝ≥0} {z : ℝ} (hz : 0 < z) : x ^ (1 / z) ≤ y ↔ x ≤ y ^ z := by
  /-
    x y : NNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LE.le (HPow.hPow x (HDiv.hDiv 1 z)) y) (LE.le x (HPow.hPow y z))
  -/
  rw [← rpow_le_rpow_iff hz, rpow_self_rpow_inv hz.ne']
  /-
    🎉 no goals
  -/


theorem lt_rpow_inv_iff {x y : ℝ≥0} {z : ℝ} (hz : 0 < z) : x < y ^ z⁻¹ ↔ x ^z < y := by
  /-
    x y : NNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LT.lt x (HPow.hPow y (Inv.inv z))) (LT.lt (HPow.hPow x z) y)
  -/
  simp only [← not_le, rpow_inv_le_iff hz]
  /-
    🎉 no goals
  -/


theorem rpow_inv_lt_iff {x y : ℝ≥0} {z : ℝ} (hz : 0 < z) : x ^ z⁻¹ < y ↔ x < y ^ z := by
  /-
    x y : NNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LT.lt (HPow.hPow x (Inv.inv z)) y) (LT.lt x (HPow.hPow y z))
  -/
  simp only [← not_le, le_rpow_inv_iff hz]
  /-
    🎉 no goals
  -/


lemma rpow_lt_rpow_of_neg (hx : 0 < x) (hxy : x < y) (hz : z < 0) : y ^ z < x ^ z :=
  Real.rpow_lt_rpow_of_neg hx hxy hz


lemma rpow_le_rpow_of_nonpos (hx : 0 < x) (hxy : x ≤ y) (hz : z ≤ 0) : y ^ z ≤ x ^ z :=
  Real.rpow_le_rpow_of_nonpos hx hxy hz


lemma rpow_lt_rpow_iff_of_neg (hx : 0 < x) (hy : 0 < y) (hz : z < 0) : x ^ z < y ^ z ↔ y < x :=
  Real.rpow_lt_rpow_iff_of_neg hx hy hz


lemma rpow_le_rpow_iff_of_neg (hx : 0 < x) (hy : 0 < y) (hz : z < 0) : x ^ z ≤ y ^ z ↔ y ≤ x :=
  Real.rpow_le_rpow_iff_of_neg hx hy hz


lemma le_rpow_inv_iff_of_pos (hy : 0 ≤ y) (hz : 0 < z) (x : ℝ≥0) : x ≤ y ^ z⁻¹ ↔ x ^ z ≤ y :=
  Real.le_rpow_inv_iff_of_pos x.2 hy hz


lemma rpow_inv_le_iff_of_pos (hy : 0 ≤ y) (hz : 0 < z) (x : ℝ≥0) : x ^ z⁻¹ ≤ y ↔ x ≤ y ^ z :=
  Real.rpow_inv_le_iff_of_pos x.2 hy hz


lemma lt_rpow_inv_iff_of_pos (hy : 0 ≤ y) (hz : 0 < z) (x : ℝ≥0) : x < y ^ z⁻¹ ↔ x ^ z < y :=
  Real.lt_rpow_inv_iff_of_pos x.2 hy hz


lemma rpow_inv_lt_iff_of_pos (hy : 0 ≤ y) (hz : 0 < z) (x : ℝ≥0) : x ^ z⁻¹ < y ↔ x < y ^ z :=
  Real.rpow_inv_lt_iff_of_pos x.2 hy hz


lemma le_rpow_inv_iff_of_neg (hx : 0 < x) (hy : 0 < y) (hz : z < 0) : x ≤ y ^ z⁻¹ ↔ y ≤ x ^ z :=
  Real.le_rpow_inv_iff_of_neg hx hy hz


lemma lt_rpow_inv_iff_of_neg (hx : 0 < x) (hy : 0 < y) (hz : z < 0) : x < y ^ z⁻¹ ↔ y < x ^ z :=
  Real.lt_rpow_inv_iff_of_neg hx hy hz


lemma rpow_inv_lt_iff_of_neg (hx : 0 < x) (hy : 0 < y) (hz : z < 0) : x ^ z⁻¹ < y ↔ y ^ z < x :=
  Real.rpow_inv_lt_iff_of_neg hx hy hz


lemma rpow_inv_le_iff_of_neg (hx : 0 < x) (hy : 0 < y) (hz : z < 0) : x ^ z⁻¹ ≤ y ↔ y ^ z ≤ x :=
  Real.rpow_inv_le_iff_of_neg hx hy hz


@[gcongr] theorem rpow_lt_rpow_of_exponent_lt {x : ℝ≥0} {y z : ℝ} (hx : 1 < x) (hyz : y < z) :
    x ^ y < x ^ z :=
  Real.rpow_lt_rpow_of_exponent_lt hx hyz


@[gcongr] theorem rpow_le_rpow_of_exponent_le {x : ℝ≥0} {y z : ℝ} (hx : 1 ≤ x) (hyz : y ≤ z) :
    x ^ y ≤ x ^ z :=
  Real.rpow_le_rpow_of_exponent_le hx hyz


theorem rpow_lt_rpow_of_exponent_gt {x : ℝ≥0} {y z : ℝ} (hx0 : 0 < x) (hx1 : x < 1) (hyz : z < y) :
    x ^ y < x ^ z :=
  Real.rpow_lt_rpow_of_exponent_gt hx0 hx1 hyz


theorem rpow_le_rpow_of_exponent_ge {x : ℝ≥0} {y z : ℝ} (hx0 : 0 < x) (hx1 : x ≤ 1) (hyz : z ≤ y) :
    x ^ y ≤ x ^ z :=
  Real.rpow_le_rpow_of_exponent_ge hx0 hx1 hyz


theorem rpow_pos {p : ℝ} {x : ℝ≥0} (hx_pos : 0 < x) : 0 < x ^ p := by
  have rpow_pos_of_nonneg : ∀ {p : ℝ}, 0 < p → 0 < x ^ p := by
    intro p hp_pos
    rw [← zero_rpow hp_pos.ne']
    exact rpow_lt_rpow hx_pos hp_pos
  /-
    p : Real
    x : NNReal
    hx_pos : LT.lt 0 x
    rpow_pos_of_nonneg : ∀ {p : Real}, LT.lt 0 p → LT.lt 0 (HPow.hPow x p)
    ⊢ LT.lt 0 (HPow.hPow x p)
  -/
  rcases lt_trichotomy (0 : ℝ) p with (hp_pos | rfl | hp_neg)
    /-
      case inl
      p : Real
      x : NNReal
      hx_pos : LT.lt 0 x
      rpow_pos_of_nonneg : ∀ {p : Real}, LT.lt 0 p → LT.lt 0 (HPow.hPow x p)
      hp_pos : LT.lt 0 p
      ⊢ LT.lt 0 (HPow.hPow x p)
    -/
  · exact rpow_pos_of_nonneg hp_pos
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      x : NNReal
      hx_pos : LT.lt 0 x
      rpow_pos_of_nonneg : ∀ {p : Real}, LT.lt 0 p → LT.lt 0 (HPow.hPow x p)
      ⊢ LT.lt 0 (HPow.hPow x 0)
    -/
  · simp only [zero_lt_one, rpow_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      p : Real
      x : NNReal
      hx_pos : LT.lt 0 x
      rpow_pos_of_nonneg : ∀ {p : Real}, LT.lt 0 p → LT.lt 0 (HPow.hPow x p)
      hp_neg : LT.lt p 0
      ⊢ LT.lt 0 (HPow.hPow x p)
    -/
  · rw [← neg_neg p, rpow_neg, inv_pos]
    /-
      case inr.inr
      p : Real
      x : NNReal
      hx_pos : LT.lt 0 x
      rpow_pos_of_nonneg : ∀ {p : Real}, LT.lt 0 p → LT.lt 0 (HPow.hPow x p)
      hp_neg : LT.lt p 0
      ⊢ LT.lt 0 (HPow.hPow x (Neg.neg p))
    -/
    exact rpow_pos_of_nonneg (neg_pos.mpr hp_neg)
    /-
      🎉 no goals
    -/


theorem rpow_lt_one {x : ℝ≥0} {z : ℝ} (hx1 : x < 1) (hz : 0 < z) : x ^ z < 1 :=
  Real.rpow_lt_one (coe_nonneg x) hx1 hz


theorem rpow_le_one {x : ℝ≥0} {z : ℝ} (hx2 : x ≤ 1) (hz : 0 ≤ z) : x ^ z ≤ 1 :=
  Real.rpow_le_one x.2 hx2 hz


theorem rpow_lt_one_of_one_lt_of_neg {x : ℝ≥0} {z : ℝ} (hx : 1 < x) (hz : z < 0) : x ^ z < 1 :=
  Real.rpow_lt_one_of_one_lt_of_neg hx hz


theorem rpow_le_one_of_one_le_of_nonpos {x : ℝ≥0} {z : ℝ} (hx : 1 ≤ x) (hz : z ≤ 0) : x ^ z ≤ 1 :=
  Real.rpow_le_one_of_one_le_of_nonpos hx hz


theorem one_lt_rpow {x : ℝ≥0} {z : ℝ} (hx : 1 < x) (hz : 0 < z) : 1 < x ^ z :=
  Real.one_lt_rpow hx hz


theorem one_le_rpow {x : ℝ≥0} {z : ℝ} (h : 1 ≤ x) (h₁ : 0 ≤ z) : 1 ≤ x ^ z :=
  Real.one_le_rpow h h₁


theorem one_lt_rpow_of_pos_of_lt_one_of_neg {x : ℝ≥0} {z : ℝ} (hx1 : 0 < x) (hx2 : x < 1)
    (hz : z < 0) : 1 < x ^ z :=
  Real.one_lt_rpow_of_pos_of_lt_one_of_neg hx1 hx2 hz


theorem one_le_rpow_of_pos_of_le_one_of_nonpos {x : ℝ≥0} {z : ℝ} (hx1 : 0 < x) (hx2 : x ≤ 1)
    (hz : z ≤ 0) : 1 ≤ x ^ z :=
  Real.one_le_rpow_of_pos_of_le_one_of_nonpos hx1 hx2 hz


theorem rpow_le_self_of_le_one {x : ℝ≥0} {z : ℝ} (hx : x ≤ 1) (h_one_le : 1 ≤ z) : x ^ z ≤ x := by
  /-
    x : NNReal
    z : Real
    hx : LE.le x 1
    h_one_le : LE.le 1 z
    ⊢ LE.le (HPow.hPow x z) x
  -/
  rcases eq_bot_or_bot_lt x with (rfl | (h : 0 < x))
    /-
      case inl
      z : Real
      h_one_le : LE.le 1 z
      hx : LE.le Bot.bot 1
      ⊢ LE.le (HPow.hPow Bot.bot z) Bot.bot
    -/
  · have : z ≠ 0 := by linarith
    /-
      case inl
      z : Real
      h_one_le : LE.le 1 z
      hx : LE.le Bot.bot 1
      this : Ne z 0
      ⊢ LE.le (HPow.hPow Bot.bot z) Bot.bot
    -/
    simp [this]
    /-
      🎉 no goals
    -/
  /-
    case inr
    x : NNReal
    z : Real
    hx : LE.le x 1
    h_one_le : LE.le 1 z
    h : LT.lt 0 x
    ⊢ LE.le (HPow.hPow x z) x
  -/
  nth_rw 2 [← NNReal.rpow_one x]
  /-
    case inr
    x : NNReal
    z : Real
    hx : LE.le x 1
    h_one_le : LE.le 1 z
    h : LT.lt 0 x
    ⊢ LE.le (HPow.hPow x z) (HPow.hPow x 1)
  -/
  exact NNReal.rpow_le_rpow_of_exponent_ge h hx h_one_le
  /-
    🎉 no goals
  -/


theorem rpow_left_injective {x : ℝ} (hx : x ≠ 0) : Function.Injective fun y : ℝ≥0 => y ^ x :=
                    /-
                      x : Real
                      hx : Ne x 0
                      y z : NNReal
                      hyz : Eq ((fun y => HPow.hPow y x) y) ((fun y => HPow.hPow y x) z)
                      ⊢ Eq y z
                    -/
  fun y z hyz => by simpa only [rpow_inv_rpow_self hx] using congr_arg (fun y => y ^ (1 / x)) hyz
                    /-
                      🎉 no goals
                    -/


theorem rpow_eq_rpow_iff {x y : ℝ≥0} {z : ℝ} (hz : z ≠ 0) : x ^ z = y ^ z ↔ x = y :=
  (rpow_left_injective hz).eq_iff


theorem rpow_left_surjective {x : ℝ} (hx : x ≠ 0) : Function.Surjective fun y : ℝ≥0 => y ^ x :=
                        /-
                          x : Real
                          hx : Ne x 0
                          y : NNReal
                          ⊢ Eq ((fun y => HPow.hPow y x) (HPow.hPow y (Inv.inv x))) y
                        -/
  fun y => ⟨y ^ x⁻¹, by simp_rw [← rpow_mul, inv_mul_cancel₀ hx, rpow_one]⟩
                        /-
                          🎉 no goals
                        -/


theorem rpow_left_bijective {x : ℝ} (hx : x ≠ 0) : Function.Bijective fun y : ℝ≥0 => y ^ x :=
  ⟨rpow_left_injective hx, rpow_left_surjective hx⟩


theorem eq_rpow_inv_iff {x y : ℝ≥0} {z : ℝ} (hz : z ≠ 0) : x = y ^ z⁻¹ ↔ x ^ z = y := by
  /-
    x y : NNReal
    z : Real
    hz : Ne z 0
    ⊢ Iff (Eq x (HPow.hPow y (Inv.inv z))) (Eq (HPow.hPow x z) y)
  -/
  rw [← rpow_eq_rpow_iff hz, ← one_div, rpow_self_rpow_inv hz]
  /-
    🎉 no goals
  -/


@[deprecated eq_rpow_inv_iff (since := "2024-07-10")]
theorem eq_rpow_one_div_iff {x y : ℝ≥0} {z : ℝ} (hz : z ≠ 0) : x = y ^ (1 / z) ↔ x ^ z = y := by
  /-
    x y : NNReal
    z : Real
    hz : Ne z 0
    ⊢ Iff (Eq x (HPow.hPow y (HDiv.hDiv 1 z))) (Eq (HPow.hPow x z) y)
  -/
  rw [← rpow_eq_rpow_iff hz, rpow_self_rpow_inv hz]
  /-
    🎉 no goals
  -/


theorem rpow_inv_eq_iff {x y : ℝ≥0} {z : ℝ} (hz : z ≠ 0) : x ^ z⁻¹ = y ↔ x = y ^ z := by
  /-
    x y : NNReal
    z : Real
    hz : Ne z 0
    ⊢ Iff (Eq (HPow.hPow x (Inv.inv z)) y) (Eq x (HPow.hPow y z))
  -/
  rw [← rpow_eq_rpow_iff hz, ← one_div, rpow_self_rpow_inv hz]
  /-
    🎉 no goals
  -/


@[deprecated rpow_inv_eq_iff (since := "2024-07-10")]
theorem rpow_one_div_eq_iff {x y : ℝ≥0} {z : ℝ} (hz : z ≠ 0) : x ^ (1 / z) = y ↔ x = y ^ z := by
  /-
    x y : NNReal
    z : Real
    hz : Ne z 0
    ⊢ Iff (Eq (HPow.hPow x (HDiv.hDiv 1 z)) y) (Eq x (HPow.hPow y z))
  -/
  rw [← rpow_eq_rpow_iff hz, rpow_self_rpow_inv hz]
  /-
    🎉 no goals
  -/


@[simp] lemma rpow_rpow_inv {y : ℝ} (hy : y ≠ 0) (x : ℝ≥0) : (x ^ y) ^ y⁻¹ = x := by
  /-
    y : Real
    hy : Ne y 0
    x : NNReal
    ⊢ Eq (HPow.hPow (HPow.hPow x y) (Inv.inv y)) x
  -/
  rw [← rpow_mul, mul_inv_cancel₀ hy, rpow_one]
  /-
    🎉 no goals
  -/


@[simp] lemma rpow_inv_rpow {y : ℝ} (hy : y ≠ 0) (x : ℝ≥0) : (x ^ y⁻¹) ^ y = x := by
  /-
    y : Real
    hy : Ne y 0
    x : NNReal
    ⊢ Eq (HPow.hPow (HPow.hPow x (Inv.inv y)) y) x
  -/
  rw [← rpow_mul, inv_mul_cancel₀ hy, rpow_one]
  /-
    🎉 no goals
  -/


theorem pow_rpow_inv_natCast (x : ℝ≥0) {n : ℕ} (hn : n ≠ 0) : (x ^ n) ^ (n⁻¹ : ℝ) = x := by
  /-
    x : NNReal
    n : Nat
    hn : Ne n 0
    ⊢ Eq (HPow.hPow (HPow.hPow x n) (Inv.inv ↑n)) x
  -/
  rw [← NNReal.coe_inj, coe_rpow, NNReal.coe_pow]
  /-
    x : NNReal
    n : Nat
    hn : Ne n 0
    ⊢ Eq (HPow.hPow (HPow.hPow (↑x) n) (Inv.inv ↑n)) ↑x
  -/
  exact Real.pow_rpow_inv_natCast x.2 hn
  /-
    🎉 no goals
  -/


theorem rpow_inv_natCast_pow (x : ℝ≥0) {n : ℕ} (hn : n ≠ 0) : (x ^ (n⁻¹ : ℝ)) ^ n = x := by
  /-
    x : NNReal
    n : Nat
    hn : Ne n 0
    ⊢ Eq (HPow.hPow (HPow.hPow x (Inv.inv ↑n)) n) x
  -/
  rw [← NNReal.coe_inj, NNReal.coe_pow, coe_rpow]
  /-
    x : NNReal
    n : Nat
    hn : Ne n 0
    ⊢ Eq (HPow.hPow (HPow.hPow (↑x) (Inv.inv ↑n)) n) ↑x
  -/
  exact Real.rpow_inv_natCast_pow x.2 hn
  /-
    🎉 no goals
  -/


theorem _root_.Real.toNNReal_rpow_of_nonneg {x y : ℝ} (hx : 0 ≤ x) :
    Real.toNNReal (x ^ y) = Real.toNNReal x ^ y := by
  /-
    x y : Real
    hx : LE.le 0 x
    ⊢ Eq (HPow.hPow x y).toNNReal (HPow.hPow x.toNNReal y)
  -/
  nth_rw 1 [← Real.coe_toNNReal x hx]
  /-
    x y : Real
    hx : LE.le 0 x
    ⊢ Eq (HPow.hPow (↑x.toNNReal) y).toNNReal (HPow.hPow x.toNNReal y)
  -/
  rw [← NNReal.coe_rpow, Real.toNNReal_coe]
  /-
    🎉 no goals
  -/


theorem strictMono_rpow_of_pos {z : ℝ} (h : 0 < z) : StrictMono fun x : ℝ≥0 => x ^ z :=
                    /-
                      z : Real
                      h : LT.lt 0 z
                      x y : NNReal
                      hxy : LT.lt x y
                      ⊢ LT.lt ((fun x => HPow.hPow x z) x) ((fun x => HPow.hPow x z) y)
                    -/
  fun x y hxy => by simp only [NNReal.rpow_lt_rpow hxy h, coe_lt_coe]
                    /-
                      🎉 no goals
                    -/


theorem monotone_rpow_of_nonneg {z : ℝ} (h : 0 ≤ z) : Monotone fun x : ℝ≥0 => x ^ z :=
                                     /-
                                       z : Real
                                       h : LE.le 0 z
                                       h0 : Eq 0 z
                                       ⊢ Monotone fun x => HPow.hPow x 0
                                     -/
  h.eq_or_lt.elim (fun h0 => h0 ▸ by simp only [rpow_zero, monotone_const]) fun h0 =>
                                     /-
                                       🎉 no goals
                                     -/
    (strictMono_rpow_of_pos h0).monotone


/-- Bundles `fun x : ℝ≥0 => x ^ y` into an order isomorphism when `y : ℝ` is positive,
where the inverse is `fun x : ℝ≥0 => x ^ (1 / y)`. -/
@[simps! apply]
def orderIsoRpow (y : ℝ) (hy : 0 < y) : ℝ≥0 ≃o ℝ≥0 :=
  (strictMono_rpow_of_pos hy).orderIsoOfRightInverse (fun x => x ^ y) (fun x => x ^ (1 / y))
    fun x => by
      /-
        x✝ : NNReal
        w y✝ z y : Real
        hy : LT.lt 0 y
        x : NNReal
        ⊢ Eq ((fun x => HPow.hPow x y) ((fun x => HPow.hPow x (HDiv.hDiv 1 y)) x)) x
      -/
      dsimp
      /-
        x✝ : NNReal
        w y✝ z y : Real
        hy : LT.lt 0 y
        x : NNReal
        ⊢ Eq (HPow.hPow (HPow.hPow x (HDiv.hDiv 1 y)) y) x
      -/
      rw [← rpow_mul, one_div_mul_cancel hy.ne.symm, rpow_one]
      /-
        🎉 no goals
      -/


theorem orderIsoRpow_symm_eq (y : ℝ) (hy : 0 < y) :
    (orderIsoRpow y hy).symm = orderIsoRpow (1 / y) (one_div_pos.2 hy) := by
  /-
    y : Real
    hy : LT.lt 0 y
    ⊢ Eq (NNReal.orderIsoRpow y hy).symm (NNReal.orderIsoRpow (HDiv.hDiv 1 y) ⋯)
  -/
  simp only [orderIsoRpow, one_div_one_div]; rfl
                                             /-
                                               🎉 no goals
                                             -/


theorem _root_.Real.nnnorm_rpow_of_nonneg {x y : ℝ} (hx : 0 ≤ x) : ‖x ^ y‖₊ = ‖x‖₊ ^ y := by
  /-
    x y : Real
    hx : LE.le 0 x
    ⊢ Eq (NNNorm.nnnorm (HPow.hPow x y)) (HPow.hPow (NNNorm.nnnorm x) y)
  -/
  ext; exact Real.norm_rpow_of_nonneg hx
       /-
         🎉 no goals
       -/


/-- The real power function `x^y` on extended nonnegative reals, defined for `x : ℝ≥0∞` and
`y : ℝ` as the restriction of the real power function if `0 < x < ⊤`, and with the natural values
for `0` and `⊤` (i.e., `0 ^ x = 0` for `x > 0`, `1` for `x = 0` and `⊤` for `x < 0`, and
`⊤ ^ x = 1 / 0 ^ x`). -/
noncomputable def rpow : ℝ≥0∞ → ℝ → ℝ≥0∞
  | some x, y => if x = 0 ∧ y < 0 then ⊤ else (x ^ y : ℝ≥0)
  | none, y => if 0 < y then ⊤ else if y = 0 then 1 else 0


noncomputable instance : Pow ℝ≥0∞ ℝ :=
  ⟨rpow⟩


@[simp]
theorem rpow_eq_pow (x : ℝ≥0∞) (y : ℝ) : rpow x y = x ^ y :=
  rfl


@[simp]
theorem rpow_zero {x : ℝ≥0∞} : x ^ (0 : ℝ) = 1 := by
  /-
    x : ENNReal
    ⊢ Eq (HPow.hPow x 0) 1
  -/
  cases x <;>
      /-
        case top
        ⊢ Eq (HPow.hPow Top.top 0) 1
      -/
      /-
        case top
        ⊢ Eq (ite (LT.lt 0 0) Top.top (ite (Eq 0 0) 1 0)) 1
      -/
      /-
        🎉 no goals
      -/
      /-
        case coe
        x✝ : NNReal
        ⊢ Eq (ite (And (Eq x✝ 0) (LT.lt 0 0)) Top.top ↑(x✝.rpow 0)) 1
      -/
      simp [lt_irrefl]
      /-
        🎉 no goals
      -/


theorem top_rpow_def (y : ℝ) : (⊤ : ℝ≥0∞) ^ y = if 0 < y then ⊤ else if y = 0 then 1 else 0 :=
  rfl


@[simp]
                                                                       /-
                                                                         y : Real
                                                                         h : LT.lt 0 y
                                                                         ⊢ Eq (HPow.hPow Top.top y) Top.top
                                                                       -/
theorem top_rpow_of_pos {y : ℝ} (h : 0 < y) : (⊤ : ℝ≥0∞) ^ y = ⊤ := by simp [top_rpow_def, h]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
theorem top_rpow_of_neg {y : ℝ} (h : y < 0) : (⊤ : ℝ≥0∞) ^ y = 0 := by
  /-
    y : Real
    h : LT.lt y 0
    ⊢ Eq (HPow.hPow Top.top y) 0
  -/
  simp [top_rpow_def, asymm h, ne_of_lt h]
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_rpow_of_pos {y : ℝ} (h : 0 < y) : (0 : ℝ≥0∞) ^ y = 0 := by
  /-
    y : Real
    h : LT.lt 0 y
    ⊢ Eq (HPow.hPow 0 y) 0
  -/
  rw [← ENNReal.coe_zero, ← ENNReal.some_eq_coe]
  /-
    y : Real
    h : LT.lt 0 y
    ⊢ Eq (HPow.hPow (Option.some 0) y) (Option.some 0)
  -/
  dsimp only [(· ^ ·), rpow, Pow.pow]
  /-
    y : Real
    h : LT.lt 0 y
    ⊢ Eq (ite (And (Eq 0 0) (LT.lt y 0)) Top.top ↑(NNReal.rpow 0 y)) (Option.some 0)
  -/
  simp [h, asymm h, ne_of_gt h]
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_rpow_of_neg {y : ℝ} (h : y < 0) : (0 : ℝ≥0∞) ^ y = ⊤ := by
  /-
    y : Real
    h : LT.lt y 0
    ⊢ Eq (HPow.hPow 0 y) Top.top
  -/
  rw [← ENNReal.coe_zero, ← ENNReal.some_eq_coe]
  /-
    y : Real
    h : LT.lt y 0
    ⊢ Eq (HPow.hPow (Option.some 0) y) Top.top
  -/
  dsimp only [(· ^ ·), rpow, Pow.pow]
  /-
    y : Real
    h : LT.lt y 0
    ⊢ Eq (ite (And (Eq 0 0) (LT.lt y 0)) Top.top ↑(NNReal.rpow 0 y)) Top.top
  -/
  simp [h, ne_of_gt h]
  /-
    🎉 no goals
  -/


theorem zero_rpow_def (y : ℝ) : (0 : ℝ≥0∞) ^ y = if 0 < y then 0 else if y = 0 then 1 else ⊤ := by
  /-
    y : Real
    ⊢ Eq (HPow.hPow 0 y) (ite (LT.lt 0 y) 0 (ite (Eq y 0) 1 Top.top))
  -/
  rcases lt_trichotomy (0 : ℝ) y with (H | rfl | H)
    /-
      case inl
      y : Real
      H : LT.lt 0 y
      ⊢ Eq (HPow.hPow 0 y) (ite (LT.lt 0 y) 0 (ite (Eq y 0) 1 Top.top))
    -/
  · simp [H, ne_of_gt, zero_rpow_of_pos, lt_irrefl]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      ⊢ Eq (HPow.hPow 0 0) (ite (LT.lt 0 0) 0 (ite (Eq 0 0) 1 Top.top))
    -/
  · simp [lt_irrefl]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      y : Real
      H : LT.lt y 0
      ⊢ Eq (HPow.hPow 0 y) (ite (LT.lt 0 y) 0 (ite (Eq y 0) 1 Top.top))
    -/
  · simp [H, asymm H, ne_of_lt, zero_rpow_of_neg]
    /-
      🎉 no goals
    -/


@[simp]
theorem zero_rpow_mul_self (y : ℝ) : (0 : ℝ≥0∞) ^ y * (0 : ℝ≥0∞) ^ y = (0 : ℝ≥0∞) ^ y := by
  /-
    y : Real
    ⊢ Eq (HMul.hMul (HPow.hPow 0 y) (HPow.hPow 0 y)) (HPow.hPow 0 y)
  -/
  rw [zero_rpow_def]
  /-
    y : Real
    ⊢ Eq (HMul.hMul (ite (LT.lt 0 y) 0 (ite (Eq y 0) 1 Top.top)) (ite (LT.lt 0 y)  …
  -/
  split_ifs
  /-
    case pos
    y : Real
    h✝ : LT.lt 0 y
    ⊢ Eq (HMul.hMul 0 0) 0
  -/
  exacts [zero_mul _, one_mul _, top_mul_top]
  /-
    🎉 no goals
  -/


@[norm_cast]
theorem coe_rpow_of_ne_zero {x : ℝ≥0} (h : x ≠ 0) (y : ℝ) : (↑(x ^ y) : ℝ≥0∞) = x ^ y := by
  /-
    x : NNReal
    h : Ne x 0
    y : Real
    ⊢ Eq (↑(HPow.hPow x y)) (HPow.hPow (↑x) y)
  -/
  rw [← ENNReal.some_eq_coe]
  /-
    x : NNReal
    h : Ne x 0
    y : Real
    ⊢ Eq (Option.some (HPow.hPow x y)) (HPow.hPow (↑x) y)
  -/
  dsimp only [(· ^ ·), Pow.pow, rpow]
  /-
    x : NNReal
    h : Ne x 0
    y : Real
    ⊢ Eq (Option.some (x.rpow y)) (ite (And (Eq x 0) (LT.lt y 0)) Top.top ↑(x.rpow …
  -/
  simp [h]
  /-
    🎉 no goals
  -/


@[norm_cast]
theorem coe_rpow_of_nonneg (x : ℝ≥0) {y : ℝ} (h : 0 ≤ y) : ↑(x ^ y) = (x : ℝ≥0∞) ^ y := by
  /-
    x : NNReal
    y : Real
    h : LE.le 0 y
    ⊢ Eq (↑(HPow.hPow x y)) (HPow.hPow (↑x) y)
  -/
  by_cases hx : x = 0
    /-
      case pos
      x : NNReal
      y : Real
      h : LE.le 0 y
      hx : Eq x 0
      ⊢ Eq (↑(HPow.hPow x y)) (HPow.hPow (↑x) y)
    -/
  · rcases le_iff_eq_or_lt.1 h with (H | H)
      /-
        case pos.inl
        x : NNReal
        y : Real
        h : LE.le 0 y
        hx : Eq x 0
        H : Eq 0 y
        ⊢ Eq (↑(HPow.hPow x y)) (HPow.hPow (↑x) y)
      -/
    · simp [hx, H.symm]
      /-
        🎉 no goals
      -/
      /-
        case pos.inr
        x : NNReal
        y : Real
        h : LE.le 0 y
        hx : Eq x 0
        H : LT.lt 0 y
        ⊢ Eq (↑(HPow.hPow x y)) (HPow.hPow (↑x) y)
      -/
    · simp [hx, zero_rpow_of_pos H, NNReal.zero_rpow (ne_of_gt H)]
      /-
        🎉 no goals
      -/
    /-
      case neg
      x : NNReal
      y : Real
      h : LE.le 0 y
      hx : Not (Eq x 0)
      ⊢ Eq (↑(HPow.hPow x y)) (HPow.hPow (↑x) y)
    -/
  · exact coe_rpow_of_ne_zero hx _
    /-
      🎉 no goals
    -/


theorem coe_rpow_def (x : ℝ≥0) (y : ℝ) :
    (x : ℝ≥0∞) ^ y = if x = 0 ∧ y < 0 then ⊤ else ↑(x ^ y) :=
  rfl


@[simp]
theorem rpow_one (x : ℝ≥0∞) : x ^ (1 : ℝ) = x := by
  /-
    x : ENNReal
    ⊢ Eq (HPow.hPow x 1) x
  -/
  cases x
    /-
      case top
      ⊢ Eq (HPow.hPow Top.top 1) Top.top
    -/
  · exact dif_pos zero_lt_one
    /-
      🎉 no goals
    -/
    /-
      case coe
      x✝ : NNReal
      ⊢ Eq (HPow.hPow (↑x✝) 1) ↑x✝
    -/
  · change ite _ _ _ = _
    /-
      case coe
      x✝ : NNReal
      ⊢ Eq (ite (And (Eq x✝ 0) (LT.lt 1 0)) Top.top ↑(HPow.hPow x✝ 1)) ↑x✝
    -/
    simp only [NNReal.rpow_one, some_eq_coe, ite_eq_right_iff, top_ne_coe, and_imp]
    /-
      case coe
      x✝ : NNReal
      ⊢ Eq x✝ 0 → LT.lt 1 0 → False
    -/
    exact fun _ => zero_le_one.not_lt
    /-
      🎉 no goals
    -/


@[simp]
theorem one_rpow (x : ℝ) : (1 : ℝ≥0∞) ^ x = 1 := by
  /-
    x : Real
    ⊢ Eq (HPow.hPow 1 x) 1
  -/
  rw [← coe_one, ← coe_rpow_of_ne_zero one_ne_zero]
  /-
    x : Real
    ⊢ Eq ↑(HPow.hPow 1 x) ↑1
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem rpow_eq_zero_iff {x : ℝ≥0∞} {y : ℝ} : x ^ y = 0 ↔ x = 0 ∧ 0 < y ∨ x = ⊤ ∧ y < 0 := by
  /-
    x : ENNReal
    y : Real
    ⊢ Iff (Eq (HPow.hPow x y) 0) (Or (And (Eq x 0) (LT.lt 0 y)) (And (Eq x Top.top …
  -/
  cases' x with x
    /-
      case top
      y : Real
      ⊢ Iff (Eq (HPow.hPow Top.top y) 0) (Or (And (Eq Top.top 0) (LT.lt 0 y)) (And ( …
    -/
  · rcases lt_trichotomy y 0 with (H | H | H) <;>
      /-
        case top.inl
        y : Real
        H : LT.lt y 0
        ⊢ Iff (Eq (HPow.hPow Top.top y) 0) (Or (And (Eq Top.top 0) (LT.lt 0 y)) (And ( …
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      simp [H, top_rpow_of_neg, top_rpow_of_pos, le_of_lt]
      /-
        🎉 no goals
      -/
    /-
      case coe
      y : Real
      x : NNReal
      ⊢ Iff (Eq (HPow.hPow (↑x) y) 0) (Or (And (Eq (↑x) 0) (LT.lt 0 y)) (And (Eq (↑x …
    -/
  · by_cases h : x = 0
      /-
        case pos
        y : Real
        x : NNReal
        h : Eq x 0
        ⊢ Iff (Eq (HPow.hPow (↑x) y) 0) (Or (And (Eq (↑x) 0) (LT.lt 0 y)) (And (Eq (↑x …
      -/
    · rcases lt_trichotomy y 0 with (H | H | H) <;>
        /-
          case pos.inl
          y : Real
          x : NNReal
          h : Eq x 0
          H : LT.lt y 0
          ⊢ Iff (Eq (HPow.hPow (↑x) y) 0) (Or (And (Eq (↑x) 0) (LT.lt 0 y)) (And (Eq (↑x …
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
        simp [h, H, zero_rpow_of_neg, zero_rpow_of_pos, le_of_lt]
        /-
          🎉 no goals
        -/
      /-
        case neg
        y : Real
        x : NNReal
        h : Not (Eq x 0)
        ⊢ Iff (Eq (HPow.hPow (↑x) y) 0) (Or (And (Eq (↑x) 0) (LT.lt 0 y)) (And (Eq (↑x …
      -/
    · simp [← coe_rpow_of_ne_zero h, h]
      /-
        🎉 no goals
      -/


lemma rpow_eq_zero_iff_of_pos {x : ℝ≥0∞} {y : ℝ} (hy : 0 < y) : x ^ y = 0 ↔ x = 0 := by
  /-
    x : ENNReal
    y : Real
    hy : LT.lt 0 y
    ⊢ Iff (Eq (HPow.hPow x y) 0) (Eq x 0)
  -/
  simp [hy, hy.not_lt]
  /-
    🎉 no goals
  -/


@[simp]
theorem rpow_eq_top_iff {x : ℝ≥0∞} {y : ℝ} : x ^ y = ⊤ ↔ x = 0 ∧ y < 0 ∨ x = ⊤ ∧ 0 < y := by
  /-
    x : ENNReal
    y : Real
    ⊢ Iff (Eq (HPow.hPow x y) Top.top) (Or (And (Eq x 0) (LT.lt y 0)) (And (Eq x T …
  -/
  cases' x with x
    /-
      case top
      y : Real
      ⊢ Iff (Eq (HPow.hPow Top.top y) Top.top) (Or (And (Eq Top.top 0) (LT.lt y 0))  …
    -/
  · rcases lt_trichotomy y 0 with (H | H | H) <;>
      /-
        case top.inl
        y : Real
        H : LT.lt y 0
        ⊢ Iff (Eq (HPow.hPow Top.top y) Top.top) (Or (And (Eq Top.top 0) (LT.lt y 0))  …
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      simp [H, top_rpow_of_neg, top_rpow_of_pos, le_of_lt]
      /-
        🎉 no goals
      -/
    /-
      case coe
      y : Real
      x : NNReal
      ⊢ Iff (Eq (HPow.hPow (↑x) y) Top.top) (Or (And (Eq (↑x) 0) (LT.lt y 0)) (And ( …
    -/
  · by_cases h : x = 0
      /-
        case pos
        y : Real
        x : NNReal
        h : Eq x 0
        ⊢ Iff (Eq (HPow.hPow (↑x) y) Top.top) (Or (And (Eq (↑x) 0) (LT.lt y 0)) (And ( …
      -/
    · rcases lt_trichotomy y 0 with (H | H | H) <;>
        /-
          case pos.inl
          y : Real
          x : NNReal
          h : Eq x 0
          H : LT.lt y 0
          ⊢ Iff (Eq (HPow.hPow (↑x) y) Top.top) (Or (And (Eq (↑x) 0) (LT.lt y 0)) (And ( …
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
        simp [h, H, zero_rpow_of_neg, zero_rpow_of_pos, le_of_lt]
        /-
          🎉 no goals
        -/
      /-
        case neg
        y : Real
        x : NNReal
        h : Not (Eq x 0)
        ⊢ Iff (Eq (HPow.hPow (↑x) y) Top.top) (Or (And (Eq (↑x) 0) (LT.lt y 0)) (And ( …
      -/
    · simp [← coe_rpow_of_ne_zero h, h]
      /-
        🎉 no goals
      -/


theorem rpow_eq_top_iff_of_pos {x : ℝ≥0∞} {y : ℝ} (hy : 0 < y) : x ^ y = ⊤ ↔ x = ⊤ := by
  /-
    x : ENNReal
    y : Real
    hy : LT.lt 0 y
    ⊢ Iff (Eq (HPow.hPow x y) Top.top) (Eq x Top.top)
  -/
  simp [rpow_eq_top_iff, hy, asymm hy]
  /-
    🎉 no goals
  -/


lemma rpow_lt_top_iff_of_pos {x : ℝ≥0∞} {y : ℝ} (hy : 0 < y) : x ^ y < ∞ ↔ x < ∞ := by
  /-
    x : ENNReal
    y : Real
    hy : LT.lt 0 y
    ⊢ Iff (LT.lt (HPow.hPow x y) Top.top) (LT.lt x Top.top)
  -/
  simp only [lt_top_iff_ne_top, Ne, rpow_eq_top_iff_of_pos hy]
  /-
    🎉 no goals
  -/


theorem rpow_eq_top_of_nonneg (x : ℝ≥0∞) {y : ℝ} (hy0 : 0 ≤ y) : x ^ y = ⊤ → x = ⊤ := by
  /-
    x : ENNReal
    y : Real
    hy0 : LE.le 0 y
    ⊢ Eq (HPow.hPow x y) Top.top → Eq x Top.top
  -/
  rw [ENNReal.rpow_eq_top_iff]
  /-
    x : ENNReal
    y : Real
    hy0 : LE.le 0 y
    ⊢ Or (And (Eq x 0) (LT.lt y 0)) (And (Eq x Top.top) (LT.lt 0 y)) → Eq x Top.top
  -/
  rintro (h|h)
    /-
      case inl
      x : ENNReal
      y : Real
      hy0 : LE.le 0 y
      h : And (Eq x 0) (LT.lt y 0)
      ⊢ Eq x Top.top
    -/
  · exfalso
    /-
      case inl
      x : ENNReal
      y : Real
      hy0 : LE.le 0 y
      h : And (Eq x 0) (LT.lt y 0)
      ⊢ False
    -/
    rw [lt_iff_not_ge] at h
    /-
      case inl
      x : ENNReal
      y : Real
      hy0 : LE.le 0 y
      h : And (Eq x 0) (Not (GE.ge y 0))
      ⊢ False
    -/
    exact h.right hy0
    /-
      🎉 no goals
    -/
    /-
      case inr
      x : ENNReal
      y : Real
      hy0 : LE.le 0 y
      h : And (Eq x Top.top) (LT.lt 0 y)
      ⊢ Eq x Top.top
    -/
  · exact h.left
    /-
      🎉 no goals
    -/


theorem rpow_ne_top_of_nonneg {x : ℝ≥0∞} {y : ℝ} (hy0 : 0 ≤ y) (h : x ≠ ⊤) : x ^ y ≠ ⊤ :=
  mt (ENNReal.rpow_eq_top_of_nonneg x hy0) h


theorem rpow_lt_top_of_nonneg {x : ℝ≥0∞} {y : ℝ} (hy0 : 0 ≤ y) (h : x ≠ ⊤) : x ^ y < ⊤ :=
  lt_top_iff_ne_top.mpr (ENNReal.rpow_ne_top_of_nonneg hy0 h)


theorem rpow_add {x : ℝ≥0∞} (y z : ℝ) (hx : x ≠ 0) (h'x : x ≠ ⊤) : x ^ (y + z) = x ^ y * x ^ z := by
  /-
    x : ENNReal
    y z : Real
    hx : Ne x 0
    h'x : Ne x Top.top
    ⊢ Eq (HPow.hPow x (HAdd.hAdd y z)) (HMul.hMul (HPow.hPow x y) (HPow.hPow x z))
  -/
  cases' x with x
    /-
      case top
      y z : Real
      hx : Ne Top.top 0
      h'x : Ne Top.top Top.top
      ⊢ Eq (HPow.hPow Top.top (HAdd.hAdd y z)) (HMul.hMul (HPow.hPow Top.top y) (HPo …
    -/
  · exact (h'x rfl).elim
    /-
      🎉 no goals
    -/
  /-
    case coe
    y z : Real
    x : NNReal
    hx : Ne (↑x) 0
    h'x : Ne (↑x) Top.top
    ⊢ Eq (HPow.hPow (↑x) (HAdd.hAdd y z)) (HMul.hMul (HPow.hPow (↑x) y) (HPow.hPow …
  -/
  have : x ≠ 0 := fun h => by simp [h] at hx
  /-
    case coe
    y z : Real
    x : NNReal
    hx : Ne (↑x) 0
    h'x : Ne (↑x) Top.top
    this : Ne x 0
    ⊢ Eq (HPow.hPow (↑x) (HAdd.hAdd y z)) (HMul.hMul (HPow.hPow (↑x) y) (HPow.hPow …
  -/
  simp [← coe_rpow_of_ne_zero this, NNReal.rpow_add this]
  /-
    🎉 no goals
  -/


theorem rpow_add_of_nonneg {x : ℝ≥0∞} (y z : ℝ) (hy : 0 ≤ y) (hz : 0 ≤ z) :
    x ^ (y + z) = x ^ y * x ^ z := by
  /-
    x : ENNReal
    y z : Real
    hy : LE.le 0 y
    hz : LE.le 0 z
    ⊢ Eq (HPow.hPow x (HAdd.hAdd y z)) (HMul.hMul (HPow.hPow x y) (HPow.hPow x z))
  -/
  induction x using recTopCoe
    /-
      case top
      y z : Real
      hy : LE.le 0 y
      hz : LE.le 0 z
      ⊢ Eq (HPow.hPow Top.top (HAdd.hAdd y z)) (HMul.hMul (HPow.hPow Top.top y) (HPo …
    -/
  · rcases hy.eq_or_lt with rfl|hy
      /-
        case top.inl
        z : Real
        hz : LE.le 0 z
        hy : LE.le 0 0
        ⊢ Eq (HPow.hPow Top.top (HAdd.hAdd 0 z)) (HMul.hMul (HPow.hPow Top.top 0) (HPo …
      -/
    · rw [rpow_zero, one_mul, zero_add]
      /-
        🎉 no goals
      -/
    /-
      case top.inr
      y z : Real
      hy✝ : LE.le 0 y
      hz : LE.le 0 z
      hy : LT.lt 0 y
      ⊢ Eq (HPow.hPow Top.top (HAdd.hAdd y z)) (HMul.hMul (HPow.hPow Top.top y) (HPo …
    -/
    rcases hz.eq_or_lt with rfl|hz
      /-
        case top.inr.inl
        y : Real
        hy✝ : LE.le 0 y
        hy : LT.lt 0 y
        hz : LE.le 0 0
        ⊢ Eq (HPow.hPow Top.top (HAdd.hAdd y 0)) (HMul.hMul (HPow.hPow Top.top y) (HPo …
      -/
    · rw [rpow_zero, mul_one, add_zero]
      /-
        🎉 no goals
      -/
    /-
      case top.inr.inr
      y z : Real
      hy✝ : LE.le 0 y
      hz✝ : LE.le 0 z
      hy : LT.lt 0 y
      hz : LT.lt 0 z
      ⊢ Eq (HPow.hPow Top.top (HAdd.hAdd y z)) (HMul.hMul (HPow.hPow Top.top y) (HPo …
    -/
    simp [top_rpow_of_pos, hy, hz, add_pos hy hz]
    /-
      🎉 no goals
    -/
  /-
    case coe
    y z : Real
    hy : LE.le 0 y
    hz : LE.le 0 z
    x✝ : NNReal
    ⊢ Eq (HPow.hPow (↑x✝) (HAdd.hAdd y z)) (HMul.hMul (HPow.hPow (↑x✝) y) (HPow.hP …
  -/
  simp [← coe_rpow_of_nonneg, hy, hz, add_nonneg hy hz, NNReal.rpow_add_of_nonneg _ hy hz]
  /-
    🎉 no goals
  -/


theorem rpow_neg (x : ℝ≥0∞) (y : ℝ) : x ^ (-y) = (x ^ y)⁻¹ := by
  /-
    x : ENNReal
    y : Real
    ⊢ Eq (HPow.hPow x (Neg.neg y)) (Inv.inv (HPow.hPow x y))
  -/
  cases' x with x
    /-
      case top
      y : Real
      ⊢ Eq (HPow.hPow Top.top (Neg.neg y)) (Inv.inv (HPow.hPow Top.top y))
    -/
  · rcases lt_trichotomy y 0 with (H | H | H) <;>
      /-
        case top.inl
        y : Real
        H : LT.lt y 0
        ⊢ Eq (HPow.hPow Top.top (Neg.neg y)) (Inv.inv (HPow.hPow Top.top y))
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      simp [top_rpow_of_pos, top_rpow_of_neg, H, neg_pos.mpr]
      /-
        🎉 no goals
      -/
    /-
      case coe
      y : Real
      x : NNReal
      ⊢ Eq (HPow.hPow (↑x) (Neg.neg y)) (Inv.inv (HPow.hPow (↑x) y))
    -/
  · by_cases h : x = 0
      /-
        case pos
        y : Real
        x : NNReal
        h : Eq x 0
        ⊢ Eq (HPow.hPow (↑x) (Neg.neg y)) (Inv.inv (HPow.hPow (↑x) y))
      -/
    · rcases lt_trichotomy y 0 with (H | H | H) <;>
        /-
          case pos.inl
          y : Real
          x : NNReal
          h : Eq x 0
          H : LT.lt y 0
          ⊢ Eq (HPow.hPow (↑x) (Neg.neg y)) (Inv.inv (HPow.hPow (↑x) y))
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
        simp [h, zero_rpow_of_pos, zero_rpow_of_neg, H, neg_pos.mpr]
        /-
          🎉 no goals
        -/
      /-
        case neg
        y : Real
        x : NNReal
        h : Not (Eq x 0)
        ⊢ Eq (HPow.hPow (↑x) (Neg.neg y)) (Inv.inv (HPow.hPow (↑x) y))
      -/
    · have A : x ^ y ≠ 0 := by simp [h]
      /-
        case neg
        y : Real
        x : NNReal
        h : Not (Eq x 0)
        A : Ne (HPow.hPow x y) 0
        ⊢ Eq (HPow.hPow (↑x) (Neg.neg y)) (Inv.inv (HPow.hPow (↑x) y))
      -/
      simp [← coe_rpow_of_ne_zero h, ← coe_inv A, NNReal.rpow_neg]
      /-
        🎉 no goals
      -/


theorem rpow_sub {x : ℝ≥0∞} (y z : ℝ) (hx : x ≠ 0) (h'x : x ≠ ⊤) : x ^ (y - z) = x ^ y / x ^ z := by
  /-
    x : ENNReal
    y z : Real
    hx : Ne x 0
    h'x : Ne x Top.top
    ⊢ Eq (HPow.hPow x (HSub.hSub y z)) (HDiv.hDiv (HPow.hPow x y) (HPow.hPow x z))
  -/
  rw [sub_eq_add_neg, rpow_add _ _ hx h'x, rpow_neg, div_eq_mul_inv]
  /-
    🎉 no goals
  -/


                                                           /-
                                                             x : ENNReal
                                                             ⊢ Eq (HPow.hPow x (-1)) (Inv.inv x)
                                                           -/
theorem rpow_neg_one (x : ℝ≥0∞) : x ^ (-1 : ℝ) = x⁻¹ := by simp [rpow_neg]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem rpow_mul (x : ℝ≥0∞) (y z : ℝ) : x ^ (y * z) = (x ^ y) ^ z := by
  /-
    x : ENNReal
    y z : Real
    ⊢ Eq (HPow.hPow x (HMul.hMul y z)) (HPow.hPow (HPow.hPow x y) z)
  -/
  cases' x with x
    /-
      case top
      y z : Real
      ⊢ Eq (HPow.hPow Top.top (HMul.hMul y z)) (HPow.hPow (HPow.hPow Top.top y) z)
    -/
  · rcases lt_trichotomy y 0 with (Hy | Hy | Hy) <;>
        /-
          case top.inl
          y z : Real
          Hy : LT.lt y 0
          ⊢ Eq (HPow.hPow Top.top (HMul.hMul y z)) (HPow.hPow (HPow.hPow Top.top y) z)
        -/
        rcases lt_trichotomy z 0 with (Hz | Hz | Hz) <;>
      simp [Hy, Hz, zero_rpow_of_neg, zero_rpow_of_pos, top_rpow_of_neg, top_rpow_of_pos,
        mul_pos_of_neg_of_neg, mul_neg_of_neg_of_pos, mul_neg_of_pos_of_neg]
    /-
      case coe
      y z : Real
      x : NNReal
      ⊢ Eq (HPow.hPow (↑x) (HMul.hMul y z)) (HPow.hPow (HPow.hPow (↑x) y) z)
    -/
  · by_cases h : x = 0
      /-
        case pos
        y z : Real
        x : NNReal
        h : Eq x 0
        ⊢ Eq (HPow.hPow (↑x) (HMul.hMul y z)) (HPow.hPow (HPow.hPow (↑x) y) z)
      -/
    · rcases lt_trichotomy y 0 with (Hy | Hy | Hy) <;>
          /-
            case pos.inl
            y z : Real
            x : NNReal
            h : Eq x 0
            Hy : LT.lt y 0
            ⊢ Eq (HPow.hPow (↑x) (HMul.hMul y z)) (HPow.hPow (HPow.hPow (↑x) y) z)
          -/
          rcases lt_trichotomy z 0 with (Hz | Hz | Hz) <;>
        simp [h, Hy, Hz, zero_rpow_of_neg, zero_rpow_of_pos, top_rpow_of_neg, top_rpow_of_pos,
          mul_pos_of_neg_of_neg, mul_neg_of_neg_of_pos, mul_neg_of_pos_of_neg]
      /-
        case neg
        y z : Real
        x : NNReal
        h : Not (Eq x 0)
        ⊢ Eq (HPow.hPow (↑x) (HMul.hMul y z)) (HPow.hPow (HPow.hPow (↑x) y) z)
      -/
    · have : x ^ y ≠ 0 := by simp [h]
      /-
        case neg
        y z : Real
        x : NNReal
        h : Not (Eq x 0)
        this : Ne (HPow.hPow x y) 0
        ⊢ Eq (HPow.hPow (↑x) (HMul.hMul y z)) (HPow.hPow (HPow.hPow (↑x) y) z)
      -/
      simp [← coe_rpow_of_ne_zero, h, this, NNReal.rpow_mul]
      /-
        🎉 no goals
      -/


@[simp, norm_cast]
theorem rpow_natCast (x : ℝ≥0∞) (n : ℕ) : x ^ (n : ℝ) = x ^ n := by
  /-
    x : ENNReal
    n : Nat
    ⊢ Eq (HPow.hPow x ↑n) (HPow.hPow x n)
  -/
  cases x
    /-
      case top
      n : Nat
      ⊢ Eq (HPow.hPow Top.top ↑n) (HPow.hPow Top.top n)
    -/
                /-
                  🎉 no goals
                -/
  · cases n <;> simp [top_rpow_of_pos (Nat.cast_add_one_pos _), top_pow (Nat.succ_pos _)]
                /-
                  🎉 no goals
                -/
    /-
      case coe
      n : Nat
      x✝ : NNReal
      ⊢ Eq (HPow.hPow ↑x✝ ↑n) (HPow.hPow (↑x✝) n)
    -/
  · simp [← coe_rpow_of_nonneg _ (Nat.cast_nonneg n)]
    /-
      🎉 no goals
    -/


@[simp]
lemma rpow_ofNat (x : ℝ≥0∞) (n : ℕ) [n.AtLeastTwo] :
    x ^ (no_index (OfNat.ofNat n) : ℝ) = x ^ (OfNat.ofNat n) :=
  rpow_natCast x n


@[simp, norm_cast]
lemma rpow_intCast (x : ℝ≥0∞) (n : ℤ) : x ^ (n : ℝ) = x ^ n := by
  /-
    x : ENNReal
    n : Int
    ⊢ Eq (HPow.hPow x ↑n) (HPow.hPow x n)
  -/
  cases n <;> simp only [Int.ofNat_eq_coe, Int.cast_natCast, rpow_natCast, zpow_natCast,
    Int.cast_negSucc, rpow_neg, zpow_negSucc]


@[deprecated (since := "2024-04-17")]
alias rpow_int_cast := rpow_intCast


theorem rpow_two (x : ℝ≥0∞) : x ^ (2 : ℝ) = x ^ 2 := rpow_ofNat x 2


theorem mul_rpow_eq_ite (x y : ℝ≥0∞) (z : ℝ) :
    (x * y) ^ z = if (x = 0 ∧ y = ⊤ ∨ x = ⊤ ∧ y = 0) ∧ z < 0 then ⊤ else x ^ z * y ^ z := by
  /-
    x y : ENNReal
    z : Real
    ⊢ Eq (HPow.hPow (HMul.hMul x y) z) (ite (And (Or (And (Eq x 0) (Eq y Top.top)) …
  -/
  rcases eq_or_ne z 0 with (rfl | hz); · simp
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case inr
    x y : ENNReal
    z : Real
    hz : Ne z 0
    ⊢ Eq (HPow.hPow (HMul.hMul x y) z) (ite (And (Or (And (Eq x 0) (Eq y Top.top)) …
  -/
  replace hz := hz.lt_or_lt
  /-
    case inr
    x y : ENNReal
    z : Real
    hz : Or (LT.lt z 0) (LT.lt 0 z)
    ⊢ Eq (HPow.hPow (HMul.hMul x y) z) (ite (And (Or (And (Eq x 0) (Eq y Top.top)) …
  -/
  wlog hxy : x ≤ y
    /-
      case inr.inr
      x y : ENNReal
      z : Real
      hz : Or (LT.lt z 0) (LT.lt 0 z)
      this : ∀ (x y : ENNReal) (z : Real), Or (LT.lt z 0) (LT.lt 0 z) → LE.le x y →  …
      hxy : Not (LE.le x y)
      ⊢ Eq (HPow.hPow (HMul.hMul x y) z) (ite (And (Or (And (Eq x 0) (Eq y Top.top)) …
    -/
                                                         /-
                                                           🎉 no goals
                                                         -/
                                                         /-
                                                           🎉 no goals
                                                         -/
  · convert this y x z hz (le_of_not_le hxy) using 2 <;> simp only [mul_comm, and_comm, or_comm]
                                                         /-
                                                           🎉 no goals
                                                         -/
  /-
    x y : ENNReal
    z : Real
    hz : Or (LT.lt z 0) (LT.lt 0 z)
    hxy : LE.le x y
    ⊢ Eq (HPow.hPow (HMul.hMul x y) z) (ite (And (Or (And (Eq x 0) (Eq y Top.top)) …
  -/
  rcases eq_or_ne x 0 with (rfl | hx0)
    /-
      case inl
      y : ENNReal
      z : Real
      hz : Or (LT.lt z 0) (LT.lt 0 z)
      hxy : LE.le 0 y
      ⊢ Eq (HPow.hPow (HMul.hMul 0 y) z) (ite (And (Or (And (Eq 0 0) (Eq y Top.top)) …
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
  · induction y <;> cases' hz with hz hz <;> simp [*, hz.not_lt]
                                             /-
                                               🎉 no goals
                                             -/
  /-
    case inr
    x y : ENNReal
    z : Real
    hz : Or (LT.lt z 0) (LT.lt 0 z)
    hxy : LE.le x y
    hx0 : Ne x 0
    ⊢ Eq (HPow.hPow (HMul.hMul x y) z) (ite (And (Or (And (Eq x 0) (Eq y Top.top)) …
  -/
  rcases eq_or_ne y 0 with (rfl | hy0)
    /-
      case inr.inl
      x : ENNReal
      z : Real
      hz : Or (LT.lt z 0) (LT.lt 0 z)
      hx0 : Ne x 0
      hxy : LE.le x 0
      ⊢ Eq (HPow.hPow (HMul.hMul x 0) z) (ite (And (Or (And (Eq x 0) (Eq 0 Top.top)) …
    -/
  · exact (hx0 (bot_unique hxy)).elim
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    x y : ENNReal
    z : Real
    hz : Or (LT.lt z 0) (LT.lt 0 z)
    hxy : LE.le x y
    hx0 : Ne x 0
    hy0 : Ne y 0
    ⊢ Eq (HPow.hPow (HMul.hMul x y) z) (ite (And (Or (And (Eq x 0) (Eq y Top.top)) …
  -/
  induction x
    /-
      case inr.inr.top
      y : ENNReal
      z : Real
      hz : Or (LT.lt z 0) (LT.lt 0 z)
      hy0 : Ne y 0
      hxy : LE.le Top.top y
      hx0 : Ne Top.top 0
      ⊢ Eq (HPow.hPow (HMul.hMul Top.top y) z) (ite (And (Or (And (Eq Top.top 0) (Eq …
    -/
                             /-
                               🎉 no goals
                             -/
  · cases' hz with hz hz <;> simp [hz, top_unique hxy]
                             /-
                               🎉 no goals
                             -/
  /-
    case inr.inr.coe
    y : ENNReal
    z : Real
    hz : Or (LT.lt z 0) (LT.lt 0 z)
    hy0 : Ne y 0
    x✝ : NNReal
    hxy : LE.le (↑x✝) y
    hx0 : Ne (↑x✝) 0
    ⊢ Eq (HPow.hPow (HMul.hMul (↑x✝) y) z) (ite (And (Or (And (Eq (↑x✝) 0) (Eq y T …
  -/
  induction y
    /-
      case inr.inr.coe.top
      z : Real
      hz : Or (LT.lt z 0) (LT.lt 0 z)
      x✝ : NNReal
      hx0 : Ne (↑x✝) 0
      hy0 : Ne Top.top 0
      hxy : LE.le (↑x✝) Top.top
      ⊢ Eq (HPow.hPow (HMul.hMul (↑x✝) Top.top) z) (ite (And (Or (And (Eq (↑x✝) 0) ( …
    -/
  · rw [ne_eq, coe_eq_zero] at hx0
    /-
      case inr.inr.coe.top
      z : Real
      hz : Or (LT.lt z 0) (LT.lt 0 z)
      x✝ : NNReal
      hx0 : Not (Eq x✝ 0)
      hy0 : Ne Top.top 0
      hxy : LE.le (↑x✝) Top.top
      ⊢ Eq (HPow.hPow (HMul.hMul (↑x✝) Top.top) z) (ite (And (Or (And (Eq (↑x✝) 0) ( …
    -/
                             /-
                               🎉 no goals
                             -/
    cases' hz with hz hz <;> simp [*]
                             /-
                               🎉 no goals
                             -/
  /-
    case inr.inr.coe.coe
    z : Real
    hz : Or (LT.lt z 0) (LT.lt 0 z)
    x✝¹ : NNReal
    hx0 : Ne (↑x✝¹) 0
    x✝ : NNReal
    hy0 : Ne (↑x✝) 0
    hxy : LE.le ↑x✝¹ ↑x✝
    ⊢ Eq (HPow.hPow (HMul.hMul ↑x✝¹ ↑x✝) z) (ite (And (Or (And (Eq (↑x✝¹) 0) (Eq ( …
  -/
  simp only [*, if_false]
  /-
    case inr.inr.coe.coe
    z : Real
    hz : Or (LT.lt z 0) (LT.lt 0 z)
    x✝¹ : NNReal
    hx0 : Ne (↑x✝¹) 0
    x✝ : NNReal
    hy0 : Ne (↑x✝) 0
    hxy : LE.le ↑x✝¹ ↑x✝
    ⊢ Eq (HPow.hPow (HMul.hMul ↑x✝¹ ↑x✝) z) (ite (And (Or (And False (Eq (↑x✝) Top …
  -/
  norm_cast at *
  /-
    case inr.inr.coe.coe
    z : Real
    x✝¹ x✝ : NNReal
    hz : Or (LT.lt z 0) (LT.lt 0 z)
    hx0 : Not (Eq x✝¹ 0)
    hy0 : Not (Eq x✝ 0)
    hxy : LE.le x✝¹ x✝
    ⊢ Eq (HPow.hPow (↑(HMul.hMul x✝¹ x✝)) z) (ite (And (Or (And False (Eq (↑x✝) To …
  -/
  rw [← coe_rpow_of_ne_zero (mul_ne_zero hx0 hy0), NNReal.mul_rpow]
  /-
    case inr.inr.coe.coe
    z : Real
    x✝¹ x✝ : NNReal
    hz : Or (LT.lt z 0) (LT.lt 0 z)
    hx0 : Not (Eq x✝¹ 0)
    hy0 : Not (Eq x✝ 0)
    hxy : LE.le x✝¹ x✝
    ⊢ Eq (↑(HMul.hMul (HPow.hPow x✝¹ z) (HPow.hPow x✝ z))) (ite (And (Or (And Fals …
  -/
  norm_cast
  /-
    🎉 no goals
  -/


theorem mul_rpow_of_ne_top {x y : ℝ≥0∞} (hx : x ≠ ⊤) (hy : y ≠ ⊤) (z : ℝ) :
                                      /-
                                        x y : ENNReal
                                        hx : Ne x Top.top
                                        hy : Ne y Top.top
                                        z : Real
                                        ⊢ Eq (HPow.hPow (HMul.hMul x y) z) (HMul.hMul (HPow.hPow x z) (HPow.hPow y z))
                                      -/
    (x * y) ^ z = x ^ z * y ^ z := by simp [*, mul_rpow_eq_ite]
                                      /-
                                        🎉 no goals
                                      -/


@[norm_cast]
theorem coe_mul_rpow (x y : ℝ≥0) (z : ℝ) : ((x : ℝ≥0∞) * y) ^ z = (x : ℝ≥0∞) ^ z * (y : ℝ≥0∞) ^ z :=
  mul_rpow_of_ne_top coe_ne_top coe_ne_top z


theorem prod_coe_rpow {ι} (s : Finset ι) (f : ι → ℝ≥0) (r : ℝ) :
    ∏ i ∈ s, (f i : ℝ≥0∞) ^ r = ((∏ i ∈ s, f i : ℝ≥0) : ℝ≥0∞) ^ r := by
  classical
  induction s using Finset.induction with
  | empty => simp
  | insert hi ih => simp_rw [prod_insert hi, ih, ← coe_mul_rpow, coe_mul]


theorem mul_rpow_of_ne_zero {x y : ℝ≥0∞} (hx : x ≠ 0) (hy : y ≠ 0) (z : ℝ) :
                                      /-
                                        x y : ENNReal
                                        hx : Ne x 0
                                        hy : Ne y 0
                                        z : Real
                                        ⊢ Eq (HPow.hPow (HMul.hMul x y) z) (HMul.hMul (HPow.hPow x z) (HPow.hPow y z))
                                      -/
    (x * y) ^ z = x ^ z * y ^ z := by simp [*, mul_rpow_eq_ite]
                                      /-
                                        🎉 no goals
                                      -/


theorem mul_rpow_of_nonneg (x y : ℝ≥0∞) {z : ℝ} (hz : 0 ≤ z) : (x * y) ^ z = x ^ z * y ^ z := by
  /-
    x y : ENNReal
    z : Real
    hz : LE.le 0 z
    ⊢ Eq (HPow.hPow (HMul.hMul x y) z) (HMul.hMul (HPow.hPow x z) (HPow.hPow y z))
  -/
  simp [hz.not_lt, mul_rpow_eq_ite]
  /-
    🎉 no goals
  -/


theorem prod_rpow_of_ne_top {ι} {s : Finset ι} {f : ι → ℝ≥0∞} (hf : ∀ i ∈ s, f i ≠ ∞) (r : ℝ) :
    ∏ i ∈ s, f i ^ r = (∏ i ∈ s, f i) ^ r := by
  classical
  induction s using Finset.induction with
  | empty => simp
  | @insert i s hi ih =>
    have h2f : ∀ i ∈ s, f i ≠ ∞ := fun i hi ↦ hf i <| mem_insert_of_mem hi
    rw [prod_insert hi, prod_insert hi, ih h2f, ← mul_rpow_of_ne_top <| hf i <| mem_insert_self ..]
    apply prod_ne_top h2f


theorem prod_rpow_of_nonneg {ι} {s : Finset ι} {f : ι → ℝ≥0∞} {r : ℝ} (hr : 0 ≤ r) :
    ∏ i ∈ s, f i ^ r = (∏ i ∈ s, f i) ^ r := by
  classical
  induction s using Finset.induction with
  | empty => simp
  | insert hi ih => simp_rw [prod_insert hi, ih, ← mul_rpow_of_nonneg _ _ hr]


theorem inv_rpow (x : ℝ≥0∞) (y : ℝ) : x⁻¹ ^ y = (x ^ y)⁻¹ := by
  /-
    x : ENNReal
    y : Real
    ⊢ Eq (HPow.hPow (Inv.inv x) y) (Inv.inv (HPow.hPow x y))
  -/
  rcases eq_or_ne y 0 with (rfl | hy); · simp only [rpow_zero, inv_one]
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case inr
    x : ENNReal
    y : Real
    hy : Ne y 0
    ⊢ Eq (HPow.hPow (Inv.inv x) y) (Inv.inv (HPow.hPow x y))
  -/
  replace hy := hy.lt_or_lt
  /-
    case inr
    x : ENNReal
    y : Real
    hy : Or (LT.lt y 0) (LT.lt 0 y)
    ⊢ Eq (HPow.hPow (Inv.inv x) y) (Inv.inv (HPow.hPow x y))
  -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  rcases eq_or_ne x 0 with (rfl | h0); · cases hy <;> simp [*]
                                                      /-
                                                        🎉 no goals
                                                      -/
  /-
    case inr.inr
    x : ENNReal
    y : Real
    hy : Or (LT.lt y 0) (LT.lt 0 y)
    h0 : Ne x 0
    ⊢ Eq (HPow.hPow (Inv.inv x) y) (Inv.inv (HPow.hPow x y))
  -/
                                                         /-
                                                           🎉 no goals
                                                         -/
  rcases eq_or_ne x ⊤ with (rfl | h_top); · cases hy <;> simp [*]
                                                         /-
                                                           🎉 no goals
                                                         -/
  /-
    case inr.inr.inr
    x : ENNReal
    y : Real
    hy : Or (LT.lt y 0) (LT.lt 0 y)
    h0 : Ne x 0
    h_top : Ne x Top.top
    ⊢ Eq (HPow.hPow (Inv.inv x) y) (Inv.inv (HPow.hPow x y))
  -/
  apply ENNReal.eq_inv_of_mul_eq_one_left
  rw [← mul_rpow_of_ne_zero (ENNReal.inv_ne_zero.2 h_top) h0, ENNReal.inv_mul_cancel h0 h_top,
    one_rpow]


theorem div_rpow_of_nonneg (x y : ℝ≥0∞) {z : ℝ} (hz : 0 ≤ z) : (x / y) ^ z = x ^ z / y ^ z := by
  /-
    x y : ENNReal
    z : Real
    hz : LE.le 0 z
    ⊢ Eq (HPow.hPow (HDiv.hDiv x y) z) (HDiv.hDiv (HPow.hPow x z) (HPow.hPow y z))
  -/
  rw [div_eq_mul_inv, mul_rpow_of_nonneg _ _ hz, inv_rpow, div_eq_mul_inv]
  /-
    🎉 no goals
  -/


theorem strictMono_rpow_of_pos {z : ℝ} (h : 0 < z) : StrictMono fun x : ℝ≥0∞ => x ^ z := by
  /-
    z : Real
    h : LT.lt 0 z
    ⊢ StrictMono fun x => HPow.hPow x z
  -/
  intro x y hxy
  /-
    z : Real
    h : LT.lt 0 z
    x y : ENNReal
    hxy : LT.lt x y
    ⊢ LT.lt ((fun x => HPow.hPow x z) x) ((fun x => HPow.hPow x z) y)
  -/
  lift x to ℝ≥0 using ne_top_of_lt hxy
  /-
    case intro
    z : Real
    h : LT.lt 0 z
    y : ENNReal
    x : NNReal
    hxy : LT.lt (↑x) y
    ⊢ LT.lt ((fun x => HPow.hPow x z) ↑x) ((fun x => HPow.hPow x z) y)
  -/
  rcases eq_or_ne y ∞ with (rfl | hy)
    /-
      case intro.inl
      z : Real
      h : LT.lt 0 z
      x : NNReal
      hxy : LT.lt (↑x) Top.top
      ⊢ LT.lt ((fun x => HPow.hPow x z) ↑x) ((fun x => HPow.hPow x z) Top.top)
    -/
  · simp only [top_rpow_of_pos h, ← coe_rpow_of_nonneg _ h.le, coe_lt_top]
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      z : Real
      h : LT.lt 0 z
      y : ENNReal
      x : NNReal
      hxy : LT.lt (↑x) y
      hy : Ne y Top.top
      ⊢ LT.lt ((fun x => HPow.hPow x z) ↑x) ((fun x => HPow.hPow x z) y)
    -/
  · lift y to ℝ≥0 using hy
    /-
      case intro.inr.intro
      z : Real
      h : LT.lt 0 z
      x y : NNReal
      hxy : LT.lt ↑x ↑y
      ⊢ LT.lt ((fun x => HPow.hPow x z) ↑x) ((fun x => HPow.hPow x z) ↑y)
    -/
    simp only [← coe_rpow_of_nonneg _ h.le, NNReal.rpow_lt_rpow (coe_lt_coe.1 hxy) h, coe_lt_coe]
    /-
      🎉 no goals
    -/


theorem monotone_rpow_of_nonneg {z : ℝ} (h : 0 ≤ z) : Monotone fun x : ℝ≥0∞ => x ^ z :=
                                     /-
                                       z : Real
                                       h : LE.le 0 z
                                       h0 : Eq 0 z
                                       ⊢ Monotone fun x => HPow.hPow x 0
                                     -/
  h.eq_or_lt.elim (fun h0 => h0 ▸ by simp only [rpow_zero, monotone_const]) fun h0 =>
                                     /-
                                       🎉 no goals
                                     -/
    (strictMono_rpow_of_pos h0).monotone


/-- Bundles `fun x : ℝ≥0∞ => x ^ y` into an order isomorphism when `y : ℝ` is positive,
where the inverse is `fun x : ℝ≥0∞ => x ^ (1 / y)`. -/
@[simps! apply]
def orderIsoRpow (y : ℝ) (hy : 0 < y) : ℝ≥0∞ ≃o ℝ≥0∞ :=
  (strictMono_rpow_of_pos hy).orderIsoOfRightInverse (fun x => x ^ y) (fun x => x ^ (1 / y))
    fun x => by
    /-
      y : Real
      hy : LT.lt 0 y
      x : ENNReal
      ⊢ Eq ((fun x => HPow.hPow x y) ((fun x => HPow.hPow x (HDiv.hDiv 1 y)) x)) x
    -/
    dsimp
    /-
      y : Real
      hy : LT.lt 0 y
      x : ENNReal
      ⊢ Eq (HPow.hPow (HPow.hPow x (HDiv.hDiv 1 y)) y) x
    -/
    rw [← rpow_mul, one_div_mul_cancel hy.ne.symm, rpow_one]
    /-
      🎉 no goals
    -/


theorem orderIsoRpow_symm_apply (y : ℝ) (hy : 0 < y) :
    (orderIsoRpow y hy).symm = orderIsoRpow (1 / y) (one_div_pos.2 hy) := by
  /-
    y : Real
    hy : LT.lt 0 y
    ⊢ Eq (ENNReal.orderIsoRpow y hy).symm (ENNReal.orderIsoRpow (HDiv.hDiv 1 y) ⋯)
  -/
  simp only [orderIsoRpow, one_div_one_div]
  /-
    y : Real
    hy : LT.lt 0 y
    ⊢ Eq (StrictMono.orderIsoOfRightInverse (fun x => HPow.hPow x y) ⋯ (fun x => H …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[gcongr] theorem rpow_le_rpow {x y : ℝ≥0∞} {z : ℝ} (h₁ : x ≤ y) (h₂ : 0 ≤ z) : x ^ z ≤ y ^ z :=
  monotone_rpow_of_nonneg h₂ h₁


@[gcongr] theorem rpow_lt_rpow {x y : ℝ≥0∞} {z : ℝ} (h₁ : x < y) (h₂ : 0 < z) : x ^ z < y ^ z :=
  strictMono_rpow_of_pos h₂ h₁


theorem rpow_le_rpow_iff {x y : ℝ≥0∞} {z : ℝ} (hz : 0 < z) : x ^ z ≤ y ^ z ↔ x ≤ y :=
  (strictMono_rpow_of_pos hz).le_iff_le


theorem rpow_lt_rpow_iff {x y : ℝ≥0∞} {z : ℝ} (hz : 0 < z) : x ^ z < y ^ z ↔ x < y :=
  (strictMono_rpow_of_pos hz).lt_iff_lt


theorem le_rpow_inv_iff {x y : ℝ≥0∞} {z : ℝ} (hz : 0 < z) : x ≤ y ^ z⁻¹ ↔ x ^ z ≤ y := by
  /-
    x y : ENNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LE.le x (HPow.hPow y (Inv.inv z))) (LE.le (HPow.hPow x z) y)
  -/
  nth_rw 1 [← rpow_one x]
  /-
    x y : ENNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LE.le (HPow.hPow x 1) (HPow.hPow y (Inv.inv z))) (LE.le (HPow.hPow x z) …
  -/
  nth_rw 1 [← @mul_inv_cancel₀ _ _ z hz.ne']
  /-
    x y : ENNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LE.le (HPow.hPow x (HMul.hMul z (Inv.inv z))) (HPow.hPow y (Inv.inv z)) …
  -/
  rw [rpow_mul, @rpow_le_rpow_iff _ _ z⁻¹ (by simp [hz])]
  /-
    🎉 no goals
  -/


@[deprecated le_rpow_inv_iff (since := "2024-07-10")]
theorem le_rpow_one_div_iff {x y : ℝ≥0∞} {z : ℝ} (hz : 0 < z) : x ≤ y ^ (1 / z) ↔ x ^ z ≤ y := by
  /-
    x y : ENNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LE.le x (HPow.hPow y (HDiv.hDiv 1 z))) (LE.le (HPow.hPow x z) y)
  -/
  nth_rw 1 [← rpow_one x]
  /-
    x y : ENNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LE.le (HPow.hPow x 1) (HPow.hPow y (HDiv.hDiv 1 z))) (LE.le (HPow.hPow  …
  -/
  nth_rw 1 [← @mul_inv_cancel₀ _ _ z hz.ne']
  /-
    x y : ENNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LE.le (HPow.hPow x (HMul.hMul z (Inv.inv z))) (HPow.hPow y (HDiv.hDiv 1 …
  -/
  rw [rpow_mul, ← one_div, @rpow_le_rpow_iff _ _ (1 / z) (by simp [hz])]
  /-
    🎉 no goals
  -/


theorem rpow_inv_lt_iff {x y : ℝ≥0∞} {z : ℝ} (hz : 0 < z) : x ^ z⁻¹ < y ↔ x < y ^ z := by
  /-
    x y : ENNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LT.lt (HPow.hPow x (Inv.inv z)) y) (LT.lt x (HPow.hPow y z))
  -/
  simp only [← not_le, le_rpow_inv_iff hz]
  /-
    🎉 no goals
  -/


theorem lt_rpow_inv_iff {x y : ℝ≥0∞} {z : ℝ} (hz : 0 < z) : x < y ^ z⁻¹ ↔ x ^ z < y := by
  /-
    x y : ENNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LT.lt x (HPow.hPow y (Inv.inv z))) (LT.lt (HPow.hPow x z) y)
  -/
  nth_rw 1 [← rpow_one x]
  /-
    x y : ENNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LT.lt (HPow.hPow x 1) (HPow.hPow y (Inv.inv z))) (LT.lt (HPow.hPow x z) …
  -/
  nth_rw 1 [← @mul_inv_cancel₀ _ _ z (ne_of_lt hz).symm]
  /-
    x y : ENNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LT.lt (HPow.hPow x (HMul.hMul z (Inv.inv z))) (HPow.hPow y (Inv.inv z)) …
  -/
  rw [rpow_mul, @rpow_lt_rpow_iff _ _ z⁻¹ (by simp [hz])]
  /-
    🎉 no goals
  -/


@[deprecated lt_rpow_inv_iff (since := "2024-07-10")]
theorem lt_rpow_one_div_iff {x y : ℝ≥0∞} {z : ℝ} (hz : 0 < z) : x < y ^ (1 / z) ↔ x ^ z < y := by
  /-
    x y : ENNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LT.lt x (HPow.hPow y (HDiv.hDiv 1 z))) (LT.lt (HPow.hPow x z) y)
  -/
  nth_rw 1 [← rpow_one x]
  /-
    x y : ENNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LT.lt (HPow.hPow x 1) (HPow.hPow y (HDiv.hDiv 1 z))) (LT.lt (HPow.hPow  …
  -/
  nth_rw 1 [← @mul_inv_cancel₀ _ _ z (ne_of_lt hz).symm]
  /-
    x y : ENNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LT.lt (HPow.hPow x (HMul.hMul z (Inv.inv z))) (HPow.hPow y (HDiv.hDiv 1 …
  -/
  rw [rpow_mul, ← one_div, @rpow_lt_rpow_iff _ _ (1 / z) (by simp [hz])]
  /-
    🎉 no goals
  -/


theorem rpow_inv_le_iff {x y : ℝ≥0∞} {z : ℝ} (hz : 0 < z) : x ^ z⁻¹ ≤ y ↔ x ≤ y ^ z := by
  /-
    x y : ENNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LE.le (HPow.hPow x (Inv.inv z)) y) (LE.le x (HPow.hPow y z))
  -/
  nth_rw 1 [← ENNReal.rpow_one y]
  /-
    x y : ENNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LE.le (HPow.hPow x (Inv.inv z)) (HPow.hPow y 1)) (LE.le x (HPow.hPow y  …
  -/
  nth_rw 1 [← @mul_inv_cancel₀ _ _ z hz.ne.symm]
  /-
    x y : ENNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LE.le (HPow.hPow x (Inv.inv z)) (HPow.hPow y (HMul.hMul z (Inv.inv z))) …
  -/
  rw [ENNReal.rpow_mul, ENNReal.rpow_le_rpow_iff (inv_pos.2 hz)]
  /-
    🎉 no goals
  -/


@[deprecated rpow_inv_le_iff (since := "2024-07-10")]
theorem rpow_one_div_le_iff {x y : ℝ≥0∞} {z : ℝ} (hz : 0 < z) : x ^ (1 / z) ≤ y ↔ x ≤ y ^ z := by
  /-
    x y : ENNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LE.le (HPow.hPow x (HDiv.hDiv 1 z)) y) (LE.le x (HPow.hPow y z))
  -/
  nth_rw 1 [← ENNReal.rpow_one y]
  /-
    x y : ENNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LE.le (HPow.hPow x (HDiv.hDiv 1 z)) (HPow.hPow y 1)) (LE.le x (HPow.hPo …
  -/
  nth_rw 2 [← @mul_inv_cancel₀ _ _ z hz.ne.symm]
  /-
    x y : ENNReal
    z : Real
    hz : LT.lt 0 z
    ⊢ Iff (LE.le (HPow.hPow x (HDiv.hDiv 1 z)) (HPow.hPow y (HMul.hMul z (Inv.inv  …
  -/
  rw [ENNReal.rpow_mul, ← one_div, ENNReal.rpow_le_rpow_iff (one_div_pos.2 hz)]
  /-
    🎉 no goals
  -/


theorem rpow_lt_rpow_of_exponent_lt {x : ℝ≥0∞} {y z : ℝ} (hx : 1 < x) (hx' : x ≠ ⊤) (hyz : y < z) :
    x ^ y < x ^ z := by
  /-
    x : ENNReal
    y z : Real
    hx : LT.lt 1 x
    hx' : Ne x Top.top
    hyz : LT.lt y z
    ⊢ LT.lt (HPow.hPow x y) (HPow.hPow x z)
  -/
  lift x to ℝ≥0 using hx'
  /-
    case intro
    y z : Real
    hyz : LT.lt y z
    x : NNReal
    hx : LT.lt 1 ↑x
    ⊢ LT.lt (HPow.hPow (↑x) y) (HPow.hPow (↑x) z)
  -/
  rw [one_lt_coe_iff] at hx
  simp [← coe_rpow_of_ne_zero (ne_of_gt (lt_trans zero_lt_one hx)),
    NNReal.rpow_lt_rpow_of_exponent_lt hx hyz]


@[gcongr] theorem rpow_le_rpow_of_exponent_le {x : ℝ≥0∞} {y z : ℝ} (hx : 1 ≤ x) (hyz : y ≤ z) :
    x ^ y ≤ x ^ z := by
  /-
    x : ENNReal
    y z : Real
    hx : LE.le 1 x
    hyz : LE.le y z
    ⊢ LE.le (HPow.hPow x y) (HPow.hPow x z)
  -/
  cases x
    /-
      case top
      y z : Real
      hyz : LE.le y z
      hx : LE.le 1 Top.top
      ⊢ LE.le (HPow.hPow Top.top y) (HPow.hPow Top.top z)
    -/
  · rcases lt_trichotomy y 0 with (Hy | Hy | Hy) <;>
    /-
      case top.inl
      y z : Real
      hyz : LE.le y z
      hx : LE.le 1 Top.top
      Hy : LT.lt y 0
      ⊢ LE.le (HPow.hPow Top.top y) (HPow.hPow Top.top z)
    -/
    rcases lt_trichotomy z 0 with (Hz | Hz | Hz) <;>
    /-
      case top.inl.inl
      y z : Real
      hyz : LE.le y z
      hx : LE.le 1 Top.top
      Hy : LT.lt y 0
      Hz : LT.lt z 0
      ⊢ LE.le (HPow.hPow Top.top y) (HPow.hPow Top.top z)
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
    simp [Hy, Hz, top_rpow_of_neg, top_rpow_of_pos, le_refl] <;>
    /-
      🎉 no goals
    -/
    /-
      case top.inr.inl.inl
      y z : Real
      hyz : LE.le y z
      hx : LE.le 1 Top.top
      Hy : Eq y 0
      Hz : LT.lt z 0
      ⊢ False
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    linarith
    /-
      🎉 no goals
    -/
    /-
      case coe
      y z : Real
      hyz : LE.le y z
      x✝ : NNReal
      hx : LE.le 1 ↑x✝
      ⊢ LE.le (HPow.hPow (↑x✝) y) (HPow.hPow (↑x✝) z)
    -/
  · simp only [one_le_coe_iff, some_eq_coe] at hx
    simp [← coe_rpow_of_ne_zero (ne_of_gt (lt_of_lt_of_le zero_lt_one hx)),
      NNReal.rpow_le_rpow_of_exponent_le hx hyz]


theorem rpow_lt_rpow_of_exponent_gt {x : ℝ≥0∞} {y z : ℝ} (hx0 : 0 < x) (hx1 : x < 1) (hyz : z < y) :
    x ^ y < x ^ z := by
  /-
    x : ENNReal
    y z : Real
    hx0 : LT.lt 0 x
    hx1 : LT.lt x 1
    hyz : LT.lt z y
    ⊢ LT.lt (HPow.hPow x y) (HPow.hPow x z)
  -/
  lift x to ℝ≥0 using ne_of_lt (lt_of_lt_of_le hx1 le_top)
  /-
    case intro
    y z : Real
    hyz : LT.lt z y
    x : NNReal
    hx0 : LT.lt 0 ↑x
    hx1 : LT.lt (↑x) 1
    ⊢ LT.lt (HPow.hPow (↑x) y) (HPow.hPow (↑x) z)
  -/
  simp only [coe_lt_one_iff, coe_pos] at hx0 hx1
  /-
    case intro
    y z : Real
    hyz : LT.lt z y
    x : NNReal
    hx0 : LT.lt 0 x
    hx1 : LT.lt x 1
    ⊢ LT.lt (HPow.hPow (↑x) y) (HPow.hPow (↑x) z)
  -/
  simp [← coe_rpow_of_ne_zero (ne_of_gt hx0), NNReal.rpow_lt_rpow_of_exponent_gt hx0 hx1 hyz]
  /-
    🎉 no goals
  -/


theorem rpow_le_rpow_of_exponent_ge {x : ℝ≥0∞} {y z : ℝ} (hx1 : x ≤ 1) (hyz : z ≤ y) :
    x ^ y ≤ x ^ z := by
  /-
    x : ENNReal
    y z : Real
    hx1 : LE.le x 1
    hyz : LE.le z y
    ⊢ LE.le (HPow.hPow x y) (HPow.hPow x z)
  -/
  lift x to ℝ≥0 using ne_of_lt (lt_of_le_of_lt hx1 coe_lt_top)
  /-
    case intro
    y z : Real
    hyz : LE.le z y
    x : NNReal
    hx1 : LE.le (↑x) 1
    ⊢ LE.le (HPow.hPow (↑x) y) (HPow.hPow (↑x) z)
  -/
  by_cases h : x = 0
    /-
      case pos
      y z : Real
      hyz : LE.le z y
      x : NNReal
      hx1 : LE.le (↑x) 1
      h : Eq x 0
      ⊢ LE.le (HPow.hPow (↑x) y) (HPow.hPow (↑x) z)
    -/
  · rcases lt_trichotomy y 0 with (Hy | Hy | Hy) <;>
    /-
      case pos.inl
      y z : Real
      hyz : LE.le z y
      x : NNReal
      hx1 : LE.le (↑x) 1
      h : Eq x 0
      Hy : LT.lt y 0
      ⊢ LE.le (HPow.hPow (↑x) y) (HPow.hPow (↑x) z)
    -/
    rcases lt_trichotomy z 0 with (Hz | Hz | Hz) <;>
    /-
      case pos.inl.inl
      y z : Real
      hyz : LE.le z y
      x : NNReal
      hx1 : LE.le (↑x) 1
      h : Eq x 0
      Hy : LT.lt y 0
      Hz : LT.lt z 0
      ⊢ LE.le (HPow.hPow (↑x) y) (HPow.hPow (↑x) z)
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
    simp [Hy, Hz, h, zero_rpow_of_neg, zero_rpow_of_pos, le_refl] <;>
    /-
      🎉 no goals
    -/
    /-
      case pos.inl.inr.inl
      y z : Real
      hyz : LE.le z y
      x : NNReal
      hx1 : LE.le (↑x) 1
      h : Eq x 0
      Hy : LT.lt y 0
      Hz : Eq z 0
      ⊢ False
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    linarith
    /-
      🎉 no goals
    -/
    /-
      case neg
      y z : Real
      hyz : LE.le z y
      x : NNReal
      hx1 : LE.le (↑x) 1
      h : Not (Eq x 0)
      ⊢ LE.le (HPow.hPow (↑x) y) (HPow.hPow (↑x) z)
    -/
  · rw [coe_le_one_iff] at hx1
    simp [← coe_rpow_of_ne_zero h,
      NNReal.rpow_le_rpow_of_exponent_ge (bot_lt_iff_ne_bot.mpr h) hx1 hyz]


theorem rpow_le_self_of_le_one {x : ℝ≥0∞} {z : ℝ} (hx : x ≤ 1) (h_one_le : 1 ≤ z) : x ^ z ≤ x := by
  /-
    x : ENNReal
    z : Real
    hx : LE.le x 1
    h_one_le : LE.le 1 z
    ⊢ LE.le (HPow.hPow x z) x
  -/
  nth_rw 2 [← ENNReal.rpow_one x]
  /-
    x : ENNReal
    z : Real
    hx : LE.le x 1
    h_one_le : LE.le 1 z
    ⊢ LE.le (HPow.hPow x z) (HPow.hPow x 1)
  -/
  exact ENNReal.rpow_le_rpow_of_exponent_ge hx h_one_le
  /-
    🎉 no goals
  -/


theorem le_rpow_self_of_one_le {x : ℝ≥0∞} {z : ℝ} (hx : 1 ≤ x) (h_one_le : 1 ≤ z) : x ≤ x ^ z := by
  /-
    x : ENNReal
    z : Real
    hx : LE.le 1 x
    h_one_le : LE.le 1 z
    ⊢ LE.le x (HPow.hPow x z)
  -/
  nth_rw 1 [← ENNReal.rpow_one x]
  /-
    x : ENNReal
    z : Real
    hx : LE.le 1 x
    h_one_le : LE.le 1 z
    ⊢ LE.le (HPow.hPow x 1) (HPow.hPow x z)
  -/
  exact ENNReal.rpow_le_rpow_of_exponent_le hx h_one_le
  /-
    🎉 no goals
  -/


theorem rpow_pos_of_nonneg {p : ℝ} {x : ℝ≥0∞} (hx_pos : 0 < x) (hp_nonneg : 0 ≤ p) : 0 < x ^ p := by
  /-
    p : Real
    x : ENNReal
    hx_pos : LT.lt 0 x
    hp_nonneg : LE.le 0 p
    ⊢ LT.lt 0 (HPow.hPow x p)
  -/
  by_cases hp_zero : p = 0
    /-
      case pos
      p : Real
      x : ENNReal
      hx_pos : LT.lt 0 x
      hp_nonneg : LE.le 0 p
      hp_zero : Eq p 0
      ⊢ LT.lt 0 (HPow.hPow x p)
    -/
  · simp [hp_zero, zero_lt_one]
    /-
      🎉 no goals
    -/
    /-
      case neg
      p : Real
      x : ENNReal
      hx_pos : LT.lt 0 x
      hp_nonneg : LE.le 0 p
      hp_zero : Not (Eq p 0)
      ⊢ LT.lt 0 (HPow.hPow x p)
    -/
  · rw [← Ne] at hp_zero
    /-
      case neg
      p : Real
      x : ENNReal
      hx_pos : LT.lt 0 x
      hp_nonneg : LE.le 0 p
      hp_zero : Ne p 0
      ⊢ LT.lt 0 (HPow.hPow x p)
    -/
    have hp_pos := lt_of_le_of_ne hp_nonneg hp_zero.symm
    /-
      case neg
      p : Real
      x : ENNReal
      hx_pos : LT.lt 0 x
      hp_nonneg : LE.le 0 p
      hp_zero : Ne p 0
      hp_pos : LT.lt 0 p
      ⊢ LT.lt 0 (HPow.hPow x p)
    -/
    rw [← zero_rpow_of_pos hp_pos]
    /-
      case neg
      p : Real
      x : ENNReal
      hx_pos : LT.lt 0 x
      hp_nonneg : LE.le 0 p
      hp_zero : Ne p 0
      hp_pos : LT.lt 0 p
      ⊢ LT.lt (HPow.hPow 0 p) (HPow.hPow x p)
    -/
    exact rpow_lt_rpow hx_pos hp_pos
    /-
      🎉 no goals
    -/


theorem rpow_pos {p : ℝ} {x : ℝ≥0∞} (hx_pos : 0 < x) (hx_ne_top : x ≠ ⊤) : 0 < x ^ p := by
  /-
    p : Real
    x : ENNReal
    hx_pos : LT.lt 0 x
    hx_ne_top : Ne x Top.top
    ⊢ LT.lt 0 (HPow.hPow x p)
  -/
  cases' lt_or_le 0 p with hp_pos hp_nonpos
    /-
      case inl
      p : Real
      x : ENNReal
      hx_pos : LT.lt 0 x
      hx_ne_top : Ne x Top.top
      hp_pos : LT.lt 0 p
      ⊢ LT.lt 0 (HPow.hPow x p)
    -/
  · exact rpow_pos_of_nonneg hx_pos (le_of_lt hp_pos)
    /-
      🎉 no goals
    -/
    /-
      case inr
      p : Real
      x : ENNReal
      hx_pos : LT.lt 0 x
      hx_ne_top : Ne x Top.top
      hp_nonpos : LE.le p 0
      ⊢ LT.lt 0 (HPow.hPow x p)
    -/
  · rw [← neg_neg p, rpow_neg, ENNReal.inv_pos]
    /-
      case inr
      p : Real
      x : ENNReal
      hx_pos : LT.lt 0 x
      hx_ne_top : Ne x Top.top
      hp_nonpos : LE.le p 0
      ⊢ Ne (HPow.hPow x (Neg.neg p)) Top.top
    -/
    exact rpow_ne_top_of_nonneg (Right.nonneg_neg_iff.mpr hp_nonpos) hx_ne_top
    /-
      🎉 no goals
    -/


theorem rpow_lt_one {x : ℝ≥0∞} {z : ℝ} (hx : x < 1) (hz : 0 < z) : x ^ z < 1 := by
  /-
    x : ENNReal
    z : Real
    hx : LT.lt x 1
    hz : LT.lt 0 z
    ⊢ LT.lt (HPow.hPow x z) 1
  -/
  lift x to ℝ≥0 using ne_of_lt (lt_of_lt_of_le hx le_top)
  /-
    case intro
    z : Real
    hz : LT.lt 0 z
    x : NNReal
    hx : LT.lt (↑x) 1
    ⊢ LT.lt (HPow.hPow (↑x) z) 1
  -/
  simp only [coe_lt_one_iff] at hx
  /-
    case intro
    z : Real
    hz : LT.lt 0 z
    x : NNReal
    hx : LT.lt x 1
    ⊢ LT.lt (HPow.hPow (↑x) z) 1
  -/
  simp [← coe_rpow_of_nonneg _ (le_of_lt hz), NNReal.rpow_lt_one hx hz]
  /-
    🎉 no goals
  -/


theorem rpow_le_one {x : ℝ≥0∞} {z : ℝ} (hx : x ≤ 1) (hz : 0 ≤ z) : x ^ z ≤ 1 := by
  /-
    x : ENNReal
    z : Real
    hx : LE.le x 1
    hz : LE.le 0 z
    ⊢ LE.le (HPow.hPow x z) 1
  -/
  lift x to ℝ≥0 using ne_of_lt (lt_of_le_of_lt hx coe_lt_top)
  /-
    case intro
    z : Real
    hz : LE.le 0 z
    x : NNReal
    hx : LE.le (↑x) 1
    ⊢ LE.le (HPow.hPow (↑x) z) 1
  -/
  simp only [coe_le_one_iff] at hx
  /-
    case intro
    z : Real
    hz : LE.le 0 z
    x : NNReal
    hx : LE.le x 1
    ⊢ LE.le (HPow.hPow (↑x) z) 1
  -/
  simp [← coe_rpow_of_nonneg _ hz, NNReal.rpow_le_one hx hz]
  /-
    🎉 no goals
  -/


theorem rpow_lt_one_of_one_lt_of_neg {x : ℝ≥0∞} {z : ℝ} (hx : 1 < x) (hz : z < 0) : x ^ z < 1 := by
  /-
    x : ENNReal
    z : Real
    hx : LT.lt 1 x
    hz : LT.lt z 0
    ⊢ LT.lt (HPow.hPow x z) 1
  -/
  cases x
    /-
      case top
      z : Real
      hz : LT.lt z 0
      hx : LT.lt 1 Top.top
      ⊢ LT.lt (HPow.hPow Top.top z) 1
    -/
  · simp [top_rpow_of_neg hz, zero_lt_one]
    /-
      🎉 no goals
    -/
    /-
      case coe
      z : Real
      hz : LT.lt z 0
      x✝ : NNReal
      hx : LT.lt 1 ↑x✝
      ⊢ LT.lt (HPow.hPow (↑x✝) z) 1
    -/
  · simp only [some_eq_coe, one_lt_coe_iff] at hx
    simp [← coe_rpow_of_ne_zero (ne_of_gt (lt_trans zero_lt_one hx)),
      NNReal.rpow_lt_one_of_one_lt_of_neg hx hz]


theorem rpow_le_one_of_one_le_of_neg {x : ℝ≥0∞} {z : ℝ} (hx : 1 ≤ x) (hz : z < 0) : x ^ z ≤ 1 := by
  /-
    x : ENNReal
    z : Real
    hx : LE.le 1 x
    hz : LT.lt z 0
    ⊢ LE.le (HPow.hPow x z) 1
  -/
  cases x
    /-
      case top
      z : Real
      hz : LT.lt z 0
      hx : LE.le 1 Top.top
      ⊢ LE.le (HPow.hPow Top.top z) 1
    -/
  · simp [top_rpow_of_neg hz, zero_lt_one]
    /-
      🎉 no goals
    -/
    /-
      case coe
      z : Real
      hz : LT.lt z 0
      x✝ : NNReal
      hx : LE.le 1 ↑x✝
      ⊢ LE.le (HPow.hPow (↑x✝) z) 1
    -/
  · simp only [one_le_coe_iff, some_eq_coe] at hx
    simp [← coe_rpow_of_ne_zero (ne_of_gt (lt_of_lt_of_le zero_lt_one hx)),
      NNReal.rpow_le_one_of_one_le_of_nonpos hx (le_of_lt hz)]


theorem one_lt_rpow {x : ℝ≥0∞} {z : ℝ} (hx : 1 < x) (hz : 0 < z) : 1 < x ^ z := by
  /-
    x : ENNReal
    z : Real
    hx : LT.lt 1 x
    hz : LT.lt 0 z
    ⊢ LT.lt 1 (HPow.hPow x z)
  -/
  cases x
    /-
      case top
      z : Real
      hz : LT.lt 0 z
      hx : LT.lt 1 Top.top
      ⊢ LT.lt 1 (HPow.hPow Top.top z)
    -/
  · simp [top_rpow_of_pos hz]
    /-
      🎉 no goals
    -/
    /-
      case coe
      z : Real
      hz : LT.lt 0 z
      x✝ : NNReal
      hx : LT.lt 1 ↑x✝
      ⊢ LT.lt 1 (HPow.hPow (↑x✝) z)
    -/
  · simp only [some_eq_coe, one_lt_coe_iff] at hx
    /-
      case coe
      z : Real
      hz : LT.lt 0 z
      x✝ : NNReal
      hx : LT.lt 1 x✝
      ⊢ LT.lt 1 (HPow.hPow (↑x✝) z)
    -/
    simp [← coe_rpow_of_nonneg _ (le_of_lt hz), NNReal.one_lt_rpow hx hz]
    /-
      🎉 no goals
    -/


theorem one_le_rpow {x : ℝ≥0∞} {z : ℝ} (hx : 1 ≤ x) (hz : 0 < z) : 1 ≤ x ^ z := by
  /-
    x : ENNReal
    z : Real
    hx : LE.le 1 x
    hz : LT.lt 0 z
    ⊢ LE.le 1 (HPow.hPow x z)
  -/
  cases x
    /-
      case top
      z : Real
      hz : LT.lt 0 z
      hx : LE.le 1 Top.top
      ⊢ LE.le 1 (HPow.hPow Top.top z)
    -/
  · simp [top_rpow_of_pos hz]
    /-
      🎉 no goals
    -/
    /-
      case coe
      z : Real
      hz : LT.lt 0 z
      x✝ : NNReal
      hx : LE.le 1 ↑x✝
      ⊢ LE.le 1 (HPow.hPow (↑x✝) z)
    -/
  · simp only [one_le_coe_iff, some_eq_coe] at hx
    /-
      case coe
      z : Real
      hz : LT.lt 0 z
      x✝ : NNReal
      hx : LE.le 1 x✝
      ⊢ LE.le 1 (HPow.hPow (↑x✝) z)
    -/
    simp [← coe_rpow_of_nonneg _ (le_of_lt hz), NNReal.one_le_rpow hx (le_of_lt hz)]
    /-
      🎉 no goals
    -/


theorem one_lt_rpow_of_pos_of_lt_one_of_neg {x : ℝ≥0∞} {z : ℝ} (hx1 : 0 < x) (hx2 : x < 1)
    (hz : z < 0) : 1 < x ^ z := by
  /-
    x : ENNReal
    z : Real
    hx1 : LT.lt 0 x
    hx2 : LT.lt x 1
    hz : LT.lt z 0
    ⊢ LT.lt 1 (HPow.hPow x z)
  -/
  lift x to ℝ≥0 using ne_of_lt (lt_of_lt_of_le hx2 le_top)
  /-
    case intro
    z : Real
    hz : LT.lt z 0
    x : NNReal
    hx1 : LT.lt 0 ↑x
    hx2 : LT.lt (↑x) 1
    ⊢ LT.lt 1 (HPow.hPow (↑x) z)
  -/
  simp only [coe_lt_one_iff, coe_pos] at hx1 hx2 ⊢
  /-
    case intro
    z : Real
    hz : LT.lt z 0
    x : NNReal
    hx1 : LT.lt 0 x
    hx2 : LT.lt x 1
    ⊢ LT.lt 1 (HPow.hPow (↑x) z)
  -/
  simp [← coe_rpow_of_ne_zero (ne_of_gt hx1), NNReal.one_lt_rpow_of_pos_of_lt_one_of_neg hx1 hx2 hz]
  /-
    🎉 no goals
  -/


theorem one_le_rpow_of_pos_of_le_one_of_neg {x : ℝ≥0∞} {z : ℝ} (hx1 : 0 < x) (hx2 : x ≤ 1)
    (hz : z < 0) : 1 ≤ x ^ z := by
  /-
    x : ENNReal
    z : Real
    hx1 : LT.lt 0 x
    hx2 : LE.le x 1
    hz : LT.lt z 0
    ⊢ LE.le 1 (HPow.hPow x z)
  -/
  lift x to ℝ≥0 using ne_of_lt (lt_of_le_of_lt hx2 coe_lt_top)
  /-
    case intro
    z : Real
    hz : LT.lt z 0
    x : NNReal
    hx1 : LT.lt 0 ↑x
    hx2 : LE.le (↑x) 1
    ⊢ LE.le 1 (HPow.hPow (↑x) z)
  -/
  simp only [coe_le_one_iff, coe_pos] at hx1 hx2 ⊢
  simp [← coe_rpow_of_ne_zero (ne_of_gt hx1),
    NNReal.one_le_rpow_of_pos_of_le_one_of_nonpos hx1 hx2 (le_of_lt hz)]


@[simp] lemma toNNReal_rpow (x : ℝ≥0∞) (z : ℝ) : (x ^ z).toNNReal = x.toNNReal ^ z := by
  /-
    x : ENNReal
    z : Real
    ⊢ Eq (HPow.hPow x z).toNNReal (HPow.hPow x.toNNReal z)
  -/
  rcases lt_trichotomy z 0 with (H | H | H)
    /-
      case inl
      x : ENNReal
      z : Real
      H : LT.lt z 0
      ⊢ Eq (HPow.hPow x z).toNNReal (HPow.hPow x.toNNReal z)
    -/
  · cases' x with x
      /-
        case inl.top
        z : Real
        H : LT.lt z 0
        ⊢ Eq (HPow.hPow Top.top z).toNNReal (HPow.hPow Top.top.toNNReal z)
      -/
    · simp [H, ne_of_lt]
      /-
        🎉 no goals
      -/
    /-
      case inl.coe
      z : Real
      H : LT.lt z 0
      x : NNReal
      ⊢ Eq (HPow.hPow (↑x) z).toNNReal (HPow.hPow (↑x).toNNReal z)
    -/
    by_cases hx : x = 0
      /-
        case pos
        z : Real
        H : LT.lt z 0
        x : NNReal
        hx : Eq x 0
        ⊢ Eq (HPow.hPow (↑x) z).toNNReal (HPow.hPow (↑x).toNNReal z)
      -/
    · simp [hx, H, ne_of_lt]
      /-
        🎉 no goals
      -/
      /-
        case neg
        z : Real
        H : LT.lt z 0
        x : NNReal
        hx : Not (Eq x 0)
        ⊢ Eq (HPow.hPow (↑x) z).toNNReal (HPow.hPow (↑x).toNNReal z)
      -/
    · simp [← coe_rpow_of_ne_zero hx]
      /-
        🎉 no goals
      -/
    /-
      case inr.inl
      x : ENNReal
      z : Real
      H : Eq z 0
      ⊢ Eq (HPow.hPow x z).toNNReal (HPow.hPow x.toNNReal z)
    -/
  · simp [H]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      x : ENNReal
      z : Real
      H : LT.lt 0 z
      ⊢ Eq (HPow.hPow x z).toNNReal (HPow.hPow x.toNNReal z)
    -/
  · cases x
      /-
        case inr.inr.top
        z : Real
        H : LT.lt 0 z
        ⊢ Eq (HPow.hPow Top.top z).toNNReal (HPow.hPow Top.top.toNNReal z)
      -/
    · simp [H, ne_of_gt]
      /-
        🎉 no goals
      -/
    /-
      case inr.inr.coe
      z : Real
      H : LT.lt 0 z
      x✝ : NNReal
      ⊢ Eq (HPow.hPow (↑x✝) z).toNNReal (HPow.hPow (↑x✝).toNNReal z)
    -/
    simp [← coe_rpow_of_nonneg _ (le_of_lt H)]
    /-
      🎉 no goals
    -/


theorem toReal_rpow (x : ℝ≥0∞) (z : ℝ) : x.toReal ^ z = (x ^ z).toReal := by
  /-
    x : ENNReal
    z : Real
    ⊢ Eq (HPow.hPow x.toReal z) (HPow.hPow x z).toReal
  -/
  rw [ENNReal.toReal, ENNReal.toReal, ← NNReal.coe_rpow, ENNReal.toNNReal_rpow]
  /-
    🎉 no goals
  -/


theorem ofReal_rpow_of_pos {x p : ℝ} (hx_pos : 0 < x) :
    ENNReal.ofReal x ^ p = ENNReal.ofReal (x ^ p) := by
  /-
    x p : Real
    hx_pos : LT.lt 0 x
    ⊢ Eq (HPow.hPow (ENNReal.ofReal x) p) (ENNReal.ofReal (HPow.hPow x p))
  -/
  simp_rw [ENNReal.ofReal]
  /-
    x p : Real
    hx_pos : LT.lt 0 x
    ⊢ Eq (HPow.hPow (↑x.toNNReal) p) ↑(HPow.hPow x p).toNNReal
  -/
  rw [← coe_rpow_of_ne_zero, coe_inj, Real.toNNReal_rpow_of_nonneg hx_pos.le]
  /-
    case h
    x p : Real
    hx_pos : LT.lt 0 x
    ⊢ Ne x.toNNReal 0
  -/
  simp [hx_pos]
  /-
    🎉 no goals
  -/


theorem ofReal_rpow_of_nonneg {x p : ℝ} (hx_nonneg : 0 ≤ x) (hp_nonneg : 0 ≤ p) :
    ENNReal.ofReal x ^ p = ENNReal.ofReal (x ^ p) := by
  /-
    x p : Real
    hx_nonneg : LE.le 0 x
    hp_nonneg : LE.le 0 p
    ⊢ Eq (HPow.hPow (ENNReal.ofReal x) p) (ENNReal.ofReal (HPow.hPow x p))
  -/
  by_cases hp0 : p = 0
    /-
      case pos
      x p : Real
      hx_nonneg : LE.le 0 x
      hp_nonneg : LE.le 0 p
      hp0 : Eq p 0
      ⊢ Eq (HPow.hPow (ENNReal.ofReal x) p) (ENNReal.ofReal (HPow.hPow x p))
    -/
  · simp [hp0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    x p : Real
    hx_nonneg : LE.le 0 x
    hp_nonneg : LE.le 0 p
    hp0 : Not (Eq p 0)
    ⊢ Eq (HPow.hPow (ENNReal.ofReal x) p) (ENNReal.ofReal (HPow.hPow x p))
  -/
  by_cases hx0 : x = 0
    /-
      case pos
      x p : Real
      hx_nonneg : LE.le 0 x
      hp_nonneg : LE.le 0 p
      hp0 : Not (Eq p 0)
      hx0 : Eq x 0
      ⊢ Eq (HPow.hPow (ENNReal.ofReal x) p) (ENNReal.ofReal (HPow.hPow x p))
    -/
  · rw [← Ne] at hp0
    /-
      case pos
      x p : Real
      hx_nonneg : LE.le 0 x
      hp_nonneg : LE.le 0 p
      hp0 : Ne p 0
      hx0 : Eq x 0
      ⊢ Eq (HPow.hPow (ENNReal.ofReal x) p) (ENNReal.ofReal (HPow.hPow x p))
    -/
    have hp_pos : 0 < p := lt_of_le_of_ne hp_nonneg hp0.symm
    /-
      case pos
      x p : Real
      hx_nonneg : LE.le 0 x
      hp_nonneg : LE.le 0 p
      hp0 : Ne p 0
      hx0 : Eq x 0
      hp_pos : LT.lt 0 p
      ⊢ Eq (HPow.hPow (ENNReal.ofReal x) p) (ENNReal.ofReal (HPow.hPow x p))
    -/
    simp [hx0, hp_pos, hp_pos.ne.symm]
    /-
      🎉 no goals
    -/
  /-
    case neg
    x p : Real
    hx_nonneg : LE.le 0 x
    hp_nonneg : LE.le 0 p
    hp0 : Not (Eq p 0)
    hx0 : Not (Eq x 0)
    ⊢ Eq (HPow.hPow (ENNReal.ofReal x) p) (ENNReal.ofReal (HPow.hPow x p))
  -/
  rw [← Ne] at hx0
  /-
    case neg
    x p : Real
    hx_nonneg : LE.le 0 x
    hp_nonneg : LE.le 0 p
    hp0 : Not (Eq p 0)
    hx0 : Ne x 0
    ⊢ Eq (HPow.hPow (ENNReal.ofReal x) p) (ENNReal.ofReal (HPow.hPow x p))
  -/
  exact ofReal_rpow_of_pos (hx_nonneg.lt_of_ne hx0.symm)
  /-
    🎉 no goals
  -/


@[simp] lemma rpow_rpow_inv {y : ℝ} (hy : y ≠ 0) (x : ℝ≥0∞) : (x ^ y) ^ y⁻¹ = x := by
  /-
    y : Real
    hy : Ne y 0
    x : ENNReal
    ⊢ Eq (HPow.hPow (HPow.hPow x y) (Inv.inv y)) x
  -/
  rw [← rpow_mul, mul_inv_cancel₀ hy, rpow_one]
  /-
    🎉 no goals
  -/


@[simp] lemma rpow_inv_rpow {y : ℝ} (hy : y ≠ 0) (x : ℝ≥0∞) : (x ^ y⁻¹) ^ y = x := by
  /-
    y : Real
    hy : Ne y 0
    x : ENNReal
    ⊢ Eq (HPow.hPow (HPow.hPow x (Inv.inv y)) y) x
  -/
  rw [← rpow_mul, inv_mul_cancel₀ hy, rpow_one]
  /-
    🎉 no goals
  -/


lemma pow_rpow_inv_natCast {n : ℕ} (hn : n ≠ 0) (x : ℝ≥0∞) : (x ^ n) ^ (n⁻¹ : ℝ) = x := by
  /-
    n : Nat
    hn : Ne n 0
    x : ENNReal
    ⊢ Eq (HPow.hPow (HPow.hPow x n) (Inv.inv ↑n)) x
  -/
  rw [← rpow_natCast, ← rpow_mul, mul_inv_cancel₀ (by positivity), rpow_one]
  /-
    🎉 no goals
  -/


lemma rpow_inv_natCast_pow {n : ℕ} (hn : n ≠ 0) (x : ℝ≥0∞) : (x ^ (n⁻¹ : ℝ)) ^ n = x := by
  /-
    n : Nat
    hn : Ne n 0
    x : ENNReal
    ⊢ Eq (HPow.hPow (HPow.hPow x (Inv.inv ↑n)) n) x
  -/
  rw [← rpow_natCast, ← rpow_mul, inv_mul_cancel₀ (by positivity), rpow_one]
  /-
    🎉 no goals
  -/


lemma rpow_natCast_mul (x : ℝ≥0∞) (n : ℕ) (z : ℝ) : x ^ (n * z) = (x ^ n) ^ z := by
  /-
    x : ENNReal
    n : Nat
    z : Real
    ⊢ Eq (HPow.hPow x (HMul.hMul (↑n) z)) (HPow.hPow (HPow.hPow x n) z)
  -/
  rw [rpow_mul, rpow_natCast]
  /-
    🎉 no goals
  -/


lemma rpow_mul_natCast (x : ℝ≥0∞) (y : ℝ) (n : ℕ) : x ^ (y * n) = (x ^ y) ^ n := by
  /-
    x : ENNReal
    y : Real
    n : Nat
    ⊢ Eq (HPow.hPow x (HMul.hMul y ↑n)) (HPow.hPow (HPow.hPow x y) n)
  -/
  rw [rpow_mul, rpow_natCast]
  /-
    🎉 no goals
  -/


lemma rpow_intCast_mul (x : ℝ≥0∞) (n : ℤ) (z : ℝ) : x ^ (n * z) = (x ^ n) ^ z := by
  /-
    x : ENNReal
    n : Int
    z : Real
    ⊢ Eq (HPow.hPow x (HMul.hMul (↑n) z)) (HPow.hPow (HPow.hPow x n) z)
  -/
  rw [rpow_mul, rpow_intCast]
  /-
    🎉 no goals
  -/


lemma rpow_mul_intCast (x : ℝ≥0∞) (y : ℝ) (n : ℤ) : x ^ (y * n) = (x ^ y) ^ n := by
  /-
    x : ENNReal
    y : Real
    n : Int
    ⊢ Eq (HPow.hPow x (HMul.hMul y ↑n)) (HPow.hPow (HPow.hPow x y) n)
  -/
  rw [rpow_mul, rpow_intCast]
  /-
    🎉 no goals
  -/


lemma rpow_left_injective {x : ℝ} (hx : x ≠ 0) : Injective fun y : ℝ≥0∞ ↦ y ^ x :=
  HasLeftInverse.injective ⟨fun y ↦ y ^ x⁻¹, rpow_rpow_inv hx⟩


theorem rpow_left_surjective {x : ℝ} (hx : x ≠ 0) : Function.Surjective fun y : ℝ≥0∞ => y ^ x :=
  HasRightInverse.surjective ⟨fun y ↦ y ^ x⁻¹, rpow_inv_rpow hx⟩


theorem rpow_left_bijective {x : ℝ} (hx : x ≠ 0) : Function.Bijective fun y : ℝ≥0∞ => y ^ x :=
  ⟨rpow_left_injective hx, rpow_left_surjective hx⟩



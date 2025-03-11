private lemma tendsto_const_rpow_inv {C : ℝ} (hC : 0 < C) :
    Tendsto (fun k : ℕ ↦ C ^ (k : ℝ)⁻¹) atTop (𝓝 1) :=
  ((continuous_iff_continuousAt.mpr fun _ ↦ continuousAt_const_rpow hC.ne').tendsto'
    0 1 (rpow_zero C)).comp <| tendsto_inv_atTop_zero.comp tendsto_natCast_atTop_atTop

--extends the lemma `tendsto_rpow_div` when the function has natural input

private lemma tendsto_nat_rpow_inv :
    Tendsto (fun k : ℕ ↦ (k : ℝ) ^ (k : ℝ)⁻¹) atTop (𝓝 1) := by
  /-
    ⊢ Filter.Tendsto (fun k => HPow.hPow (↑k) (Inv.inv ↑k)) Filter.atTop (nhds 1)
  -/
  simp_rw [← one_div]
  /-
    ⊢ Filter.Tendsto (fun k => HPow.hPow (↑k) (HDiv.hDiv 1 ↑k)) Filter.atTop (nhds …
  -/
  exact Tendsto.comp tendsto_rpow_div tendsto_natCast_atTop_atTop
  /-
    🎉 no goals
  -/

-- Multiplication by a constant moves in a List.sum

private lemma list_mul_sum {R : Type*} [CommSemiring R] {T : Type*} (l : List T) (y : R) (x : R) :
    (l.mapIdx fun i _ => x * y ^ i).sum = x * (l.mapIdx fun i _ => y ^ i).sum := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    T : Type u_2
    l : List T
    y x : R
    ⊢ Eq (List.mapIdx (fun i x_1 => HMul.hMul x (HPow.hPow y i)) l).sum (HMul.hMul …
  -/
  simp_rw [← smul_eq_mul, List.smul_sum, List.mapIdx_eq_enum_map]
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    T : Type u_2
    l : List T
    y x : R
    ⊢ Eq (List.map (Function.uncurry fun i x_1 => HSMul.hSMul x (HPow.hPow y i)) l …
  -/
  congr 1
  /-
    case e_a
    R : Type u_1
    inst✝ : CommSemiring R
    T : Type u_2
    l : List T
    y x : R
    ⊢ Eq (List.map (Function.uncurry fun i x_1 => HSMul.hSMul x (HPow.hPow y i)) l …
  -/
  simp
  /-
    🎉 no goals
  -/

-- Geometric sum for lists

private lemma list_geom {T : Type*} {F : Type*} [Field F] (l : List T) {y : F} (hy : y ≠ 1) :
    (l.mapIdx fun i _ => y ^ i).sum = (y ^ l.length - 1) / (y - 1) := by
  /-
    T : Type u_1
    F : Type u_2
    inst✝ : Field F
    l : List T
    y : F
    hy : Ne y 1
    ⊢ Eq (List.mapIdx (fun i x => HPow.hPow y i) l).sum (HDiv.hDiv (HSub.hSub (HPo …
  -/
  rw [← geom_sum_eq hy l.length, List.mapIdx_eq_enum_map, Finset.sum_range, ← Fin.sum_univ_get']
  /-
    T : Type u_1
    F : Type u_2
    inst✝ : Field F
    l : List T
    y : F
    hy : Ne y 1
    ⊢ Eq (Finset.univ.sum fun i => Function.uncurry (fun i x => HPow.hPow y i) (Ge …
  -/
  simp only [List.getElem_enum, Function.uncurry_apply_pair]
  /-
    T : Type u_1
    F : Type u_2
    inst✝ : Field F
    l : List T
    y : F
    hy : Ne y 1
    ⊢ Eq (Finset.univ.sum fun x => HPow.hPow y ↑x) (Finset.univ.sum fun i => HPow. …
  -/
  let e : Fin l.enum.length ≃ Fin l.length := finCongr List.enum_length
  /-
    T : Type u_1
    F : Type u_2
    inst✝ : Field F
    l : List T
    y : F
    hy : Ne y 1
    e : Equiv (Fin l.enum.length) (Fin l.length) := finCongr ⋯
    ⊢ Eq (Finset.univ.sum fun x => HPow.hPow y ↑x) (Finset.univ.sum fun i => HPow. …
  -/
  exact Fintype.sum_bijective e e.bijective _ _ fun _ ↦ rfl
  /-
    🎉 no goals
  -/


/-- Values of an absolute value on the rationals are determined by the values on the natural
numbers. -/
lemma eq_on_nat_iff_eq : (∀ n : ℕ , f n = g n) ↔ f = g := by
  /-
    f g : AbsoluteValue Rat Real
    ⊢ Iff (∀ (n : Nat), Eq (f ↑n) (g ↑n)) (Eq f g)
  -/
  refine ⟨fun h ↦ ?_, fun h n ↦ congrFun (congrArg DFunLike.coe h) ↑n⟩
  /-
    f g : AbsoluteValue Rat Real
    h : ∀ (n : Nat), Eq (f ↑n) (g ↑n)
    ⊢ Eq f g
  -/
  ext1 z
  /-
    case a
    f g : AbsoluteValue Rat Real
    h : ∀ (n : Nat), Eq (f ↑n) (g ↑n)
    z : Rat
    ⊢ Eq (f z) (g z)
  -/
  rw [← Rat.num_div_den z, map_div₀, map_div₀, h, eq_on_nat_iff_eq_on_int.mp h]
  /-
    🎉 no goals
  -/


/-- The equivalence class of an absolute value on the rationals is determined by its values on
the natural numbers. -/
lemma equiv_on_nat_iff_equiv : (∃ c : ℝ, 0 < c ∧ ∀ n : ℕ , f n ^ c = g n) ↔ f ≈ g := by
  /-
    f g : AbsoluteValue Rat Real
    ⊢ Iff (Exists fun c => And (LT.lt 0 c) (∀ (n : Nat), Eq (HPow.hPow (f ↑n) c) ( …
  -/
  refine ⟨fun ⟨c, hc, h⟩ ↦ ⟨c, hc, ?_⟩, fun ⟨c, hc, h⟩ ↦ ⟨c, hc, (congrFun h ·)⟩⟩
  /-
    f g : AbsoluteValue Rat Real
    x✝ : Exists fun c => And (LT.lt 0 c) (∀ (n : Nat), Eq (HPow.hPow (f ↑n) c) (g  …
    c : Real
    hc : LT.lt 0 c
    h : ∀ (n : Nat), Eq (HPow.hPow (f ↑n) c) (g ↑n)
    ⊢ Eq (fun x => HPow.hPow (f x) c) ⇑g
  -/
  ext1 x
  rw [← Rat.num_div_den x, map_div₀, map_div₀, div_rpow (by positivity) (by positivity), h x.den,
    ← apply_natAbs_eq,← apply_natAbs_eq, h (natAbs x.num)]


/-- The real-valued `AbsoluteValue` corresponding to the p-adic norm on `ℚ`. -/
def padic (p : ℕ) [Fact p.Prime] : AbsoluteValue ℚ ℝ where
  toFun x := (padicNorm p x : ℝ)
                 /-
                   f g : AbsoluteValue Rat Real
                   p : Nat
                   inst✝ : Fact (Nat.Prime p)
                   ⊢ ∀ (x y : Rat), Eq ((fun x => ↑(padicNorm p x)) (HMul.hMul x y)) (HMul.hMul ( …
                 -/
  map_mul' := by simp only [padicNorm.mul, Rat.cast_mul, forall_const]
                 /-
                   🎉 no goals
                 -/
  nonneg' x := cast_nonneg.mpr <| padicNorm.nonneg x
  eq_zero' x :=
    ⟨fun H ↦ padicNorm.zero_of_padicNorm_eq_zero <| cast_eq_zero.mp H,
      fun H ↦ cast_eq_zero.mpr <| H ▸ padicNorm.zero (p := p)⟩
                    /-
                      f g : AbsoluteValue Rat Real
                      p : Nat
                      inst✝ : Fact (Nat.Prime p)
                      x y : Rat
                      ⊢ LE.le ({ toFun := fun x => ↑(padicNorm p x), map_mul' := ⋯ }.toFun (HAdd.hAd …
                    -/
  add_le' x y := by simp only; exact_mod_cast padicNorm.triangle_ineq x y
                               /-
                                 🎉 no goals
                               -/


@[simp] lemma padic_eq_padicNorm (p : ℕ) [Fact p.Prime] (r : ℚ) :
    padic p r = padicNorm p r := rfl


lemma padic_le_one (p : ℕ) [Fact p.Prime] (n : ℤ) : padic p n ≤ 1 := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    n : Int
    ⊢ LE.le ((Rat.AbsoluteValue.padic p) ↑n) 1
  -/
  simp only [padic_eq_padicNorm]
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    n : Int
    ⊢ LE.le (↑(padicNorm p ↑n)) 1
  -/
  exact_mod_cast padicNorm.of_int n
  /-
    🎉 no goals
  -/

-- ## Step 1: define `p = minimal n s. t. 0 < f n < 1`


include hf_nontriv bdd in
/-- There exists a minimal positive integer with absolute value smaller than 1. -/
lemma exists_minimal_nat_zero_lt_and_lt_one :
    ∃ p : ℕ, (0 < f p ∧ f p < 1) ∧ ∀ m : ℕ, 0 < f m ∧ f m < 1 → p ≤ m := by
  -- There is a positive integer with absolute value different from one.
  obtain ⟨n, hn1, hn2⟩ : ∃ n : ℕ, n ≠ 0 ∧ f n ≠ 1 := by
    contrapose! hf_nontriv
    rw [AbsoluteValue.trivial, ← eq_on_nat_iff_eq]
    intro n
    rcases eq_or_ne n 0 with rfl | hn0
    · simp
    · simp [hn0, hf_nontriv n hn0]
  /-
    case intro.intro
    f : AbsoluteValue Rat Real
    hf_nontriv : Ne f AbsoluteValue.trivial
    bdd : ∀ (n : Nat), LE.le (f ↑n) 1
    n : Nat
    hn1 : Ne n 0
    hn2 : Ne (f ↑n) 1
    ⊢ Exists fun p => And (And (LT.lt 0 (f ↑p)) (LT.lt (f ↑p) 1)) (∀ (m : Nat), An …
  -/
  set P := {m : ℕ | 0 < f ↑m ∧ f ↑m < 1} -- p is going to be the minimum of this set.
  have hP : P.Nonempty :=
    ⟨n, map_pos_of_ne_zero f (Nat.cast_ne_zero.mpr hn1), lt_of_le_of_ne (bdd n) hn2⟩
  /-
    case intro.intro
    f : AbsoluteValue Rat Real
    hf_nontriv : Ne f AbsoluteValue.trivial
    bdd : ∀ (n : Nat), LE.le (f ↑n) 1
    n : Nat
    hn1 : Ne n 0
    hn2 : Ne (f ↑n) 1
    P : Set Nat := setOf fun m => And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1)
    hP : P.Nonempty
    ⊢ Exists fun p => And (And (LT.lt 0 (f ↑p)) (LT.lt (f ↑p) 1)) (∀ (m : Nat), An …
  -/
  exact ⟨sInf P, Nat.sInf_mem hP, fun m hm ↦ Nat.sInf_le hm⟩
  /-
    🎉 no goals
  -/

-- ## Step 2: p is prime


include hp0 hp1 hmin in
/-- The minimal positive integer with absolute value smaller than 1 is a prime number.-/
lemma is_prime_of_minimal_nat_zero_lt_and_lt_one : p.Prime := by
  /-
    f : AbsoluteValue Rat Real
    p : Nat
    hp0 : LT.lt 0 (f ↑p)
    hp1 : LT.lt (f ↑p) 1
    hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
    ⊢ Nat.Prime p
  -/
  rw [← Nat.irreducible_iff_nat_prime]
  /-
    f : AbsoluteValue Rat Real
    p : Nat
    hp0 : LT.lt 0 (f ↑p)
    hp1 : LT.lt (f ↑p) 1
    hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
    ⊢ Irreducible p
  -/
  constructor -- Two goals: p is not a unit and any product giving p must contain a unit.
    /-
      case not_unit
      f : AbsoluteValue Rat Real
      p : Nat
      hp0 : LT.lt 0 (f ↑p)
      hp1 : LT.lt (f ↑p) 1
      hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
      ⊢ Not (IsUnit p)
    -/
  · rw [Nat.isUnit_iff]
    /-
      case not_unit
      f : AbsoluteValue Rat Real
      p : Nat
      hp0 : LT.lt 0 (f ↑p)
      hp1 : LT.lt (f ↑p) 1
      hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
      ⊢ Not (Eq p 1)
    -/
    rintro rfl
    /-
      case not_unit
      f : AbsoluteValue Rat Real
      hp0 : LT.lt 0 (f ↑1)
      hp1 : LT.lt (f ↑1) 1
      hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le 1 m
      ⊢ False
    -/
    simp only [Nat.cast_one, map_one, lt_self_iff_false] at hp1
    /-
      🎉 no goals
    -/
    /-
      case isUnit_or_isUnit'
      f : AbsoluteValue Rat Real
      p : Nat
      hp0 : LT.lt 0 (f ↑p)
      hp1 : LT.lt (f ↑p) 1
      hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
      ⊢ ∀ (a b : Nat), Eq p (HMul.hMul a b) → Or (IsUnit a) (IsUnit b)
    -/
  · rintro a b rfl
    /-
      case isUnit_or_isUnit'
      f : AbsoluteValue Rat Real
      a b : Nat
      hp0 : LT.lt 0 (f ↑(HMul.hMul a b))
      hp1 : LT.lt (f ↑(HMul.hMul a b)) 1
      hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le (HMul.hMul a …
      ⊢ Or (IsUnit a) (IsUnit b)
    -/
    rw [Nat.isUnit_iff, Nat.isUnit_iff]
    /-
      case isUnit_or_isUnit'
      f : AbsoluteValue Rat Real
      a b : Nat
      hp0 : LT.lt 0 (f ↑(HMul.hMul a b))
      hp1 : LT.lt (f ↑(HMul.hMul a b)) 1
      hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le (HMul.hMul a …
      ⊢ Or (Eq a 1) (Eq b 1)
    -/
    by_contra! con
    /-
      case isUnit_or_isUnit'
      f : AbsoluteValue Rat Real
      a b : Nat
      hp0 : LT.lt 0 (f ↑(HMul.hMul a b))
      hp1 : LT.lt (f ↑(HMul.hMul a b)) 1
      hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le (HMul.hMul a …
      con : And (Ne a 1) (Ne b 1)
      ⊢ False
    -/
    obtain ⟨ha₁, hb₁⟩ := con
    obtain ⟨ha₀, hb₀⟩ : a ≠ 0 ∧ b ≠ 0 := by
      refine mul_ne_zero_iff.mp fun h ↦ ?_
      rwa [h, Nat.cast_zero, map_zero, lt_self_iff_false] at hp0
    /-
      case isUnit_or_isUnit'.intro.intro
      f : AbsoluteValue Rat Real
      a b : Nat
      hp0 : LT.lt 0 (f ↑(HMul.hMul a b))
      hp1 : LT.lt (f ↑(HMul.hMul a b)) 1
      hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le (HMul.hMul a …
      ha₁ : Ne a 1
      hb₁ : Ne b 1
      ha₀ : Ne a 0
      hb₀ : Ne b 0
      ⊢ False
    -/
    have hap : a < a * b := lt_mul_of_one_lt_right (by omega) (by omega)
    /-
      case isUnit_or_isUnit'.intro.intro
      f : AbsoluteValue Rat Real
      a b : Nat
      hp0 : LT.lt 0 (f ↑(HMul.hMul a b))
      hp1 : LT.lt (f ↑(HMul.hMul a b)) 1
      hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le (HMul.hMul a …
      ha₁ : Ne a 1
      hb₁ : Ne b 1
      ha₀ : Ne a 0
      hb₀ : Ne b 0
      hap : LT.lt a (HMul.hMul a b)
      ⊢ False
    -/
    have hbp : b < a * b := lt_mul_of_one_lt_left (by omega) (by omega)
    have ha :=
      le_of_not_lt <| not_and.mp ((hmin a).mt hap.not_le) (map_pos_of_ne_zero f (mod_cast ha₀))
    have hb :=
      le_of_not_lt <| not_and.mp ((hmin b).mt hbp.not_le) (map_pos_of_ne_zero f (mod_cast hb₀))
    /-
      case isUnit_or_isUnit'.intro.intro
      f : AbsoluteValue Rat Real
      a b : Nat
      hp0 : LT.lt 0 (f ↑(HMul.hMul a b))
      hp1 : LT.lt (f ↑(HMul.hMul a b)) 1
      hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le (HMul.hMul a …
      ha₁ : Ne a 1
      hb₁ : Ne b 1
      ha₀ : Ne a 0
      hb₀ : Ne b 0
      hap : LT.lt a (HMul.hMul a b)
      hbp : LT.lt b (HMul.hMul a b)
      ha : LE.le 1 (f ↑a)
      hb : LE.le 1 (f ↑b)
      ⊢ False
    -/
    rw [Nat.cast_mul, map_mul] at hp1
    /-
      case isUnit_or_isUnit'.intro.intro
      f : AbsoluteValue Rat Real
      a b : Nat
      hp0 : LT.lt 0 (f ↑(HMul.hMul a b))
      hp1 : LT.lt (HMul.hMul (f ↑a) (f ↑b)) 1
      hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le (HMul.hMul a …
      ha₁ : Ne a 1
      hb₁ : Ne b 1
      ha₀ : Ne a 0
      hb₀ : Ne b 0
      hap : LT.lt a (HMul.hMul a b)
      hbp : LT.lt b (HMul.hMul a b)
      ha : LE.le 1 (f ↑a)
      hb : LE.le 1 (f ↑b)
      ⊢ False
    -/
    exact ((one_le_mul_of_one_le_of_one_le ha hb).trans_lt hp1).false
    /-
      🎉 no goals
    -/

-- ## Step 3: if p does not divide m, then f m = 1


include hp0 hp1 hmin bdd in
/-- A natural number not divible by `p` has absolute value 1. -/
lemma eq_one_of_not_dvd {m : ℕ} (hpm : ¬ p ∣ m) : f m = 1 := by
  /-
    f : AbsoluteValue Rat Real
    bdd : ∀ (n : Nat), LE.le (f ↑n) 1
    p : Nat
    hp0 : LT.lt 0 (f ↑p)
    hp1 : LT.lt (f ↑p) 1
    hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
    m : Nat
    hpm : Not (Dvd.dvd p m)
    ⊢ Eq (f ↑m) 1
  -/
  apply le_antisymm (bdd m)
  /-
    f : AbsoluteValue Rat Real
    bdd : ∀ (n : Nat), LE.le (f ↑n) 1
    p : Nat
    hp0 : LT.lt 0 (f ↑p)
    hp1 : LT.lt (f ↑p) 1
    hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
    m : Nat
    hpm : Not (Dvd.dvd p m)
    ⊢ LE.le 1 (f ↑m)
  -/
  by_contra! hm
  /-
    f : AbsoluteValue Rat Real
    bdd : ∀ (n : Nat), LE.le (f ↑n) 1
    p : Nat
    hp0 : LT.lt 0 (f ↑p)
    hp1 : LT.lt (f ↑p) 1
    hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
    m : Nat
    hpm : Not (Dvd.dvd p m)
    hm : LT.lt (f ↑m) 1
    ⊢ False
  -/
  set M := f p ⊔ f m with hM
  /-
    f : AbsoluteValue Rat Real
    bdd : ∀ (n : Nat), LE.le (f ↑n) 1
    p : Nat
    hp0 : LT.lt 0 (f ↑p)
    hp1 : LT.lt (f ↑p) 1
    hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
    m : Nat
    hpm : Not (Dvd.dvd p m)
    hm : LT.lt (f ↑m) 1
    M : Real := Max.max (f ↑p) (f ↑m)
    hM : Eq M (Max.max (f ↑p) (f ↑m))
    ⊢ False
  -/
  set k := Nat.ceil (M.logb (1 / 2)) + 1 with hk
  obtain ⟨a, b, bezout⟩ : IsCoprime (p ^ k : ℤ) (m ^ k) :=
    is_prime_of_minimal_nat_zero_lt_and_lt_one hp0 hp1 hmin
      |>.coprime_iff_not_dvd |>.mpr hpm |>.isCoprime |>.pow
  have le_half {x} (hx0 : 0 < x) (hx1 : x < 1) (hxM : x ≤ M) : x ^ k < 1 / 2 := by
    calc
    x ^ k = x ^ (k : ℝ) := (rpow_natCast x k).symm
    _ < x ^ M.logb (1 / 2) := by
      apply rpow_lt_rpow_of_exponent_gt hx0 hx1
      rw [hk]
      push_cast
      exact lt_add_of_le_of_pos (Nat.le_ceil _) zero_lt_one
    _ ≤ x ^ x.logb (1 / 2) := by
      apply rpow_le_rpow_of_exponent_ge hx0 hx1.le
      simp only [one_div, ← log_div_log, log_inv, neg_div, ← div_neg, hM]
      gcongr
      simp only [Left.neg_pos_iff]
      exact log_neg (lt_sup_iff.mpr <| .inl hp0) (sup_lt_iff.mpr ⟨hp1, hm⟩)
    _ = 1 / 2 := rpow_logb hx0 hx1.ne one_half_pos
  /-
    case intro.intro
    f : AbsoluteValue Rat Real
    bdd : ∀ (n : Nat), LE.le (f ↑n) 1
    p : Nat
    hp0 : LT.lt 0 (f ↑p)
    hp1 : LT.lt (f ↑p) 1
    hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
    m : Nat
    hpm : Not (Dvd.dvd p m)
    hm : LT.lt (f ↑m) 1
    M : Real := Max.max (f ↑p) (f ↑m)
    hM : Eq M (Max.max (f ↑p) (f ↑m))
    k : Nat := HAdd.hAdd (Nat.ceil (Real.logb M (1 / 2))) 1
    hk : Eq k (HAdd.hAdd (Nat.ceil (Real.logb M (1 / 2))) 1)
    a b : Int
    bezout : Eq (HAdd.hAdd (HMul.hMul a (HPow.hPow (↑p) k)) (HMul.hMul b (HPow.hPo …
    le_half : ∀ {x : Real}, LT.lt 0 x → LT.lt x 1 → LE.le x M → LT.lt (HPow.hPow x …
    ⊢ False
  -/
  apply lt_irrefl (1 : ℝ)
  calc
  1 = f 1 := (map_one f).symm
  _ = f (a * p ^ k + b * m ^ k) := by rw_mod_cast [bezout]; norm_cast
  _ ≤ f (a * p ^ k) + f (b * m ^ k) := f.add_le' ..
  _ ≤ 1 * (f p) ^ k + 1 * (f m) ^ k := by
    simp only [map_mul, map_pow]
    gcongr
    all_goals rw [← apply_natAbs_eq]; apply bdd
  _ = (f p) ^ k + (f m) ^ k := by simp only [one_mul]
  _ < 1 := by
    have hm₀ : 0 < f m := f.pos <| Nat.cast_ne_zero.mpr fun H ↦ hpm <| H ▸ dvd_zero p
    linarith only [le_half hp0 hp1 le_sup_left, le_half hm₀ hm le_sup_right]

-- ## Step 4: f p = p ^ (-t) for some positive real t


include hp0 hp1 hmin in
/-- The absolute value of `p` is `p ^ (-t)` for some positive real number `t`. -/
lemma exists_pos_eq_pow_neg : ∃ t : ℝ, 0 < t ∧ f p = p ^ (-t) := by
  /-
    f : AbsoluteValue Rat Real
    p : Nat
    hp0 : LT.lt 0 (f ↑p)
    hp1 : LT.lt (f ↑p) 1
    hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
    ⊢ Exists fun t => And (LT.lt 0 t) (Eq (f ↑p) (HPow.hPow (↑p) (Neg.neg t)))
  -/
  have pprime := is_prime_of_minimal_nat_zero_lt_and_lt_one hp0 hp1 hmin
  /-
    f : AbsoluteValue Rat Real
    p : Nat
    hp0 : LT.lt 0 (f ↑p)
    hp1 : LT.lt (f ↑p) 1
    hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
    pprime : Nat.Prime p
    ⊢ Exists fun t => And (LT.lt 0 t) (Eq (f ↑p) (HPow.hPow (↑p) (Neg.neg t)))
  -/
  refine ⟨- logb p (f p), Left.neg_pos_iff.mpr <| logb_neg (mod_cast pprime.one_lt) hp0 hp1, ?_⟩
  /-
    f : AbsoluteValue Rat Real
    p : Nat
    hp0 : LT.lt 0 (f ↑p)
    hp1 : LT.lt (f ↑p) 1
    hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
    pprime : Nat.Prime p
    ⊢ Eq (f ↑p) (HPow.hPow (↑p) (Neg.neg (Neg.neg (Real.logb (↑p) (f ↑p)))))
  -/
  rw [neg_neg]
  /-
    f : AbsoluteValue Rat Real
    p : Nat
    hp0 : LT.lt 0 (f ↑p)
    hp1 : LT.lt (f ↑p) 1
    hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
    pprime : Nat.Prime p
    ⊢ Eq (f ↑p) (HPow.hPow (↑p) (Real.logb (↑p) (f ↑p)))
  -/
  exact (rpow_logb (mod_cast pprime.pos) (mod_cast pprime.ne_one) hp0).symm
  /-
    🎉 no goals
  -/

-- ## Non-archimedean case: end goal


include hf_nontriv bdd in
/-- If `f` is bounded and not trivial, then it is equivalent to a p-adic absolute value. -/
theorem equiv_padic_of_bounded :
    ∃! p, ∃ (_ : Fact p.Prime), f ≈ (padic p) := by
  /-
    f : AbsoluteValue Rat Real
    hf_nontriv : Ne f AbsoluteValue.trivial
    bdd : ∀ (n : Nat), LE.le (f ↑n) 1
    ⊢ ExistsUnique fun p => Exists fun x => HasEquiv.Equiv f (Rat.AbsoluteValue.pa …
  -/
  obtain ⟨p, hfp, hmin⟩ := exists_minimal_nat_zero_lt_and_lt_one hf_nontriv bdd
  /-
    case intro.intro
    f : AbsoluteValue Rat Real
    hf_nontriv : Ne f AbsoluteValue.trivial
    bdd : ∀ (n : Nat), LE.le (f ↑n) 1
    p : Nat
    hfp : And (LT.lt 0 (f ↑p)) (LT.lt (f ↑p) 1)
    hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
    ⊢ ExistsUnique fun p => Exists fun x => HasEquiv.Equiv f (Rat.AbsoluteValue.pa …
  -/
  have hprime := is_prime_of_minimal_nat_zero_lt_and_lt_one hfp.1 hfp.2 hmin
  /-
    case intro.intro
    f : AbsoluteValue Rat Real
    hf_nontriv : Ne f AbsoluteValue.trivial
    bdd : ∀ (n : Nat), LE.le (f ↑n) 1
    p : Nat
    hfp : And (LT.lt 0 (f ↑p)) (LT.lt (f ↑p) 1)
    hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
    hprime : Nat.Prime p
    ⊢ ExistsUnique fun p => Exists fun x => HasEquiv.Equiv f (Rat.AbsoluteValue.pa …
  -/
  have hprime_fact : Fact p.Prime := ⟨hprime⟩
  /-
    case intro.intro
    f : AbsoluteValue Rat Real
    hf_nontriv : Ne f AbsoluteValue.trivial
    bdd : ∀ (n : Nat), LE.le (f ↑n) 1
    p : Nat
    hfp : And (LT.lt 0 (f ↑p)) (LT.lt (f ↑p) 1)
    hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
    hprime : Nat.Prime p
    hprime_fact : Fact (Nat.Prime p)
    ⊢ ExistsUnique fun p => Exists fun x => HasEquiv.Equiv f (Rat.AbsoluteValue.pa …
  -/
  obtain ⟨t, h⟩ := exists_pos_eq_pow_neg hfp.1 hfp.2 hmin
  /-
    case intro.intro.intro
    f : AbsoluteValue Rat Real
    hf_nontriv : Ne f AbsoluteValue.trivial
    bdd : ∀ (n : Nat), LE.le (f ↑n) 1
    p : Nat
    hfp : And (LT.lt 0 (f ↑p)) (LT.lt (f ↑p) 1)
    hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
    hprime : Nat.Prime p
    hprime_fact : Fact (Nat.Prime p)
    t : Real
    h : And (LT.lt 0 t) (Eq (f ↑p) (HPow.hPow (↑p) (Neg.neg t)))
    ⊢ ExistsUnique fun p => Exists fun x => HasEquiv.Equiv f (Rat.AbsoluteValue.pa …
  -/
  simp_rw [← equiv_on_nat_iff_equiv]
  /-
    case intro.intro.intro
    f : AbsoluteValue Rat Real
    hf_nontriv : Ne f AbsoluteValue.trivial
    bdd : ∀ (n : Nat), LE.le (f ↑n) 1
    p : Nat
    hfp : And (LT.lt 0 (f ↑p)) (LT.lt (f ↑p) 1)
    hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
    hprime : Nat.Prime p
    hprime_fact : Fact (Nat.Prime p)
    t : Real
    h : And (LT.lt 0 t) (Eq (f ↑p) (HPow.hPow (↑p) (Neg.neg t)))
    ⊢ ExistsUnique fun p => Exists fun h => Exists fun c => And (LT.lt 0 c) (∀ (n  …
  -/
  refine ⟨p, ⟨hprime_fact, t⁻¹, inv_pos_of_pos h.1, fun n ↦ ?_⟩, fun q ⟨hq_prime, h_equiv⟩ ↦ ?_⟩
    /-
      case intro.intro.intro.refine_1
      f : AbsoluteValue Rat Real
      hf_nontriv : Ne f AbsoluteValue.trivial
      bdd : ∀ (n : Nat), LE.le (f ↑n) 1
      p : Nat
      hfp : And (LT.lt 0 (f ↑p)) (LT.lt (f ↑p) 1)
      hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
      hprime : Nat.Prime p
      hprime_fact : Fact (Nat.Prime p)
      t : Real
      h : And (LT.lt 0 t) (Eq (f ↑p) (HPow.hPow (↑p) (Neg.neg t)))
      n : Nat
      ⊢ Eq (HPow.hPow (f ↑n) (Inv.inv t)) ((Rat.AbsoluteValue.padic p) ↑n)
    -/
  · have ht : t⁻¹ ≠ 0 := inv_ne_zero h.1.ne'
    /-
      case intro.intro.intro.refine_1
      f : AbsoluteValue Rat Real
      hf_nontriv : Ne f AbsoluteValue.trivial
      bdd : ∀ (n : Nat), LE.le (f ↑n) 1
      p : Nat
      hfp : And (LT.lt 0 (f ↑p)) (LT.lt (f ↑p) 1)
      hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
      hprime : Nat.Prime p
      hprime_fact : Fact (Nat.Prime p)
      t : Real
      h : And (LT.lt 0 t) (Eq (f ↑p) (HPow.hPow (↑p) (Neg.neg t)))
      n : Nat
      ht : Ne (Inv.inv t) 0
      ⊢ Eq (HPow.hPow (f ↑n) (Inv.inv t)) ((Rat.AbsoluteValue.padic p) ↑n)
    -/
    rcases eq_or_ne n 0 with rfl | hn -- Separate cases n = 0 and n ≠ 0
      /-
        case intro.intro.intro.refine_1.inl
        f : AbsoluteValue Rat Real
        hf_nontriv : Ne f AbsoluteValue.trivial
        bdd : ∀ (n : Nat), LE.le (f ↑n) 1
        p : Nat
        hfp : And (LT.lt 0 (f ↑p)) (LT.lt (f ↑p) 1)
        hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
        hprime : Nat.Prime p
        hprime_fact : Fact (Nat.Prime p)
        t : Real
        h : And (LT.lt 0 t) (Eq (f ↑p) (HPow.hPow (↑p) (Neg.neg t)))
        ht : Ne (Inv.inv t) 0
        ⊢ Eq (HPow.hPow (f ↑0) (Inv.inv t)) ((Rat.AbsoluteValue.padic p) ↑0)
      -/
    · simp [ht]
      /-
        🎉 no goals
      -/
    · /- Any natural number can be written as a power of p times a natural number not divisible
      by p  -/
      /-
        case intro.intro.intro.refine_1.inr
        f : AbsoluteValue Rat Real
        hf_nontriv : Ne f AbsoluteValue.trivial
        bdd : ∀ (n : Nat), LE.le (f ↑n) 1
        p : Nat
        hfp : And (LT.lt 0 (f ↑p)) (LT.lt (f ↑p) 1)
        hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
        hprime : Nat.Prime p
        hprime_fact : Fact (Nat.Prime p)
        t : Real
        h : And (LT.lt 0 t) (Eq (f ↑p) (HPow.hPow (↑p) (Neg.neg t)))
        n : Nat
        ht : Ne (Inv.inv t) 0
        hn : Ne n 0
        ⊢ Eq (HPow.hPow (f ↑n) (Inv.inv t)) ((Rat.AbsoluteValue.padic p) ↑n)
      -/
      rcases Nat.exists_eq_pow_mul_and_not_dvd hn p hprime.ne_one with ⟨e, m, hpm, rfl⟩
      simp only [Nat.cast_mul, Nat.cast_pow, map_mul, map_pow, h.2,
        eq_one_of_not_dvd bdd hfp.1 hfp.2 hmin hpm, padic_eq_padicNorm,
        padicNorm.padicNorm_p_of_prime, cast_inv, cast_natCast, inv_pow]
      /-
        case intro.intro.intro.refine_1.inr.intro.intro.intro
        f : AbsoluteValue Rat Real
        hf_nontriv : Ne f AbsoluteValue.trivial
        bdd : ∀ (n : Nat), LE.le (f ↑n) 1
        p : Nat
        hfp : And (LT.lt 0 (f ↑p)) (LT.lt (f ↑p) 1)
        hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
        hprime : Nat.Prime p
        hprime_fact : Fact (Nat.Prime p)
        t : Real
        h : And (LT.lt 0 t) (Eq (f ↑p) (HPow.hPow (↑p) (Neg.neg t)))
        ht : Ne (Inv.inv t) 0
        e m : Nat
        hpm : Not (Dvd.dvd p m)
        hn : Ne (HMul.hMul (HPow.hPow p e) m) 0
        ⊢ Eq (HPow.hPow (HMul.hMul (HPow.hPow (HPow.hPow (↑p) (Neg.neg t)) e) 1) (Inv. …
      -/
      rw [← padicNorm.nat_eq_one_iff] at hpm
      simp only [← rpow_natCast, p.cast_nonneg, ← rpow_mul, neg_mul, mul_one, ← rpow_neg, hpm,
        cast_one]
      /-
        case intro.intro.intro.refine_1.inr.intro.intro.intro
        f : AbsoluteValue Rat Real
        hf_nontriv : Ne f AbsoluteValue.trivial
        bdd : ∀ (n : Nat), LE.le (f ↑n) 1
        p : Nat
        hfp : And (LT.lt 0 (f ↑p)) (LT.lt (f ↑p) 1)
        hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
        hprime : Nat.Prime p
        hprime_fact : Fact (Nat.Prime p)
        t : Real
        h : And (LT.lt 0 t) (Eq (f ↑p) (HPow.hPow (↑p) (Neg.neg t)))
        ht : Ne (Inv.inv t) 0
        e m : Nat
        hpm : Eq (padicNorm p ↑m) 1
        hn : Ne (HMul.hMul (HPow.hPow p e) m) 0
        ⊢ Eq (HPow.hPow (↑p) (Neg.neg (HMul.hMul (HMul.hMul t ↑e) (Inv.inv t)))) (HPow …
      -/
      congr
      /-
        case intro.intro.intro.refine_1.inr.intro.intro.intro.e_a.e_a
        f : AbsoluteValue Rat Real
        hf_nontriv : Ne f AbsoluteValue.trivial
        bdd : ∀ (n : Nat), LE.le (f ↑n) 1
        p : Nat
        hfp : And (LT.lt 0 (f ↑p)) (LT.lt (f ↑p) 1)
        hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
        hprime : Nat.Prime p
        hprime_fact : Fact (Nat.Prime p)
        t : Real
        h : And (LT.lt 0 t) (Eq (f ↑p) (HPow.hPow (↑p) (Neg.neg t)))
        ht : Ne (Inv.inv t) 0
        e m : Nat
        hpm : Eq (padicNorm p ↑m) 1
        hn : Ne (HMul.hMul (HPow.hPow p e) m) 0
        ⊢ Eq (HMul.hMul (HMul.hMul t ↑e) (Inv.inv t)) ↑e
      -/
      field_simp [h.1.ne']
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.intro.refine_2
      f : AbsoluteValue Rat Real
      hf_nontriv : Ne f AbsoluteValue.trivial
      bdd : ∀ (n : Nat), LE.le (f ↑n) 1
      p : Nat
      hfp : And (LT.lt 0 (f ↑p)) (LT.lt (f ↑p) 1)
      hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
      hprime : Nat.Prime p
      hprime_fact : Fact (Nat.Prime p)
      t : Real
      h : And (LT.lt 0 t) (Eq (f ↑p) (HPow.hPow (↑p) (Neg.neg t)))
      q : Nat
      x✝ : (fun p => Exists fun h => Exists fun c => And (LT.lt 0 c) (∀ (n : Nat), E …
      hq_prime : Fact (Nat.Prime q)
      h_equiv : Exists fun c => And (LT.lt 0 c) (∀ (n : Nat), Eq (HPow.hPow (f ↑n) c …
      ⊢ Eq q p
    -/
  · by_contra! hne
    /-
      case intro.intro.intro.refine_2
      f : AbsoluteValue Rat Real
      hf_nontriv : Ne f AbsoluteValue.trivial
      bdd : ∀ (n : Nat), LE.le (f ↑n) 1
      p : Nat
      hfp : And (LT.lt 0 (f ↑p)) (LT.lt (f ↑p) 1)
      hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
      hprime : Nat.Prime p
      hprime_fact : Fact (Nat.Prime p)
      t : Real
      h : And (LT.lt 0 t) (Eq (f ↑p) (HPow.hPow (↑p) (Neg.neg t)))
      q : Nat
      x✝ : (fun p => Exists fun h => Exists fun c => And (LT.lt 0 c) (∀ (n : Nat), E …
      hq_prime : Fact (Nat.Prime q)
      h_equiv : Exists fun c => And (LT.lt 0 c) (∀ (n : Nat), Eq (HPow.hPow (f ↑n) c …
      hne : Ne q p
      ⊢ False
    -/
    apply hq_prime.elim.prime.ne_one
    /-
      case intro.intro.intro.refine_2
      f : AbsoluteValue Rat Real
      hf_nontriv : Ne f AbsoluteValue.trivial
      bdd : ∀ (n : Nat), LE.le (f ↑n) 1
      p : Nat
      hfp : And (LT.lt 0 (f ↑p)) (LT.lt (f ↑p) 1)
      hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
      hprime : Nat.Prime p
      hprime_fact : Fact (Nat.Prime p)
      t : Real
      h : And (LT.lt 0 t) (Eq (f ↑p) (HPow.hPow (↑p) (Neg.neg t)))
      q : Nat
      x✝ : (fun p => Exists fun h => Exists fun c => And (LT.lt 0 c) (∀ (n : Nat), E …
      hq_prime : Fact (Nat.Prime q)
      h_equiv : Exists fun c => And (LT.lt 0 c) (∀ (n : Nat), Eq (HPow.hPow (f ↑n) c …
      hne : Ne q p
      ⊢ Eq q 1
    -/
    rw [ne_comm, ← Nat.coprime_primes hprime hq_prime.elim, hprime.coprime_iff_not_dvd] at hne
    /-
      case intro.intro.intro.refine_2
      f : AbsoluteValue Rat Real
      hf_nontriv : Ne f AbsoluteValue.trivial
      bdd : ∀ (n : Nat), LE.le (f ↑n) 1
      p : Nat
      hfp : And (LT.lt 0 (f ↑p)) (LT.lt (f ↑p) 1)
      hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
      hprime : Nat.Prime p
      hprime_fact : Fact (Nat.Prime p)
      t : Real
      h : And (LT.lt 0 t) (Eq (f ↑p) (HPow.hPow (↑p) (Neg.neg t)))
      q : Nat
      x✝ : (fun p => Exists fun h => Exists fun c => And (LT.lt 0 c) (∀ (n : Nat), E …
      hq_prime : Fact (Nat.Prime q)
      h_equiv : Exists fun c => And (LT.lt 0 c) (∀ (n : Nat), Eq (HPow.hPow (f ↑n) c …
      hne : Not (Dvd.dvd p q)
      ⊢ Eq q 1
    -/
    rcases h_equiv with ⟨c, _, h_eq⟩
    /-
      case intro.intro.intro.refine_2.intro.intro
      f : AbsoluteValue Rat Real
      hf_nontriv : Ne f AbsoluteValue.trivial
      bdd : ∀ (n : Nat), LE.le (f ↑n) 1
      p : Nat
      hfp : And (LT.lt 0 (f ↑p)) (LT.lt (f ↑p) 1)
      hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
      hprime : Nat.Prime p
      hprime_fact : Fact (Nat.Prime p)
      t : Real
      h : And (LT.lt 0 t) (Eq (f ↑p) (HPow.hPow (↑p) (Neg.neg t)))
      q : Nat
      x✝ : (fun p => Exists fun h => Exists fun c => And (LT.lt 0 c) (∀ (n : Nat), E …
      hq_prime : Fact (Nat.Prime q)
      hne : Not (Dvd.dvd p q)
      c : Real
      left✝ : LT.lt 0 c
      h_eq : ∀ (n : Nat), Eq (HPow.hPow (f ↑n) c) ((Rat.AbsoluteValue.padic q) ↑n)
      ⊢ Eq q 1
    -/
    have h_eq' := h_eq q
    simp only [eq_one_of_not_dvd bdd hfp.1 hfp.2 hmin hne, one_rpow, padic_eq_padicNorm,
      padicNorm.padicNorm_p_of_prime, cast_inv, cast_natCast, eq_comm, inv_eq_one] at h_eq'
    /-
      case intro.intro.intro.refine_2.intro.intro
      f : AbsoluteValue Rat Real
      hf_nontriv : Ne f AbsoluteValue.trivial
      bdd : ∀ (n : Nat), LE.le (f ↑n) 1
      p : Nat
      hfp : And (LT.lt 0 (f ↑p)) (LT.lt (f ↑p) 1)
      hmin : ∀ (m : Nat), And (LT.lt 0 (f ↑m)) (LT.lt (f ↑m) 1) → LE.le p m
      hprime : Nat.Prime p
      hprime_fact : Fact (Nat.Prime p)
      t : Real
      h : And (LT.lt 0 t) (Eq (f ↑p) (HPow.hPow (↑p) (Neg.neg t)))
      q : Nat
      x✝ : (fun p => Exists fun h => Exists fun c => And (LT.lt 0 c) (∀ (n : Nat), E …
      hq_prime : Fact (Nat.Prime q)
      hne : Not (Dvd.dvd p q)
      c : Real
      left✝ : LT.lt 0 c
      h_eq : ∀ (n : Nat), Eq (HPow.hPow (f ↑n) c) ((Rat.AbsoluteValue.padic q) ↑n)
      h_eq' : Eq (↑q) 1
      ⊢ Eq q 1
    -/
    exact_mod_cast h_eq'
    /-
      🎉 no goals
    -/


/-- The standard absolute value on `ℚ`. We name it `real` because it corresponds to the
unique real place of `ℚ`. -/
def real : AbsoluteValue ℚ ℝ where
  toFun x := |x|
                     /-
                       f g : AbsoluteValue Rat Real
                       x y : Rat
                       ⊢ Eq ((fun x => abs ↑x) (HMul.hMul x y)) (HMul.hMul ((fun x => abs ↑x) x) ((fu …
                     -/
  map_mul' x y := by simpa using abs_mul (x : ℝ) (y : ℝ)
                     /-
                       🎉 no goals
                     -/
                  /-
                    f g : AbsoluteValue Rat Real
                    x : Rat
                    ⊢ LE.le 0 ({ toFun := fun x => abs ↑x, map_mul' := ⋯ }.toFun x)
                  -/
  nonneg' x := by simp
                  /-
                    🎉 no goals
                  -/
                   /-
                     f g : AbsoluteValue Rat Real
                     x : Rat
                     ⊢ Iff (Eq ({ toFun := fun x => abs ↑x, map_mul' := ⋯ }.toFun x) 0) (Eq x 0)
                   -/
  eq_zero' x := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      f g : AbsoluteValue Rat Real
                      x y : Rat
                      ⊢ LE.le ({ toFun := fun x => abs ↑x, map_mul' := ⋯ }.toFun (HAdd.hAdd x y)) (H …
                    -/
  add_le' x y := by simpa using abs_add_le (x : ℝ) (y : ℝ)
                    /-
                      🎉 no goals
                    -/


@[simp] lemma real_eq_abs (r : ℚ) : real r = |r| :=
  (cast_abs r).symm

-- ## Preliminary result


/-- Given any two integers `n`, `m` with `m > 1`, the absolute value of `n` is bounded by
`m + m * f m + m * (f m) ^ 2 + ... + m * (f m) ^ d` where `d` is the number of digits of the
expansion of `n` in base `m`. -/
lemma apply_le_sum_digits (n : ℕ) {m : ℕ} (hm : 1 < m) :
    f n ≤ ((Nat.digits m n).mapIdx fun i _ ↦ m * (f m) ^ i).sum := by
  /-
    f : AbsoluteValue Rat Real
    n m : Nat
    hm : LT.lt 1 m
    ⊢ LE.le (f ↑n) (List.mapIdx (fun i x => HMul.hMul (↑m) (HPow.hPow (f ↑m) i)) ( …
  -/
  set L := Nat.digits m n
  /-
    f : AbsoluteValue Rat Real
    n m : Nat
    hm : LT.lt 1 m
    L : List Nat := m.digits n
    ⊢ LE.le (f ↑n) (List.mapIdx (fun i x => HMul.hMul (↑m) (HPow.hPow (f ↑m) i)) L …
  -/
  set L' : List ℚ := List.map Nat.cast (L.mapIdx fun i a ↦ (a * m ^ i)) with hL'
  -- If `c` is a digit in the expansion of `n` in base `m`, then `f c` is less than `m`.
  have hcoef {c : ℕ} (hc : c ∈ Nat.digits m n) : f c < m :=
    lt_of_le_of_lt (f.apply_nat_le_self c) (mod_cast Nat.digits_lt_base hm hc)
  calc
  f n = f ((Nat.ofDigits m L : ℕ) : ℚ) := by rw [Nat.ofDigits_digits m n]
    _ = f L'.sum := by rw [Nat.ofDigits_eq_sum_mapIdx]; norm_cast
    _ ≤ (L'.map f).sum := listSum_le f L'
    _ ≤ (L.mapIdx fun i _ ↦ m * (f m) ^ i).sum := ?_
  /-
    f : AbsoluteValue Rat Real
    n m : Nat
    hm : LT.lt 1 m
    L : List Nat := m.digits n
    L' : List Rat := List.map Nat.cast (List.mapIdx (fun i a => HMul.hMul a (HPow. …
    hL' : Eq L' (List.map Nat.cast (List.mapIdx (fun i a => HMul.hMul a (HPow.hPow …
    hcoef : ∀ {c : Nat}, Membership.mem (m.digits n) c → LT.lt (f ↑c) ↑m
    ⊢ LE.le (List.map (⇑f) L').sum (List.mapIdx (fun i x => HMul.hMul (↑m) (HPow.h …
  -/
  simp only [hL', List.mapIdx_eq_enum_map, List.map_map]
  /-
    f : AbsoluteValue Rat Real
    n m : Nat
    hm : LT.lt 1 m
    L : List Nat := m.digits n
    L' : List Rat := List.map Nat.cast (List.mapIdx (fun i a => HMul.hMul a (HPow. …
    hL' : Eq L' (List.map Nat.cast (List.mapIdx (fun i a => HMul.hMul a (HPow.hPow …
    hcoef : ∀ {c : Nat}, Membership.mem (m.digits n) c → LT.lt (f ↑c) ↑m
    ⊢ LE.le (List.map (Function.comp (⇑f) (Function.comp Nat.cast (Function.uncurr …
  -/
  refine List.sum_le_sum fun ⟨i, a⟩ hia ↦ ?_
  /-
    f : AbsoluteValue Rat Real
    n m : Nat
    hm : LT.lt 1 m
    L : List Nat := m.digits n
    L' : List Rat := List.map Nat.cast (List.mapIdx (fun i a => HMul.hMul a (HPow. …
    hL' : Eq L' (List.map Nat.cast (List.mapIdx (fun i a => HMul.hMul a (HPow.hPow …
    hcoef : ∀ {c : Nat}, Membership.mem (m.digits n) c → LT.lt (f ↑c) ↑m
    x✝ : Prod Nat Nat
    i a : Nat
    hia : Membership.mem L.enum { fst := i, snd := a }
    ⊢ LE.le (Function.comp (⇑f) (Function.comp Nat.cast (Function.uncurry fun i a  …
  -/
  dsimp only [Function.comp_apply, Function.uncurry_apply_pair]
  /-
    f : AbsoluteValue Rat Real
    n m : Nat
    hm : LT.lt 1 m
    L : List Nat := m.digits n
    L' : List Rat := List.map Nat.cast (List.mapIdx (fun i a => HMul.hMul a (HPow. …
    hL' : Eq L' (List.map Nat.cast (List.mapIdx (fun i a => HMul.hMul a (HPow.hPow …
    hcoef : ∀ {c : Nat}, Membership.mem (m.digits n) c → LT.lt (f ↑c) ↑m
    x✝ : Prod Nat Nat
    i a : Nat
    hia : Membership.mem L.enum { fst := i, snd := a }
    ⊢ LE.le (f ↑(HMul.hMul a (HPow.hPow m i))) (HMul.hMul (↑m) (HPow.hPow (f ↑m) i))
  -/
  replace hia := List.mem_enumFrom hia
  /-
    f : AbsoluteValue Rat Real
    n m : Nat
    hm : LT.lt 1 m
    L : List Nat := m.digits n
    L' : List Rat := List.map Nat.cast (List.mapIdx (fun i a => HMul.hMul a (HPow. …
    hL' : Eq L' (List.map Nat.cast (List.mapIdx (fun i a => HMul.hMul a (HPow.hPow …
    hcoef : ∀ {c : Nat}, Membership.mem (m.digits n) c → LT.lt (f ↑c) ↑m
    x✝ : Prod Nat Nat
    i a : Nat
    hia✝ : Membership.mem L.enum { fst := i, snd := a }
    hia : And (LE.le 0 i) (And (LT.lt i (HAdd.hAdd 0 L.length)) (Eq a (GetElem.get …
    ⊢ LE.le (f ↑(HMul.hMul a (HPow.hPow m i))) (HMul.hMul (↑m) (HPow.hPow (f ↑m) i))
  -/
  push_cast
  /-
    f : AbsoluteValue Rat Real
    n m : Nat
    hm : LT.lt 1 m
    L : List Nat := m.digits n
    L' : List Rat := List.map Nat.cast (List.mapIdx (fun i a => HMul.hMul a (HPow. …
    hL' : Eq L' (List.map Nat.cast (List.mapIdx (fun i a => HMul.hMul a (HPow.hPow …
    hcoef : ∀ {c : Nat}, Membership.mem (m.digits n) c → LT.lt (f ↑c) ↑m
    x✝ : Prod Nat Nat
    i a : Nat
    hia✝ : Membership.mem L.enum { fst := i, snd := a }
    hia : And (LE.le 0 i) (And (LT.lt i (HAdd.hAdd 0 L.length)) (Eq a (GetElem.get …
    ⊢ LE.le (f (HMul.hMul (↑a) (HPow.hPow (↑m) i))) (HMul.hMul (↑m) (HPow.hPow (f  …
  -/
  rw [map_mul, map_pow]
  /-
    f : AbsoluteValue Rat Real
    n m : Nat
    hm : LT.lt 1 m
    L : List Nat := m.digits n
    L' : List Rat := List.map Nat.cast (List.mapIdx (fun i a => HMul.hMul a (HPow. …
    hL' : Eq L' (List.map Nat.cast (List.mapIdx (fun i a => HMul.hMul a (HPow.hPow …
    hcoef : ∀ {c : Nat}, Membership.mem (m.digits n) c → LT.lt (f ↑c) ↑m
    x✝ : Prod Nat Nat
    i a : Nat
    hia✝ : Membership.mem L.enum { fst := i, snd := a }
    hia : And (LE.le 0 i) (And (LT.lt i (HAdd.hAdd 0 L.length)) (Eq a (GetElem.get …
    ⊢ LE.le (HMul.hMul (f ↑a) (HPow.hPow (f ↑m) i)) (HMul.hMul (↑m) (HPow.hPow (f  …
  -/
  refine mul_le_mul_of_nonneg_right ?_ <| pow_nonneg (f.nonneg _) i
  /-
    f : AbsoluteValue Rat Real
    n m : Nat
    hm : LT.lt 1 m
    L : List Nat := m.digits n
    L' : List Rat := List.map Nat.cast (List.mapIdx (fun i a => HMul.hMul a (HPow. …
    hL' : Eq L' (List.map Nat.cast (List.mapIdx (fun i a => HMul.hMul a (HPow.hPow …
    hcoef : ∀ {c : Nat}, Membership.mem (m.digits n) c → LT.lt (f ↑c) ↑m
    x✝ : Prod Nat Nat
    i a : Nat
    hia✝ : Membership.mem L.enum { fst := i, snd := a }
    hia : And (LE.le 0 i) (And (LT.lt i (HAdd.hAdd 0 L.length)) (Eq a (GetElem.get …
    ⊢ LE.le (f ↑a) ↑m
  -/
  simp only [zero_le, zero_add, tsub_zero, true_and] at hia
  /-
    f : AbsoluteValue Rat Real
    n m : Nat
    hm : LT.lt 1 m
    L : List Nat := m.digits n
    L' : List Rat := List.map Nat.cast (List.mapIdx (fun i a => HMul.hMul a (HPow. …
    hL' : Eq L' (List.map Nat.cast (List.mapIdx (fun i a => HMul.hMul a (HPow.hPow …
    hcoef : ∀ {c : Nat}, Membership.mem (m.digits n) c → LT.lt (f ↑c) ↑m
    x✝ : Prod Nat Nat
    i a : Nat
    hia✝ : Membership.mem L.enum { fst := i, snd := a }
    hia : And (LT.lt i L.length) (Eq a (GetElem.getElem L i ⋯))
    ⊢ LE.le (f ↑a) ↑m
  -/
  exact (hcoef (List.mem_iff_get.mpr ⟨⟨i, hia.1⟩, hia.2.symm⟩)).le
  /-
    🎉 no goals
  -/

-- ## Step 1: if f is an AbsoluteValue and f n > 1 for some natural n, then f n > 1 for all n ≥ 2


/-- If `f n > 1` for some `n` then `f n > 1` for all `n ≥ 2` -/
lemma one_lt_of_not_bounded (notbdd : ¬ ∀ n : ℕ, f n ≤ 1) {n₀ : ℕ} (hn₀ : 1 < n₀) : 1 < f n₀ := by
  /-
    f : AbsoluteValue Rat Real
    notbdd : Not (∀ (n : Nat), LE.le (f ↑n) 1)
    n₀ : Nat
    hn₀ : LT.lt 1 n₀
    ⊢ LT.lt 1 (f ↑n₀)
  -/
  contrapose! notbdd with h
  /-
    f : AbsoluteValue Rat Real
    n₀ : Nat
    hn₀ : LT.lt 1 n₀
    h : LE.le (f ↑n₀) 1
    ⊢ ∀ (n : Nat), LE.le (f ↑n) 1
  -/
  intro n
  have h_ineq1 {m : ℕ} (hm : 1 ≤ m) : f m ≤ n₀ * (logb n₀ m + 1) := by
    /- L is the string of digits of `n` in the base `n₀`-/
    set L := Nat.digits n₀ m
    calc
    f m ≤ (L.mapIdx fun i _ ↦ n₀ * f n₀ ^ i).sum := apply_le_sum_digits m hn₀
    _ ≤ (L.mapIdx fun _ _ ↦ (n₀ : ℝ)).sum := by
      simp only [List.mapIdx_eq_enum_map, List.map_map]
      refine List.sum_le_sum fun ⟨i, a⟩ _ ↦ ?_
      simp only [Function.comp_apply, Function.uncurry_apply_pair]
      exact mul_le_of_le_of_le_one' (mod_cast le_refl n₀) (pow_le_one₀ (by positivity) h)
        (by positivity) (by positivity)
    _ = n₀ * (Nat.log n₀ m + 1) := by
      rw [List.mapIdx_eq_enum_map, List.eq_replicate_of_mem (a := (n₀ : ℝ))
        (l := List.map (Function.uncurry fun _ _ ↦ n₀) (List.enum L)),
        List.sum_replicate, List.length_map, List.enum_length, nsmul_eq_mul, mul_comm,
        Nat.digits_len n₀ m hn₀ (not_eq_zero_of_lt hm), Nat.cast_add_one]
      simp +contextual
    _ ≤ n₀ * (logb n₀ m + 1) := by gcongr; exact natLog_le_logb ..
  -- For h_ineq2 we need to exclude the case n = 0.
  /-
    f : AbsoluteValue Rat Real
    n₀ : Nat
    hn₀ : LT.lt 1 n₀
    h : LE.le (f ↑n₀) 1
    n : Nat
    h_ineq1 : ∀ {m : Nat}, LE.le 1 m → LE.le (f ↑m) (HMul.hMul (↑n₀) (HAdd.hAdd (R …
    ⊢ LE.le (f ↑n) 1
  -/
  rcases eq_or_ne n 0 with rfl | h₀
    /-
      case inl
      f : AbsoluteValue Rat Real
      n₀ : Nat
      hn₀ : LT.lt 1 n₀
      h : LE.le (f ↑n₀) 1
      h_ineq1 : ∀ {m : Nat}, LE.le 1 m → LE.le (f ↑m) (HMul.hMul (↑n₀) (HAdd.hAdd (R …
      ⊢ LE.le (f ↑0) 1
    -/
  · simp
    /-
      🎉 no goals
    -/
  have h_ineq2 (k : ℕ) (hk : 0 < k) :
      f n ≤ (n₀ * (logb n₀ n + 1)) ^ (k : ℝ)⁻¹ * k ^ (k : ℝ)⁻¹ := by
    have : 0 ≤ logb n₀ n := logb_nonneg (one_lt_cast.mpr hn₀) (mod_cast Nat.one_le_of_lt h₀.bot_lt)
    calc
    f n = (f ↑(n ^ k)) ^ (k : ℝ)⁻¹ := by
      rw [Nat.cast_pow, map_pow, ← rpow_natCast, rpow_rpow_inv (by positivity) (by positivity)]
    _  ≤ (n₀ * (logb n₀ ↑(n ^ k) + 1)) ^ (k : ℝ)⁻¹ := by
      gcongr
      exact h_ineq1 <| one_le_pow₀ (one_le_iff_ne_zero.mpr h₀)
    _  = (n₀ * (k * logb n₀ n + 1)) ^ (k : ℝ)⁻¹ := by
      rw [Nat.cast_pow, logb_pow]
    _  ≤ (n₀ * (k * logb n₀ n + k)) ^ (k : ℝ)⁻¹ := by
      gcongr
      exact one_le_cast.mpr hk
    _ = (n₀ * (logb n₀ n + 1)) ^ (k : ℝ)⁻¹ * k ^ (k : ℝ)⁻¹ := by
      rw [← mul_rpow (by positivity) (by positivity), mul_assoc, add_mul, one_mul,
        mul_comm _ (k : ℝ)]
-- For 0 < logb n₀ n below we also need to exclude n = 1.
  /-
    case inr
    f : AbsoluteValue Rat Real
    n₀ : Nat
    hn₀ : LT.lt 1 n₀
    h : LE.le (f ↑n₀) 1
    n : Nat
    h_ineq1 : ∀ {m : Nat}, LE.le 1 m → LE.le (f ↑m) (HMul.hMul (↑n₀) (HAdd.hAdd (R …
    h₀ : Ne n 0
    h_ineq2 : ∀ (k : Nat), LT.lt 0 k → LE.le (f ↑n) (HMul.hMul (HPow.hPow (HMul.hM …
    ⊢ LE.le (f ↑n) 1
  -/
  rcases eq_or_ne n 1 with rfl | h₁
    /-
      case inr.inl
      f : AbsoluteValue Rat Real
      n₀ : Nat
      hn₀ : LT.lt 1 n₀
      h : LE.le (f ↑n₀) 1
      h_ineq1 : ∀ {m : Nat}, LE.le 1 m → LE.le (f ↑m) (HMul.hMul (↑n₀) (HAdd.hAdd (R …
      h₀ : Ne 1 0
      h_ineq2 : ∀ (k : Nat), LT.lt 0 k → LE.le (f ↑1) (HMul.hMul (HPow.hPow (HMul.hM …
      ⊢ LE.le (f ↑1) 1
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    f : AbsoluteValue Rat Real
    n₀ : Nat
    hn₀ : LT.lt 1 n₀
    h : LE.le (f ↑n₀) 1
    n : Nat
    h_ineq1 : ∀ {m : Nat}, LE.le 1 m → LE.le (f ↑m) (HMul.hMul (↑n₀) (HAdd.hAdd (R …
    h₀ : Ne n 0
    h_ineq2 : ∀ (k : Nat), LT.lt 0 k → LE.le (f ↑n) (HMul.hMul (HPow.hPow (HMul.hM …
    h₁ : Ne n 1
    ⊢ LE.le (f ↑n) 1
  -/
  refine le_of_tendsto_of_tendsto tendsto_const_nhds ?_ (eventually_atTop.mpr ⟨1, h_ineq2⟩)
  /-
    case inr.inr
    f : AbsoluteValue Rat Real
    n₀ : Nat
    hn₀ : LT.lt 1 n₀
    h : LE.le (f ↑n₀) 1
    n : Nat
    h_ineq1 : ∀ {m : Nat}, LE.le 1 m → LE.le (f ↑m) (HMul.hMul (↑n₀) (HAdd.hAdd (R …
    h₀ : Ne n 0
    h_ineq2 : ∀ (k : Nat), LT.lt 0 k → LE.le (f ↑n) (HMul.hMul (HPow.hPow (HMul.hM …
    h₁ : Ne n 1
    ⊢ Filter.Tendsto (fun b => HMul.hMul (HPow.hPow (HMul.hMul (↑n₀) (HAdd.hAdd (R …
  -/
  nth_rw 2 [← mul_one 1]
  /-
    case inr.inr
    f : AbsoluteValue Rat Real
    n₀ : Nat
    hn₀ : LT.lt 1 n₀
    h : LE.le (f ↑n₀) 1
    n : Nat
    h_ineq1 : ∀ {m : Nat}, LE.le 1 m → LE.le (f ↑m) (HMul.hMul (↑n₀) (HAdd.hAdd (R …
    h₀ : Ne n 0
    h_ineq2 : ∀ (k : Nat), LT.lt 0 k → LE.le (f ↑n) (HMul.hMul (HPow.hPow (HMul.hM …
    h₁ : Ne n 1
    ⊢ Filter.Tendsto (fun b => HMul.hMul (HPow.hPow (HMul.hMul (↑n₀) (HAdd.hAdd (R …
  -/
  have : 0 < logb n₀ n := logb_pos (mod_cast hn₀) (by norm_cast; omega)
  /-
    case inr.inr
    f : AbsoluteValue Rat Real
    n₀ : Nat
    hn₀ : LT.lt 1 n₀
    h : LE.le (f ↑n₀) 1
    n : Nat
    h_ineq1 : ∀ {m : Nat}, LE.le 1 m → LE.le (f ↑m) (HMul.hMul (↑n₀) (HAdd.hAdd (R …
    h₀ : Ne n 0
    h_ineq2 : ∀ (k : Nat), LT.lt 0 k → LE.le (f ↑n) (HMul.hMul (HPow.hPow (HMul.hM …
    h₁ : Ne n 1
    this : LT.lt 0 (Real.logb ↑n₀ ↑n)
    ⊢ Filter.Tendsto (fun b => HMul.hMul (HPow.hPow (HMul.hMul (↑n₀) (HAdd.hAdd (R …
  -/
  exact (tendsto_const_rpow_inv (by positivity)).mul tendsto_nat_rpow_inv
  /-
    🎉 no goals
  -/

-- ## Step 2: given m, n ≥ 2 and |m| = m^s, |n| = n^t for s, t > 0, we have t ≤ s


include hm notbdd in
private lemma expr_pos : 0 < m * f m / (f m - 1) := by
  apply div_pos (mul_pos (mod_cast zero_lt_of_lt hm)
    (map_pos_of_ne_zero f (mod_cast ne_zero_of_lt hm)))
  /-
    f : AbsoluteValue Rat Real
    m : Nat
    hm : LT.lt 1 m
    notbdd : Not (∀ (n : Nat), LE.le (f ↑n) 1)
    ⊢ LT.lt 0 (HSub.hSub (f ↑m) 1)
  -/
  linarith only [one_lt_of_not_bounded notbdd hm]
  /-
    🎉 no goals
  -/


include hn hm notbdd in
private lemma param_upperbound {k : ℕ} (hk : k ≠ 0) :
    f n ≤ (m * f m / (f m - 1)) ^ (k : ℝ)⁻¹ * f m ^ logb m n := by
  have h_ineq1 {m n : ℕ} (hm : 1 < m) (hn : 1 < n) :
      f n ≤ (m * f m / (f m - 1)) * f m ^ logb m n := by
    let d := Nat.log m n
    calc
    f n ≤ ((Nat.digits m n).mapIdx fun i _ ↦ m * f m ^ i).sum := apply_le_sum_digits n hm
    _ = m * ((Nat.digits m n).mapIdx fun i _ ↦ f m ^ i).sum := list_mul_sum (m.digits n) (f m) m
    _ = m * ((f m ^ (d + 1) - 1) / (f m - 1)) := by
      rw [list_geom _ (ne_of_gt (one_lt_of_not_bounded notbdd hm)),
        ← Nat.digits_len m n hm (not_eq_zero_of_lt hn)]
    _ ≤ m * ((f m ^ (d + 1)) / (f m - 1)) := by
      gcongr
      · linarith only [one_lt_of_not_bounded notbdd hm]
      · simp
    _ = ↑m * f ↑m / (f ↑m - 1) * f ↑m ^ d := by ring
    _ ≤ ↑m * f ↑m / (f ↑m - 1) * f ↑m ^ logb ↑m ↑n := by
      gcongr
      · exact (expr_pos hm notbdd).le
      · rw [← rpow_natCast, rpow_le_rpow_left_iff (one_lt_of_not_bounded notbdd hm)]
        exact natLog_le_logb n m
  apply le_of_pow_le_pow_left₀ hk <| mul_nonneg (rpow_nonneg (expr_pos hm notbdd).le _)
    (rpow_nonneg (apply_nonneg f ↑m) _)
  /-
    f : AbsoluteValue Rat Real
    m n : Nat
    hm : LT.lt 1 m
    hn : LT.lt 1 n
    notbdd : Not (∀ (n : Nat), LE.le (f ↑n) 1)
    k : Nat
    hk : Ne k 0
    h_ineq1 : ∀ {m n : Nat}, LT.lt 1 m → LT.lt 1 n → LE.le (f ↑n) (HMul.hMul (HDiv …
    ⊢ LE.le (HPow.hPow (f ↑n) k) (HPow.hPow (HMul.hMul (HPow.hPow (HDiv.hDiv (HMul …
  -/
  nth_rewrite 2 [← rpow_natCast]
  rw [mul_rpow (rpow_nonneg (expr_pos hm notbdd).le _) (rpow_nonneg (apply_nonneg f ↑m) _),
    ← rpow_mul (expr_pos hm notbdd).le, ← rpow_mul (apply_nonneg f ↑m),
    inv_mul_cancel₀ (mod_cast hk), rpow_one, mul_comm (logb ..)]
  calc
    (f n) ^ k = f ↑(n ^ k) := by simp
    _ ≤ (m * f m / (f m - 1)) * f m ^ logb m ↑(n ^ k) := h_ineq1 hm (Nat.one_lt_pow hk hn)
    _ = (m * f m / (f m - 1)) * f m ^ (k * logb m n) := by rw [Nat.cast_pow, logb_pow]


include hm hn notbdd in
/-- Given two natural numbers `n, m` greater than 1 we have `f n ≤ f m ^ logb m n`. -/
lemma le_pow_log : f n ≤ f m ^ logb m n := by
  have : Tendsto (fun k : ℕ ↦ (m * f m / (f m - 1)) ^ (k : ℝ)⁻¹ * f m ^ logb m n)
      atTop (𝓝 (f m ^ logb m n)) := by
    nth_rw 2 [← one_mul (f ↑m ^ logb ↑m ↑n)]
    exact (tendsto_const_rpow_inv (expr_pos hm notbdd)).mul_const _
  exact le_of_tendsto_of_tendsto (tendsto_const_nhds (x:= f ↑n)) this <|
    eventually_atTop.mpr ⟨2, fun b hb ↦ param_upperbound hm hn notbdd (not_eq_zero_of_lt hb)⟩


include hm hn notbdd in
/-- Given `m, n ≥ 2` and `f m = m ^ s`, `f n = n ^ t` for `s, t > 0`, we have `t ≤ s`. -/
private lemma le_of_eq_pow {s t : ℝ} (hfm : f m = m ^ s) (hfn : f n = n ^ t)  : t ≤ s := by
  /-
    f : AbsoluteValue Rat Real
    m n : Nat
    hm : LT.lt 1 m
    hn : LT.lt 1 n
    notbdd : Not (∀ (n : Nat), LE.le (f ↑n) 1)
    s t : Real
    hfm : Eq (f ↑m) (HPow.hPow (↑m) s)
    hfn : Eq (f ↑n) (HPow.hPow (↑n) t)
    ⊢ LE.le t s
  -/
  rw [← rpow_le_rpow_left_iff (x := n) (mod_cast hn), ← hfn]
  /-
    f : AbsoluteValue Rat Real
    m n : Nat
    hm : LT.lt 1 m
    hn : LT.lt 1 n
    notbdd : Not (∀ (n : Nat), LE.le (f ↑n) 1)
    s t : Real
    hfm : Eq (f ↑m) (HPow.hPow (↑m) s)
    hfn : Eq (f ↑n) (HPow.hPow (↑n) t)
    ⊢ LE.le (f ↑n) (HPow.hPow (↑n) s)
  -/
  apply le_trans <| le_pow_log hm hn notbdd
  rw [hfm, ← rpow_mul (Nat.cast_nonneg m), mul_comm, rpow_mul (Nat.cast_nonneg m),
    rpow_logb (mod_cast zero_lt_of_lt hm) (mod_cast hm.ne') (mod_cast zero_lt_of_lt hn)]


include hm hn notbdd in
private lemma eq_of_eq_pow {s t : ℝ} (hfm : f m = m ^ s) (hfn : f n = n ^ t) : s = t :=
  le_antisymm (le_of_eq_pow hn hm notbdd hfn hfm) (le_of_eq_pow hm hn notbdd hfm hfn)

-- ## Archimedean case: end goal


include notbdd in
/-- If `f` is not bounded and not trivial, then it is equivalent to the standard absolute value on
`ℚ`. -/
theorem equiv_real_of_unbounded : f ≈ real := by
  /-
    f : AbsoluteValue Rat Real
    notbdd : Not (∀ (n : Nat), LE.le (f ↑n) 1)
    ⊢ HasEquiv.Equiv f Rat.AbsoluteValue.real
  -/
  obtain ⟨m, hm⟩ := Classical.exists_not_of_not_forall notbdd
  have oneltm : 1 < m := by
    contrapose! hm
    rcases le_one_iff_eq_zero_or_eq_one.mp hm with rfl | rfl <;> simp
  /-
    case intro
    f : AbsoluteValue Rat Real
    notbdd : Not (∀ (n : Nat), LE.le (f ↑n) 1)
    m : Nat
    hm : Not (LE.le (f ↑m) 1)
    oneltm : LT.lt 1 m
    ⊢ HasEquiv.Equiv f Rat.AbsoluteValue.real
  -/
  rw [← equiv_on_nat_iff_equiv]
  /-
    case intro
    f : AbsoluteValue Rat Real
    notbdd : Not (∀ (n : Nat), LE.le (f ↑n) 1)
    m : Nat
    hm : Not (LE.le (f ↑m) 1)
    oneltm : LT.lt 1 m
    ⊢ Exists fun c => And (LT.lt 0 c) (∀ (n : Nat), Eq (HPow.hPow (f ↑n) c) (Rat.A …
  -/
  set s := logb m (f m) with hs
  refine ⟨s⁻¹,
    inv_pos.mpr (logb_pos (Nat.one_lt_cast.mpr oneltm) (one_lt_of_not_bounded notbdd oneltm)),
    fun n ↦ ?_⟩
  /-
    case intro
    f : AbsoluteValue Rat Real
    notbdd : Not (∀ (n : Nat), LE.le (f ↑n) 1)
    m : Nat
    hm : Not (LE.le (f ↑m) 1)
    oneltm : LT.lt 1 m
    s : Real := Real.logb (↑m) (f ↑m)
    hs : Eq s (Real.logb (↑m) (f ↑m))
    n : Nat
    ⊢ Eq (HPow.hPow (f ↑n) (Inv.inv s)) (Rat.AbsoluteValue.real ↑n)
  -/
  rcases lt_trichotomy n 1 with h | rfl | h
    /-
      case intro.inl
      f : AbsoluteValue Rat Real
      notbdd : Not (∀ (n : Nat), LE.le (f ↑n) 1)
      m : Nat
      hm : Not (LE.le (f ↑m) 1)
      oneltm : LT.lt 1 m
      s : Real := Real.logb (↑m) (f ↑m)
      hs : Eq s (Real.logb (↑m) (f ↑m))
      n : Nat
      h : LT.lt n 1
      ⊢ Eq (HPow.hPow (f ↑n) (Inv.inv s)) (Rat.AbsoluteValue.real ↑n)
    -/
  · obtain rfl : n = 0 := by omega
    have : (logb (↑m) (f ↑m))⁻¹ ≠ 0 := by
      simp only [ne_eq, inv_eq_zero, logb_eq_zero, Nat.cast_eq_zero, Nat.cast_eq_one, map_eq_zero,
        not_or]
      exact ⟨not_eq_zero_of_lt oneltm, oneltm.ne', by norm_cast,
        not_eq_zero_of_lt oneltm, ne_of_not_le hm, by linarith only [apply_nonneg f ↑m]⟩
    /-
      case intro.inl
      f : AbsoluteValue Rat Real
      notbdd : Not (∀ (n : Nat), LE.le (f ↑n) 1)
      m : Nat
      hm : Not (LE.le (f ↑m) 1)
      oneltm : LT.lt 1 m
      s : Real := Real.logb (↑m) (f ↑m)
      hs : Eq s (Real.logb (↑m) (f ↑m))
      h : LT.lt 0 1
      this : Ne (Inv.inv (Real.logb (↑m) (f ↑m))) 0
      ⊢ Eq (HPow.hPow (f ↑0) (Inv.inv s)) (Rat.AbsoluteValue.real ↑0)
    -/
    simp [hs, this]
    /-
      🎉 no goals
    -/
    /-
      case intro.inr.inl
      f : AbsoluteValue Rat Real
      notbdd : Not (∀ (n : Nat), LE.le (f ↑n) 1)
      m : Nat
      hm : Not (LE.le (f ↑m) 1)
      oneltm : LT.lt 1 m
      s : Real := Real.logb (↑m) (f ↑m)
      hs : Eq s (Real.logb (↑m) (f ↑m))
      ⊢ Eq (HPow.hPow (f ↑1) (Inv.inv s)) (Rat.AbsoluteValue.real ↑1)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case intro.inr.inr
      f : AbsoluteValue Rat Real
      notbdd : Not (∀ (n : Nat), LE.le (f ↑n) 1)
      m : Nat
      hm : Not (LE.le (f ↑m) 1)
      oneltm : LT.lt 1 m
      s : Real := Real.logb (↑m) (f ↑m)
      hs : Eq s (Real.logb (↑m) (f ↑m))
      n : Nat
      h : LT.lt 1 n
      ⊢ Eq (HPow.hPow (f ↑n) (Inv.inv s)) (Rat.AbsoluteValue.real ↑n)
    -/
  · simp only [real_eq_abs, abs_cast, Rat.cast_natCast]
    rw [rpow_inv_eq (apply_nonneg f ↑n) (Nat.cast_nonneg n)
      (logb_ne_zero_of_pos_of_ne_one (one_lt_cast.mpr oneltm) (by linarith only [hm])
      (by linarith only [hm]))]
    have hfm : f m = m ^ s := by
      rw [rpow_logb (mod_cast zero_lt_of_lt oneltm) (mod_cast oneltm.ne') (by linarith only [hm])]
    have hfn : f n = n ^ logb n (f n) := by
      rw [rpow_logb (mod_cast zero_lt_of_lt h) (mod_cast h.ne')
        (by apply map_pos_of_ne_zero; exact_mod_cast not_eq_zero_of_lt h)]
    /-
      case intro.inr.inr
      f : AbsoluteValue Rat Real
      notbdd : Not (∀ (n : Nat), LE.le (f ↑n) 1)
      m : Nat
      hm : Not (LE.le (f ↑m) 1)
      oneltm : LT.lt 1 m
      s : Real := Real.logb (↑m) (f ↑m)
      hs : Eq s (Real.logb (↑m) (f ↑m))
      n : Nat
      h : LT.lt 1 n
      hfm : Eq (f ↑m) (HPow.hPow (↑m) s)
      hfn : Eq (f ↑n) (HPow.hPow (↑n) (Real.logb (↑n) (f ↑n)))
      ⊢ Eq (f ↑n) (HPow.hPow (↑n) (Real.logb (↑m) (f ↑m)))
    -/
    rwa [← hs, eq_of_eq_pow oneltm h notbdd hfm hfn]
    /-
      🎉 no goals
    -/


/-- **Ostrowski's Theorem**: every absolute value (with values in `ℝ`) on `ℚ` is equivalent
to either the standard absolute value or a `p`-adic absolute value for a prime `p`. -/
theorem equiv_real_or_padic (f : AbsoluteValue ℚ ℝ) (hf_nontriv : f ≠ .trivial) :
    f ≈ real ∨ ∃! p, ∃ (_ : Fact p.Prime), f ≈ (padic p) := by
  /-
    f : AbsoluteValue Rat Real
    hf_nontriv : Ne f AbsoluteValue.trivial
    ⊢ Or (HasEquiv.Equiv f Rat.AbsoluteValue.real) (ExistsUnique fun p => Exists f …
  -/
  by_cases bdd : ∀ n : ℕ, f n ≤ 1
    /-
      case pos
      f : AbsoluteValue Rat Real
      hf_nontriv : Ne f AbsoluteValue.trivial
      bdd : ∀ (n : Nat), LE.le (f ↑n) 1
      ⊢ Or (HasEquiv.Equiv f Rat.AbsoluteValue.real) (ExistsUnique fun p => Exists f …
    -/
  · exact .inr <| equiv_padic_of_bounded hf_nontriv bdd
    /-
      🎉 no goals
    -/
    /-
      case neg
      f : AbsoluteValue Rat Real
      hf_nontriv : Ne f AbsoluteValue.trivial
      bdd : Not (∀ (n : Nat), LE.le (f ↑n) 1)
      ⊢ Or (HasEquiv.Equiv f Rat.AbsoluteValue.real) (ExistsUnique fun p => Exists f …
    -/
  · exact .inl <| equiv_real_of_unbounded bdd
    /-
      🎉 no goals
    -/


/-- The standard absolute value on `ℚ` is not equivalent to any `p`-adic absolute value. -/
lemma not_real_equiv_padic (p : ℕ) [Fact p.Prime] : ¬ real ≈ (padic p) := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    ⊢ Not (HasEquiv.Equiv Rat.AbsoluteValue.real (Rat.AbsoluteValue.padic p))
  -/
  rintro ⟨c, hc₀, hc⟩
  /-
    case intro.intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    c : Real
    hc₀ : LT.lt 0 c
    hc : Eq (fun x => HPow.hPow (Rat.AbsoluteValue.real x) c) ⇑(Rat.AbsoluteValue. …
    ⊢ False
  -/
  apply_fun (· 2) at hc
  /-
    case intro.intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    c : Real
    hc₀ : LT.lt 0 c
    hc : Eq ((fun x => HPow.hPow (Rat.AbsoluteValue.real x) c) 2) ((Rat.AbsoluteVa …
    ⊢ False
  -/
  simp only [real_eq_abs, abs_ofNat, cast_ofNat] at hc
  /-
    case intro.intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    c : Real
    hc₀ : LT.lt 0 c
    hc : Eq (HPow.hPow 2 c) ((Rat.AbsoluteValue.padic p) 2)
    ⊢ False
  -/
  exact ((padic_le_one p 2).trans_lt <| one_lt_rpow one_lt_two hc₀).ne' hc
  /-
    🎉 no goals
  -/



/-- The Gaussian integers, defined as `ℤ√(-1)`. -/
abbrev GaussianInt : Type :=
  Zsqrtd (-1)


local notation "ℤ[i]" => GaussianInt


instance : Repr ℤ[i] :=
  ⟨fun x _ => "⟨" ++ repr x.re ++ ", " ++ repr x.im ++ "⟩"⟩


instance instCommRing : CommRing ℤ[i] :=
  Zsqrtd.commRing


/-- The embedding of the Gaussian integers into the complex numbers, as a ring homomorphism. -/
def toComplex : ℤ[i] →+* ℂ :=
                     /-
                       ⊢ Eq (HMul.hMul Complex.I Complex.I) ↑(-1)
                     -/
  Zsqrtd.lift ⟨I, by simp⟩
                     /-
                       🎉 no goals
                     -/


instance : Coe ℤ[i] ℂ :=
  ⟨toComplex⟩


theorem toComplex_def (x : ℤ[i]) : (x : ℂ) = x.re + x.im * I :=
  rfl


                                                                           /-
                                                                             x y : Int
                                                                             ⊢ Eq (GaussianInt.toComplex { re := x, im := y }) (HAdd.hAdd (↑x) (HMul.hMul ( …
                                                                           -/
theorem toComplex_def' (x y : ℤ) : ((⟨x, y⟩ : ℤ[i]) : ℂ) = x + y * I := by simp [toComplex_def]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


theorem toComplex_def₂ (x : ℤ[i]) : (x : ℂ) = ⟨x.re, x.im⟩ := by
  /-
    x : GaussianInt
    ⊢ Eq (GaussianInt.toComplex x) { re := ↑x.re, im := ↑x.im }
  -/
                        /-
                          🎉 no goals
                        -/
  apply Complex.ext <;> simp [toComplex_def]
                        /-
                          🎉 no goals
                        -/


@[simp]
                                                                    /-
                                                                      x : GaussianInt
                                                                      ⊢ Eq (↑x.re) (GaussianInt.toComplex x).re
                                                                    -/
theorem to_real_re (x : ℤ[i]) : ((x.re : ℤ) : ℝ) = (x : ℂ).re := by simp [toComplex_def]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
                                                                    /-
                                                                      x : GaussianInt
                                                                      ⊢ Eq (↑x.im) (GaussianInt.toComplex x).im
                                                                    -/
theorem to_real_im (x : ℤ[i]) : ((x.im : ℤ) : ℝ) = (x : ℂ).im := by simp [toComplex_def]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
                                                                    /-
                                                                      x y : Int
                                                                      ⊢ Eq (GaussianInt.toComplex { re := x, im := y }).re ↑x
                                                                    -/
theorem toComplex_re (x y : ℤ) : ((⟨x, y⟩ : ℤ[i]) : ℂ).re = x := by simp [toComplex_def]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
                                                                    /-
                                                                      x y : Int
                                                                      ⊢ Eq (GaussianInt.toComplex { re := x, im := y }).im ↑y
                                                                    -/
theorem toComplex_im (x y : ℤ) : ((⟨x, y⟩ : ℤ[i]) : ℂ).im = y := by simp [toComplex_def]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem toComplex_add (x y : ℤ[i]) : ((x + y : ℤ[i]) : ℂ) = x + y :=
  toComplex.map_add _ _


theorem toComplex_mul (x y : ℤ[i]) : ((x * y : ℤ[i]) : ℂ) = x * y :=
  toComplex.map_mul _ _


theorem toComplex_one : ((1 : ℤ[i]) : ℂ) = 1 :=
  toComplex.map_one


theorem toComplex_zero : ((0 : ℤ[i]) : ℂ) = 0 :=
  toComplex.map_zero


theorem toComplex_neg (x : ℤ[i]) : ((-x : ℤ[i]) : ℂ) = -x :=
  toComplex.map_neg _


theorem toComplex_sub (x y : ℤ[i]) : ((x - y : ℤ[i]) : ℂ) = x - y :=
  toComplex.map_sub _ _


@[simp]
theorem toComplex_star (x : ℤ[i]) : ((star x : ℤ[i]) : ℂ) = conj (x : ℂ) := by
  /-
    x : GaussianInt
    ⊢ Eq (GaussianInt.toComplex (Star.star x)) ((starRingEnd Complex) (GaussianInt …
  -/
  rw [toComplex_def₂, toComplex_def₂]
  /-
    x : GaussianInt
    ⊢ Eq { re := ↑(Star.star x).re, im := ↑(Star.star x).im } ((starRingEnd Comple …
  -/
  exact congr_arg₂ _ rfl (Int.cast_neg _)
  /-
    🎉 no goals
  -/


@[simp]
theorem toComplex_inj {x y : ℤ[i]} : (x : ℂ) = y ↔ x = y := by
  /-
    x y : GaussianInt
    ⊢ Iff (Eq (GaussianInt.toComplex x) (GaussianInt.toComplex y)) (Eq x y)
  -/
  cases x; cases y; simp [toComplex_def₂]
                    /-
                      🎉 no goals
                    -/


lemma toComplex_injective : Function.Injective GaussianInt.toComplex :=
  fun ⦃_ _⦄ ↦ toComplex_inj.mp


@[simp]
theorem toComplex_eq_zero {x : ℤ[i]} : (x : ℂ) = 0 ↔ x = 0 := by
  /-
    x : GaussianInt
    ⊢ Iff (Eq (GaussianInt.toComplex x) 0) (Eq x 0)
  -/
  rw [← toComplex_zero, toComplex_inj]
  /-
    🎉 no goals
  -/


@[simp]
theorem intCast_real_norm (x : ℤ[i]) : (x.norm : ℝ) = Complex.normSq (x : ℂ) := by
  /-
    x : GaussianInt
    ⊢ Eq (↑(Zsqrtd.norm x)) (Complex.normSq (GaussianInt.toComplex x))
  -/
  rw [Zsqrtd.norm, normSq]; simp
                            /-
                              🎉 no goals
                            -/


@[deprecated (since := "2024-04-17")]
alias int_cast_real_norm := intCast_real_norm


@[simp]
theorem intCast_complex_norm (x : ℤ[i]) : (x.norm : ℂ) = Complex.normSq (x : ℂ) := by
  /-
    x : GaussianInt
    ⊢ Eq ↑(Zsqrtd.norm x) ↑(Complex.normSq (GaussianInt.toComplex x))
  -/
  cases x; rw [Zsqrtd.norm, normSq]; simp
                                     /-
                                       🎉 no goals
                                     -/


@[deprecated (since := "2024-04-17")]
alias int_cast_complex_norm := intCast_complex_norm


theorem norm_nonneg (x : ℤ[i]) : 0 ≤ norm x :=
                         /-
                           x : GaussianInt
                           ⊢ LE.le (-1) 0
                         -/
  Zsqrtd.norm_nonneg (by norm_num) _
                         /-
                           🎉 no goals
                         -/


@[simp]
                                                           /-
                                                             x : GaussianInt
                                                             ⊢ Iff (Eq (Zsqrtd.norm x) 0) (Eq x 0)
                                                           -/
theorem norm_eq_zero {x : ℤ[i]} : norm x = 0 ↔ x = 0 := by rw [← @Int.cast_inj ℝ _ _ _]; simp
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


theorem norm_pos {x : ℤ[i]} : 0 < norm x ↔ x ≠ 0 := by
  /-
    x : GaussianInt
    ⊢ Iff (LT.lt 0 (Zsqrtd.norm x)) (Ne x 0)
  -/
  rw [lt_iff_le_and_ne, Ne, eq_comm, norm_eq_zero]; simp [norm_nonneg]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem abs_natCast_norm (x : ℤ[i]) : (x.norm.natAbs : ℤ) = x.norm :=
  Int.natAbs_of_nonneg (norm_nonneg _)


@[deprecated (since := "2024-04-05")] alias abs_coe_nat_norm := abs_natCast_norm


@[simp]
theorem natCast_natAbs_norm {α : Type*} [Ring α] (x : ℤ[i]) : (x.norm.natAbs : α) = x.norm := by
  /-
    α : Type u_1
    inst✝ : Ring α
    x : GaussianInt
    ⊢ Eq ↑(Zsqrtd.norm x).natAbs ↑(Zsqrtd.norm x)
  -/
  rw [← Int.cast_natCast, abs_natCast_norm]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias nat_cast_natAbs_norm := natCast_natAbs_norm


theorem natAbs_norm_eq (x : ℤ[i]) :
    x.norm.natAbs = x.re.natAbs * x.re.natAbs + x.im.natAbs * x.im.natAbs :=
                      /-
                        x : GaussianInt
                        ⊢ Eq (Int.ofNat (Zsqrtd.norm x).natAbs) (Int.ofNat (HAdd.hAdd (HMul.hMul x.re. …
                      -/
  Int.ofNat.inj <| by simp; simp [Zsqrtd.norm]
                            /-
                              🎉 no goals
                            -/


instance : Div ℤ[i] :=
  ⟨fun x y =>
    let n := (norm y : ℚ)⁻¹
    let c := star y
    ⟨round ((x * c).re * n : ℚ), round ((x * c).im * n : ℚ)⟩⟩


theorem div_def (x y : ℤ[i]) :
    x / y = ⟨round ((x * star y).re / norm y : ℚ), round ((x * star y).im / norm y : ℚ)⟩ :=
                            /-
                              x y : GaussianInt
                              ⊢ Eq { re := round (HMul.hMul (↑(HMul.hMul x (Star.star y)).re) (Inv.inv ↑(Zsq …
                            -/
  show Zsqrtd.mk _ _ = _ by simp [div_eq_mul_inv]
                            /-
                              🎉 no goals
                            -/


theorem toComplex_div_re (x y : ℤ[i]) : ((x / y : ℤ[i]) : ℂ).re = round (x / y : ℂ).re := by
  /-
    x y : GaussianInt
    ⊢ Eq (GaussianInt.toComplex (HDiv.hDiv x y)).re ↑(round (HDiv.hDiv (GaussianIn …
  -/
  rw [div_def, ← @Rat.round_cast ℝ _ _]
  /-
    x y : GaussianInt
    ⊢ Eq (GaussianInt.toComplex { re := round ↑(HDiv.hDiv ↑(HMul.hMul x (Star.star …
  -/
  simp [-Rat.round_cast, mul_assoc, div_eq_mul_inv, mul_add, add_mul]
  /-
    🎉 no goals
  -/


theorem toComplex_div_im (x y : ℤ[i]) : ((x / y : ℤ[i]) : ℂ).im = round (x / y : ℂ).im := by
  /-
    x y : GaussianInt
    ⊢ Eq (GaussianInt.toComplex (HDiv.hDiv x y)).im ↑(round (HDiv.hDiv (GaussianIn …
  -/
  rw [div_def, ← @Rat.round_cast ℝ _ _, ← @Rat.round_cast ℝ _ _]
  /-
    x y : GaussianInt
    ⊢ Eq (GaussianInt.toComplex { re := round ↑(HDiv.hDiv ↑(HMul.hMul x (Star.star …
  -/
  simp [-Rat.round_cast, mul_assoc, div_eq_mul_inv, mul_add, add_mul]
  /-
    🎉 no goals
  -/


theorem normSq_le_normSq_of_re_le_of_im_le {x y : ℂ} (hre : |x.re| ≤ |y.re|)
    (him : |x.im| ≤ |y.im|) : Complex.normSq x ≤ Complex.normSq y := by
  rw [normSq_apply, normSq_apply, ← _root_.abs_mul_self, _root_.abs_mul, ←
      _root_.abs_mul_self y.re, _root_.abs_mul y.re, ← _root_.abs_mul_self x.im,
      _root_.abs_mul x.im, ← _root_.abs_mul_self y.im, _root_.abs_mul y.im]
  exact
      add_le_add (mul_self_le_mul_self (abs_nonneg _) hre) (mul_self_le_mul_self (abs_nonneg _) him)


theorem normSq_div_sub_div_lt_one (x y : ℤ[i]) :
    Complex.normSq ((x / y : ℂ) - ((x / y : ℤ[i]) : ℂ)) < 1 :=
  calc
    Complex.normSq ((x / y : ℂ) - ((x / y : ℤ[i]) : ℂ))
    _ = Complex.normSq
      ((x / y : ℂ).re - ((x / y : ℤ[i]) : ℂ).re + ((x / y : ℂ).im - ((x / y : ℤ[i]) : ℂ).im) *
        I : ℂ) :=
                        /-
                          x y : GaussianInt
                          ⊢ Eq (HSub.hSub (HDiv.hDiv (GaussianInt.toComplex x) (GaussianInt.toComplex y) …
                        -/
                                              /-
                                                🎉 no goals
                                              -/
      congr_arg _ <| by apply Complex.ext <;> simp
                                              /-
                                                🎉 no goals
                                              -/
    _ ≤ Complex.normSq (1 / 2 + 1 / 2 * I) := by
      /-
        x y : GaussianInt
        ⊢ LE.le (Complex.normSq (HAdd.hAdd (HSub.hSub ↑(HDiv.hDiv (GaussianInt.toCompl …
      -/
      have : |(2⁻¹ : ℝ)| = 2⁻¹ := abs_of_nonneg (by norm_num)
      exact normSq_le_normSq_of_re_le_of_im_le
        (by rw [toComplex_div_re]; simp [normSq, this]; simpa using abs_sub_round (x / y : ℂ).re)
        (by rw [toComplex_div_im]; simp [normSq, this]; simpa using abs_sub_round (x / y : ℂ).im)
                /-
                  x y : GaussianInt
                  ⊢ LT.lt (Complex.normSq (HAdd.hAdd (1 / 2) (HMul.hMul (1 / 2) Complex.I))) 1
                -/
    _ < 1 := by simp [normSq]; norm_num
                               /-
                                 🎉 no goals
                               -/


instance : Mod ℤ[i] :=
  ⟨fun x y => x - y * (x / y)⟩


theorem mod_def (x y : ℤ[i]) : x % y = x - y * (x / y) :=
  rfl


theorem norm_mod_lt (x : ℤ[i]) {y : ℤ[i]} (hy : y ≠ 0) : (x % y).norm < y.norm :=
                           /-
                             x y : GaussianInt
                             hy : Ne y 0
                             ⊢ Ne (GaussianInt.toComplex y) 0
                           -/
  have : (y : ℂ) ≠ 0 := by rwa [Ne, ← toComplex_zero, toComplex_inj]
                           /-
                             🎉 no goals
                           -/
  (@Int.cast_lt ℝ _ _ _ _).1 <|
    calc
                                                                                 /-
                                                                                   x y : GaussianInt
                                                                                   hy : Ne y 0
                                                                                   this : Ne (GaussianInt.toComplex y) 0
                                                                                   ⊢ Eq (↑(Zsqrtd.norm (HMod.hMod x y))) (Complex.normSq (HSub.hSub (GaussianInt. …
                                                                                 -/
      ↑(Zsqrtd.norm (x % y)) = Complex.normSq (x - y * (x / y : ℤ[i]) : ℂ) := by simp [mod_def]
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
      _ = Complex.normSq (y : ℂ) * Complex.normSq (x / y - (x / y : ℤ[i]) : ℂ) := by
        /-
          x y : GaussianInt
          hy : Ne y 0
          this : Ne (GaussianInt.toComplex y) 0
          ⊢ Eq (Complex.normSq (HSub.hSub (GaussianInt.toComplex x) (HMul.hMul (Gaussian …
        -/
        rw [← normSq_mul, mul_sub, mul_div_cancel₀ _ this]
        /-
          🎉 no goals
        -/
      _ < Complex.normSq (y : ℂ) * 1 :=
        (mul_lt_mul_of_pos_left (normSq_div_sub_div_lt_one _ _) (normSq_pos.2 this))
                              /-
                                x y : GaussianInt
                                hy : Ne y 0
                                this : Ne (GaussianInt.toComplex y) 0
                                ⊢ Eq (HMul.hMul (Complex.normSq (GaussianInt.toComplex y)) 1) ↑(Zsqrtd.norm y)
                              -/
      _ = Zsqrtd.norm y := by simp
                              /-
                                🎉 no goals
                              -/


theorem natAbs_norm_mod_lt (x : ℤ[i]) {y : ℤ[i]} (hy : y ≠ 0) :
    (x % y).norm.natAbs < y.norm.natAbs :=
                     /-
                       x y : GaussianInt
                       hy : Ne y 0
                       ⊢ LT.lt ↑(Zsqrtd.norm (HMod.hMod x y)).natAbs ↑(Zsqrtd.norm y).natAbs
                     -/
  Int.ofNat_lt.1 (by simp [-Int.ofNat_lt, norm_mod_lt x hy])
                     /-
                       🎉 no goals
                     -/


theorem norm_le_norm_mul_left (x : ℤ[i]) {y : ℤ[i]} (hy : y ≠ 0) :
    (norm x).natAbs ≤ (norm (x * y)).natAbs := by
  /-
    x y : GaussianInt
    hy : Ne y 0
    ⊢ LE.le (Zsqrtd.norm x).natAbs (Zsqrtd.norm (HMul.hMul x y)).natAbs
  -/
  rw [Zsqrtd.norm_mul, Int.natAbs_mul]
  exact le_mul_of_one_le_right (Nat.zero_le _) (Int.ofNat_le.1 (by
    rw [abs_natCast_norm]
    exact Int.add_one_le_of_lt (norm_pos.2 hy)))


instance instNontrivial : Nontrivial ℤ[i] :=
             /-
               ⊢ Ne 0 1
             -/
  ⟨⟨0, 1, by decide⟩⟩
             /-
               🎉 no goals
             -/


instance : EuclideanDomain ℤ[i] :=
  { GaussianInt.instCommRing,
    GaussianInt.instNontrivial with
    quotient := (· / ·)
    remainder := (· % ·)
                        /-
                          ⊢ ∀ (a : GaussianInt), Eq ((fun x1 x2 => HDiv.hDiv x1 x2) a 0) 0
                        -/
    quotient_zero := by simp [div_def]; rfl
                                        /-
                                          🎉 no goals
                                        -/
                                                   /-
                                                     x✝¹ x✝ : GaussianInt
                                                     ⊢ Eq (HAdd.hAdd (HMul.hMul x✝ ((fun x1 x2 => HDiv.hDiv x1 x2) x✝¹ x✝)) ((fun x …
                                                   -/
    quotient_mul_add_remainder_eq := fun _ _ => by simp [mod_def]
                                                   /-
                                                     🎉 no goals
                                                   -/
    r := _
    r_wellFounded := (measure (Int.natAbs ∘ norm)).wf
    remainder_lt := natAbs_norm_mod_lt
    mul_left_not_lt := fun a _ hb0 => not_lt_of_ge <| norm_le_norm_mul_left a hb0 }


theorem sq_add_sq_of_nat_prime_of_not_irreducible (p : ℕ) [hp : Fact p.Prime]
    (hpi : ¬Irreducible (p : ℤ[i])) : ∃ a b, a ^ 2 + b ^ 2 = p :=
  have hpu : ¬IsUnit (p : ℤ[i]) :=
    mt norm_eq_one_iff.2 <| by
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        hpi : Not (Irreducible ↑p)
        ⊢ Not (Eq (Zsqrtd.norm ↑p).natAbs 1)
      -/
      rw [norm_natCast, Int.natAbs_mul, mul_eq_one]
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        hpi : Not (Irreducible ↑p)
        ⊢ Not (And (Eq (↑p).natAbs 1) (Eq (↑p).natAbs 1))
      -/
      exact fun h => (ne_of_lt hp.1.one_lt).symm h.1
      /-
        🎉 no goals
      -/
  have hab : ∃ a b, (p : ℤ[i]) = a * b ∧ ¬IsUnit a ∧ ¬IsUnit b := by
    -- Porting note: was
    -- simpa [irreducible_iff, hpu, not_forall, not_or] using hpi
    simpa only [true_and, not_false_iff, exists_prop, irreducible_iff, hpu, not_forall, not_or]
      using hpi
  let ⟨a, b, hpab, hau, hbu⟩ := hab
  have hnap : (norm a).natAbs = p :=
    ((hp.1.mul_eq_prime_sq_iff (mt norm_eq_one_iff.1 hau) (mt norm_eq_one_iff.1 hbu)).1 <| by
        /-
          p : Nat
          hp : Fact (Nat.Prime p)
          hpi : Not (Irreducible ↑p)
          hpu : Not (IsUnit ↑p)
          hab : Exists fun a => Exists fun b => And (Eq (↑p) (HMul.hMul a b)) (And (Not  …
          a b : GaussianInt
          hpab : Eq (↑p) (HMul.hMul a b)
          hau : Not (IsUnit a)
          hbu : Not (IsUnit b)
          ⊢ Eq (HMul.hMul (Zsqrtd.norm a).natAbs (Zsqrtd.norm b).natAbs) (HPow.hPow p 2)
        -/
        rw [← Int.natCast_inj, Int.natCast_pow, sq, ← @norm_natCast (-1), hpab]; simp).1
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
                                /-
                                  p : Nat
                                  hp : Fact (Nat.Prime p)
                                  hpi : Not (Irreducible ↑p)
                                  hpu : Not (IsUnit ↑p)
                                  hab : Exists fun a => Exists fun b => And (Eq (↑p) (HMul.hMul a b)) (And (Not  …
                                  a b : GaussianInt
                                  hpab : Eq (↑p) (HMul.hMul a b)
                                  hau : Not (IsUnit a)
                                  hbu : Not (IsUnit b)
                                  hnap : Eq (Zsqrtd.norm a).natAbs p
                                  ⊢ Eq (HAdd.hAdd (HPow.hPow a.re.natAbs 2) (HPow.hPow a.im.natAbs 2)) p
                                -/
  ⟨a.re.natAbs, a.im.natAbs, by simpa [natAbs_norm_eq, sq] using hnap⟩
                                /-
                                  🎉 no goals
                                -/



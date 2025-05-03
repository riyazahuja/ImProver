theorem padic_polynomial_dist {p : ℕ} [Fact p.Prime] (F : Polynomial ℤ_[p]) (x y : ℤ_[p]) :
    ‖F.eval x - F.eval y‖ ≤ ‖x - y‖ :=
  let ⟨z, hz⟩ := F.evalSubFactor x y
  calc
                                                /-
                                                  p : Nat
                                                  inst✝ : Fact (Nat.Prime p)
                                                  F : Polynomial (PadicInt p)
                                                  x y z : PadicInt p
                                                  hz : Eq (HSub.hSub (Polynomial.eval x F) (Polynomial.eval y F)) (HMul.hMul z ( …
                                                  ⊢ Eq (Norm.norm (HSub.hSub (Polynomial.eval x F) (Polynomial.eval y F))) (HMul …
                                                -/
    ‖F.eval x - F.eval y‖ = ‖z‖ * ‖x - y‖ := by simp [hz]
                                                /-
                                                  🎉 no goals
                                                -/
                          /-
                            p : Nat
                            inst✝ : Fact (Nat.Prime p)
                            F : Polynomial (PadicInt p)
                            x y z : PadicInt p
                            hz : Eq (HSub.hSub (Polynomial.eval x F) (Polynomial.eval y F)) (HMul.hMul z ( …
                            ⊢ LE.le (HMul.hMul (Norm.norm z) (Norm.norm (HSub.hSub x y))) (HMul.hMul 1 (No …
                          -/
    _ ≤ 1 * ‖x - y‖ := by gcongr; apply PadicInt.norm_le_one
                                  /-
                                    🎉 no goals
                                  -/
                      /-
                        p : Nat
                        inst✝ : Fact (Nat.Prime p)
                        F : Polynomial (PadicInt p)
                        x y z : PadicInt p
                        hz : Eq (HSub.hSub (Polynomial.eval x F) (Polynomial.eval y F)) (HMul.hMul z ( …
                        ⊢ Eq (HMul.hMul 1 (Norm.norm (HSub.hSub x y))) (Norm.norm (HSub.hSub x y))
                      -/
    _ = ‖x - y‖ := by simp
                      /-
                        🎉 no goals
                      -/


private theorem comp_tendsto_lim {p : ℕ} [Fact p.Prime] {F : Polynomial ℤ_[p]}
    (ncs : CauSeq ℤ_[p] norm) : Tendsto (fun i => F.eval (ncs i)) atTop (𝓝 (F.eval ncs.lim)) :=
  Filter.Tendsto.comp (@Polynomial.continuousAt _ _ _ _ F _) ncs.tendsto_limit


private theorem ncs_tendsto_lim :
    Tendsto (fun i => ‖F.derivative.eval (ncs i)‖) atTop (𝓝 ‖F.derivative.eval ncs.lim‖) :=
  Tendsto.comp (continuous_iff_continuousAt.1 continuous_norm _) (comp_tendsto_lim _)


private theorem ncs_tendsto_const :
    Tendsto (fun i => ‖F.derivative.eval (ncs i)‖) atTop (𝓝 ‖F.derivative.eval a‖) := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    ncs : CauSeq (PadicInt p) Norm.norm
    F : Polynomial (PadicInt p)
    a : PadicInt p
    ncs_der_val : ∀ (n : Nat), Eq (Norm.norm (Polynomial.eval (↑ncs n) (Polynomial …
    ⊢ Filter.Tendsto (fun i => Norm.norm (Polynomial.eval (↑ncs i) (Polynomial.der …
  -/
  convert @tendsto_const_nhds ℝ ℕ _ _ _; rw [ncs_der_val]
                                         /-
                                           🎉 no goals
                                         -/


private theorem norm_deriv_eq : ‖F.derivative.eval ncs.lim‖ = ‖F.derivative.eval a‖ :=
  tendsto_nhds_unique ncs_tendsto_lim (ncs_tendsto_const ncs_der_val)


private theorem tendsto_zero_of_norm_tendsto_zero : Tendsto (fun i => F.eval (ncs i)) atTop (𝓝 0) :=
                                          /-
                                            p : Nat
                                            inst✝ : Fact (Nat.Prime p)
                                            ncs : CauSeq (PadicInt p) Norm.norm
                                            F : Polynomial (PadicInt p)
                                            hnorm : Filter.Tendsto (fun i => Norm.norm (Polynomial.eval (↑ncs i) F)) Filte …
                                            ⊢ Filter.Tendsto (fun e => Norm.norm (HSub.hSub (Polynomial.eval (↑ncs e) F) 0 …
                                          -/
  tendsto_iff_norm_sub_tendsto_zero.2 (by simpa using hnorm)
                                          /-
                                            🎉 no goals
                                          -/


theorem limit_zero_of_norm_tendsto_zero : F.eval ncs.lim = 0 :=
  tendsto_nhds_unique (comp_tendsto_lim _) (tendsto_zero_of_norm_tendsto_zero hnorm)


/-- `T` is an auxiliary value that is used to control the behavior of the polynomial `F`. -/
private def T_gen : ℝ := ‖F.eval a / ((F.derivative.eval a ^ 2 : ℤ_[p]) : ℚ_[p])‖


local notation "T" => @T_gen p _ F a


private theorem T_def : T = ‖F.eval a‖ / ‖F.derivative.eval a‖ ^ 2 := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    F : Polynomial (PadicInt p)
    a : PadicInt p
    ⊢ Eq (T_gen p F a) (HDiv.hDiv (Norm.norm (Polynomial.eval a F)) (HPow.hPow (No …
  -/
  simp [T_gen, ← PadicInt.norm_def]
  /-
    🎉 no goals
  -/


private theorem T_nonneg : 0 ≤ T := norm_nonneg _


private theorem T_pow_nonneg (n : ℕ) : 0 ≤ T ^ n := pow_nonneg T_nonneg _


private theorem deriv_sq_norm_pos : 0 < ‖F.derivative.eval a‖ ^ 2 :=
  lt_of_le_of_lt (norm_nonneg _) hnorm


private theorem deriv_sq_norm_ne_zero : ‖F.derivative.eval a‖ ^ 2 ≠ 0 :=
  ne_of_gt (deriv_sq_norm_pos hnorm)


private theorem deriv_norm_ne_zero : ‖F.derivative.eval a‖ ≠ 0 := fun h =>
                                  /-
                                    p : Nat
                                    inst✝ : Fact (Nat.Prime p)
                                    F : Polynomial (PadicInt p)
                                    a : PadicInt p
                                    hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
                                    h : Eq (Norm.norm (Polynomial.eval a (Polynomial.derivative F))) 0
                                    ⊢ Eq (HPow.hPow (Norm.norm (Polynomial.eval a (Polynomial.derivative F))) 2) 0
                                  -/
  deriv_sq_norm_ne_zero hnorm (by simp [*, sq])
                                  /-
                                    🎉 no goals
                                  -/


private theorem deriv_norm_pos : 0 < ‖F.derivative.eval a‖ :=
  lt_of_le_of_ne (norm_nonneg _) (Ne.symm (deriv_norm_ne_zero hnorm))


private theorem deriv_ne_zero : F.derivative.eval a ≠ 0 :=
  mt norm_eq_zero.2 (deriv_norm_ne_zero hnorm)



private theorem T_lt_one : T < 1 := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    F : Polynomial (PadicInt p)
    a : PadicInt p
    hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
    ⊢ LT.lt (T_gen p F a) 1
  -/
  have h := (div_lt_one (deriv_sq_norm_pos hnorm)).2 hnorm
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    F : Polynomial (PadicInt p)
    a : PadicInt p
    hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
    h : LT.lt (HDiv.hDiv (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm ( …
    ⊢ LT.lt (T_gen p F a) 1
  -/
  rw [T_def]; exact h
              /-
                🎉 no goals
              -/


private theorem T_pow {n : ℕ} (hn : n ≠ 0) : T ^ n < 1 := pow_lt_one₀ T_nonneg (T_lt_one hnorm) hn


private theorem T_pow' (n : ℕ) : T ^ 2 ^ n < 1 := T_pow hnorm (pow_ne_zero _ two_ne_zero)

-- Porting note: renamed this `def` and used a local notation to provide arguments automatically

/-- We will construct a sequence of elements of ℤ_p satisfying successive values of `ih`. -/
private def ih_gen (n : ℕ) (z : ℤ_[p]) : Prop :=
  ‖F.derivative.eval z‖ = ‖F.derivative.eval a‖ ∧ ‖F.eval z‖ ≤ ‖F.derivative.eval a‖ ^ 2 * T ^ 2 ^ n


local notation "ih" => @ih_gen p _ F a


private theorem ih_0 : ih 0 a :=
           /-
             p : Nat
             inst✝ : Fact (Nat.Prime p)
             F : Polynomial (PadicInt p)
             a : PadicInt p
             hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
             ⊢ LE.le (Norm.norm (Polynomial.eval a F)) (HMul.hMul (HPow.hPow (Norm.norm (Po …
           -/
  ⟨rfl, by simp [T_def, mul_div_cancel₀ _ (ne_of_gt (deriv_sq_norm_pos hnorm))]⟩
           /-
             🎉 no goals
           -/


private theorem calc_norm_le_one {n : ℕ} {z : ℤ_[p]} (hz : ih n z) :
    ‖(↑(F.eval z) : ℚ_[p]) / ↑(F.derivative.eval z)‖ ≤ 1 :=
  calc
    ‖(↑(F.eval z) : ℚ_[p]) / ↑(F.derivative.eval z)‖ =
        ‖(↑(F.eval z) : ℚ_[p])‖ / ‖(↑(F.derivative.eval z) : ℚ_[p])‖ :=
      norm_div _ _
                                                 /-
                                                   p : Nat
                                                   inst✝ : Fact (Nat.Prime p)
                                                   F : Polynomial (PadicInt p)
                                                   a : PadicInt p
                                                   hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
                                                   n : Nat
                                                   z : PadicInt p
                                                   hz : ih_gen n z
                                                   ⊢ Eq (HDiv.hDiv (Norm.norm ↑(Polynomial.eval z F)) (Norm.norm ↑(Polynomial.eva …
                                                 -/
    _ = ‖F.eval z‖ / ‖F.derivative.eval a‖ := by simp [hz.1]
                                                 /-
                                                   🎉 no goals
                                                 -/
    _ ≤ ‖F.derivative.eval a‖ ^ 2 * T ^ 2 ^ n / ‖F.derivative.eval a‖ := by
      /-
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        F : Polynomial (PadicInt p)
        a : PadicInt p
        hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
        n : Nat
        z : PadicInt p
        hz : ih_gen n z
        ⊢ LE.le (HDiv.hDiv (Norm.norm (Polynomial.eval z F)) (Norm.norm (Polynomial.ev …
      -/
      gcongr
      /-
        case hab
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        F : Polynomial (PadicInt p)
        a : PadicInt p
        hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
        n : Nat
        z : PadicInt p
        hz : ih_gen n z
        ⊢ LE.le (Norm.norm (Polynomial.eval z F)) (HMul.hMul (HPow.hPow (Norm.norm (Po …
      -/
      apply hz.2
      /-
        🎉 no goals
      -/
    _ = ‖F.derivative.eval a‖ * T ^ 2 ^ n := div_sq_cancel _ _
    _ ≤ 1 := mul_le_one₀ (PadicInt.norm_le_one _) (T_pow_nonneg _) (le_of_lt (T_pow' hnorm _))



private theorem calc_deriv_dist {z z' z1 : ℤ_[p]} (hz' : z' = z - z1)
    (hz1 : ‖z1‖ = ‖F.eval z‖ / ‖F.derivative.eval a‖) {n} (hz : ih n z) :
    ‖F.derivative.eval z' - F.derivative.eval z‖ < ‖F.derivative.eval a‖ :=
  calc
    ‖F.derivative.eval z' - F.derivative.eval z‖ ≤ ‖z' - z‖ := padic_polynomial_dist _ _ _
                   /-
                     p : Nat
                     inst✝ : Fact (Nat.Prime p)
                     F : Polynomial (PadicInt p)
                     a : PadicInt p
                     hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
                     z z' z1 : PadicInt p
                     hz' : Eq z' (HSub.hSub z z1)
                     hz1 : Eq (Norm.norm z1) (HDiv.hDiv (Norm.norm (Polynomial.eval z F)) (Norm.nor …
                     n : Nat
                     hz : ih_gen n z
                     ⊢ Eq (Norm.norm (HSub.hSub z' z)) (Norm.norm z1)
                   -/
    _ = ‖z1‖ := by simp only [sub_eq_add_neg, add_assoc, hz', add_add_neg_cancel'_right, norm_neg]
                   /-
                     🎉 no goals
                   -/
    _ = ‖F.eval z‖ / ‖F.derivative.eval a‖ := hz1
    _ ≤ ‖F.derivative.eval a‖ ^ 2 * T ^ 2 ^ n / ‖F.derivative.eval a‖ := by
      /-
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        F : Polynomial (PadicInt p)
        a : PadicInt p
        hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
        z z' z1 : PadicInt p
        hz' : Eq z' (HSub.hSub z z1)
        hz1 : Eq (Norm.norm z1) (HDiv.hDiv (Norm.norm (Polynomial.eval z F)) (Norm.nor …
        n : Nat
        hz : ih_gen n z
        ⊢ LE.le (HDiv.hDiv (Norm.norm (Polynomial.eval z F)) (Norm.norm (Polynomial.ev …
      -/
      gcongr
      /-
        case hab
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        F : Polynomial (PadicInt p)
        a : PadicInt p
        hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
        z z' z1 : PadicInt p
        hz' : Eq z' (HSub.hSub z z1)
        hz1 : Eq (Norm.norm z1) (HDiv.hDiv (Norm.norm (Polynomial.eval z F)) (Norm.nor …
        n : Nat
        hz : ih_gen n z
        ⊢ LE.le (Norm.norm (Polynomial.eval z F)) (HMul.hMul (HPow.hPow (Norm.norm (Po …
      -/
      apply hz.2
      /-
        🎉 no goals
      -/
    _ = ‖F.derivative.eval a‖ * T ^ 2 ^ n := div_sq_cancel _ _
    _ < ‖F.derivative.eval a‖ := (mul_lt_iff_lt_one_right (deriv_norm_pos hnorm)).2
      (T_pow' hnorm _)



private def calc_eval_z' {z z' z1 : ℤ_[p]} (hz' : z' = z - z1) {n} (hz : ih n z)
    (h1 : ‖(↑(F.eval z) : ℚ_[p]) / ↑(F.derivative.eval z)‖ ≤ 1) (hzeq : z1 = ⟨_, h1⟩) :
    { q : ℤ_[p] // F.eval z' = q * z1 ^ 2 } := by
  have hdzne : F.derivative.eval z ≠ 0 :=
    mt norm_eq_zero.2 (by rw [hz.1]; apply deriv_norm_ne_zero; assumption)
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    F : Polynomial (PadicInt p)
    a : PadicInt p
    hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
    z z' z1 : PadicInt p
    hz' : Eq z' (HSub.hSub z z1)
    n : Nat
    hz : ih_gen n z
    h1 : LE.le (Norm.norm (HDiv.hDiv ↑(Polynomial.eval z F) ↑(Polynomial.eval z (P …
    hzeq : Eq z1 ⟨HDiv.hDiv ↑(Polynomial.eval z F) ↑(Polynomial.eval z (Polynomial …
    hdzne : Ne (Polynomial.eval z (Polynomial.derivative F)) 0
    ⊢ Subtype fun q => Eq (Polynomial.eval z' F) (HMul.hMul q (HPow.hPow z1 2))
  -/
  have hdzne' : (↑(F.derivative.eval z) : ℚ_[p]) ≠ 0 := fun h => hdzne (Subtype.ext_iff_val.2 h)
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    F : Polynomial (PadicInt p)
    a : PadicInt p
    hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
    z z' z1 : PadicInt p
    hz' : Eq z' (HSub.hSub z z1)
    n : Nat
    hz : ih_gen n z
    h1 : LE.le (Norm.norm (HDiv.hDiv ↑(Polynomial.eval z F) ↑(Polynomial.eval z (P …
    hzeq : Eq z1 ⟨HDiv.hDiv ↑(Polynomial.eval z F) ↑(Polynomial.eval z (Polynomial …
    hdzne : Ne (Polynomial.eval z (Polynomial.derivative F)) 0
    hdzne' : Ne (↑(Polynomial.eval z (Polynomial.derivative F))) 0
    ⊢ Subtype fun q => Eq (Polynomial.eval z' F) (HMul.hMul q (HPow.hPow z1 2))
  -/
  obtain ⟨q, hq⟩ := F.binomExpansion z (-z1)
  have : ‖(↑(F.derivative.eval z) * (↑(F.eval z) / ↑(F.derivative.eval z)) : ℚ_[p])‖ ≤ 1 := by
    rw [padicNormE.mul]
    exact mul_le_one₀ (PadicInt.norm_le_one _) (norm_nonneg _) h1
  have : F.derivative.eval z * -z1 = -F.eval z := by
    calc
      F.derivative.eval z * -z1 =
          F.derivative.eval z * -⟨↑(F.eval z) / ↑(F.derivative.eval z), h1⟩ := by rw [hzeq]
      _ = -(F.derivative.eval z * ⟨↑(F.eval z) / ↑(F.derivative.eval z), h1⟩) := mul_neg _ _
      _ = -⟨F.derivative.eval z * (F.eval z / (F.derivative.eval z : ℤ_[p]) : ℚ_[p]), this⟩ :=
        (Subtype.ext <| by simp only [PadicInt.coe_neg, PadicInt.coe_mul, Subtype.coe_mk])
      _ = -F.eval z := by simp only [mul_div_cancel₀ _ hdzne', Subtype.coe_eta]

  /-
    case mk
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    F : Polynomial (PadicInt p)
    a : PadicInt p
    hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
    z z' z1 : PadicInt p
    hz' : Eq z' (HSub.hSub z z1)
    n : Nat
    hz : ih_gen n z
    h1 : LE.le (Norm.norm (HDiv.hDiv ↑(Polynomial.eval z F) ↑(Polynomial.eval z (P …
    hzeq : Eq z1 ⟨HDiv.hDiv ↑(Polynomial.eval z F) ↑(Polynomial.eval z (Polynomial …
    hdzne : Ne (Polynomial.eval z (Polynomial.derivative F)) 0
    hdzne' : Ne (↑(Polynomial.eval z (Polynomial.derivative F))) 0
    q : PadicInt p
    hq : Eq (Polynomial.eval (HAdd.hAdd z (Neg.neg z1)) F) (HAdd.hAdd (HAdd.hAdd ( …
    this✝ : LE.le (Norm.norm (HMul.hMul (↑(Polynomial.eval z (Polynomial.derivativ …
    this : Eq (HMul.hMul (Polynomial.eval z (Polynomial.derivative F)) (Neg.neg z1 …
    ⊢ Subtype fun q => Eq (Polynomial.eval z' F) (HMul.hMul q (HPow.hPow z1 2))
  -/
  exact ⟨q, by simpa only [sub_eq_add_neg, this, hz', add_neg_cancel, neg_sq, zero_add] using hq⟩
  /-
    🎉 no goals
  -/


private def calc_eval_z'_norm {z z' z1 : ℤ_[p]} {n} (hz : ih n z) {q} (heq : F.eval z' = q * z1 ^ 2)
    (h1 : ‖(↑(F.eval z) : ℚ_[p]) / ↑(F.derivative.eval z)‖ ≤ 1) (hzeq : z1 = ⟨_, h1⟩) :
    ‖F.eval z'‖ ≤ ‖F.derivative.eval a‖ ^ 2 * T ^ 2 ^ (n + 1) := by
  calc
    ‖F.eval z'‖ = ‖q‖ * ‖z1‖ ^ 2 := by simp [heq]
    _ ≤ 1 * ‖z1‖ ^ 2 := by gcongr; apply PadicInt.norm_le_one
    _ = ‖F.eval z‖ ^ 2 / ‖F.derivative.eval a‖ ^ 2 := by simp [hzeq, hz.1, div_pow]
    _ ≤ (‖F.derivative.eval a‖ ^ 2 * T ^ 2 ^ n) ^ 2 / ‖F.derivative.eval a‖ ^ 2 := by
      gcongr
      exact hz.2
    _ = (‖F.derivative.eval a‖ ^ 2) ^ 2 * (T ^ 2 ^ n) ^ 2 / ‖F.derivative.eval a‖ ^ 2 := by
      simp only [mul_pow]
    _ = ‖F.derivative.eval a‖ ^ 2 * (T ^ 2 ^ n) ^ 2 := div_sq_cancel _ _
    _ = ‖F.derivative.eval a‖ ^ 2 * T ^ 2 ^ (n + 1) := by rw [← pow_mul, pow_succ 2]


-- Porting note: unsupported option eqn_compiler.zeta
-- set_option eqn_compiler.zeta true


/-- Given `z : ℤ_[p]` satisfying `ih n z`, construct `z' : ℤ_[p]` satisfying `ih (n+1) z'`. We need
the hypothesis `ih n z`, since otherwise `z'` is not necessarily an integer. -/
private def ih_n {n : ℕ} {z : ℤ_[p]} (hz : ih n z) : { z' : ℤ_[p] // ih (n + 1) z' } :=
  have h1 : ‖(↑(F.eval z) : ℚ_[p]) / ↑(F.derivative.eval z)‖ ≤ 1 := calc_norm_le_one hnorm hz
  let z1 : ℤ_[p] := ⟨_, h1⟩
  let z' : ℤ_[p] := z - z1
  ⟨z',
    have hdist : ‖F.derivative.eval z' - F.derivative.eval z‖ < ‖F.derivative.eval a‖ :=
                                    /-
                                      p : Nat
                                      inst✝ : Fact (Nat.Prime p)
                                      F : Polynomial (PadicInt p)
                                      a : PadicInt p
                                      hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
                                      n : Nat
                                      z : PadicInt p
                                      hz : ih_gen n z
                                      h1 : LE.le (Norm.norm (HDiv.hDiv ↑(Polynomial.eval z F) ↑(Polynomial.eval z (P …
                                      z1 : PadicInt p := ⟨HDiv.hDiv ↑(Polynomial.eval z F) ↑(Polynomial.eval z (Poly …
                                      z' : PadicInt p := HSub.hSub z z1
                                      ⊢ Eq (Norm.norm z1) (HDiv.hDiv (Norm.norm (Polynomial.eval z F)) (Norm.norm (P …
                                    -/
      calc_deriv_dist hnorm rfl (by simp [z1, hz.1]) hz
                                    /-
                                      🎉 no goals
                                    -/
    have hfeq : ‖F.derivative.eval z'‖ = ‖F.derivative.eval a‖ := by
      /-
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        F : Polynomial (PadicInt p)
        a : PadicInt p
        hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
        n : Nat
        z : PadicInt p
        hz : ih_gen n z
        h1 : LE.le (Norm.norm (HDiv.hDiv ↑(Polynomial.eval z F) ↑(Polynomial.eval z (P …
        z1 : PadicInt p := ⟨HDiv.hDiv ↑(Polynomial.eval z F) ↑(Polynomial.eval z (Poly …
        z' : PadicInt p := HSub.hSub z z1
        hdist : LT.lt (Norm.norm (HSub.hSub (Polynomial.eval z' (Polynomial.derivative …
        ⊢ Eq (Norm.norm (Polynomial.eval z' (Polynomial.derivative F))) (Norm.norm (Po …
      -/
      rw [sub_eq_add_neg, ← hz.1, ← norm_neg (F.derivative.eval z)] at hdist
      /-
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        F : Polynomial (PadicInt p)
        a : PadicInt p
        hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
        n : Nat
        z : PadicInt p
        hz : ih_gen n z
        h1 : LE.le (Norm.norm (HDiv.hDiv ↑(Polynomial.eval z F) ↑(Polynomial.eval z (P …
        z1 : PadicInt p := ⟨HDiv.hDiv ↑(Polynomial.eval z F) ↑(Polynomial.eval z (Poly …
        z' : PadicInt p := HSub.hSub z z1
        hdist : LT.lt (Norm.norm (HAdd.hAdd (Polynomial.eval z' (Polynomial.derivative …
        ⊢ Eq (Norm.norm (Polynomial.eval z' (Polynomial.derivative F))) (Norm.norm (Po …
      -/
      have := PadicInt.norm_eq_of_norm_add_lt_right hdist
      /-
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        F : Polynomial (PadicInt p)
        a : PadicInt p
        hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
        n : Nat
        z : PadicInt p
        hz : ih_gen n z
        h1 : LE.le (Norm.norm (HDiv.hDiv ↑(Polynomial.eval z F) ↑(Polynomial.eval z (P …
        z1 : PadicInt p := ⟨HDiv.hDiv ↑(Polynomial.eval z F) ↑(Polynomial.eval z (Poly …
        z' : PadicInt p := HSub.hSub z z1
        hdist : LT.lt (Norm.norm (HAdd.hAdd (Polynomial.eval z' (Polynomial.derivative …
        this : Eq (Norm.norm (Polynomial.eval z' (Polynomial.derivative F))) (Norm.nor …
        ⊢ Eq (Norm.norm (Polynomial.eval z' (Polynomial.derivative F))) (Norm.norm (Po …
      -/
      rwa [norm_neg, hz.1] at this
      /-
        🎉 no goals
      -/
    let ⟨_, heq⟩ := calc_eval_z' hnorm rfl hz h1 rfl
    have hnle : ‖F.eval z'‖ ≤ ‖F.derivative.eval a‖ ^ 2 * T ^ 2 ^ (n + 1) :=
      calc_eval_z'_norm hz heq h1 rfl
    ⟨hfeq, hnle⟩⟩

-- Porting note: unsupported option eqn_compiler.zeta
-- set_option eqn_compiler.zeta false


private def newton_seq_aux : ∀ n : ℕ, { z : ℤ_[p] // ih n z }
  | 0 => ⟨a, ih_0 hnorm⟩
  | k + 1 => ih_n hnorm (newton_seq_aux k).2

-- Porting note: renamed this `def` and used a local notation to provide arguments automatically

private def newton_seq_gen (n : ℕ) : ℤ_[p] :=
  (newton_seq_aux hnorm n).1


local notation "newton_seq" => newton_seq_gen hnorm


private theorem newton_seq_deriv_norm (n : ℕ) :
    ‖F.derivative.eval (newton_seq n)‖ = ‖F.derivative.eval a‖ :=
  (newton_seq_aux hnorm n).2.1


private theorem newton_seq_norm_le (n : ℕ) :
    ‖F.eval (newton_seq n)‖ ≤ ‖F.derivative.eval a‖ ^ 2 * T ^ 2 ^ n :=
  (newton_seq_aux hnorm n).2.2


private theorem newton_seq_norm_eq (n : ℕ) :
    ‖newton_seq (n + 1) - newton_seq n‖ =
    ‖F.eval (newton_seq n)‖ / ‖F.derivative.eval (newton_seq n)‖ := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    F : Polynomial (PadicInt p)
    a : PadicInt p
    hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
    n : Nat
    ⊢ Eq (Norm.norm (HSub.hSub (newton_seq_gen hnorm (HAdd.hAdd n 1)) (newton_seq_ …
  -/
  rw [newton_seq_gen, newton_seq_gen, newton_seq_aux, ih_n]
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    F : Polynomial (PadicInt p)
    a : PadicInt p
    hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
    n : Nat
    ⊢ Eq (Norm.norm (HSub.hSub ↑⟨HSub.hSub ↑(newton_seq_aux hnorm n) ⟨HDiv.hDiv ↑( …
  -/
  simp [sub_eq_add_neg, add_comm]
  /-
    🎉 no goals
  -/


private theorem newton_seq_succ_dist (n : ℕ) :
    ‖newton_seq (n + 1) - newton_seq n‖ ≤ ‖F.derivative.eval a‖ * T ^ 2 ^ n :=
  calc
    ‖newton_seq (n + 1) - newton_seq n‖ =
        ‖F.eval (newton_seq n)‖ / ‖F.derivative.eval (newton_seq n)‖ :=
      newton_seq_norm_eq hnorm _
                                                              /-
                                                                p : Nat
                                                                inst✝ : Fact (Nat.Prime p)
                                                                F : Polynomial (PadicInt p)
                                                                a : PadicInt p
                                                                hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
                                                                n : Nat
                                                                ⊢ Eq (HDiv.hDiv (Norm.norm (Polynomial.eval (newton_seq_gen hnorm n) F)) (Norm …
                                                              -/
    _ = ‖F.eval (newton_seq n)‖ / ‖F.derivative.eval a‖ := by rw [newton_seq_deriv_norm]
                                                              /-
                                                                🎉 no goals
                                                              -/
    _ ≤ ‖F.derivative.eval a‖ ^ 2 * T ^ 2 ^ n / ‖F.derivative.eval a‖ :=
      ((div_le_div_iff_of_pos_right (deriv_norm_pos hnorm)).2 (newton_seq_norm_le hnorm _))
    _ = ‖F.derivative.eval a‖ * T ^ 2 ^ n := div_sq_cancel _ _


private theorem newton_seq_dist_aux (n : ℕ) :
    ∀ k : ℕ, ‖newton_seq (n + k) - newton_seq n‖ ≤ ‖F.derivative.eval a‖ * T ^ 2 ^ n
            /-
              p : Nat
              inst✝ : Fact (Nat.Prime p)
              F : Polynomial (PadicInt p)
              a : PadicInt p
              hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
              n : Nat
              ⊢ LE.le (Norm.norm (HSub.hSub (newton_seq_gen hnorm (HAdd.hAdd n 0)) (newton_s …
            -/
  | 0 => by simp [T_pow_nonneg, mul_nonneg]
            /-
              🎉 no goals
            -/
  | k + 1 =>
    have : 2 ^ n ≤ 2 ^ (n + k) := by
      /-
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        F : Polynomial (PadicInt p)
        a : PadicInt p
        hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
        n k : Nat
        ⊢ LE.le (HPow.hPow 2 n) (HPow.hPow 2 (HAdd.hAdd n k))
      -/
      apply pow_right_mono₀
        /-
          case h
          p : Nat
          inst✝ : Fact (Nat.Prime p)
          F : Polynomial (PadicInt p)
          a : PadicInt p
          hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
          n k : Nat
          ⊢ LE.le 1 2
        -/
      · norm_num
        /-
          🎉 no goals
        -/
        /-
          case a
          p : Nat
          inst✝ : Fact (Nat.Prime p)
          F : Polynomial (PadicInt p)
          a : PadicInt p
          hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
          n k : Nat
          ⊢ LE.le n (HAdd.hAdd n k)
        -/
      · apply Nat.le_add_right
        /-
          🎉 no goals
        -/
    calc
      ‖newton_seq (n + (k + 1)) - newton_seq n‖ = ‖newton_seq (n + k + 1) - newton_seq n‖ := by
        /-
          p : Nat
          inst✝ : Fact (Nat.Prime p)
          F : Polynomial (PadicInt p)
          a : PadicInt p
          hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
          n k : Nat
          this : LE.le (HPow.hPow 2 n) (HPow.hPow 2 (HAdd.hAdd n k))
          ⊢ Eq (Norm.norm (HSub.hSub (newton_seq_gen hnorm (HAdd.hAdd n (HAdd.hAdd k 1)) …
        -/
        rw [add_assoc]
        /-
          🎉 no goals
        -/
      _ = ‖newton_seq (n + k + 1) - newton_seq (n + k) + (newton_seq (n + k) - newton_seq n)‖ := by
        /-
          p : Nat
          inst✝ : Fact (Nat.Prime p)
          F : Polynomial (PadicInt p)
          a : PadicInt p
          hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
          n k : Nat
          this : LE.le (HPow.hPow 2 n) (HPow.hPow 2 (HAdd.hAdd n k))
          ⊢ Eq (Norm.norm (HSub.hSub (newton_seq_gen hnorm (HAdd.hAdd (HAdd.hAdd n k) 1) …
        -/
        rw [← sub_add_sub_cancel]
        /-
          🎉 no goals
        -/
      _ ≤ max ‖newton_seq (n + k + 1) - newton_seq (n + k)‖ ‖newton_seq (n + k) - newton_seq n‖ :=
        (PadicInt.nonarchimedean _ _)
      _ ≤ max (‖F.derivative.eval a‖ * T ^ 2 ^ (n + k)) (‖F.derivative.eval a‖ * T ^ 2 ^ n) :=
        (max_le_max (newton_seq_succ_dist _ _) (newton_seq_dist_aux _ _))
      _ = ‖F.derivative.eval a‖ * T ^ 2 ^ n :=
        max_eq_right <|
          mul_le_mul_of_nonneg_left (pow_le_pow_of_le_one (norm_nonneg _)
            (le_of_lt (T_lt_one hnorm)) this) (norm_nonneg _)


private theorem newton_seq_dist {n k : ℕ} (hnk : n ≤ k) :
    ‖newton_seq k - newton_seq n‖ ≤ ‖F.derivative.eval a‖ * T ^ 2 ^ n := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    F : Polynomial (PadicInt p)
    a : PadicInt p
    hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
    n k : Nat
    hnk : LE.le n k
    ⊢ LE.le (Norm.norm (HSub.hSub (newton_seq_gen hnorm k) (newton_seq_gen hnorm n …
  -/
  have hex : ∃ m, k = n + m := Nat.exists_eq_add_of_le hnk
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    F : Polynomial (PadicInt p)
    a : PadicInt p
    hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
    n k : Nat
    hnk : LE.le n k
    hex : Exists fun m => Eq k (HAdd.hAdd n m)
    ⊢ LE.le (Norm.norm (HSub.hSub (newton_seq_gen hnorm k) (newton_seq_gen hnorm n …
  -/
  let ⟨_, hex'⟩ := hex
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    F : Polynomial (PadicInt p)
    a : PadicInt p
    hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
    n k : Nat
    hnk : LE.le n k
    hex : Exists fun m => Eq k (HAdd.hAdd n m)
    w✝ : Nat
    hex' : Eq k (HAdd.hAdd n w✝)
    ⊢ LE.le (Norm.norm (HSub.hSub (newton_seq_gen hnorm k) (newton_seq_gen hnorm n …
  -/
  rw [hex']; apply newton_seq_dist_aux
             /-
               🎉 no goals
             -/


private theorem bound' : Tendsto (fun n : ℕ => ‖F.derivative.eval a‖ * T ^ 2 ^ n) atTop (𝓝 0) := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    F : Polynomial (PadicInt p)
    a : PadicInt p
    hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
    ⊢ Filter.Tendsto (fun n => HMul.hMul (Norm.norm (Polynomial.eval a (Polynomial …
  -/
  rw [← mul_zero ‖F.derivative.eval a‖]
  exact
    tendsto_const_nhds.mul
      (Tendsto.comp (tendsto_pow_atTop_nhds_zero_of_lt_one (norm_nonneg _) (T_lt_one hnorm))
        (Nat.tendsto_pow_atTop_atTop_of_one_lt (by norm_num)))


private theorem bound :
    ∀ {ε}, ε > 0 → ∃ N : ℕ, ∀ {n}, n ≥ N → ‖F.derivative.eval a‖ * T ^ 2 ^ n < ε := fun hε ↦
  eventually_atTop.1 <| (bound' hnorm).eventually <| gt_mem_nhds hε


private theorem bound'_sq :
    Tendsto (fun n : ℕ => ‖F.derivative.eval a‖ ^ 2 * T ^ 2 ^ n) atTop (𝓝 0) := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    F : Polynomial (PadicInt p)
    a : PadicInt p
    hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
    ⊢ Filter.Tendsto (fun n => HMul.hMul (HPow.hPow (Norm.norm (Polynomial.eval a  …
  -/
  rw [← mul_zero ‖F.derivative.eval a‖, sq]
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    F : Polynomial (PadicInt p)
    a : PadicInt p
    hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
    ⊢ Filter.Tendsto (fun n => HMul.hMul (HMul.hMul (Norm.norm (Polynomial.eval a  …
  -/
  simp only [mul_assoc]
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    F : Polynomial (PadicInt p)
    a : PadicInt p
    hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
    ⊢ Filter.Tendsto (fun n => HMul.hMul (Norm.norm (Polynomial.eval a (Polynomial …
  -/
  apply Tendsto.mul
    /-
      case hf
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      F : Polynomial (PadicInt p)
      a : PadicInt p
      hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
      ⊢ Filter.Tendsto (fun x => Norm.norm (Polynomial.eval a (Polynomial.derivative …
    -/
  · apply tendsto_const_nhds
    /-
      🎉 no goals
    -/
    /-
      case hg
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      F : Polynomial (PadicInt p)
      a : PadicInt p
      hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
      ⊢ Filter.Tendsto (fun x => HMul.hMul (Norm.norm (Polynomial.eval a (Polynomial …
    -/
  · apply bound'
    /-
      case hg.hnorm
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      F : Polynomial (PadicInt p)
      a : PadicInt p
      hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
      ⊢ LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynomial.ev …
    -/
    assumption
    /-
      🎉 no goals
    -/


private theorem newton_seq_is_cauchy : IsCauSeq norm newton_seq := fun _ε hε ↦
  (bound hnorm hε).imp fun _N hN _j hj ↦ (newton_seq_dist hnorm hj).trans_lt <| hN le_rfl


private def newton_cau_seq : CauSeq ℤ_[p] norm := ⟨_, newton_seq_is_cauchy hnorm⟩

-- Porting note: renamed this `def` and used a local notation to provide arguments automatically

private def soln_gen : ℤ_[p] := (newton_cau_seq hnorm).lim


local notation "soln" => soln_gen hnorm


private theorem soln_spec {ε : ℝ} (hε : ε > 0) :
    ∃ N : ℕ, ∀ {i : ℕ}, i ≥ N → ‖soln - newton_cau_seq hnorm i‖ < ε :=
  Setoid.symm (CauSeq.equiv_lim (newton_cau_seq hnorm)) _ hε


private theorem soln_deriv_norm : ‖F.derivative.eval soln‖ = ‖F.derivative.eval a‖ :=
  norm_deriv_eq (newton_seq_deriv_norm hnorm)


private theorem newton_seq_norm_tendsto_zero :
    Tendsto (fun i => ‖F.eval (newton_cau_seq hnorm i)‖) atTop (𝓝 0) :=
  squeeze_zero (fun _ => norm_nonneg _) (newton_seq_norm_le hnorm) (bound'_sq hnorm)


private theorem newton_seq_dist_tendsto' :
    Tendsto (fun n => ‖newton_cau_seq hnorm n - a‖) atTop (𝓝 ‖soln - a‖) :=
  (continuous_norm.tendsto _).comp ((newton_cau_seq hnorm).tendsto_limit.sub tendsto_const_nhds)


private theorem eval_soln : F.eval soln = 0 :=
  limit_zero_of_norm_tendsto_zero (newton_seq_norm_tendsto_zero hnorm)


private theorem T_pos : T > 0 := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    F : Polynomial (PadicInt p)
    a : PadicInt p
    hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
    hnsol : Ne (Polynomial.eval a F) 0
    ⊢ GT.gt (T_gen p F a) 0
  -/
  rw [T_def]
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    F : Polynomial (PadicInt p)
    a : PadicInt p
    hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
    hnsol : Ne (Polynomial.eval a F) 0
    ⊢ GT.gt (HDiv.hDiv (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Po …
  -/
  exact div_pos (norm_pos_iff.2 hnsol) (deriv_sq_norm_pos hnorm)
  /-
    🎉 no goals
  -/


private theorem newton_seq_succ_dist_weak (n : ℕ) :
    ‖newton_seq (n + 2) - newton_seq (n + 1)‖ < ‖F.eval a‖ / ‖F.derivative.eval a‖ :=
  have : 2 ≤ 2 ^ (n + 1) := by
    /-
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      F : Polynomial (PadicInt p)
      a : PadicInt p
      hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
      hnsol : Ne (Polynomial.eval a F) 0
      n : Nat
      ⊢ LE.le 2 (HPow.hPow 2 (HAdd.hAdd n 1))
    -/
    have := pow_right_mono₀ (by norm_num : 1 ≤ 2) (Nat.le_add_left _ _ : 1 ≤ n + 1)
    /-
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      F : Polynomial (PadicInt p)
      a : PadicInt p
      hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
      hnsol : Ne (Polynomial.eval a F) 0
      n : Nat
      this : LE.le ((fun x => HPow.hPow 2 x) 1) ((fun x => HPow.hPow 2 x) (HAdd.hAdd …
      ⊢ LE.le 2 (HPow.hPow 2 (HAdd.hAdd n 1))
    -/
    simpa using this
    /-
      🎉 no goals
    -/
  calc
    ‖newton_seq (n + 2) - newton_seq (n + 1)‖ ≤ ‖F.derivative.eval a‖ * T ^ 2 ^ (n + 1) :=
      newton_seq_succ_dist hnorm _
    _ ≤ ‖F.derivative.eval a‖ * T ^ 2 :=
      (mul_le_mul_of_nonneg_left (pow_le_pow_of_le_one (norm_nonneg _)
        (le_of_lt (T_lt_one hnorm)) this) (norm_nonneg _))
    _ < ‖F.derivative.eval a‖ * T ^ 1 :=
      (mul_lt_mul_of_pos_left (pow_lt_pow_right_of_lt_one₀ (T_pos hnorm hnsol)
                             /-
                               p : Nat
                               inst✝ : Fact (Nat.Prime p)
                               F : Polynomial (PadicInt p)
                               a : PadicInt p
                               hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
                               hnsol : Ne (Polynomial.eval a F) 0
                               n : Nat
                               this : LE.le 2 (HPow.hPow 2 (HAdd.hAdd n 1))
                               ⊢ LT.lt 1 2
                             -/
        (T_lt_one hnorm) (by norm_num)) (deriv_norm_pos hnorm))
                             /-
                               🎉 no goals
                             -/
    _ = ‖F.eval a‖ / ‖F.derivative.eval a‖ := by
      rw [T_gen, sq, pow_one, norm_div, ← mul_div_assoc, PadicInt.padic_norm_e_of_padicInt,
        PadicInt.coe_mul, padicNormE.mul]
      /-
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        F : Polynomial (PadicInt p)
        a : PadicInt p
        hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
        hnsol : Ne (Polynomial.eval a F) 0
        n : Nat
        this : LE.le 2 (HPow.hPow 2 (HAdd.hAdd n 1))
        ⊢ Eq (HDiv.hDiv (HMul.hMul (Norm.norm (Polynomial.eval a (Polynomial.derivativ …
      -/
      apply mul_div_mul_left
      /-
        case hc
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        F : Polynomial (PadicInt p)
        a : PadicInt p
        hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
        hnsol : Ne (Polynomial.eval a F) 0
        n : Nat
        this : LE.le 2 (HPow.hPow 2 (HAdd.hAdd n 1))
        ⊢ Ne (Norm.norm (Polynomial.eval a (Polynomial.derivative F))) 0
      -/
      apply deriv_norm_ne_zero; assumption
                                /-
                                  🎉 no goals
                                -/


private theorem newton_seq_dist_to_a :
    ∀ n : ℕ, 0 < n → ‖newton_seq n - a‖ = ‖F.eval a‖ / ‖F.derivative.eval a‖
                /-
                  p : Nat
                  inst✝ : Fact (Nat.Prime p)
                  F : Polynomial (PadicInt p)
                  a : PadicInt p
                  hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
                  hnsol : Ne (Polynomial.eval a F) 0
                  _h : LT.lt 0 1
                  ⊢ Eq (Norm.norm (HSub.hSub (newton_seq_gen hnorm 1) a)) (HDiv.hDiv (Norm.norm  …
                -/
  | 1, _h => by simp [sub_eq_add_neg, add_assoc, newton_seq_gen, newton_seq_aux, ih_n]
                /-
                  🎉 no goals
                -/
  | k + 2, _h =>
    have hlt : ‖newton_seq (k + 2) - newton_seq (k + 1)‖ < ‖newton_seq (k + 1) - a‖ := by
      /-
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        F : Polynomial (PadicInt p)
        a : PadicInt p
        hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
        hnsol : Ne (Polynomial.eval a F) 0
        k : Nat
        _h : LT.lt 0 (HAdd.hAdd k 2)
        ⊢ LT.lt (Norm.norm (HSub.hSub (newton_seq_gen hnorm (HAdd.hAdd k 2)) (newton_s …
      -/
      rw [newton_seq_dist_to_a (k + 1) (succ_pos _)]; apply newton_seq_succ_dist_weak
      /-
        case hnsol
        p : Nat
        inst✝ : Fact (Nat.Prime p)
        F : Polynomial (PadicInt p)
        a : PadicInt p
        hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
        hnsol : Ne (Polynomial.eval a F) 0
        k : Nat
        _h : LT.lt 0 (HAdd.hAdd k 2)
        ⊢ Ne (Polynomial.eval a F) 0
      -/
      assumption
      /-
        🎉 no goals
      -/
    have hne' : ‖newton_seq (k + 2) - newton_seq (k + 1)‖ ≠ ‖newton_seq (k + 1) - a‖ := ne_of_lt hlt
    calc
      ‖newton_seq (k + 2) - a‖ =
          ‖newton_seq (k + 2) - newton_seq (k + 1) + (newton_seq (k + 1) - a)‖ := by
        /-
          p : Nat
          inst✝ : Fact (Nat.Prime p)
          F : Polynomial (PadicInt p)
          a : PadicInt p
          hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
          hnsol : Ne (Polynomial.eval a F) 0
          k : Nat
          _h : LT.lt 0 (HAdd.hAdd k 2)
          hlt : LT.lt (Norm.norm (HSub.hSub (newton_seq_gen hnorm (HAdd.hAdd k 2)) (newt …
          hne' : Ne (Norm.norm (HSub.hSub (newton_seq_gen hnorm (HAdd.hAdd k 2)) (newton …
          ⊢ Eq (Norm.norm (HSub.hSub (newton_seq_gen hnorm (HAdd.hAdd k 2)) a)) (Norm.no …
        -/
        rw [← sub_add_sub_cancel]
        /-
          🎉 no goals
        -/
      _ = max ‖newton_seq (k + 2) - newton_seq (k + 1)‖ ‖newton_seq (k + 1) - a‖ :=
        (PadicInt.norm_add_eq_max_of_ne hne')
      _ = ‖newton_seq (k + 1) - a‖ := max_eq_right_of_lt hlt
      _ = ‖Polynomial.eval a F‖ / ‖Polynomial.eval a (Polynomial.derivative F)‖ :=
        newton_seq_dist_to_a (k + 1) (succ_pos _)


private theorem newton_seq_dist_tendsto :
    Tendsto (fun n => ‖newton_cau_seq hnorm n - a‖)
    atTop (𝓝 (‖F.eval a‖ / ‖F.derivative.eval a‖)) :=
  tendsto_const_nhds.congr' (eventually_atTop.2
    ⟨1, fun _ hx => (newton_seq_dist_to_a hnorm hnsol _ hx).symm⟩)


private theorem soln_dist_to_a : ‖soln - a‖ = ‖F.eval a‖ / ‖F.derivative.eval a‖ :=
  tendsto_nhds_unique (newton_seq_dist_tendsto' hnorm) (newton_seq_dist_tendsto hnorm hnsol)


private theorem soln_dist_to_a_lt_deriv : ‖soln - a‖ < ‖F.derivative.eval a‖ := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    F : Polynomial (PadicInt p)
    a : PadicInt p
    hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
    hnsol : Ne (Polynomial.eval a F) 0
    ⊢ LT.lt (Norm.norm (HSub.hSub (soln_gen hnorm) a)) (Norm.norm (Polynomial.eval …
  -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
  rw [soln_dist_to_a, div_lt_iff₀ (deriv_norm_pos _), ← sq] <;> assumption
                                                                /-
                                                                  🎉 no goals
                                                                -/


private theorem soln_unique (z : ℤ_[p]) (hev : F.eval z = 0)
    (hnlt : ‖z - a‖ < ‖F.derivative.eval a‖) : z = soln :=
  have soln_dist : ‖z - soln‖ < ‖F.derivative.eval a‖ :=
    calc
                                              /-
                                                p : Nat
                                                inst✝ : Fact (Nat.Prime p)
                                                F : Polynomial (PadicInt p)
                                                a : PadicInt p
                                                hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
                                                hnsol : Ne (Polynomial.eval a F) 0
                                                z : PadicInt p
                                                hev : Eq (Polynomial.eval z F) 0
                                                hnlt : LT.lt (Norm.norm (HSub.hSub z a)) (Norm.norm (Polynomial.eval a (Polyno …
                                                ⊢ Eq (Norm.norm (HSub.hSub z (soln_gen hnorm))) (Norm.norm (HAdd.hAdd (HSub.hS …
                                              -/
      ‖z - soln‖ = ‖z - a + (a - soln)‖ := by rw [sub_add_sub_cancel]
                                              /-
                                                🎉 no goals
                                              -/
      _ ≤ max ‖z - a‖ ‖a - soln‖ := PadicInt.nonarchimedean _ _
      _ < ‖F.derivative.eval a‖ :=
        max_lt hnlt ((norm_sub_rev soln a ▸ (soln_dist_to_a_lt_deriv hnorm)) hnsol)

  let h := z - soln
  let ⟨q, hq⟩ := F.binomExpansion soln h
  have : (F.derivative.eval soln + q * h) * h = 0 :=
    Eq.symm
      (calc
                                    /-
                                      p : Nat
                                      inst✝ : Fact (Nat.Prime p)
                                      F : Polynomial (PadicInt p)
                                      a : PadicInt p
                                      hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
                                      hnsol : Ne (Polynomial.eval a F) 0
                                      z : PadicInt p
                                      hev : Eq (Polynomial.eval z F) 0
                                      hnlt : LT.lt (Norm.norm (HSub.hSub z a)) (Norm.norm (Polynomial.eval a (Polyno …
                                      soln_dist : LT.lt (Norm.norm (HSub.hSub z (soln_gen hnorm))) (Norm.norm (Polyn …
                                      h : PadicInt p := HSub.hSub z (soln_gen hnorm)
                                      q : PadicInt p
                                      hq : Eq (Polynomial.eval (HAdd.hAdd (soln_gen hnorm) h) F) (HAdd.hAdd (HAdd.hA …
                                      ⊢ Eq 0 (Polynomial.eval (HAdd.hAdd (soln_gen hnorm) h) F)
                                    -/
        0 = F.eval (soln + h) := by simp [h, hev]
                                    /-
                                      🎉 no goals
                                    -/
                                                         /-
                                                           p : Nat
                                                           inst✝ : Fact (Nat.Prime p)
                                                           F : Polynomial (PadicInt p)
                                                           a : PadicInt p
                                                           hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
                                                           hnsol : Ne (Polynomial.eval a F) 0
                                                           z : PadicInt p
                                                           hev : Eq (Polynomial.eval z F) 0
                                                           hnlt : LT.lt (Norm.norm (HSub.hSub z a)) (Norm.norm (Polynomial.eval a (Polyno …
                                                           soln_dist : LT.lt (Norm.norm (HSub.hSub z (soln_gen hnorm))) (Norm.norm (Polyn …
                                                           h : PadicInt p := HSub.hSub z (soln_gen hnorm)
                                                           q : PadicInt p
                                                           hq : Eq (Polynomial.eval (HAdd.hAdd (soln_gen hnorm) h) F) (HAdd.hAdd (HAdd.hA …
                                                           ⊢ Eq (Polynomial.eval (HAdd.hAdd (soln_gen hnorm) h) F) (HAdd.hAdd (HMul.hMul  …
                                                         -/
        _ = F.derivative.eval soln * h + q * h ^ 2 := by rw [hq, eval_soln, zero_add]
                                                         /-
                                                           🎉 no goals
                                                         -/
                                                       /-
                                                         p : Nat
                                                         inst✝ : Fact (Nat.Prime p)
                                                         F : Polynomial (PadicInt p)
                                                         a : PadicInt p
                                                         hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
                                                         hnsol : Ne (Polynomial.eval a F) 0
                                                         z : PadicInt p
                                                         hev : Eq (Polynomial.eval z F) 0
                                                         hnlt : LT.lt (Norm.norm (HSub.hSub z a)) (Norm.norm (Polynomial.eval a (Polyno …
                                                         soln_dist : LT.lt (Norm.norm (HSub.hSub z (soln_gen hnorm))) (Norm.norm (Polyn …
                                                         h : PadicInt p := HSub.hSub z (soln_gen hnorm)
                                                         q : PadicInt p
                                                         hq : Eq (Polynomial.eval (HAdd.hAdd (soln_gen hnorm) h) F) (HAdd.hAdd (HAdd.hA …
                                                         ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.eval (soln_gen hnorm) (Polynomial.deriv …
                                                       -/
        _ = (F.derivative.eval soln + q * h) * h := by rw [sq, right_distrib, mul_assoc]
                                                       /-
                                                         🎉 no goals
                                                       -/
        )
  have : h = 0 :=
    by_contra fun hne =>
      have : F.derivative.eval soln + q * h = 0 :=
        (eq_zero_or_eq_zero_of_mul_eq_zero this).resolve_right hne
                                                   /-
                                                     p : Nat
                                                     inst✝ : Fact (Nat.Prime p)
                                                     F : Polynomial (PadicInt p)
                                                     a : PadicInt p
                                                     hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
                                                     hnsol : Ne (Polynomial.eval a F) 0
                                                     z : PadicInt p
                                                     hev : Eq (Polynomial.eval z F) 0
                                                     hnlt : LT.lt (Norm.norm (HSub.hSub z a)) (Norm.norm (Polynomial.eval a (Polyno …
                                                     soln_dist : LT.lt (Norm.norm (HSub.hSub z (soln_gen hnorm))) (Norm.norm (Polyn …
                                                     h : PadicInt p := HSub.hSub z (soln_gen hnorm)
                                                     q : PadicInt p
                                                     hq : Eq (Polynomial.eval (HAdd.hAdd (soln_gen hnorm) h) F) (HAdd.hAdd (HAdd.hA …
                                                     this✝ : Eq (HMul.hMul (HAdd.hAdd (Polynomial.eval (soln_gen hnorm) (Polynomial …
                                                     hne : Not (Eq h 0)
                                                     this : Eq (HAdd.hAdd (Polynomial.eval (soln_gen hnorm) (Polynomial.derivative  …
                                                     ⊢ Eq (Polynomial.eval (soln_gen hnorm) (Polynomial.derivative F)) (HMul.hMul ( …
                                                   -/
      have : F.derivative.eval soln = -q * h := by simpa using eq_neg_of_add_eq_zero_left this
                                                   /-
                                                     🎉 no goals
                                                   -/
      lt_irrefl ‖F.derivative.eval soln‖
        (calc
                                                    /-
                                                      p : Nat
                                                      inst✝ : Fact (Nat.Prime p)
                                                      F : Polynomial (PadicInt p)
                                                      a : PadicInt p
                                                      hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
                                                      hnsol : Ne (Polynomial.eval a F) 0
                                                      z : PadicInt p
                                                      hev : Eq (Polynomial.eval z F) 0
                                                      hnlt : LT.lt (Norm.norm (HSub.hSub z a)) (Norm.norm (Polynomial.eval a (Polyno …
                                                      soln_dist : LT.lt (Norm.norm (HSub.hSub z (soln_gen hnorm))) (Norm.norm (Polyn …
                                                      h : PadicInt p := HSub.hSub z (soln_gen hnorm)
                                                      q : PadicInt p
                                                      hq : Eq (Polynomial.eval (HAdd.hAdd (soln_gen hnorm) h) F) (HAdd.hAdd (HAdd.hA …
                                                      this✝¹ : Eq (HMul.hMul (HAdd.hAdd (Polynomial.eval (soln_gen hnorm) (Polynomia …
                                                      hne : Not (Eq h 0)
                                                      this✝ : Eq (HAdd.hAdd (Polynomial.eval (soln_gen hnorm) (Polynomial.derivative …
                                                      this : Eq (Polynomial.eval (soln_gen hnorm) (Polynomial.derivative F)) (HMul.h …
                                                      ⊢ Eq (Norm.norm (Polynomial.eval (soln_gen hnorm) (Polynomial.derivative F)))  …
                                                    -/
          ‖F.derivative.eval soln‖ = ‖-q * h‖ := by rw [this]
                                                    /-
                                                      🎉 no goals
                                                    -/
          _ ≤ 1 * ‖h‖ := by
            /-
              p : Nat
              inst✝ : Fact (Nat.Prime p)
              F : Polynomial (PadicInt p)
              a : PadicInt p
              hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
              hnsol : Ne (Polynomial.eval a F) 0
              z : PadicInt p
              hev : Eq (Polynomial.eval z F) 0
              hnlt : LT.lt (Norm.norm (HSub.hSub z a)) (Norm.norm (Polynomial.eval a (Polyno …
              soln_dist : LT.lt (Norm.norm (HSub.hSub z (soln_gen hnorm))) (Norm.norm (Polyn …
              h : PadicInt p := HSub.hSub z (soln_gen hnorm)
              q : PadicInt p
              hq : Eq (Polynomial.eval (HAdd.hAdd (soln_gen hnorm) h) F) (HAdd.hAdd (HAdd.hA …
              this✝¹ : Eq (HMul.hMul (HAdd.hAdd (Polynomial.eval (soln_gen hnorm) (Polynomia …
              hne : Not (Eq h 0)
              this✝ : Eq (HAdd.hAdd (Polynomial.eval (soln_gen hnorm) (Polynomial.derivative …
              this : Eq (Polynomial.eval (soln_gen hnorm) (Polynomial.derivative F)) (HMul.h …
              ⊢ LE.le (Norm.norm (HMul.hMul (Neg.neg q) h)) (HMul.hMul 1 (Norm.norm h))
            -/
            rw [PadicInt.norm_mul]
            /-
              p : Nat
              inst✝ : Fact (Nat.Prime p)
              F : Polynomial (PadicInt p)
              a : PadicInt p
              hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
              hnsol : Ne (Polynomial.eval a F) 0
              z : PadicInt p
              hev : Eq (Polynomial.eval z F) 0
              hnlt : LT.lt (Norm.norm (HSub.hSub z a)) (Norm.norm (Polynomial.eval a (Polyno …
              soln_dist : LT.lt (Norm.norm (HSub.hSub z (soln_gen hnorm))) (Norm.norm (Polyn …
              h : PadicInt p := HSub.hSub z (soln_gen hnorm)
              q : PadicInt p
              hq : Eq (Polynomial.eval (HAdd.hAdd (soln_gen hnorm) h) F) (HAdd.hAdd (HAdd.hA …
              this✝¹ : Eq (HMul.hMul (HAdd.hAdd (Polynomial.eval (soln_gen hnorm) (Polynomia …
              hne : Not (Eq h 0)
              this✝ : Eq (HAdd.hAdd (Polynomial.eval (soln_gen hnorm) (Polynomial.derivative …
              this : Eq (Polynomial.eval (soln_gen hnorm) (Polynomial.derivative F)) (HMul.h …
              ⊢ LE.le (HMul.hMul (Norm.norm (Neg.neg q)) (Norm.norm h)) (HMul.hMul 1 (Norm.n …
            -/
            exact mul_le_mul_of_nonneg_right (PadicInt.norm_le_one _) (norm_nonneg _)
            /-
              🎉 no goals
            -/
                               /-
                                 p : Nat
                                 inst✝ : Fact (Nat.Prime p)
                                 F : Polynomial (PadicInt p)
                                 a : PadicInt p
                                 hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
                                 hnsol : Ne (Polynomial.eval a F) 0
                                 z : PadicInt p
                                 hev : Eq (Polynomial.eval z F) 0
                                 hnlt : LT.lt (Norm.norm (HSub.hSub z a)) (Norm.norm (Polynomial.eval a (Polyno …
                                 soln_dist : LT.lt (Norm.norm (HSub.hSub z (soln_gen hnorm))) (Norm.norm (Polyn …
                                 h : PadicInt p := HSub.hSub z (soln_gen hnorm)
                                 q : PadicInt p
                                 hq : Eq (Polynomial.eval (HAdd.hAdd (soln_gen hnorm) h) F) (HAdd.hAdd (HAdd.hA …
                                 this✝¹ : Eq (HMul.hMul (HAdd.hAdd (Polynomial.eval (soln_gen hnorm) (Polynomia …
                                 hne : Not (Eq h 0)
                                 this✝ : Eq (HAdd.hAdd (Polynomial.eval (soln_gen hnorm) (Polynomial.derivative …
                                 this : Eq (Polynomial.eval (soln_gen hnorm) (Polynomial.derivative F)) (HMul.h …
                                 ⊢ Eq (HMul.hMul 1 (Norm.norm h)) (Norm.norm (HSub.hSub z (soln_gen hnorm)))
                               -/
          _ = ‖z - soln‖ := by simp [h]
                               /-
                                 🎉 no goals
                               -/
                                             /-
                                               p : Nat
                                               inst✝ : Fact (Nat.Prime p)
                                               F : Polynomial (PadicInt p)
                                               a : PadicInt p
                                               hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
                                               hnsol : Ne (Polynomial.eval a F) 0
                                               z : PadicInt p
                                               hev : Eq (Polynomial.eval z F) 0
                                               hnlt : LT.lt (Norm.norm (HSub.hSub z a)) (Norm.norm (Polynomial.eval a (Polyno …
                                               soln_dist : LT.lt (Norm.norm (HSub.hSub z (soln_gen hnorm))) (Norm.norm (Polyn …
                                               h : PadicInt p := HSub.hSub z (soln_gen hnorm)
                                               q : PadicInt p
                                               hq : Eq (Polynomial.eval (HAdd.hAdd (soln_gen hnorm) h) F) (HAdd.hAdd (HAdd.hA …
                                               this✝¹ : Eq (HMul.hMul (HAdd.hAdd (Polynomial.eval (soln_gen hnorm) (Polynomia …
                                               hne : Not (Eq h 0)
                                               this✝ : Eq (HAdd.hAdd (Polynomial.eval (soln_gen hnorm) (Polynomial.derivative …
                                               this : Eq (Polynomial.eval (soln_gen hnorm) (Polynomial.derivative F)) (HMul.h …
                                               ⊢ LT.lt (Norm.norm (HSub.hSub z (soln_gen hnorm))) (Norm.norm (Polynomial.eval …
                                             -/
          _ < ‖F.derivative.eval soln‖ := by rw [soln_deriv_norm]; apply soln_dist
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
          )
                        /-
                          p : Nat
                          inst✝ : Fact (Nat.Prime p)
                          F : Polynomial (PadicInt p)
                          a : PadicInt p
                          hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
                          hnsol : Ne (Polynomial.eval a F) 0
                          z : PadicInt p
                          hev : Eq (Polynomial.eval z F) 0
                          hnlt : LT.lt (Norm.norm (HSub.hSub z a)) (Norm.norm (Polynomial.eval a (Polyno …
                          soln_dist : LT.lt (Norm.norm (HSub.hSub z (soln_gen hnorm))) (Norm.norm (Polyn …
                          h : PadicInt p := HSub.hSub z (soln_gen hnorm)
                          q : PadicInt p
                          hq : Eq (Polynomial.eval (HAdd.hAdd (soln_gen hnorm) h) F) (HAdd.hAdd (HAdd.hA …
                          this✝ : Eq (HMul.hMul (HAdd.hAdd (Polynomial.eval (soln_gen hnorm) (Polynomial …
                          this : Eq h 0
                          ⊢ Eq (HSub.hSub z (soln_gen hnorm)) 0
                        -/
  eq_of_sub_eq_zero (by rw [← this])
                        /-
                          🎉 no goals
                        -/


private theorem a_soln_is_unique (ha : F.eval a = 0) (z' : ℤ_[p]) (hz' : F.eval z' = 0)
    (hnormz' : ‖z' - a‖ < ‖F.derivative.eval a‖) : z' = a :=
  let h := z' - a
  let ⟨q, hq⟩ := F.binomExpansion a h
  have : (F.derivative.eval a + q * h) * h = 0 :=
    Eq.symm
      (calc
                                                                /-
                                                                  p : Nat
                                                                  inst✝ : Fact (Nat.Prime p)
                                                                  F : Polynomial (PadicInt p)
                                                                  a : PadicInt p
                                                                  ha : Eq (Polynomial.eval a F) 0
                                                                  z' : PadicInt p
                                                                  hz' : Eq (Polynomial.eval z' F) 0
                                                                  hnormz' : LT.lt (Norm.norm (HSub.hSub z' a)) (Norm.norm (Polynomial.eval a (Po …
                                                                  h : PadicInt p := HSub.hSub z' a
                                                                  q : PadicInt p
                                                                  hq : Eq (Polynomial.eval (HAdd.hAdd a h) F) (HAdd.hAdd (HAdd.hAdd (Polynomial. …
                                                                  ⊢ Eq 0 (Polynomial.eval (HAdd.hAdd a (HSub.hSub z' a)) F)
                                                                -/
        0 = F.eval (a + h) := show 0 = F.eval (a + (z' - a)) by rw [add_comm]; simp [hz']
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
                                                      /-
                                                        p : Nat
                                                        inst✝ : Fact (Nat.Prime p)
                                                        F : Polynomial (PadicInt p)
                                                        a : PadicInt p
                                                        ha : Eq (Polynomial.eval a F) 0
                                                        z' : PadicInt p
                                                        hz' : Eq (Polynomial.eval z' F) 0
                                                        hnormz' : LT.lt (Norm.norm (HSub.hSub z' a)) (Norm.norm (Polynomial.eval a (Po …
                                                        h : PadicInt p := HSub.hSub z' a
                                                        q : PadicInt p
                                                        hq : Eq (Polynomial.eval (HAdd.hAdd a h) F) (HAdd.hAdd (HAdd.hAdd (Polynomial. …
                                                        ⊢ Eq (Polynomial.eval (HAdd.hAdd a h) F) (HAdd.hAdd (HMul.hMul (Polynomial.eva …
                                                      -/
        _ = F.derivative.eval a * h + q * h ^ 2 := by rw [hq, ha, zero_add]
                                                      /-
                                                        🎉 no goals
                                                      -/
                                                    /-
                                                      p : Nat
                                                      inst✝ : Fact (Nat.Prime p)
                                                      F : Polynomial (PadicInt p)
                                                      a : PadicInt p
                                                      ha : Eq (Polynomial.eval a F) 0
                                                      z' : PadicInt p
                                                      hz' : Eq (Polynomial.eval z' F) 0
                                                      hnormz' : LT.lt (Norm.norm (HSub.hSub z' a)) (Norm.norm (Polynomial.eval a (Po …
                                                      h : PadicInt p := HSub.hSub z' a
                                                      q : PadicInt p
                                                      hq : Eq (Polynomial.eval (HAdd.hAdd a h) F) (HAdd.hAdd (HAdd.hAdd (Polynomial. …
                                                      ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.eval a (Polynomial.derivative F)) h) (H …
                                                    -/
        _ = (F.derivative.eval a + q * h) * h := by rw [sq, right_distrib, mul_assoc]
                                                    /-
                                                      🎉 no goals
                                                    -/
        )
  have : h = 0 :=
    by_contra fun hne =>
      have : F.derivative.eval a + q * h = 0 :=
        (eq_zero_or_eq_zero_of_mul_eq_zero this).resolve_right hne
                                                /-
                                                  p : Nat
                                                  inst✝ : Fact (Nat.Prime p)
                                                  F : Polynomial (PadicInt p)
                                                  a : PadicInt p
                                                  ha : Eq (Polynomial.eval a F) 0
                                                  z' : PadicInt p
                                                  hz' : Eq (Polynomial.eval z' F) 0
                                                  hnormz' : LT.lt (Norm.norm (HSub.hSub z' a)) (Norm.norm (Polynomial.eval a (Po …
                                                  h : PadicInt p := HSub.hSub z' a
                                                  q : PadicInt p
                                                  hq : Eq (Polynomial.eval (HAdd.hAdd a h) F) (HAdd.hAdd (HAdd.hAdd (Polynomial. …
                                                  this✝ : Eq (HMul.hMul (HAdd.hAdd (Polynomial.eval a (Polynomial.derivative F)) …
                                                  hne : Not (Eq h 0)
                                                  this : Eq (HAdd.hAdd (Polynomial.eval a (Polynomial.derivative F)) (HMul.hMul  …
                                                  ⊢ Eq (Polynomial.eval a (Polynomial.derivative F)) (HMul.hMul (Neg.neg q) h)
                                                -/
      have : F.derivative.eval a = -q * h := by simpa using eq_neg_of_add_eq_zero_left this
                                                /-
                                                  🎉 no goals
                                                -/
      lt_irrefl ‖F.derivative.eval a‖
        (calc
                                                  /-
                                                    p : Nat
                                                    inst✝ : Fact (Nat.Prime p)
                                                    F : Polynomial (PadicInt p)
                                                    a : PadicInt p
                                                    ha : Eq (Polynomial.eval a F) 0
                                                    z' : PadicInt p
                                                    hz' : Eq (Polynomial.eval z' F) 0
                                                    hnormz' : LT.lt (Norm.norm (HSub.hSub z' a)) (Norm.norm (Polynomial.eval a (Po …
                                                    h : PadicInt p := HSub.hSub z' a
                                                    q : PadicInt p
                                                    hq : Eq (Polynomial.eval (HAdd.hAdd a h) F) (HAdd.hAdd (HAdd.hAdd (Polynomial. …
                                                    this✝¹ : Eq (HMul.hMul (HAdd.hAdd (Polynomial.eval a (Polynomial.derivative F) …
                                                    hne : Not (Eq h 0)
                                                    this✝ : Eq (HAdd.hAdd (Polynomial.eval a (Polynomial.derivative F)) (HMul.hMul …
                                                    this : Eq (Polynomial.eval a (Polynomial.derivative F)) (HMul.hMul (Neg.neg q) …
                                                    ⊢ Eq (Norm.norm (Polynomial.eval a (Polynomial.derivative F))) (HMul.hMul (Nor …
                                                  -/
          ‖F.derivative.eval a‖ = ‖q‖ * ‖h‖ := by simp [this]
                                                  /-
                                                    🎉 no goals
                                                  -/
                            /-
                              p : Nat
                              inst✝ : Fact (Nat.Prime p)
                              F : Polynomial (PadicInt p)
                              a : PadicInt p
                              ha : Eq (Polynomial.eval a F) 0
                              z' : PadicInt p
                              hz' : Eq (Polynomial.eval z' F) 0
                              hnormz' : LT.lt (Norm.norm (HSub.hSub z' a)) (Norm.norm (Polynomial.eval a (Po …
                              h : PadicInt p := HSub.hSub z' a
                              q : PadicInt p
                              hq : Eq (Polynomial.eval (HAdd.hAdd a h) F) (HAdd.hAdd (HAdd.hAdd (Polynomial. …
                              this✝¹ : Eq (HMul.hMul (HAdd.hAdd (Polynomial.eval a (Polynomial.derivative F) …
                              hne : Not (Eq h 0)
                              this✝ : Eq (HAdd.hAdd (Polynomial.eval a (Polynomial.derivative F)) (HMul.hMul …
                              this : Eq (Polynomial.eval a (Polynomial.derivative F)) (HMul.hMul (Neg.neg q) …
                              ⊢ LE.le (HMul.hMul (Norm.norm q) (Norm.norm h)) (HMul.hMul 1 (Norm.norm h))
                            -/
          _ ≤ 1 * ‖h‖ := by gcongr; apply PadicInt.norm_le_one
                                    /-
                                      🎉 no goals
                                    -/
                                          /-
                                            p : Nat
                                            inst✝ : Fact (Nat.Prime p)
                                            F : Polynomial (PadicInt p)
                                            a : PadicInt p
                                            ha : Eq (Polynomial.eval a F) 0
                                            z' : PadicInt p
                                            hz' : Eq (Polynomial.eval z' F) 0
                                            hnormz' : LT.lt (Norm.norm (HSub.hSub z' a)) (Norm.norm (Polynomial.eval a (Po …
                                            h : PadicInt p := HSub.hSub z' a
                                            q : PadicInt p
                                            hq : Eq (Polynomial.eval (HAdd.hAdd a h) F) (HAdd.hAdd (HAdd.hAdd (Polynomial. …
                                            this✝¹ : Eq (HMul.hMul (HAdd.hAdd (Polynomial.eval a (Polynomial.derivative F) …
                                            hne : Not (Eq h 0)
                                            this✝ : Eq (HAdd.hAdd (Polynomial.eval a (Polynomial.derivative F)) (HMul.hMul …
                                            this : Eq (Polynomial.eval a (Polynomial.derivative F)) (HMul.hMul (Neg.neg q) …
                                            ⊢ LT.lt (HMul.hMul 1 (Norm.norm h)) (Norm.norm (Polynomial.eval a (Polynomial. …
                                          -/
          _ < ‖F.derivative.eval a‖ := by simpa
                                          /-
                                            🎉 no goals
                                          -/
          )
                        /-
                          p : Nat
                          inst✝ : Fact (Nat.Prime p)
                          F : Polynomial (PadicInt p)
                          a : PadicInt p
                          ha : Eq (Polynomial.eval a F) 0
                          z' : PadicInt p
                          hz' : Eq (Polynomial.eval z' F) 0
                          hnormz' : LT.lt (Norm.norm (HSub.hSub z' a)) (Norm.norm (Polynomial.eval a (Po …
                          h : PadicInt p := HSub.hSub z' a
                          q : PadicInt p
                          hq : Eq (Polynomial.eval (HAdd.hAdd a h) F) (HAdd.hAdd (HAdd.hAdd (Polynomial. …
                          this✝ : Eq (HMul.hMul (HAdd.hAdd (Polynomial.eval a (Polynomial.derivative F)) …
                          this : Eq h 0
                          ⊢ Eq (HSub.hSub z' a) 0
                        -/
  eq_of_sub_eq_zero (by rw [← this])
                        /-
                          🎉 no goals
                        -/


private theorem a_is_soln (ha : F.eval a = 0) :
    F.eval a = 0 ∧
      ‖a - a‖ < ‖F.derivative.eval a‖ ∧
        ‖F.derivative.eval a‖ = ‖F.derivative.eval a‖ ∧
          ∀ z', F.eval z' = 0 → ‖z' - a‖ < ‖F.derivative.eval a‖ → z' = a :=
          /-
            p : Nat
            inst✝ : Fact (Nat.Prime p)
            F : Polynomial (PadicInt p)
            a : PadicInt p
            hnorm : LT.lt (Norm.norm (Polynomial.eval a F)) (HPow.hPow (Norm.norm (Polynom …
            ha : Eq (Polynomial.eval a F) 0
            ⊢ LT.lt (Norm.norm (HSub.hSub a a)) (Norm.norm (Polynomial.eval a (Polynomial. …
          -/
  ⟨ha, by simp [deriv_ne_zero hnorm], rfl, a_soln_is_unique ha⟩
          /-
            🎉 no goals
          -/


theorem hensels_lemma :
    ∃ z : ℤ_[p],
      F.eval z = 0 ∧
        ‖z - a‖ < ‖F.derivative.eval a‖ ∧
          ‖F.derivative.eval z‖ = ‖F.derivative.eval a‖ ∧
            ∀ z', F.eval z' = 0 → ‖z' - a‖ < ‖F.derivative.eval a‖ → z' = z := by
  classical
  exact if ha : F.eval a = 0 then ⟨a, a_is_soln hnorm ha⟩
  else by
    exact ⟨soln_gen hnorm, eval_soln hnorm,
      soln_dist_to_a_lt_deriv hnorm ha, soln_deriv_norm hnorm, fun z => soln_unique hnorm ha z⟩


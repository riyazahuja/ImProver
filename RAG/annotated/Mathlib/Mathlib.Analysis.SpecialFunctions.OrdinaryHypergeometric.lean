/-- The coefficients in the ordinary hypergeometric sum. -/
noncomputable abbrev ordinaryHypergeometricCoefficient (a b c : 𝕂) (n : ℕ) := ((n !⁻¹ : 𝕂) *
    (ascPochhammer 𝕂 n).eval a * (ascPochhammer 𝕂 n).eval b * ((ascPochhammer 𝕂 n).eval c)⁻¹)


/-- `ordinaryHypergeometricSeries 𝔸 (a b c : 𝕂)` is a `FormalMultilinearSeries`.
Its sum is the `ordinaryHypergeometric` map. -/
noncomputable def ordinaryHypergeometricSeries (a b c : 𝕂) : FormalMultilinearSeries 𝕂 𝔸 𝔸 :=
  ofScalars 𝔸 (ordinaryHypergeometricCoefficient a b c)


/-- `ordinaryHypergeometric (a b c : 𝕂) : 𝔸 → 𝔸` is the ordinary hypergeometric map, defined as the
sum of the `FormalMultilinearSeries` `ordinaryHypergeometricSeries 𝔸 a b c`.

Note that this takes the junk value `0` outside the radius of convergence.
-/
noncomputable def ordinaryHypergeometric (x : 𝔸) : 𝔸 :=
  (ordinaryHypergeometricSeries 𝔸 a b c).sum x


@[inherit_doc]
notation "₂F₁" => ordinaryHypergeometric


theorem ordinaryHypergeometricSeries_apply_eq (x : 𝔸) (n : ℕ) :
    (ordinaryHypergeometricSeries 𝔸 a b c n fun _ => x) =
      ((n !⁻¹ : 𝕂) * (ascPochhammer 𝕂 n).eval a * (ascPochhammer 𝕂 n).eval b *
        ((ascPochhammer 𝕂 n).eval c)⁻¹ ) • x ^ n := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁴ : Field 𝕂
    inst✝³ : Ring 𝔸
    inst✝² : Algebra 𝕂 𝔸
    inst✝¹ : TopologicalSpace 𝔸
    inst✝ : TopologicalRing 𝔸
    a b c : 𝕂
    x : 𝔸
    n : Nat
    ⊢ Eq ((ordinaryHypergeometricSeries 𝔸 a b c n) fun x_1 => x) (HSMul.hSMul (HMu …
  -/
  rw [ordinaryHypergeometricSeries, ofScalars_apply_eq]
  /-
    🎉 no goals
  -/


/-- This naming follows the convention of `NormedSpace.expSeries_apply_eq'`. -/
theorem ordinaryHypergeometricSeries_apply_eq' (x : 𝔸) :
    (fun n => ordinaryHypergeometricSeries 𝔸 a b c n fun _ => x) =
      fun n => ((n !⁻¹ : 𝕂) * (ascPochhammer 𝕂 n).eval a * (ascPochhammer 𝕂 n).eval b *
        ((ascPochhammer 𝕂 n).eval c)⁻¹ ) • x ^ n := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁴ : Field 𝕂
    inst✝³ : Ring 𝔸
    inst✝² : Algebra 𝕂 𝔸
    inst✝¹ : TopologicalSpace 𝔸
    inst✝ : TopologicalRing 𝔸
    a b c : 𝕂
    x : 𝔸
    ⊢ Eq (fun n => (ordinaryHypergeometricSeries 𝔸 a b c n) fun x_1 => x) fun n => …
  -/
  rw [ordinaryHypergeometricSeries, ofScalars_apply_eq']
  /-
    🎉 no goals
  -/


theorem ordinaryHypergeometric_sum_eq (x : 𝔸) : (ordinaryHypergeometricSeries 𝔸 a b c).sum x =
    ∑' n : ℕ, ((n !⁻¹ : 𝕂) * (ascPochhammer 𝕂 n).eval a * (ascPochhammer 𝕂 n).eval b *
      ((ascPochhammer 𝕂 n).eval c)⁻¹ ) • x ^ n :=
  tsum_congr fun n => ordinaryHypergeometricSeries_apply_eq a b c x n


theorem ordinaryHypergeometric_eq_tsum : ₂F₁ a b c =
    fun (x : 𝔸) => ∑' n : ℕ, ((n !⁻¹ : 𝕂) * (ascPochhammer 𝕂 n).eval a *
      (ascPochhammer 𝕂 n).eval b * ((ascPochhammer 𝕂 n).eval c)⁻¹ ) • x ^ n :=
  funext (ordinaryHypergeometric_sum_eq a b c)


theorem ordinaryHypergeometricSeries_apply_zero (n : ℕ) :
    (ordinaryHypergeometricSeries 𝔸 a b c n fun _ => 0) = Pi.single (f := fun _ => 𝔸) 0 1 n := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁴ : Field 𝕂
    inst✝³ : Ring 𝔸
    inst✝² : Algebra 𝕂 𝔸
    inst✝¹ : TopologicalSpace 𝔸
    inst✝ : TopologicalRing 𝔸
    a b c : 𝕂
    n : Nat
    ⊢ Eq ((ordinaryHypergeometricSeries 𝔸 a b c n) fun x => 0) (Pi.single 0 1 n)
  -/
  rw [ordinaryHypergeometricSeries, ofScalars_apply_eq, ordinaryHypergeometricCoefficient]
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁴ : Field 𝕂
    inst✝³ : Ring 𝔸
    inst✝² : Algebra 𝕂 𝔸
    inst✝¹ : TopologicalSpace 𝔸
    inst✝ : TopologicalRing 𝔸
    a b c : 𝕂
    n : Nat
    ⊢ Eq (HSMul.hSMul (HMul.hMul (HMul.hMul (HMul.hMul (Inv.inv ↑n.factorial) (Pol …
  -/
              /-
                🎉 no goals
              -/
  cases n <;> simp
              /-
                🎉 no goals
              -/


@[simp]
theorem ordinaryHypergeometric_zero : ₂F₁ a b c (0 : 𝔸) = 1 := by
  simp [ordinaryHypergeometric_eq_tsum, ← ordinaryHypergeometricSeries_apply_eq,
    ordinaryHypergeometricSeries_apply_zero]


theorem ordinaryHypergeometricSeries_symm :
    ordinaryHypergeometricSeries 𝔸 a b c = ordinaryHypergeometricSeries 𝔸 b a c := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁴ : Field 𝕂
    inst✝³ : Ring 𝔸
    inst✝² : Algebra 𝕂 𝔸
    inst✝¹ : TopologicalSpace 𝔸
    inst✝ : TopologicalRing 𝔸
    a b c : 𝕂
    ⊢ Eq (ordinaryHypergeometricSeries 𝔸 a b c) (ordinaryHypergeometricSeries 𝔸 b  …
  -/
  unfold ordinaryHypergeometricSeries ordinaryHypergeometricCoefficient
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁴ : Field 𝕂
    inst✝³ : Ring 𝔸
    inst✝² : Algebra 𝕂 𝔸
    inst✝¹ : TopologicalSpace 𝔸
    inst✝ : TopologicalRing 𝔸
    a b c : 𝕂
    ⊢ Eq (FormalMultilinearSeries.ofScalars 𝔸 fun n => HMul.hMul (HMul.hMul (HMul. …
  -/
  simp [mul_assoc, mul_left_comm]
  /-
    🎉 no goals
  -/


/-- If any parameter to the series is a sufficiently large nonpositive integer, then the series
term is zero. -/
lemma ordinaryHypergeometricSeries_eq_zero_of_neg_nat {n k : ℕ} (habc : k = -a ∨ k = -b ∨ k = -c)
    (hk : k < n) : ordinaryHypergeometricSeries 𝔸 a b c n = 0 := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁴ : Field 𝕂
    inst✝³ : Ring 𝔸
    inst✝² : Algebra 𝕂 𝔸
    inst✝¹ : TopologicalSpace 𝔸
    inst✝ : TopologicalRing 𝔸
    a b c : 𝕂
    n k : Nat
    habc : Or (Eq (↑k) (Neg.neg a)) (Or (Eq (↑k) (Neg.neg b)) (Eq (↑k) (Neg.neg c)))
    hk : LT.lt k n
    ⊢ Eq (ordinaryHypergeometricSeries 𝔸 a b c n) 0
  -/
  rw [ordinaryHypergeometricSeries, ofScalars]
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝⁴ : Field 𝕂
    inst✝³ : Ring 𝔸
    inst✝² : Algebra 𝕂 𝔸
    inst✝¹ : TopologicalSpace 𝔸
    inst✝ : TopologicalRing 𝔸
    a b c : 𝕂
    n k : Nat
    habc : Or (Eq (↑k) (Neg.neg a)) (Or (Eq (↑k) (Neg.neg b)) (Eq (↑k) (Neg.neg c)))
    hk : LT.lt k n
    ⊢ Eq (HSMul.hSMul (ordinaryHypergeometricCoefficient a b c n) (ContinuousMulti …
  -/
  rcases habc with h | h | h
  all_goals
    ext
    simp [(ascPochhammer_eval_eq_zero_iff n _).2 ⟨k, hk, h⟩]


theorem ordinaryHypergeometric_radius_top_of_neg_nat₁ {k : ℕ} :
    (ordinaryHypergeometricSeries 𝔸 (-(k : 𝕂)) b c).radius = ⊤ := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝² : RCLike 𝕂
    inst✝¹ : NormedDivisionRing 𝔸
    inst✝ : NormedAlgebra 𝕂 𝔸
    b c : 𝕂
    k : Nat
    ⊢ Eq (ordinaryHypergeometricSeries 𝔸 (Neg.neg ↑k) b c).radius Top.top
  -/
  refine FormalMultilinearSeries.radius_eq_top_of_forall_image_add_eq_zero _ (1 + k) fun n ↦ ?_
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝² : RCLike 𝕂
    inst✝¹ : NormedDivisionRing 𝔸
    inst✝ : NormedAlgebra 𝕂 𝔸
    b c : 𝕂
    k n : Nat
    ⊢ Eq (ordinaryHypergeometricSeries 𝔸 (Neg.neg ↑k) b c (HAdd.hAdd n (HAdd.hAdd  …
  -/
  exact ordinaryHypergeometricSeries_eq_zero_of_neg_nat (-(k : 𝕂)) b c (by aesop) (by omega)
  /-
    🎉 no goals
  -/


theorem ordinaryHypergeometric_radius_top_of_neg_nat₂ {k : ℕ} :
    (ordinaryHypergeometricSeries 𝔸 a (-(k : 𝕂)) c).radius = ⊤ := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝² : RCLike 𝕂
    inst✝¹ : NormedDivisionRing 𝔸
    inst✝ : NormedAlgebra 𝕂 𝔸
    a c : 𝕂
    k : Nat
    ⊢ Eq (ordinaryHypergeometricSeries 𝔸 a (Neg.neg ↑k) c).radius Top.top
  -/
  rw [ordinaryHypergeometricSeries_symm]
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝² : RCLike 𝕂
    inst✝¹ : NormedDivisionRing 𝔸
    inst✝ : NormedAlgebra 𝕂 𝔸
    a c : 𝕂
    k : Nat
    ⊢ Eq (ordinaryHypergeometricSeries 𝔸 (Neg.neg ↑k) a c).radius Top.top
  -/
  exact ordinaryHypergeometric_radius_top_of_neg_nat₁ 𝔸 a c
  /-
    🎉 no goals
  -/


theorem ordinaryHypergeometric_radius_top_of_neg_nat₃ {k : ℕ} :
    (ordinaryHypergeometricSeries 𝔸 a b (-(k : 𝕂))).radius = ⊤ := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝² : RCLike 𝕂
    inst✝¹ : NormedDivisionRing 𝔸
    inst✝ : NormedAlgebra 𝕂 𝔸
    a b : 𝕂
    k : Nat
    ⊢ Eq (ordinaryHypergeometricSeries 𝔸 a b (Neg.neg ↑k)).radius Top.top
  -/
  refine FormalMultilinearSeries.radius_eq_top_of_forall_image_add_eq_zero _ (1 + k) fun n ↦ ?_
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝² : RCLike 𝕂
    inst✝¹ : NormedDivisionRing 𝔸
    inst✝ : NormedAlgebra 𝕂 𝔸
    a b : 𝕂
    k n : Nat
    ⊢ Eq (ordinaryHypergeometricSeries 𝔸 a b (Neg.neg ↑k) (HAdd.hAdd n (HAdd.hAdd  …
  -/
  exact ordinaryHypergeometricSeries_eq_zero_of_neg_nat a b (-(k : 𝕂)) (by aesop) (by omega)
  /-
    🎉 no goals
  -/


/-- An iff variation on `ordinaryHypergeometricSeries_eq_zero_of_nonpos_int` for `[RCLike 𝕂]`. -/
lemma ordinaryHypergeometricSeries_eq_zero_iff (n : ℕ) :
    ordinaryHypergeometricSeries 𝔸 a b c n = 0 ↔ ∃ k < n, k = -a ∨ k = -b ∨ k = -c := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝² : RCLike 𝕂
    inst✝¹ : NormedDivisionRing 𝔸
    inst✝ : NormedAlgebra 𝕂 𝔸
    a b c : 𝕂
    n : Nat
    ⊢ Iff (Eq (ordinaryHypergeometricSeries 𝔸 a b c n) 0) (Exists fun k => And (LT …
  -/
  refine ⟨fun h ↦ ?_, fun zero ↦ ?_⟩
    /-
      case refine_1
      𝕂 : Type u_1
      𝔸 : Type u_2
      inst✝² : RCLike 𝕂
      inst✝¹ : NormedDivisionRing 𝔸
      inst✝ : NormedAlgebra 𝕂 𝔸
      a b c : 𝕂
      n : Nat
      h : Eq (ordinaryHypergeometricSeries 𝔸 a b c n) 0
      ⊢ Exists fun k => And (LT.lt k n) (Or (Eq (↑k) (Neg.neg a)) (Or (Eq (↑k) (Neg. …
    -/
  · rw [ordinaryHypergeometricSeries, ofScalars_eq_zero] at h
    /-
      case refine_1
      𝕂 : Type u_1
      𝔸 : Type u_2
      inst✝² : RCLike 𝕂
      inst✝¹ : NormedDivisionRing 𝔸
      inst✝ : NormedAlgebra 𝕂 𝔸
      a b c : 𝕂
      n : Nat
      h : Eq (ordinaryHypergeometricCoefficient a b c n) 0
      ⊢ Exists fun k => And (LT.lt k n) (Or (Eq (↑k) (Neg.neg a)) (Or (Eq (↑k) (Neg. …
    -/
    simp only [_root_.mul_eq_zero, inv_eq_zero] at h
    /-
      case refine_1
      𝕂 : Type u_1
      𝔸 : Type u_2
      inst✝² : RCLike 𝕂
      inst✝¹ : NormedDivisionRing 𝔸
      inst✝ : NormedAlgebra 𝕂 𝔸
      a b c : 𝕂
      n : Nat
      h : Or (Or (Or (Eq (↑n.factorial) 0) (Eq (Polynomial.eval a (ascPochhammer 𝕂 n …
      ⊢ Exists fun k => And (LT.lt k n) (Or (Eq (↑k) (Neg.neg a)) (Or (Eq (↑k) (Neg. …
    -/
    rcases h with ((hn | h) | h) | h
      /-
        case refine_1.inl.inl.inl
        𝕂 : Type u_1
        𝔸 : Type u_2
        inst✝² : RCLike 𝕂
        inst✝¹ : NormedDivisionRing 𝔸
        inst✝ : NormedAlgebra 𝕂 𝔸
        a b c : 𝕂
        n : Nat
        hn : Eq (↑n.factorial) 0
        ⊢ Exists fun k => And (LT.lt k n) (Or (Eq (↑k) (Neg.neg a)) (Or (Eq (↑k) (Neg. …
      -/
    · simp [Nat.factorial_ne_zero] at hn
      /-
        🎉 no goals
      -/
    all_goals
      obtain ⟨kn, hkn, hn⟩ := (ascPochhammer_eval_eq_zero_iff _ _).1 h
      exact ⟨kn, hkn, by tauto⟩
    /-
      case refine_2
      𝕂 : Type u_1
      𝔸 : Type u_2
      inst✝² : RCLike 𝕂
      inst✝¹ : NormedDivisionRing 𝔸
      inst✝ : NormedAlgebra 𝕂 𝔸
      a b c : 𝕂
      n : Nat
      zero : Exists fun k => And (LT.lt k n) (Or (Eq (↑k) (Neg.neg a)) (Or (Eq (↑k)  …
      ⊢ Eq (ordinaryHypergeometricSeries 𝔸 a b c n) 0
    -/
  · obtain ⟨_, h, hn⟩ := zero
    /-
      case refine_2.intro.intro
      𝕂 : Type u_1
      𝔸 : Type u_2
      inst✝² : RCLike 𝕂
      inst✝¹ : NormedDivisionRing 𝔸
      inst✝ : NormedAlgebra 𝕂 𝔸
      a b c : 𝕂
      n w✝ : Nat
      h : LT.lt w✝ n
      hn : Or (Eq (↑w✝) (Neg.neg a)) (Or (Eq (↑w✝) (Neg.neg b)) (Eq (↑w✝) (Neg.neg c …
      ⊢ Eq (ordinaryHypergeometricSeries 𝔸 a b c n) 0
    -/
    exact ordinaryHypergeometricSeries_eq_zero_of_neg_nat a b c hn h
    /-
      🎉 no goals
    -/


theorem ordinaryHypergeometricSeries_norm_div_succ_norm (n : ℕ)
    (habc : ∀ kn < n, (↑kn ≠ -a ∧ ↑kn ≠ -b ∧ ↑kn ≠ -c)) :
      ‖ordinaryHypergeometricCoefficient a b c n‖ / ‖ordinaryHypergeometricCoefficient a b c n.succ‖
      = ‖a + n‖⁻¹ * ‖b + n‖⁻¹ * ‖c + n‖ * ‖1 + (n : 𝕂)‖ := by
  simp only [mul_inv_rev, factorial_succ, cast_mul, cast_add,
    cast_one, ascPochhammer_succ_eval, norm_mul, norm_inv]
  calc
    _ = ‖Polynomial.eval a (ascPochhammer 𝕂 n)‖ * ‖Polynomial.eval a (ascPochhammer 𝕂 n)‖⁻¹ *
        ‖Polynomial.eval b (ascPochhammer 𝕂 n)‖ * ‖Polynomial.eval b (ascPochhammer 𝕂 n)‖⁻¹ *
        ‖Polynomial.eval c (ascPochhammer 𝕂 n)‖⁻¹⁻¹ * ‖Polynomial.eval c (ascPochhammer 𝕂 n)‖⁻¹ *
        ‖(n ! : 𝕂)‖⁻¹⁻¹ * ‖(n ! : 𝕂)‖⁻¹ * ‖a + n‖⁻¹ * ‖b + n‖⁻¹ * ‖c + n‖⁻¹⁻¹ *
        ‖1 + (n : 𝕂)‖⁻¹⁻¹ := by ring_nf
    _ = _ := by
      simp only [inv_inv]
      repeat rw [DivisionRing.mul_inv_cancel, one_mul]
      all_goals
        rw [norm_ne_zero_iff]
      any_goals
        apply (ascPochhammer_eval_eq_zero_iff n _).not.2
        push_neg
        exact fun kn hkn ↦ by simp [habc kn hkn]
      exact cast_ne_zero.2 (factorial_ne_zero n)


/-- The radius of convergence of `ordinaryHypergeometricSeries` is unity if none of the parameters
are non-positive integers. -/
theorem ordinaryHypergeometricSeries_radius_eq_one
    (habc : ∀ kn : ℕ, ↑kn ≠ -a ∧ ↑kn ≠ -b ∧ ↑kn ≠ -c) :
      (ordinaryHypergeometricSeries 𝔸 a b c).radius = 1 := by
  /-
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝² : RCLike 𝕂
    inst✝¹ : NormedDivisionRing 𝔸
    inst✝ : NormedAlgebra 𝕂 𝔸
    a b c : 𝕂
    habc : ∀ (kn : Nat), And (Ne (↑kn) (Neg.neg a)) (And (Ne (↑kn) (Neg.neg b)) (N …
    ⊢ Eq (ordinaryHypergeometricSeries 𝔸 a b c).radius 1
  -/
  convert ofScalars_radius_eq_of_tendsto 𝔸 _ one_ne_zero ?_
  suffices Tendsto (fun k : ℕ ↦ (a + k)⁻¹ * (b + k)⁻¹ * (c + k) * ((1 : 𝕂) + k)) atTop (𝓝 1) by
    simp_rw [ordinaryHypergeometricSeries_norm_div_succ_norm a b c _ (fun n _ ↦ habc n)]
    simp [← norm_mul, ← norm_inv]
    convert Filter.Tendsto.norm this
    exact norm_one.symm
  have (k : ℕ) : (a + k)⁻¹ * (b + k)⁻¹ * (c + k) * ((1 : 𝕂) + k) =
        (c + k) / (a + k) * ((1 + k) / (b + k)) := by field_simp
  /-
    case convert_5
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝² : RCLike 𝕂
    inst✝¹ : NormedDivisionRing 𝔸
    inst✝ : NormedAlgebra 𝕂 𝔸
    a b c : 𝕂
    habc : ∀ (kn : Nat), And (Ne (↑kn) (Neg.neg a)) (And (Ne (↑kn) (Neg.neg b)) (N …
    this : ∀ (k : Nat), Eq (HMul.hMul (HMul.hMul (HMul.hMul (Inv.inv (HAdd.hAdd a  …
    ⊢ Filter.Tendsto (fun k => HMul.hMul (HMul.hMul (HMul.hMul (Inv.inv (HAdd.hAdd …
  -/
  simp_rw [this]
  /-
    case convert_5
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝² : RCLike 𝕂
    inst✝¹ : NormedDivisionRing 𝔸
    inst✝ : NormedAlgebra 𝕂 𝔸
    a b c : 𝕂
    habc : ∀ (kn : Nat), And (Ne (↑kn) (Neg.neg a)) (And (Ne (↑kn) (Neg.neg b)) (N …
    this : ∀ (k : Nat), Eq (HMul.hMul (HMul.hMul (HMul.hMul (Inv.inv (HAdd.hAdd a  …
    ⊢ Filter.Tendsto (fun k => HMul.hMul (HDiv.hDiv (HAdd.hAdd c ↑k) (HAdd.hAdd a  …
  -/
  apply (mul_one (1 : 𝕂)) ▸ Filter.Tendsto.mul <;>
  /-
    case convert_5.hf
    𝕂 : Type u_1
    𝔸 : Type u_2
    inst✝² : RCLike 𝕂
    inst✝¹ : NormedDivisionRing 𝔸
    inst✝ : NormedAlgebra 𝕂 𝔸
    a b c : 𝕂
    habc : ∀ (kn : Nat), And (Ne (↑kn) (Neg.neg a)) (And (Ne (↑kn) (Neg.neg b)) (N …
    this : ∀ (k : Nat), Eq (HMul.hMul (HMul.hMul (HMul.hMul (Inv.inv (HAdd.hAdd a  …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (HAdd.hAdd c ↑x) (HAdd.hAdd a ↑x)) Filter …
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
  convert RCLike.tendsto_add_mul_div_add_mul_atTop_nhds _ _ (1 : 𝕂) one_ne_zero <;> simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/



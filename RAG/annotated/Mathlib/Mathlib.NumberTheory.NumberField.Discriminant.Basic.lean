open scoped Classical in
theorem _root_.NumberField.mixedEmbedding.volume_fundamentalDomain_latticeBasis :
    volume (fundamentalDomain (latticeBasis K)) =
      (2 : ℝ≥0∞)⁻¹ ^ nrComplexPlaces K * sqrt ‖discr K‖₊ := by
  let f : Module.Free.ChooseBasisIndex ℤ (𝓞 K) ≃ (K →+* ℂ) :=
    (canonicalEmbedding.latticeBasis K).indexEquiv (Pi.basisFun ℂ _)
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    f : Equiv (Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)) (R …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (ZSpan.fundamentalDomain (NumberField. …
  -/
  let e : (index K) ≃ Module.Free.ChooseBasisIndex ℤ (𝓞 K) := (indexEquiv K).trans f.symm
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    f : Equiv (Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)) (R …
    e : Equiv (NumberField.mixedEmbedding.index K) (Module.Free.ChooseBasisIndex I …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (ZSpan.fundamentalDomain (NumberField. …
  -/
  let M := (mixedEmbedding.stdBasis K).toMatrix ((latticeBasis K).reindex e.symm)
  let N := Algebra.embeddingsMatrixReindex ℚ ℂ (integralBasis K ∘ f.symm)
    RingHom.equivRatAlgHom
  suffices M.map ofRealHom = matrixToStdBasis K *
      (Matrix.reindex (indexEquiv K).symm (indexEquiv K).symm N).transpose by
    calc volume (fundamentalDomain (latticeBasis K))
      _ = ‖((mixedEmbedding.stdBasis K).toMatrix ((latticeBasis K).reindex e.symm)).det‖₊ := by
        rw [← fundamentalDomain_reindex _ e.symm, ← norm_toNNReal, measure_fundamentalDomain
          ((latticeBasis K).reindex e.symm), volume_fundamentalDomain_stdBasis, mul_one]
        rfl
      _ = ‖(matrixToStdBasis K).det * N.det‖₊ := by
        rw [← nnnorm_real, ← ofRealHom_eq_coe, RingHom.map_det, RingHom.mapMatrix_apply, this,
          det_mul, det_transpose, det_reindex_self]
      _ = (2 : ℝ≥0∞)⁻¹ ^ Fintype.card {w : InfinitePlace K // IsComplex w} * sqrt ‖N.det ^ 2‖₊ := by
        have : ‖Complex.I‖₊ = 1 := by rw [← norm_toNNReal, norm_eq_abs, abs_I, Real.toNNReal_one]
        rw [det_matrixToStdBasis, nnnorm_mul, nnnorm_pow, nnnorm_mul, this, mul_one, nnnorm_inv,
          coe_mul, ENNReal.coe_pow, ← norm_toNNReal, RCLike.norm_two, Real.toNNReal_ofNat,
          coe_inv two_ne_zero, coe_ofNat, nnnorm_pow, NNReal.sqrt_sq]
      _ = (2 : ℝ≥0∞)⁻¹ ^ Fintype.card { w // IsComplex w } * NNReal.sqrt ‖discr K‖₊ := by
        rw [← Algebra.discr_eq_det_embeddingsMatrixReindex_pow_two, Algebra.discr_reindex,
          ← coe_discr, map_intCast, ← Complex.nnnorm_intCast]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    f : Equiv (Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)) (R …
    e : Equiv (NumberField.mixedEmbedding.index K) (Module.Free.ChooseBasisIndex I …
    M : Matrix (NumberField.mixedEmbedding.index K) (NumberField.mixedEmbedding.in …
    N : Matrix (RingHom K Complex) (RingHom K Complex) Complex := Algebra.embeddin …
    ⊢ Eq (M.map ⇑Complex.ofRealHom) (HMul.hMul (NumberField.mixedEmbedding.matrixT …
  -/
  ext : 2
  /-
    case a
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    f : Equiv (Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)) (R …
    e : Equiv (NumberField.mixedEmbedding.index K) (Module.Free.ChooseBasisIndex I …
    M : Matrix (NumberField.mixedEmbedding.index K) (NumberField.mixedEmbedding.in …
    N : Matrix (RingHom K Complex) (RingHom K Complex) Complex := Algebra.embeddin …
    i✝ j✝ : NumberField.mixedEmbedding.index K
    ⊢ Eq (M.map (⇑Complex.ofRealHom) i✝ j✝) (HMul.hMul (NumberField.mixedEmbedding …
  -/
  dsimp only [M]
  rw [Matrix.map_apply, Basis.toMatrix_apply, Basis.coe_reindex, Function.comp_apply,
    Equiv.symm_symm, latticeBasis_apply, ← commMap_canonical_eq_mixed, Complex.ofRealHom_eq_coe,
    stdBasis_repr_eq_matrixToStdBasis_mul K _ (fun _ => rfl)]
  /-
    case a
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    f : Equiv (Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)) (R …
    e : Equiv (NumberField.mixedEmbedding.index K) (Module.Free.ChooseBasisIndex I …
    M : Matrix (NumberField.mixedEmbedding.index K) (NumberField.mixedEmbedding.in …
    N : Matrix (RingHom K Complex) (RingHom K Complex) Complex := Algebra.embeddin …
    i✝ j✝ : NumberField.mixedEmbedding.index K
    ⊢ Eq ((NumberField.mixedEmbedding.matrixToStdBasis K).mulVec (Function.comp (( …
  -/
  rfl
  /-
    🎉 no goals
  -/


open scoped Classical in
theorem _root_.NumberField.mixedEmbedding.covolume_integerLattice :
    /-
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      ⊢ MeasureTheory.Measure (NumberField.mixedEmbedding.mixedSpace K)
    -/
    ZLattice.covolume (mixedEmbedding.integerLattice K) =
    /-
      🎉 no goals
    -/
      (2 ⁻¹) ^ nrComplexPlaces K * √|discr K| := by
  rw [ZLattice.covolume_eq_measure_fundamentalDomain _ _ (fundamentalDomain_integerLattice K),
    volume_fundamentalDomain_latticeBasis, ENNReal.toReal_mul, ENNReal.toReal_pow,
    ENNReal.toReal_inv, toReal_ofNat, ENNReal.coe_toReal, Real.coe_sqrt, coe_nnnorm,
    Int.norm_eq_abs]


open scoped Classical in
theorem _root_.NumberField.mixedEmbedding.covolume_idealLattice (I : (FractionalIdeal (𝓞 K)⁰ K)ˣ) :
    /-
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
      ⊢ MeasureTheory.Measure (NumberField.mixedEmbedding.mixedSpace K)
    -/
    ZLattice.covolume (mixedEmbedding.idealLattice K I) =
    /-
      🎉 no goals
    -/
      (FractionalIdeal.absNorm (I : FractionalIdeal (𝓞 K)⁰ K)) *
        (2 ⁻¹) ^ nrComplexPlaces K * √|discr K| := by
  rw [ZLattice.covolume_eq_measure_fundamentalDomain _ _ (fundamentalDomain_idealLattice K I),
    volume_fundamentalDomain_fractionalIdealLatticeBasis, volume_fundamentalDomain_latticeBasis,
    ENNReal.toReal_mul, ENNReal.toReal_mul, ENNReal.toReal_pow, ENNReal.toReal_inv, toReal_ofNat,
    ENNReal.coe_toReal, Real.coe_sqrt, coe_nnnorm, Int.norm_eq_abs,
    ENNReal.toReal_ofReal (Rat.cast_nonneg.mpr (FractionalIdeal.absNorm_nonneg I.val)), mul_assoc]


theorem exists_ne_zero_mem_ideal_of_norm_le_mul_sqrt_discr (I : (FractionalIdeal (𝓞 K)⁰ K)ˣ) :
    ∃ a ∈ (I : FractionalIdeal (𝓞 K)⁰ K), a ≠ 0 ∧
      |Algebra.norm ℚ (a : K)| ≤ FractionalIdeal.absNorm I.1 * (4 / π) ^ nrComplexPlaces K *
        (finrank ℚ K).factorial / (finrank ℚ K) ^ (finrank ℚ K) * Real.sqrt |discr K| := by
  classical
  -- The smallest possible value for `exists_ne_zero_mem_ideal_of_norm_le`
  let B := (minkowskiBound K I * (convexBodySumFactor K)⁻¹).toReal ^ (1 / (finrank ℚ K : ℝ))
  have h_le : (minkowskiBound K I) ≤ volume (convexBodySum K B) := by
    refine le_of_eq ?_
    rw [convexBodySum_volume, ← ENNReal.ofReal_pow (by positivity), ← Real.rpow_natCast,
      ← Real.rpow_mul toReal_nonneg, div_mul_cancel₀, Real.rpow_one, ofReal_toReal, mul_comm,
      mul_assoc, ← coe_mul, inv_mul_cancel₀ (convexBodySumFactor_ne_zero K), ENNReal.coe_one,
      mul_one]
    · exact mul_ne_top (ne_of_lt (minkowskiBound_lt_top K I)) coe_ne_top
    · exact (Nat.cast_ne_zero.mpr (ne_of_gt finrank_pos))
  convert exists_ne_zero_mem_ideal_of_norm_le K I h_le
  rw [div_pow B, ← Real.rpow_natCast B, ← Real.rpow_mul (by positivity), div_mul_cancel₀ _
    (Nat.cast_ne_zero.mpr <| ne_of_gt finrank_pos), Real.rpow_one, mul_comm_div, mul_div_assoc']
  congr 1
  rw [eq_comm]
  calc
    _ = FractionalIdeal.absNorm I.1 * (2 : ℝ)⁻¹ ^ nrComplexPlaces K * sqrt ‖discr K‖₊ *
          (2 : ℝ) ^ finrank ℚ K * ((2 : ℝ) ^ nrRealPlaces K * (π / 2) ^ nrComplexPlaces K /
            (Nat.factorial (finrank ℚ K)))⁻¹ := by
      simp_rw [minkowskiBound, convexBodySumFactor,
        volume_fundamentalDomain_fractionalIdealLatticeBasis,
        volume_fundamentalDomain_latticeBasis, toReal_mul, toReal_pow, toReal_inv, coe_toReal,
        toReal_ofNat, mixedEmbedding.finrank, mul_assoc]
      rw [ENNReal.toReal_ofReal (Rat.cast_nonneg.mpr (FractionalIdeal.absNorm_nonneg I.1))]
      simp_rw [NNReal.coe_inv, NNReal.coe_div, NNReal.coe_mul, NNReal.coe_pow, NNReal.coe_div,
        coe_real_pi, NNReal.coe_ofNat, NNReal.coe_natCast]
    _ = FractionalIdeal.absNorm I.1 * (2 : ℝ) ^ (finrank ℚ K - nrComplexPlaces K - nrRealPlaces K +
          nrComplexPlaces K : ℤ) * Real.sqrt ‖discr K‖ * Nat.factorial (finrank ℚ K) *
            π⁻¹ ^ (nrComplexPlaces K) := by
      simp_rw [inv_div, div_eq_mul_inv, mul_inv, ← zpow_neg_one, ← zpow_natCast, mul_zpow,
        ← zpow_mul, neg_one_mul, mul_neg_one, neg_neg, Real.coe_sqrt, coe_nnnorm, sub_eq_add_neg,
        zpow_add₀ (two_ne_zero : (2 : ℝ) ≠ 0)]
      ring
    _ = FractionalIdeal.absNorm I.1 * (2 : ℝ) ^ (2 * nrComplexPlaces K : ℤ) * Real.sqrt ‖discr K‖ *
          Nat.factorial (finrank ℚ K) * π⁻¹ ^ (nrComplexPlaces K) := by
      congr
      rw [← card_add_two_mul_card_eq_rank, Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat]
      ring
    _ = FractionalIdeal.absNorm I.1 * (4 / π) ^ nrComplexPlaces K * (finrank ℚ K).factorial *
          Real.sqrt |discr K| := by
      rw [Int.norm_eq_abs, zpow_mul, show (2 : ℝ) ^ (2 : ℤ) = 4 by norm_cast, div_pow,
        inv_eq_one_div, div_pow, one_pow, zpow_natCast]
      ring


theorem exists_ne_zero_mem_ringOfIntegers_of_norm_le_mul_sqrt_discr :
    ∃ (a : 𝓞 K), a ≠ 0 ∧
      |Algebra.norm ℚ (a : K)| ≤ (4 / π) ^ nrComplexPlaces K *
        (finrank ℚ K).factorial / (finrank ℚ K) ^ (finrank ℚ K) * Real.sqrt |discr K| := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ Exists fun a => And (Ne a 0) (LE.le (↑(abs ((Algebra.norm Rat) ↑a))) (HMul.h …
  -/
  obtain ⟨_, h_mem, h_nz, h_nm⟩ := exists_ne_zero_mem_ideal_of_norm_le_mul_sqrt_discr K ↑1
  /-
    case intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    w✝ : K
    h_mem : Membership.mem (↑1) w✝
    h_nz : Ne w✝ 0
    h_nm : LE.le (↑(abs ((Algebra.norm Rat) w✝))) (HMul.hMul (HDiv.hDiv (HMul.hMul …
    ⊢ Exists fun a => And (Ne a 0) (LE.le (↑(abs ((Algebra.norm Rat) ↑a))) (HMul.h …
  -/
  obtain ⟨a, rfl⟩ := (FractionalIdeal.mem_one_iff _).mp h_mem
  /-
    case intro.intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    a : NumberField.RingOfIntegers K
    h_mem : Membership.mem (↑1) ((algebraMap (NumberField.RingOfIntegers K) K) a)
    h_nz : Ne ((algebraMap (NumberField.RingOfIntegers K) K) a) 0
    h_nm : LE.le (↑(abs ((Algebra.norm Rat) ((algebraMap (NumberField.RingOfIntege …
    ⊢ Exists fun a => And (Ne a 0) (LE.le (↑(abs ((Algebra.norm Rat) ↑a))) (HMul.h …
  -/
  refine ⟨a, ne_zero_of_map h_nz, ?_⟩
  /-
    case intro.intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    a : NumberField.RingOfIntegers K
    h_mem : Membership.mem (↑1) ((algebraMap (NumberField.RingOfIntegers K) K) a)
    h_nz : Ne ((algebraMap (NumberField.RingOfIntegers K) K) a) 0
    h_nm : LE.le (↑(abs ((Algebra.norm Rat) ((algebraMap (NumberField.RingOfIntege …
    ⊢ LE.le (↑(abs ((Algebra.norm Rat) ↑a))) (HMul.hMul (HDiv.hDiv (HMul.hMul (HPo …
  -/
  simp_rw [Units.val_one, FractionalIdeal.absNorm_one, Rat.cast_one, one_mul] at h_nm
  /-
    case intro.intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    a : NumberField.RingOfIntegers K
    h_mem : Membership.mem (↑1) ((algebraMap (NumberField.RingOfIntegers K) K) a)
    h_nz : Ne ((algebraMap (NumberField.RingOfIntegers K) K) a) 0
    h_nm : LE.le (↑(abs ((Algebra.norm Rat) ((algebraMap (NumberField.RingOfIntege …
    ⊢ LE.le (↑(abs ((Algebra.norm Rat) ↑a))) (HMul.hMul (HDiv.hDiv (HMul.hMul (HPo …
  -/
  exact h_nm
  /-
    🎉 no goals
  -/


theorem abs_discr_ge (h : 1 < finrank ℚ K) :
    (4 / 9 : ℝ) * (3 * π / 4) ^ finrank ℚ K ≤ |discr K| := by
  -- We use `exists_ne_zero_mem_ringOfIntegers_of_norm_le_mul_sqrt_discr` to get a nonzero
  -- algebraic integer `x` of small norm and the fact that `1 ≤ |Norm x|` to get a lower bound
  -- on `sqrt |discr K|`.
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    h : LT.lt 1 (Module.finrank Rat K)
    ⊢ LE.le (HMul.hMul (4 / 9) (HPow.hPow (HDiv.hDiv (HMul.hMul 3 Real.pi) 4) (Mod …
  -/
  obtain ⟨x, h_nz, h_bd⟩ := exists_ne_zero_mem_ringOfIntegers_of_norm_le_mul_sqrt_discr K
  have h_nm : (1 : ℝ) ≤ |Algebra.norm ℚ (x : K)| := by
    rw [← Algebra.coe_norm_int, ← Int.cast_one, ← Int.cast_abs, Rat.cast_intCast, Int.cast_le]
    exact Int.one_le_abs (Algebra.norm_ne_zero_iff.mpr h_nz)
  /-
    case intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    h : LT.lt 1 (Module.finrank Rat K)
    x : NumberField.RingOfIntegers K
    h_nz : Ne x 0
    h_bd : LE.le (↑(abs ((Algebra.norm Rat) ↑x))) (HMul.hMul (HDiv.hDiv (HMul.hMul …
    h_nm : LE.le 1 ↑(abs ((Algebra.norm Rat) ↑x))
    ⊢ LE.le (HMul.hMul (4 / 9) (HPow.hPow (HDiv.hDiv (HMul.hMul 3 Real.pi) 4) (Mod …
  -/
  replace h_bd := le_trans h_nm h_bd
  rw [← inv_mul_le_iff₀ (by positivity), inv_div, mul_one, Real.le_sqrt (by positivity)
    (by positivity), ← Int.cast_abs, div_pow, mul_pow, ← pow_mul, ← pow_mul] at h_bd
  /-
    case intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    h : LT.lt 1 (Module.finrank Rat K)
    x : NumberField.RingOfIntegers K
    h_nz : Ne x 0
    h_nm : LE.le 1 ↑(abs ((Algebra.norm Rat) ↑x))
    h_bd : LE.le (HDiv.hDiv (HPow.hPow (↑(Module.finrank Rat K)) (HMul.hMul (Modul …
    ⊢ LE.le (HMul.hMul (4 / 9) (HPow.hPow (HDiv.hDiv (HMul.hMul 3 Real.pi) 4) (Mod …
  -/
  refine le_trans ?_ h_bd
  -- The sequence `a n` is a lower bound for `|discr K|`. We prove below by induction an uniform
  -- lower bound for this sequence from which we deduce the result.
  /-
    case intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    h : LT.lt 1 (Module.finrank Rat K)
    x : NumberField.RingOfIntegers K
    h_nz : Ne x 0
    h_nm : LE.le 1 ↑(abs ((Algebra.norm Rat) ↑x))
    h_bd : LE.le (HDiv.hDiv (HPow.hPow (↑(Module.finrank Rat K)) (HMul.hMul (Modul …
    ⊢ LE.le (HMul.hMul (4 / 9) (HPow.hPow (HDiv.hDiv (HMul.hMul 3 Real.pi) 4) (Mod …
  -/
  let a : ℕ → ℝ := fun n => (n : ℝ) ^ (n * 2) / ((4 / π) ^ n * (n.factorial : ℝ) ^ 2)
  suffices ∀ n, 2 ≤ n → (4 / 9 : ℝ) * (3 * π / 4) ^ n ≤ a n by
    refine le_trans (this (finrank ℚ K) h) ?_
    simp only [a]
    gcongr
    · exact (one_le_div Real.pi_pos).2 Real.pi_le_four
    · rw [← card_add_two_mul_card_eq_rank, mul_comm]
      exact Nat.le_add_left _ _
  /-
    case intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    h : LT.lt 1 (Module.finrank Rat K)
    x : NumberField.RingOfIntegers K
    h_nz : Ne x 0
    h_nm : LE.le 1 ↑(abs ((Algebra.norm Rat) ↑x))
    h_bd : LE.le (HDiv.hDiv (HPow.hPow (↑(Module.finrank Rat K)) (HMul.hMul (Modul …
    a : Nat → Real := fun n => HDiv.hDiv (HPow.hPow (↑n) (HMul.hMul n 2)) (HMul.hM …
    ⊢ ∀ (n : Nat), LE.le 2 n → LE.le (HMul.hMul (4 / 9) (HPow.hPow (HDiv.hDiv (HMu …
  -/
  intro n hn
  induction n, hn using Nat.le_induction with
  | base => exact le_of_eq <| by norm_num [a, Nat.factorial_two]; field_simp; ring
  | succ m _ h_m =>
      suffices (3 : ℝ) ≤ (1 + 1 / m : ℝ) ^ (2 * m) by
        convert_to _ ≤ (a m) * (1 + 1 / m : ℝ) ^ (2 * m) / (4 / π)
        · simp_rw [a, add_mul, one_mul, pow_succ, Nat.factorial_succ]
          field_simp; ring
        · rw [_root_.le_div_iff₀ (by positivity), pow_succ]
          convert (mul_le_mul h_m this (by positivity) (by positivity)) using 1
          field_simp; ring
      refine le_trans (le_of_eq (by field_simp; norm_num)) (one_add_mul_le_pow ?_ (2 * m))
      exact le_trans (by norm_num : (-2 : ℝ) ≤ 0) (by positivity)


/-- **Hermite-Minkowski Theorem**. A nontrivial number field has discriminant greater than `2`. -/
theorem abs_discr_gt_two (h : 1 < finrank ℚ K) : 2 < |discr K| := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    h : LT.lt 1 (Module.finrank Rat K)
    ⊢ LT.lt 2 (abs (NumberField.discr K))
  -/
  rw [← Nat.succ_le_iff] at h
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    h : LE.le (Nat.succ 1) (Module.finrank Rat K)
    ⊢ LT.lt 2 (abs (NumberField.discr K))
  -/
  rify
  calc
    (2 : ℝ) < (4 / 9) * (3 * π / 4) ^ 2 := by
      nlinarith [Real.pi_gt_three]
    _ ≤ (4 / 9 : ℝ) * (3 * π / 4) ^ finrank ℚ K := by
      gcongr
      linarith [Real.pi_gt_three]
    _ ≤ |(discr K : ℝ)| := mod_cast abs_discr_ge h


theorem finite_of_finite_generating_set {p : IntermediateField ℚ A → Prop}
    (S : Set {F : IntermediateField ℚ A // p F}) {T : Set A}
    (hT : T.Finite) (h : ∀ F ∈ S, ∃ x ∈ T, F = ℚ⟮x⟯) :
    S.Finite := by
  /-
    A : Type u_2
    inst✝¹ : Field A
    inst✝ : CharZero A
    p : IntermediateField Rat A → Prop
    S : Set (Subtype fun F => p F)
    T : Set A
    hT : T.Finite
    h : ∀ (F : Subtype fun F => p F), Membership.mem S F → Exists fun x => And (Me …
    ⊢ S.Finite
  -/
  rw [← Set.finite_coe_iff] at hT
  refine Set.finite_coe_iff.mp <| Finite.of_injective
    (fun ⟨F, hF⟩ ↦ (⟨(h F hF).choose, (h F hF).choose_spec.1⟩ : T)) (fun _ _ h_eq ↦ ?_)
  /-
    A : Type u_2
    inst✝¹ : Field A
    inst✝ : CharZero A
    p : IntermediateField Rat A → Prop
    S : Set (Subtype fun F => p F)
    T : Set A
    hT : Finite ↑T
    h : ∀ (F : Subtype fun F => p F), Membership.mem S F → Exists fun x => And (Me …
    x✝¹ x✝ : ↑S
    h_eq : Eq ((fun x => NumberField.hermiteTheorem.finite_of_finite_generating_se …
    ⊢ Eq x✝¹ x✝
  -/
  rw [Subtype.ext_iff_val, Subtype.ext_iff_val]
  /-
    A : Type u_2
    inst✝¹ : Field A
    inst✝ : CharZero A
    p : IntermediateField Rat A → Prop
    S : Set (Subtype fun F => p F)
    T : Set A
    hT : Finite ↑T
    h : ∀ (F : Subtype fun F => p F), Membership.mem S F → Exists fun x => And (Me …
    x✝¹ x✝ : ↑S
    h_eq : Eq ((fun x => NumberField.hermiteTheorem.finite_of_finite_generating_se …
    ⊢ Eq ↑↑x✝¹ ↑↑x✝
  -/
  convert congr_arg (ℚ⟮·⟯) (Subtype.mk_eq_mk.mp h_eq)
  /-
    case h.e'_2
    A : Type u_2
    inst✝¹ : Field A
    inst✝ : CharZero A
    p : IntermediateField Rat A → Prop
    S : Set (Subtype fun F => p F)
    T : Set A
    hT : Finite ↑T
    h : ∀ (F : Subtype fun F => p F), Membership.mem S F → Exists fun x => And (Me …
    x✝¹ x✝ : ↑S
    h_eq : Eq ((fun x => NumberField.hermiteTheorem.finite_of_finite_generating_se …
    ⊢ Eq (↑↑x✝¹) (IntermediateField.adjoin Rat (Singleton.singleton ⋯.choose))
  -/
  all_goals exact (h _ (Subtype.mem _)).choose_spec.2
  /-
    🎉 no goals
  -/


/-- An upper bound on the degree of a number field `K` with `|discr K| ≤ N`,
see `rank_le_rankOfDiscrBdd`. -/
noncomputable abbrev rankOfDiscrBdd : ℕ :=
  max 1 (Nat.floor ((Real.log ((9 / 4 : ℝ) * N) / Real.log (3 * π / 4))))


/-- An upper bound on the Minkowski bound of a number field `K` with `|discr K| ≤ N`;
see `minkowskiBound_lt_boundOfDiscBdd`. -/
noncomputable abbrev boundOfDiscBdd : ℝ≥0 := sqrt N * (2 : ℝ≥0) ^ rankOfDiscrBdd N + 1


include hK in
/-- If `|discr K| ≤ N` then the degree of `K` is at most `rankOfDiscrBdd`. -/
theorem rank_le_rankOfDiscrBdd :
    finrank ℚ K ≤ rankOfDiscrBdd N := by
  have h_nz : N ≠ 0 := by
    refine fun h ↦ discr_ne_zero K ?_
    rwa [h, Nat.cast_zero, abs_nonpos_iff] at hK
  have h₂ : 1 < 3 * π / 4 := by
    rw [_root_.lt_div_iff₀ (by positivity), ← _root_.div_lt_iff₀' (by positivity), one_mul]
    linarith [Real.pi_gt_three]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    N : Nat
    hK : LE.le (abs (NumberField.discr K)) ↑N
    h_nz : Ne N 0
    h₂ : LT.lt 1 (HDiv.hDiv (HMul.hMul 3 Real.pi) 4)
    ⊢ LE.le (Module.finrank Rat K) (NumberField.hermiteTheorem.rankOfDiscrBdd N)
  -/
  obtain h | h := lt_or_le 1 (finrank ℚ K)
    /-
      case inl
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      N : Nat
      hK : LE.le (abs (NumberField.discr K)) ↑N
      h_nz : Ne N 0
      h₂ : LT.lt 1 (HDiv.hDiv (HMul.hMul 3 Real.pi) 4)
      h : LT.lt 1 (Module.finrank Rat K)
      ⊢ LE.le (Module.finrank Rat K) (NumberField.hermiteTheorem.rankOfDiscrBdd N)
    -/
  · apply le_max_of_le_right
    /-
      case inl.a
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      N : Nat
      hK : LE.le (abs (NumberField.discr K)) ↑N
      h_nz : Ne N 0
      h₂ : LT.lt 1 (HDiv.hDiv (HMul.hMul 3 Real.pi) 4)
      h : LT.lt 1 (Module.finrank Rat K)
      ⊢ LE.le (Module.finrank Rat K) (Nat.floor (HDiv.hDiv (Real.log (HMul.hMul (9 / …
    -/
    rw [Nat.le_floor_iff]
      /-
        case inl.a
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        N : Nat
        hK : LE.le (abs (NumberField.discr K)) ↑N
        h_nz : Ne N 0
        h₂ : LT.lt 1 (HDiv.hDiv (HMul.hMul 3 Real.pi) 4)
        h : LT.lt 1 (Module.finrank Rat K)
        ⊢ LE.le (↑(Module.finrank Rat K)) (HDiv.hDiv (Real.log (HMul.hMul (9 / 4) ↑N)) …
      -/
    · have h := le_trans (abs_discr_ge h) (Int.cast_le.mpr hK)
      /-
        case inl.a
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        N : Nat
        hK : LE.le (abs (NumberField.discr K)) ↑N
        h_nz : Ne N 0
        h₂ : LT.lt 1 (HDiv.hDiv (HMul.hMul 3 Real.pi) 4)
        h✝ : LT.lt 1 (Module.finrank Rat K)
        h : LE.le (HMul.hMul (4 / 9) (HPow.hPow (HDiv.hDiv (HMul.hMul 3 Real.pi) 4) (M …
        ⊢ LE.le (↑(Module.finrank Rat K)) (HDiv.hDiv (Real.log (HMul.hMul (9 / 4) ↑N)) …
      -/
      contrapose! h
      /-
        case inl.a
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        N : Nat
        hK : LE.le (abs (NumberField.discr K)) ↑N
        h_nz : Ne N 0
        h₂ : LT.lt 1 (HDiv.hDiv (HMul.hMul 3 Real.pi) 4)
        h✝ : LT.lt 1 (Module.finrank Rat K)
        h : LT.lt (HDiv.hDiv (Real.log (HMul.hMul (9 / 4) ↑N)) (Real.log (HDiv.hDiv (H …
        ⊢ LT.lt (↑↑N) (HMul.hMul (4 / 9) (HPow.hPow (HDiv.hDiv (HMul.hMul 3 Real.pi) 4 …
      -/
      rw [← Real.rpow_natCast]
      /-
        case inl.a
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        N : Nat
        hK : LE.le (abs (NumberField.discr K)) ↑N
        h_nz : Ne N 0
        h₂ : LT.lt 1 (HDiv.hDiv (HMul.hMul 3 Real.pi) 4)
        h✝ : LT.lt 1 (Module.finrank Rat K)
        h : LT.lt (HDiv.hDiv (Real.log (HMul.hMul (9 / 4) ↑N)) (Real.log (HDiv.hDiv (H …
        ⊢ LT.lt (↑↑N) (HMul.hMul (4 / 9) (HPow.hPow (HDiv.hDiv (HMul.hMul 3 Real.pi) 4 …
      -/
      rw [Real.log_div_log] at h
      refine lt_of_le_of_lt ?_ (mul_lt_mul_of_pos_left
        (Real.rpow_lt_rpow_of_exponent_lt h₂ h) (by positivity : (0 : ℝ) < 4 / 9))
      rw [Real.rpow_logb (lt_trans zero_lt_one h₂) (ne_of_gt h₂) (by positivity), ← mul_assoc,
            ← inv_div, inv_mul_cancel₀ (by norm_num), one_mul, Int.cast_natCast]
      /-
        case inl.a
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        N : Nat
        hK : LE.le (abs (NumberField.discr K)) ↑N
        h_nz : Ne N 0
        h₂ : LT.lt 1 (HDiv.hDiv (HMul.hMul 3 Real.pi) 4)
        h : LT.lt 1 (Module.finrank Rat K)
        ⊢ LE.le 0 (HDiv.hDiv (Real.log (HMul.hMul (9 / 4) ↑N)) (Real.log (HDiv.hDiv (H …
      -/
    · refine div_nonneg (Real.log_nonneg ?_) (Real.log_nonneg (le_of_lt h₂))
      rw [mul_comm, ← mul_div_assoc, _root_.le_div_iff₀ (by positivity), one_mul,
        ← _root_.div_le_iff₀ (by positivity)]
      /-
        case inl.a
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        N : Nat
        hK : LE.le (abs (NumberField.discr K)) ↑N
        h_nz : Ne N 0
        h₂ : LT.lt 1 (HDiv.hDiv (HMul.hMul 3 Real.pi) 4)
        h : LT.lt 1 (Module.finrank Rat K)
        ⊢ LE.le (4 / 9) ↑N
      -/
      exact le_trans (by norm_num) (Nat.one_le_cast.mpr (Nat.one_le_iff_ne_zero.mpr h_nz))
      /-
        🎉 no goals
      -/
    /-
      case inr
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      N : Nat
      hK : LE.le (abs (NumberField.discr K)) ↑N
      h_nz : Ne N 0
      h₂ : LT.lt 1 (HDiv.hDiv (HMul.hMul 3 Real.pi) 4)
      h : LE.le (Module.finrank Rat K) 1
      ⊢ LE.le (Module.finrank Rat K) (NumberField.hermiteTheorem.rankOfDiscrBdd N)
    -/
  · exact le_max_of_le_left h
    /-
      🎉 no goals
    -/


include hK in
/-- If `|discr K| ≤ N` then the Minkowski bound of `K` is less than `boundOfDiscrBdd`. -/
theorem minkowskiBound_lt_boundOfDiscBdd : minkowskiBound K ↑1 < boundOfDiscBdd N := by
  have : boundOfDiscBdd N - 1 < boundOfDiscBdd N := by
    simp_rw [boundOfDiscBdd, add_tsub_cancel_right, lt_add_iff_pos_right, zero_lt_one]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    N : Nat
    hK : LE.le (abs (NumberField.discr K)) ↑N
    this : LT.lt (HSub.hSub (NumberField.hermiteTheorem.boundOfDiscBdd N) 1) (Numb …
    ⊢ LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) ↑(NumberField.hermiteT …
  -/
  refine lt_of_le_of_lt ?_ (coe_lt_coe.mpr this)
  rw [minkowskiBound, volume_fundamentalDomain_fractionalIdealLatticeBasis, boundOfDiscBdd,
    add_tsub_cancel_right, Units.val_one, FractionalIdeal.absNorm_one, Rat.cast_one,
    ENNReal.ofReal_one, one_mul, mixedEmbedding.finrank, volume_fundamentalDomain_latticeBasis,
    coe_mul, ENNReal.coe_pow, coe_ofNat, show sqrt N = (1 : ℝ≥0∞) * sqrt N by rw [one_mul]]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    N : Nat
    hK : LE.le (abs (NumberField.discr K)) ↑N
    this : LT.lt (HSub.hSub (NumberField.hermiteTheorem.boundOfDiscBdd N) 1) (Numb …
    ⊢ LE.le (HMul.hMul (HMul.hMul (HPow.hPow (Inv.inv 2) (NumberField.InfinitePlac …
  -/
  gcongr
    /-
      case h₁.h₁
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      N : Nat
      hK : LE.le (abs (NumberField.discr K)) ↑N
      this : LT.lt (HSub.hSub (NumberField.hermiteTheorem.boundOfDiscBdd N) 1) (Numb …
      ⊢ LE.le (HPow.hPow (Inv.inv 2) (NumberField.InfinitePlace.nrComplexPlaces K)) 1
    -/
  · exact pow_le_one₀ (by positivity) (by norm_num)
    /-
      🎉 no goals
    -/
  · rwa [← NNReal.coe_le_coe, coe_nnnorm, Int.norm_eq_abs, ← Int.cast_abs,
      NNReal.coe_natCast, ← Int.cast_natCast, Int.cast_le]
    /-
      case h₂.ha
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      N : Nat
      hK : LE.le (abs (NumberField.discr K)) ↑N
      this : LT.lt (HSub.hSub (NumberField.hermiteTheorem.boundOfDiscBdd N) 1) (Numb …
      ⊢ LE.le 1 2
    -/
  · exact one_le_two
    /-
      🎉 no goals
    -/
    /-
      case h₂.hmn
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      N : Nat
      hK : LE.le (abs (NumberField.discr K)) ↑N
      this : LT.lt (HSub.hSub (NumberField.hermiteTheorem.boundOfDiscBdd N) 1) (Numb …
      ⊢ LE.le (Module.finrank Rat K) (NumberField.hermiteTheorem.rankOfDiscrBdd N)
    -/
  · exact rank_le_rankOfDiscrBdd hK
    /-
      🎉 no goals
    -/


include hK in
theorem natDegree_le_rankOfDiscrBdd (a : 𝓞 K) (h : ℚ⟮(a : K)⟯ = ⊤) :
    natDegree (minpoly ℤ (a : K)) ≤ rankOfDiscrBdd N := by
  rw [Field.primitive_element_iff_minpoly_natDegree_eq,
    minpoly.isIntegrallyClosed_eq_field_fractions' ℚ a.isIntegral_coe,
    (minpoly.monic a.isIntegral_coe).natDegree_map] at h
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    N : Nat
    hK : LE.le (abs (NumberField.discr K)) ↑N
    a : NumberField.RingOfIntegers K
    h : Eq (minpoly Int ((algebraMap (NumberField.RingOfIntegers K) K) a)).natDegr …
    ⊢ LE.le (minpoly Int ↑a).natDegree (NumberField.hermiteTheorem.rankOfDiscrBdd N)
  -/
  exact h.symm ▸ rank_le_rankOfDiscrBdd hK
  /-
    🎉 no goals
  -/


theorem finite_of_discr_bdd_of_isReal :
    {K : { F : IntermediateField ℚ A // FiniteDimensional ℚ F} |
      haveI :  NumberField K := @NumberField.mk _ _ inferInstance K.prop
      {w : InfinitePlace K | IsReal w}.Nonempty ∧ |discr K| ≤ N }.Finite := by
  classical
  -- The bound on the degree of the generating polynomials
  let D := rankOfDiscrBdd N
  -- The bound on the Minkowski bound
  let B := boundOfDiscBdd N
  -- The bound on the coefficients of the generating polynomials
  let C := Nat.ceil ((max B 1) ^ D *  Nat.choose D (D / 2))
  refine finite_of_finite_generating_set A _ (bUnion_roots_finite (algebraMap ℤ A) D
      (Set.finite_Icc (-C : ℤ) C)) (fun ⟨K, hK₀⟩ ⟨hK₁, hK₂⟩ ↦ ?_)
  -- We now need to prove that each field is generated by an element of the union of the rootset
  simp_rw [Set.mem_iUnion]
  -- this is purely an optimization
  have : CharZero K := SubsemiringClass.instCharZero K
  haveI : NumberField K := @NumberField.mk _ _ inferInstance hK₀
  obtain ⟨w₀, hw₀⟩ := hK₁
  suffices minkowskiBound K ↑1 < (convexBodyLTFactor K) * B by
    obtain ⟨x, hx₁, hx₂⟩ := exists_primitive_element_lt_of_isReal K hw₀ this
    have hx := x.isIntegral_coe
    refine ⟨x, ⟨⟨minpoly ℤ (x : K), ⟨?_, fun i ↦ ?_⟩, ?_⟩, ?_⟩⟩
    · exact natDegree_le_rankOfDiscrBdd hK₂ x hx₁
    · rw [Set.mem_Icc, ← abs_le, ← @Int.cast_le ℝ]
      refine (Eq.trans_le ?_ <| Embeddings.coeff_bdd_of_norm_le
          ((le_iff_le (x : K) _).mp (fun w ↦ le_of_lt (hx₂ w))) i).trans ?_
      · rw [minpoly.isIntegrallyClosed_eq_field_fractions' ℚ hx, coeff_map, eq_intCast,
          Int.norm_cast_rat, Int.norm_eq_abs, Int.cast_abs]
      · refine le_trans ?_ (Nat.le_ceil _)
        rw [show max ↑(max (B : ℝ≥0) 1) (1 : ℝ) = max (B : ℝ) 1 by simp, val_eq_coe, NNReal.coe_mul,
          NNReal.coe_pow, NNReal.coe_max, NNReal.coe_one, NNReal.coe_natCast]
        gcongr
        · exact le_max_right _ 1
        · exact rank_le_rankOfDiscrBdd hK₂
        · exact (Nat.choose_le_choose _ (rank_le_rankOfDiscrBdd hK₂)).trans
            (Nat.choose_le_middle _ _)
    · refine mem_rootSet.mpr ⟨minpoly.ne_zero hx, ?_⟩
      exact (aeval_algebraMap_eq_zero_iff A (x : K) _).mpr (minpoly.aeval ℤ (x : K))
    · rw [← (IntermediateField.lift_injective _).eq_iff, eq_comm] at hx₁
      convert hx₁
      · simp only [IntermediateField.lift_top]
      · simp only [IntermediateField.lift_adjoin, Set.image_singleton]
  calc
    minkowskiBound K 1 < B := minkowskiBound_lt_boundOfDiscBdd hK₂
    _ = 1 * B := by rw [one_mul]
    _ ≤ convexBodyLTFactor K * B := by gcongr; exact mod_cast one_le_convexBodyLTFactor K


theorem finite_of_discr_bdd_of_isComplex :
    {K : { F : IntermediateField ℚ A // FiniteDimensional ℚ F} |
      haveI :  NumberField K := @NumberField.mk _ _ inferInstance K.prop
      {w : InfinitePlace K | IsComplex w}.Nonempty ∧ |discr K| ≤ N }.Finite := by
  classical
  -- The bound on the degree of the generating polynomials
  let D := rankOfDiscrBdd N
  -- The bound on the Minkowski bound
  let B := boundOfDiscBdd N
  -- The bound on the coefficients of the generating polynomials
  let C := Nat.ceil ((max (sqrt (1 + B ^ 2)) 1) ^ D * Nat.choose D (D / 2))
  refine finite_of_finite_generating_set A _ (bUnion_roots_finite (algebraMap ℤ A) D
      (Set.finite_Icc (-C : ℤ) C)) (fun ⟨K, hK₀⟩ ⟨hK₁, hK₂⟩ ↦ ?_)
  -- We now need to prove that each field is generated by an element of the union of the rootset
  simp_rw [Set.mem_iUnion]
  -- this is purely an optimization
  have : CharZero K := SubsemiringClass.instCharZero K
  haveI : NumberField K := @NumberField.mk _ _ inferInstance hK₀
  obtain ⟨w₀, hw₀⟩ := hK₁
  suffices minkowskiBound K ↑1 < (convexBodyLT'Factor K) * boundOfDiscBdd N by
    obtain ⟨x, hx₁, hx₂⟩ := exists_primitive_element_lt_of_isComplex K hw₀ this
    have hx := x.isIntegral_coe
    refine ⟨x, ⟨⟨minpoly ℤ (x : K), ⟨?_, fun i ↦ ?_⟩, ?_⟩, ?_⟩⟩
    · exact natDegree_le_rankOfDiscrBdd hK₂ x hx₁
    · rw [Set.mem_Icc, ← abs_le, ← @Int.cast_le ℝ]
      refine (Eq.trans_le ?_ <| Embeddings.coeff_bdd_of_norm_le
          ((le_iff_le (x : K) _).mp (fun w ↦ le_of_lt (hx₂ w))) i).trans ?_
      · rw [minpoly.isIntegrallyClosed_eq_field_fractions' ℚ hx, coeff_map, eq_intCast,
          Int.norm_cast_rat, Int.norm_eq_abs, Int.cast_abs]
      · refine le_trans ?_ (Nat.le_ceil _)
        rw [val_eq_coe, NNReal.coe_mul, NNReal.coe_pow, NNReal.coe_max, NNReal.coe_one,
          Real.coe_sqrt, NNReal.coe_add 1, NNReal.coe_one, NNReal.coe_pow]
        gcongr
        · exact le_max_right _ 1
        · exact rank_le_rankOfDiscrBdd hK₂
        · rw [NNReal.coe_natCast, Nat.cast_le]
          exact (Nat.choose_le_choose _ (rank_le_rankOfDiscrBdd hK₂)).trans
            (Nat.choose_le_middle _ _)
    · refine mem_rootSet.mpr ⟨minpoly.ne_zero hx, ?_⟩
      exact (aeval_algebraMap_eq_zero_iff A (x : K) _).mpr (minpoly.aeval ℤ (x : K))
    · rw [← (IntermediateField.lift_injective _).eq_iff, eq_comm] at hx₁
      convert hx₁
      · simp only [IntermediateField.lift_top]
      · simp only [IntermediateField.lift_adjoin, Set.image_singleton]
  calc
    minkowskiBound K 1 < B := minkowskiBound_lt_boundOfDiscBdd hK₂
    _ = 1 * B := by rw [one_mul]
    _ ≤ convexBodyLT'Factor K * B := by gcongr; exact mod_cast one_le_convexBodyLT'Factor K


/-- **Hermite Theorem**. Let `N` be an integer. There are only finitely many number fields
(in some fixed extension of `ℚ`) of discriminant bounded by `N`. -/
theorem _root_.NumberField.finite_of_discr_bdd :
    {K : { F : IntermediateField ℚ A // FiniteDimensional ℚ F} |
      haveI :  NumberField K := @NumberField.mk _ _ inferInstance K.prop
      |discr K| ≤ N }.Finite := by
  refine Set.Finite.subset (Set.Finite.union (finite_of_discr_bdd_of_isReal A N)
    (finite_of_discr_bdd_of_isComplex A N)) ?_
  /-
    A : Type u_2
    inst✝¹ : Field A
    inst✝ : CharZero A
    N : Nat
    ⊢ HasSubset.Subset (setOf fun K => LE.le (abs (NumberField.discr (Subtype fun  …
  -/
  rintro ⟨K, hK₀⟩ hK₁
  -- this is purely an optimization
  /-
    case mk
    A : Type u_2
    inst✝¹ : Field A
    inst✝ : CharZero A
    N : Nat
    K : IntermediateField Rat A
    hK₀ : FiniteDimensional Rat (Subtype fun x => Membership.mem K x)
    hK₁ : Membership.mem (setOf fun K => LE.le (abs (NumberField.discr (Subtype fu …
    ⊢ Membership.mem (Union.union (setOf fun K => And (setOf fun w => w.IsReal).No …
  -/
  have : CharZero K := SubsemiringClass.instCharZero K
  /-
    case mk
    A : Type u_2
    inst✝¹ : Field A
    inst✝ : CharZero A
    N : Nat
    K : IntermediateField Rat A
    hK₀ : FiniteDimensional Rat (Subtype fun x => Membership.mem K x)
    hK₁ : Membership.mem (setOf fun K => LE.le (abs (NumberField.discr (Subtype fu …
    this : CharZero (Subtype fun x => Membership.mem K x)
    ⊢ Membership.mem (Union.union (setOf fun K => And (setOf fun w => w.IsReal).No …
  -/
  haveI : NumberField K := @NumberField.mk _ _ inferInstance hK₀
  /-
    case mk
    A : Type u_2
    inst✝¹ : Field A
    inst✝ : CharZero A
    N : Nat
    K : IntermediateField Rat A
    hK₀ : FiniteDimensional Rat (Subtype fun x => Membership.mem K x)
    hK₁ : Membership.mem (setOf fun K => LE.le (abs (NumberField.discr (Subtype fu …
    this✝ : CharZero (Subtype fun x => Membership.mem K x)
    this : NumberField (Subtype fun x => Membership.mem K x)
    ⊢ Membership.mem (Union.union (setOf fun K => And (setOf fun w => w.IsReal).No …
  -/
  obtain ⟨w₀⟩ := (inferInstance : Nonempty (InfinitePlace K))
  /-
    case mk.intro
    A : Type u_2
    inst✝¹ : Field A
    inst✝ : CharZero A
    N : Nat
    K : IntermediateField Rat A
    hK₀ : FiniteDimensional Rat (Subtype fun x => Membership.mem K x)
    hK₁ : Membership.mem (setOf fun K => LE.le (abs (NumberField.discr (Subtype fu …
    this✝ : CharZero (Subtype fun x => Membership.mem K x)
    this : NumberField (Subtype fun x => Membership.mem K x)
    w₀ : NumberField.InfinitePlace (Subtype fun x => Membership.mem K x)
    ⊢ Membership.mem (Union.union (setOf fun K => And (setOf fun w => w.IsReal).No …
  -/
  by_cases hw₀ : IsReal w₀
    /-
      case pos
      A : Type u_2
      inst✝¹ : Field A
      inst✝ : CharZero A
      N : Nat
      K : IntermediateField Rat A
      hK₀ : FiniteDimensional Rat (Subtype fun x => Membership.mem K x)
      hK₁ : Membership.mem (setOf fun K => LE.le (abs (NumberField.discr (Subtype fu …
      this✝ : CharZero (Subtype fun x => Membership.mem K x)
      this : NumberField (Subtype fun x => Membership.mem K x)
      w₀ : NumberField.InfinitePlace (Subtype fun x => Membership.mem K x)
      hw₀ : w₀.IsReal
      ⊢ Membership.mem (Union.union (setOf fun K => And (setOf fun w => w.IsReal).No …
    -/
  · apply Set.mem_union_left
    /-
      case pos.a
      A : Type u_2
      inst✝¹ : Field A
      inst✝ : CharZero A
      N : Nat
      K : IntermediateField Rat A
      hK₀ : FiniteDimensional Rat (Subtype fun x => Membership.mem K x)
      hK₁ : Membership.mem (setOf fun K => LE.le (abs (NumberField.discr (Subtype fu …
      this✝ : CharZero (Subtype fun x => Membership.mem K x)
      this : NumberField (Subtype fun x => Membership.mem K x)
      w₀ : NumberField.InfinitePlace (Subtype fun x => Membership.mem K x)
      hw₀ : w₀.IsReal
      ⊢ Membership.mem (setOf fun K => And (setOf fun w => w.IsReal).Nonempty (LE.le …
    -/
    exact ⟨⟨w₀, hw₀⟩, hK₁⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      A : Type u_2
      inst✝¹ : Field A
      inst✝ : CharZero A
      N : Nat
      K : IntermediateField Rat A
      hK₀ : FiniteDimensional Rat (Subtype fun x => Membership.mem K x)
      hK₁ : Membership.mem (setOf fun K => LE.le (abs (NumberField.discr (Subtype fu …
      this✝ : CharZero (Subtype fun x => Membership.mem K x)
      this : NumberField (Subtype fun x => Membership.mem K x)
      w₀ : NumberField.InfinitePlace (Subtype fun x => Membership.mem K x)
      hw₀ : Not w₀.IsReal
      ⊢ Membership.mem (Union.union (setOf fun K => And (setOf fun w => w.IsReal).No …
    -/
  · apply Set.mem_union_right
    /-
      case neg.a
      A : Type u_2
      inst✝¹ : Field A
      inst✝ : CharZero A
      N : Nat
      K : IntermediateField Rat A
      hK₀ : FiniteDimensional Rat (Subtype fun x => Membership.mem K x)
      hK₁ : Membership.mem (setOf fun K => LE.le (abs (NumberField.discr (Subtype fu …
      this✝ : CharZero (Subtype fun x => Membership.mem K x)
      this : NumberField (Subtype fun x => Membership.mem K x)
      w₀ : NumberField.InfinitePlace (Subtype fun x => Membership.mem K x)
      hw₀ : Not w₀.IsReal
      ⊢ Membership.mem (setOf fun K => And (setOf fun w => w.IsComplex).Nonempty (LE …
    -/
    exact ⟨⟨w₀, not_isReal_iff_isComplex.mp hw₀⟩, hK₁⟩
    /-
      🎉 no goals
    -/



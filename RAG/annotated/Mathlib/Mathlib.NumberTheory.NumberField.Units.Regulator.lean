/-- The regulator of a number field `K`. -/
                     /-
                       K : Type u_1
                       inst✝¹ : Field K
                       inst✝ : NumberField K
                       ⊢ MeasureTheory.Measure ((Subtype fun w => Ne w NumberField.Units.dirichletUni …
                     -/
def regulator : ℝ := ZLattice.covolume (unitLattice K)
                     /-
                       🎉 no goals
                     -/


theorem regulator_ne_zero : regulator K ≠ 0 := ZLattice.covolume_ne_zero (unitLattice K) volume


theorem regulator_pos : 0 < regulator K := ZLattice.covolume_pos (unitLattice K) volume


theorem regulator_eq_det' (e : {w : InfinitePlace K // w ≠ w₀} ≃ Fin (rank K)) :
    regulator K = |(Matrix.of fun i ↦
      logEmbedding K (Additive.ofMul (fundSystem K (e i)))).det| := by
  simp_rw [regulator, ZLattice.covolume_eq_det _
    (((basisModTorsion K).map (logEmbeddingEquiv K)).reindex e.symm), Basis.coe_reindex,
    Function.comp_def, Basis.map_apply, ← fundSystem_mk, Equiv.symm_symm, logEmbeddingEquiv_apply]


/-- Let `u : Fin (rank K) → (𝓞 K)ˣ` be a family of units and let `w₁` and `w₂` be two infinite
places. Then, the two square matrices with entries `(mult w * log w (u i))_i, {w ≠ w_i}`, `i = 1,2`,
have the same determinant in absolute value. -/
theorem abs_det_eq_abs_det (u : Fin (rank K) → (𝓞 K)ˣ)
    {w₁ w₂ : InfinitePlace K} (e₁ : {w // w ≠ w₁} ≃ Fin (rank K))
    (e₂ : {w // w ≠ w₂} ≃ Fin (rank K)) :
    |(Matrix.of fun i w : {w // w ≠ w₁} ↦ (mult w.val : ℝ) * (w.val (u (e₁ i) : K)).log).det| =
    |(Matrix.of fun i w : {w // w ≠ w₂} ↦ (mult w.val : ℝ) * (w.val (u (e₂ i) : K)).log).det| := by
  -- We construct an equiv `Fin (rank K + 1) ≃ InfinitePlace K` from `e₂.symm`
  let f : Fin (rank K + 1) ≃ InfinitePlace K :=
    (finSuccEquiv _).trans ((Equiv.optionSubtype _).symm e₁.symm).val
  -- And `g` corresponds to the restriction of `f⁻¹` to `{w // w ≠ w₂}`
  let g : {w // w ≠ w₂} ≃ Fin (rank K) :=
    (Equiv.subtypeEquiv f.symm (fun _ ↦ by simp [f])).trans
      (finSuccAboveEquiv (f.symm w₂)).symm
  have h_col := congr_arg abs <| Matrix.det_permute (g.trans e₂.symm)
    (Matrix.of fun i w : {w // w ≠ w₂} ↦ (mult w.val : ℝ) * (w.val (u (e₂ i) : K)).log)
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    u : Fin (NumberField.Units.rank K) → Units (NumberField.RingOfIntegers K)
    w₁ w₂ : NumberField.InfinitePlace K
    e₁ : Equiv (Subtype fun w => Ne w w₁) (Fin (NumberField.Units.rank K))
    e₂ : Equiv (Subtype fun w => Ne w w₂) (Fin (NumberField.Units.rank K))
    f : Equiv (Fin (HAdd.hAdd (NumberField.Units.rank K) 1)) (NumberField.Infinite …
    g : Equiv (Subtype fun w => Ne w w₂) (Fin (NumberField.Units.rank K)) := (f.sy …
    h_col : Eq (abs ((Matrix.of fun i w => HMul.hMul (↑(↑w).mult) (Real.log (↑w (( …
    ⊢ Eq (abs (Matrix.of fun i w => HMul.hMul (↑(↑w).mult) (Real.log (↑w ((algebra …
  -/
  rw [abs_mul, ← Int.cast_abs, Equiv.Perm.sign_abs, Int.cast_one, one_mul] at h_col
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    u : Fin (NumberField.Units.rank K) → Units (NumberField.RingOfIntegers K)
    w₁ w₂ : NumberField.InfinitePlace K
    e₁ : Equiv (Subtype fun w => Ne w w₁) (Fin (NumberField.Units.rank K))
    e₂ : Equiv (Subtype fun w => Ne w w₂) (Fin (NumberField.Units.rank K))
    f : Equiv (Fin (HAdd.hAdd (NumberField.Units.rank K) 1)) (NumberField.Infinite …
    g : Equiv (Subtype fun w => Ne w w₂) (Fin (NumberField.Units.rank K)) := (f.sy …
    h_col : Eq (abs ((Matrix.of fun i w => HMul.hMul (↑(↑w).mult) (Real.log (↑w (( …
    ⊢ Eq (abs (Matrix.of fun i w => HMul.hMul (↑(↑w).mult) (Real.log (↑w ((algebra …
  -/
  rw [← h_col]
  have h := congr_arg abs <| Matrix.submatrix_succAbove_det_eq_negOnePow_submatrix_succAbove_det'
    (Matrix.of fun i w ↦ (mult (f w) : ℝ) * ((f w) (u i)).log) ?_ 0 (f.symm w₂)
    /-
      case refine_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      u : Fin (NumberField.Units.rank K) → Units (NumberField.RingOfIntegers K)
      w₁ w₂ : NumberField.InfinitePlace K
      e₁ : Equiv (Subtype fun w => Ne w w₁) (Fin (NumberField.Units.rank K))
      e₂ : Equiv (Subtype fun w => Ne w w₂) (Fin (NumberField.Units.rank K))
      f : Equiv (Fin (HAdd.hAdd (NumberField.Units.rank K) 1)) (NumberField.Infinite …
      g : Equiv (Subtype fun w => Ne w w₂) (Fin (NumberField.Units.rank K)) := (f.sy …
      h_col : Eq (abs ((Matrix.of fun i w => HMul.hMul (↑(↑w).mult) (Real.log (↑w (( …
      h : Eq (abs ((Matrix.of fun i w => HMul.hMul (↑(f w).mult) (Real.log ((f w) (( …
      ⊢ Eq (abs (Matrix.of fun i w => HMul.hMul (↑(↑w).mult) (Real.log (↑w ((algebra …
    -/
  · rw [← Matrix.det_reindex_self e₁, ← Matrix.det_reindex_self g]
      /-
        case refine_2
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        u : Fin (NumberField.Units.rank K) → Units (NumberField.RingOfIntegers K)
        w₁ w₂ : NumberField.InfinitePlace K
        e₁ : Equiv (Subtype fun w => Ne w w₁) (Fin (NumberField.Units.rank K))
        e₂ : Equiv (Subtype fun w => Ne w w₂) (Fin (NumberField.Units.rank K))
        f : Equiv (Fin (HAdd.hAdd (NumberField.Units.rank K) 1)) (NumberField.Infinite …
        g : Equiv (Subtype fun w => Ne w w₂) (Fin (NumberField.Units.rank K)) := (f.sy …
        h_col : Eq (abs ((Matrix.of fun i w => HMul.hMul (↑(↑w).mult) (Real.log (↑w (( …
        h : Eq (abs ((Matrix.of fun i w => HMul.hMul (↑(f w).mult) (Real.log ((f w) (( …
        ⊢ Eq (abs ((Matrix.reindex e₁ e₁) (Matrix.of fun i w => HMul.hMul (↑(↑w).mult) …
      -/
    · rw [Units.smul_def, abs_zsmul, Int.abs_negOnePow, one_smul] at h
      /-
        case refine_2
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        u : Fin (NumberField.Units.rank K) → Units (NumberField.RingOfIntegers K)
        w₁ w₂ : NumberField.InfinitePlace K
        e₁ : Equiv (Subtype fun w => Ne w w₁) (Fin (NumberField.Units.rank K))
        e₂ : Equiv (Subtype fun w => Ne w w₂) (Fin (NumberField.Units.rank K))
        f : Equiv (Fin (HAdd.hAdd (NumberField.Units.rank K) 1)) (NumberField.Infinite …
        g : Equiv (Subtype fun w => Ne w w₂) (Fin (NumberField.Units.rank K)) := (f.sy …
        h_col : Eq (abs ((Matrix.of fun i w => HMul.hMul (↑(↑w).mult) (Real.log (↑w (( …
        h : Eq (abs ((Matrix.of fun i w => HMul.hMul (↑(f w).mult) (Real.log ((f w) (( …
        ⊢ Eq (abs ((Matrix.reindex e₁ e₁) (Matrix.of fun i w => HMul.hMul (↑(↑w).mult) …
      -/
      convert h
        /-
          case h.e'_2.h.e'_4.h.e'_6
          K : Type u_1
          inst✝¹ : Field K
          inst✝ : NumberField K
          u : Fin (NumberField.Units.rank K) → Units (NumberField.RingOfIntegers K)
          w₁ w₂ : NumberField.InfinitePlace K
          e₁ : Equiv (Subtype fun w => Ne w w₁) (Fin (NumberField.Units.rank K))
          e₂ : Equiv (Subtype fun w => Ne w w₂) (Fin (NumberField.Units.rank K))
          f : Equiv (Fin (HAdd.hAdd (NumberField.Units.rank K) 1)) (NumberField.Infinite …
          g : Equiv (Subtype fun w => Ne w w₂) (Fin (NumberField.Units.rank K)) := (f.sy …
          h_col : Eq (abs ((Matrix.of fun i w => HMul.hMul (↑(↑w).mult) (Real.log (↑w (( …
          h : Eq (abs ((Matrix.of fun i w => HMul.hMul (↑(f w).mult) (Real.log ((f w) (( …
          ⊢ Eq ((Matrix.reindex e₁ e₁) (Matrix.of fun i w => HMul.hMul (↑(↑w).mult) (Rea …
        -/
      · ext; simp only [ne_eq, Matrix.reindex_apply, Matrix.submatrix_apply, Matrix.of_apply,
          Equiv.apply_symm_apply, Equiv.trans_apply, Fin.succAbove_zero, id_eq, finSuccEquiv_succ,
          Equiv.optionSubtype_symm_apply_apply_coe, f]
        /-
          case h.e'_3.h.e'_4.h.e'_6
          K : Type u_1
          inst✝¹ : Field K
          inst✝ : NumberField K
          u : Fin (NumberField.Units.rank K) → Units (NumberField.RingOfIntegers K)
          w₁ w₂ : NumberField.InfinitePlace K
          e₁ : Equiv (Subtype fun w => Ne w w₁) (Fin (NumberField.Units.rank K))
          e₂ : Equiv (Subtype fun w => Ne w w₂) (Fin (NumberField.Units.rank K))
          f : Equiv (Fin (HAdd.hAdd (NumberField.Units.rank K) 1)) (NumberField.Infinite …
          g : Equiv (Subtype fun w => Ne w w₂) (Fin (NumberField.Units.rank K)) := (f.sy …
          h_col : Eq (abs ((Matrix.of fun i w => HMul.hMul (↑(↑w).mult) (Real.log (↑w (( …
          h : Eq (abs ((Matrix.of fun i w => HMul.hMul (↑(f w).mult) (Real.log ((f w) (( …
          ⊢ Eq ((Matrix.reindex g g) ((Matrix.of fun i w => HMul.hMul (↑(↑w).mult) (Real …
        -/
      · ext; simp only [ne_eq, Equiv.coe_trans, Matrix.reindex_apply, Matrix.submatrix_apply,
                                                                                /-
                                                                                  case h.e'_3.h.e'_4.h.e'_6.a
                                                                                  K : Type u_1
                                                                                  inst✝¹ : Field K
                                                                                  inst✝ : NumberField K
                                                                                  u : Fin (NumberField.Units.rank K) → Units (NumberField.RingOfIntegers K)
                                                                                  w₁ w₂ : NumberField.InfinitePlace K
                                                                                  e₁ : Equiv (Subtype fun w => Ne w w₁) (Fin (NumberField.Units.rank K))
                                                                                  e₂ : Equiv (Subtype fun w => Ne w w₂) (Fin (NumberField.Units.rank K))
                                                                                  f : Equiv (Fin (HAdd.hAdd (NumberField.Units.rank K) 1)) (NumberField.Infinite …
                                                                                  g : Equiv (Subtype fun w => Ne w w₂) (Fin (NumberField.Units.rank K)) := (f.sy …
                                                                                  h_col : Eq (abs ((Matrix.of fun i w => HMul.hMul (↑(↑w).mult) (Real.log (↑w (( …
                                                                                  h : Eq (abs ((Matrix.of fun i w => HMul.hMul (↑(f w).mult) (Real.log ((f w) (( …
                                                                                  i✝ j✝ : Fin (NumberField.Units.rank K)
                                                                                  ⊢ Eq (HMul.hMul (↑(↑(g.symm j✝)).mult) (Real.log (↑(g.symm j✝) ((algebraMap (N …
                                                                                -/
          Function.comp_apply, Equiv.apply_symm_apply, id_eq, Matrix.of_apply]; rfl
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
    /-
      case refine_1
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      u : Fin (NumberField.Units.rank K) → Units (NumberField.RingOfIntegers K)
      w₁ w₂ : NumberField.InfinitePlace K
      e₁ : Equiv (Subtype fun w => Ne w w₁) (Fin (NumberField.Units.rank K))
      e₂ : Equiv (Subtype fun w => Ne w w₂) (Fin (NumberField.Units.rank K))
      f : Equiv (Fin (HAdd.hAdd (NumberField.Units.rank K) 1)) (NumberField.Infinite …
      g : Equiv (Subtype fun w => Ne w w₂) (Fin (NumberField.Units.rank K)) := (f.sy …
      h_col : Eq (abs ((Matrix.of fun i w => HMul.hMul (↑(↑w).mult) (Real.log (↑w (( …
      ⊢ ∀ (i : Fin (NumberField.Units.rank K)), Eq (Finset.univ.sum fun j => Matrix. …
    -/
  · intro _
    /-
      case refine_1
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      u : Fin (NumberField.Units.rank K) → Units (NumberField.RingOfIntegers K)
      w₁ w₂ : NumberField.InfinitePlace K
      e₁ : Equiv (Subtype fun w => Ne w w₁) (Fin (NumberField.Units.rank K))
      e₂ : Equiv (Subtype fun w => Ne w w₂) (Fin (NumberField.Units.rank K))
      f : Equiv (Fin (HAdd.hAdd (NumberField.Units.rank K) 1)) (NumberField.Infinite …
      g : Equiv (Subtype fun w => Ne w w₂) (Fin (NumberField.Units.rank K)) := (f.sy …
      h_col : Eq (abs ((Matrix.of fun i w => HMul.hMul (↑(↑w).mult) (Real.log (↑w (( …
      i✝ : Fin (NumberField.Units.rank K)
      ⊢ Eq (Finset.univ.sum fun j => Matrix.of (fun i w => HMul.hMul (↑(f w).mult) ( …
    -/
    simp_rw [Matrix.of_apply, ← Real.log_pow]
    rw [← Real.log_prod, Equiv.prod_comp f (fun w ↦ (w (u _) ^ (mult w))), prod_eq_abs_norm,
      Units.norm, Rat.cast_one, Real.log_one]
    /-
      case refine_1.hf
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      u : Fin (NumberField.Units.rank K) → Units (NumberField.RingOfIntegers K)
      w₁ w₂ : NumberField.InfinitePlace K
      e₁ : Equiv (Subtype fun w => Ne w w₁) (Fin (NumberField.Units.rank K))
      e₂ : Equiv (Subtype fun w => Ne w w₂) (Fin (NumberField.Units.rank K))
      f : Equiv (Fin (HAdd.hAdd (NumberField.Units.rank K) 1)) (NumberField.Infinite …
      g : Equiv (Subtype fun w => Ne w w₂) (Fin (NumberField.Units.rank K)) := (f.sy …
      h_col : Eq (abs ((Matrix.of fun i w => HMul.hMul (↑(↑w).mult) (Real.log (↑w (( …
      i✝ : Fin (NumberField.Units.rank K)
      ⊢ ∀ (x : Fin (HAdd.hAdd (NumberField.Units.rank K) 1)), Membership.mem Finset. …
    -/
    exact fun _ _ ↦ pow_ne_zero _ <| (map_ne_zero _).mpr (coe_ne_zero _)
    /-
      🎉 no goals
    -/


/-- For any infinite place `w'`, the regulator is equal to the absolute value of the determinant
of the matrix `(mult w * log w (fundSystem K i)))_i, {w ≠ w'}`. -/
theorem regulator_eq_det (w' : InfinitePlace K) (e : {w // w ≠ w'} ≃ Fin (rank K)) :
    regulator K =
      |(Matrix.of fun i w : {w // w ≠ w'} ↦ (mult w.val : ℝ) *
        Real.log (w.val (fundSystem K (e i) : K))).det| := by
  let e' : {w : InfinitePlace K // w ≠ w₀} ≃ Fin (rank K) := Fintype.equivOfCardEq (by
    rw [Fintype.card_subtype_compl, Fintype.card_ofSubsingleton, Fintype.card_fin, rank])
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    w' : NumberField.InfinitePlace K
    e : Equiv (Subtype fun w => Ne w w') (Fin (NumberField.Units.rank K))
    e' : Equiv (Subtype fun w => Ne w NumberField.Units.dirichletUnitTheorem.w₀) ( …
    ⊢ Eq (NumberField.Units.regulator K) (abs (Matrix.of fun i w => HMul.hMul (↑(↑ …
  -/
  simp_rw [regulator_eq_det' K e', logEmbedding, AddMonoidHom.coe_mk, ZeroHom.coe_mk]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    w' : NumberField.InfinitePlace K
    e : Equiv (Subtype fun w => Ne w w') (Fin (NumberField.Units.rank K))
    e' : Equiv (Subtype fun w => Ne w NumberField.Units.dirichletUnitTheorem.w₀) ( …
    ⊢ Eq (abs (Matrix.of fun i w => HMul.hMul (↑(↑w).mult) (Real.log (↑w ((algebra …
  -/
  exact abs_det_eq_abs_det K (fun i ↦ fundSystem K i) e' e
  /-
    🎉 no goals
  -/



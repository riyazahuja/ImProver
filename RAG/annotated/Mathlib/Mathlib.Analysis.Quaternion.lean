@[inherit_doc] scoped[Quaternion] notation "ℍ" => Quaternion ℝ


instance : Inner ℝ ℍ :=
  ⟨fun a b => (a * star b).re⟩


theorem inner_self (a : ℍ) : ⟪a, a⟫ = normSq a :=
  rfl


theorem inner_def (a b : ℍ) : ⟪a, b⟫ = (a * star b).re :=
  rfl


noncomputable instance : NormedAddCommGroup ℍ :=
  @InnerProductSpace.Core.toNormedAddCommGroup ℝ ℍ _ _ _
    { toInner := inferInstance
                                 /-
                                   x y : Quaternion Real
                                   ⊢ Eq ((starRingEnd Real) (Inner.inner y x)) (Inner.inner x y)
                                 -/
      conj_symm := fun x y => by simp [inner_def, mul_comm]
                                 /-
                                   🎉 no goals
                                 -/
      nonneg_re := fun _ => normSq_nonneg
      definite := fun _ => normSq_eq_zero.1
                                  /-
                                    x y z : Quaternion Real
                                    ⊢ Eq (Inner.inner (HAdd.hAdd x y) z) (HAdd.hAdd (Inner.inner x z) (Inner.inner …
                                  -/
      add_left := fun x y z => by simp only [inner_def, add_mul, add_re]
                                  /-
                                    🎉 no goals
                                  -/
                                   /-
                                     x y : Quaternion Real
                                     r : Real
                                     ⊢ Eq (Inner.inner (HSMul.hSMul r x) y) (HMul.hMul ((starRingEnd Real) r) (Inne …
                                   -/
      smul_left := fun x y r => by simp [inner_def] }
                                   /-
                                     🎉 no goals
                                   -/


noncomputable instance : InnerProductSpace ℝ ℍ :=
  InnerProductSpace.ofCore _


theorem normSq_eq_norm_mul_self (a : ℍ) : normSq a = ‖a‖ * ‖a‖ := by
  /-
    a : Quaternion Real
    ⊢ Eq (Quaternion.normSq a) (HMul.hMul (Norm.norm a) (Norm.norm a))
  -/
  rw [← inner_self, real_inner_self_eq_norm_mul_norm]
  /-
    🎉 no goals
  -/


instance : NormOneClass ℍ :=
      /-
        ⊢ Eq (Norm.norm 1) 1
      -/
  ⟨by rw [norm_eq_sqrt_real_inner, inner_self, normSq.map_one, Real.sqrt_one]⟩
      /-
        🎉 no goals
      -/


@[simp, norm_cast]
theorem norm_coe (a : ℝ) : ‖(a : ℍ)‖ = ‖a‖ := by
  /-
    a : Real
    ⊢ Eq (Norm.norm ↑a) (Norm.norm a)
  -/
  rw [norm_eq_sqrt_real_inner, inner_self, normSq_coe, Real.sqrt_sq_eq_abs, Real.norm_eq_abs]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem nnnorm_coe (a : ℝ) : ‖(a : ℍ)‖₊ = ‖a‖₊ :=
  Subtype.ext <| norm_coe a

-- Porting note https://github.com/leanprover-community/mathlib4/issues/10959
-- simp cannot prove this

@[simp, nolint simpNF]
theorem norm_star (a : ℍ) : ‖star a‖ = ‖a‖ := by
  /-
    a : Quaternion Real
    ⊢ Eq (Norm.norm (Star.star a)) (Norm.norm a)
  -/
  simp_rw [norm_eq_sqrt_real_inner, inner_self, normSq_star]
  /-
    🎉 no goals
  -/

-- Porting note https://github.com/leanprover-community/mathlib4/issues/10959
-- simp cannot prove this

@[simp, nolint simpNF]
theorem nnnorm_star (a : ℍ) : ‖star a‖₊ = ‖a‖₊ :=
  Subtype.ext <| norm_star a


noncomputable instance : NormedDivisionRing ℍ where
  dist_eq _ _ := rfl
  norm_mul' a b := by
    /-
      a b : Quaternion Real
      ⊢ Eq (Norm.norm (HMul.hMul a b)) (HMul.hMul (Norm.norm a) (Norm.norm b))
    -/
    simp only [norm_eq_sqrt_real_inner, inner_self, normSq.map_mul]
    /-
      a b : Quaternion Real
      ⊢ Eq (HMul.hMul (Quaternion.normSq a) (Quaternion.normSq b)).sqrt (HMul.hMul ( …
    -/
    exact Real.sqrt_mul normSq_nonneg _
    /-
      🎉 no goals
    -/


noncomputable instance : NormedAlgebra ℝ ℍ where
  norm_smul_le := norm_smul_le
  toAlgebra := Quaternion.algebra


instance : CStarRing ℍ where
  norm_mul_self_le x :=
    le_of_eq <| Eq.symm <| (norm_mul _ _).trans <| congr_arg (· * ‖x‖) (norm_star x)


/-- Coercion from `ℂ` to `ℍ`. -/
@[coe] def coeComplex (z : ℂ) : ℍ := ⟨z.re, z.im, 0, 0⟩


instance : Coe ℂ ℍ := ⟨coeComplex⟩


@[simp, norm_cast]
theorem coeComplex_re (z : ℂ) : (z : ℍ).re = z.re :=
  rfl


@[simp, norm_cast]
theorem coeComplex_imI (z : ℂ) : (z : ℍ).imI = z.im :=
  rfl


@[simp, norm_cast]
theorem coeComplex_imJ (z : ℂ) : (z : ℍ).imJ = 0 :=
  rfl


@[simp, norm_cast]
theorem coeComplex_imK (z : ℂ) : (z : ℍ).imK = 0 :=
  rfl


@[simp, norm_cast]
                                                                /-
                                                                  z w : Complex
                                                                  ⊢ Eq (↑(HAdd.hAdd z w)) (HAdd.hAdd ↑z ↑w)
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
theorem coeComplex_add (z w : ℂ) : ↑(z + w) = (z + w : ℍ) := by ext <;> simp
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp, norm_cast]
                                                                /-
                                                                  z w : Complex
                                                                  ⊢ Eq (↑(HMul.hMul z w)) (HMul.hMul ↑z ↑w)
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
theorem coeComplex_mul (z w : ℂ) : ↑(z * w) = (z * w : ℍ) := by ext <;> simp
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp, norm_cast]
theorem coeComplex_zero : ((0 : ℂ) : ℍ) = 0 :=
  rfl


@[simp, norm_cast]
theorem coeComplex_one : ((1 : ℂ) : ℍ) = 1 :=
  rfl


@[simp, norm_cast]
                                                                           /-
                                                                             r : Real
                                                                             z : Complex
                                                                             ⊢ Eq (HSMul.hSMul r ↑z) (HMul.hMul ↑r ↑z)
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
theorem coe_real_complex_mul (r : ℝ) (z : ℂ) : (r • z : ℍ) = ↑r * ↑z := by ext <;> simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


@[simp, norm_cast]
theorem coeComplex_coe (r : ℝ) : ((r : ℂ) : ℍ) = r :=
  rfl


/-- Coercion `ℂ →ₐ[ℝ] ℍ` as an algebra homomorphism. -/
def ofComplex : ℂ →ₐ[ℝ] ℍ where
  toFun := (↑)
  map_one' := rfl
  map_zero' := rfl
  map_add' := coeComplex_add
  map_mul' := coeComplex_mul
  commutes' _ := rfl


@[simp]
theorem coe_ofComplex : ⇑ofComplex = coeComplex := rfl


/-- The norm of the components as a euclidean vector equals the norm of the quaternion. -/
theorem norm_piLp_equiv_symm_equivTuple (x : ℍ) :
    ‖(WithLp.equiv 2 (Fin 4 → _)).symm (equivTuple ℝ x)‖ = ‖x‖ := by
  rw [norm_eq_sqrt_real_inner, norm_eq_sqrt_real_inner, inner_self, normSq_def', PiLp.inner_apply,
    Fin.sum_univ_four]
  /-
    x : Quaternion Real
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (Inner.inner ((WithLp.equiv 2 (Fin 4 → R …
  -/
  simp_rw [RCLike.inner_apply, starRingEnd_apply, star_trivial, ← sq]
  /-
    x : Quaternion Real
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HPow.hPow ((WithLp.equiv 2 (Fin 4 → Rea …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `QuaternionAlgebra.linearEquivTuple` as a `LinearIsometryEquiv`. -/
@[simps apply symm_apply]
noncomputable def linearIsometryEquivTuple : ℍ ≃ₗᵢ[ℝ] EuclideanSpace ℝ (Fin 4) :=
  { (QuaternionAlgebra.linearEquivTuple (-1 : ℝ) (-1 : ℝ)).trans
      (WithLp.linearEquiv 2 ℝ (Fin 4 → ℝ)).symm with
    toFun := fun a => !₂[a.1, a.2, a.3, a.4]
    invFun := fun a => ⟨a 0, a 1, a 2, a 3⟩
    norm_map' := norm_piLp_equiv_symm_equivTuple }


@[continuity]
theorem continuous_coe : Continuous (coe : ℝ → ℍ) :=
  continuous_algebraMap ℝ ℍ


@[continuity]
theorem continuous_normSq : Continuous (normSq : ℍ → ℝ) := by
  simpa [← normSq_eq_norm_mul_self] using
    (continuous_norm.mul continuous_norm : Continuous fun q : ℍ => ‖q‖ * ‖q‖)


@[continuity]
theorem continuous_re : Continuous fun q : ℍ => q.re :=
  (continuous_apply 0).comp linearIsometryEquivTuple.continuous


@[continuity]
theorem continuous_imI : Continuous fun q : ℍ => q.imI :=
  (continuous_apply 1).comp linearIsometryEquivTuple.continuous


@[continuity]
theorem continuous_imJ : Continuous fun q : ℍ => q.imJ :=
  (continuous_apply 2).comp linearIsometryEquivTuple.continuous


@[continuity]
theorem continuous_imK : Continuous fun q : ℍ => q.imK :=
  (continuous_apply 3).comp linearIsometryEquivTuple.continuous


@[continuity]
theorem continuous_im : Continuous fun q : ℍ => q.im := by
  /-
    ⊢ Continuous fun q => q.im
  -/
  simpa only [← sub_self_re] using continuous_id.sub (continuous_coe.comp continuous_re)
  /-
    🎉 no goals
  -/


instance : CompleteSpace ℍ :=
  haveI : IsUniformEmbedding linearIsometryEquivTuple.toLinearEquiv.toEquiv.symm :=
    linearIsometryEquivTuple.toContinuousLinearEquiv.symm.isUniformEmbedding
  (completeSpace_congr this).1 inferInstance


@[simp, norm_cast]
theorem hasSum_coe {f : α → ℝ} {r : ℝ} : HasSum (fun a => (f a : ℍ)) (↑r : ℍ) ↔ HasSum f r :=
               /-
                 α : Type u_1
                 f : α → Real
                 r : Real
                 h : HasSum (fun a => ↑(f a)) ↑r
                 ⊢ HasSum f r
               -/
  ⟨fun h => by simpa only using h.map (show ℍ →ₗ[ℝ] ℝ from QuaternionAlgebra.reₗ _ _) continuous_re,
               /-
                 🎉 no goals
               -/
                /-
                  α : Type u_1
                  f : α → Real
                  r : Real
                  h : HasSum f r
                  ⊢ HasSum (fun a => ↑(f a)) ↑r
                -/
    fun h => by simpa only using h.map (algebraMap ℝ ℍ) (continuous_algebraMap _ _)⟩
                /-
                  🎉 no goals
                -/


@[simp, norm_cast]
theorem summable_coe {f : α → ℝ} : (Summable fun a => (f a : ℍ)) ↔ Summable f := by
  simpa only using
    Summable.map_iff_of_leftInverse (algebraMap ℝ ℍ) (show ℍ →ₗ[ℝ] ℝ from QuaternionAlgebra.reₗ _ _)
      (continuous_algebraMap _ _) continuous_re coe_re


@[norm_cast]
theorem tsum_coe (f : α → ℝ) : (∑' a, (f a : ℍ)) = ↑(∑' a, f a) := by
  /-
    α : Type u_1
    f : α → Real
    ⊢ Eq (tsum fun a => ↑(f a)) ↑(tsum fun a => f a)
  -/
  by_cases hf : Summable f
    /-
      case pos
      α : Type u_1
      f : α → Real
      hf : Summable f
      ⊢ Eq (tsum fun a => ↑(f a)) ↑(tsum fun a => f a)
    -/
  · exact (hasSum_coe.mpr hf.hasSum).tsum_eq
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      f : α → Real
      hf : Not (Summable f)
      ⊢ Eq (tsum fun a => ↑(f a)) ↑(tsum fun a => f a)
    -/
  · simp [tsum_eq_zero_of_not_summable hf, tsum_eq_zero_of_not_summable (summable_coe.not.mpr hf)]
    /-
      🎉 no goals
    -/



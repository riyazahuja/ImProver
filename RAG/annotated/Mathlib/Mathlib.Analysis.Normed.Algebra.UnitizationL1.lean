/-- The natural map between `Unitization 𝕜 A` and `𝕜 × A`, transferred to their `WithLp 1`
synonyms. -/
noncomputable def unitization_addEquiv_prod : WithLp 1 (Unitization 𝕜 A) ≃+ WithLp 1 (𝕜 × A) :=
  (WithLp.linearEquiv 1 𝕜 (Unitization 𝕜 A)).toAddEquiv.trans <|
    (addEquiv 𝕜 A).trans (WithLp.linearEquiv 1 𝕜 (𝕜 × A)).symm.toAddEquiv


noncomputable instance instUnitizationNormedAddCommGroup :
    NormedAddCommGroup (WithLp 1 (Unitization 𝕜 A)) :=
  NormedAddCommGroup.induced (WithLp 1 (Unitization 𝕜 A)) (WithLp 1 (𝕜 × A))
    (unitization_addEquiv_prod 𝕜 A) (AddEquiv.injective _)


/-- Bundle `WithLp.unitization_addEquiv_prod` as a `UniformEquiv`. -/
noncomputable def uniformEquiv_unitization_addEquiv_prod :
    WithLp 1 (Unitization 𝕜 A) ≃ᵤ WithLp 1 (𝕜 × A) :=
  { unitization_addEquiv_prod 𝕜 A with
    uniformContinuous_invFun := uniformContinuous_comap' uniformContinuous_id
    uniformContinuous_toFun := uniformContinuous_iff.mpr le_rfl }


instance instCompleteSpace [CompleteSpace 𝕜] [CompleteSpace A] :
    CompleteSpace (WithLp 1 (Unitization 𝕜 A)) :=
  completeSpace_congr (uniformEquiv_unitization_addEquiv_prod 𝕜 A).isUniformEmbedding |>.mpr
    CompleteSpace.prod


open ENNReal in
lemma unitization_norm_def (x : WithLp 1 (Unitization 𝕜 A)) :
    ‖x‖ = ‖(WithLp.equiv 1 _ x).fst‖ + ‖(WithLp.equiv 1 _ x).snd‖ := calc
  ‖x‖ = (‖(WithLp.equiv 1 _ x).fst‖ ^ (1 : ℝ≥0∞).toReal +
      ‖(WithLp.equiv 1 _ x).snd‖ ^ (1 : ℝ≥0∞).toReal) ^ (1 / (1 : ℝ≥0∞).toReal) :=
                                /-
                                  𝕜 : Type u_1
                                  A : Type u_2
                                  inst✝² : NormedField 𝕜
                                  inst✝¹ : NonUnitalNormedRing A
                                  inst✝ : NormedSpace 𝕜 A
                                  x : WithLp 1 (Unitization 𝕜 A)
                                  ⊢ LT.lt 0 (ENNReal.toReal 1)
                                -/
    WithLp.prod_norm_eq_add (by simp : 0 < (1 : ℝ≥0∞).toReal) _
                                /-
                                  🎉 no goals
                                -/
                                                                      /-
                                                                        𝕜 : Type u_1
                                                                        A : Type u_2
                                                                        inst✝² : NormedField 𝕜
                                                                        inst✝¹ : NonUnitalNormedRing A
                                                                        inst✝ : NormedSpace 𝕜 A
                                                                        x : WithLp 1 (Unitization 𝕜 A)
                                                                        ⊢ Eq (HPow.hPow (HAdd.hAdd (HPow.hPow (Norm.norm ((WithLp.equiv 1 (Unitization …
                                                                      -/
  _   = ‖(WithLp.equiv 1 _ x).fst‖ + ‖(WithLp.equiv 1 _ x).snd‖ := by simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


lemma unitization_nnnorm_def (x : WithLp 1 (Unitization 𝕜 A)) :
    ‖x‖₊ = ‖(WithLp.equiv 1 _ x).fst‖₊ + ‖(WithLp.equiv 1 _ x).snd‖₊ :=
  Subtype.ext <| unitization_norm_def x


lemma unitization_norm_inr (x : A) : ‖(WithLp.equiv 1 (Unitization 𝕜 A)).symm x‖ = ‖x‖ := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : NonUnitalNormedRing A
    inst✝ : NormedSpace 𝕜 A
    x : A
    ⊢ Eq (Norm.norm ((WithLp.equiv 1 (Unitization 𝕜 A)).symm ↑x)) (Norm.norm x)
  -/
  simp [unitization_norm_def]
  /-
    🎉 no goals
  -/


lemma unitization_nnnorm_inr (x : A) : ‖(WithLp.equiv 1 (Unitization 𝕜 A)).symm x‖₊ = ‖x‖₊ := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : NonUnitalNormedRing A
    inst✝ : NormedSpace 𝕜 A
    x : A
    ⊢ Eq (NNNorm.nnnorm ((WithLp.equiv 1 (Unitization 𝕜 A)).symm ↑x)) (NNNorm.nnno …
  -/
  simp [unitization_nnnorm_def]
  /-
    🎉 no goals
  -/


lemma unitization_isometry_inr :
    Isometry (fun x : A ↦ (WithLp.equiv 1 (Unitization 𝕜 A)).symm x) :=
  AddMonoidHomClass.isometry_of_norm
    ((WithLp.linearEquiv 1 𝕜 (Unitization 𝕜 A)).symm.comp <| Unitization.inrHom 𝕜 A)
    unitization_norm_inr


instance instUnitizationRing : Ring (WithLp 1 (Unitization 𝕜 A)) :=
  inferInstanceAs (Ring (Unitization 𝕜 A))


@[simp]
lemma unitization_mul (x y : WithLp 1 (Unitization 𝕜 A)) :
    WithLp.equiv 1 _ (x * y) = (WithLp.equiv 1 _ x) * (WithLp.equiv 1 _ y) :=
  rfl


instance {R : Type*} [CommSemiring R] [Algebra R 𝕜] [DistribMulAction R A] [IsScalarTower R 𝕜 A] :
    Algebra R (WithLp 1 (Unitization 𝕜 A)) :=
  inferInstanceAs (Algebra R (Unitization 𝕜 A))


@[simp]
lemma unitization_algebraMap (r : 𝕜) :
    WithLp.equiv 1 _ (algebraMap 𝕜 (WithLp 1 (Unitization 𝕜 A)) r) =
      algebraMap 𝕜 (Unitization 𝕜 A) r :=
  rfl


/-- `WithLp.equiv` bundled as an algebra isomorphism with `Unitization 𝕜 A`. -/
@[simps!]
def unitizationAlgEquiv (R : Type*) [CommSemiring R] [Algebra R 𝕜] [DistribMulAction R A]
    [IsScalarTower R 𝕜 A] : WithLp 1 (Unitization 𝕜 A) ≃ₐ[R] Unitization 𝕜 A :=
  { WithLp.equiv 1 (Unitization 𝕜 A) with
    map_mul' := fun _ _ ↦ rfl
    map_add' := fun _ _ ↦ rfl
    commutes' := fun _ ↦ rfl }


noncomputable instance instUnitizationNormedRing : NormedRing (WithLp 1 (Unitization 𝕜 A)) where
  dist_eq := dist_eq_norm
  norm_mul x y := by
    /-
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁴ : NormedField 𝕜
      inst✝³ : NonUnitalNormedRing A
      inst✝² : NormedSpace 𝕜 A
      inst✝¹ : IsScalarTower 𝕜 A A
      inst✝ : SMulCommClass 𝕜 A A
      x y : WithLp 1 (Unitization 𝕜 A)
      ⊢ LE.le (Norm.norm (HMul.hMul x y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
    -/
    simp_rw [unitization_norm_def, add_mul, mul_add, unitization_mul, fst_mul, snd_mul]
    /-
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁴ : NormedField 𝕜
      inst✝³ : NonUnitalNormedRing A
      inst✝² : NormedSpace 𝕜 A
      inst✝¹ : IsScalarTower 𝕜 A A
      inst✝ : SMulCommClass 𝕜 A A
      x y : WithLp 1 (Unitization 𝕜 A)
      ⊢ LE.le (HAdd.hAdd (Norm.norm (HMul.hMul ((WithLp.equiv 1 (Unitization 𝕜 A)) x …
    -/
    rw [add_assoc, add_assoc]
    /-
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁴ : NormedField 𝕜
      inst✝³ : NonUnitalNormedRing A
      inst✝² : NormedSpace 𝕜 A
      inst✝¹ : IsScalarTower 𝕜 A A
      inst✝ : SMulCommClass 𝕜 A A
      x y : WithLp 1 (Unitization 𝕜 A)
      ⊢ LE.le (HAdd.hAdd (Norm.norm (HMul.hMul ((WithLp.equiv 1 (Unitization 𝕜 A)) x …
    -/
    gcongr
      /-
        case h₁
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁴ : NormedField 𝕜
        inst✝³ : NonUnitalNormedRing A
        inst✝² : NormedSpace 𝕜 A
        inst✝¹ : IsScalarTower 𝕜 A A
        inst✝ : SMulCommClass 𝕜 A A
        x y : WithLp 1 (Unitization 𝕜 A)
        ⊢ LE.le (Norm.norm (HMul.hMul ((WithLp.equiv 1 (Unitization 𝕜 A)) x).fst ((Wit …
      -/
    · exact norm_mul_le _ _
      /-
        🎉 no goals
      -/
      /-
        case h₂
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁴ : NormedField 𝕜
        inst✝³ : NonUnitalNormedRing A
        inst✝² : NormedSpace 𝕜 A
        inst✝¹ : IsScalarTower 𝕜 A A
        inst✝ : SMulCommClass 𝕜 A A
        x y : WithLp 1 (Unitization 𝕜 A)
        ⊢ LE.le (Norm.norm (HAdd.hAdd (HSMul.hSMul ((WithLp.equiv 1 (Unitization 𝕜 A)) …
      -/
    · apply (norm_add_le _ _).trans
      /-
        case h₂
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁴ : NormedField 𝕜
        inst✝³ : NonUnitalNormedRing A
        inst✝² : NormedSpace 𝕜 A
        inst✝¹ : IsScalarTower 𝕜 A A
        inst✝ : SMulCommClass 𝕜 A A
        x y : WithLp 1 (Unitization 𝕜 A)
        ⊢ LE.le (HAdd.hAdd (Norm.norm (HSMul.hSMul ((WithLp.equiv 1 (Unitization 𝕜 A)) …
      -/
      gcongr
        /-
          case h₂.h₁
          𝕜 : Type u_1
          A : Type u_2
          inst✝⁴ : NormedField 𝕜
          inst✝³ : NonUnitalNormedRing A
          inst✝² : NormedSpace 𝕜 A
          inst✝¹ : IsScalarTower 𝕜 A A
          inst✝ : SMulCommClass 𝕜 A A
          x y : WithLp 1 (Unitization 𝕜 A)
          ⊢ LE.le (Norm.norm (HSMul.hSMul ((WithLp.equiv 1 (Unitization 𝕜 A)) x).fst ((W …
        -/
      · simp [norm_smul]
        /-
          🎉 no goals
        -/
        /-
          case h₂.h₂
          𝕜 : Type u_1
          A : Type u_2
          inst✝⁴ : NormedField 𝕜
          inst✝³ : NonUnitalNormedRing A
          inst✝² : NormedSpace 𝕜 A
          inst✝¹ : IsScalarTower 𝕜 A A
          inst✝ : SMulCommClass 𝕜 A A
          x y : WithLp 1 (Unitization 𝕜 A)
          ⊢ LE.le (Norm.norm (HAdd.hAdd (HSMul.hSMul ((WithLp.equiv 1 (Unitization 𝕜 A)) …
        -/
      · apply (norm_add_le _ _).trans
        /-
          case h₂.h₂
          𝕜 : Type u_1
          A : Type u_2
          inst✝⁴ : NormedField 𝕜
          inst✝³ : NonUnitalNormedRing A
          inst✝² : NormedSpace 𝕜 A
          inst✝¹ : IsScalarTower 𝕜 A A
          inst✝ : SMulCommClass 𝕜 A A
          x y : WithLp 1 (Unitization 𝕜 A)
          ⊢ LE.le (HAdd.hAdd (Norm.norm (HSMul.hSMul ((WithLp.equiv 1 (Unitization 𝕜 A)) …
        -/
        gcongr
          /-
            case h₂.h₂.h₁
            𝕜 : Type u_1
            A : Type u_2
            inst✝⁴ : NormedField 𝕜
            inst✝³ : NonUnitalNormedRing A
            inst✝² : NormedSpace 𝕜 A
            inst✝¹ : IsScalarTower 𝕜 A A
            inst✝ : SMulCommClass 𝕜 A A
            x y : WithLp 1 (Unitization 𝕜 A)
            ⊢ LE.le (Norm.norm (HSMul.hSMul ((WithLp.equiv 1 (Unitization 𝕜 A)) y).fst ((W …
          -/
        · simp [norm_smul, mul_comm]
          /-
            🎉 no goals
          -/
          /-
            case h₂.h₂.h₂
            𝕜 : Type u_1
            A : Type u_2
            inst✝⁴ : NormedField 𝕜
            inst✝³ : NonUnitalNormedRing A
            inst✝² : NormedSpace 𝕜 A
            inst✝¹ : IsScalarTower 𝕜 A A
            inst✝ : SMulCommClass 𝕜 A A
            x y : WithLp 1 (Unitization 𝕜 A)
            ⊢ LE.le (Norm.norm (HMul.hMul ((WithLp.equiv 1 (Unitization 𝕜 A)) x).snd ((Wit …
          -/
        · exact norm_mul_le _ _
          /-
            🎉 no goals
          -/


noncomputable instance instUnitizationNormedAlgebra :
    NormedAlgebra 𝕜 (WithLp 1 (Unitization 𝕜 A)) where
  norm_smul_le r x := by
    /-
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁴ : NormedField 𝕜
      inst✝³ : NonUnitalNormedRing A
      inst✝² : NormedSpace 𝕜 A
      inst✝¹ : IsScalarTower 𝕜 A A
      inst✝ : SMulCommClass 𝕜 A A
      r : 𝕜
      x : WithLp 1 (Unitization 𝕜 A)
      ⊢ LE.le (Norm.norm (HSMul.hSMul r x)) (HMul.hMul (Norm.norm r) (Norm.norm x))
    -/
    simp_rw [unitization_norm_def, equiv_smul, fst_smul, snd_smul, norm_smul, mul_add]
    /-
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁴ : NormedField 𝕜
      inst✝³ : NonUnitalNormedRing A
      inst✝² : NormedSpace 𝕜 A
      inst✝¹ : IsScalarTower 𝕜 A A
      inst✝ : SMulCommClass 𝕜 A A
      r : 𝕜
      x : WithLp 1 (Unitization 𝕜 A)
      ⊢ LE.le (HAdd.hAdd (HMul.hMul (Norm.norm r) (Norm.norm ((WithLp.equiv 1 (Uniti …
    -/
    exact le_rfl
    /-
      🎉 no goals
    -/



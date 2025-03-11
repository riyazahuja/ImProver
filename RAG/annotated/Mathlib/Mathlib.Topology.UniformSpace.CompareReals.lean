/-- The metric space uniform structure on ℚ (which presupposes the existence
of real numbers) agrees with the one coming directly from (abs : ℚ → ℚ). -/
theorem Rat.uniformSpace_eq :
    (AbsoluteValue.abs : AbsoluteValue ℚ ℚ).uniformSpace = PseudoMetricSpace.toUniformSpace := by
  /-
    ⊢ Eq AbsoluteValue.abs.uniformSpace PseudoMetricSpace.toUniformSpace
  -/
  ext s
  /-
    case h.h
    s : Set (Prod Rat Rat)
    ⊢ Iff (Membership.mem (uniformity Rat) s) (Membership.mem (uniformity Rat) s)
  -/
  rw [(AbsoluteValue.hasBasis_uniformity _).mem_iff, Metric.uniformity_basis_dist_rat.mem_iff]
  simp only [Rat.dist_eq, AbsoluteValue.abs_apply, ← Rat.cast_sub, ← Rat.cast_abs, Rat.cast_lt,
    abs_sub_comm]


/-- Cauchy reals packaged as a completion of ℚ using the absolute value route. -/
def rationalCauSeqPkg : @AbstractCompletion ℚ <| (@AbsoluteValue.abs ℚ _).uniformSpace :=
  @AbstractCompletion.mk
    (space := ℝ)
    (coe := ((↑) : ℚ → ℝ))
                         /-
                           ⊢ UniformSpace Real
                         -/
    (uniformStruct := by infer_instance)
                         /-
                           🎉 no goals
                         -/
                    /-
                      ⊢ CompleteSpace Real
                    -/
    (complete := by infer_instance)
                    /-
                      🎉 no goals
                    -/
                      /-
                        ⊢ T0Space Real
                      -/
    (separation := by infer_instance)
                      /-
                        🎉 no goals
                      -/
    (isUniformInducing := by
      /-
        ⊢ IsUniformInducing Rat.cast
      -/
      rw [Rat.uniformSpace_eq]
      /-
        ⊢ IsUniformInducing Rat.cast
      -/
      exact Rat.isUniformEmbedding_coe_real.isUniformInducing)
      /-
        🎉 no goals
      -/
    (dense := Rat.isDenseEmbedding_coe_real.dense)


/-- Type wrapper around ℚ to make sure the absolute value uniform space instance is picked up
instead of the metric space one. We proved in `Rat.uniformSpace_eq` that they are equal,
but they are not definitionaly equal, so it would confuse the type class system (and probably
also human readers). -/
def Q :=
  ℚ deriving CommRing, Inhabited


instance uniformSpace : UniformSpace Q :=
  (@AbsoluteValue.abs ℚ _).uniformSpace


/-- Real numbers constructed as in Bourbaki. -/
def Bourbakiℝ : Type :=
  Completion Q deriving Inhabited


instance Bourbaki.uniformSpace : UniformSpace Bourbakiℝ :=
  Completion.uniformSpace Q


/-- Bourbaki reals packaged as a completion of Q using the general theory. -/
def bourbakiPkg : AbstractCompletion Q :=
  Completion.cPkg


/-- The uniform bijection between Bourbaki and Cauchy reals. -/
noncomputable def compareEquiv : Bourbakiℝ ≃ᵤ ℝ :=
  bourbakiPkg.compareEquiv rationalCauSeqPkg


theorem compare_uc : UniformContinuous compareEquiv :=
  bourbakiPkg.uniformContinuous_compareEquiv rationalCauSeqPkg


theorem compare_uc_symm : UniformContinuous compareEquiv.symm :=
  bourbakiPkg.uniformContinuous_compareEquiv_symm rationalCauSeqPkg



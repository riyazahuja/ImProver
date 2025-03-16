/-- The reals are a conditionally complete linearly ordered field. -/
noncomputable instance : ConditionallyCompleteLinearOrderedField ℝ :=
  { (inferInstance : LinearOrderedField ℝ),
    (inferInstance : ConditionallyCompleteLinearOrder ℝ) with }


/-- There exists no nontrivial ring homomorphism `ℝ →+* ℝ`. -/
instance Real.RingHom.unique : Unique (ℝ →+* ℝ) where
  default := RingHom.id ℝ
  uniq f := congr_arg OrderRingHom.toRingHom (@Subsingleton.elim (ℝ →+*o ℝ) _
      ⟨f, ringHom_monotone (fun r hr => ⟨√r, sq_sqrt hr⟩) f⟩ default)


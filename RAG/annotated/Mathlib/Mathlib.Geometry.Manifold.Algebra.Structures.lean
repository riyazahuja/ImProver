/-- A smooth (semi)ring is a (semi)ring `R` where addition and multiplication are smooth.
If `R` is a ring, then negation is automatically smooth, as it is multiplication with `-1`. -/
class SmoothRing (I : ModelWithCorners 𝕜 E H) (R : Type*) [Semiring R] [TopologicalSpace R]
    [ChartedSpace H R] extends SmoothAdd I R : Prop where
  smooth_mul : ContMDiff (I.prod I) I ⊤ fun p : R × R => p.1 * p.2

-- see Note [lower instance priority]

instance (priority := 100) SmoothRing.toSmoothMul (I : ModelWithCorners 𝕜 E H) (R : Type*)
    [Semiring R] [TopologicalSpace R] [ChartedSpace H R] [h : SmoothRing I R] :
    SmoothMul I R :=
  { h with }

-- see Note [lower instance priority]

instance (priority := 100) SmoothRing.toLieAddGroup (I : ModelWithCorners 𝕜 E H) (R : Type*)
    [Ring R] [TopologicalSpace R] [ChartedSpace H R] [SmoothRing I R] : LieAddGroup I R where
  compatible := StructureGroupoid.compatible (contDiffGroupoid ∞ I)
  smooth_add := contMDiff_add I
                   /-
                     𝕜 : Type u_1
                     inst✝⁷ : NontriviallyNormedField 𝕜
                     H : Type u_2
                     inst✝⁶ : TopologicalSpace H
                     E : Type u_3
                     inst✝⁵ : NormedAddCommGroup E
                     inst✝⁴ : NormedSpace 𝕜 E
                     I : ModelWithCorners 𝕜 E H
                     R : Type u_4
                     inst✝³ : Ring R
                     inst✝² : TopologicalSpace R
                     inst✝¹ : ChartedSpace H R
                     inst✝ : SmoothRing I R
                     ⊢ ContMDiff I I Top.top fun a => Neg.neg a
                   -/
  smooth_neg := by simpa only [neg_one_mul] using contMDiff_mul_left (G := R) (a := -1)
                   /-
                     🎉 no goals
                   -/


instance (priority := 100) fieldSmoothRing {𝕜 : Type*} [NontriviallyNormedField 𝕜] :
    SmoothRing 𝓘(𝕜) 𝕜 :=
  { normedSpaceLieAddGroup with
    smooth_mul := by
      /-
        𝕜 : Type u_1
        inst✝ : NontriviallyNormedField 𝕜
        ⊢ ContMDiff ((modelWithCornersSelf 𝕜 𝕜).prod (modelWithCornersSelf 𝕜 𝕜)) (mode …
      -/
      rw [contMDiff_iff]
      /-
        𝕜 : Type u_1
        inst✝ : NontriviallyNormedField 𝕜
        ⊢ And (Continuous fun p => HMul.hMul p.1 p.2) (∀ (x : Prod 𝕜 𝕜) (y : 𝕜), ContD …
      -/
      refine ⟨continuous_mul, fun x y => ?_⟩
      /-
        𝕜 : Type u_1
        inst✝ : NontriviallyNormedField 𝕜
        x : Prod 𝕜 𝕜
        y : 𝕜
        ⊢ ContDiffOn 𝕜 (↑Top.top) (Function.comp (↑(extChartAt (modelWithCornersSelf 𝕜 …
      -/
      simp only [mfld_simps]
      /-
        𝕜 : Type u_1
        inst✝ : NontriviallyNormedField 𝕜
        x : Prod 𝕜 𝕜
        y : 𝕜
        ⊢ ContDiffOn 𝕜 (↑Top.top) (fun p => HMul.hMul p.1 p.2) Set.univ
      -/
      rw [contDiffOn_univ]
      /-
        𝕜 : Type u_1
        inst✝ : NontriviallyNormedField 𝕜
        x : Prod 𝕜 𝕜
        y : 𝕜
        ⊢ ContDiff 𝕜 ↑Top.top fun p => HMul.hMul p.1 p.2
      -/
      exact contDiff_mul }
      /-
        🎉 no goals
      -/


/-- A smooth (semi)ring is a topological (semi)ring. This is not an instance for technical reasons,
see note [Design choices about smooth algebraic structures]. -/
theorem topologicalSemiring_of_smooth [Semiring R] [SmoothRing I R] : TopologicalSemiring R :=
  { continuousMul_of_smooth I, continuousAdd_of_smooth I with }


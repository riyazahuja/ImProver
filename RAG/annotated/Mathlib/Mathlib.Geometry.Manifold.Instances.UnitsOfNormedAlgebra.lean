local notation "∞" => (⊤ : ℕ∞)


instance : ChartedSpace R Rˣ :=
  isOpenEmbedding_val.singletonChartedSpace


theorem chartAt_apply {a : Rˣ} {b : Rˣ} : chartAt R a b = b :=
  rfl


theorem chartAt_source {a : Rˣ} : (chartAt R a).source = Set.univ :=
  rfl


instance : SmoothManifoldWithCorners 𝓘(𝕜, R) Rˣ :=
  isOpenEmbedding_val.singleton_smoothManifoldWithCorners


/-- For a complete normed ring `R`, the embedding of the units `Rˣ` into `R` is a smooth map between
manifolds. -/
lemma contMDiff_val {m : ℕ∞} : ContMDiff 𝓘(𝕜, R) 𝓘(𝕜, R) m (val : Rˣ → R) :=
  contMDiff_isOpenEmbedding Units.isOpenEmbedding_val


/-- The units of a complete normed ring form a Lie group. -/
instance : LieGroup 𝓘(𝕜, R) Rˣ where
  smooth_mul := by
    /-
      R : Type u_1
      inst✝³ : NormedRing R
      inst✝² : CompleteSpace R
      𝕜 : Type u_2
      inst✝¹ : NontriviallyNormedField 𝕜
      inst✝ : NormedAlgebra 𝕜 R
      ⊢ ContMDiff ((modelWithCornersSelf 𝕜 R).prod (modelWithCornersSelf 𝕜 R)) (mode …
    -/
    apply ContMDiff.of_comp_isOpenEmbedding Units.isOpenEmbedding_val
    have : (val : Rˣ → R) ∘ (fun x : Rˣ × Rˣ => x.1 * x.2) =
      (fun x : R × R => x.1 * x.2) ∘ (fun x : Rˣ × Rˣ => (x.1, x.2)) := by ext; simp
    /-
      R : Type u_1
      inst✝³ : NormedRing R
      inst✝² : CompleteSpace R
      𝕜 : Type u_2
      inst✝¹ : NontriviallyNormedField 𝕜
      inst✝ : NormedAlgebra 𝕜 R
      this : Eq (Function.comp Units.val fun x => HMul.hMul x.1 x.2) (Function.comp  …
      ⊢ ContMDiff ((modelWithCornersSelf 𝕜 R).prod (modelWithCornersSelf 𝕜 R)) (mode …
    -/
    rw [this]
    have : ContMDiff (𝓘(𝕜, R).prod 𝓘(𝕜, R)) 𝓘(𝕜, R × R) ∞
      (fun x : Rˣ × Rˣ => ((x.1 : R), (x.2 : R))) :=
      (contMDiff_val.comp contMDiff_fst).prod_mk_space (contMDiff_val.comp contMDiff_snd)
    /-
      R : Type u_1
      inst✝³ : NormedRing R
      inst✝² : CompleteSpace R
      𝕜 : Type u_2
      inst✝¹ : NontriviallyNormedField 𝕜
      inst✝ : NormedAlgebra 𝕜 R
      this✝ : Eq (Function.comp Units.val fun x => HMul.hMul x.1 x.2) (Function.comp …
      this : ContMDiff ((modelWithCornersSelf 𝕜 R).prod (modelWithCornersSelf 𝕜 R))  …
      ⊢ ContMDiff ((modelWithCornersSelf 𝕜 R).prod (modelWithCornersSelf 𝕜 R)) (mode …
    -/
    refine ContMDiff.comp ?_ this
    /-
      R : Type u_1
      inst✝³ : NormedRing R
      inst✝² : CompleteSpace R
      𝕜 : Type u_2
      inst✝¹ : NontriviallyNormedField 𝕜
      inst✝ : NormedAlgebra 𝕜 R
      this✝ : Eq (Function.comp Units.val fun x => HMul.hMul x.1 x.2) (Function.comp …
      this : ContMDiff ((modelWithCornersSelf 𝕜 R).prod (modelWithCornersSelf 𝕜 R))  …
      ⊢ ContMDiff (modelWithCornersSelf 𝕜 (Prod R R)) (modelWithCornersSelf 𝕜 R) Top …
    -/
    rw [contMDiff_iff_contDiff]
    /-
      R : Type u_1
      inst✝³ : NormedRing R
      inst✝² : CompleteSpace R
      𝕜 : Type u_2
      inst✝¹ : NontriviallyNormedField 𝕜
      inst✝ : NormedAlgebra 𝕜 R
      this✝ : Eq (Function.comp Units.val fun x => HMul.hMul x.1 x.2) (Function.comp …
      this : ContMDiff ((modelWithCornersSelf 𝕜 R).prod (modelWithCornersSelf 𝕜 R))  …
      ⊢ ContDiff 𝕜 ↑Top.top fun x => HMul.hMul x.1 x.2
    -/
    exact contDiff_mul
    /-
      🎉 no goals
    -/
  smooth_inv := by
    /-
      R : Type u_1
      inst✝³ : NormedRing R
      inst✝² : CompleteSpace R
      𝕜 : Type u_2
      inst✝¹ : NontriviallyNormedField 𝕜
      inst✝ : NormedAlgebra 𝕜 R
      ⊢ ContMDiff (modelWithCornersSelf 𝕜 R) (modelWithCornersSelf 𝕜 R) Top.top fun  …
    -/
    apply ContMDiff.of_comp_isOpenEmbedding Units.isOpenEmbedding_val
    /-
      R : Type u_1
      inst✝³ : NormedRing R
      inst✝² : CompleteSpace R
      𝕜 : Type u_2
      inst✝¹ : NontriviallyNormedField 𝕜
      inst✝ : NormedAlgebra 𝕜 R
      ⊢ ContMDiff (modelWithCornersSelf 𝕜 R) (modelWithCornersSelf 𝕜 R) Top.top (Fun …
    -/
    have : (val : Rˣ → R) ∘ (fun x : Rˣ => x⁻¹) = Ring.inverse ∘ val := by ext; simp
    /-
      R : Type u_1
      inst✝³ : NormedRing R
      inst✝² : CompleteSpace R
      𝕜 : Type u_2
      inst✝¹ : NontriviallyNormedField 𝕜
      inst✝ : NormedAlgebra 𝕜 R
      this : Eq (Function.comp Units.val fun x => Inv.inv x) (Function.comp Ring.inv …
      ⊢ ContMDiff (modelWithCornersSelf 𝕜 R) (modelWithCornersSelf 𝕜 R) Top.top (Fun …
    -/
    rw [this, ContMDiff]
    /-
      R : Type u_1
      inst✝³ : NormedRing R
      inst✝² : CompleteSpace R
      𝕜 : Type u_2
      inst✝¹ : NontriviallyNormedField 𝕜
      inst✝ : NormedAlgebra 𝕜 R
      this : Eq (Function.comp Units.val fun x => Inv.inv x) (Function.comp Ring.inv …
      ⊢ ∀ (x : Units R), ContMDiffAt (modelWithCornersSelf 𝕜 R) (modelWithCornersSel …
    -/
    refine fun x => ContMDiffAt.comp x ?_ (contMDiff_val x)
    /-
      R : Type u_1
      inst✝³ : NormedRing R
      inst✝² : CompleteSpace R
      𝕜 : Type u_2
      inst✝¹ : NontriviallyNormedField 𝕜
      inst✝ : NormedAlgebra 𝕜 R
      this : Eq (Function.comp Units.val fun x => Inv.inv x) (Function.comp Ring.inv …
      x : Units R
      ⊢ ContMDiffAt (modelWithCornersSelf 𝕜 R) (modelWithCornersSelf 𝕜 R) Top.top Ri …
    -/
    rw [contMDiffAt_iff_contDiffAt]
    /-
      R : Type u_1
      inst✝³ : NormedRing R
      inst✝² : CompleteSpace R
      𝕜 : Type u_2
      inst✝¹ : NontriviallyNormedField 𝕜
      inst✝ : NormedAlgebra 𝕜 R
      this : Eq (Function.comp Units.val fun x => Inv.inv x) (Function.comp Ring.inv …
      x : Units R
      ⊢ ContDiffAt 𝕜 (↑Top.top) Ring.inverse ↑x
    -/
    exact contDiffAt_ring_inverse _ _
    /-
      🎉 no goals
    -/



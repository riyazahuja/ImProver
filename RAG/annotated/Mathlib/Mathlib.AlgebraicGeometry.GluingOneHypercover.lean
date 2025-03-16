/-- The 1-hypercover of `D.glued` in the big Zariski site that is given by the
open cover `D.U` from the glue data `D`.
The "covering of the intersection of two such open subsets" is the trivial
covering given by `D.V`. -/
@[simps]
noncomputable def oneHypercover : Scheme.zariskiTopology.OneHypercover D.glued where
  I₀ := D.J
  X := D.U
  f := D.ι
  I₁ _ _ := PUnit
  Y i₁ i₂ _ := D.V (i₁, i₂)
  p₁ i₁ i₂ _ := D.f i₁ i₂
  p₂ i₁ i₂ _ := D.t i₁ i₂ ≫ D.f i₂ i₁
                  /-
                    D : AlgebraicGeometry.Scheme.GlueData
                    i₁ i₂ : D.J
                    x✝ : (fun x x => PUnit.{u + 1}) i₁ i₂
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i₁ i₂ x => D.f i₁ i₂) i₁ i₂ x✝) …
                  -/
  w i₁ i₂ _ := by simp only [Category.assoc, Scheme.GlueData.glue_condition]
                  /-
                    🎉 no goals
                  -/
  mem₀ := by
    /-
      D : AlgebraicGeometry.Scheme.GlueData
      ⊢ Membership.mem (AlgebraicGeometry.Scheme.zariskiTopology D.glued) { I₀ := D. …
    -/
    refine zariskiTopology.superset_covering ?_ (grothendieckTopology_cover D.openCover)
    /-
      D : AlgebraicGeometry.Scheme.GlueData
      ⊢ LE.le (CategoryTheory.Sieve.generate (CategoryTheory.Presieve.ofArrows D.ope …
    -/
    rw [Sieve.generate_le_iff]
    /-
      D : AlgebraicGeometry.Scheme.GlueData
      ⊢ LE.le (CategoryTheory.Presieve.ofArrows D.openCover.obj D.openCover.map) { I …
    -/
    rintro W _ ⟨i⟩
    /-
      case mk
      D : AlgebraicGeometry.Scheme.GlueData
      Y : AlgebraicGeometry.Scheme
      i : D.openCover.J
      ⊢ Membership.mem { I₀ := D.J, X := D.U, f := D.ι, I₁ := fun x x => PUnit.{u +  …
    -/
    exact ⟨_, 𝟙 _, _, ⟨i⟩, by simp; rfl⟩
    /-
      🎉 no goals
    -/
  mem₁ i₁ i₂ W p₁ p₂ fac := by
    /-
      D : AlgebraicGeometry.Scheme.GlueData
      i₁ i₂ : { I₀ := D.J, X := D.U, f := D.ι, I₁ := fun x x => PUnit.{u + 1}, Y :=  …
      W : AlgebraicGeometry.Scheme
      p₁ : Quiver.Hom W ({ I₀ := D.J, X := D.U, f := D.ι, I₁ := fun x x => PUnit.{u  …
      p₂ : Quiver.Hom W ({ I₀ := D.J, X := D.U, f := D.ι, I₁ := fun x x => PUnit.{u  …
      fac : Eq (CategoryTheory.CategoryStruct.comp p₁ ({ I₀ := D.J, X := D.U, f := D …
      ⊢ Membership.mem (AlgebraicGeometry.Scheme.zariskiTopology W) ({ I₀ := D.J, X  …
    -/
    refine zariskiTopology.superset_covering (fun T g _ ↦ ?_) (zariskiTopology.top_mem _)
    have ⟨φ, h₁, h₂⟩ := PullbackCone.IsLimit.lift' (D.vPullbackConeIsLimit i₁ i₂)
      (g ≫ p₁) (g ≫ p₂) (by simpa using g ≫= fac)
    /-
      D : AlgebraicGeometry.Scheme.GlueData
      i₁ i₂ : { I₀ := D.J, X := D.U, f := D.ι, I₁ := fun x x => PUnit.{u + 1}, Y :=  …
      W : AlgebraicGeometry.Scheme
      p₁ : Quiver.Hom W ({ I₀ := D.J, X := D.U, f := D.ι, I₁ := fun x x => PUnit.{u  …
      p₂ : Quiver.Hom W ({ I₀ := D.J, X := D.U, f := D.ι, I₁ := fun x x => PUnit.{u  …
      fac : Eq (CategoryTheory.CategoryStruct.comp p₁ ({ I₀ := D.J, X := D.U, f := D …
      T : AlgebraicGeometry.Scheme
      g : Quiver.Hom T W
      x✝ : Top.top.arrows g
      φ : Quiver.Hom T (D.vPullbackCone i₁ i₂).pt
      h₁ : Eq (CategoryTheory.CategoryStruct.comp φ (D.vPullbackCone i₁ i₂).fst) (Ca …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp φ (D.vPullbackCone i₁ i₂).snd) (Ca …
      ⊢ ({ I₀ := D.J, X := D.U, f := D.ι, I₁ := fun x x => PUnit.{u + 1}, Y := fun i …
    -/
    exact ⟨⟨⟩, φ, h₁.symm, h₂.symm⟩
    /-
      🎉 no goals
    -/


/-- Constructor for sections over `D.glued` of a sheaf of types on the big Zariski site. -/
noncomputable def sheafValGluedMk : F.val.obj (op D.glued) :=
  Multifork.IsLimit.sectionsEquiv (D.oneHypercover.isLimitMultifork F)
    { val := s
      property := fun _ ↦ h _ _ }


@[simp]
lemma sheafValGluedMk_val (j : D.J) : F.val.map (D.ι j).op (D.sheafValGluedMk s h) = s j :=
  Multifork.IsLimit.sectionsEquiv_apply_val (D.oneHypercover.isLimitMultifork F) _ _



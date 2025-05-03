/-- A functor preserves homology when it preserves both kernels and cokernels. -/
class PreservesHomology (F : C ⥤ D) [PreservesZeroMorphisms F] : Prop where
  /-- the functor preserves kernels -/
  preservesKernels ⦃X Y : C⦄ (f : X ⟶ Y) : PreservesLimit (parallelPair f 0) F := by
    infer_instance
  /-- the functor preserves cokernels -/
  preservesCokernels ⦃X Y : C⦄ (f : X ⟶ Y) : PreservesColimit (parallelPair f 0) F := by
    infer_instance


/-- A functor which preserves homology preserves kernels. -/
lemma PreservesHomology.preservesKernel [F.PreservesHomology] {X Y : C} (f : X ⟶ Y) :
    PreservesLimit (parallelPair f 0) F :=
  PreservesHomology.preservesKernels _


/-- A functor which preserves homology preserves cokernels. -/
lemma PreservesHomology.preservesCokernel [F.PreservesHomology] {X Y : C} (f : X ⟶ Y) :
    PreservesColimit (parallelPair f 0) F :=
  PreservesHomology.preservesCokernels _


noncomputable instance preservesHomologyOfExact
    [PreservesFiniteLimits F] [PreservesFiniteColimits F] :
  F.PreservesHomology where


/-- A left homology data `h` of a short complex `S` is preserved by a functor `F` is
`F` preserves the kernel of `S.g : S.X₂ ⟶ S.X₃` and the cokernel of `h.f' : S.X₁ ⟶ h.K`. -/
class IsPreservedBy [F.PreservesZeroMorphisms] : Prop where
  /-- the functor preserves the kernel of `S.g : S.X₂ ⟶ S.X₃`. -/
  g : PreservesLimit (parallelPair S.g 0) F
  /-- the functor preserves the cokernel of `h.f' : S.X₁ ⟶ h.K`. -/
  f' : PreservesColimit (parallelPair h.f' 0) F


noncomputable instance isPreservedBy_of_preservesHomology [F.PreservesHomology] :
    h.IsPreservedBy F where
  g := Functor.PreservesHomology.preservesKernel _ _
  f' := Functor.PreservesHomology.preservesCokernel _ _


include h in
/-- When a left homology data is preserved by a functor `F`, this functor
preserves the kernel of `S.g : S.X₂ ⟶ S.X₃`. -/
lemma IsPreservedBy.hg : PreservesLimit (parallelPair S.g 0) F :=
  @IsPreservedBy.g _ _ _ _ _ _ _ h F _ _


/-- When a left homology data `h` is preserved by a functor `F`, this functor
preserves the cokernel of `h.f' : S.X₁ ⟶ h.K`. -/
lemma IsPreservedBy.hf' : PreservesColimit (parallelPair h.f' 0) F := IsPreservedBy.f'


/-- When a left homology data `h` of a short complex `S` is preserved by a functor `F`,
this is the induced left homology data `h.map F` for the short complex `S.map F`. -/
@[simps]
noncomputable def map : (S.map F).LeftHomologyData := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{?u.12258, u_1} C
    inst✝⁴ : CategoryTheory.Category.{?u.12262, u_2} D
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
    S S₁ S₂ : CategoryTheory.ShortComplex C
    h : S.LeftHomologyData
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    inst✝ : h.IsPreservedBy F
    ⊢ (S.map F).LeftHomologyData
  -/
  have := IsPreservedBy.hg h F
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{?u.12258, u_1} C
    inst✝⁴ : CategoryTheory.Category.{?u.12262, u_2} D
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
    S S₁ S₂ : CategoryTheory.ShortComplex C
    h : S.LeftHomologyData
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    inst✝ : h.IsPreservedBy F
    this : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPai …
    ⊢ (S.map F).LeftHomologyData
  -/
  have := IsPreservedBy.hf' h F
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{?u.12258, u_1} C
    inst✝⁴ : CategoryTheory.Category.{?u.12262, u_2} D
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
    S S₁ S₂ : CategoryTheory.ShortComplex C
    h : S.LeftHomologyData
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    inst✝ : h.IsPreservedBy F
    this✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
    this : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelP …
    ⊢ (S.map F).LeftHomologyData
  -/
  have wi : F.map h.i ≫ F.map S.g = 0 := by rw [← F.map_comp, h.wi, F.map_zero]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{?u.12258, u_1} C
    inst✝⁴ : CategoryTheory.Category.{?u.12262, u_2} D
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
    S S₁ S₂ : CategoryTheory.ShortComplex C
    h : S.LeftHomologyData
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    inst✝ : h.IsPreservedBy F
    this✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
    this : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelP …
    wi : Eq (CategoryTheory.CategoryStruct.comp (F.map h.i) (F.map S.g)) 0
    ⊢ (S.map F).LeftHomologyData
  -/
  have hi := KernelFork.mapIsLimit _ h.hi F
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{?u.12258, u_1} C
    inst✝⁴ : CategoryTheory.Category.{?u.12262, u_2} D
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
    S S₁ S₂ : CategoryTheory.ShortComplex C
    h : S.LeftHomologyData
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    inst✝ : h.IsPreservedBy F
    this✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
    this : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelP …
    wi : Eq (CategoryTheory.CategoryStruct.comp (F.map h.i) (F.map S.g)) 0
    hi : CategoryTheory.Limits.IsLimit ((CategoryTheory.Limits.KernelFork.ofι h.i  …
    ⊢ (S.map F).LeftHomologyData
  -/
  let f' : F.obj S.X₁ ⟶ F.obj h.K := hi.lift (KernelFork.ofι (S.map F).f (S.map F).zero)
  have hf' : f' = F.map h.f' := Fork.IsLimit.hom_ext hi (by
    rw [Fork.IsLimit.lift_ι hi]
    simp only [KernelFork.map_ι, Fork.ι_ofι, map_f, ← F.map_comp, f'_i])
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{?u.12258, u_1} C
    inst✝⁴ : CategoryTheory.Category.{?u.12262, u_2} D
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
    S S₁ S₂ : CategoryTheory.ShortComplex C
    h : S.LeftHomologyData
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    inst✝ : h.IsPreservedBy F
    this✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
    this : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelP …
    wi : Eq (CategoryTheory.CategoryStruct.comp (F.map h.i) (F.map S.g)) 0
    hi : CategoryTheory.Limits.IsLimit ((CategoryTheory.Limits.KernelFork.ofι h.i  …
    f' : Quiver.Hom (F.obj S.X₁) (F.obj h.K) := hi.lift (CategoryTheory.Limits.Ker …
    hf' : Eq f' (F.map h.f')
    ⊢ (S.map F).LeftHomologyData
  -/
  have wπ : f' ≫ F.map h.π = 0 := by rw [hf', ← F.map_comp, f'_π, F.map_zero]
  have hπ : IsColimit (CokernelCofork.ofπ (F.map h.π) wπ) := by
    let e : parallelPair f' 0 ≅ parallelPair (F.map h.f') 0 :=
      parallelPair.ext (Iso.refl _) (Iso.refl _) (by simpa using hf') (by simp)
    refine IsColimit.precomposeInvEquiv e _
      (IsColimit.ofIsoColimit (CokernelCofork.mapIsColimit _ h.hπ' F) ?_)
    exact Cofork.ext (Iso.refl _) (by simp [e])
  exact
    { K := F.obj h.K
      H := F.obj h.H
      i := F.map h.i
      π := F.map h.π
      wi := wi
      hi := hi
      wπ := wπ
      hπ := hπ }


@[simp]
lemma map_f' : (h.map F).f' = F.map h.f' := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    h : S.LeftHomologyData
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    inst✝ : h.IsPreservedBy F
    ⊢ Eq (h.map F).f' (F.map h.f')
  -/
  rw [← cancel_mono (h.map F).i, f'_i, map_f, map_i, ← F.map_comp, f'_i]
  /-
    🎉 no goals
  -/


/-- Given a left homology map data `ψ : LeftHomologyMapData φ h₁ h₂` such that
both left homology data `h₁` and `h₂` are preserved by a functor `F`, this is
the induced left homology map data for the morphism `F.mapShortComplex.map φ`. -/
@[simps]
def LeftHomologyMapData.map {φ : S₁ ⟶ S₂} {h₁ : S₁.LeftHomologyData}
    {h₂ : S₂.LeftHomologyData} (ψ : LeftHomologyMapData φ h₁ h₂) (F : C ⥤ D)
    [F.PreservesZeroMorphisms] [h₁.IsPreservedBy F] [h₂.IsPreservedBy F] :
    LeftHomologyMapData (F.mapShortComplex.map φ) (h₁.map F) (h₂.map F) where
  φK := F.map ψ.φK
  φH := F.map ψ.φH
              /-
                C : Type u_1
                D : Type u_2
                inst✝⁶ : CategoryTheory.Category.{?u.23316, u_1} C
                inst✝⁵ : CategoryTheory.Category.{?u.23320, u_2} D
                inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
                inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
                S S₁ S₂ : CategoryTheory.ShortComplex C
                φ : Quiver.Hom S₁ S₂
                h₁ : S₁.LeftHomologyData
                h₂ : S₂.LeftHomologyData
                ψ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
                F : CategoryTheory.Functor C D
                inst✝² : F.PreservesZeroMorphisms
                inst✝¹ : h₁.IsPreservedBy F
                inst✝ : h₂.IsPreservedBy F
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ψ.φK) (h₂.map F).i) (CategoryT …
              -/
  commi := by simpa only [F.map_comp] using F.congr_map ψ.commi
              /-
                🎉 no goals
              -/
               /-
                 C : Type u_1
                 D : Type u_2
                 inst✝⁶ : CategoryTheory.Category.{?u.23316, u_1} C
                 inst✝⁵ : CategoryTheory.Category.{?u.23320, u_2} D
                 inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
                 inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
                 S S₁ S₂ : CategoryTheory.ShortComplex C
                 φ : Quiver.Hom S₁ S₂
                 h₁ : S₁.LeftHomologyData
                 h₂ : S₂.LeftHomologyData
                 ψ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
                 F : CategoryTheory.Functor C D
                 inst✝² : F.PreservesZeroMorphisms
                 inst✝¹ : h₁.IsPreservedBy F
                 inst✝ : h₂.IsPreservedBy F
                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (h₁.map F).f' (F.map ψ.φK)) (Category …
               -/
  commf' := by simpa only [LeftHomologyData.map_f', F.map_comp] using F.congr_map ψ.commf'
               /-
                 🎉 no goals
               -/
              /-
                C : Type u_1
                D : Type u_2
                inst✝⁶ : CategoryTheory.Category.{?u.23316, u_1} C
                inst✝⁵ : CategoryTheory.Category.{?u.23320, u_2} D
                inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
                inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
                S S₁ S₂ : CategoryTheory.ShortComplex C
                φ : Quiver.Hom S₁ S₂
                h₁ : S₁.LeftHomologyData
                h₂ : S₂.LeftHomologyData
                ψ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
                F : CategoryTheory.Functor C D
                inst✝² : F.PreservesZeroMorphisms
                inst✝¹ : h₁.IsPreservedBy F
                inst✝ : h₂.IsPreservedBy F
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (h₁.map F).π (F.map ψ.φH)) (CategoryT …
              -/
  commπ := by simpa only [F.map_comp] using F.congr_map ψ.commπ
              /-
                🎉 no goals
              -/


/-- A right homology data `h` of a short complex `S` is preserved by a functor `F` is
`F` preserves the cokernel of `S.f : S.X₁ ⟶ S.X₂` and the kernel of `h.g' : h.Q ⟶ S.X₃`. -/
class IsPreservedBy [F.PreservesZeroMorphisms] : Prop where
  /-- the functor preserves the cokernel of `S.f : S.X₁ ⟶ S.X₂`. -/
  f : PreservesColimit (parallelPair S.f 0) F
  /-- the functor preserves the kernel of `h.g' : h.Q ⟶ S.X₃`. -/
  g' : PreservesLimit (parallelPair h.g' 0) F


noncomputable instance isPreservedBy_of_preservesHomology [F.PreservesHomology] :
    h.IsPreservedBy F where
  f := Functor.PreservesHomology.preservesCokernel F _
  g' := Functor.PreservesHomology.preservesKernel F _


include h in
/-- When a right homology data is preserved by a functor `F`, this functor
preserves the cokernel of `S.f : S.X₁ ⟶ S.X₂`. -/
lemma IsPreservedBy.hf : PreservesColimit (parallelPair S.f 0) F :=
  @IsPreservedBy.f _ _ _ _ _ _ _ h F _ _


/-- When a right homology data `h` is preserved by a functor `F`, this functor
preserves the kernel of `h.g' : h.Q ⟶ S.X₃`. -/
lemma IsPreservedBy.hg' : PreservesLimit (parallelPair h.g' 0) F :=
  @IsPreservedBy.g' _ _ _ _ _ _ _ h F _ _


/-- When a right homology data `h` of a short complex `S` is preserved by a functor `F`,
this is the induced right homology data `h.map F` for the short complex `S.map F`. -/
@[simps]
noncomputable def map : (S.map F).RightHomologyData := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{?u.31923, u_1} C
    inst✝⁴ : CategoryTheory.Category.{?u.31927, u_2} D
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
    S S₁ S₂ : CategoryTheory.ShortComplex C
    h : S.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    inst✝ : h.IsPreservedBy F
    ⊢ (S.map F).RightHomologyData
  -/
  have := IsPreservedBy.hf h F
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{?u.31923, u_1} C
    inst✝⁴ : CategoryTheory.Category.{?u.31927, u_2} D
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
    S S₁ S₂ : CategoryTheory.ShortComplex C
    h : S.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    inst✝ : h.IsPreservedBy F
    this : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelP …
    ⊢ (S.map F).RightHomologyData
  -/
  have := IsPreservedBy.hg' h F
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{?u.31923, u_1} C
    inst✝⁴ : CategoryTheory.Category.{?u.31927, u_2} D
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
    S S₁ S₂ : CategoryTheory.ShortComplex C
    h : S.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    inst✝ : h.IsPreservedBy F
    this✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
    this : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPai …
    ⊢ (S.map F).RightHomologyData
  -/
  have wp : F.map S.f ≫ F.map h.p = 0 := by rw [← F.map_comp, h.wp, F.map_zero]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{?u.31923, u_1} C
    inst✝⁴ : CategoryTheory.Category.{?u.31927, u_2} D
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
    S S₁ S₂ : CategoryTheory.ShortComplex C
    h : S.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    inst✝ : h.IsPreservedBy F
    this✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
    this : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPai …
    wp : Eq (CategoryTheory.CategoryStruct.comp (F.map S.f) (F.map h.p)) 0
    ⊢ (S.map F).RightHomologyData
  -/
  have hp := CokernelCofork.mapIsColimit _ h.hp F
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{?u.31923, u_1} C
    inst✝⁴ : CategoryTheory.Category.{?u.31927, u_2} D
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
    S S₁ S₂ : CategoryTheory.ShortComplex C
    h : S.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    inst✝ : h.IsPreservedBy F
    this✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
    this : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPai …
    wp : Eq (CategoryTheory.CategoryStruct.comp (F.map S.f) (F.map h.p)) 0
    hp : CategoryTheory.Limits.IsColimit ((CategoryTheory.Limits.CokernelCofork.of …
    ⊢ (S.map F).RightHomologyData
  -/
  let g' : F.obj h.Q ⟶ F.obj S.X₃ := hp.desc (CokernelCofork.ofπ (S.map F).g (S.map F).zero)
  have hg' : g' = F.map h.g' := by
    apply Cofork.IsColimit.hom_ext hp
    rw [Cofork.IsColimit.π_desc hp]
    simp only [Cofork.π_ofπ, CokernelCofork.map_π, map_g, ← F.map_comp, p_g']
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{?u.31923, u_1} C
    inst✝⁴ : CategoryTheory.Category.{?u.31927, u_2} D
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
    S S₁ S₂ : CategoryTheory.ShortComplex C
    h : S.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    inst✝ : h.IsPreservedBy F
    this✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
    this : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPai …
    wp : Eq (CategoryTheory.CategoryStruct.comp (F.map S.f) (F.map h.p)) 0
    hp : CategoryTheory.Limits.IsColimit ((CategoryTheory.Limits.CokernelCofork.of …
    g' : Quiver.Hom (F.obj h.Q) (F.obj S.X₃) := hp.desc (CategoryTheory.Limits.Cok …
    hg' : Eq g' (F.map h.g')
    ⊢ (S.map F).RightHomologyData
  -/
  have wι : F.map h.ι ≫ g' = 0 := by rw [hg', ← F.map_comp, ι_g', F.map_zero]
  have hι : IsLimit (KernelFork.ofι (F.map h.ι) wι) := by
    let e : parallelPair g' 0 ≅ parallelPair (F.map h.g') 0 :=
      parallelPair.ext (Iso.refl _) (Iso.refl _) (by simpa using hg') (by simp)
    refine IsLimit.postcomposeHomEquiv e _
      (IsLimit.ofIsoLimit (KernelFork.mapIsLimit _ h.hι' F) ?_)
    exact Fork.ext (Iso.refl _) (by simp [e])
  exact
    { Q := F.obj h.Q
      H := F.obj h.H
      p := F.map h.p
      ι := F.map h.ι
      wp := wp
      hp := hp
      wι := wι
      hι := hι }


@[simp]
lemma map_g' : (h.map F).g' = F.map h.g' := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} D
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    h : S.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    inst✝ : h.IsPreservedBy F
    ⊢ Eq (h.map F).g' (F.map h.g')
  -/
  rw [← cancel_epi (h.map F).p, p_g', map_g, map_p, ← F.map_comp, p_g']
  /-
    🎉 no goals
  -/


/-- Given a right homology map data `ψ : RightHomologyMapData φ h₁ h₂` such that
both right homology data `h₁` and `h₂` are preserved by a functor `F`, this is
the induced right homology map data for the morphism `F.mapShortComplex.map φ`. -/
@[simps]
def RightHomologyMapData.map {φ : S₁ ⟶ S₂} {h₁ : S₁.RightHomologyData}
    {h₂ : S₂.RightHomologyData} (ψ : RightHomologyMapData φ h₁ h₂) (F : C ⥤ D)
    [F.PreservesZeroMorphisms] [h₁.IsPreservedBy F] [h₂.IsPreservedBy F] :
    RightHomologyMapData (F.mapShortComplex.map φ) (h₁.map F) (h₂.map F) where
  φQ := F.map ψ.φQ
  φH := F.map ψ.φH
              /-
                C : Type u_1
                D : Type u_2
                inst✝⁶ : CategoryTheory.Category.{?u.42756, u_1} C
                inst✝⁵ : CategoryTheory.Category.{?u.42760, u_2} D
                inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
                inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
                S S₁ S₂ : CategoryTheory.ShortComplex C
                φ : Quiver.Hom S₁ S₂
                h₁ : S₁.RightHomologyData
                h₂ : S₂.RightHomologyData
                ψ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
                F : CategoryTheory.Functor C D
                inst✝² : F.PreservesZeroMorphisms
                inst✝¹ : h₁.IsPreservedBy F
                inst✝ : h₂.IsPreservedBy F
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (h₁.map F).p (F.map ψ.φQ)) (CategoryT …
              -/
  commp := by simpa only [F.map_comp] using F.congr_map ψ.commp
              /-
                🎉 no goals
              -/
               /-
                 C : Type u_1
                 D : Type u_2
                 inst✝⁶ : CategoryTheory.Category.{?u.42756, u_1} C
                 inst✝⁵ : CategoryTheory.Category.{?u.42760, u_2} D
                 inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
                 inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
                 S S₁ S₂ : CategoryTheory.ShortComplex C
                 φ : Quiver.Hom S₁ S₂
                 h₁ : S₁.RightHomologyData
                 h₂ : S₂.RightHomologyData
                 ψ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
                 F : CategoryTheory.Functor C D
                 inst✝² : F.PreservesZeroMorphisms
                 inst✝¹ : h₁.IsPreservedBy F
                 inst✝ : h₂.IsPreservedBy F
                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ψ.φQ) (h₂.map F).g') (Category …
               -/
  commg' := by simpa only [RightHomologyData.map_g', F.map_comp] using F.congr_map ψ.commg'
               /-
                 🎉 no goals
               -/
              /-
                C : Type u_1
                D : Type u_2
                inst✝⁶ : CategoryTheory.Category.{?u.42756, u_1} C
                inst✝⁵ : CategoryTheory.Category.{?u.42760, u_2} D
                inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
                inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
                S S₁ S₂ : CategoryTheory.ShortComplex C
                φ : Quiver.Hom S₁ S₂
                h₁ : S₁.RightHomologyData
                h₂ : S₂.RightHomologyData
                ψ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
                F : CategoryTheory.Functor C D
                inst✝² : F.PreservesZeroMorphisms
                inst✝¹ : h₁.IsPreservedBy F
                inst✝ : h₂.IsPreservedBy F
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ψ.φH) (h₂.map F).ι) (CategoryT …
              -/
  commι := by simpa only [F.map_comp] using F.congr_map ψ.commι
              /-
                🎉 no goals
              -/


/-- When a homology data `h` of a short complex `S` is such that both `h.left` and
`h.right` are preserved by a functor `F`, this is the induced homology data
`h.map F` for the short complex `S.map F`. -/
@[simps]
noncomputable def HomologyData.map (h : S.HomologyData) (F : C ⥤ D) [F.PreservesZeroMorphisms]
    [h.left.IsPreservedBy F] [h.right.IsPreservedBy F] :
    (S.map F).HomologyData where
  left := h.left.map F
  right := h.right.map F
  iso := F.mapIso h.iso
             /-
               C : Type u_1
               D : Type u_2
               inst✝⁶ : CategoryTheory.Category.{?u.45936, u_1} C
               inst✝⁵ : CategoryTheory.Category.{?u.45940, u_2} D
               inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
               inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
               S S₁ S₂ : CategoryTheory.ShortComplex C
               h : S.HomologyData
               F : CategoryTheory.Functor C D
               inst✝² : F.PreservesZeroMorphisms
               inst✝¹ : h.left.IsPreservedBy F
               inst✝ : h.right.IsPreservedBy F
               ⊢ Eq (CategoryTheory.CategoryStruct.comp (h.left.map F).π (CategoryTheory.Cate …
             -/
  comm := by simpa only [F.map_comp] using F.congr_map h.comm
             /-
               🎉 no goals
             -/


/-- Given a homology map data `ψ : HomologyMapData φ h₁ h₂` such that
`h₁.left`, `h₁.right`, `h₂.left` and `h₂.right` are all preserved by a functor `F`, this is
the induced homology map data for the morphism `F.mapShortComplex.map φ`. -/
@[simps]
def HomologyMapData.map {φ : S₁ ⟶ S₂} {h₁ : S₁.HomologyData} {h₂ : S₂.HomologyData}
    (ψ : HomologyMapData φ h₁ h₂) (F : C ⥤ D) [F.PreservesZeroMorphisms]
    [h₁.left.IsPreservedBy F] [h₁.right.IsPreservedBy F]
    [h₂.left.IsPreservedBy F] [h₂.right.IsPreservedBy F] :
    HomologyMapData (F.mapShortComplex.map φ) (h₁.map F) (h₂.map F) where
  left := ψ.left.map F
  right := ψ.right.map F


/-- A functor preserves the left homology of a short complex `S` if it preserves all the
left homology data of `S`. -/
class PreservesLeftHomologyOf : Prop where
  /-- the functor preserves all the left homology data of the short complex -/
  isPreservedBy : ∀ (h : S.LeftHomologyData), h.IsPreservedBy F


/-- A functor preserves the right homology of a short complex `S` if it preserves all the
right homology data of `S`. -/
class PreservesRightHomologyOf : Prop where
  /-- the functor preserves all the right homology data of the short complex -/
  isPreservedBy : ∀ (h : S.RightHomologyData), h.IsPreservedBy F


instance PreservesHomology.preservesLeftHomologyOf [F.PreservesHomology] :
    F.PreservesLeftHomologyOf S := ⟨inferInstance⟩


instance PreservesHomology.preservesRightHomologyOf [F.PreservesHomology] :
    F.PreservesRightHomologyOf S := ⟨inferInstance⟩


/-- If a functor preserves a certain left homology data of a short complex `S`, then it
preserves the left homology of `S`. -/
lemma PreservesLeftHomologyOf.mk' (h : S.LeftHomologyData) [h.IsPreservedBy F] :
    F.PreservesLeftHomologyOf S where
  isPreservedBy h' :=
    { g := ShortComplex.LeftHomologyData.IsPreservedBy.hg h F
      f' := by
        /-
          C : Type u_1
          D : Type u_2
          inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
          inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
          F : CategoryTheory.Functor C D
          inst✝¹ : F.PreservesZeroMorphisms
          S : CategoryTheory.ShortComplex C
          h : S.LeftHomologyData
          inst✝ : h.IsPreservedBy F
          h' : S.LeftHomologyData
          ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelPair h …
        -/
        have := ShortComplex.LeftHomologyData.IsPreservedBy.hf' h F
        let e : parallelPair h.f' 0 ≅ parallelPair h'.f' 0 :=
          parallelPair.ext (Iso.refl _) (ShortComplex.cyclesMapIso' (Iso.refl S) h h')
            (by simp) (by simp)
        /-
          C : Type u_1
          D : Type u_2
          inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
          inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
          F : CategoryTheory.Functor C D
          inst✝¹ : F.PreservesZeroMorphisms
          S : CategoryTheory.ShortComplex C
          h : S.LeftHomologyData
          inst✝ : h.IsPreservedBy F
          h' : S.LeftHomologyData
          this : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelP …
          e : CategoryTheory.Iso (CategoryTheory.Limits.parallelPair h.f' 0) (CategoryTh …
          ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelPair h …
        -/
        exact preservesColimit_of_iso_diagram F e }
        /-
          🎉 no goals
        -/


/-- If a functor preserves a certain right homology data of a short complex `S`, then it
preserves the right homology of `S`. -/
lemma PreservesRightHomologyOf.mk' (h : S.RightHomologyData) [h.IsPreservedBy F] :
    F.PreservesRightHomologyOf S where
  isPreservedBy h' :=
    { f := ShortComplex.RightHomologyData.IsPreservedBy.hf h F
      g' := by
        /-
          C : Type u_1
          D : Type u_2
          inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
          inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
          F : CategoryTheory.Functor C D
          inst✝¹ : F.PreservesZeroMorphisms
          S : CategoryTheory.ShortComplex C
          h : S.RightHomologyData
          inst✝ : h.IsPreservedBy F
          h' : S.RightHomologyData
          ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPair h'. …
        -/
        have := ShortComplex.RightHomologyData.IsPreservedBy.hg' h F
        let e : parallelPair h.g' 0 ≅ parallelPair h'.g' 0 :=
          parallelPair.ext (ShortComplex.opcyclesMapIso' (Iso.refl S) h h') (Iso.refl _)
            (by simp) (by simp)
        /-
          C : Type u_1
          D : Type u_2
          inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
          inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
          F : CategoryTheory.Functor C D
          inst✝¹ : F.PreservesZeroMorphisms
          S : CategoryTheory.ShortComplex C
          h : S.RightHomologyData
          inst✝ : h.IsPreservedBy F
          h' : S.RightHomologyData
          this : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPai …
          e : CategoryTheory.Iso (CategoryTheory.Limits.parallelPair h.g' 0) (CategoryTh …
          ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPair h'. …
        -/
        exact preservesLimit_of_iso_diagram F e }
        /-
          🎉 no goals
        -/


instance LeftHomologyData.isPreservedBy_of_preserves [F.PreservesLeftHomologyOf S] :
    h₁.IsPreservedBy F :=
  Functor.PreservesLeftHomologyOf.isPreservedBy _


instance RightHomologyData.isPreservedBy_of_preserves [F.PreservesRightHomologyOf S] :
    h₂.IsPreservedBy F :=
  Functor.PreservesRightHomologyOf.isPreservedBy _


instance hasLeftHomology_of_preserves [S.HasLeftHomology] [F.PreservesLeftHomologyOf S] :
    (S.map F).HasLeftHomology :=
  HasLeftHomology.mk' (S.leftHomologyData.map F)


instance hasLeftHomology_of_preserves' [S.HasLeftHomology] [F.PreservesLeftHomologyOf S] :
    (F.mapShortComplex.obj S).HasLeftHomology := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    h₁ : S.LeftHomologyData
    h₂ : S.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : S.HasLeftHomology
    inst✝ : F.PreservesLeftHomologyOf S
    ⊢ (F.mapShortComplex.obj S).HasLeftHomology
  -/
  dsimp; infer_instance
         /-
           🎉 no goals
         -/


instance hasRightHomology_of_preserves [S.HasRightHomology] [F.PreservesRightHomologyOf S] :
    (S.map F).HasRightHomology :=
  HasRightHomology.mk' (S.rightHomologyData.map F)


instance hasRightHomology_of_preserves' [S.HasRightHomology] [F.PreservesRightHomologyOf S] :
    (F.mapShortComplex.obj S).HasRightHomology := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    h₁ : S.LeftHomologyData
    h₂ : S.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : S.HasRightHomology
    inst✝ : F.PreservesRightHomologyOf S
    ⊢ (F.mapShortComplex.obj S).HasRightHomology
  -/
  dsimp; infer_instance
         /-
           🎉 no goals
         -/


instance hasHomology_of_preserves [S.HasHomology] [F.PreservesLeftHomologyOf S]
    [F.PreservesRightHomologyOf S] :
    (S.map F).HasHomology :=
  HasHomology.mk' (S.homologyData.map F)


instance hasHomology_of_preserves' [S.HasHomology] [F.PreservesLeftHomologyOf S]
    [F.PreservesRightHomologyOf S] :
    (F.mapShortComplex.obj S).HasHomology := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    h₁ : S.LeftHomologyData
    h₂ : S.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝³ : F.PreservesZeroMorphisms
    inst✝² : S.HasHomology
    inst✝¹ : F.PreservesLeftHomologyOf S
    inst✝ : F.PreservesRightHomologyOf S
    ⊢ (F.mapShortComplex.obj S).HasHomology
  -/
  dsimp; infer_instance
         /-
           🎉 no goals
         -/


lemma map_cyclesMap' : F.map (ShortComplex.cyclesMap' φ hl₁ hl₂) =
    ShortComplex.cyclesMap' (F.mapShortComplex.map φ) (hl₁.map F) (hl₂.map F) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    hl₁ : S₁.LeftHomologyData
    hl₂ : S₂.LeftHomologyData
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : hl₁.IsPreservedBy F
    inst✝ : hl₂.IsPreservedBy F
    ⊢ Eq (F.map (CategoryTheory.ShortComplex.cyclesMap' φ hl₁ hl₂)) (CategoryTheor …
  -/
  have γ : ShortComplex.LeftHomologyMapData φ hl₁ hl₂ := default
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    hl₁ : S₁.LeftHomologyData
    hl₂ : S₂.LeftHomologyData
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : hl₁.IsPreservedBy F
    inst✝ : hl₂.IsPreservedBy F
    γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ hl₁ hl₂
    ⊢ Eq (F.map (CategoryTheory.ShortComplex.cyclesMap' φ hl₁ hl₂)) (CategoryTheor …
  -/
  rw [γ.cyclesMap'_eq, (γ.map F).cyclesMap'_eq,  ShortComplex.LeftHomologyMapData.map_φK]
  /-
    🎉 no goals
  -/


lemma map_leftHomologyMap' : F.map (ShortComplex.leftHomologyMap' φ hl₁ hl₂) =
    ShortComplex.leftHomologyMap' (F.mapShortComplex.map φ) (hl₁.map F) (hl₂.map F) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    hl₁ : S₁.LeftHomologyData
    hl₂ : S₂.LeftHomologyData
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : hl₁.IsPreservedBy F
    inst✝ : hl₂.IsPreservedBy F
    ⊢ Eq (F.map (CategoryTheory.ShortComplex.leftHomologyMap' φ hl₁ hl₂)) (Categor …
  -/
  have γ : ShortComplex.LeftHomologyMapData φ hl₁ hl₂ := default
  rw [γ.leftHomologyMap'_eq, (γ.map F).leftHomologyMap'_eq,
    ShortComplex.LeftHomologyMapData.map_φH]


lemma map_opcyclesMap' : F.map (ShortComplex.opcyclesMap' φ hr₁ hr₂) =
    ShortComplex.opcyclesMap' (F.mapShortComplex.map φ) (hr₁.map F) (hr₂.map F) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    hr₁ : S₁.RightHomologyData
    hr₂ : S₂.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : hr₁.IsPreservedBy F
    inst✝ : hr₂.IsPreservedBy F
    ⊢ Eq (F.map (CategoryTheory.ShortComplex.opcyclesMap' φ hr₁ hr₂)) (CategoryThe …
  -/
  have γ : ShortComplex.RightHomologyMapData φ hr₁ hr₂ := default
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    hr₁ : S₁.RightHomologyData
    hr₂ : S₂.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : hr₁.IsPreservedBy F
    inst✝ : hr₂.IsPreservedBy F
    γ : CategoryTheory.ShortComplex.RightHomologyMapData φ hr₁ hr₂
    ⊢ Eq (F.map (CategoryTheory.ShortComplex.opcyclesMap' φ hr₁ hr₂)) (CategoryThe …
  -/
  rw [γ.opcyclesMap'_eq, (γ.map F).opcyclesMap'_eq,  ShortComplex.RightHomologyMapData.map_φQ]
  /-
    🎉 no goals
  -/


lemma map_rightHomologyMap' : F.map (ShortComplex.rightHomologyMap' φ hr₁ hr₂) =
    ShortComplex.rightHomologyMap' (F.mapShortComplex.map φ) (hr₁.map F) (hr₂.map F) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    hr₁ : S₁.RightHomologyData
    hr₂ : S₂.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : hr₁.IsPreservedBy F
    inst✝ : hr₂.IsPreservedBy F
    ⊢ Eq (F.map (CategoryTheory.ShortComplex.rightHomologyMap' φ hr₁ hr₂)) (Catego …
  -/
  have γ : ShortComplex.RightHomologyMapData φ hr₁ hr₂ := default
  rw [γ.rightHomologyMap'_eq, (γ.map F).rightHomologyMap'_eq,
    ShortComplex.RightHomologyMapData.map_φH]


lemma HomologyData.map_homologyMap'
    [h₁.left.IsPreservedBy F] [h₁.right.IsPreservedBy F]
    [h₂.left.IsPreservedBy F] [h₂.right.IsPreservedBy F] :
    F.map (ShortComplex.homologyMap' φ h₁ h₂) =
      ShortComplex.homologyMap' (F.mapShortComplex.map φ) (h₁.map F) (h₂.map F) :=
  LeftHomologyData.map_leftHomologyMap' _ _ _ _


/-- When a functor `F` preserves the left homology of a short complex `S`, this is the
canonical isomorphism `(S.map F).cycles ≅ F.obj S.cycles`. -/
noncomputable def mapCyclesIso [S.HasLeftHomology] [F.PreservesLeftHomologyOf S] :
    (S.map F).cycles ≅ F.obj S.cycles :=
  (S.leftHomologyData.map F).cyclesIso


@[reassoc (attr := simp)]
lemma mapCyclesIso_hom_iCycles [S.HasLeftHomology] [F.PreservesLeftHomologyOf S] :
    (S.mapCyclesIso F).hom ≫ F.map S.iCycles = (S.map F).iCycles := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : S.HasLeftHomology
    inst✝ : F.PreservesLeftHomologyOf S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.mapCyclesIso F).hom (F.map S.iCycl …
  -/
  apply LeftHomologyData.cyclesIso_hom_comp_i
  /-
    🎉 no goals
  -/


/-- When a functor `F` preserves the left homology of a short complex `S`, this is the
canonical isomorphism `(S.map F).leftHomology ≅ F.obj S.leftHomology`. -/
noncomputable def mapLeftHomologyIso [S.HasLeftHomology] [F.PreservesLeftHomologyOf S] :
    (S.map F).leftHomology ≅ F.obj S.leftHomology :=
  (S.leftHomologyData.map F).leftHomologyIso


/-- When a functor `F` preserves the right homology of a short complex `S`, this is the
canonical isomorphism `(S.map F).opcycles ≅ F.obj S.opcycles`. -/
noncomputable def mapOpcyclesIso [S.HasRightHomology] [F.PreservesRightHomologyOf S] :
    (S.map F).opcycles ≅ F.obj S.opcycles :=
  (S.rightHomologyData.map F).opcyclesIso


/-- When a functor `F` preserves the right homology of a short complex `S`, this is the
canonical isomorphism `(S.map F).rightHomology ≅ F.obj S.rightHomology`. -/
noncomputable def mapRightHomologyIso [S.HasRightHomology] [F.PreservesRightHomologyOf S] :
    (S.map F).rightHomology ≅ F.obj S.rightHomology :=
  (S.rightHomologyData.map F).rightHomologyIso


/-- When a functor `F` preserves the left homology of a short complex `S`, this is the
canonical isomorphism `(S.map F).homology ≅ F.obj S.homology`. -/
noncomputable def mapHomologyIso [S.HasHomology] [(S.map F).HasHomology]
    [F.PreservesLeftHomologyOf S] :
    (S.map F).homology ≅ F.obj S.homology :=
  (S.homologyData.left.map F).homologyIso


/-- When a functor `F` preserves the right homology of a short complex `S`, this is the
canonical isomorphism `(S.map F).homology ≅ F.obj S.homology`. -/
noncomputable def mapHomologyIso' [S.HasHomology] [(S.map F).HasHomology]
    [F.PreservesRightHomologyOf S] :
    (S.map F).homology ≅ F.obj S.homology :=
  (S.homologyData.right.map F).homologyIso ≪≫ F.mapIso S.homologyData.right.homologyIso.symm


lemma LeftHomologyData.mapCyclesIso_eq [S.HasLeftHomology]
    [F.PreservesLeftHomologyOf S] :
    S.mapCyclesIso F = (hl.map F).cyclesIso ≪≫ F.mapIso hl.cyclesIso.symm := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    hl : S.LeftHomologyData
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : S.HasLeftHomology
    inst✝ : F.PreservesLeftHomologyOf S
    ⊢ Eq (S.mapCyclesIso F) ((hl.map F).cyclesIso.trans (F.mapIso hl.cyclesIso.sym …
  -/
  ext
  /-
    case w
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    hl : S.LeftHomologyData
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : S.HasLeftHomology
    inst✝ : F.PreservesLeftHomologyOf S
    ⊢ Eq (S.mapCyclesIso F).hom ((hl.map F).cyclesIso.trans (F.mapIso hl.cyclesIso …
  -/
  dsimp [mapCyclesIso, cyclesIso]
  simp only [map_cyclesMap', ← cyclesMap'_comp, Functor.map_id, comp_id,
    Functor.mapShortComplex_obj]


lemma LeftHomologyData.mapLeftHomologyIso_eq [S.HasLeftHomology]
    [F.PreservesLeftHomologyOf S] :
    S.mapLeftHomologyIso F = (hl.map F).leftHomologyIso ≪≫ F.mapIso hl.leftHomologyIso.symm := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    hl : S.LeftHomologyData
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : S.HasLeftHomology
    inst✝ : F.PreservesLeftHomologyOf S
    ⊢ Eq (S.mapLeftHomologyIso F) ((hl.map F).leftHomologyIso.trans (F.mapIso hl.l …
  -/
  ext
  /-
    case w.h
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    hl : S.LeftHomologyData
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : S.HasLeftHomology
    inst✝ : F.PreservesLeftHomologyOf S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map F).leftHomologyπ (S.mapLeftHom …
  -/
  dsimp [mapLeftHomologyIso, leftHomologyIso]
  simp only [map_leftHomologyMap', ← leftHomologyMap'_comp, Functor.map_id, comp_id,
    Functor.mapShortComplex_obj]


lemma RightHomologyData.mapOpcyclesIso_eq [S.HasRightHomology]
    [F.PreservesRightHomologyOf S] :
    S.mapOpcyclesIso F = (hr.map F).opcyclesIso ≪≫ F.mapIso hr.opcyclesIso.symm := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    hr : S.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : S.HasRightHomology
    inst✝ : F.PreservesRightHomologyOf S
    ⊢ Eq (S.mapOpcyclesIso F) ((hr.map F).opcyclesIso.trans (F.mapIso hr.opcyclesI …
  -/
  ext
  /-
    case w.h
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    hr : S.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : S.HasRightHomology
    inst✝ : F.PreservesRightHomologyOf S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map F).pOpcycles (S.mapOpcyclesIso …
  -/
  dsimp [mapOpcyclesIso, opcyclesIso]
  simp only [map_opcyclesMap', ← opcyclesMap'_comp, Functor.map_id, comp_id,
    Functor.mapShortComplex_obj]


lemma RightHomologyData.mapRightHomologyIso_eq [S.HasRightHomology]
    [F.PreservesRightHomologyOf S] :
    S.mapRightHomologyIso F = (hr.map F).rightHomologyIso ≪≫
      F.mapIso hr.rightHomologyIso.symm := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    hr : S.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : S.HasRightHomology
    inst✝ : F.PreservesRightHomologyOf S
    ⊢ Eq (S.mapRightHomologyIso F) ((hr.map F).rightHomologyIso.trans (F.mapIso hr …
  -/
  ext
  /-
    case w
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    hr : S.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : S.HasRightHomology
    inst✝ : F.PreservesRightHomologyOf S
    ⊢ Eq (S.mapRightHomologyIso F).hom ((hr.map F).rightHomologyIso.trans (F.mapIs …
  -/
  dsimp [mapRightHomologyIso, rightHomologyIso]
  simp only [map_rightHomologyMap', ← rightHomologyMap'_comp, Functor.map_id, comp_id,
    Functor.mapShortComplex_obj]


lemma LeftHomologyData.mapHomologyIso_eq [S.HasHomology]
    [(S.map F).HasHomology] [F.PreservesLeftHomologyOf S] :
    S.mapHomologyIso F = (hl.map F).homologyIso ≪≫ F.mapIso hl.homologyIso.symm := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    hl : S.LeftHomologyData
    F : CategoryTheory.Functor C D
    inst✝³ : F.PreservesZeroMorphisms
    inst✝² : S.HasHomology
    inst✝¹ : (S.map F).HasHomology
    inst✝ : F.PreservesLeftHomologyOf S
    ⊢ Eq (S.mapHomologyIso F) ((hl.map F).homologyIso.trans (F.mapIso hl.homologyI …
  -/
  ext
  dsimp only [mapHomologyIso, homologyIso, ShortComplex.leftHomologyIso,
    leftHomologyMapIso', leftHomologyIso, Functor.mapIso,
    Iso.symm, Iso.trans, Iso.refl]
  simp only [F.map_comp, map_leftHomologyMap', ← leftHomologyMap'_comp, comp_id,
    Functor.map_id, Functor.mapShortComplex_obj]


lemma RightHomologyData.mapHomologyIso'_eq [S.HasHomology]
    [(S.map F).HasHomology] [F.PreservesRightHomologyOf S] :
    S.mapHomologyIso' F = (hr.map F).homologyIso ≪≫ F.mapIso hr.homologyIso.symm := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    hr : S.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝³ : F.PreservesZeroMorphisms
    inst✝² : S.HasHomology
    inst✝¹ : (S.map F).HasHomology
    inst✝ : F.PreservesRightHomologyOf S
    ⊢ Eq (S.mapHomologyIso' F) ((hr.map F).homologyIso.trans (F.mapIso hr.homology …
  -/
  ext
  dsimp only [Iso.trans, Iso.symm, Iso.refl, Functor.mapIso, mapHomologyIso', homologyIso,
    rightHomologyIso, rightHomologyMapIso', ShortComplex.rightHomologyIso]
  /-
    case w
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    hr : S.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝³ : F.PreservesZeroMorphisms
    inst✝² : S.HasHomology
    inst✝¹ : (S.map F).HasHomology
    inst✝ : F.PreservesRightHomologyOf S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [assoc, F.map_comp, map_rightHomologyMap', ← rightHomologyMap'_comp_assoc]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma mapCyclesIso_hom_naturality [S₁.HasLeftHomology] [S₂.HasLeftHomology]
    [F.PreservesLeftHomologyOf S₁] [F.PreservesLeftHomologyOf S₂] :
    cyclesMap (F.mapShortComplex.map φ) ≫ (S₂.mapCyclesIso F).hom =
      (S₁.mapCyclesIso F).hom ≫ F.map (cyclesMap φ) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁶ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms D
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    F : CategoryTheory.Functor C D
    inst✝⁴ : F.PreservesZeroMorphisms
    inst✝³ : S₁.HasLeftHomology
    inst✝² : S₂.HasLeftHomology
    inst✝¹ : F.PreservesLeftHomologyOf S₁
    inst✝ : F.PreservesLeftHomologyOf S₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.cyclesMa …
  -/
  dsimp only [cyclesMap, mapCyclesIso, LeftHomologyData.cyclesIso, cyclesMapIso', Iso.refl]
  simp only [LeftHomologyData.map_cyclesMap', Functor.mapShortComplex_obj, ← cyclesMap'_comp,
    comp_id, id_comp]


@[reassoc]
lemma mapCyclesIso_inv_naturality [S₁.HasLeftHomology] [S₂.HasLeftHomology]
    [F.PreservesLeftHomologyOf S₁] [F.PreservesLeftHomologyOf S₂] :
    F.map (cyclesMap φ) ≫ (S₂.mapCyclesIso F).inv =
      (S₁.mapCyclesIso F).inv ≫ cyclesMap (F.mapShortComplex.map φ) := by
  rw [← cancel_epi (S₁.mapCyclesIso F).hom, ← mapCyclesIso_hom_naturality_assoc,
    Iso.hom_inv_id, comp_id, Iso.hom_inv_id_assoc]


@[reassoc]
lemma mapLeftHomologyIso_hom_naturality [S₁.HasLeftHomology] [S₂.HasLeftHomology]
    [F.PreservesLeftHomologyOf S₁] [F.PreservesLeftHomologyOf S₂] :
    leftHomologyMap (F.mapShortComplex.map φ) ≫ (S₂.mapLeftHomologyIso F).hom =
      (S₁.mapLeftHomologyIso F).hom ≫ F.map (leftHomologyMap φ) := by
  dsimp only [leftHomologyMap, mapLeftHomologyIso, LeftHomologyData.leftHomologyIso,
    leftHomologyMapIso', Iso.refl]
  simp only [LeftHomologyData.map_leftHomologyMap', Functor.mapShortComplex_obj,
    ← leftHomologyMap'_comp, comp_id, id_comp]


@[reassoc]
lemma mapLeftHomologyIso_inv_naturality [S₁.HasLeftHomology] [S₂.HasLeftHomology]
    [F.PreservesLeftHomologyOf S₁] [F.PreservesLeftHomologyOf S₂] :
    F.map (leftHomologyMap φ) ≫ (S₂.mapLeftHomologyIso F).inv =
      (S₁.mapLeftHomologyIso F).inv ≫ leftHomologyMap (F.mapShortComplex.map φ) := by
  rw [← cancel_epi (S₁.mapLeftHomologyIso F).hom, ← mapLeftHomologyIso_hom_naturality_assoc,
    Iso.hom_inv_id, comp_id, Iso.hom_inv_id_assoc]


@[reassoc]
lemma mapOpcyclesIso_hom_naturality [S₁.HasRightHomology] [S₂.HasRightHomology]
    [F.PreservesRightHomologyOf S₁] [F.PreservesRightHomologyOf S₂] :
    opcyclesMap (F.mapShortComplex.map φ) ≫ (S₂.mapOpcyclesIso F).hom =
      (S₁.mapOpcyclesIso F).hom ≫ F.map (opcyclesMap φ) := by
  dsimp only [opcyclesMap, mapOpcyclesIso, RightHomologyData.opcyclesIso,
    opcyclesMapIso', Iso.refl]
  simp only [RightHomologyData.map_opcyclesMap', Functor.mapShortComplex_obj, ← opcyclesMap'_comp,
    comp_id, id_comp]


@[reassoc]
lemma mapOpcyclesIso_inv_naturality [S₁.HasRightHomology] [S₂.HasRightHomology]
    [F.PreservesRightHomologyOf S₁] [F.PreservesRightHomologyOf S₂] :
    F.map (opcyclesMap φ) ≫ (S₂.mapOpcyclesIso F).inv =
      (S₁.mapOpcyclesIso F).inv ≫ opcyclesMap (F.mapShortComplex.map φ) := by
  rw [← cancel_epi (S₁.mapOpcyclesIso F).hom, ← mapOpcyclesIso_hom_naturality_assoc,
    Iso.hom_inv_id, comp_id, Iso.hom_inv_id_assoc]


@[reassoc]
lemma mapRightHomologyIso_hom_naturality [S₁.HasRightHomology] [S₂.HasRightHomology]
    [F.PreservesRightHomologyOf S₁] [F.PreservesRightHomologyOf S₂] :
    rightHomologyMap (F.mapShortComplex.map φ) ≫ (S₂.mapRightHomologyIso F).hom =
      (S₁.mapRightHomologyIso F).hom ≫ F.map (rightHomologyMap φ) := by
  dsimp only [rightHomologyMap, mapRightHomologyIso, RightHomologyData.rightHomologyIso,
    rightHomologyMapIso', Iso.refl]
  simp only [RightHomologyData.map_rightHomologyMap', Functor.mapShortComplex_obj,
    ← rightHomologyMap'_comp, comp_id, id_comp]


@[reassoc]
lemma mapRightHomologyIso_inv_naturality [S₁.HasRightHomology] [S₂.HasRightHomology]
    [F.PreservesRightHomologyOf S₁] [F.PreservesRightHomologyOf S₂] :
    F.map (rightHomologyMap φ) ≫ (S₂.mapRightHomologyIso F).inv =
      (S₁.mapRightHomologyIso F).inv ≫ rightHomologyMap (F.mapShortComplex.map φ) := by
  rw [← cancel_epi (S₁.mapRightHomologyIso F).hom, ← mapRightHomologyIso_hom_naturality_assoc,
    Iso.hom_inv_id, comp_id, Iso.hom_inv_id_assoc]


@[reassoc]
lemma mapHomologyIso_hom_naturality [S₁.HasHomology] [S₂.HasHomology]
    [(S₁.map F).HasHomology] [(S₂.map F).HasHomology]
    [F.PreservesLeftHomologyOf S₁] [F.PreservesLeftHomologyOf S₂] :
    @homologyMap _ _ _ (S₁.map F) (S₂.map F) (F.mapShortComplex.map φ) _ _ ≫
      (S₂.mapHomologyIso F).hom = (S₁.mapHomologyIso F).hom ≫ F.map (homologyMap φ) := by
  dsimp only [homologyMap, homologyMap', mapHomologyIso, LeftHomologyData.homologyIso,
    LeftHomologyData.leftHomologyIso, leftHomologyMapIso', leftHomologyIso,
    Iso.symm, Iso.trans, Iso.refl]
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹⁰ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁹ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁷ : CategoryTheory.Limits.HasZeroMorphisms D
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    F : CategoryTheory.Functor C D
    inst✝⁶ : F.PreservesZeroMorphisms
    inst✝⁵ : S₁.HasHomology
    inst✝⁴ : S₂.HasHomology
    inst✝³ : (S₁.map F).HasHomology
    inst✝² : (S₂.map F).HasHomology
    inst✝¹ : F.PreservesLeftHomologyOf S₁
    inst✝ : F.PreservesLeftHomologyOf S₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.leftHomo …
  -/
  simp only [LeftHomologyData.map_leftHomologyMap', ← leftHomologyMap'_comp, comp_id, id_comp]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma mapHomologyIso_inv_naturality [S₁.HasHomology] [S₂.HasHomology]
    [(S₁.map F).HasHomology] [(S₂.map F).HasHomology]
    [F.PreservesLeftHomologyOf S₁] [F.PreservesLeftHomologyOf S₂] :
    F.map (homologyMap φ) ≫ (S₂.mapHomologyIso F).inv =
      (S₁.mapHomologyIso F).inv ≫
      @homologyMap _ _ _ (S₁.map F) (S₂.map F) (F.mapShortComplex.map φ) _ _ := by
  rw [← cancel_epi (S₁.mapHomologyIso F).hom, ← mapHomologyIso_hom_naturality_assoc,
    Iso.hom_inv_id, comp_id, Iso.hom_inv_id_assoc]


@[reassoc]
lemma mapHomologyIso'_hom_naturality [S₁.HasHomology] [S₂.HasHomology]
    [(S₁.map F).HasHomology] [(S₂.map F).HasHomology]
    [F.PreservesRightHomologyOf S₁] [F.PreservesRightHomologyOf S₂] :
    @homologyMap _ _ _ (S₁.map F) (S₂.map F) (F.mapShortComplex.map φ) _ _ ≫
      (S₂.mapHomologyIso' F).hom = (S₁.mapHomologyIso' F).hom ≫ F.map (homologyMap φ) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹⁰ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁹ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁷ : CategoryTheory.Limits.HasZeroMorphisms D
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    F : CategoryTheory.Functor C D
    inst✝⁶ : F.PreservesZeroMorphisms
    inst✝⁵ : S₁.HasHomology
    inst✝⁴ : S₂.HasHomology
    inst✝³ : (S₁.map F).HasHomology
    inst✝² : (S₂.map F).HasHomology
    inst✝¹ : F.PreservesRightHomologyOf S₁
    inst✝ : F.PreservesRightHomologyOf S₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.homology …
  -/
  dsimp only [Iso.trans, Iso.symm, Functor.mapIso, mapHomologyIso']
  simp only [← RightHomologyData.rightHomologyIso_hom_naturality_assoc _
    ((homologyData S₁).right.map F) ((homologyData S₂).right.map F), assoc,
    ← RightHomologyData.map_rightHomologyMap', ← F.map_comp,
    RightHomologyData.rightHomologyIso_inv_naturality _
      (homologyData S₁).right (homologyData S₂).right]


@[reassoc]
lemma mapHomologyIso'_inv_naturality [S₁.HasHomology] [S₂.HasHomology]
    [(S₁.map F).HasHomology] [(S₂.map F).HasHomology]
    [F.PreservesRightHomologyOf S₁] [F.PreservesRightHomologyOf S₂] :
    F.map (homologyMap φ) ≫ (S₂.mapHomologyIso' F).inv = (S₁.mapHomologyIso' F).inv ≫
      @homologyMap _ _ _ (S₁.map F) (S₂.map F) (F.mapShortComplex.map φ) _ _ := by
  rw [← cancel_epi (S₁.mapHomologyIso' F).hom, ← mapHomologyIso'_hom_naturality_assoc,
    Iso.hom_inv_id, comp_id, Iso.hom_inv_id_assoc]


lemma mapHomologyIso'_eq_mapHomologyIso [S.HasHomology] [F.PreservesLeftHomologyOf S]
    [F.PreservesRightHomologyOf S] :
    S.mapHomologyIso' F = S.mapHomologyIso F := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    F : CategoryTheory.Functor C D
    inst✝³ : F.PreservesZeroMorphisms
    inst✝² : S.HasHomology
    inst✝¹ : F.PreservesLeftHomologyOf S
    inst✝ : F.PreservesRightHomologyOf S
    ⊢ Eq (S.mapHomologyIso' F) (S.mapHomologyIso F)
  -/
  ext
  /-
    case w
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    F : CategoryTheory.Functor C D
    inst✝³ : F.PreservesZeroMorphisms
    inst✝² : S.HasHomology
    inst✝¹ : F.PreservesLeftHomologyOf S
    inst✝ : F.PreservesRightHomologyOf S
    ⊢ Eq (S.mapHomologyIso' F).hom (S.mapHomologyIso F).hom
  -/
  rw [S.homologyData.left.mapHomologyIso_eq F, S.homologyData.right.mapHomologyIso'_eq F]
  dsimp only [Iso.trans, Iso.symm, Iso.refl, Functor.mapIso, RightHomologyData.homologyIso,
    rightHomologyIso, RightHomologyData.rightHomologyIso, LeftHomologyData.homologyIso,
    leftHomologyIso, LeftHomologyData.leftHomologyIso]
  simp only [RightHomologyData.map_H, rightHomologyMapIso'_inv, rightHomologyMapIso'_hom, assoc,
    Functor.map_comp, RightHomologyData.map_rightHomologyMap', Functor.mapShortComplex_obj,
    Functor.map_id, LeftHomologyData.map_H, leftHomologyMapIso'_inv, leftHomologyMapIso'_hom,
    LeftHomologyData.map_leftHomologyMap', ← rightHomologyMap'_comp_assoc, ← leftHomologyMap'_comp,
    ← leftHomologyMap'_comp_assoc, id_comp]
  /-
    case w
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    F : CategoryTheory.Functor C D
    inst✝³ : F.PreservesZeroMorphisms
    inst✝² : S.HasHomology
    inst✝¹ : F.PreservesLeftHomologyOf S
    inst✝ : F.PreservesRightHomologyOf S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map F).homologyData.iso.hom (Categ …
  -/
  have γ : HomologyMapData (𝟙 (S.map F)) (map S F).homologyData (S.homologyData.map F) := default
  /-
    case w
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    F : CategoryTheory.Functor C D
    inst✝³ : F.PreservesZeroMorphisms
    inst✝² : S.HasHomology
    inst✝¹ : F.PreservesLeftHomologyOf S
    inst✝ : F.PreservesRightHomologyOf S
    γ : CategoryTheory.ShortComplex.HomologyMapData (CategoryTheory.CategoryStruct …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map F).homologyData.iso.hom (Categ …
  -/
  have eq := γ.comm
  /-
    case w
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    F : CategoryTheory.Functor C D
    inst✝³ : F.PreservesZeroMorphisms
    inst✝² : S.HasHomology
    inst✝¹ : F.PreservesLeftHomologyOf S
    inst✝ : F.PreservesRightHomologyOf S
    γ : CategoryTheory.ShortComplex.HomologyMapData (CategoryTheory.CategoryStruct …
    eq : Eq (CategoryTheory.CategoryStruct.comp γ.left.φH (S.homologyData.map F).i …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map F).homologyData.iso.hom (Categ …
  -/
  rw [← γ.left.leftHomologyMap'_eq, ← γ.right.rightHomologyMap'_eq] at eq
  /-
    case w
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    F : CategoryTheory.Functor C D
    inst✝³ : F.PreservesZeroMorphisms
    inst✝² : S.HasHomology
    inst✝¹ : F.PreservesLeftHomologyOf S
    inst✝ : F.PreservesRightHomologyOf S
    γ : CategoryTheory.ShortComplex.HomologyMapData (CategoryTheory.CategoryStruct …
    eq : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.leftH …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map F).homologyData.iso.hom (Categ …
  -/
  dsimp at eq
  /-
    case w
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    F : CategoryTheory.Functor C D
    inst✝³ : F.PreservesZeroMorphisms
    inst✝² : S.HasHomology
    inst✝¹ : F.PreservesLeftHomologyOf S
    inst✝ : F.PreservesRightHomologyOf S
    γ : CategoryTheory.ShortComplex.HomologyMapData (CategoryTheory.CategoryStruct …
    eq : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.leftH …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map F).homologyData.iso.hom (Categ …
  -/
  simp only [← reassoc_of% eq, ← F.map_comp, Iso.hom_inv_id, F.map_id, comp_id]
  /-
    🎉 no goals
  -/


/-- Given a natural transformation `τ : F ⟶ G` between functors `C ⥤ D` which preserve
the left homology of a short complex `S`, and a left homology data for `S`,
this is the left homology map data for the morphism `S.mapNatTrans τ`
obtained by evaluating `τ`. -/
@[simps]
def LeftHomologyMapData.natTransApp (h : LeftHomologyData S) (τ : F ⟶ G) :
    LeftHomologyMapData (S.mapNatTrans τ) (h.map F) (h.map G) where
  φK := τ.app h.K
  φH := τ.app h.H


/-- Given a natural transformation `τ : F ⟶ G` between functors `C ⥤ D` which preserve
the right homology of a short complex `S`, and a right homology data for `S`,
this is the right homology map data for the morphism `S.mapNatTrans τ`
obtained by evaluating `τ`. -/
@[simps]
def RightHomologyMapData.natTransApp (h : RightHomologyData S) (τ : F ⟶ G) :
    RightHomologyMapData (S.mapNatTrans τ) (h.map F) (h.map G) where
  φQ := τ.app h.Q
  φH := τ.app h.H


/-- Given a natural transformation `τ : F ⟶ G` between functors `C ⥤ D` which preserve
the homology of a short complex `S`, and a homology data for `S`,
this is the homology map data for the morphism `S.mapNatTrans τ`
obtained by evaluating `τ`. -/
@[simps]
def HomologyMapData.natTransApp (h : HomologyData S) (τ : F ⟶ G) :
    HomologyMapData (S.mapNatTrans τ) (h.map F) (h.map G) where
  left := LeftHomologyMapData.natTransApp h.left τ
  right := RightHomologyMapData.natTransApp h.right τ


lemma homologyMap_mapNatTrans [S.HasHomology] (τ : F ⟶ G) :
    homologyMap (S.mapNatTrans τ) =
      (S.mapHomologyIso F).hom ≫ τ.app S.homology ≫ (S.mapHomologyIso G).inv :=
  (LeftHomologyMapData.natTransApp S.homologyData.left τ).homologyMap_eq


/-- The natural isomorphism
`F.mapShortComplex ⋙ cyclesFunctor D ≅ cyclesFunctor C ⋙ F`
for a functor `F : C ⥤ D` which preserves homology. --/
noncomputable def cyclesFunctorIso [F.PreservesHomology] :
    F.mapShortComplex ⋙ ShortComplex.cyclesFunctor D ≅
      ShortComplex.cyclesFunctor C ⋙ F :=
  NatIso.ofComponents (fun S => S.mapCyclesIso F)
    (fun f => ShortComplex.mapCyclesIso_hom_naturality f F)


/-- The natural isomorphism
`F.mapShortComplex ⋙ leftHomologyFunctor D ≅ leftHomologyFunctor C ⋙ F`
for a functor `F : C ⥤ D` which preserves homology. --/
noncomputable def leftHomologyFunctorIso [F.PreservesHomology] :
    F.mapShortComplex ⋙ ShortComplex.leftHomologyFunctor D ≅
      ShortComplex.leftHomologyFunctor C ⋙ F :=
  NatIso.ofComponents (fun S => S.mapLeftHomologyIso F)
    (fun f => ShortComplex.mapLeftHomologyIso_hom_naturality f F)


/-- The natural isomorphism
`F.mapShortComplex ⋙ opcyclesFunctor D ≅ opcyclesFunctor C ⋙ F`
for a functor `F : C ⥤ D` which preserves homology. --/
noncomputable def opcyclesFunctorIso [F.PreservesHomology] :
    F.mapShortComplex ⋙ ShortComplex.opcyclesFunctor D ≅
      ShortComplex.opcyclesFunctor C ⋙ F :=
  NatIso.ofComponents (fun S => S.mapOpcyclesIso F)
    (fun f => ShortComplex.mapOpcyclesIso_hom_naturality f F)


/-- The natural isomorphism
`F.mapShortComplex ⋙ rightHomologyFunctor D ≅ rightHomologyFunctor C ⋙ F`
for a functor `F : C ⥤ D` which preserves homology. --/
noncomputable def rightHomologyFunctorIso [F.PreservesHomology] :
    F.mapShortComplex ⋙ ShortComplex.rightHomologyFunctor D ≅
      ShortComplex.rightHomologyFunctor C ⋙ F :=
  NatIso.ofComponents (fun S => S.mapRightHomologyIso F)
    (fun f => ShortComplex.mapRightHomologyIso_hom_naturality f F)


/-- The natural isomorphism
`F.mapShortComplex ⋙ homologyFunctor D ≅ homologyFunctor C ⋙ F`
for a functor `F : C ⥤ D` which preserves homology. --/
noncomputable def homologyFunctorIso
    [CategoryWithHomology C] [CategoryWithHomology D] [F.PreservesHomology] :
    F.mapShortComplex ⋙ ShortComplex.homologyFunctor D ≅
      ShortComplex.homologyFunctor C ⋙ F :=
  NatIso.ofComponents (fun S => S.mapHomologyIso F)
    (fun f => ShortComplex.mapHomologyIso_hom_naturality f F)


lemma LeftHomologyMapData.quasiIso_map_iff
    [(F.mapShortComplex.obj S₁).HasHomology]
    [(F.mapShortComplex.obj S₂).HasHomology]
    [hl₁.IsPreservedBy F] [hl₂.IsPreservedBy F] :
    QuasiIso (F.mapShortComplex.map φ) ↔ IsIso (F.map ψl.φH) :=
  (ψl.map F).quasiIso_iff


lemma RightHomologyMapData.quasiIso_map_iff
    [(F.mapShortComplex.obj S₁).HasHomology]
    [(F.mapShortComplex.obj S₂).HasHomology]
    [hr₁.IsPreservedBy F] [hr₂.IsPreservedBy F] :
    QuasiIso (F.mapShortComplex.map φ) ↔ IsIso (F.map ψr.φH) :=
  (ψr.map F).quasiIso_iff


instance quasiIso_map_of_preservesLeftHomology
    [F.PreservesLeftHomologyOf S₁] [F.PreservesLeftHomologyOf S₂]
    [QuasiIso φ] : QuasiIso (F.mapShortComplex.map φ) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    h₁ : S.LeftHomologyData
    h₂ : S.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝⁷ : F.PreservesZeroMorphisms
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    hl₁ : S₁.LeftHomologyData
    hr₁ : S₁.RightHomologyData
    hl₂ : S₂.LeftHomologyData
    hr₂ : S₂.RightHomologyData
    ψl : CategoryTheory.ShortComplex.LeftHomologyMapData φ hl₁ hl₂
    ψr : CategoryTheory.ShortComplex.RightHomologyMapData φ hr₁ hr₂
    inst✝⁶ : S₁.HasHomology
    inst✝⁵ : S₂.HasHomology
    inst✝⁴ : (F.mapShortComplex.obj S₁).HasHomology
    inst✝³ : (F.mapShortComplex.obj S₂).HasHomology
    inst✝² : F.PreservesLeftHomologyOf S₁
    inst✝¹ : F.PreservesLeftHomologyOf S₂
    inst✝ : CategoryTheory.ShortComplex.QuasiIso φ
    ⊢ CategoryTheory.ShortComplex.QuasiIso (F.mapShortComplex.map φ)
  -/
  have γ : LeftHomologyMapData φ S₁.leftHomologyData S₂.leftHomologyData := default
  have : IsIso γ.φH := by
    rw [← γ.quasiIso_iff]
    infer_instance
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    h₁ : S.LeftHomologyData
    h₂ : S.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝⁷ : F.PreservesZeroMorphisms
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    hl₁ : S₁.LeftHomologyData
    hr₁ : S₁.RightHomologyData
    hl₂ : S₂.LeftHomologyData
    hr₂ : S₂.RightHomologyData
    ψl : CategoryTheory.ShortComplex.LeftHomologyMapData φ hl₁ hl₂
    ψr : CategoryTheory.ShortComplex.RightHomologyMapData φ hr₁ hr₂
    inst✝⁶ : S₁.HasHomology
    inst✝⁵ : S₂.HasHomology
    inst✝⁴ : (F.mapShortComplex.obj S₁).HasHomology
    inst✝³ : (F.mapShortComplex.obj S₂).HasHomology
    inst✝² : F.PreservesLeftHomologyOf S₁
    inst✝¹ : F.PreservesLeftHomologyOf S₂
    inst✝ : CategoryTheory.ShortComplex.QuasiIso φ
    γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ S₁.leftHomologyData S₂.l …
    this : CategoryTheory.IsIso γ.φH
    ⊢ CategoryTheory.ShortComplex.QuasiIso (F.mapShortComplex.map φ)
  -/
  rw [(γ.map F).quasiIso_iff, LeftHomologyMapData.map_φH]
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    h₁ : S.LeftHomologyData
    h₂ : S.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝⁷ : F.PreservesZeroMorphisms
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    hl₁ : S₁.LeftHomologyData
    hr₁ : S₁.RightHomologyData
    hl₂ : S₂.LeftHomologyData
    hr₂ : S₂.RightHomologyData
    ψl : CategoryTheory.ShortComplex.LeftHomologyMapData φ hl₁ hl₂
    ψr : CategoryTheory.ShortComplex.RightHomologyMapData φ hr₁ hr₂
    inst✝⁶ : S₁.HasHomology
    inst✝⁵ : S₂.HasHomology
    inst✝⁴ : (F.mapShortComplex.obj S₁).HasHomology
    inst✝³ : (F.mapShortComplex.obj S₂).HasHomology
    inst✝² : F.PreservesLeftHomologyOf S₁
    inst✝¹ : F.PreservesLeftHomologyOf S₂
    inst✝ : CategoryTheory.ShortComplex.QuasiIso φ
    γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ S₁.leftHomologyData S₂.l …
    this : CategoryTheory.IsIso γ.φH
    ⊢ CategoryTheory.IsIso (F.map γ.φH)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma quasiIso_map_iff_of_preservesLeftHomology
    [F.PreservesLeftHomologyOf S₁] [F.PreservesLeftHomologyOf S₂]
    [F.ReflectsIsomorphisms] :
    QuasiIso (F.mapShortComplex.map φ) ↔ QuasiIso φ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms D
    F : CategoryTheory.Functor C D
    inst✝⁷ : F.PreservesZeroMorphisms
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝⁶ : S₁.HasHomology
    inst✝⁵ : S₂.HasHomology
    inst✝⁴ : (F.mapShortComplex.obj S₁).HasHomology
    inst✝³ : (F.mapShortComplex.obj S₂).HasHomology
    inst✝² : F.PreservesLeftHomologyOf S₁
    inst✝¹ : F.PreservesLeftHomologyOf S₂
    inst✝ : F.ReflectsIsomorphisms
    ⊢ Iff (CategoryTheory.ShortComplex.QuasiIso (F.mapShortComplex.map φ)) (Catego …
  -/
  have γ : LeftHomologyMapData φ S₁.leftHomologyData S₂.leftHomologyData := default
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms D
    F : CategoryTheory.Functor C D
    inst✝⁷ : F.PreservesZeroMorphisms
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝⁶ : S₁.HasHomology
    inst✝⁵ : S₂.HasHomology
    inst✝⁴ : (F.mapShortComplex.obj S₁).HasHomology
    inst✝³ : (F.mapShortComplex.obj S₂).HasHomology
    inst✝² : F.PreservesLeftHomologyOf S₁
    inst✝¹ : F.PreservesLeftHomologyOf S₂
    inst✝ : F.ReflectsIsomorphisms
    γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ S₁.leftHomologyData S₂.l …
    ⊢ Iff (CategoryTheory.ShortComplex.QuasiIso (F.mapShortComplex.map φ)) (Catego …
  -/
  rw [γ.quasiIso_iff, (γ.map F).quasiIso_iff, LeftHomologyMapData.map_φH]
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms D
    F : CategoryTheory.Functor C D
    inst✝⁷ : F.PreservesZeroMorphisms
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝⁶ : S₁.HasHomology
    inst✝⁵ : S₂.HasHomology
    inst✝⁴ : (F.mapShortComplex.obj S₁).HasHomology
    inst✝³ : (F.mapShortComplex.obj S₂).HasHomology
    inst✝² : F.PreservesLeftHomologyOf S₁
    inst✝¹ : F.PreservesLeftHomologyOf S₂
    inst✝ : F.ReflectsIsomorphisms
    γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ S₁.leftHomologyData S₂.l …
    ⊢ Iff (CategoryTheory.IsIso (F.map γ.φH)) (CategoryTheory.IsIso γ.φH)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝¹¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹⁰ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms D
      F : CategoryTheory.Functor C D
      inst✝⁷ : F.PreservesZeroMorphisms
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      inst✝⁶ : S₁.HasHomology
      inst✝⁵ : S₂.HasHomology
      inst✝⁴ : (F.mapShortComplex.obj S₁).HasHomology
      inst✝³ : (F.mapShortComplex.obj S₂).HasHomology
      inst✝² : F.PreservesLeftHomologyOf S₁
      inst✝¹ : F.PreservesLeftHomologyOf S₂
      inst✝ : F.ReflectsIsomorphisms
      γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ S₁.leftHomologyData S₂.l …
      ⊢ CategoryTheory.IsIso (F.map γ.φH) → CategoryTheory.IsIso γ.φH
    -/
  · intro
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝¹¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹⁰ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms D
      F : CategoryTheory.Functor C D
      inst✝⁷ : F.PreservesZeroMorphisms
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      inst✝⁶ : S₁.HasHomology
      inst✝⁵ : S₂.HasHomology
      inst✝⁴ : (F.mapShortComplex.obj S₁).HasHomology
      inst✝³ : (F.mapShortComplex.obj S₂).HasHomology
      inst✝² : F.PreservesLeftHomologyOf S₁
      inst✝¹ : F.PreservesLeftHomologyOf S₂
      inst✝ : F.ReflectsIsomorphisms
      γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ S₁.leftHomologyData S₂.l …
      a✝ : CategoryTheory.IsIso (F.map γ.φH)
      ⊢ CategoryTheory.IsIso γ.φH
    -/
    exact isIso_of_reflects_iso _ F
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝¹¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹⁰ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms D
      F : CategoryTheory.Functor C D
      inst✝⁷ : F.PreservesZeroMorphisms
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      inst✝⁶ : S₁.HasHomology
      inst✝⁵ : S₂.HasHomology
      inst✝⁴ : (F.mapShortComplex.obj S₁).HasHomology
      inst✝³ : (F.mapShortComplex.obj S₂).HasHomology
      inst✝² : F.PreservesLeftHomologyOf S₁
      inst✝¹ : F.PreservesLeftHomologyOf S₂
      inst✝ : F.ReflectsIsomorphisms
      γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ S₁.leftHomologyData S₂.l …
      ⊢ CategoryTheory.IsIso γ.φH → CategoryTheory.IsIso (F.map γ.φH)
    -/
  · intro
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝¹¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹⁰ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms D
      F : CategoryTheory.Functor C D
      inst✝⁷ : F.PreservesZeroMorphisms
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      inst✝⁶ : S₁.HasHomology
      inst✝⁵ : S₂.HasHomology
      inst✝⁴ : (F.mapShortComplex.obj S₁).HasHomology
      inst✝³ : (F.mapShortComplex.obj S₂).HasHomology
      inst✝² : F.PreservesLeftHomologyOf S₁
      inst✝¹ : F.PreservesLeftHomologyOf S₂
      inst✝ : F.ReflectsIsomorphisms
      γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ S₁.leftHomologyData S₂.l …
      a✝ : CategoryTheory.IsIso γ.φH
      ⊢ CategoryTheory.IsIso (F.map γ.φH)
    -/
    infer_instance
    /-
      🎉 no goals
    -/


instance quasiIso_map_of_preservesRightHomology
    [F.PreservesRightHomologyOf S₁] [F.PreservesRightHomologyOf S₂]
    [QuasiIso φ] : QuasiIso (F.mapShortComplex.map φ) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    h₁ : S.LeftHomologyData
    h₂ : S.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝⁷ : F.PreservesZeroMorphisms
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    hl₁ : S₁.LeftHomologyData
    hr₁ : S₁.RightHomologyData
    hl₂ : S₂.LeftHomologyData
    hr₂ : S₂.RightHomologyData
    ψl : CategoryTheory.ShortComplex.LeftHomologyMapData φ hl₁ hl₂
    ψr : CategoryTheory.ShortComplex.RightHomologyMapData φ hr₁ hr₂
    inst✝⁶ : S₁.HasHomology
    inst✝⁵ : S₂.HasHomology
    inst✝⁴ : (F.mapShortComplex.obj S₁).HasHomology
    inst✝³ : (F.mapShortComplex.obj S₂).HasHomology
    inst✝² : F.PreservesRightHomologyOf S₁
    inst✝¹ : F.PreservesRightHomologyOf S₂
    inst✝ : CategoryTheory.ShortComplex.QuasiIso φ
    ⊢ CategoryTheory.ShortComplex.QuasiIso (F.mapShortComplex.map φ)
  -/
  have γ : RightHomologyMapData φ S₁.rightHomologyData S₂.rightHomologyData := default
  have : IsIso γ.φH := by
    rw [← γ.quasiIso_iff]
    infer_instance
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    h₁ : S.LeftHomologyData
    h₂ : S.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝⁷ : F.PreservesZeroMorphisms
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    hl₁ : S₁.LeftHomologyData
    hr₁ : S₁.RightHomologyData
    hl₂ : S₂.LeftHomologyData
    hr₂ : S₂.RightHomologyData
    ψl : CategoryTheory.ShortComplex.LeftHomologyMapData φ hl₁ hl₂
    ψr : CategoryTheory.ShortComplex.RightHomologyMapData φ hr₁ hr₂
    inst✝⁶ : S₁.HasHomology
    inst✝⁵ : S₂.HasHomology
    inst✝⁴ : (F.mapShortComplex.obj S₁).HasHomology
    inst✝³ : (F.mapShortComplex.obj S₂).HasHomology
    inst✝² : F.PreservesRightHomologyOf S₁
    inst✝¹ : F.PreservesRightHomologyOf S₂
    inst✝ : CategoryTheory.ShortComplex.QuasiIso φ
    γ : CategoryTheory.ShortComplex.RightHomologyMapData φ S₁.rightHomologyData S₂ …
    this : CategoryTheory.IsIso γ.φH
    ⊢ CategoryTheory.ShortComplex.QuasiIso (F.mapShortComplex.map φ)
  -/
  rw [(γ.map F).quasiIso_iff, RightHomologyMapData.map_φH]
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    h₁ : S.LeftHomologyData
    h₂ : S.RightHomologyData
    F : CategoryTheory.Functor C D
    inst✝⁷ : F.PreservesZeroMorphisms
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    hl₁ : S₁.LeftHomologyData
    hr₁ : S₁.RightHomologyData
    hl₂ : S₂.LeftHomologyData
    hr₂ : S₂.RightHomologyData
    ψl : CategoryTheory.ShortComplex.LeftHomologyMapData φ hl₁ hl₂
    ψr : CategoryTheory.ShortComplex.RightHomologyMapData φ hr₁ hr₂
    inst✝⁶ : S₁.HasHomology
    inst✝⁵ : S₂.HasHomology
    inst✝⁴ : (F.mapShortComplex.obj S₁).HasHomology
    inst✝³ : (F.mapShortComplex.obj S₂).HasHomology
    inst✝² : F.PreservesRightHomologyOf S₁
    inst✝¹ : F.PreservesRightHomologyOf S₂
    inst✝ : CategoryTheory.ShortComplex.QuasiIso φ
    γ : CategoryTheory.ShortComplex.RightHomologyMapData φ S₁.rightHomologyData S₂ …
    this : CategoryTheory.IsIso γ.φH
    ⊢ CategoryTheory.IsIso (F.map γ.φH)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma quasiIso_map_iff_of_preservesRightHomology
    [F.PreservesRightHomologyOf S₁] [F.PreservesRightHomologyOf S₂]
    [F.ReflectsIsomorphisms] :
    QuasiIso (F.mapShortComplex.map φ) ↔ QuasiIso φ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms D
    F : CategoryTheory.Functor C D
    inst✝⁷ : F.PreservesZeroMorphisms
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝⁶ : S₁.HasHomology
    inst✝⁵ : S₂.HasHomology
    inst✝⁴ : (F.mapShortComplex.obj S₁).HasHomology
    inst✝³ : (F.mapShortComplex.obj S₂).HasHomology
    inst✝² : F.PreservesRightHomologyOf S₁
    inst✝¹ : F.PreservesRightHomologyOf S₂
    inst✝ : F.ReflectsIsomorphisms
    ⊢ Iff (CategoryTheory.ShortComplex.QuasiIso (F.mapShortComplex.map φ)) (Catego …
  -/
  have γ : RightHomologyMapData φ S₁.rightHomologyData S₂.rightHomologyData := default
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms D
    F : CategoryTheory.Functor C D
    inst✝⁷ : F.PreservesZeroMorphisms
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝⁶ : S₁.HasHomology
    inst✝⁵ : S₂.HasHomology
    inst✝⁴ : (F.mapShortComplex.obj S₁).HasHomology
    inst✝³ : (F.mapShortComplex.obj S₂).HasHomology
    inst✝² : F.PreservesRightHomologyOf S₁
    inst✝¹ : F.PreservesRightHomologyOf S₂
    inst✝ : F.ReflectsIsomorphisms
    γ : CategoryTheory.ShortComplex.RightHomologyMapData φ S₁.rightHomologyData S₂ …
    ⊢ Iff (CategoryTheory.ShortComplex.QuasiIso (F.mapShortComplex.map φ)) (Catego …
  -/
  rw [γ.quasiIso_iff, (γ.map F).quasiIso_iff, RightHomologyMapData.map_φH]
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms D
    F : CategoryTheory.Functor C D
    inst✝⁷ : F.PreservesZeroMorphisms
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝⁶ : S₁.HasHomology
    inst✝⁵ : S₂.HasHomology
    inst✝⁴ : (F.mapShortComplex.obj S₁).HasHomology
    inst✝³ : (F.mapShortComplex.obj S₂).HasHomology
    inst✝² : F.PreservesRightHomologyOf S₁
    inst✝¹ : F.PreservesRightHomologyOf S₂
    inst✝ : F.ReflectsIsomorphisms
    γ : CategoryTheory.ShortComplex.RightHomologyMapData φ S₁.rightHomologyData S₂ …
    ⊢ Iff (CategoryTheory.IsIso (F.map γ.φH)) (CategoryTheory.IsIso γ.φH)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝¹¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹⁰ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms D
      F : CategoryTheory.Functor C D
      inst✝⁷ : F.PreservesZeroMorphisms
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      inst✝⁶ : S₁.HasHomology
      inst✝⁵ : S₂.HasHomology
      inst✝⁴ : (F.mapShortComplex.obj S₁).HasHomology
      inst✝³ : (F.mapShortComplex.obj S₂).HasHomology
      inst✝² : F.PreservesRightHomologyOf S₁
      inst✝¹ : F.PreservesRightHomologyOf S₂
      inst✝ : F.ReflectsIsomorphisms
      γ : CategoryTheory.ShortComplex.RightHomologyMapData φ S₁.rightHomologyData S₂ …
      ⊢ CategoryTheory.IsIso (F.map γ.φH) → CategoryTheory.IsIso γ.φH
    -/
  · intro
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝¹¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹⁰ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms D
      F : CategoryTheory.Functor C D
      inst✝⁷ : F.PreservesZeroMorphisms
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      inst✝⁶ : S₁.HasHomology
      inst✝⁵ : S₂.HasHomology
      inst✝⁴ : (F.mapShortComplex.obj S₁).HasHomology
      inst✝³ : (F.mapShortComplex.obj S₂).HasHomology
      inst✝² : F.PreservesRightHomologyOf S₁
      inst✝¹ : F.PreservesRightHomologyOf S₂
      inst✝ : F.ReflectsIsomorphisms
      γ : CategoryTheory.ShortComplex.RightHomologyMapData φ S₁.rightHomologyData S₂ …
      a✝ : CategoryTheory.IsIso (F.map γ.φH)
      ⊢ CategoryTheory.IsIso γ.φH
    -/
    exact isIso_of_reflects_iso _ F
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝¹¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹⁰ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms D
      F : CategoryTheory.Functor C D
      inst✝⁷ : F.PreservesZeroMorphisms
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      inst✝⁶ : S₁.HasHomology
      inst✝⁵ : S₂.HasHomology
      inst✝⁴ : (F.mapShortComplex.obj S₁).HasHomology
      inst✝³ : (F.mapShortComplex.obj S₂).HasHomology
      inst✝² : F.PreservesRightHomologyOf S₁
      inst✝¹ : F.PreservesRightHomologyOf S₂
      inst✝ : F.ReflectsIsomorphisms
      γ : CategoryTheory.ShortComplex.RightHomologyMapData φ S₁.rightHomologyData S₂ …
      ⊢ CategoryTheory.IsIso γ.φH → CategoryTheory.IsIso (F.map γ.φH)
    -/
  · intro
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝¹¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹⁰ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms D
      F : CategoryTheory.Functor C D
      inst✝⁷ : F.PreservesZeroMorphisms
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      inst✝⁶ : S₁.HasHomology
      inst✝⁵ : S₂.HasHomology
      inst✝⁴ : (F.mapShortComplex.obj S₁).HasHomology
      inst✝³ : (F.mapShortComplex.obj S₂).HasHomology
      inst✝² : F.PreservesRightHomologyOf S₁
      inst✝¹ : F.PreservesRightHomologyOf S₂
      inst✝ : F.ReflectsIsomorphisms
      γ : CategoryTheory.ShortComplex.RightHomologyMapData φ S₁.rightHomologyData S₂ …
      a✝ : CategoryTheory.IsIso γ.φH
      ⊢ CategoryTheory.IsIso (F.map γ.φH)
    -/
    infer_instance
    /-
      🎉 no goals
    -/


/-- If a short complex `S` is such that `S.f = 0` and that the kernel of `S.g` is preserved
by a functor `F`, then `F` preserves the left homology of `S`. -/
lemma preservesLeftHomology_of_zero_f (hf : S.f = 0)
    [PreservesLimit (parallelPair S.g 0) F] :
    F.PreservesLeftHomologyOf S := ⟨fun h =>
            /-
              C : Type u_1
              D : Type u_2
              inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
              inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
              inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
              inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
              F : CategoryTheory.Functor C D
              inst✝¹ : F.PreservesZeroMorphisms
              S : CategoryTheory.ShortComplex C
              hf : Eq S.f 0
              inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
              h : S.LeftHomologyData
              ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPair S.g …
            -/
  { g := by infer_instance
            /-
              🎉 no goals
            -/
    f' := Limits.preservesCokernel_zero' _ _
          /-
            C : Type u_1
            D : Type u_2
            inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
            inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
            inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
            inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
            F : CategoryTheory.Functor C D
            inst✝¹ : F.PreservesZeroMorphisms
            S : CategoryTheory.ShortComplex C
            hf : Eq S.f 0
            inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
            h : S.LeftHomologyData
            ⊢ Eq h.f' 0
          -/
      (by rw [← cancel_mono h.i, h.f'_i, zero_comp, hf]) }⟩
          /-
            🎉 no goals
          -/


/-- If a short complex `S` is such that `S.g = 0` and that the cokernel of `S.f` is preserved
by a functor `F`, then `F` preserves the right homology of `S`. -/
lemma preservesRightHomology_of_zero_g (hg : S.g = 0)
    [PreservesColimit (parallelPair S.f 0) F] :
    F.PreservesRightHomologyOf S := ⟨fun h =>
            /-
              C : Type u_1
              D : Type u_2
              inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
              inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
              inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
              inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
              F : CategoryTheory.Functor C D
              inst✝¹ : F.PreservesZeroMorphisms
              S : CategoryTheory.ShortComplex C
              hg : Eq S.g 0
              inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
              h : S.RightHomologyData
              ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelPair S …
            -/
  { f := by infer_instance
            /-
              🎉 no goals
            -/
    g' := Limits.preservesKernel_zero' _ _
          /-
            C : Type u_1
            D : Type u_2
            inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
            inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
            inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
            inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
            F : CategoryTheory.Functor C D
            inst✝¹ : F.PreservesZeroMorphisms
            S : CategoryTheory.ShortComplex C
            hg : Eq S.g 0
            inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
            h : S.RightHomologyData
            ⊢ Eq h.g' 0
          -/
      (by rw [← cancel_epi h.p, h.p_g', comp_zero, hg]) }⟩
          /-
            🎉 no goals
          -/


/-- If a short complex `S` is such that `S.g = 0` and that the cokernel of `S.f` is preserved
by a functor `F`, then `F` preserves the left homology of `S`. -/
lemma preservesLeftHomology_of_zero_g (hg : S.g = 0)
    [PreservesColimit (parallelPair S.f 0) F] :
    F.PreservesLeftHomologyOf S := ⟨fun h =>
  { g := by
      /-
        C : Type u_1
        D : Type u_2
        inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.PreservesZeroMorphisms
        S : CategoryTheory.ShortComplex C
        hg : Eq S.g 0
        inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
        h : S.LeftHomologyData
        ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPair S.g …
      -/
      rw [hg]
      /-
        C : Type u_1
        D : Type u_2
        inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.PreservesZeroMorphisms
        S : CategoryTheory.ShortComplex C
        hg : Eq S.g 0
        inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
        h : S.LeftHomologyData
        ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPair 0 0 …
      -/
      infer_instance
      /-
        🎉 no goals
      -/
    f' := by
      /-
        C : Type u_1
        D : Type u_2
        inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.PreservesZeroMorphisms
        S : CategoryTheory.ShortComplex C
        hg : Eq S.g 0
        inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
        h : S.LeftHomologyData
        ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelPair h …
      -/
      have := h.isIso_i hg
      let e : parallelPair h.f' 0 ≅ parallelPair S.f 0 :=
        parallelPair.ext (Iso.refl _) (asIso h.i) (by aesop_cat) (by aesop_cat)
      /-
        C : Type u_1
        D : Type u_2
        inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.PreservesZeroMorphisms
        S : CategoryTheory.ShortComplex C
        hg : Eq S.g 0
        inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
        h : S.LeftHomologyData
        this : CategoryTheory.IsIso h.i
        e : CategoryTheory.Iso (CategoryTheory.Limits.parallelPair h.f' 0) (CategoryTh …
        ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelPair h …
      -/
      exact Limits.preservesColimit_of_iso_diagram F e.symm}⟩
      /-
        🎉 no goals
      -/


/-- If a short complex `S` is such that `S.f = 0` and that the kernel of `S.g` is preserved
by a functor `F`, then `F` preserves the right homology of `S`. -/
lemma preservesRightHomology_of_zero_f (hf : S.f = 0)
    [PreservesLimit (parallelPair S.g 0) F] :
    F.PreservesRightHomologyOf S := ⟨fun h =>
  { f := by
      /-
        C : Type u_1
        D : Type u_2
        inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.PreservesZeroMorphisms
        S : CategoryTheory.ShortComplex C
        hf : Eq S.f 0
        inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
        h : S.RightHomologyData
        ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelPair S …
      -/
      rw [hf]
      /-
        C : Type u_1
        D : Type u_2
        inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.PreservesZeroMorphisms
        S : CategoryTheory.ShortComplex C
        hf : Eq S.f 0
        inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
        h : S.RightHomologyData
        ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelPair 0 …
      -/
      infer_instance
      /-
        🎉 no goals
      -/
    g' := by
      /-
        C : Type u_1
        D : Type u_2
        inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.PreservesZeroMorphisms
        S : CategoryTheory.ShortComplex C
        hf : Eq S.f 0
        inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
        h : S.RightHomologyData
        ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPair h.g …
      -/
      have := h.isIso_p hf
      let e : parallelPair S.g 0 ≅ parallelPair h.g' 0 :=
        parallelPair.ext (asIso h.p) (Iso.refl _) (by aesop_cat) (by aesop_cat)
      /-
        C : Type u_1
        D : Type u_2
        inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.PreservesZeroMorphisms
        S : CategoryTheory.ShortComplex C
        hf : Eq S.f 0
        inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
        h : S.RightHomologyData
        this : CategoryTheory.IsIso h.p
        e : CategoryTheory.Iso (CategoryTheory.Limits.parallelPair S.g 0) (CategoryThe …
        ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPair h.g …
      -/
      exact Limits.preservesLimit_of_iso_diagram F e }⟩
      /-
        🎉 no goals
      -/


lemma NatTrans.app_homology {F G : C ⥤ D} (τ : F ⟶ G)
    (S : ShortComplex C) [S.HasHomology] [F.PreservesZeroMorphisms] [G.PreservesZeroMorphisms]
    [F.PreservesLeftHomologyOf S] [G.PreservesLeftHomologyOf S] [F.PreservesRightHomologyOf S]
    [G.PreservesRightHomologyOf S] :
    τ.app S.homology = (S.mapHomologyIso F).inv ≫
      ShortComplex.homologyMap (S.mapNatTrans τ) ≫ (S.mapHomologyIso G).hom := by
  rw [ShortComplex.homologyMap_mapNatTrans, assoc, assoc, Iso.inv_hom_id,
    comp_id, Iso.inv_hom_id_assoc]



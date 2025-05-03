/-- The pseudofunctor from `LocallyDiscrete CommRingCatᵒᵖ` to `Cat` which sends
a commutative ring `R` to its category of modules. The functoriality is given by
the restriction of scalars. -/
@[simps! obj map mapId mapComp]
noncomputable def CommRingCat.moduleCatRestrictScalarsPseudofunctor :
    Pseudofunctor (LocallyDiscrete CommRingCat.{u}ᵒᵖ) Cat :=
  /-
    ⊢ ∀ {b₀ b₁ b₂ b₃ : Opposite CommRingCat} (f : Quiver.Hom b₀ b₁) (g : Quiver.Ho …
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  LocallyDiscrete.mkPseudofunctor
  /-
    🎉 no goals
  -/
    (fun R ↦ Cat.of (ModuleCat.{v} R.unop))
    (fun f ↦ restrictScalars f.unop.hom)
    (fun R ↦ restrictScalarsId R.unop)
    (fun f g ↦ restrictScalarsComp g.unop.hom f.unop.hom)


/-- The pseudofunctor from `LocallyDiscrete RingCatᵒᵖ` to `Cat` which sends a ring `R`
to its category of modules. The functoriality is given by the restriction of scalars. -/
@[simps! obj map mapId mapComp]
noncomputable def RingCat.moduleCatRestrictScalarsPseudofunctor :
    Pseudofunctor (LocallyDiscrete RingCat.{u}ᵒᵖ) Cat :=
  /-
    ⊢ ∀ {b₀ b₁ b₂ b₃ : Opposite RingCat} (f : Quiver.Hom b₀ b₁) (g : Quiver.Hom b₁ …
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  LocallyDiscrete.mkPseudofunctor
  /-
    🎉 no goals
  -/
    (fun R ↦ Cat.of (ModuleCat.{v} R.unop))
    (fun f ↦ restrictScalars f.unop.hom)
    (fun R ↦ restrictScalarsId R.unop)
    (fun f g ↦ restrictScalarsComp g.unop.hom f.unop.hom)


/-- The pseudofunctor from `LocallyDiscrete CommRingCat` to `Cat` which sends
a commutative ring `R` to its category of modules. The functoriality is given by
the extension of scalars. -/
@[simps! obj map mapId mapComp]
noncomputable def CommRingCat.moduleCatExtendScalarsPseudofunctor :
    Pseudofunctor (LocallyDiscrete CommRingCat.{u}) Cat :=
  LocallyDiscrete.mkPseudofunctor
    (fun R ↦ Cat.of (ModuleCat.{u} R))
    (fun f ↦ extendScalars f.hom)
    (fun R ↦ extendScalarsId R)
    (fun f g ↦ extendScalarsComp f.hom g.hom)
    (fun _ _ _ ↦ extendScalars_assoc' _ _ _)
    (fun _ ↦ extendScalars_id_comp _)
    (fun _ ↦ extendScalars_comp_id _)


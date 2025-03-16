/-- The free condensed abelian group on a finite set. -/
abbrev finFree : FintypeCat.{u} ⥤ CondensedMod.{u} R :=
  FintypeCat.toProfinite ⋙ profiniteToCondensed ⋙ free R


/-- The free condensed abelian group on a profinite space. -/
abbrev profiniteFree : Profinite.{u} ⥤ CondensedMod.{u} R :=
  profiniteToCondensed ⋙ free R


/-- The functor sending a profinite space `S` to the condensed abelian group `R[S]^\solid`. -/
def profiniteSolid : Profinite.{u} ⥤ CondensedMod.{u} R :=
  Functor.rightKanExtension FintypeCat.toProfinite (finFree R)


/-- The natural transformation `FintypeCat.toProfinite ⋙ profiniteSolid R ⟶ finFree R`
which is part of the assertion that `profiniteSolid R` is the (pointwise) right
Kan extension of `finFree R` along `FintypeCat.toProfinite`. -/
def profiniteSolidCounit : FintypeCat.toProfinite ⋙ profiniteSolid R ⟶ finFree R :=
  Functor.rightKanExtensionCounit FintypeCat.toProfinite (finFree R)


instance : (profiniteSolid R).IsRightKanExtension (profiniteSolidCounit R) := by
  /-
    R : Type (u + 1)
    inst✝ : Ring R
    ⊢ (Condensed.profiniteSolid R).IsRightKanExtension (Condensed.profiniteSolidCo …
  -/
  dsimp only [profiniteSolidCounit, profiniteSolid]
  /-
    R : Type (u + 1)
    inst✝ : Ring R
    ⊢ (FintypeCat.toProfinite.rightKanExtension (Condensed.finFree R)).IsRightKanE …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The functor `Profinite.{u} ⥤ CondensedMod.{u} R` is a pointwise
right Kan extension of `finFree R : FintypeCat.{u} ⥤ CondensedMod.{u} R`
along `FintypeCat.toProfinite`. -/
def profiniteSolidIsPointwiseRightKanExtension :
    (Functor.RightExtension.mk _ (profiniteSolidCounit R)).IsPointwiseRightKanExtension :=
  Functor.isPointwiseRightKanExtensionOfIsRightKanExtension _ _


/-- The natural transformation `R[S] ⟶ R[S]^\solid`. -/
def profiniteSolidification : profiniteFree R ⟶ profiniteSolid.{u} R :=
  (profiniteSolid R).liftOfIsRightKanExtension (profiniteSolidCounit R) _ (𝟙 _)


/-- The predicate on condensed abelian groups describing the property of being solid. -/
class CondensedMod.IsSolid (A : CondensedMod.{u} R) : Prop where
  isIso_solidification_map : ∀ X : Profinite.{u}, IsIso ((yoneda.obj A).map
    ((profiniteSolidification R).app X).op)


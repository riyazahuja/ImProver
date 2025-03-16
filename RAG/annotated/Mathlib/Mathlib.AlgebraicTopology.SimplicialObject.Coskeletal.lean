/-- The identity natural transformation exhibits a simplicial set as a right extension of its
restriction along `(Truncated.inclusion n).op`.-/
@[simps!]
def rightExtensionInclusion :
    RightExtension (Truncated.inclusion n).op
      ((Truncated.inclusion n).op ⋙ X) := RightExtension.mk _ (𝟙 _)


/-- A simplicial object `X` is `n`-coskeletal when it is the right Kan extension of its restriction
along `(Truncated.inclusion n).op` via the identity natural transformation. -/
@[mk_iff]
class IsCoskeletal : Prop where
  isRightKanExtension : IsRightKanExtension X (𝟙 ((Truncated.inclusion n).op ⋙ X))


/-- If `X` is `n`-coskeletal, then `Truncated.rightExtensionInclusion X n` is a terminal object in
the category `RightExtension (Truncated.inclusion n).op (Truncated.inclusion.op ⋙ X)`. -/
noncomputable def IsCoskeletal.isUniversalOfIsRightKanExtension [X.IsCoskeletal n] :
    (rightExtensionInclusion X n).IsUniversal := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    inst✝¹ : ∀ (F : CategoryTheory.Functor (Opposite (SimplexCategory.Truncated n) …
    inst✝ : X.IsCoskeletal n
    ⊢ CategoryTheory.CostructuredArrow.IsUniversal (CategoryTheory.SimplicialObjec …
  -/
  apply Functor.isUniversalOfIsRightKanExtension
  /-
    🎉 no goals
  -/


theorem isCoskeletal_iff_isIso : X.IsCoskeletal n ↔ IsIso ((coskAdj n).unit.app X) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    inst✝ : ∀ (F : CategoryTheory.Functor (Opposite (SimplexCategory.Truncated n)) …
    ⊢ Iff (X.IsCoskeletal n) (CategoryTheory.IsIso ((CategoryTheory.SimplicialObje …
  -/
  rw [isCoskeletal_iff]
  exact isRightKanExtension_iff_isIso ((coskAdj n).unit.app X)
    ((coskAdj n).counit.app _) (𝟙 _) ((coskAdj n).left_triangle_components X)


instance [X.IsCoskeletal n] : IsIso ((coskAdj n).unit.app X) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    inst✝¹ : ∀ (F : CategoryTheory.Functor (Opposite (SimplexCategory.Truncated n) …
    inst✝ : X.IsCoskeletal n
    ⊢ CategoryTheory.IsIso ((CategoryTheory.SimplicialObject.coskAdj n).unit.app X)
  -/
  rw [← isCoskeletal_iff_isIso]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    inst✝¹ : ∀ (F : CategoryTheory.Functor (Opposite (SimplexCategory.Truncated n) …
    inst✝ : X.IsCoskeletal n
    ⊢ X.IsCoskeletal n
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The canonical isomorphism `X ≅ (cosk n).obj X` defined when `X` is coskeletal and the
`n`-coskeleton functor exists.-/
@[simps! hom]
noncomputable def isoCoskOfIsCoskeletal [X.IsCoskeletal n] : X ≅ (cosk n).obj X :=
  asIso ((coskAdj n).unit.app X)



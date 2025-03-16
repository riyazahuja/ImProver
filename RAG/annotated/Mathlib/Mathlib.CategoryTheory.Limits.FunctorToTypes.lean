theorem jointly_surjective (k : K) {t : Cocone F} (h : IsColimit t) (x : t.pt.obj k)
    [∀ k, HasColimit (F.flip.obj k)] : ∃ j y, x = (t.ι.app j).app k y := by
  /-
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    F : CategoryTheory.Functor J (CategoryTheory.Functor K (Type w))
    k : K
    t : CategoryTheory.Limits.Cocone F
    h : CategoryTheory.Limits.IsColimit t
    x : t.pt.obj k
    inst✝ : ∀ (k : K), CategoryTheory.Limits.HasColimit (F.flip.obj k)
    ⊢ Exists fun j => Exists fun y => Eq x ((t.ι.app j).app k y)
  -/
  let hev := isColimitOfPreserves ((evaluation _ _).obj k) h
  /-
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    F : CategoryTheory.Functor J (CategoryTheory.Functor K (Type w))
    k : K
    t : CategoryTheory.Limits.Cocone F
    h : CategoryTheory.Limits.IsColimit t
    x : t.pt.obj k
    inst✝ : ∀ (k : K), CategoryTheory.Limits.HasColimit (F.flip.obj k)
    hev : CategoryTheory.Limits.IsColimit (((CategoryTheory.evaluation K (Type w)) …
    ⊢ Exists fun j => Exists fun y => Eq x ((t.ι.app j).app k y)
  -/
  obtain ⟨j, y, rfl⟩ := Types.jointly_surjective _ hev x
  /-
    case intro.intro
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    F : CategoryTheory.Functor J (CategoryTheory.Functor K (Type w))
    k : K
    t : CategoryTheory.Limits.Cocone F
    h : CategoryTheory.Limits.IsColimit t
    inst✝ : ∀ (k : K), CategoryTheory.Limits.HasColimit (F.flip.obj k)
    hev : CategoryTheory.Limits.IsColimit (((CategoryTheory.evaluation K (Type w)) …
    j : J
    y : (F.comp ((CategoryTheory.evaluation K (Type w)).obj k)).obj j
    ⊢ Exists fun j_1 => Exists fun y_1 => Eq ((((CategoryTheory.evaluation K (Type …
  -/
  exact ⟨j, y, by simp⟩
  /-
    🎉 no goals
  -/


theorem jointly_surjective' [∀ k, HasColimit (F.flip.obj k)] (k : K) (x : (colimit F).obj k) :
    ∃ j y, x = (colimit.ι F j).app k y :=
  jointly_surjective _ _ (colimit.isColimit _) x


theorem colimit.map_ι_apply [HasColimit F] (j : J) {k k' : K} {f : k ⟶ k'} {x} :
    (colimit F).map f ((colimit.ι F j).app _ x) = (colimit.ι F j).app _ ((F.obj j).map f x) :=
  congrFun ((colimit.ι F j).naturality _).symm _



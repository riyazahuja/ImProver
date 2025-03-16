/-- The Zariski pretopology on the category of schemes. -/
def zariskiPretopology : Pretopology (Scheme.{u}) :=
  pretopology @IsOpenImmersion


/-- The Zariski topology on the category of schemes. -/
abbrev zariskiTopology : GrothendieckTopology (Scheme.{u}) :=
  zariskiPretopology.toGrothendieck


instance subcanonical_zariskiTopology : zariskiTopology.Subcanonical := by
  /-
    ⊢ AlgebraicGeometry.Scheme.zariskiTopology.Subcanonical
  -/
  apply GrothendieckTopology.Subcanonical.of_isSheaf_yoneda_obj
  /-
    case h
    ⊢ ∀ (X : AlgebraicGeometry.Scheme), CategoryTheory.Presieve.IsSheaf AlgebraicG …
  -/
  intro X
  /-
    case h
    X : AlgebraicGeometry.Scheme
    ⊢ CategoryTheory.Presieve.IsSheaf AlgebraicGeometry.Scheme.zariskiTopology (Ca …
  -/
  rw [Presieve.isSheaf_pretopology]
  /-
    case h
    X : AlgebraicGeometry.Scheme
    ⊢ ∀ {X_1 : AlgebraicGeometry.Scheme} (R : CategoryTheory.Presieve X_1), Member …
  -/
  rintro Y S ⟨𝓤,rfl⟩ x hx
  let e : Y ⟶ X := 𝓤.glueMorphisms (fun j => x (𝓤.map _) (.mk _)) <| by
    intro i j
    apply hx
    exact Limits.pullback.condition
  /-
    case h.intro
    X Y : AlgebraicGeometry.Scheme
    𝓤 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) Y
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj X) (Ca …
    hx : x.Compatible
    e : Quiver.Hom Y X := 𝓤.glueMorphisms (fun j => x (𝓤.map j) ⋯) ⋯
    ⊢ ExistsUnique fun t => x.IsAmalgamation t
  -/
  refine ⟨e, ?_, ?_⟩
    /-
      case h.intro.refine_1
      X Y : AlgebraicGeometry.Scheme
      𝓤 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) Y
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj X) (Ca …
      hx : x.Compatible
      e : Quiver.Hom Y X := 𝓤.glueMorphisms (fun j => x (𝓤.map j) ⋯) ⋯
      ⊢ (fun t => x.IsAmalgamation t) e
    -/
  · rintro Z e ⟨j⟩
    /-
      case h.intro.refine_1.mk
      X Y✝ : AlgebraicGeometry.Scheme
      𝓤 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) Y✝
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj X) (Ca …
      hx : x.Compatible
      e : Quiver.Hom Y✝ X := 𝓤.glueMorphisms (fun j => x (𝓤.map j) ⋯) ⋯
      Y : AlgebraicGeometry.Scheme
      j : 𝓤.J
      ⊢ Eq ((CategoryTheory.yoneda.obj X).map (𝓤.map j).op e) (x (𝓤.map j) ⋯)
    -/
    dsimp [e]
    /-
      case h.intro.refine_1.mk
      X Y✝ : AlgebraicGeometry.Scheme
      𝓤 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) Y✝
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj X) (Ca …
      hx : x.Compatible
      e : Quiver.Hom Y✝ X := 𝓤.glueMorphisms (fun j => x (𝓤.map j) ⋯) ⋯
      Y : AlgebraicGeometry.Scheme
      j : 𝓤.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (𝓤.map j) (𝓤.glueMorphisms (fun j =>  …
    -/
    rw [𝓤.ι_glueMorphisms]
    /-
      🎉 no goals
    -/
    /-
      case h.intro.refine_2
      X Y : AlgebraicGeometry.Scheme
      𝓤 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) Y
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj X) (Ca …
      hx : x.Compatible
      e : Quiver.Hom Y X := 𝓤.glueMorphisms (fun j => x (𝓤.map j) ⋯) ⋯
      ⊢ ∀ (y : (CategoryTheory.yoneda.obj X).obj { unop := Y }), (fun t => x.IsAmalg …
    -/
  · intro e' h
    /-
      case h.intro.refine_2
      X Y : AlgebraicGeometry.Scheme
      𝓤 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) Y
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj X) (Ca …
      hx : x.Compatible
      e : Quiver.Hom Y X := 𝓤.glueMorphisms (fun j => x (𝓤.map j) ⋯) ⋯
      e' : (CategoryTheory.yoneda.obj X).obj { unop := Y }
      h : x.IsAmalgamation e'
      ⊢ Eq e' e
    -/
    apply 𝓤.hom_ext
    /-
      case h.intro.refine_2.h
      X Y : AlgebraicGeometry.Scheme
      𝓤 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) Y
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj X) (Ca …
      hx : x.Compatible
      e : Quiver.Hom Y X := 𝓤.glueMorphisms (fun j => x (𝓤.map j) ⋯) ⋯
      e' : (CategoryTheory.yoneda.obj X).obj { unop := Y }
      h : x.IsAmalgamation e'
      ⊢ ∀ (x : 𝓤.J), Eq (CategoryTheory.CategoryStruct.comp (𝓤.map x) e') (CategoryT …
    -/
    intro j
    /-
      case h.intro.refine_2.h
      X Y : AlgebraicGeometry.Scheme
      𝓤 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) Y
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj X) (Ca …
      hx : x.Compatible
      e : Quiver.Hom Y X := 𝓤.glueMorphisms (fun j => x (𝓤.map j) ⋯) ⋯
      e' : (CategoryTheory.yoneda.obj X).obj { unop := Y }
      h : x.IsAmalgamation e'
      j : 𝓤.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (𝓤.map j) e') (CategoryTheory.Categor …
    -/
    rw [𝓤.ι_glueMorphisms]
    /-
      case h.intro.refine_2.h
      X Y : AlgebraicGeometry.Scheme
      𝓤 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) Y
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj X) (Ca …
      hx : x.Compatible
      e : Quiver.Hom Y X := 𝓤.glueMorphisms (fun j => x (𝓤.map j) ⋯) ⋯
      e' : (CategoryTheory.yoneda.obj X).obj { unop := Y }
      h : x.IsAmalgamation e'
      j : 𝓤.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (𝓤.map j) e') (x (𝓤.map j) ⋯)
    -/
    exact h (𝓤.map j) (.mk j)
    /-
      🎉 no goals
    -/



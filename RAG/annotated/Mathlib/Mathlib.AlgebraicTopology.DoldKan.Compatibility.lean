/-- A basic equivalence `A ≅ B'` obtained by composing `eA : A ≅ A'` and `e' : A' ≅ B'`. -/
@[simps! functor inverse unitIso_hom_app]
def equivalence₀ : A ≌ B' :=
  eA.trans e'


/-- An intermediate equivalence `A ≅ B'` whose functor is `F` and whose inverse is
`e'.inverse ⋙ eA.inverse`. -/
@[simps! functor]
def equivalence₁ : A ≌ B' := (equivalence₀ eA e').changeFunctor hF


theorem equivalence₁_inverse : (equivalence₁ hF).inverse = e'.inverse ⋙ eA.inverse :=
  rfl


/-- The counit isomorphism of the equivalence `equivalence₁` between `A` and `B'`. -/
@[simps!]
def equivalence₁CounitIso : (e'.inverse ⋙ eA.inverse) ⋙ F ≅ 𝟭 B' :=
  calc
    (e'.inverse ⋙ eA.inverse) ⋙ F ≅ (e'.inverse ⋙ eA.inverse) ⋙ eA.functor ⋙ e'.functor :=
      isoWhiskerLeft _ hF.symm
    _ ≅ e'.inverse ⋙ (eA.inverse ⋙ eA.functor) ⋙ e'.functor := Iso.refl _
    _ ≅ e'.inverse ⋙ 𝟭 _ ⋙ e'.functor := isoWhiskerLeft _ (isoWhiskerRight eA.counitIso _)
    _ ≅ e'.inverse ⋙ e'.functor := Iso.refl _
    _ ≅ 𝟭 B' := e'.counitIso


theorem equivalence₁CounitIso_eq : (equivalence₁ hF).counitIso = equivalence₁CounitIso hF := by
  /-
    A : Type u_1
    A' : Type u_2
    B' : Type u_4
    inst✝² : CategoryTheory.Category.{u_6, u_1} A
    inst✝¹ : CategoryTheory.Category.{u_7, u_2} A'
    inst✝ : CategoryTheory.Category.{u_5, u_4} B'
    eA : CategoryTheory.Equivalence A A'
    e' : CategoryTheory.Equivalence A' B'
    F : CategoryTheory.Functor A B'
    hF : CategoryTheory.Iso (eA.functor.comp e'.functor) F
    ⊢ Eq (AlgebraicTopology.DoldKan.Compatibility.equivalence₁ hF).counitIso (Alge …
  -/
  ext Y
  /-
    case w.w.h
    A : Type u_1
    A' : Type u_2
    B' : Type u_4
    inst✝² : CategoryTheory.Category.{u_6, u_1} A
    inst✝¹ : CategoryTheory.Category.{u_7, u_2} A'
    inst✝ : CategoryTheory.Category.{u_5, u_4} B'
    eA : CategoryTheory.Equivalence A A'
    e' : CategoryTheory.Equivalence A' B'
    F : CategoryTheory.Functor A B'
    hF : CategoryTheory.Iso (eA.functor.comp e'.functor) F
    Y : B'
    ⊢ Eq ((AlgebraicTopology.DoldKan.Compatibility.equivalence₁ hF).counitIso.hom. …
  -/
  simp [equivalence₁, equivalence₀]
  /-
    🎉 no goals
  -/


/-- The unit isomorphism of the equivalence `equivalence₁` between `A` and `B'`. -/
@[simps!]
def equivalence₁UnitIso : 𝟭 A ≅ F ⋙ e'.inverse ⋙ eA.inverse :=
  calc
    𝟭 A ≅ eA.functor ⋙ eA.inverse := eA.unitIso
    _ ≅ eA.functor ⋙ 𝟭 A' ⋙ eA.inverse := Iso.refl _
    _ ≅ eA.functor ⋙ (e'.functor ⋙ e'.inverse) ⋙ eA.inverse :=
      isoWhiskerLeft _ (isoWhiskerRight e'.unitIso _)
    _ ≅ (eA.functor ⋙ e'.functor) ⋙ e'.inverse ⋙ eA.inverse := Iso.refl _
    _ ≅ F ⋙ e'.inverse ⋙ eA.inverse := isoWhiskerRight hF _


theorem equivalence₁UnitIso_eq : (equivalence₁ hF).unitIso = equivalence₁UnitIso hF := by
  /-
    A : Type u_1
    A' : Type u_2
    B' : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_1} A
    inst✝¹ : CategoryTheory.Category.{u_7, u_2} A'
    inst✝ : CategoryTheory.Category.{u_6, u_4} B'
    eA : CategoryTheory.Equivalence A A'
    e' : CategoryTheory.Equivalence A' B'
    F : CategoryTheory.Functor A B'
    hF : CategoryTheory.Iso (eA.functor.comp e'.functor) F
    ⊢ Eq (AlgebraicTopology.DoldKan.Compatibility.equivalence₁ hF).unitIso (Algebr …
  -/
  ext X
  /-
    case w.w.h
    A : Type u_1
    A' : Type u_2
    B' : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_1} A
    inst✝¹ : CategoryTheory.Category.{u_7, u_2} A'
    inst✝ : CategoryTheory.Category.{u_6, u_4} B'
    eA : CategoryTheory.Equivalence A A'
    e' : CategoryTheory.Equivalence A' B'
    F : CategoryTheory.Functor A B'
    hF : CategoryTheory.Iso (eA.functor.comp e'.functor) F
    X : A
    ⊢ Eq ((AlgebraicTopology.DoldKan.Compatibility.equivalence₁ hF).unitIso.hom.ap …
  -/
  simp [equivalence₁]
  /-
    🎉 no goals
  -/


/-- An intermediate equivalence `A ≅ B` obtained as the composition of `equivalence₁` and
the inverse of `eB : B ≌ B'`. -/
@[simps! functor]
def equivalence₂ : A ≌ B :=
  (equivalence₁ hF).trans eB.symm


theorem equivalence₂_inverse :
    (equivalence₂ eB hF).inverse = eB.functor ⋙ e'.inverse ⋙ eA.inverse :=
  rfl


/-- The counit isomorphism of the equivalence `equivalence₂` between `A` and `B`. -/
@[simps!]
def equivalence₂CounitIso : (eB.functor ⋙ e'.inverse ⋙ eA.inverse) ⋙ F ⋙ eB.inverse ≅ 𝟭 B :=
  calc
    (eB.functor ⋙ e'.inverse ⋙ eA.inverse) ⋙ F ⋙ eB.inverse ≅
        eB.functor ⋙ (e'.inverse ⋙ eA.inverse ⋙ F) ⋙ eB.inverse :=
      Iso.refl _
    _ ≅ eB.functor ⋙ 𝟭 _ ⋙ eB.inverse :=
      isoWhiskerLeft _ (isoWhiskerRight (equivalence₁CounitIso hF) _)
    _ ≅ eB.functor ⋙ eB.inverse := Iso.refl _
    _ ≅ 𝟭 B := eB.unitIso.symm


theorem equivalence₂CounitIso_eq :
    (equivalence₂ eB hF).counitIso = equivalence₂CounitIso eB hF := by
  /-
    A : Type u_1
    A' : Type u_2
    B : Type u_3
    B' : Type u_4
    inst✝³ : CategoryTheory.Category.{u_6, u_1} A
    inst✝² : CategoryTheory.Category.{u_7, u_2} A'
    inst✝¹ : CategoryTheory.Category.{u_5, u_3} B
    inst✝ : CategoryTheory.Category.{u_8, u_4} B'
    eA : CategoryTheory.Equivalence A A'
    eB : CategoryTheory.Equivalence B B'
    e' : CategoryTheory.Equivalence A' B'
    F : CategoryTheory.Functor A B'
    hF : CategoryTheory.Iso (eA.functor.comp e'.functor) F
    ⊢ Eq (AlgebraicTopology.DoldKan.Compatibility.equivalence₂ eB hF).counitIso (A …
  -/
  ext Y'
  /-
    case w.w.h
    A : Type u_1
    A' : Type u_2
    B : Type u_3
    B' : Type u_4
    inst✝³ : CategoryTheory.Category.{u_6, u_1} A
    inst✝² : CategoryTheory.Category.{u_7, u_2} A'
    inst✝¹ : CategoryTheory.Category.{u_5, u_3} B
    inst✝ : CategoryTheory.Category.{u_8, u_4} B'
    eA : CategoryTheory.Equivalence A A'
    eB : CategoryTheory.Equivalence B B'
    e' : CategoryTheory.Equivalence A' B'
    F : CategoryTheory.Functor A B'
    hF : CategoryTheory.Iso (eA.functor.comp e'.functor) F
    Y' : B
    ⊢ Eq ((AlgebraicTopology.DoldKan.Compatibility.equivalence₂ eB hF).counitIso.h …
  -/
  dsimp [equivalence₂, Iso.refl]
  simp only [equivalence₁CounitIso_eq, equivalence₂CounitIso_hom_app,
    equivalence₁CounitIso_hom_app, Functor.map_comp, assoc]


/-- The unit isomorphism of the equivalence `equivalence₂` between `A` and `B`. -/
@[simps!]
def equivalence₂UnitIso : 𝟭 A ≅ (F ⋙ eB.inverse) ⋙ eB.functor ⋙ e'.inverse ⋙ eA.inverse :=
  calc
    𝟭 A ≅ F ⋙ e'.inverse ⋙ eA.inverse := equivalence₁UnitIso hF
    _ ≅ F ⋙ 𝟭 B' ⋙ e'.inverse ⋙ eA.inverse := Iso.refl _
    _ ≅ F ⋙ (eB.inverse ⋙ eB.functor) ⋙ e'.inverse ⋙ eA.inverse :=
      isoWhiskerLeft _ (isoWhiskerRight eB.counitIso.symm _)
    _ ≅ (F ⋙ eB.inverse) ⋙ eB.functor ⋙ e'.inverse ⋙ eA.inverse := Iso.refl _


theorem equivalence₂UnitIso_eq : (equivalence₂ eB hF).unitIso = equivalence₂UnitIso eB hF := by
  /-
    A : Type u_1
    A' : Type u_2
    B : Type u_3
    B' : Type u_4
    inst✝³ : CategoryTheory.Category.{u_5, u_1} A
    inst✝² : CategoryTheory.Category.{u_7, u_2} A'
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} B
    inst✝ : CategoryTheory.Category.{u_8, u_4} B'
    eA : CategoryTheory.Equivalence A A'
    eB : CategoryTheory.Equivalence B B'
    e' : CategoryTheory.Equivalence A' B'
    F : CategoryTheory.Functor A B'
    hF : CategoryTheory.Iso (eA.functor.comp e'.functor) F
    ⊢ Eq (AlgebraicTopology.DoldKan.Compatibility.equivalence₂ eB hF).unitIso (Alg …
  -/
  ext X
  /-
    case w.w.h
    A : Type u_1
    A' : Type u_2
    B : Type u_3
    B' : Type u_4
    inst✝³ : CategoryTheory.Category.{u_5, u_1} A
    inst✝² : CategoryTheory.Category.{u_7, u_2} A'
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} B
    inst✝ : CategoryTheory.Category.{u_8, u_4} B'
    eA : CategoryTheory.Equivalence A A'
    eB : CategoryTheory.Equivalence B B'
    e' : CategoryTheory.Equivalence A' B'
    F : CategoryTheory.Functor A B'
    hF : CategoryTheory.Iso (eA.functor.comp e'.functor) F
    X : A
    ⊢ Eq ((AlgebraicTopology.DoldKan.Compatibility.equivalence₂ eB hF).unitIso.hom …
  -/
  dsimp [equivalence₂]
  simp only [equivalence₂UnitIso_hom_app, equivalence₁UnitIso_eq, equivalence₁UnitIso_hom_app,
      assoc, NatIso.cancel_natIso_hom_left]
  /-
    case w.w.h
    A : Type u_1
    A' : Type u_2
    B : Type u_3
    B' : Type u_4
    inst✝³ : CategoryTheory.Category.{u_5, u_1} A
    inst✝² : CategoryTheory.Category.{u_7, u_2} A'
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} B
    inst✝ : CategoryTheory.Category.{u_8, u_4} B'
    eA : CategoryTheory.Equivalence A A'
    eB : CategoryTheory.Equivalence B B'
    e' : CategoryTheory.Equivalence A' B'
    F : CategoryTheory.Functor A B'
    hF : CategoryTheory.Iso (eA.functor.comp e'.functor) F
    X : A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (eA.unitIso.hom.app X) (CategoryTheor …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The equivalence `A ≅ B` whose functor is `F ⋙ eB.inverse` and
whose inverse is `G : B ≅ A`. -/
@[simps! inverse]
def equivalence : A ≌ B :=
  ((equivalence₂ eB hF).changeInverse
    (calc eB.functor ⋙ e'.inverse ⋙ eA.inverse ≅
        (eB.functor ⋙ e'.inverse) ⋙ eA.inverse := (Functor.associator _ _ _).symm
    _ ≅ (G ⋙ eA.functor) ⋙ eA.inverse := isoWhiskerRight hG _
    _ ≅ G ⋙ 𝟭 A := isoWhiskerLeft _ eA.unitIso.symm
    _ ≅ G := G.rightUnitor))


theorem equivalence_functor : (equivalence hF hG).functor = F ⋙ eB.inverse :=
  rfl


/-- The isomorphism `eB.functor ⋙ e'.inverse ⋙ e'.functor ≅ eB.functor` deduced
from the counit isomorphism of `e'`. -/
@[simps! hom_app]
def τ₀ : eB.functor ⋙ e'.inverse ⋙ e'.functor ≅ eB.functor :=
  calc
    eB.functor ⋙ e'.inverse ⋙ e'.functor ≅ eB.functor ⋙ 𝟭 _ := isoWhiskerLeft _ e'.counitIso
    _ ≅ eB.functor := Functor.rightUnitor _


/-- The isomorphism `eB.functor ⋙ e'.inverse ⋙ e'.functor ≅ eB.functor` deduced
from the isomorphisms `hF : eA.functor ⋙ e'.functor ≅ F`,
`hG : eB.functor ⋙ e'.inverse ≅ G ⋙ eA.functor` and the datum of
an isomorphism `η : G ⋙ F ≅ eB.functor`. -/
@[simps! hom_app]
def τ₁ (η : G ⋙ F ≅ eB.functor) : eB.functor ⋙ e'.inverse ⋙ e'.functor ≅ eB.functor :=
  calc
    eB.functor ⋙ e'.inverse ⋙ e'.functor ≅ (eB.functor ⋙ e'.inverse) ⋙ e'.functor :=
        Iso.refl _
    _ ≅ (G ⋙ eA.functor) ⋙ e'.functor := isoWhiskerRight hG _
                                          /-
                                            A : Type u_1
                                            A' : Type u_2
                                            B : Type u_3
                                            B' : Type u_4
                                            inst✝³ : CategoryTheory.Category.{?u.47633, u_1} A
                                            inst✝² : CategoryTheory.Category.{?u.47637, u_2} A'
                                            inst✝¹ : CategoryTheory.Category.{?u.47641, u_3} B
                                            inst✝ : CategoryTheory.Category.{?u.47645, u_4} B'
                                            eA : CategoryTheory.Equivalence A A'
                                            eB : CategoryTheory.Equivalence B B'
                                            e' : CategoryTheory.Equivalence A' B'
                                            F : CategoryTheory.Functor A B'
                                            hF : CategoryTheory.Iso (eA.functor.comp e'.functor) F
                                            G : CategoryTheory.Functor B A
                                            hG : CategoryTheory.Iso (eB.functor.comp e'.inverse) (G.comp eA.functor)
                                            η : CategoryTheory.Iso (G.comp F) eB.functor
                                            ⊢ CategoryTheory.Iso ((G.comp eA.functor).comp e'.functor) (G.comp (eA.functor …
                                          -/
    _ ≅ G ⋙ eA.functor ⋙ e'.functor := by rfl
                                          /-
                                            🎉 no goals
                                          -/
    _ ≅ G ⋙ F := isoWhiskerLeft _ hF
    _ ≅ eB.functor := η


/-- The counit isomorphism of `equivalence`. -/
@[simps!]
def equivalenceCounitIso : G ⋙ F ⋙ eB.inverse ≅ 𝟭 B :=
  calc
    G ⋙ F ⋙ eB.inverse ≅ (G ⋙ F) ⋙ eB.inverse := Iso.refl _
    _ ≅ eB.functor ⋙ eB.inverse := isoWhiskerRight η _
    _ ≅ 𝟭 B := eB.unitIso.symm


theorem equivalenceCounitIso_eq (hη : τ₀ = τ₁ hF hG η) :
    (equivalence hF hG).counitIso = equivalenceCounitIso η := by
  /-
    A : Type u_1
    A' : Type u_2
    B : Type u_3
    B' : Type u_4
    inst✝³ : CategoryTheory.Category.{u_8, u_1} A
    inst✝² : CategoryTheory.Category.{u_7, u_2} A'
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} B
    inst✝ : CategoryTheory.Category.{u_5, u_4} B'
    eA : CategoryTheory.Equivalence A A'
    eB : CategoryTheory.Equivalence B B'
    e' : CategoryTheory.Equivalence A' B'
    F : CategoryTheory.Functor A B'
    hF : CategoryTheory.Iso (eA.functor.comp e'.functor) F
    G : CategoryTheory.Functor B A
    hG : CategoryTheory.Iso (eB.functor.comp e'.inverse) (G.comp eA.functor)
    η : CategoryTheory.Iso (G.comp F) eB.functor
    hη : Eq AlgebraicTopology.DoldKan.Compatibility.τ₀ (AlgebraicTopology.DoldKan. …
    ⊢ Eq (AlgebraicTopology.DoldKan.Compatibility.equivalence hF hG).counitIso (Al …
  -/
  ext1; apply NatTrans.ext; ext Y
  /-
    case w.app.h
    A : Type u_1
    A' : Type u_2
    B : Type u_3
    B' : Type u_4
    inst✝³ : CategoryTheory.Category.{u_8, u_1} A
    inst✝² : CategoryTheory.Category.{u_7, u_2} A'
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} B
    inst✝ : CategoryTheory.Category.{u_5, u_4} B'
    eA : CategoryTheory.Equivalence A A'
    eB : CategoryTheory.Equivalence B B'
    e' : CategoryTheory.Equivalence A' B'
    F : CategoryTheory.Functor A B'
    hF : CategoryTheory.Iso (eA.functor.comp e'.functor) F
    G : CategoryTheory.Functor B A
    hG : CategoryTheory.Iso (eB.functor.comp e'.inverse) (G.comp eA.functor)
    η : CategoryTheory.Iso (G.comp F) eB.functor
    hη : Eq AlgebraicTopology.DoldKan.Compatibility.τ₀ (AlgebraicTopology.DoldKan. …
    Y : B
    ⊢ Eq ((AlgebraicTopology.DoldKan.Compatibility.equivalence hF hG).counitIso.ho …
  -/
  dsimp [equivalence]
  simp only [comp_id, id_comp, Functor.map_comp, equivalence₂CounitIso_eq,
    equivalence₂CounitIso_hom_app, assoc, equivalenceCounitIso_hom_app]
  /-
    case w.app.h
    A : Type u_1
    A' : Type u_2
    B : Type u_3
    B' : Type u_4
    inst✝³ : CategoryTheory.Category.{u_8, u_1} A
    inst✝² : CategoryTheory.Category.{u_7, u_2} A'
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} B
    inst✝ : CategoryTheory.Category.{u_5, u_4} B'
    eA : CategoryTheory.Equivalence A A'
    eB : CategoryTheory.Equivalence B B'
    e' : CategoryTheory.Equivalence A' B'
    F : CategoryTheory.Functor A B'
    hF : CategoryTheory.Iso (eA.functor.comp e'.functor) F
    G : CategoryTheory.Functor B A
    hG : CategoryTheory.Iso (eB.functor.comp e'.inverse) (G.comp eA.functor)
    η : CategoryTheory.Iso (G.comp F) eB.functor
    hη : Eq AlgebraicTopology.DoldKan.Compatibility.τ₀ (AlgebraicTopology.DoldKan. …
    Y : B
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (eB.inverse.map (F.map (eA.unitIso.ho …
  -/
  simp only [← eB.inverse.map_comp_assoc, ← τ₀_hom_app, hη, τ₁_hom_app]
  /-
    case w.app.h
    A : Type u_1
    A' : Type u_2
    B : Type u_3
    B' : Type u_4
    inst✝³ : CategoryTheory.Category.{u_8, u_1} A
    inst✝² : CategoryTheory.Category.{u_7, u_2} A'
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} B
    inst✝ : CategoryTheory.Category.{u_5, u_4} B'
    eA : CategoryTheory.Equivalence A A'
    eB : CategoryTheory.Equivalence B B'
    e' : CategoryTheory.Equivalence A' B'
    F : CategoryTheory.Functor A B'
    hF : CategoryTheory.Iso (eA.functor.comp e'.functor) F
    G : CategoryTheory.Functor B A
    hG : CategoryTheory.Iso (eB.functor.comp e'.inverse) (G.comp eA.functor)
    η : CategoryTheory.Iso (G.comp F) eB.functor
    hη : Eq AlgebraicTopology.DoldKan.Compatibility.τ₀ (AlgebraicTopology.DoldKan. …
    Y : B
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (eB.inverse.map (CategoryTheory.Categ …
  -/
  erw [hF.inv.naturality_assoc, hF.inv.naturality_assoc]
  /-
    case w.app.h
    A : Type u_1
    A' : Type u_2
    B : Type u_3
    B' : Type u_4
    inst✝³ : CategoryTheory.Category.{u_8, u_1} A
    inst✝² : CategoryTheory.Category.{u_7, u_2} A'
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} B
    inst✝ : CategoryTheory.Category.{u_5, u_4} B'
    eA : CategoryTheory.Equivalence A A'
    eB : CategoryTheory.Equivalence B B'
    e' : CategoryTheory.Equivalence A' B'
    F : CategoryTheory.Functor A B'
    hF : CategoryTheory.Iso (eA.functor.comp e'.functor) F
    G : CategoryTheory.Functor B A
    hG : CategoryTheory.Iso (eB.functor.comp e'.inverse) (G.comp eA.functor)
    η : CategoryTheory.Iso (G.comp F) eB.functor
    hη : Eq AlgebraicTopology.DoldKan.Compatibility.τ₀ (AlgebraicTopology.DoldKan. …
    Y : B
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (eB.inverse.map (CategoryTheory.Categ …
  -/
  dsimp
  /-
    case w.app.h
    A : Type u_1
    A' : Type u_2
    B : Type u_3
    B' : Type u_4
    inst✝³ : CategoryTheory.Category.{u_8, u_1} A
    inst✝² : CategoryTheory.Category.{u_7, u_2} A'
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} B
    inst✝ : CategoryTheory.Category.{u_5, u_4} B'
    eA : CategoryTheory.Equivalence A A'
    eB : CategoryTheory.Equivalence B B'
    e' : CategoryTheory.Equivalence A' B'
    F : CategoryTheory.Functor A B'
    hF : CategoryTheory.Iso (eA.functor.comp e'.functor) F
    G : CategoryTheory.Functor B A
    hG : CategoryTheory.Iso (eB.functor.comp e'.inverse) (G.comp eA.functor)
    η : CategoryTheory.Iso (G.comp F) eB.functor
    hη : Eq AlgebraicTopology.DoldKan.Compatibility.τ₀ (AlgebraicTopology.DoldKan. …
    Y : B
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (eB.inverse.map (CategoryTheory.Categ …
  -/
  congr 2
  simp only [← e'.functor.map_comp_assoc, Equivalence.fun_inv_map, assoc,
    Iso.inv_hom_id_app_assoc, hG.inv_hom_id_app]
  /-
    case w.app.h.e_a.e_a
    A : Type u_1
    A' : Type u_2
    B : Type u_3
    B' : Type u_4
    inst✝³ : CategoryTheory.Category.{u_8, u_1} A
    inst✝² : CategoryTheory.Category.{u_7, u_2} A'
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} B
    inst✝ : CategoryTheory.Category.{u_5, u_4} B'
    eA : CategoryTheory.Equivalence A A'
    eB : CategoryTheory.Equivalence B B'
    e' : CategoryTheory.Equivalence A' B'
    F : CategoryTheory.Functor A B'
    hF : CategoryTheory.Iso (eA.functor.comp e'.functor) F
    G : CategoryTheory.Functor B A
    hG : CategoryTheory.Iso (eB.functor.comp e'.inverse) (G.comp eA.functor)
    η : CategoryTheory.Iso (G.comp F) eB.functor
    hη : Eq AlgebraicTopology.DoldKan.Compatibility.τ₀ (AlgebraicTopology.DoldKan. …
    Y : B
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (hF.inv.app (G.obj Y)) (CategoryTheor …
  -/
  dsimp
  /-
    case w.app.h.e_a.e_a
    A : Type u_1
    A' : Type u_2
    B : Type u_3
    B' : Type u_4
    inst✝³ : CategoryTheory.Category.{u_8, u_1} A
    inst✝² : CategoryTheory.Category.{u_7, u_2} A'
    inst✝¹ : CategoryTheory.Category.{u_6, u_3} B
    inst✝ : CategoryTheory.Category.{u_5, u_4} B'
    eA : CategoryTheory.Equivalence A A'
    eB : CategoryTheory.Equivalence B B'
    e' : CategoryTheory.Equivalence A' B'
    F : CategoryTheory.Functor A B'
    hF : CategoryTheory.Iso (eA.functor.comp e'.functor) F
    G : CategoryTheory.Functor B A
    hG : CategoryTheory.Iso (eB.functor.comp e'.inverse) (G.comp eA.functor)
    η : CategoryTheory.Iso (G.comp F) eB.functor
    hη : Eq AlgebraicTopology.DoldKan.Compatibility.τ₀ (AlgebraicTopology.DoldKan. …
    Y : B
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (hF.inv.app (G.obj Y)) (CategoryTheor …
  -/
  rw [comp_id, eA.functor_unitIso_comp, e'.functor.map_id, id_comp, hF.inv_hom_id_app_assoc]
  /-
    🎉 no goals
  -/


/-- The isomorphism `eA.functor ≅ F ⋙ e'.inverse` deduced from the
unit isomorphism of `e'` and the isomorphism `hF : eA.functor ⋙ e'.functor ≅ F`. -/
@[simps!]
def υ : eA.functor ≅ F ⋙ e'.inverse :=
  calc
    eA.functor ≅ eA.functor ⋙ 𝟭 A' := (Functor.leftUnitor _).symm
    _ ≅ eA.functor ⋙ e'.functor ⋙ e'.inverse := isoWhiskerLeft _ e'.unitIso
    _ ≅ (eA.functor ⋙ e'.functor) ⋙ e'.inverse := Iso.refl _
    _ ≅ F ⋙ e'.inverse := isoWhiskerRight hF _


/-- The unit isomorphism of `equivalence`. -/
@[simps!]
def equivalenceUnitIso : 𝟭 A ≅ (F ⋙ eB.inverse) ⋙ G :=
  calc
    𝟭 A ≅ eA.functor ⋙ eA.inverse := eA.unitIso
    _ ≅ (F ⋙ e'.inverse) ⋙ eA.inverse := isoWhiskerRight ε _
    _ ≅ F ⋙ 𝟭 B' ⋙ e'.inverse ⋙ eA.inverse := Iso.refl _
    _ ≅ F ⋙ (eB.inverse ⋙ eB.functor) ⋙ e'.inverse ⋙ eA.inverse :=
      isoWhiskerLeft _ (isoWhiskerRight eB.counitIso.symm _)
    _ ≅ (F ⋙ eB.inverse) ⋙ (eB.functor ⋙ e'.inverse) ⋙ eA.inverse := Iso.refl _
    _ ≅ (F ⋙ eB.inverse) ⋙ (G ⋙ eA.functor) ⋙ eA.inverse :=
      isoWhiskerLeft _ (isoWhiskerRight hG _)
    _ ≅ (F ⋙ eB.inverse ⋙ G) ⋙ eA.functor ⋙ eA.inverse := Iso.refl _
    _ ≅ (F ⋙ eB.inverse ⋙ G) ⋙ 𝟭 A := isoWhiskerLeft _ eA.unitIso.symm
    _ ≅ (F ⋙ eB.inverse) ⋙ G := Iso.refl _


theorem equivalenceUnitIso_eq (hε : υ hF = ε) :
    (equivalence hF hG).unitIso = equivalenceUnitIso hG ε := by
  /-
    A : Type u_1
    A' : Type u_2
    B : Type u_3
    B' : Type u_4
    inst✝³ : CategoryTheory.Category.{u_6, u_1} A
    inst✝² : CategoryTheory.Category.{u_5, u_2} A'
    inst✝¹ : CategoryTheory.Category.{u_8, u_3} B
    inst✝ : CategoryTheory.Category.{u_7, u_4} B'
    eA : CategoryTheory.Equivalence A A'
    eB : CategoryTheory.Equivalence B B'
    e' : CategoryTheory.Equivalence A' B'
    F : CategoryTheory.Functor A B'
    hF : CategoryTheory.Iso (eA.functor.comp e'.functor) F
    G : CategoryTheory.Functor B A
    hG : CategoryTheory.Iso (eB.functor.comp e'.inverse) (G.comp eA.functor)
    ε : CategoryTheory.Iso eA.functor (F.comp e'.inverse)
    hε : Eq (AlgebraicTopology.DoldKan.Compatibility.υ hF) ε
    ⊢ Eq (AlgebraicTopology.DoldKan.Compatibility.equivalence hF hG).unitIso (Alge …
  -/
  ext1; apply NatTrans.ext; ext X
  /-
    case w.app.h
    A : Type u_1
    A' : Type u_2
    B : Type u_3
    B' : Type u_4
    inst✝³ : CategoryTheory.Category.{u_6, u_1} A
    inst✝² : CategoryTheory.Category.{u_5, u_2} A'
    inst✝¹ : CategoryTheory.Category.{u_8, u_3} B
    inst✝ : CategoryTheory.Category.{u_7, u_4} B'
    eA : CategoryTheory.Equivalence A A'
    eB : CategoryTheory.Equivalence B B'
    e' : CategoryTheory.Equivalence A' B'
    F : CategoryTheory.Functor A B'
    hF : CategoryTheory.Iso (eA.functor.comp e'.functor) F
    G : CategoryTheory.Functor B A
    hG : CategoryTheory.Iso (eB.functor.comp e'.inverse) (G.comp eA.functor)
    ε : CategoryTheory.Iso eA.functor (F.comp e'.inverse)
    hε : Eq (AlgebraicTopology.DoldKan.Compatibility.υ hF) ε
    X : A
    ⊢ Eq ((AlgebraicTopology.DoldKan.Compatibility.equivalence hF hG).unitIso.hom. …
  -/
  dsimp [equivalence]
  /-
    case w.app.h
    A : Type u_1
    A' : Type u_2
    B : Type u_3
    B' : Type u_4
    inst✝³ : CategoryTheory.Category.{u_6, u_1} A
    inst✝² : CategoryTheory.Category.{u_5, u_2} A'
    inst✝¹ : CategoryTheory.Category.{u_8, u_3} B
    inst✝ : CategoryTheory.Category.{u_7, u_4} B'
    eA : CategoryTheory.Equivalence A A'
    eB : CategoryTheory.Equivalence B B'
    e' : CategoryTheory.Equivalence A' B'
    F : CategoryTheory.Functor A B'
    hF : CategoryTheory.Iso (eA.functor.comp e'.functor) F
    G : CategoryTheory.Functor B A
    hG : CategoryTheory.Iso (eB.functor.comp e'.inverse) (G.comp eA.functor)
    ε : CategoryTheory.Iso eA.functor (F.comp e'.inverse)
    hε : Eq (AlgebraicTopology.DoldKan.Compatibility.υ hF) ε
    X : A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.DoldKan.Compatibi …
  -/
  simp only [assoc, comp_id, equivalenceUnitIso_hom_app]
  /-
    case w.app.h
    A : Type u_1
    A' : Type u_2
    B : Type u_3
    B' : Type u_4
    inst✝³ : CategoryTheory.Category.{u_6, u_1} A
    inst✝² : CategoryTheory.Category.{u_5, u_2} A'
    inst✝¹ : CategoryTheory.Category.{u_8, u_3} B
    inst✝ : CategoryTheory.Category.{u_7, u_4} B'
    eA : CategoryTheory.Equivalence A A'
    eB : CategoryTheory.Equivalence B B'
    e' : CategoryTheory.Equivalence A' B'
    F : CategoryTheory.Functor A B'
    hF : CategoryTheory.Iso (eA.functor.comp e'.functor) F
    G : CategoryTheory.Functor B A
    hG : CategoryTheory.Iso (eB.functor.comp e'.inverse) (G.comp eA.functor)
    ε : CategoryTheory.Iso eA.functor (F.comp e'.inverse)
    hε : Eq (AlgebraicTopology.DoldKan.Compatibility.υ hF) ε
    X : A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.DoldKan.Compatibi …
  -/
  erw [id_comp]
  simp only [equivalence₂UnitIso_eq eB hF, equivalence₂UnitIso_hom_app,
    ← eA.inverse.map_comp_assoc, assoc, ← hε, υ_hom_app]



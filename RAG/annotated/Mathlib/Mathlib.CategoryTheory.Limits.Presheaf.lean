/--
The functor taking `(E : ℰ) (c : Cᵒᵖ)` to the homset `(A.obj C ⟶ E)`. It is shown in `L_adjunction`
that this functor has a left adjoint (provided `E` has colimits) given by taking colimits over
categories of elements.
In the case where `ℰ = Cᵒᵖ ⥤ Type u` and `A = yoneda`, this functor is isomorphic to the identity.

Defined as in [MM92], Chapter I, Section 5, Theorem 2.
-/
@[simps!]
def restrictedYoneda : ℰ ⥤ Cᵒᵖ ⥤ Type v₁ :=
  yoneda ⋙ (whiskeringLeft _ _ (Type v₁)).obj (Functor.op A)


/-- Auxiliary definition for `restrictedYonedaHomEquiv`. -/
def restrictedYonedaHomEquiv' (P : Cᵒᵖ ⥤ Type v₁) (E : ℰ) :
    (CostructuredArrow.proj yoneda P ⋙ A ⟶
      (Functor.const (CostructuredArrow yoneda P)).obj E) ≃
      (P ⟶ (restrictedYoneda A).obj E) where
  toFun f :=
    { app := fun _ x => f.app (CostructuredArrow.mk (yonedaEquiv.symm x))
      naturality := fun {X₁ X₂} φ => by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          E : ℰ
          f : Quiver.Hom ((CategoryTheory.CostructuredArrow.proj CategoryTheory.yoneda P …
          X₁ X₂ : Opposite C
          φ : Quiver.Hom X₁ X₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.map φ) ((fun x x_1 => f.app (Categ …
        -/
        ext x
        let ψ : CostructuredArrow.mk (yonedaEquiv.symm (P.toPrefunctor.map φ x)) ⟶
          CostructuredArrow.mk (yonedaEquiv.symm x) := CostructuredArrow.homMk φ.unop (by
            dsimp [yonedaEquiv]
            aesop_cat )
        /-
          case h
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          E : ℰ
          f : Quiver.Hom ((CategoryTheory.CostructuredArrow.proj CategoryTheory.yoneda P …
          X₁ X₂ : Opposite C
          φ : Quiver.Hom X₁ X₂
          x : P.obj X₁
          ψ : Quiver.Hom (CategoryTheory.CostructuredArrow.mk (CategoryTheory.yonedaEqui …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.map φ) ((fun x x_1 => f.app (Categ …
        -/
        simpa using (f.naturality ψ).symm }
        /-
          🎉 no goals
        -/
  invFun g :=
    { app := fun y => yonedaEquiv (y.hom ≫ g)
      naturality := fun {X₁ X₂} φ => by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          E : ℰ
          g : Quiver.Hom P ((CategoryTheory.Presheaf.restrictedYoneda A).obj E)
          X₁ X₂ : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P
          φ : Quiver.Hom X₁ X₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.CostructuredArrow.p …
        -/
        dsimp
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          E : ℰ
          g : Quiver.Hom P ((CategoryTheory.Presheaf.restrictedYoneda A).obj E)
          X₁ X₂ : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P
          φ : Quiver.Hom X₁ X₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (A.map φ.left) (CategoryTheory.yoneda …
        -/
        rw [← CostructuredArrow.w φ]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          E : ℰ
          g : Quiver.Hom P ((CategoryTheory.Presheaf.restrictedYoneda A).obj E)
          X₁ X₂ : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P
          φ : Quiver.Hom X₁ X₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (A.map φ.left) (CategoryTheory.yoneda …
        -/
        dsimp [yonedaEquiv]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          E : ℰ
          g : Quiver.Hom P ((CategoryTheory.Presheaf.restrictedYoneda A).obj E)
          X₁ X₂ : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P
          φ : Quiver.Hom X₁ X₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (A.map φ.left) (g.app { unop := X₂.le …
        -/
        simp only [comp_id, id_comp]
        refine (congr_fun (g.naturality φ.left.op) (X₂.hom.app (Opposite.op X₂.left)
          (𝟙 _))).symm.trans ?_
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          E : ℰ
          g : Quiver.Hom P ((CategoryTheory.Presheaf.restrictedYoneda A).obj E)
          X₁ X₂ : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P
          φ : Quiver.Hom X₁ X₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.map φ.left.op) (g.app { unop := X₁ …
        -/
        dsimp
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          E : ℰ
          g : Quiver.Hom P ((CategoryTheory.Presheaf.restrictedYoneda A).obj E)
          X₁ X₂ : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P
          φ : Quiver.Hom X₁ X₂
          ⊢ Eq (g.app { unop := X₁.left } (P.map φ.left.op (X₂.hom.app { unop := X₂.left …
        -/
        apply congr_arg
        /-
          case h
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          E : ℰ
          g : Quiver.Hom P ((CategoryTheory.Presheaf.restrictedYoneda A).obj E)
          X₁ X₂ : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P
          φ : Quiver.Hom X₁ X₂
          ⊢ Eq (P.map φ.left.op (X₂.hom.app { unop := X₂.left } (CategoryTheory.Category …
        -/
        simpa using congr_fun (X₂.hom.naturality φ.left.op).symm (𝟙 _) }
        /-
          🎉 no goals
        -/
  left_inv f := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
      A : CategoryTheory.Functor C ℰ
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      E : ℰ
      f : Quiver.Hom ((CategoryTheory.CostructuredArrow.proj CategoryTheory.yoneda P …
      ⊢ Eq ((fun g => { app := fun y => CategoryTheory.yonedaEquiv (CategoryTheory.C …
    -/
    ext ⟨X, ⟨⟨⟩⟩, φ⟩
    suffices yonedaEquiv.symm (φ.app (Opposite.op X) (𝟙 X)) = φ by
      dsimp
      erw [yonedaEquiv_apply]
      dsimp [CostructuredArrow.mk]
      erw [this]
    /-
      case w.h.mk.mk.unit
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
      A : CategoryTheory.Functor C ℰ
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      E : ℰ
      f : Quiver.Hom ((CategoryTheory.CostructuredArrow.proj CategoryTheory.yoneda P …
      X : C
      φ : Quiver.Hom (CategoryTheory.yoneda.obj X) ((CategoryTheory.Functor.fromPUni …
      ⊢ Eq (CategoryTheory.yonedaEquiv.symm (φ.app { unop := X } (CategoryTheory.Cat …
    -/
    exact yonedaEquiv.injective (by aesop_cat)
    /-
      🎉 no goals
    -/
  right_inv g := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
      A : CategoryTheory.Functor C ℰ
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      E : ℰ
      g : Quiver.Hom P ((CategoryTheory.Presheaf.restrictedYoneda A).obj E)
      ⊢ Eq ((fun f => { app := fun x x_1 => f.app (CategoryTheory.CostructuredArrow. …
    -/
    ext X x
    /-
      case w.h.h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
      A : CategoryTheory.Functor C ℰ
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      E : ℰ
      g : Quiver.Hom P ((CategoryTheory.Presheaf.restrictedYoneda A).obj E)
      X : Opposite C
      x : P.obj X
      ⊢ Eq (((fun f => { app := fun x x_1 => f.app (CategoryTheory.CostructuredArrow …
    -/
    dsimp
    /-
      case w.h.h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
      A : CategoryTheory.Functor C ℰ
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      E : ℰ
      g : Quiver.Hom P ((CategoryTheory.Presheaf.restrictedYoneda A).obj E)
      X : Opposite C
      x : P.obj X
      ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.CategoryStruct.comp (Category …
    -/
    erw [yonedaEquiv_apply]
    /-
      case w.h.h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
      A : CategoryTheory.Functor C ℰ
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      E : ℰ
      g : Quiver.Hom P ((CategoryTheory.Presheaf.restrictedYoneda A).obj E)
      X : Opposite C
      x : P.obj X
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.yonedaEquiv.symm x)  …
    -/
    dsimp
    /-
      case w.h.h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
      A : CategoryTheory.Functor C ℰ
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      E : ℰ
      g : Quiver.Hom P ((CategoryTheory.Presheaf.restrictedYoneda A).obj E)
      X : Opposite C
      x : P.obj X
      ⊢ Eq (g.app X ((CategoryTheory.yonedaEquiv.symm x).app X (CategoryTheory.Categ …
    -/
    rw [yonedaEquiv_symm_app_apply]
    /-
      case w.h.h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
      A : CategoryTheory.Functor C ℰ
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      E : ℰ
      g : Quiver.Hom P ((CategoryTheory.Presheaf.restrictedYoneda A).obj E)
      X : Opposite C
      x : P.obj X
      ⊢ Eq (g.app X (P.map (CategoryTheory.CategoryStruct.id (Opposite.unop X)).op x …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Auxiliary definition for `yonedaAdjunction`. -/
noncomputable def restrictedYonedaHomEquiv (P : Cᵒᵖ ⥤ Type v₁) (E : ℰ) :
    (L.obj P ⟶ E) ≃ (P ⟶ (restrictedYoneda A).obj E) :=
  ((Functor.isPointwiseLeftKanExtensionOfIsLeftKanExtension _ α P).homEquiv E).trans
    (restrictedYonedaHomEquiv' A P E)


/-- If `L : (Cᵒᵖ ⥤ Type v₁) ⥤ ℰ` is a pointwise left Kan extension
of a functor `A : C ⥤ ℰ` along the Yoneda embedding,
then `L` is a left adjoint of `restrictedYoneda A : ℰ ⥤ Cᵒᵖ ⥤ Type v₁` -/
noncomputable def yonedaAdjunction : L ⊣ restrictedYoneda A :=
  Adjunction.mkOfHomEquiv
    { homEquiv := restrictedYonedaHomEquiv L α
      homEquiv_naturality_left_symm := fun {P Q X} f g => by
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
          L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
          α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
          inst✝ : L.IsLeftKanExtension α
          P Q : CategoryTheory.Functor (Opposite C) (Type v₁)
          X : ℰ
          f : Quiver.Hom P Q
          g : Quiver.Hom Q ((CategoryTheory.Presheaf.restrictedYoneda A).obj X)
          ⊢ Eq ((CategoryTheory.Presheaf.restrictedYonedaHomEquiv L α P X).symm (Categor …
        -/
        obtain ⟨g, rfl⟩ := (restrictedYonedaHomEquiv L α Q X).surjective g
        /-
          case intro
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
          L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
          α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
          inst✝ : L.IsLeftKanExtension α
          P Q : CategoryTheory.Functor (Opposite C) (Type v₁)
          X : ℰ
          f : Quiver.Hom P Q
          g : Quiver.Hom (L.obj Q) X
          ⊢ Eq ((CategoryTheory.Presheaf.restrictedYonedaHomEquiv L α P X).symm (Categor …
        -/
        apply (restrictedYonedaHomEquiv L α P X).injective
        /-
          case intro.a
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
          L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
          α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
          inst✝ : L.IsLeftKanExtension α
          P Q : CategoryTheory.Functor (Opposite C) (Type v₁)
          X : ℰ
          f : Quiver.Hom P Q
          g : Quiver.Hom (L.obj Q) X
          ⊢ Eq ((CategoryTheory.Presheaf.restrictedYonedaHomEquiv L α P X) ((CategoryThe …
        -/
        simp only [Equiv.apply_symm_apply, Equiv.symm_apply_apply]
        /-
          case intro.a
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
          L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
          α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
          inst✝ : L.IsLeftKanExtension α
          P Q : CategoryTheory.Functor (Opposite C) (Type v₁)
          X : ℰ
          f : Quiver.Hom P Q
          g : Quiver.Hom (L.obj Q) X
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f ((CategoryTheory.Presheaf.restricte …
        -/
        ext Y y
        /-
          case intro.a.w.h.h
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
          L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
          α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
          inst✝ : L.IsLeftKanExtension α
          P Q : CategoryTheory.Functor (Opposite C) (Type v₁)
          X : ℰ
          f : Quiver.Hom P Q
          g : Quiver.Hom (L.obj Q) X
          Y : Opposite C
          y : P.obj Y
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp f ((CategoryTheory.Presheaf.restrict …
        -/
        dsimp [restrictedYonedaHomEquiv, restrictedYonedaHomEquiv', IsColimit.homEquiv]
        /-
          case intro.a.w.h.h
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
          L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
          α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
          inst✝ : L.IsLeftKanExtension α
          P Q : CategoryTheory.Functor (Opposite C) (Type v₁)
          X : ℰ
          f : Quiver.Hom P Q
          g : Quiver.Hom (L.obj Q) X
          Y : Opposite C
          y : P.obj Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        rw [assoc, assoc, ← L.map_comp_assoc]
        /-
          case intro.a.w.h.h
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
          L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
          α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
          inst✝ : L.IsLeftKanExtension α
          P Q : CategoryTheory.Functor (Opposite C) (Type v₁)
          X : ℰ
          f : Quiver.Hom P Q
          g : Quiver.Hom (L.obj Q) X
          Y : Opposite C
          y : P.obj Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.app (Opposite.unop Y)) (CategoryTh …
        -/
        congr 3
        /-
          case intro.a.w.h.h.e_a.e_a.e_a
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
          L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
          α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
          inst✝ : L.IsLeftKanExtension α
          P Q : CategoryTheory.Functor (Opposite C) (Type v₁)
          X : ℰ
          f : Quiver.Hom P Q
          g : Quiver.Hom (L.obj Q) X
          Y : Opposite C
          y : P.obj Y
          ⊢ Eq (CategoryTheory.yonedaEquiv.symm (f.app Y y)) (CategoryTheory.CategoryStr …
        -/
        apply yonedaEquiv.injective
        /-
          case intro.a.w.h.h.e_a.e_a.e_a.a
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
          L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
          α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
          inst✝ : L.IsLeftKanExtension α
          P Q : CategoryTheory.Functor (Opposite C) (Type v₁)
          X : ℰ
          f : Quiver.Hom P Q
          g : Quiver.Hom (L.obj Q) X
          Y : Opposite C
          y : P.obj Y
          ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.yonedaEquiv.symm (f.app Y y)) …
        -/
        simp [yonedaEquiv]
        /-
          🎉 no goals
        -/
      homEquiv_naturality_right := fun {P X Y} f g => by
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
          L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
          α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
          inst✝ : L.IsLeftKanExtension α
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          X Y : ℰ
          f : Quiver.Hom (L.obj P) X
          g : Quiver.Hom X Y
          ⊢ Eq ((CategoryTheory.Presheaf.restrictedYonedaHomEquiv L α P Y) (CategoryTheo …
        -/
        apply (restrictedYonedaHomEquiv L α P Y).symm.injective
        /-
          case a
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
          L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
          α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
          inst✝ : L.IsLeftKanExtension α
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          X Y : ℰ
          f : Quiver.Hom (L.obj P) X
          g : Quiver.Hom X Y
          ⊢ Eq ((CategoryTheory.Presheaf.restrictedYonedaHomEquiv L α P Y).symm ((Catego …
        -/
        simp only [Equiv.symm_apply_apply]
        /-
          case a
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
          L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
          α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
          inst✝ : L.IsLeftKanExtension α
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          X Y : ℰ
          f : Quiver.Hom (L.obj P) X
          g : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f g) ((CategoryTheory.Presheaf.restri …
        -/
        dsimp [restrictedYonedaHomEquiv, restrictedYonedaHomEquiv', IsColimit.homEquiv]
        /-
          case a
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
          L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
          α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
          inst✝ : L.IsLeftKanExtension α
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          X Y : ℰ
          f : Quiver.Hom (L.obj P) X
          g : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f g) ((CategoryTheory.Functor.isPoint …
        -/
        apply (Functor.isPointwiseLeftKanExtensionOfIsLeftKanExtension L α P).hom_ext
        /-
          case a
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
          L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
          α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
          inst✝ : L.IsLeftKanExtension α
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          X Y : ℰ
          f : Quiver.Hom (L.obj P) X
          g : Quiver.Hom X Y
          ⊢ ∀ (j : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P), Eq (Catego …
        -/
        intro p
        /-
          case a
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
          L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
          α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
          inst✝ : L.IsLeftKanExtension α
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          X Y : ℰ
          f : Quiver.Hom (L.obj P) X
          g : Quiver.Hom X Y
          p : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.LeftExtensi …
        -/
        rw [IsColimit.fac]
        /-
          case a
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
          L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
          α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
          inst✝ : L.IsLeftKanExtension α
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          X Y : ℰ
          f : Quiver.Hom (L.obj P) X
          g : Quiver.Hom X Y
          p : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.LeftExtensi …
        -/
        dsimp [restrictedYoneda, yonedaEquiv]
        /-
          case a
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
          L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
          α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
          inst✝ : L.IsLeftKanExtension α
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          X Y : ℰ
          f : Quiver.Hom (L.obj P) X
          g : Quiver.Hom X Y
          p : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        simp only [assoc]
        /-
          case a
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
          L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
          α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
          inst✝ : L.IsLeftKanExtension α
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          X Y : ℰ
          f : Quiver.Hom (L.obj P) X
          g : Quiver.Hom X Y
          p : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.app p.left) (CategoryTheory.Catego …
        -/
        congr 3
        /-
          case a.e_a.e_a.e_a
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
          L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
          α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
          inst✝ : L.IsLeftKanExtension α
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          X Y : ℰ
          f : Quiver.Hom (L.obj P) X
          g : Quiver.Hom X Y
          p : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P
          ⊢ Eq p.hom { app := fun x f => P.map f.op (p.hom.app { unop := p.left } (Categ …
        -/
        apply yonedaEquiv.injective
        /-
          case a.e_a.e_a.e_a.a
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
          L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
          α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
          inst✝ : L.IsLeftKanExtension α
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          X Y : ℰ
          f : Quiver.Hom (L.obj P) X
          g : Quiver.Hom X Y
          p : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P
          ⊢ Eq (CategoryTheory.yonedaEquiv p.hom) (CategoryTheory.yonedaEquiv { app := f …
        -/
        simp [yonedaEquiv] }
        /-
          🎉 no goals
        -/


include α in
/-- Any left Kan extension along the Yoneda embedding preserves colimits. -/
lemma preservesColimitsOfSize_of_isLeftKanExtension :
    PreservesColimitsOfSize.{v₃, u₃} L :=
  (yonedaAdjunction L α).leftAdjoint_preservesColimits


lemma isIso_of_isLeftKanExtension : IsIso α :=
  (Functor.isPointwiseLeftKanExtensionOfIsLeftKanExtension _ α).isIso_hom


/-- See Property 2 of https://ncatlab.org/nlab/show/Yoneda+extension#properties. -/
noncomputable instance preservesColimitsOfSize_leftKanExtension :
    PreservesColimitsOfSize.{v₃, u₃} (yoneda.leftKanExtension A) :=
  (yonedaAdjunction _ (yoneda.leftKanExtensionUnit A)).leftAdjoint_preservesColimits


instance : IsIso (yoneda.leftKanExtensionUnit A) :=
  isIso_of_isLeftKanExtension _ (yoneda.leftKanExtensionUnit A)


/-- A pointwise left Kan extension along the Yoneda embedding is an extension. -/
noncomputable def isExtensionAlongYoneda :
    yoneda ⋙ yoneda.leftKanExtension A ≅ A :=
  (asIso (yoneda.leftKanExtensionUnit A)).symm


/-- A functor to the presheaf category in which everything in the image is representable (witnessed
by the fact that it factors through the yoneda embedding).
`coconeOfRepresentable` gives a cocone for this functor which is a colimit and has point `P`.
-/
@[reducible]
def functorToRepresentables (P : Cᵒᵖ ⥤ Type v₁) : P.Elementsᵒᵖ ⥤ Cᵒᵖ ⥤ Type v₁ :=
  (CategoryOfElements.π P).leftOp ⋙ yoneda


/-- This is a cocone with point `P` for the functor `functorToRepresentables P`. It is shown in
`colimitOfRepresentable P` that this cocone is a colimit: that is, we have exhibited an arbitrary
presheaf `P` as a colimit of representables.

The construction of [MM92], Chapter I, Section 5, Corollary 3.
-/
@[simps]
noncomputable def coconeOfRepresentable (P : Cᵒᵖ ⥤ Type v₁) :
    Cocone (functorToRepresentables P) where
  pt := P
  ι :=
    { app := fun x => yonedaEquiv.symm x.unop.2
      naturality := fun {x₁ x₂} f => by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          x₁ x₂ : Opposite P.Elements
          f : Quiver.Hom x₁ x₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Presheaf.functorToRe …
        -/
        dsimp
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          x₁ x₂ : Opposite P.Elements
          f : Quiver.Hom x₁ x₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map (↑f.unop). …
        -/
        rw [comp_id, ← yonedaEquiv_symm_map]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          x₁ x₂ : Opposite P.Elements
          f : Quiver.Hom x₁ x₂
          ⊢ Eq (CategoryTheory.yonedaEquiv.symm (P.map (↑f.unop) (Opposite.unop x₂).snd) …
        -/
        congr 1
        /-
          case h.e_6.h
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          x₁ x₂ : Opposite P.Elements
          f : Quiver.Hom x₁ x₂
          ⊢ Eq (P.map (↑f.unop) (Opposite.unop x₂).snd) (Opposite.unop x₁).snd
        -/
        rw [f.unop.2] }
        /-
          🎉 no goals
        -/


/-- The legs of the cocone `coconeOfRepresentable` are natural in the choice of presheaf. -/
theorem coconeOfRepresentable_naturality {P₁ P₂ : Cᵒᵖ ⥤ Type v₁} (α : P₁ ⟶ P₂) (j : P₁.Elementsᵒᵖ) :
    (coconeOfRepresentable P₁).ι.app j ≫ α =
      (coconeOfRepresentable P₂).ι.app ((CategoryOfElements.map α).op.obj j) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P₁ P₂ : CategoryTheory.Functor (Opposite C) (Type v₁)
    α : Quiver.Hom P₁ P₂
    j : Opposite P₁.Elements
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Presheaf.coconeOfRep …
  -/
  ext T f
  /-
    case w.h.h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    P₁ P₂ : CategoryTheory.Functor (Opposite C) (Type v₁)
    α : Quiver.Hom P₁ P₂
    j : Opposite P₁.Elements
    T : Opposite C
    f : ((CategoryTheory.Presheaf.functorToRepresentables P₁).obj j).obj T
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Presheaf.coconeOfRe …
  -/
  simpa [coconeOfRepresentable_ι_app] using FunctorToTypes.naturality _ _ α f.op _
  /-
    🎉 no goals
  -/


/-- The cocone with point `P` given by `coconeOfRepresentable` is a colimit:
that is, we have exhibited an arbitrary presheaf `P` as a colimit of representables.

The result of [MM92], Chapter I, Section 5, Corollary 3.
-/
noncomputable def colimitOfRepresentable (P : Cᵒᵖ ⥤ Type v₁) :
    IsColimit (coconeOfRepresentable P) where
  desc s :=
    { app := fun X x => (s.ι.app (Opposite.op (Functor.elementsMk P X x))).app X (𝟙 _)
      naturality := fun X Y f => by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          s : CategoryTheory.Limits.Cocone (CategoryTheory.Presheaf.functorToRepresentab …
          X Y : Opposite C
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Presheaf.coconeOfRep …
        -/
        ext x
        have eq₁ := congr_fun (congr_app (s.w (CategoryOfElements.homMk (P.elementsMk X x)
          (P.elementsMk Y (P.map f x)) f rfl).op) Y) (𝟙 _)
        /-
          case h
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
          A : CategoryTheory.Functor C ℰ
          P : CategoryTheory.Functor (Opposite C) (Type v₁)
          s : CategoryTheory.Limits.Cocone (CategoryTheory.Presheaf.functorToRepresentab …
          X Y : Opposite C
          f : Quiver.Hom X Y
          x : (CategoryTheory.Presheaf.coconeOfRepresentable P).pt.obj X
          eq₁ : Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Presheaf.functo …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Presheaf.coconeOfRep …
        -/
        dsimp at eq₁ ⊢
        simpa [← eq₁, id_comp] using
          congr_fun ((s.ι.app (Opposite.op (P.elementsMk X x))).naturality f) (𝟙 _) }
  fac s j := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
      A : CategoryTheory.Functor C ℰ
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Presheaf.functorToRepresentab …
      j : Opposite P.Elements
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Presheaf.coconeOfRep …
    -/
    ext X x
    /-
      case w.h.h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
      A : CategoryTheory.Functor C ℰ
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Presheaf.functorToRepresentab …
      j : Opposite P.Elements
      X : Opposite C
      x : ((CategoryTheory.Presheaf.functorToRepresentables P).obj j).obj X
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Presheaf.coconeOfRe …
    -/
    let φ : j.unop ⟶ Functor.elementsMk P X ((yonedaEquiv.symm j.unop.2).app X x) := ⟨x.op, rfl⟩
    /-
      case w.h.h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
      A : CategoryTheory.Functor C ℰ
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Presheaf.functorToRepresentab …
      j : Opposite P.Elements
      X : Opposite C
      x : ((CategoryTheory.Presheaf.functorToRepresentables P).obj j).obj X
      φ : Quiver.Hom (Opposite.unop j) (P.elementsMk X ((CategoryTheory.yonedaEquiv. …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Presheaf.coconeOfRe …
    -/
    simpa using congr_fun (congr_app (s.ι.naturality φ.op).symm X) (𝟙 _)
    /-
      🎉 no goals
    -/
  uniq s m hm := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
      A : CategoryTheory.Functor C ℰ
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Presheaf.functorToRepresentab …
      m : Quiver.Hom (CategoryTheory.Presheaf.coconeOfRepresentable P).pt s.pt
      hm : ∀ (j : Opposite P.Elements), Eq (CategoryTheory.CategoryStruct.comp ((Cat …
      ⊢ Eq m ((fun s => { app := fun X x => (s.ι.app { unop := P.elementsMk X x }).a …
    -/
    ext X x
    /-
      case w.h.h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
      A : CategoryTheory.Functor C ℰ
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Presheaf.functorToRepresentab …
      m : Quiver.Hom (CategoryTheory.Presheaf.coconeOfRepresentable P).pt s.pt
      hm : ∀ (j : Opposite P.Elements), Eq (CategoryTheory.CategoryStruct.comp ((Cat …
      X : Opposite C
      x : (CategoryTheory.Presheaf.coconeOfRepresentable P).pt.obj X
      ⊢ Eq (m.app X x) (((fun s => { app := fun X x => (s.ι.app { unop := P.elements …
    -/
    dsimp
    /-
      case w.h.h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
      A : CategoryTheory.Functor C ℰ
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Presheaf.functorToRepresentab …
      m : Quiver.Hom (CategoryTheory.Presheaf.coconeOfRepresentable P).pt s.pt
      hm : ∀ (j : Opposite P.Elements), Eq (CategoryTheory.CategoryStruct.comp ((Cat …
      X : Opposite C
      x : (CategoryTheory.Presheaf.coconeOfRepresentable P).pt.obj X
      ⊢ Eq (m.app X x) ((s.ι.app { unop := P.elementsMk X x }).app X (CategoryTheory …
    -/
    rw [← hm]
    /-
      case w.h.h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
      A : CategoryTheory.Functor C ℰ
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Presheaf.functorToRepresentab …
      m : Quiver.Hom (CategoryTheory.Presheaf.coconeOfRepresentable P).pt s.pt
      hm : ∀ (j : Opposite P.Elements), Eq (CategoryTheory.CategoryStruct.comp ((Cat …
      X : Opposite C
      x : (CategoryTheory.Presheaf.coconeOfRepresentable P).pt.obj X
      ⊢ Eq (m.app X x) ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Preshea …
    -/
    apply congr_arg
    /-
      case w.h.h.h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝ : CategoryTheory.Category.{v₁, u₂} ℰ
      A : CategoryTheory.Functor C ℰ
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Presheaf.functorToRepresentab …
      m : Quiver.Hom (CategoryTheory.Presheaf.coconeOfRepresentable P).pt s.pt
      hm : ∀ (j : Opposite P.Elements), Eq (CategoryTheory.CategoryStruct.comp ((Cat …
      X : Opposite C
      x : (CategoryTheory.Presheaf.coconeOfRepresentable P).pt.obj X
      ⊢ Eq x (((CategoryTheory.Presheaf.coconeOfRepresentable P).ι.app { unop := P.e …
    -/
    simp [coconeOfRepresentable_ι_app, yonedaEquiv]
    /-
      🎉 no goals
    -/


instance [L.IsLeftKanExtension α] : IsIso α :=
  (Functor.isPointwiseLeftKanExtensionOfIsLeftKanExtension L α).isIso_hom


lemma isLeftKanExtension_along_yoneda_iff :
    L.IsLeftKanExtension α ↔
      (IsIso α ∧ PreservesColimitsOfSize.{v₁, max u₁ v₁} L) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    ℰ : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} ℰ
    A : CategoryTheory.Functor C ℰ
    inst✝ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
    L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
    α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
    ⊢ Iff (L.IsLeftKanExtension α) (And (CategoryTheory.IsIso α) (CategoryTheory.L …
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} ℰ
      A : CategoryTheory.Functor C ℰ
      inst✝ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
      α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
      ⊢ L.IsLeftKanExtension α → And (CategoryTheory.IsIso α) (CategoryTheory.Limits …
    -/
  · intro
    exact ⟨inferInstance, preservesColimits_of_natIso
      (Functor.leftKanExtensionUnique _ (yoneda.leftKanExtensionUnit A) _ α)⟩
    /-
      case mpr
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} ℰ
      A : CategoryTheory.Functor C ℰ
      inst✝ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
      α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
      ⊢ And (CategoryTheory.IsIso α) (CategoryTheory.Limits.PreservesColimitsOfSize. …
    -/
  · rintro ⟨_, _⟩
    apply Functor.LeftExtension.IsPointwiseLeftKanExtension.isLeftKanExtension
      (E := Functor.LeftExtension.mk _ α)
    /-
      case mpr.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} ℰ
      A : CategoryTheory.Functor C ℰ
      inst✝ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
      α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
      left✝ : CategoryTheory.IsIso α
      right✝ : CategoryTheory.Limits.PreservesColimitsOfSize.{v₁, max u₁ v₁, max u₁  …
      ⊢ (CategoryTheory.Functor.LeftExtension.mk L α).IsPointwiseLeftKanExtension
    -/
    intro P
    /-
      case mpr.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} ℰ
      A : CategoryTheory.Functor C ℰ
      inst✝ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
      α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
      left✝ : CategoryTheory.IsIso α
      right✝ : CategoryTheory.Limits.PreservesColimitsOfSize.{v₁, max u₁ v₁, max u₁  …
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      ⊢ (CategoryTheory.Functor.LeftExtension.mk L α).IsPointwiseLeftKanExtensionAt P
    -/
    dsimp [Functor.LeftExtension.IsPointwiseLeftKanExtensionAt]
    /-
      case mpr.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} ℰ
      A : CategoryTheory.Functor C ℰ
      inst✝ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
      α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
      left✝ : CategoryTheory.IsIso α
      right✝ : CategoryTheory.Limits.PreservesColimitsOfSize.{v₁, max u₁ v₁, max u₁  …
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      ⊢ CategoryTheory.Limits.IsColimit ((CategoryTheory.Functor.LeftExtension.mk L  …
    -/
    apply IsColimit.ofWhiskerEquivalence (CategoryOfElements.costructuredArrowYonedaEquivalence _)
    let e : CategoryOfElements.toCostructuredArrow P ⋙ CostructuredArrow.proj yoneda P ⋙ A ≅
        functorToRepresentables P ⋙ L :=
      isoWhiskerLeft _ (isoWhiskerLeft _ (asIso α)) ≪≫
        isoWhiskerLeft _ (Functor.associator _ _ _).symm ≪≫
        (Functor.associator _ _ _).symm ≪≫ isoWhiskerRight (Iso.refl _) L
    /-
      case mpr.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} ℰ
      A : CategoryTheory.Functor C ℰ
      inst✝ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
      α : Quiver.Hom A (CategoryTheory.yoneda.comp L)
      left✝ : CategoryTheory.IsIso α
      right✝ : CategoryTheory.Limits.PreservesColimitsOfSize.{v₁, max u₁ v₁, max u₁  …
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      e : CategoryTheory.Iso ((CategoryTheory.CategoryOfElements.toCostructuredArrow …
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cocone.whisker (Categ …
    -/
    apply (IsColimit.precomposeHomEquiv e.symm _).1
    exact IsColimit.ofIsoColimit (isColimitOfPreserves L (colimitOfRepresentable P))
      (Cocones.ext (Iso.refl _))


lemma isLeftKanExtension_of_preservesColimits
    (L : (Cᵒᵖ ⥤ Type v₁) ⥤ ℰ) (e : A ≅ yoneda ⋙ L)
    [PreservesColimitsOfSize.{v₁, max u₁ v₁} L] :
    L.IsLeftKanExtension e.hom := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    ℰ : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
    A : CategoryTheory.Functor C ℰ
    inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
    L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
    e : CategoryTheory.Iso A (CategoryTheory.yoneda.comp L)
    inst✝ : CategoryTheory.Limits.PreservesColimitsOfSize.{v₁, max u₁ v₁, max u₁ v …
    ⊢ L.IsLeftKanExtension e.hom
  -/
  rw [isLeftKanExtension_along_yoneda_iff]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    ℰ : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
    A : CategoryTheory.Functor C ℰ
    inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
    L : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) ℰ
    e : CategoryTheory.Iso A (CategoryTheory.yoneda.comp L)
    inst✝ : CategoryTheory.Limits.PreservesColimitsOfSize.{v₁, max u₁ v₁, max u₁ v …
    ⊢ And (CategoryTheory.IsIso e.hom) (CategoryTheory.Limits.PreservesColimitsOfS …
  -/
  exact ⟨inferInstance, ⟨inferInstance⟩⟩
  /-
    🎉 no goals
  -/


/-- Show that `yoneda.leftKanExtension A` is the unique colimit-preserving
functor which extends `A` to the presheaf category.

The second part of [MM92], Chapter I, Section 5, Corollary 4.
See Property 3 of https://ncatlab.org/nlab/show/Yoneda+extension#properties.
-/
noncomputable def uniqueExtensionAlongYoneda (L : (Cᵒᵖ ⥤ Type v₁) ⥤ ℰ) (e : A ≅ yoneda ⋙ L)
    [PreservesColimitsOfSize.{v₁, max u₁ v₁} L] : L ≅ yoneda.leftKanExtension A :=
  have := isLeftKanExtension_of_preservesColimits L e
  Functor.leftKanExtensionUnique _ e.hom _ (yoneda.leftKanExtensionUnit A)


instance (L : (Cᵒᵖ ⥤ Type v₁) ⥤ ℰ) [PreservesColimitsOfSize.{v₁, max u₁ v₁} L]
    [yoneda.HasPointwiseLeftKanExtension (yoneda ⋙ L)] :
    L.IsLeftKanExtension (𝟙 _ : yoneda ⋙ L ⟶ _) :=
  isLeftKanExtension_of_preservesColimits _ (Iso.refl _)


/-- If `L` preserves colimits and `ℰ` has them, then it is a left adjoint. Note this is a (partial)
converse to `leftAdjointPreservesColimits`.
-/
lemma isLeftAdjoint_of_preservesColimits (L : (C ⥤ Type v₁) ⥤ ℰ)
    [PreservesColimitsOfSize.{v₁, max u₁ v₁} L]
    [yoneda.HasPointwiseLeftKanExtension
      (yoneda ⋙ (opOpEquivalence C).congrLeft.functor.comp L)] :
    L.IsLeftAdjoint :=
  ⟨_, ⟨((opOpEquivalence C).congrLeft.symm.toAdjunction.comp
    (yonedaAdjunction _ (𝟙 _))).ofNatIsoLeft ((opOpEquivalence C).congrLeft.invFunIdAssoc L)⟩⟩


instance (X : C) (Y : F.op.LeftExtension (yoneda.obj X)) :
    Unique (Functor.LeftExtension.mk _ (yonedaMap F X) ⟶ Y) where
  default := StructuredArrow.homMk
      (yonedaEquiv.symm (yonedaEquiv (F := F.op.comp Y.right) Y.hom)) (by
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
          A✝ A : CategoryTheory.Functor C ℰ
          inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₁, u₂} D
          F : CategoryTheory.Functor C D
          X : C
          Y : F.op.LeftExtension (CategoryTheory.yoneda.obj X)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LeftExtension …
        -/
        ext Z f
        /-
          case w.h.h
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          ℰ : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
          A✝ A : CategoryTheory.Functor C ℰ
          inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₁, u₂} D
          F : CategoryTheory.Functor C D
          X : C
          Y : F.op.LeftExtension (CategoryTheory.yoneda.obj X)
          Z : Opposite C
          f : ((CategoryTheory.Functor.fromPUnit (CategoryTheory.yoneda.obj X)).obj (Cat …
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LeftExtensio …
        -/
        simpa using congr_fun (Y.hom.naturality f.op).symm (𝟙 _))
        /-
          🎉 no goals
        -/
  uniq φ := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      X : C
      Y : F.op.LeftExtension (CategoryTheory.yoneda.obj X)
      φ : Quiver.Hom (CategoryTheory.Functor.LeftExtension.mk (CategoryTheory.yoneda …
      ⊢ Eq φ Inhabited.default
    -/
    ext1
    /-
      case h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      X : C
      Y : F.op.LeftExtension (CategoryTheory.yoneda.obj X)
      φ : Quiver.Hom (CategoryTheory.Functor.LeftExtension.mk (CategoryTheory.yoneda …
      ⊢ Eq φ.right Inhabited.default.right
    -/
    apply yonedaEquiv.injective
    /-
      case h.a
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      X : C
      Y : F.op.LeftExtension (CategoryTheory.yoneda.obj X)
      φ : Quiver.Hom (CategoryTheory.Functor.LeftExtension.mk (CategoryTheory.yoneda …
      ⊢ Eq (CategoryTheory.yonedaEquiv φ.right) (CategoryTheory.yonedaEquiv Inhabite …
    -/
    dsimp
    /-
      case h.a
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      X : C
      Y : F.op.LeftExtension (CategoryTheory.yoneda.obj X)
      φ : Quiver.Hom (CategoryTheory.Functor.LeftExtension.mk (CategoryTheory.yoneda …
      ⊢ Eq (CategoryTheory.yonedaEquiv φ.right) (CategoryTheory.yonedaEquiv (Categor …
    -/
    simp only [Equiv.apply_symm_apply, ← StructuredArrow.w φ]
    /-
      case h.a
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      X : C
      Y : F.op.LeftExtension (CategoryTheory.yoneda.obj X)
      φ : Quiver.Hom (CategoryTheory.Functor.LeftExtension.mk (CategoryTheory.yoneda …
      ⊢ Eq (CategoryTheory.yonedaEquiv φ.right) (CategoryTheory.yonedaEquiv (Categor …
    -/
    dsimp [yonedaEquiv]
    /-
      case h.a
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      X : C
      Y : F.op.LeftExtension (CategoryTheory.yoneda.obj X)
      φ : Quiver.Hom (CategoryTheory.Functor.LeftExtension.mk (CategoryTheory.yoneda …
      ⊢ Eq (φ.right.app { unop := F.obj X } (CategoryTheory.CategoryStruct.id (F.obj …
    -/
    simp only [yonedaMap_app_apply, Functor.map_id]
    /-
      🎉 no goals
    -/


/-- Given `F : C ⥤ D` and `X : C`, `yoneda.obj (F.obj X) : Dᵒᵖ ⥤ Type _` is the
left Kan extension of `yoneda.obj X : Cᵒᵖ ⥤ Type _` along `F.op`. -/
instance (X : C) : (yoneda.obj (F.obj X)).IsLeftKanExtension (yonedaMap F X) :=
  ⟨⟨Limits.IsInitial.ofUnique _⟩⟩


/-- `F ⋙ yoneda` is naturally isomorphic to `yoneda ⋙ F.op.lan`. -/
noncomputable def compYonedaIsoYonedaCompLan :
    F ⋙ yoneda ≅ yoneda ⋙ F.op.lan :=
  NatIso.ofComponents (fun X => Functor.leftKanExtensionUnique _
    (yonedaMap F X) (F.op.lan.obj _) (F.op.lanUnit.app (yoneda.obj X))) (fun {X Y} f => by
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        ℰ : Type u₂
        inst✝³ : CategoryTheory.Category.{v₁, u₂} ℰ
        A✝ A : CategoryTheory.Functor C ℰ
        inst✝² : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
        F : CategoryTheory.Functor C D
        inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
        X Y : C
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.comp CategoryTheory.yoneda).map f …
      -/
      apply yonedaEquiv.injective
      have eq₁ := congr_fun ((yoneda.obj (F.obj Y)).descOfIsLeftKanExtension_fac_app
        (yonedaMap F Y) (F.op.lan.obj (yoneda.obj Y)) (F.op.lanUnit.app (yoneda.obj Y)) _) f
      have eq₂ := congr_fun (((yoneda.obj (F.obj X)).descOfIsLeftKanExtension_fac_app
        (yonedaMap F X) (F.op.lan.obj (yoneda.obj X)) (F.op.lanUnit.app (yoneda.obj X))) _) (𝟙 _)
      /-
        case a
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        ℰ : Type u₂
        inst✝³ : CategoryTheory.Category.{v₁, u₂} ℰ
        A✝ A : CategoryTheory.Functor C ℰ
        inst✝² : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
        F : CategoryTheory.Functor C D
        inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
        X Y : C
        f : Quiver.Hom X Y
        eq₁ : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.yonedaMap F Y).a …
        eq₂ : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.yonedaMap F X).a …
        ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.CategoryStruct.comp ((F.comp  …
      -/
      have eq₃ := congr_fun (congr_app (F.op.lanUnit.naturality (yoneda.map f)) _) (𝟙 _)
      /-
        case a
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        ℰ : Type u₂
        inst✝³ : CategoryTheory.Category.{v₁, u₂} ℰ
        A✝ A : CategoryTheory.Functor C ℰ
        inst✝² : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
        F : CategoryTheory.Functor C D
        inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
        X Y : C
        f : Quiver.Hom X Y
        eq₁ : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.yonedaMap F Y).a …
        eq₂ : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.yonedaMap F X).a …
        eq₃ : Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Cat …
        ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.CategoryStruct.comp ((F.comp  …
      -/
      dsimp at eq₁ eq₂ eq₃
      /-
        case a
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        ℰ : Type u₂
        inst✝³ : CategoryTheory.Category.{v₁, u₂} ℰ
        A✝ A : CategoryTheory.Functor C ℰ
        inst✝² : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
        F : CategoryTheory.Functor C D
        inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
        X Y : C
        f : Quiver.Hom X Y
        eq₁ : Eq (((CategoryTheory.yoneda.obj (F.obj Y)).descOfIsLeftKanExtension (Cat …
        eq₂ : Eq (((CategoryTheory.yoneda.obj (F.obj X)).descOfIsLeftKanExtension (Cat …
        eq₃ : Eq ((F.op.lanUnit.app (CategoryTheory.yoneda.obj Y)).app { unop := X } ( …
        ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.CategoryStruct.comp ((F.comp  …
      -/
      simp only [Functor.map_id] at eq₂
      /-
        case a
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        ℰ : Type u₂
        inst✝³ : CategoryTheory.Category.{v₁, u₂} ℰ
        A✝ A : CategoryTheory.Functor C ℰ
        inst✝² : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
        F : CategoryTheory.Functor C D
        inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
        X Y : C
        f : Quiver.Hom X Y
        eq₁ : Eq (((CategoryTheory.yoneda.obj (F.obj Y)).descOfIsLeftKanExtension (Cat …
        eq₃ : Eq ((F.op.lanUnit.app (CategoryTheory.yoneda.obj Y)).app { unop := X } ( …
        eq₂ : Eq (((CategoryTheory.yoneda.obj (F.obj X)).descOfIsLeftKanExtension (Cat …
        ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.CategoryStruct.comp ((F.comp  …
      -/
      simp only [id_comp] at eq₃
      /-
        case a
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        ℰ : Type u₂
        inst✝³ : CategoryTheory.Category.{v₁, u₂} ℰ
        A✝ A : CategoryTheory.Functor C ℰ
        inst✝² : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
        F : CategoryTheory.Functor C D
        inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
        X Y : C
        f : Quiver.Hom X Y
        eq₁ : Eq (((CategoryTheory.yoneda.obj (F.obj Y)).descOfIsLeftKanExtension (Cat …
        eq₂ : Eq (((CategoryTheory.yoneda.obj (F.obj X)).descOfIsLeftKanExtension (Cat …
        eq₃ : Eq ((F.op.lanUnit.app (CategoryTheory.yoneda.obj Y)).app { unop := X } f …
        ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.CategoryStruct.comp ((F.comp  …
      -/
      simp [yonedaEquiv, eq₁, eq₂, eq₃])
      /-
        🎉 no goals
      -/


@[simp]
lemma compYonedaIsoYonedaCompLan_inv_app_app_apply_eq_id (X : C) :
    ((compYonedaIsoYonedaCompLan F).inv.app X).app (Opposite.op (F.obj X))
      ((F.op.lanUnit.app (yoneda.obj X)).app _ (𝟙 X)) = 𝟙 _ :=
  (congr_fun (Functor.descOfIsLeftKanExtension_fac_app _
                                                                                          /-
                                                                                            C : Type u₁
                                                                                            inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                                                            D : Type u₂
                                                                                            inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
                                                                                            F : CategoryTheory.Functor C D
                                                                                            inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
                                                                                            X : C
                                                                                            ⊢ Eq ((CategoryTheory.yonedaMap F X).app { unop := X } (CategoryTheory.Categor …
                                                                                          -/
    (F.op.lanUnit.app (yoneda.obj X)) _ (yonedaMap F X) (Opposite.op X)) (𝟙 _)).trans (by simp)
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


/-- Auxiliary definition for `presheafHom`. -/
def coconeApp {P : Cᵒᵖ ⥤ Type v₁} (x : P.Elements) :
    yoneda.obj x.1.unop ⟶ F.op ⋙ G.obj P := yonedaEquiv.symm
      ((G.map (yonedaEquiv.symm x.2)).app _ ((φ.app x.1.unop).app _ (𝟙 _)))


@[reassoc (attr := simp)]
lemma coconeApp_naturality {P : Cᵒᵖ ⥤ Type v₁} {x y : P.Elements} (f : x ⟶ y) :
    yoneda.map f.1.unop ≫ coconeApp φ x = coconeApp φ y := by
  have eq₁ : yoneda.map f.1.unop ≫ yonedaEquiv.symm x.2 = yonedaEquiv.symm y.2 :=
    yonedaEquiv.injective
      (by simpa only [Equiv.apply_symm_apply, ← yonedaEquiv_naturality] using f.2)
  have eq₂ := congr_fun ((G.map (yonedaEquiv.symm x.2)).naturality (F.map f.1.unop).op)
    ((φ.app x.1.unop).app _ (𝟙 _))
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) (Ca …
    φ : Quiver.Hom (F.comp CategoryTheory.yoneda) (CategoryTheory.yoneda.comp G)
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    x y : P.Elements
    f : Quiver.Hom x y
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map (↑f).u …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp ((G.obj (CategoryTheory.yoneda.ob …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map (↑f).unop) …
  -/
  have eq₃ := congr_fun (congr_app (φ.naturality f.1.unop) _) (𝟙 _)
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) (Ca …
    φ : Quiver.Hom (F.comp CategoryTheory.yoneda) (CategoryTheory.yoneda.comp G)
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    x y : P.Elements
    f : Quiver.Hom x y
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map (↑f).u …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp ((G.obj (CategoryTheory.yoneda.ob …
    eq₃ : Eq ((CategoryTheory.CategoryStruct.comp ((F.comp CategoryTheory.yoneda). …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map (↑f).unop) …
  -/
  have eq₄ := congr_fun ((φ.app x.1.unop).naturality (F.map f.1.unop).op)
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) (Ca …
    φ : Quiver.Hom (F.comp CategoryTheory.yoneda) (CategoryTheory.yoneda.comp G)
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    x y : P.Elements
    f : Quiver.Hom x y
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map (↑f).u …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp ((G.obj (CategoryTheory.yoneda.ob …
    eq₃ : Eq ((CategoryTheory.CategoryStruct.comp ((F.comp CategoryTheory.yoneda). …
    eq₄ : ∀ (a : ((F.comp CategoryTheory.yoneda).obj (Opposite.unop x.fst)).obj {  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map (↑f).unop) …
  -/
  dsimp at eq₂ eq₃ eq₄
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) (Ca …
    φ : Quiver.Hom (F.comp CategoryTheory.yoneda) (CategoryTheory.yoneda.comp G)
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    x y : P.Elements
    f : Quiver.Hom x y
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map (↑f).u …
    eq₂ : Eq ((G.map (CategoryTheory.yonedaEquiv.symm x.snd)).app { unop := F.obj  …
    eq₃ : Eq ((φ.app (Opposite.unop x.fst)).app { unop := F.toPrefunctor.1 (Opposi …
    eq₄ : ∀ (a : Quiver.Hom (F.obj (Opposite.unop x.fst)) (F.obj (Opposite.unop x. …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map (↑f).unop) …
  -/
  apply yonedaEquiv.injective
  /-
    case a
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) (Ca …
    φ : Quiver.Hom (F.comp CategoryTheory.yoneda) (CategoryTheory.yoneda.comp G)
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    x y : P.Elements
    f : Quiver.Hom x y
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map (↑f).u …
    eq₂ : Eq ((G.map (CategoryTheory.yonedaEquiv.symm x.snd)).app { unop := F.obj  …
    eq₃ : Eq ((φ.app (Opposite.unop x.fst)).app { unop := F.toPrefunctor.1 (Opposi …
    eq₄ : ∀ (a : Quiver.Hom (F.obj (Opposite.unop x.fst)) (F.obj (Opposite.unop x. …
    ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.CategoryStruct.comp (Category …
  -/
  dsimp only [coconeApp]
  /-
    case a
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) (Ca …
    φ : Quiver.Hom (F.comp CategoryTheory.yoneda) (CategoryTheory.yoneda.comp G)
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    x y : P.Elements
    f : Quiver.Hom x y
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map (↑f).u …
    eq₂ : Eq ((G.map (CategoryTheory.yonedaEquiv.symm x.snd)).app { unop := F.obj  …
    eq₃ : Eq ((φ.app (Opposite.unop x.fst)).app { unop := F.toPrefunctor.1 (Opposi …
    eq₄ : ∀ (a : Quiver.Hom (F.obj (Opposite.unop x.fst)) (F.obj (Opposite.unop x. …
    ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.CategoryStruct.comp (Category …
  -/
  rw [Equiv.apply_symm_apply, ← yonedaEquiv_naturality, Equiv.apply_symm_apply]
  /-
    case a
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) (Ca …
    φ : Quiver.Hom (F.comp CategoryTheory.yoneda) (CategoryTheory.yoneda.comp G)
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    x y : P.Elements
    f : Quiver.Hom x y
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map (↑f).u …
    eq₂ : Eq ((G.map (CategoryTheory.yonedaEquiv.symm x.snd)).app { unop := F.obj  …
    eq₃ : Eq ((φ.app (Opposite.unop x.fst)).app { unop := F.toPrefunctor.1 (Opposi …
    eq₄ : ∀ (a : Quiver.Hom (F.obj (Opposite.unop x.fst)) (F.obj (Opposite.unop x. …
    ⊢ Eq ((F.op.comp (G.obj P)).map (↑f).unop.op ((G.map (CategoryTheory.yonedaEqu …
  -/
  simp [← eq₁, ← eq₂, ← eq₃, ← eq₄, Functor.map_comp, FunctorToTypes.comp, id_comp, comp_id]
  /-
    🎉 no goals
  -/


/-- Given functors `F : C ⥤ D` and `G : (Cᵒᵖ ⥤ Type v₁) ⥤ (Dᵒᵖ ⥤ Type v₁)`, and
a natural transformation `φ : F ⋙ yoneda ⟶ yoneda ⋙ G`, this is the
(natural) morphism `P ⟶ F.op ⋙ G.obj P` for all `P : Cᵒᵖ ⥤ Type v₁` that is
determined by `φ`. -/
noncomputable def presheafHom (P : Cᵒᵖ ⥤ Type v₁) : P ⟶ F.op ⋙ G.obj P :=
  (colimitOfRepresentable P).desc
    (Cocone.mk _ { app := fun x => coconeApp φ x.unop })


lemma yonedaEquiv_ι_presheafHom (P : Cᵒᵖ ⥤ Type v₁) {X : C} (f : yoneda.obj X ⟶ P) :
    yonedaEquiv (f ≫ presheafHom φ P) =
      (G.map f).app (Opposite.op (F.obj X)) ((φ.app X).app _ (𝟙 _)) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) (Ca …
    φ : Quiver.Hom (F.comp CategoryTheory.yoneda) (CategoryTheory.yoneda.comp G)
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    X : C
    f : Quiver.Hom (CategoryTheory.yoneda.obj X) P
    ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.CategoryStruct.comp f (Catego …
  -/
  obtain ⟨x, rfl⟩ := yonedaEquiv.symm.surjective f
  /-
    case intro
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) (Ca …
    φ : Quiver.Hom (F.comp CategoryTheory.yoneda) (CategoryTheory.yoneda.comp G)
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    X : C
    x : P.obj { unop := X }
    ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.CategoryStruct.comp (Category …
  -/
  erw [(colimitOfRepresentable P).fac _ (Opposite.op (P.elementsMk _ x))]
  /-
    case intro
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) (Ca …
    φ : Quiver.Hom (F.comp CategoryTheory.yoneda) (CategoryTheory.yoneda.comp G)
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    X : C
    x : P.obj { unop := X }
    ⊢ Eq (CategoryTheory.yonedaEquiv ({ pt := { obj := fun X => (G.obj P).obj (F.o …
  -/
  dsimp only [coconeApp]
  /-
    case intro
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) (Ca …
    φ : Quiver.Hom (F.comp CategoryTheory.yoneda) (CategoryTheory.yoneda.comp G)
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    X : C
    x : P.obj { unop := X }
    ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.yonedaEquiv.symm ((G.map (Cat …
  -/
  apply Equiv.apply_symm_apply
  /-
    🎉 no goals
  -/


lemma yonedaEquiv_presheafHom_yoneda_obj (X : C) :
    yonedaEquiv (presheafHom φ (yoneda.obj X)) =
      ((φ.app X).app (F.op.obj (Opposite.op X)) (𝟙 _)) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) (Ca …
    φ : Quiver.Hom (F.comp CategoryTheory.yoneda) (CategoryTheory.yoneda.comp G)
    X : C
    ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.Presheaf.compYonedaIsoYonedaC …
  -/
  simpa using yonedaEquiv_ι_presheafHom φ (yoneda.obj X) (𝟙 _)
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma presheafHom_naturality {P Q : Cᵒᵖ ⥤ Type v₁} (f : P ⟶ Q) :
    presheafHom φ P ≫ whiskerLeft F.op (G.map f) = f ≫ presheafHom φ Q :=
  hom_ext_yoneda (fun X p => yonedaEquiv.injective (by
    rw [← assoc p f, yonedaEquiv_ι_presheafHom, ← assoc,
      yonedaEquiv_comp, yonedaEquiv_ι_presheafHom,
      whiskerLeft_app, Functor.map_comp, FunctorToTypes.comp]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) (Ca …
      φ : Quiver.Hom (F.comp CategoryTheory.yoneda) (CategoryTheory.yoneda.comp G)
      P Q : CategoryTheory.Functor (Opposite C) (Type v₁)
      f : Quiver.Hom P Q
      X : C
      p : Quiver.Hom (CategoryTheory.yoneda.obj X) P
      ⊢ Eq ((G.map f).app (F.op.obj { unop := X }) ((G.map p).app { unop := F.obj X  …
    -/
    dsimp))
    /-
      🎉 no goals
    -/


/-- Given functors `F : C ⥤ D` and `G : (Cᵒᵖ ⥤ Type v₁) ⥤ (Dᵒᵖ ⥤ Type v₁)`,
and a natural transformation `φ : F ⋙ yoneda ⟶ yoneda ⋙ G`, this is
the canonical natural transformation `F.op.lan ⟶ G`, which is part of the
that `F.op.lan : (Cᵒᵖ ⥤ Type v₁) ⥤ Dᵒᵖ ⥤ Type v₁` is the left Kan extension
of `F ⋙ yoneda : C ⥤ Dᵒᵖ ⥤ Type v₁` along `yoneda : C ⥤ Cᵒᵖ ⥤ Type v₁`. -/
noncomputable def natTrans : F.op.lan ⟶ G where
  app P := (F.op.lan.obj P).descOfIsLeftKanExtension (F.op.lanUnit.app P) _ (presheafHom φ P)
  naturality {P Q} f := by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C ℰ
      inst✝² : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      X : C
      G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) (Ca …
      φ : Quiver.Hom (F.comp CategoryTheory.yoneda) (CategoryTheory.yoneda.comp G)
      inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
      P Q : CategoryTheory.Functor (Opposite C) (Type v₁)
      f : Quiver.Hom P Q
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.op.lan.map f) ((fun P => (F.op.lan …
    -/
    apply (F.op.lan.obj P).hom_ext_of_isLeftKanExtension (F.op.lanUnit.app P)
    /-
      case hγ
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C ℰ
      inst✝² : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      X : C
      G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) (Ca …
      φ : Quiver.Hom (F.comp CategoryTheory.yoneda) (CategoryTheory.yoneda.comp G)
      inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
      P Q : CategoryTheory.Functor (Opposite C) (Type v₁)
      f : Quiver.Hom P Q
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.op.lanUnit.app P) (CategoryTheory. …
    -/
    have eq := F.op.lanUnit.naturality f
    /-
      case hγ
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C ℰ
      inst✝² : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      X : C
      G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) (Ca …
      φ : Quiver.Hom (F.comp CategoryTheory.yoneda) (CategoryTheory.yoneda.comp G)
      inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
      P Q : CategoryTheory.Functor (Opposite C) (Type v₁)
      f : Quiver.Hom P Q
      eq : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Categ …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.op.lanUnit.app P) (CategoryTheory. …
    -/
    dsimp at eq ⊢
    rw [Functor.descOfIsLeftKanExtension_fac_assoc, ← reassoc_of% eq,
      Functor.descOfIsLeftKanExtension_fac, presheafHom_naturality]


lemma natTrans_app_yoneda_obj (X : C) : (natTrans φ).app (yoneda.obj X) =
    (compYonedaIsoYonedaCompLan F).inv.app X ≫ φ.app X := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) (Ca …
    φ : Quiver.Hom (F.comp CategoryTheory.yoneda) (CategoryTheory.yoneda.comp G)
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
    X : C
    ⊢ Eq ((CategoryTheory.Presheaf.compYonedaIsoYonedaCompLan.natTrans φ).app (Cat …
  -/
  dsimp [natTrans]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) (Ca …
    φ : Quiver.Hom (F.comp CategoryTheory.yoneda) (CategoryTheory.yoneda.comp G)
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
    X : C
    ⊢ Eq ((F.op.lan.obj (CategoryTheory.yoneda.obj X)).descOfIsLeftKanExtension (F …
  -/
  apply (F.op.lan.obj (yoneda.obj X)).hom_ext_of_isLeftKanExtension (F.op.lanUnit.app _)
  /-
    case hγ
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) (Ca …
    φ : Quiver.Hom (F.comp CategoryTheory.yoneda) (CategoryTheory.yoneda.comp G)
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.op.lanUnit.app (CategoryTheory.yon …
  -/
  rw [Functor.descOfIsLeftKanExtension_fac]
  /-
    case hγ
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) (Ca …
    φ : Quiver.Hom (F.comp CategoryTheory.yoneda) (CategoryTheory.yoneda.comp G)
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
    X : C
    ⊢ Eq (CategoryTheory.Presheaf.compYonedaIsoYonedaCompLan.presheafHom φ (Catego …
  -/
  apply yonedaEquiv.injective
  /-
    case hγ.a
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) (Ca …
    φ : Quiver.Hom (F.comp CategoryTheory.yoneda) (CategoryTheory.yoneda.comp G)
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
    X : C
    ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.Presheaf.compYonedaIsoYonedaC …
  -/
  rw [yonedaEquiv_presheafHom_yoneda_obj]
  /-
    case hγ.a
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) (Type v₁)) (Ca …
    φ : Quiver.Hom (F.comp CategoryTheory.yoneda) (CategoryTheory.yoneda.comp G)
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
    X : C
    ⊢ Eq ((φ.app X).app (F.op.obj { unop := X }) (CategoryTheory.CategoryStruct.id …
  -/
  exact congr_arg _ (compYonedaIsoYonedaCompLan_inv_app_app_apply_eq_id F X).symm
  /-
    🎉 no goals
  -/


/-- Given a functor `F : C ⥤ D`, this definition is part of the verification that
`Functor.LeftExtension.mk F.op.lan (compYonedaIsoYonedaCompLan F).hom`
is universal, i.e. that  `F.op.lan : (Cᵒᵖ ⥤ Type v₁) ⥤ Dᵒᵖ ⥤ Type v₁` is the
left Kan extension of `F ⋙ yoneda : C ⥤ Dᵒᵖ ⥤ Type v₁`
along `yoneda : C ⥤ Cᵒᵖ ⥤ Type v₁`. -/
noncomputable def extensionHom (Φ : yoneda.LeftExtension (F ⋙ yoneda)) :
    Functor.LeftExtension.mk F.op.lan (compYonedaIsoYonedaCompLan F).hom ⟶ Φ :=
  StructuredArrow.homMk (natTrans Φ.hom) (by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C ℰ
      inst✝² : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
      Φ : CategoryTheory.yoneda.LeftExtension (F.comp CategoryTheory.yoneda)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LeftExtension …
    -/
    ext X : 2
    /-
      case w.h
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C ℰ
      inst✝² : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
      Φ : CategoryTheory.yoneda.LeftExtension (F.comp CategoryTheory.yoneda)
      X : C
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LeftExtensio …
    -/
    dsimp
    /-
      case w.h
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      ℰ : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C ℰ
      inst✝² : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
      Φ : CategoryTheory.yoneda.LeftExtension (F.comp CategoryTheory.yoneda)
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Presheaf.compYonedaI …
    -/
    rw [natTrans_app_yoneda_obj, Iso.hom_inv_id_app_assoc])
    /-
      🎉 no goals
    -/


@[ext]
lemma hom_ext {Φ : yoneda.LeftExtension (F ⋙ yoneda)}
    (f g : Functor.LeftExtension.mk F.op.lan (compYonedaIsoYonedaCompLan F).hom ⟶ Φ) :
    f = g := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
    Φ : CategoryTheory.yoneda.LeftExtension (F.comp CategoryTheory.yoneda)
    f g : Quiver.Hom (CategoryTheory.Functor.LeftExtension.mk F.op.lan (CategoryTh …
    ⊢ Eq f g
  -/
  ext P : 3
  /-
    case h.w.h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
    Φ : CategoryTheory.yoneda.LeftExtension (F.comp CategoryTheory.yoneda)
    f g : Quiver.Hom (CategoryTheory.Functor.LeftExtension.mk F.op.lan (CategoryTh …
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    ⊢ Eq (f.right.app P) (g.right.app P)
  -/
  apply (F.op.lan.obj P).hom_ext_of_isLeftKanExtension (F.op.lanUnit.app P)
  /-
    case h.w.h.hγ
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
    Φ : CategoryTheory.yoneda.LeftExtension (F.comp CategoryTheory.yoneda)
    f g : Quiver.Hom (CategoryTheory.Functor.LeftExtension.mk F.op.lan (CategoryTh …
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.op.lanUnit.app P) (CategoryTheory. …
  -/
  apply (colimitOfRepresentable P).hom_ext
  /-
    case h.w.h.hγ
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
    Φ : CategoryTheory.yoneda.LeftExtension (F.comp CategoryTheory.yoneda)
    f g : Quiver.Hom (CategoryTheory.Functor.LeftExtension.mk F.op.lan (CategoryTh …
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    ⊢ ∀ (j : Opposite P.Elements), Eq (CategoryTheory.CategoryStruct.comp ((Catego …
  -/
  intro x
  /-
    case h.w.h.hγ
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
    Φ : CategoryTheory.yoneda.LeftExtension (F.comp CategoryTheory.yoneda)
    f g : Quiver.Hom (CategoryTheory.Functor.LeftExtension.mk F.op.lan (CategoryTh …
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    x : Opposite P.Elements
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Presheaf.coconeOfRep …
  -/
  have eq := F.op.lanUnit.naturality (yonedaEquiv.symm x.unop.2)
  have eq₁ := congr_fun (congr_app (congr_app (StructuredArrow.w f) x.unop.1.unop)
    (F.op.obj x.unop.1)) (𝟙 _)
  have eq₂ := congr_fun (congr_app (congr_app (StructuredArrow.w g) x.unop.1.unop)
    (F.op.obj x.unop.1)) (𝟙 _)
  /-
    case h.w.h.hγ
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
    Φ : CategoryTheory.yoneda.LeftExtension (F.comp CategoryTheory.yoneda)
    f g : Quiver.Hom (CategoryTheory.Functor.LeftExtension.mk F.op.lan (CategoryTh …
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    x : Opposite P.Elements
    eq : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Categ …
    eq₁ : Eq (((CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LeftExt …
    eq₂ : Eq (((CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LeftExt …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Presheaf.coconeOfRep …
  -/
  dsimp at eq₁ eq₂ eq ⊢
  /-
    case h.w.h.hγ
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
    Φ : CategoryTheory.yoneda.LeftExtension (F.comp CategoryTheory.yoneda)
    f g : Quiver.Hom (CategoryTheory.Functor.LeftExtension.mk F.op.lan (CategoryTh …
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    x : Opposite P.Elements
    eq : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yonedaEquiv.symm ( …
    eq₁ : Eq ((f.right.app (CategoryTheory.yoneda.obj (Opposite.unop (Opposite.uno …
    eq₂ : Eq ((g.right.app (CategoryTheory.yoneda.obj (Opposite.unop (Opposite.uno …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yonedaEquiv.symm (Opp …
  -/
  simp only [reassoc_of% eq, ← whiskerLeft_comp]
  /-
    case h.w.h.hγ
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
    Φ : CategoryTheory.yoneda.LeftExtension (F.comp CategoryTheory.yoneda)
    f g : Quiver.Hom (CategoryTheory.Functor.LeftExtension.mk F.op.lan (CategoryTh …
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    x : Opposite P.Elements
    eq : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yonedaEquiv.symm ( …
    eq₁ : Eq ((f.right.app (CategoryTheory.yoneda.obj (Opposite.unop (Opposite.uno …
    eq₂ : Eq ((g.right.app (CategoryTheory.yoneda.obj (Opposite.unop (Opposite.uno …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.op.lanUnit.app (CategoryTheory.yon …
  -/
  congr 2
  simp only [← cancel_epi ((compYonedaIsoYonedaCompLan F).hom.app x.unop.1.unop),
    NatTrans.naturality]
  /-
    case h.w.h.hγ.e_a.e_α
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
    Φ : CategoryTheory.yoneda.LeftExtension (F.comp CategoryTheory.yoneda)
    f g : Quiver.Hom (CategoryTheory.Functor.LeftExtension.mk F.op.lan (CategoryTh …
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    x : Opposite P.Elements
    eq : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yonedaEquiv.symm ( …
    eq₁ : Eq ((f.right.app (CategoryTheory.yoneda.obj (Opposite.unop (Opposite.uno …
    eq₂ : Eq ((g.right.app (CategoryTheory.yoneda.obj (Opposite.unop (Opposite.uno …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Presheaf.compYonedaI …
  -/
  apply yonedaEquiv.injective
  /-
    case h.w.h.hγ.e_a.e_α.a
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
    Φ : CategoryTheory.yoneda.LeftExtension (F.comp CategoryTheory.yoneda)
    f g : Quiver.Hom (CategoryTheory.Functor.LeftExtension.mk F.op.lan (CategoryTh …
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    x : Opposite P.Elements
    eq : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yonedaEquiv.symm ( …
    eq₁ : Eq ((f.right.app (CategoryTheory.yoneda.obj (Opposite.unop (Opposite.uno …
    eq₂ : Eq ((g.right.app (CategoryTheory.yoneda.obj (Opposite.unop (Opposite.uno …
    ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.CategoryStruct.comp ((Categor …
  -/
  dsimp [yonedaEquiv_apply]
  /-
    case h.w.h.hγ.e_a.e_α.a
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : ∀ (P : CategoryTheory.Functor (Opposite C) (Type v₁)), F.op.HasLeftKan …
    Φ : CategoryTheory.yoneda.LeftExtension (F.comp CategoryTheory.yoneda)
    f g : Quiver.Hom (CategoryTheory.Functor.LeftExtension.mk F.op.lan (CategoryTh …
    P : CategoryTheory.Functor (Opposite C) (Type v₁)
    x : Opposite P.Elements
    eq : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yonedaEquiv.symm ( …
    eq₁ : Eq ((f.right.app (CategoryTheory.yoneda.obj (Opposite.unop (Opposite.uno …
    eq₂ : Eq ((g.right.app (CategoryTheory.yoneda.obj (Opposite.unop (Opposite.uno …
    ⊢ Eq ((Φ.right.map (CategoryTheory.yonedaEquiv.symm (Opposite.unop x).snd)).ap …
  -/
  rw [eq₁, eq₂]
  /-
    🎉 no goals
  -/


noncomputable instance (Φ : StructuredArrow (F ⋙ yoneda)
    ((whiskeringLeft C (Cᵒᵖ ⥤ Type v₁) (Dᵒᵖ ⥤ Type v₁)).obj yoneda)) :
    Unique (Functor.LeftExtension.mk F.op.lan (compYonedaIsoYonedaCompLan F).hom ⟶ Φ) where
  default := compYonedaIsoYonedaCompLan.extensionHom Φ
  uniq _ := compYonedaIsoYonedaCompLan.hom_ext _ _


/-- Given a functor `F : C ⥤ D`, `F.op.lan : (Cᵒᵖ ⥤ Type v₁) ⥤ Dᵒᵖ ⥤ Type v₁` is the
left Kan extension of `F ⋙ yoneda : C ⥤ Dᵒᵖ ⥤ Type v₁` along `yoneda : C ⥤ Cᵒᵖ ⥤ Type v₁`. -/
instance : F.op.lan.IsLeftKanExtension (compYonedaIsoYonedaCompLan F).hom :=
  ⟨⟨Limits.IsInitial.ofUnique _⟩⟩


/-- For a presheaf `P`, consider the forgetful functor from the category of representable
    presheaves over `P` to the category of presheaves. There is a tautological cocone over this
    functor whose leg for a natural transformation `V ⟶ P` with `V` representable is just that
    natural transformation. -/
@[simps]
def tautologicalCocone : Cocone (CostructuredArrow.proj yoneda P ⋙ yoneda) where
  pt := P
  ι := { app := fun X => X.hom }


/-- The tautological cocone with point `P` is a colimit cocone, exhibiting `P` as a colimit of
    representables.

    Proposition 2.6.3(i) in [Kashiwara2006] -/
def isColimitTautologicalCocone : IsColimit (tautologicalCocone P) where
  desc := fun s => by
    /-
      C✝ : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C✝ ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.CostructuredArrow.proj Categ …
      ⊢ Quiver.Hom (CategoryTheory.Presheaf.tautologicalCocone P).pt s.pt
    -/
    refine ⟨fun X t => yonedaEquiv (s.ι.app (CostructuredArrow.mk (yonedaEquiv.symm t))), ?_⟩
    /-
      C✝ : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C✝ ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.CostructuredArrow.proj Categ …
      ⊢ ∀ ⦃X Y : Opposite C⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
    -/
    intros X Y f
    /-
      C✝ : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C✝ ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.CostructuredArrow.proj Categ …
      X Y : Opposite C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Presheaf.tautologica …
    -/
    ext t
    /-
      case h
      C✝ : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C✝ ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.CostructuredArrow.proj Categ …
      X Y : Opposite C
      f : Quiver.Hom X Y
      t : (CategoryTheory.Presheaf.tautologicalCocone P).pt.obj X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Presheaf.tautologica …
    -/
    dsimp
    /-
      case h
      C✝ : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C✝ ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.CostructuredArrow.proj Categ …
      X Y : Opposite C
      f : Quiver.Hom X Y
      t : (CategoryTheory.Presheaf.tautologicalCocone P).pt.obj X
      ⊢ Eq (CategoryTheory.yonedaEquiv (s.ι.app (CategoryTheory.CostructuredArrow.mk …
    -/
    rw [yonedaEquiv_naturality', yonedaEquiv_symm_map]
    simpa using (s.ι.naturality
      (CostructuredArrow.homMk' (CostructuredArrow.mk (yonedaEquiv.symm t)) f.unop)).symm
  fac := by
    /-
      C✝ : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C✝ ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      ⊢ ∀ (s : CategoryTheory.Limits.Cocone ((CategoryTheory.CostructuredArrow.proj  …
    -/
    intro s t
    /-
      C✝ : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C✝ ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.CostructuredArrow.proj Categ …
      t : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Presheaf.tautologica …
    -/
    dsimp
    /-
      C✝ : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C✝ ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.CostructuredArrow.proj Categ …
      t : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P
      ⊢ Eq (CategoryTheory.CategoryStruct.comp t.hom { app := fun X t => CategoryThe …
    -/
    apply yonedaEquiv.injective
    /-
      case a
      C✝ : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C✝ ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.CostructuredArrow.proj Categ …
      t : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P
      ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.CategoryStruct.comp t.hom { a …
    -/
    rw [yonedaEquiv_comp]
    /-
      case a
      C✝ : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C✝ ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.CostructuredArrow.proj Categ …
      t : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P
      ⊢ Eq ({ app := fun X t => CategoryTheory.yonedaEquiv (s.ι.app (CategoryTheory. …
    -/
    dsimp only
    /-
      case a
      C✝ : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C✝ ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.CostructuredArrow.proj Categ …
      t : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P
      ⊢ Eq (CategoryTheory.yonedaEquiv (s.ι.app (CategoryTheory.CostructuredArrow.mk …
    -/
    rw [Equiv.symm_apply_apply]
    /-
      case a
      C✝ : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C✝ ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.CostructuredArrow.proj Categ …
      t : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P
      ⊢ Eq (CategoryTheory.yonedaEquiv (s.ι.app (CategoryTheory.CostructuredArrow.mk …
    -/
    rfl
    /-
      🎉 no goals
    -/
  uniq := by
    /-
      C✝ : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C✝ ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      ⊢ ∀ (s : CategoryTheory.Limits.Cocone ((CategoryTheory.CostructuredArrow.proj  …
    -/
    intro s j h
    /-
      C✝ : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C✝ ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.CostructuredArrow.proj Categ …
      j : Quiver.Hom (CategoryTheory.Presheaf.tautologicalCocone P).pt s.pt
      h : ∀ (j_1 : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P), Eq (Ca …
      ⊢ Eq j ((fun s => { app := fun X t => CategoryTheory.yonedaEquiv (s.ι.app (Cat …
    -/
    ext V x
    /-
      case w.h.h
      C✝ : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C✝ ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.CostructuredArrow.proj Categ …
      j : Quiver.Hom (CategoryTheory.Presheaf.tautologicalCocone P).pt s.pt
      h : ∀ (j_1 : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P), Eq (Ca …
      V : Opposite C
      x : (CategoryTheory.Presheaf.tautologicalCocone P).pt.obj V
      ⊢ Eq (j.app V x) (((fun s => { app := fun X t => CategoryTheory.yonedaEquiv (s …
    -/
    obtain ⟨t, rfl⟩ := yonedaEquiv.surjective x
    /-
      case w.h.h.intro
      C✝ : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C✝ ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.CostructuredArrow.proj Categ …
      j : Quiver.Hom (CategoryTheory.Presheaf.tautologicalCocone P).pt s.pt
      h : ∀ (j_1 : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P), Eq (Ca …
      V : Opposite C
      t : Quiver.Hom (CategoryTheory.yoneda.obj (Opposite.unop V)) (CategoryTheory.P …
      ⊢ Eq (j.app V (CategoryTheory.yonedaEquiv t)) (((fun s => { app := fun X t =>  …
    -/
    dsimp
    /-
      case w.h.h.intro
      C✝ : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C✝ ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.CostructuredArrow.proj Categ …
      j : Quiver.Hom (CategoryTheory.Presheaf.tautologicalCocone P).pt s.pt
      h : ∀ (j_1 : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P), Eq (Ca …
      V : Opposite C
      t : Quiver.Hom (CategoryTheory.yoneda.obj (Opposite.unop V)) (CategoryTheory.P …
      ⊢ Eq (j.app V (CategoryTheory.yonedaEquiv t)) (CategoryTheory.yonedaEquiv (s.ι …
    -/
    rw [Equiv.symm_apply_apply, ← yonedaEquiv_comp]
    /-
      case w.h.h.intro
      C✝ : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
      ℰ : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} ℰ
      A✝ A : CategoryTheory.Functor C✝ ℰ
      inst✝¹ : CategoryTheory.yoneda.HasPointwiseLeftKanExtension A
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      P : CategoryTheory.Functor (Opposite C) (Type v₁)
      s : CategoryTheory.Limits.Cocone ((CategoryTheory.CostructuredArrow.proj Categ …
      j : Quiver.Hom (CategoryTheory.Presheaf.tautologicalCocone P).pt s.pt
      h : ∀ (j_1 : CategoryTheory.CostructuredArrow CategoryTheory.yoneda P), Eq (Ca …
      V : Opposite C
      t : Quiver.Hom (CategoryTheory.yoneda.obj (Opposite.unop V)) (CategoryTheory.P …
      ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.CategoryStruct.comp t j)) (Ca …
    -/
    exact congr_arg _ (h (CostructuredArrow.mk t))
    /-
      🎉 no goals
    -/


/-- Given a functor `F : I ⥤ C`, a cocone `c` on `F ⋙ yoneda : I ⥤ Cᵒᵖ ⥤ Type v₁` induces a
    functor `I ⥤ CostructuredArrow yoneda c.pt` which maps `i : I` to the leg
    `yoneda.obj (F.obj i) ⟶ c.pt`. If `c` is a colimit cocone, then that functor is
    final.

    Proposition 2.6.3(ii) in [Kashiwara2006] -/
theorem final_toCostructuredArrow_comp_pre {c : Cocone (F ⋙ yoneda)} (hc : IsColimit c) :
    Functor.Final (c.toCostructuredArrow ⋙ CostructuredArrow.pre F yoneda c.pt) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    I : Type v₁
    inst✝ : CategoryTheory.SmallCategory I
    F : CategoryTheory.Functor I C
    c : CategoryTheory.Limits.Cocone (F.comp CategoryTheory.yoneda)
    hc : CategoryTheory.Limits.IsColimit c
    ⊢ (c.toCostructuredArrow.comp (CategoryTheory.CostructuredArrow.pre F Category …
  -/
  apply Functor.final_of_isTerminal_colimit_comp_yoneda

  suffices IsTerminal (colimit ((c.toCostructuredArrow ⋙ CostructuredArrow.pre F yoneda c.pt) ⋙
      CostructuredArrow.toOver yoneda c.pt)) by
    apply IsTerminal.isTerminalOfObj (overEquivPresheafCostructuredArrow c.pt).inverse
    apply IsTerminal.ofIso this
    refine ?_ ≪≫ (preservesColimitIso (overEquivPresheafCostructuredArrow c.pt).inverse _).symm
    apply HasColimit.isoOfNatIso
    exact isoWhiskerLeft _
      (CostructuredArrow.toOverCompOverEquivPresheafCostructuredArrow c.pt).isoCompInverse

  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    I : Type v₁
    inst✝ : CategoryTheory.SmallCategory I
    F : CategoryTheory.Functor I C
    c : CategoryTheory.Limits.Cocone (F.comp CategoryTheory.yoneda)
    hc : CategoryTheory.Limits.IsColimit c
    ⊢ CategoryTheory.Limits.IsTerminal (CategoryTheory.Limits.colimit ((c.toCostru …
  -/
  apply IsTerminal.ofIso Over.mkIdTerminal
  let isc : IsColimit ((Over.forget _).mapCocone _) := isColimitOfPreserves _
    (colimit.isColimit ((c.toCostructuredArrow ⋙ CostructuredArrow.pre F yoneda c.pt) ⋙
      CostructuredArrow.toOver yoneda c.pt))
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    I : Type v₁
    inst✝ : CategoryTheory.SmallCategory I
    F : CategoryTheory.Functor I C
    c : CategoryTheory.Limits.Cocone (F.comp CategoryTheory.yoneda)
    hc : CategoryTheory.Limits.IsColimit c
    isc : CategoryTheory.Limits.IsColimit ((CategoryTheory.Over.forget c.pt).mapCo …
    ⊢ CategoryTheory.Iso (CategoryTheory.Over.mk (CategoryTheory.CategoryStruct.id …
  -/
  exact Over.isoMk (hc.coconePointUniqueUpToIso isc) (hc.hom_ext fun i => by simp)
  /-
    🎉 no goals
  -/



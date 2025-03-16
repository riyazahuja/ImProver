/-- If `C` is a full subcategory of `C'` and `D` is a full subcategory of `D'`, then we can restrict
an adjunction `L' ⊣ R'` where `L' : C' ⥤ D'` and `R' : D' ⥤ C'` to `C` and `D`.
The construction here is slightly more general, in that `C` is required only to have a full and
faithful "inclusion" functor `iC : C ⥤ C'` (and similarly `iD : D ⥤ D'`) which commute (up to
natural isomorphism) with the proposed restrictions.
-/
noncomputable def restrictFullyFaithful : L ⊣ R :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun X Y =>
        calc
          (L.obj X ⟶ Y) ≃ (iD.obj (L.obj X) ⟶ iD.obj Y) := hiD.homEquiv
          _ ≃ (L'.obj (iC.obj X) ⟶ iD.obj Y) := Iso.homCongr (comm1.symm.app X) (Iso.refl _)
          _ ≃ (iC.obj X ⟶ R'.obj (iD.obj Y)) := adj.homEquiv _ _
          _ ≃ (iC.obj X ⟶ iC.obj (R.obj Y)) := Iso.homCongr (Iso.refl _) (comm2.app Y)
          _ ≃ (X ⟶ R.obj Y) := hiC.homEquiv.symm

      homEquiv_naturality_left_symm := fun {X' X Y} f g => by
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          C' : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} C'
          D' : Type u₄
          inst✝ : CategoryTheory.Category.{v₄, u₄} D'
          iC : CategoryTheory.Functor C C'
          iD : CategoryTheory.Functor D D'
          L' : CategoryTheory.Functor C' D'
          R' : CategoryTheory.Functor D' C'
          adj : CategoryTheory.Adjunction L' R'
          hiC : iC.FullyFaithful
          hiD : iD.FullyFaithful
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          comm1 : CategoryTheory.Iso (iC.comp L') (L.comp iD)
          comm2 : CategoryTheory.Iso (iD.comp R') (R.comp iC)
          X' X : C
          Y : D
          f : Quiver.Hom X' X
          g : Quiver.Hom X (R.obj Y)
          ⊢ Eq (((fun X Y => Trans.trans (Trans.trans (Trans.trans (Trans.trans hiD.homE …
        -/
        apply hiD.map_injective
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          C' : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} C'
          D' : Type u₄
          inst✝ : CategoryTheory.Category.{v₄, u₄} D'
          iC : CategoryTheory.Functor C C'
          iD : CategoryTheory.Functor D D'
          L' : CategoryTheory.Functor C' D'
          R' : CategoryTheory.Functor D' C'
          adj : CategoryTheory.Adjunction L' R'
          hiC : iC.FullyFaithful
          hiD : iD.FullyFaithful
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          comm1 : CategoryTheory.Iso (iC.comp L') (L.comp iD)
          comm2 : CategoryTheory.Iso (iD.comp R') (R.comp iC)
          X' X : C
          Y : D
          f : Quiver.Hom X' X
          g : Quiver.Hom X (R.obj Y)
          ⊢ Eq (iD.map (((fun X Y => Trans.trans (Trans.trans (Trans.trans (Trans.trans  …
        -/
        simpa [Trans.trans] using (comm1.inv.naturality_assoc f _).symm
        /-
          🎉 no goals
        -/
      homEquiv_naturality_right := fun {X Y' Y} f g => by
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          C' : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} C'
          D' : Type u₄
          inst✝ : CategoryTheory.Category.{v₄, u₄} D'
          iC : CategoryTheory.Functor C C'
          iD : CategoryTheory.Functor D D'
          L' : CategoryTheory.Functor C' D'
          R' : CategoryTheory.Functor D' C'
          adj : CategoryTheory.Adjunction L' R'
          hiC : iC.FullyFaithful
          hiD : iD.FullyFaithful
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          comm1 : CategoryTheory.Iso (iC.comp L') (L.comp iD)
          comm2 : CategoryTheory.Iso (iD.comp R') (R.comp iC)
          X : C
          Y' Y : D
          f : Quiver.Hom (L.obj X) Y'
          g : Quiver.Hom Y' Y
          ⊢ Eq (((fun X Y => Trans.trans (Trans.trans (Trans.trans (Trans.trans hiD.homE …
        -/
        apply hiC.map_injective
        suffices R'.map (iD.map g) ≫ comm2.hom.app Y = comm2.hom.app Y' ≫ iC.map (R.map g) by
          simp [Trans.trans, this]
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          C' : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} C'
          D' : Type u₄
          inst✝ : CategoryTheory.Category.{v₄, u₄} D'
          iC : CategoryTheory.Functor C C'
          iD : CategoryTheory.Functor D D'
          L' : CategoryTheory.Functor C' D'
          R' : CategoryTheory.Functor D' C'
          adj : CategoryTheory.Adjunction L' R'
          hiC : iC.FullyFaithful
          hiD : iD.FullyFaithful
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          comm1 : CategoryTheory.Iso (iC.comp L') (L.comp iD)
          comm2 : CategoryTheory.Iso (iD.comp R') (R.comp iC)
          X : C
          Y' Y : D
          f : Quiver.Hom (L.obj X) Y'
          g : Quiver.Hom Y' Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (R'.map (iD.map g)) (comm2.hom.app Y) …
        -/
        apply comm2.hom.naturality g }
        /-
          🎉 no goals
        -/


@[simp, reassoc]
lemma map_restrictFullyFaithful_unit_app (X : C) :
    iC.map ((adj.restrictFullyFaithful hiC hiD comm1 comm2).unit.app X) =
    adj.unit.app (iC.obj X) ≫ R'.map (comm1.hom.app X) ≫ comm2.hom.app (L.obj X) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    C' : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C'
    D' : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D'
    iC : CategoryTheory.Functor C C'
    iD : CategoryTheory.Functor D D'
    L' : CategoryTheory.Functor C' D'
    R' : CategoryTheory.Functor D' C'
    adj : CategoryTheory.Adjunction L' R'
    hiC : iC.FullyFaithful
    hiD : iD.FullyFaithful
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    comm1 : CategoryTheory.Iso (iC.comp L') (L.comp iD)
    comm2 : CategoryTheory.Iso (iD.comp R') (R.comp iC)
    X : C
    ⊢ Eq (iC.map ((adj.restrictFullyFaithful hiC hiD comm1 comm2).unit.app X)) (Ca …
  -/
  simp [restrictFullyFaithful]
  /-
    🎉 no goals
  -/


@[simp, reassoc]
lemma map_restrictFullyFaithful_counit_app (X : D) :
    iD.map ((adj.restrictFullyFaithful hiC hiD comm1 comm2).counit.app X) =
    comm1.inv.app (R.obj X) ≫ L'.map (comm2.inv.app X) ≫ adj.counit.app (iD.obj X) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    C' : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C'
    D' : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D'
    iC : CategoryTheory.Functor C C'
    iD : CategoryTheory.Functor D D'
    L' : CategoryTheory.Functor C' D'
    R' : CategoryTheory.Functor D' C'
    adj : CategoryTheory.Adjunction L' R'
    hiC : iC.FullyFaithful
    hiD : iD.FullyFaithful
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    comm1 : CategoryTheory.Iso (iC.comp L') (L.comp iD)
    comm2 : CategoryTheory.Iso (iD.comp R') (R.comp iC)
    X : D
    ⊢ Eq (iD.map ((adj.restrictFullyFaithful hiC hiD comm1 comm2).counit.app X)) ( …
  -/
  dsimp [restrictFullyFaithful]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    C' : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C'
    D' : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D'
    iC : CategoryTheory.Functor C C'
    iD : CategoryTheory.Functor D D'
    L' : CategoryTheory.Functor C' D'
    R' : CategoryTheory.Functor D' C'
    adj : CategoryTheory.Adjunction L' R'
    hiC : iC.FullyFaithful
    hiD : iD.FullyFaithful
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    comm1 : CategoryTheory.Iso (iC.comp L') (L.comp iD)
    comm2 : CategoryTheory.Iso (iD.comp R') (R.comp iC)
    X : D
    ⊢ Eq (iD.map (hiD.preimage (CategoryTheory.CategoryStruct.comp (comm1.inv.app  …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma restrictFullyFaithful_homEquiv_apply {X : C} {Y : D} (f : L.obj X ⟶ Y) :
    (adj.restrictFullyFaithful hiC hiD comm1 comm2).homEquiv X Y f =
      hiC.preimage (adj.unit.app (iC.obj X) ≫ R'.map (comm1.hom.app X) ≫
        R'.map (iD.map f) ≫ comm2.hom.app Y) := by
  -- This proof was just `simp [restrictFullyFaithful]` before https://github.com/leanprover-community/mathlib4/pull/16317
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    C' : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C'
    D' : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D'
    iC : CategoryTheory.Functor C C'
    iD : CategoryTheory.Functor D D'
    L' : CategoryTheory.Functor C' D'
    R' : CategoryTheory.Functor D' C'
    adj : CategoryTheory.Adjunction L' R'
    hiC : iC.FullyFaithful
    hiD : iD.FullyFaithful
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    comm1 : CategoryTheory.Iso (iC.comp L') (L.comp iD)
    comm2 : CategoryTheory.Iso (iD.comp R') (R.comp iC)
    X : C
    Y : D
    f : Quiver.Hom (L.obj X) Y
    ⊢ Eq (((adj.restrictFullyFaithful hiC hiD comm1 comm2).homEquiv X Y) f) (hiC.p …
  -/
  apply hiC.map_injective
  simp only [homEquiv_apply, Functor.comp_obj, Functor.map_comp, map_restrictFullyFaithful_unit_app,
    Functor.id_obj, assoc, Functor.FullyFaithful.map_preimage]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    C' : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C'
    D' : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D'
    iC : CategoryTheory.Functor C C'
    iD : CategoryTheory.Functor D D'
    L' : CategoryTheory.Functor C' D'
    R' : CategoryTheory.Functor D' C'
    adj : CategoryTheory.Adjunction L' R'
    hiC : iC.FullyFaithful
    hiD : iD.FullyFaithful
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    comm1 : CategoryTheory.Iso (iC.comp L') (L.comp iD)
    comm2 : CategoryTheory.Iso (iD.comp R') (R.comp iC)
    X : C
    Y : D
    f : Quiver.Hom (L.obj X) Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app (iC.obj X)) (CategoryTh …
  -/
  congr 2
  /-
    case e_a.e_a
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    C' : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C'
    D' : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D'
    iC : CategoryTheory.Functor C C'
    iD : CategoryTheory.Functor D D'
    L' : CategoryTheory.Functor C' D'
    R' : CategoryTheory.Functor D' C'
    adj : CategoryTheory.Adjunction L' R'
    hiC : iC.FullyFaithful
    hiD : iD.FullyFaithful
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    comm1 : CategoryTheory.Iso (iC.comp L') (L.comp iD)
    comm2 : CategoryTheory.Iso (iD.comp R') (R.comp iC)
    X : C
    Y : D
    f : Quiver.Hom (L.obj X) Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (comm2.hom.app (L.obj X)) (iC.map (R. …
  -/
  exact (comm2.hom.naturality _).symm
  /-
    🎉 no goals
  -/



/-- A functor preserves monomorphisms if it maps monomorphisms to monomorphisms. -/
class PreservesMonomorphisms (F : C ⥤ D) : Prop where
  /-- A functor preserves monomorphisms if it maps monomorphisms to monomorphisms. -/
  preserves : ∀ {X Y : C} (f : X ⟶ Y) [Mono f], Mono (F.map f)


instance map_mono (F : C ⥤ D) [PreservesMonomorphisms F] {X Y : C} (f : X ⟶ Y) [Mono f] :
    Mono (F.map f) :=
  PreservesMonomorphisms.preserves f


/-- A functor preserves epimorphisms if it maps epimorphisms to epimorphisms. -/
class PreservesEpimorphisms (F : C ⥤ D) : Prop where
  /-- A functor preserves epimorphisms if it maps epimorphisms to epimorphisms. -/
  preserves : ∀ {X Y : C} (f : X ⟶ Y) [Epi f], Epi (F.map f)


instance map_epi (F : C ⥤ D) [PreservesEpimorphisms F] {X Y : C} (f : X ⟶ Y) [Epi f] :
    Epi (F.map f) :=
  PreservesEpimorphisms.preserves f


/-- A functor reflects monomorphisms if morphisms that are mapped to monomorphisms are themselves
    monomorphisms. -/
class ReflectsMonomorphisms (F : C ⥤ D) : Prop where
   /-- A functor reflects monomorphisms if morphisms that are mapped to monomorphisms are themselves
    monomorphisms. -/
  reflects : ∀ {X Y : C} (f : X ⟶ Y), Mono (F.map f) → Mono f


theorem mono_of_mono_map (F : C ⥤ D) [ReflectsMonomorphisms F] {X Y : C} {f : X ⟶ Y}
    (h : Mono (F.map f)) : Mono f :=
  ReflectsMonomorphisms.reflects f h


/-- A functor reflects epimorphisms if morphisms that are mapped to epimorphisms are themselves
    epimorphisms. -/
class ReflectsEpimorphisms (F : C ⥤ D) : Prop where
  /-- A functor reflects epimorphisms if morphisms that are mapped to epimorphisms are themselves
      epimorphisms. -/
  reflects : ∀ {X Y : C} (f : X ⟶ Y), Epi (F.map f) → Epi f


theorem epi_of_epi_map (F : C ⥤ D) [ReflectsEpimorphisms F] {X Y : C} {f : X ⟶ Y}
    (h : Epi (F.map f)) : Epi f :=
  ReflectsEpimorphisms.reflects f h


instance preservesMonomorphisms_comp (F : C ⥤ D) (G : D ⥤ E) [PreservesMonomorphisms F]
    [PreservesMonomorphisms G] : PreservesMonomorphisms (F ⋙ G) where
  preserves f h := by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      inst✝¹ : F.PreservesMonomorphisms
      inst✝ : G.PreservesMonomorphisms
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      h : CategoryTheory.Mono f
      ⊢ CategoryTheory.Mono ((F.comp G).map f)
    -/
    rw [comp_map]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      inst✝¹ : F.PreservesMonomorphisms
      inst✝ : G.PreservesMonomorphisms
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      h : CategoryTheory.Mono f
      ⊢ CategoryTheory.Mono (G.map (F.map f))
    -/
    exact inferInstance
    /-
      🎉 no goals
    -/


instance preservesEpimorphisms_comp (F : C ⥤ D) (G : D ⥤ E) [PreservesEpimorphisms F]
    [PreservesEpimorphisms G] : PreservesEpimorphisms (F ⋙ G) where
  preserves f h := by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      inst✝¹ : F.PreservesEpimorphisms
      inst✝ : G.PreservesEpimorphisms
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      h : CategoryTheory.Epi f
      ⊢ CategoryTheory.Epi ((F.comp G).map f)
    -/
    rw [comp_map]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      inst✝¹ : F.PreservesEpimorphisms
      inst✝ : G.PreservesEpimorphisms
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      h : CategoryTheory.Epi f
      ⊢ CategoryTheory.Epi (G.map (F.map f))
    -/
    exact inferInstance
    /-
      🎉 no goals
    -/


instance reflectsMonomorphisms_comp (F : C ⥤ D) (G : D ⥤ E) [ReflectsMonomorphisms F]
    [ReflectsMonomorphisms G] : ReflectsMonomorphisms (F ⋙ G) where
  reflects _ h := F.mono_of_mono_map (G.mono_of_mono_map h)


instance reflectsEpimorphisms_comp (F : C ⥤ D) (G : D ⥤ E) [ReflectsEpimorphisms F]
    [ReflectsEpimorphisms G] : ReflectsEpimorphisms (F ⋙ G) where
  reflects _ h := F.epi_of_epi_map (G.epi_of_epi_map h)


theorem preservesEpimorphisms_of_preserves_of_reflects (F : C ⥤ D) (G : D ⥤ E)
    [PreservesEpimorphisms (F ⋙ G)] [ReflectsEpimorphisms G] : PreservesEpimorphisms F :=
  ⟨fun f _ => G.epi_of_epi_map <| show Epi ((F ⋙ G).map f) from inferInstance⟩


theorem preservesMonomorphisms_of_preserves_of_reflects (F : C ⥤ D) (G : D ⥤ E)
    [PreservesMonomorphisms (F ⋙ G)] [ReflectsMonomorphisms G] : PreservesMonomorphisms F :=
  ⟨fun f _ => G.mono_of_mono_map <| show Mono ((F ⋙ G).map f) from inferInstance⟩


theorem reflectsEpimorphisms_of_preserves_of_reflects (F : C ⥤ D) (G : D ⥤ E)
    [PreservesEpimorphisms G] [ReflectsEpimorphisms (F ⋙ G)] : ReflectsEpimorphisms F :=
  ⟨fun f _ => (F ⋙ G).epi_of_epi_map <| show Epi (G.map (F.map f)) from inferInstance⟩


theorem reflectsMonomorphisms_of_preserves_of_reflects (F : C ⥤ D) (G : D ⥤ E)
    [PreservesMonomorphisms G] [ReflectsMonomorphisms (F ⋙ G)] : ReflectsMonomorphisms F :=
  ⟨fun f _ => (F ⋙ G).mono_of_mono_map <| show Mono (G.map (F.map f)) from inferInstance⟩


theorem preservesMonomorphisms.of_iso {F G : C ⥤ D} [PreservesMonomorphisms F] (α : F ≅ G) :
    PreservesMonomorphisms G :=
  { preserves := fun {X} {Y} f h => by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        F G : CategoryTheory.Functor C D
        inst✝ : F.PreservesMonomorphisms
        α : CategoryTheory.Iso F G
        X Y : C
        f : Quiver.Hom X Y
        h : CategoryTheory.Mono f
        ⊢ CategoryTheory.Mono (G.map f)
      -/
      suffices G.map f = (α.app X).inv ≫ F.map f ≫ (α.app Y).hom from this ▸ mono_comp _ _
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        F G : CategoryTheory.Functor C D
        inst✝ : F.PreservesMonomorphisms
        α : CategoryTheory.Iso F G
        X Y : C
        f : Quiver.Hom X Y
        h : CategoryTheory.Mono f
        ⊢ Eq (G.map f) (CategoryTheory.CategoryStruct.comp (α.app X).inv (CategoryTheo …
      -/
      rw [Iso.eq_inv_comp, Iso.app_hom, Iso.app_hom, NatTrans.naturality] }
      /-
        🎉 no goals
      -/


theorem preservesMonomorphisms.iso_iff {F G : C ⥤ D} (α : F ≅ G) :
    PreservesMonomorphisms F ↔ PreservesMonomorphisms G :=
  ⟨fun _ => preservesMonomorphisms.of_iso α, fun _ => preservesMonomorphisms.of_iso α.symm⟩


theorem preservesEpimorphisms.of_iso {F G : C ⥤ D} [PreservesEpimorphisms F] (α : F ≅ G) :
    PreservesEpimorphisms G :=
  { preserves := fun {X} {Y} f h => by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        F G : CategoryTheory.Functor C D
        inst✝ : F.PreservesEpimorphisms
        α : CategoryTheory.Iso F G
        X Y : C
        f : Quiver.Hom X Y
        h : CategoryTheory.Epi f
        ⊢ CategoryTheory.Epi (G.map f)
      -/
      suffices G.map f = (α.app X).inv ≫ F.map f ≫ (α.app Y).hom from this ▸ epi_comp _ _
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        F G : CategoryTheory.Functor C D
        inst✝ : F.PreservesEpimorphisms
        α : CategoryTheory.Iso F G
        X Y : C
        f : Quiver.Hom X Y
        h : CategoryTheory.Epi f
        ⊢ Eq (G.map f) (CategoryTheory.CategoryStruct.comp (α.app X).inv (CategoryTheo …
      -/
      rw [Iso.eq_inv_comp, Iso.app_hom, Iso.app_hom, NatTrans.naturality] }
      /-
        🎉 no goals
      -/


theorem preservesEpimorphisms.iso_iff {F G : C ⥤ D} (α : F ≅ G) :
    PreservesEpimorphisms F ↔ PreservesEpimorphisms G :=
  ⟨fun _ => preservesEpimorphisms.of_iso α, fun _ => preservesEpimorphisms.of_iso α.symm⟩


theorem reflectsMonomorphisms.of_iso {F G : C ⥤ D} [ReflectsMonomorphisms F] (α : F ≅ G) :
    ReflectsMonomorphisms G :=
  { reflects := fun {X} {Y} f h => by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        F G : CategoryTheory.Functor C D
        inst✝ : F.ReflectsMonomorphisms
        α : CategoryTheory.Iso F G
        X Y : C
        f : Quiver.Hom X Y
        h : CategoryTheory.Mono (G.map f)
        ⊢ CategoryTheory.Mono f
      -/
      apply F.mono_of_mono_map
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        F G : CategoryTheory.Functor C D
        inst✝ : F.ReflectsMonomorphisms
        α : CategoryTheory.Iso F G
        X Y : C
        f : Quiver.Hom X Y
        h : CategoryTheory.Mono (G.map f)
        ⊢ CategoryTheory.Mono (F.map f)
      -/
      suffices F.map f = (α.app X).hom ≫ G.map f ≫ (α.app Y).inv from this ▸ mono_comp _ _
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        F G : CategoryTheory.Functor C D
        inst✝ : F.ReflectsMonomorphisms
        α : CategoryTheory.Iso F G
        X Y : C
        f : Quiver.Hom X Y
        h : CategoryTheory.Mono (G.map f)
        ⊢ Eq (F.map f) (CategoryTheory.CategoryStruct.comp (α.app X).hom (CategoryTheo …
      -/
      rw [← Category.assoc, Iso.eq_comp_inv, Iso.app_hom, Iso.app_hom, NatTrans.naturality] }
      /-
        🎉 no goals
      -/


theorem reflectsMonomorphisms.iso_iff {F G : C ⥤ D} (α : F ≅ G) :
    ReflectsMonomorphisms F ↔ ReflectsMonomorphisms G :=
  ⟨fun _ => reflectsMonomorphisms.of_iso α, fun _ => reflectsMonomorphisms.of_iso α.symm⟩


theorem reflectsEpimorphisms.of_iso {F G : C ⥤ D} [ReflectsEpimorphisms F] (α : F ≅ G) :
    ReflectsEpimorphisms G :=
  { reflects := fun {X} {Y} f h => by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        F G : CategoryTheory.Functor C D
        inst✝ : F.ReflectsEpimorphisms
        α : CategoryTheory.Iso F G
        X Y : C
        f : Quiver.Hom X Y
        h : CategoryTheory.Epi (G.map f)
        ⊢ CategoryTheory.Epi f
      -/
      apply F.epi_of_epi_map
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        F G : CategoryTheory.Functor C D
        inst✝ : F.ReflectsEpimorphisms
        α : CategoryTheory.Iso F G
        X Y : C
        f : Quiver.Hom X Y
        h : CategoryTheory.Epi (G.map f)
        ⊢ CategoryTheory.Epi (F.map f)
      -/
      suffices F.map f = (α.app X).hom ≫ G.map f ≫ (α.app Y).inv from this ▸ epi_comp _ _
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        F G : CategoryTheory.Functor C D
        inst✝ : F.ReflectsEpimorphisms
        α : CategoryTheory.Iso F G
        X Y : C
        f : Quiver.Hom X Y
        h : CategoryTheory.Epi (G.map f)
        ⊢ Eq (F.map f) (CategoryTheory.CategoryStruct.comp (α.app X).hom (CategoryTheo …
      -/
      rw [← Category.assoc, Iso.eq_comp_inv, Iso.app_hom, Iso.app_hom, NatTrans.naturality] }
      /-
        🎉 no goals
      -/


theorem reflectsEpimorphisms.iso_iff {F G : C ⥤ D} (α : F ≅ G) :
    ReflectsEpimorphisms F ↔ ReflectsEpimorphisms G :=
  ⟨fun _ => reflectsEpimorphisms.of_iso α, fun _ => reflectsEpimorphisms.of_iso α.symm⟩


theorem preservesEpimorphsisms_of_adjunction {F : C ⥤ D} {G : D ⥤ C} (adj : F ⊣ G) :
    PreservesEpimorphisms F :=
  { preserves := fun {X} {Y} f hf =>
      ⟨by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          G : CategoryTheory.Functor D C
          adj : CategoryTheory.Adjunction F G
          X Y : C
          f : Quiver.Hom X Y
          hf : CategoryTheory.Epi f
          ⊢ ∀ {Z : D} (g h : Quiver.Hom (F.obj Y) Z), Eq (CategoryTheory.CategoryStruct. …
        -/
        intro Z g h H
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          G : CategoryTheory.Functor D C
          adj : CategoryTheory.Adjunction F G
          X Y : C
          f : Quiver.Hom X Y
          hf : CategoryTheory.Epi f
          Z : D
          g h : Quiver.Hom (F.obj Y) Z
          H : Eq (CategoryTheory.CategoryStruct.comp (F.map f) g) (CategoryTheory.Catego …
          ⊢ Eq g h
        -/
        replace H := congr_arg (adj.homEquiv X Z) H
        rwa [adj.homEquiv_naturality_left, adj.homEquiv_naturality_left, cancel_epi,
          Equiv.apply_eq_iff_eq] at H⟩ }


instance (priority := 100) preservesEpimorphisms_of_isLeftAdjoint (F : C ⥤ D) [IsLeftAdjoint F] :
    PreservesEpimorphisms F :=
  preservesEpimorphsisms_of_adjunction (Adjunction.ofIsLeftAdjoint F)


theorem preservesMonomorphisms_of_adjunction {F : C ⥤ D} {G : D ⥤ C} (adj : F ⊣ G) :
    PreservesMonomorphisms G :=
  { preserves := fun {X} {Y} f hf =>
      ⟨by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          G : CategoryTheory.Functor D C
          adj : CategoryTheory.Adjunction F G
          X Y : D
          f : Quiver.Hom X Y
          hf : CategoryTheory.Mono f
          ⊢ ∀ {Z : C} (g h : Quiver.Hom Z (G.obj X)), Eq (CategoryTheory.CategoryStruct. …
        -/
        intro Z g h H
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          G : CategoryTheory.Functor D C
          adj : CategoryTheory.Adjunction F G
          X Y : D
          f : Quiver.Hom X Y
          hf : CategoryTheory.Mono f
          Z : C
          g h : Quiver.Hom Z (G.obj X)
          H : Eq (CategoryTheory.CategoryStruct.comp g (G.map f)) (CategoryTheory.Catego …
          ⊢ Eq g h
        -/
        replace H := congr_arg (adj.homEquiv Z Y).symm H
        rwa [adj.homEquiv_naturality_right_symm, adj.homEquiv_naturality_right_symm, cancel_mono,
          Equiv.apply_eq_iff_eq] at H⟩ }


instance (priority := 100) preservesMonomorphisms_of_isRightAdjoint (F : C ⥤ D) [IsRightAdjoint F] :
    PreservesMonomorphisms F :=
  preservesMonomorphisms_of_adjunction (Adjunction.ofIsRightAdjoint F)


instance (priority := 100) reflectsMonomorphisms_of_faithful (F : C ⥤ D) [Faithful F] :
    ReflectsMonomorphisms F where
  reflects {X} {Y} f _ :=
    ⟨fun {Z} g h hgh =>
                                                     /-
                                                       C : Type u₁
                                                       inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                                       D : Type u₂
                                                       inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                                       E : Type u₃
                                                       inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
                                                       F : CategoryTheory.Functor C D
                                                       inst✝ : F.Faithful
                                                       X Y : C
                                                       f : Quiver.Hom X Y
                                                       x✝ : CategoryTheory.Mono (F.map f)
                                                       Z : C
                                                       g h : Quiver.Hom Z X
                                                       hgh : Eq (CategoryTheory.CategoryStruct.comp g f) (CategoryTheory.CategoryStru …
                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map g) (F.map f)) (CategoryTheory. …
                                                     -/
      F.map_injective ((cancel_mono (F.map f)).1 (by rw [← F.map_comp, hgh, F.map_comp]))⟩
                                                     /-
                                                       🎉 no goals
                                                     -/


instance (priority := 100) reflectsEpimorphisms_of_faithful (F : C ⥤ D) [Faithful F] :
    ReflectsEpimorphisms F where
  reflects {X} {Y} f _ :=
    ⟨fun {Z} g h hgh =>
                                                    /-
                                                      C : Type u₁
                                                      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                                      D : Type u₂
                                                      inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                                      E : Type u₃
                                                      inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
                                                      F : CategoryTheory.Functor C D
                                                      inst✝ : F.Faithful
                                                      X Y : C
                                                      f : Quiver.Hom X Y
                                                      x✝ : CategoryTheory.Epi (F.map f)
                                                      Z : C
                                                      g h : Quiver.Hom Y Z
                                                      hgh : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStru …
                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) (F.map g)) (CategoryTheory. …
                                                    -/
      F.map_injective ((cancel_epi (F.map f)).1 (by rw [← F.map_comp, hgh, F.map_comp]))⟩
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- If `F` is a fully faithful functor, split epimorphisms are preserved and reflected by `F`. -/
noncomputable def splitEpiEquiv [Full F] [Faithful F] : SplitEpi f ≃ SplitEpi (F.map f) where
  toFun f := f.map F
  invFun s := ⟨F.preimage s.section_, by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor C D
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      s : CategoryTheory.SplitEpi (F.map f)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.preimage s.section_) f) (CategoryT …
    -/
    apply F.map_injective
    /-
      case a
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor C D
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      s : CategoryTheory.SplitEpi (F.map f)
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (F.preimage s.section_) f)) (F …
    -/
    simp only [map_comp, map_preimage, map_id]
    /-
      case a
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor C D
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      s : CategoryTheory.SplitEpi (F.map f)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp s.section_ (F.map f)) (CategoryTheory …
    -/
    apply SplitEpi.id⟩
    /-
      🎉 no goals
    -/
                 /-
                   C : Type u₁
                   inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                   D : Type u₂
                   inst✝³ : CategoryTheory.Category.{v₂, u₂} D
                   E : Type u₃
                   inst✝² : CategoryTheory.Category.{v₃, u₃} E
                   F : CategoryTheory.Functor C D
                   X Y : C
                   f : Quiver.Hom X Y
                   inst✝¹ : F.Full
                   inst✝ : F.Faithful
                   ⊢ Function.LeftInverse (fun s => { section_ := F.preimage s.section_, id := ⋯  …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                    /-
                      C : Type u₁
                      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                      D : Type u₂
                      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
                      E : Type u₃
                      inst✝² : CategoryTheory.Category.{v₃, u₃} E
                      F : CategoryTheory.Functor C D
                      X Y : C
                      f : Quiver.Hom X Y
                      inst✝¹ : F.Full
                      inst✝ : F.Faithful
                      x : CategoryTheory.SplitEpi (F.map f)
                      ⊢ Eq ((fun f_1 => f_1.map F) ((fun s => { section_ := F.preimage s.section_, i …
                    -/
  right_inv x := by aesop_cat
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem isSplitEpi_iff [Full F] [Faithful F] : IsSplitEpi (F.map f) ↔ IsSplitEpi f := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    ⊢ Iff (CategoryTheory.IsSplitEpi (F.map f)) (CategoryTheory.IsSplitEpi f)
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      ⊢ CategoryTheory.IsSplitEpi (F.map f) → CategoryTheory.IsSplitEpi f
    -/
  · intro h
    /-
      case mp
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      h : CategoryTheory.IsSplitEpi (F.map f)
      ⊢ CategoryTheory.IsSplitEpi f
    -/
    exact IsSplitEpi.mk' ((splitEpiEquiv F f).invFun h.exists_splitEpi.some)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      ⊢ CategoryTheory.IsSplitEpi f → CategoryTheory.IsSplitEpi (F.map f)
    -/
  · intro h
    /-
      case mpr
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      h : CategoryTheory.IsSplitEpi f
      ⊢ CategoryTheory.IsSplitEpi (F.map f)
    -/
    exact IsSplitEpi.mk' ((splitEpiEquiv F f).toFun h.exists_splitEpi.some)
    /-
      🎉 no goals
    -/


/-- If `F` is a fully faithful functor, split monomorphisms are preserved and reflected by `F`. -/
noncomputable def splitMonoEquiv [Full F] [Faithful F] : SplitMono f ≃ SplitMono (F.map f) where
  toFun f := f.map F
  invFun s := ⟨F.preimage s.retraction, by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor C D
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      s : CategoryTheory.SplitMono (F.map f)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (F.preimage s.retraction)) (Categor …
    -/
    apply F.map_injective
    /-
      case a
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor C D
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      s : CategoryTheory.SplitMono (F.map f)
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp f (F.preimage s.retraction)))  …
    -/
    simp only [map_comp, map_preimage, map_id]
    /-
      case a
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor C D
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      s : CategoryTheory.SplitMono (F.map f)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) s.retraction) (CategoryTheo …
    -/
    apply SplitMono.id⟩
    /-
      🎉 no goals
    -/
                 /-
                   C : Type u₁
                   inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                   D : Type u₂
                   inst✝³ : CategoryTheory.Category.{v₂, u₂} D
                   E : Type u₃
                   inst✝² : CategoryTheory.Category.{v₃, u₃} E
                   F : CategoryTheory.Functor C D
                   X Y : C
                   f : Quiver.Hom X Y
                   inst✝¹ : F.Full
                   inst✝ : F.Faithful
                   ⊢ Function.LeftInverse (fun s => { retraction := F.preimage s.retraction, id : …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                    /-
                      C : Type u₁
                      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                      D : Type u₂
                      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
                      E : Type u₃
                      inst✝² : CategoryTheory.Category.{v₃, u₃} E
                      F : CategoryTheory.Functor C D
                      X Y : C
                      f : Quiver.Hom X Y
                      inst✝¹ : F.Full
                      inst✝ : F.Faithful
                      x : CategoryTheory.SplitMono (F.map f)
                      ⊢ Eq ((fun f_1 => f_1.map F) ((fun s => { retraction := F.preimage s.retractio …
                    -/
  right_inv x := by aesop_cat
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem isSplitMono_iff [Full F] [Faithful F] : IsSplitMono (F.map f) ↔ IsSplitMono f := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    ⊢ Iff (CategoryTheory.IsSplitMono (F.map f)) (CategoryTheory.IsSplitMono f)
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      ⊢ CategoryTheory.IsSplitMono (F.map f) → CategoryTheory.IsSplitMono f
    -/
  · intro h
    /-
      case mp
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      h : CategoryTheory.IsSplitMono (F.map f)
      ⊢ CategoryTheory.IsSplitMono f
    -/
    exact IsSplitMono.mk' ((splitMonoEquiv F f).invFun h.exists_splitMono.some)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      ⊢ CategoryTheory.IsSplitMono f → CategoryTheory.IsSplitMono (F.map f)
    -/
  · intro h
    /-
      case mpr
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      h : CategoryTheory.IsSplitMono f
      ⊢ CategoryTheory.IsSplitMono (F.map f)
    -/
    exact IsSplitMono.mk' ((splitMonoEquiv F f).toFun h.exists_splitMono.some)
    /-
      🎉 no goals
    -/


@[simp]
theorem epi_map_iff_epi [hF₁ : PreservesEpimorphisms F] [hF₂ : ReflectsEpimorphisms F] :
    Epi (F.map f) ↔ Epi f := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y : C
    f : Quiver.Hom X Y
    hF₁ : F.PreservesEpimorphisms
    hF₂ : F.ReflectsEpimorphisms
    ⊢ Iff (CategoryTheory.Epi (F.map f)) (CategoryTheory.Epi f)
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y : C
      f : Quiver.Hom X Y
      hF₁ : F.PreservesEpimorphisms
      hF₂ : F.ReflectsEpimorphisms
      ⊢ CategoryTheory.Epi (F.map f) → CategoryTheory.Epi f
    -/
  · exact F.epi_of_epi_map
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y : C
      f : Quiver.Hom X Y
      hF₁ : F.PreservesEpimorphisms
      hF₂ : F.ReflectsEpimorphisms
      ⊢ CategoryTheory.Epi f → CategoryTheory.Epi (F.map f)
    -/
  · intro h
    /-
      case mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y : C
      f : Quiver.Hom X Y
      hF₁ : F.PreservesEpimorphisms
      hF₂ : F.ReflectsEpimorphisms
      h : CategoryTheory.Epi f
      ⊢ CategoryTheory.Epi (F.map f)
    -/
    exact F.map_epi f
    /-
      🎉 no goals
    -/


@[simp]
theorem mono_map_iff_mono [hF₁ : PreservesMonomorphisms F] [hF₂ : ReflectsMonomorphisms F] :
    Mono (F.map f) ↔ Mono f := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    X Y : C
    f : Quiver.Hom X Y
    hF₁ : F.PreservesMonomorphisms
    hF₂ : F.ReflectsMonomorphisms
    ⊢ Iff (CategoryTheory.Mono (F.map f)) (CategoryTheory.Mono f)
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y : C
      f : Quiver.Hom X Y
      hF₁ : F.PreservesMonomorphisms
      hF₂ : F.ReflectsMonomorphisms
      ⊢ CategoryTheory.Mono (F.map f) → CategoryTheory.Mono f
    -/
  · exact F.mono_of_mono_map
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y : C
      f : Quiver.Hom X Y
      hF₁ : F.PreservesMonomorphisms
      hF₂ : F.ReflectsMonomorphisms
      ⊢ CategoryTheory.Mono f → CategoryTheory.Mono (F.map f)
    -/
  · intro h
    /-
      case mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      X Y : C
      f : Quiver.Hom X Y
      hF₁ : F.PreservesMonomorphisms
      hF₂ : F.ReflectsMonomorphisms
      h : CategoryTheory.Mono f
      ⊢ CategoryTheory.Mono (F.map f)
    -/
    exact F.map_mono f
    /-
      🎉 no goals
    -/


/-- If `F : C ⥤ D` is an equivalence of categories and `C` is a `split_epi_category`,
then `D` also is. -/
theorem splitEpiCategoryImpOfIsEquivalence [IsEquivalence F] [SplitEpiCategory C] :
    SplitEpiCategory D :=
  ⟨fun {X} {Y} f => by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.IsEquivalence
      inst✝ : CategoryTheory.SplitEpiCategory C
      X Y : D
      f : Quiver.Hom X Y
      ⊢ ∀ [inst : CategoryTheory.Epi f], CategoryTheory.IsSplitEpi f
    -/
    intro
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝² : F.IsEquivalence
      inst✝¹ : CategoryTheory.SplitEpiCategory C
      X Y : D
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.Epi f
      ⊢ CategoryTheory.IsSplitEpi f
    -/
    rw [← F.inv.isSplitEpi_iff f]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝² : F.IsEquivalence
      inst✝¹ : CategoryTheory.SplitEpiCategory C
      X Y : D
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.Epi f
      ⊢ CategoryTheory.IsSplitEpi (F.inv.map f)
    -/
    apply isSplitEpi_of_epi⟩
    /-
      🎉 no goals
    -/


theorem strongEpi_map_of_strongEpi (adj : F ⊣ F') (f : A ⟶ B) [F'.PreservesMonomorphisms]
    [F.PreservesEpimorphisms] [StrongEpi f] : StrongEpi (F.map f) :=
  ⟨inferInstance, fun X Y Z => by
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      F' : CategoryTheory.Functor D C
      A B : C
      adj : CategoryTheory.Adjunction F F'
      f : Quiver.Hom A B
      inst✝² : F'.PreservesMonomorphisms
      inst✝¹ : F.PreservesEpimorphisms
      inst✝ : CategoryTheory.StrongEpi f
      X Y : D
      Z : Quiver.Hom X Y
      ⊢ ∀ [inst : CategoryTheory.Mono Z], CategoryTheory.HasLiftingProperty (F.map f …
    -/
    intro
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      F' : CategoryTheory.Functor D C
      A B : C
      adj : CategoryTheory.Adjunction F F'
      f : Quiver.Hom A B
      inst✝³ : F'.PreservesMonomorphisms
      inst✝² : F.PreservesEpimorphisms
      inst✝¹ : CategoryTheory.StrongEpi f
      X Y : D
      Z : Quiver.Hom X Y
      inst✝ : CategoryTheory.Mono Z
      ⊢ CategoryTheory.HasLiftingProperty (F.map f) Z
    -/
    rw [adj.hasLiftingProperty_iff]
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      F' : CategoryTheory.Functor D C
      A B : C
      adj : CategoryTheory.Adjunction F F'
      f : Quiver.Hom A B
      inst✝³ : F'.PreservesMonomorphisms
      inst✝² : F.PreservesEpimorphisms
      inst✝¹ : CategoryTheory.StrongEpi f
      X Y : D
      Z : Quiver.Hom X Y
      inst✝ : CategoryTheory.Mono Z
      ⊢ CategoryTheory.HasLiftingProperty f (F'.map Z)
    -/
    infer_instance⟩
    /-
      🎉 no goals
    -/


instance strongEpi_map_of_isEquivalence [F.IsEquivalence] (f : A ⟶ B) [_h : StrongEpi f] :
    StrongEpi (F.map f) :=
  F.asEquivalence.toAdjunction.strongEpi_map_of_strongEpi f


instance (adj : F ⊣ F') {X : C} {Y : D} (f : F.obj X ⟶ Y) [hf : Mono f] [F.ReflectsMonomorphisms] :
    Mono (adj.homEquiv _ _ f) :=
  F.mono_of_mono_map <| by
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      F' : CategoryTheory.Functor D C
      A B : C
      adj : CategoryTheory.Adjunction F F'
      X : C
      Y : D
      f : Quiver.Hom (F.obj X) Y
      hf : CategoryTheory.Mono f
      inst✝ : F.ReflectsMonomorphisms
      ⊢ CategoryTheory.Mono (F.map ((adj.homEquiv X Y) f))
    -/
    rw [← (homEquiv adj X Y).symm_apply_apply f] at hf
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      F' : CategoryTheory.Functor D C
      A B : C
      adj : CategoryTheory.Adjunction F F'
      X : C
      Y : D
      f : Quiver.Hom (F.obj X) Y
      hf : CategoryTheory.Mono ((adj.homEquiv X Y).symm ((adj.homEquiv X Y) f))
      inst✝ : F.ReflectsMonomorphisms
      ⊢ CategoryTheory.Mono (F.map ((adj.homEquiv X Y) f))
    -/
    exact mono_of_mono_fac (adj.homEquiv_counit _ _ _).symm
    /-
      🎉 no goals
    -/


@[simp]
theorem strongEpi_map_iff_strongEpi_of_isEquivalence [IsEquivalence F] :
    StrongEpi (F.map f) ↔ StrongEpi f := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    A B : C
    f : Quiver.Hom A B
    inst✝ : F.IsEquivalence
    ⊢ Iff (CategoryTheory.StrongEpi (F.map f)) (CategoryTheory.StrongEpi f)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      A B : C
      f : Quiver.Hom A B
      inst✝ : F.IsEquivalence
      ⊢ CategoryTheory.StrongEpi (F.map f) → CategoryTheory.StrongEpi f
    -/
  · intro
    have e : Arrow.mk f ≅ Arrow.mk (F.inv.map (F.map f)) :=
      Arrow.isoOfNatIso F.asEquivalence.unitIso (Arrow.mk f)
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      A B : C
      f : Quiver.Hom A B
      inst✝ : F.IsEquivalence
      a✝ : CategoryTheory.StrongEpi (F.map f)
      e : CategoryTheory.Iso (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk (F …
      ⊢ CategoryTheory.StrongEpi f
    -/
    rw [StrongEpi.iff_of_arrow_iso e]
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      A B : C
      f : Quiver.Hom A B
      inst✝ : F.IsEquivalence
      a✝ : CategoryTheory.StrongEpi (F.map f)
      e : CategoryTheory.Iso (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk (F …
      ⊢ CategoryTheory.StrongEpi (F.inv.map (F.map f))
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      A B : C
      f : Quiver.Hom A B
      inst✝ : F.IsEquivalence
      ⊢ CategoryTheory.StrongEpi f → CategoryTheory.StrongEpi (F.map f)
    -/
  · intro
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      A B : C
      f : Quiver.Hom A B
      inst✝ : F.IsEquivalence
      a✝ : CategoryTheory.StrongEpi f
      ⊢ CategoryTheory.StrongEpi (F.map f)
    -/
    infer_instance
    /-
      🎉 no goals
    -/



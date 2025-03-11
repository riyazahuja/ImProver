/-- A functor preserves zero morphisms if it sends zero morphisms to zero morphisms. -/
class PreservesZeroMorphisms (F : C ⥤ D) : Prop where
  /-- For any pair objects `F (0: X ⟶ Y) = (0 : F X ⟶ F Y)` -/
  map_zero : ∀ X Y : C, F.map (0 : X ⟶ Y) = 0 := by aesop


@[simp]
protected theorem map_zero (F : C ⥤ D) [PreservesZeroMorphisms F] (X Y : C) :
    F.map (0 : X ⟶ Y) = 0 :=
  PreservesZeroMorphisms.map_zero _ _


lemma map_isZero (F : C ⥤ D) [PreservesZeroMorphisms F] {X : C} (hX : IsZero X) :
    IsZero (F.obj X) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
    F : CategoryTheory.Functor C D
    inst✝ : F.PreservesZeroMorphisms
    X : C
    hX : CategoryTheory.Limits.IsZero X
    ⊢ CategoryTheory.Limits.IsZero (F.obj X)
  -/
  simp only [IsZero.iff_id_eq_zero] at hX ⊢
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
    F : CategoryTheory.Functor C D
    inst✝ : F.PreservesZeroMorphisms
    X : C
    hX : Eq (CategoryTheory.CategoryStruct.id X) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.id (F.obj X)) 0
  -/
  rw [← F.map_id, hX, F.map_zero]
  /-
    🎉 no goals
  -/


theorem zero_of_map_zero (F : C ⥤ D) [PreservesZeroMorphisms F] [Faithful F] {X Y : C} (f : X ⟶ Y)
    (h : F.map f = 0) : f = 0 :=
  F.map_injective <| h.trans <| Eq.symm <| F.map_zero _ _


theorem map_eq_zero_iff (F : C ⥤ D) [PreservesZeroMorphisms F] [Faithful F] {X Y : C} {f : X ⟶ Y} :
    F.map f = 0 ↔ f = 0 :=
  ⟨F.zero_of_map_zero _, by
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.PreservesZeroMorphisms
      inst✝ : F.Faithful
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq f 0 → Eq (F.map f) 0
    -/
    rintro rfl
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.PreservesZeroMorphisms
      inst✝ : F.Faithful
      X Y : C
      ⊢ Eq (F.map 0) 0
    -/
    exact F.map_zero _ _⟩
    /-
      🎉 no goals
    -/


instance (priority := 100) preservesZeroMorphisms_of_isLeftAdjoint (F : C ⥤ D) [IsLeftAdjoint F] :
    PreservesZeroMorphisms F where
  map_zero X Y := by
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms E
      F : CategoryTheory.Functor C D
      inst✝ : F.IsLeftAdjoint
      X Y : C
      ⊢ Eq (F.map 0) 0
    -/
    let adj := Adjunction.ofIsLeftAdjoint F
    calc
      F.map (0 : X ⟶ Y) = F.map 0 ≫ F.map (adj.unit.app Y) ≫ adj.counit.app (F.obj Y) := ?_
      _ = F.map 0 ≫ F.map ((rightAdjoint F).map (0 : F.obj X ⟶ _)) ≫ adj.counit.app (F.obj Y) := ?_
      _ = 0 := ?_
      /-
        case calc_1
        C : Type u₁
        inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms E
        F : CategoryTheory.Functor C D
        inst✝ : F.IsLeftAdjoint
        X Y : C
        adj : CategoryTheory.Adjunction F F.rightAdjoint := CategoryTheory.Adjunction. …
        ⊢ Eq (F.map 0) (CategoryTheory.CategoryStruct.comp (F.map 0) (CategoryTheory.C …
      -/
    · rw [Adjunction.left_triangle_components]
      /-
        case calc_1
        C : Type u₁
        inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms E
        F : CategoryTheory.Functor C D
        inst✝ : F.IsLeftAdjoint
        X Y : C
        adj : CategoryTheory.Adjunction F F.rightAdjoint := CategoryTheory.Adjunction. …
        ⊢ Eq (F.map 0) (CategoryTheory.CategoryStruct.comp (F.map 0) (CategoryTheory.C …
      -/
      exact (Category.comp_id _).symm
      /-
        🎉 no goals
      -/
      /-
        case calc_2
        C : Type u₁
        inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms E
        F : CategoryTheory.Functor C D
        inst✝ : F.IsLeftAdjoint
        X Y : C
        adj : CategoryTheory.Adjunction F F.rightAdjoint := CategoryTheory.Adjunction. …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map 0) (CategoryTheory.CategoryStr …
      -/
    · simp only [← Category.assoc, ← F.map_comp, zero_comp]
      /-
        🎉 no goals
      -/
      /-
        case calc_3
        C : Type u₁
        inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms E
        F : CategoryTheory.Functor C D
        inst✝ : F.IsLeftAdjoint
        X Y : C
        adj : CategoryTheory.Adjunction F F.rightAdjoint := CategoryTheory.Adjunction. …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map 0) (CategoryTheory.CategoryStr …
      -/
    · simp only [Adjunction.counit_naturality, comp_zero]
      /-
        🎉 no goals
      -/


instance (priority := 100) preservesZeroMorphisms_of_isRightAdjoint (G : C ⥤ D) [IsRightAdjoint G] :
    PreservesZeroMorphisms G where
  map_zero X Y := by
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms E
      G : CategoryTheory.Functor C D
      inst✝ : G.IsRightAdjoint
      X Y : C
      ⊢ Eq (G.map 0) 0
    -/
    let adj := Adjunction.ofIsRightAdjoint G
    calc
      G.map (0 : X ⟶ Y) = adj.unit.app (G.obj X) ≫ G.map (adj.counit.app X) ≫ G.map 0 := ?_
      _ = adj.unit.app (G.obj X) ≫ G.map ((leftAdjoint G).map (0 : _ ⟶ G.obj X)) ≫ G.map 0 := ?_
      _ = 0 := ?_
      /-
        case calc_1
        C : Type u₁
        inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms E
        G : CategoryTheory.Functor C D
        inst✝ : G.IsRightAdjoint
        X Y : C
        adj : CategoryTheory.Adjunction G.leftAdjoint G := CategoryTheory.Adjunction.o …
        ⊢ Eq (G.map 0) (CategoryTheory.CategoryStruct.comp (adj.unit.app (G.obj X)) (C …
      -/
    · rw [Adjunction.right_triangle_components_assoc]
      /-
        🎉 no goals
      -/
      /-
        case calc_2
        C : Type u₁
        inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms E
        G : CategoryTheory.Functor C D
        inst✝ : G.IsRightAdjoint
        X Y : C
        adj : CategoryTheory.Adjunction G.leftAdjoint G := CategoryTheory.Adjunction.o …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app (G.obj X)) (CategoryThe …
      -/
    · simp only [← G.map_comp, comp_zero]
      /-
        🎉 no goals
      -/
      /-
        case calc_3
        C : Type u₁
        inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms E
        G : CategoryTheory.Functor C D
        inst✝ : G.IsRightAdjoint
        X Y : C
        adj : CategoryTheory.Adjunction G.leftAdjoint G := CategoryTheory.Adjunction.o …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app (G.obj X)) (CategoryThe …
      -/
    · simp only [id_obj, comp_obj, Adjunction.unit_naturality_assoc, zero_comp]
      /-
        🎉 no goals
      -/


instance (priority := 100) preservesZeroMorphisms_of_full (F : C ⥤ D) [Full F] :
    PreservesZeroMorphisms F where
  map_zero X Y :=
    calc
                                                                               /-
                                                                                 C : Type u₁
                                                                                 inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
                                                                                 D : Type u₂
                                                                                 inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
                                                                                 E : Type u₃
                                                                                 inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
                                                                                 inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                                 inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
                                                                                 inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms E
                                                                                 F : CategoryTheory.Functor C D
                                                                                 inst✝ : F.Full
                                                                                 X Y : C
                                                                                 ⊢ Eq (F.map 0) (F.map (CategoryTheory.CategoryStruct.comp 0 (F.preimage 0)))
                                                                               -/
      F.map (0 : X ⟶ Y) = F.map (0 ≫ F.preimage (0 : F.obj Y ⟶ F.obj Y)) := by rw [zero_comp]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
                  /-
                    C : Type u₁
                    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
                    D : Type u₂
                    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
                    E : Type u₃
                    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
                    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                    inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
                    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms E
                    F : CategoryTheory.Functor C D
                    inst✝ : F.Full
                    X Y : C
                    ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp 0 (F.preimage 0))) 0
                  -/
      _ = 0 := by rw [F.map_comp, F.map_preimage, comp_zero]
                  /-
                    🎉 no goals
                  -/


instance preservesZeroMorphisms_comp (F : C ⥤ D) (G : D ⥤ E)
    [F.PreservesZeroMorphisms] [G.PreservesZeroMorphisms] :
                                          /-
                                            C : Type u₁
                                            inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
                                            D : Type u₂
                                            inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
                                            E : Type u₃
                                            inst✝⁵ : CategoryTheory.Category.{v₃, u₃} E
                                            inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
                                            inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
                                            inst✝² : CategoryTheory.Limits.HasZeroMorphisms E
                                            F : CategoryTheory.Functor C D
                                            G : CategoryTheory.Functor D E
                                            inst✝¹ : F.PreservesZeroMorphisms
                                            inst✝ : G.PreservesZeroMorphisms
                                            ⊢ ∀ (X Y : C), Eq ((F.comp G).map 0) 0
                                          -/
    (F ⋙ G).PreservesZeroMorphisms := ⟨by simp⟩
                                          /-
                                            🎉 no goals
                                          -/


lemma preservesZeroMorphisms_of_iso {F₁ F₂ : C ⥤ D} [F₁.PreservesZeroMorphisms] (e : F₁ ≅ F₂) :
    F₂.PreservesZeroMorphisms where
  map_zero X Y := by simp only [← cancel_epi (e.hom.app X), ← e.hom.naturality,
    F₁.map_zero, zero_comp, comp_zero]


instance preservesZeroMorphisms_evaluation_obj (j : D) :
    PreservesZeroMorphisms ((evaluation D C).obj j) where


instance (F : C ⥤ D ⥤ E) [∀ X, (F.obj X).PreservesZeroMorphisms] :
    F.flip.PreservesZeroMorphisms where


instance (F : C ⥤ D ⥤ E) [F.PreservesZeroMorphisms] (Y : D) :
    (F.flip.obj Y).PreservesZeroMorphisms where


/-- A functor that preserves zero morphisms also preserves the zero object. -/
@[simps]
def mapZeroObject [PreservesZeroMorphisms F] : F.obj 0 ≅ 0 where
  hom := 0
  inv := 0
                   /-
                     C : Type u₁
                     inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
                     D : Type u₂
                     inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
                     E : Type u₃
                     inst✝⁵ : CategoryTheory.Category.{v₃, u₃} E
                     inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
                     inst✝³ : CategoryTheory.Limits.HasZeroObject D
                     inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                     inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
                     F : CategoryTheory.Functor C D
                     inst✝ : F.PreservesZeroMorphisms
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 0) (CategoryTheory.CategoryStruct.i …
                   -/
  hom_inv_id := by rw [← F.map_id, id_zero, F.map_zero, zero_comp]
                   /-
                     🎉 no goals
                   -/
                   /-
                     C : Type u₁
                     inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
                     D : Type u₂
                     inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
                     E : Type u₃
                     inst✝⁵ : CategoryTheory.Category.{v₃, u₃} E
                     inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
                     inst✝³ : CategoryTheory.Limits.HasZeroObject D
                     inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                     inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
                     F : CategoryTheory.Functor C D
                     inst✝ : F.PreservesZeroMorphisms
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 0) (CategoryTheory.CategoryStruct.i …
                   -/
  inv_hom_id := by rw [id_zero, comp_zero]
                   /-
                     🎉 no goals
                   -/


theorem preservesZeroMorphisms_of_map_zero_object (i : F.obj 0 ≅ 0) : PreservesZeroMorphisms F where
  map_zero X Y :=
    calc
                                                            /-
                                                              C : Type u₁
                                                              inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
                                                              D : Type u₂
                                                              inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
                                                              inst✝³ : CategoryTheory.Limits.HasZeroObject C
                                                              inst✝² : CategoryTheory.Limits.HasZeroObject D
                                                              inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                              inst✝ : CategoryTheory.Limits.HasZeroMorphisms D
                                                              F : CategoryTheory.Functor C D
                                                              i : CategoryTheory.Iso (F.obj 0) 0
                                                              X Y : C
                                                              ⊢ Eq (F.map 0) (CategoryTheory.CategoryStruct.comp (F.map 0) (F.map 0))
                                                            -/
      F.map (0 : X ⟶ Y) = F.map (0 : X ⟶ 0) ≫ F.map 0 := by rw [← Functor.map_comp, comp_zero]
                                                            /-
                                                              🎉 no goals
                                                            -/
                                                    /-
                                                      C : Type u₁
                                                      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
                                                      D : Type u₂
                                                      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
                                                      inst✝³ : CategoryTheory.Limits.HasZeroObject C
                                                      inst✝² : CategoryTheory.Limits.HasZeroObject D
                                                      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                      inst✝ : CategoryTheory.Limits.HasZeroMorphisms D
                                                      F : CategoryTheory.Functor C D
                                                      i : CategoryTheory.Iso (F.obj 0) 0
                                                      X Y : C
                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map 0) (F.map 0)) (CategoryTheory. …
                                                    -/
      _ = F.map 0 ≫ (i.hom ≫ i.inv) ≫ F.map 0 := by rw [Iso.hom_inv_id, Category.id_comp]
                                                    /-
                                                      🎉 no goals
                                                    -/
                  /-
                    C : Type u₁
                    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
                    D : Type u₂
                    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
                    inst✝³ : CategoryTheory.Limits.HasZeroObject C
                    inst✝² : CategoryTheory.Limits.HasZeroObject D
                    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                    inst✝ : CategoryTheory.Limits.HasZeroMorphisms D
                    F : CategoryTheory.Functor C D
                    i : CategoryTheory.Iso (F.obj 0) 0
                    X Y : C
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map 0) (CategoryTheory.CategoryStr …
                  -/
      _ = 0 := by simp only [zero_of_to_zero i.hom, zero_comp, comp_zero]
                  /-
                    🎉 no goals
                  -/


instance (priority := 100) preservesZeroMorphisms_of_preserves_initial_object
    [PreservesColimit (Functor.empty.{0} C) F] : PreservesZeroMorphisms F :=
  preservesZeroMorphisms_of_map_zero_object <|
    F.mapIso HasZeroObject.zeroIsoInitial ≪≫
      PreservesInitial.iso F ≪≫ HasZeroObject.zeroIsoInitial.symm


instance (priority := 100) preservesZeroMorphisms_of_preserves_terminal_object
    [PreservesLimit (Functor.empty.{0} C) F] : PreservesZeroMorphisms F :=
  preservesZeroMorphisms_of_map_zero_object <|
    F.mapIso HasZeroObject.zeroIsoTerminal ≪≫
      PreservesTerminal.iso F ≪≫ HasZeroObject.zeroIsoTerminal.symm


/-- Preserving zero morphisms implies preserving terminal objects. -/
lemma preservesTerminalObject_of_preservesZeroMorphisms [PreservesZeroMorphisms F] :
    PreservesLimit (Functor.empty.{0} C) F :=
  preservesTerminal_of_iso F <|
    F.mapIso HasZeroObject.zeroIsoTerminal.symm ≪≫ mapZeroObject F ≪≫ HasZeroObject.zeroIsoTerminal


/-- Preserving zero morphisms implies preserving terminal objects. -/
lemma preservesInitialObject_of_preservesZeroMorphisms [PreservesZeroMorphisms F] :
    PreservesColimit (Functor.empty.{0} C) F :=
  preservesInitial_of_iso F <|
    HasZeroObject.zeroIsoInitial.symm ≪≫
      (mapZeroObject F).symm ≪≫ (F.mapIso HasZeroObject.zeroIsoInitial.symm).symm


/-- A zero functor preserves limits. -/
lemma preservesLimitsOfShape_of_isZero : PreservesLimitsOfShape J G where
  preservesLimit {K} := ⟨fun _ => ⟨by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      inst✝² : CategoryTheory.Limits.HasZeroObject D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
      G : CategoryTheory.Functor C D
      hG : CategoryTheory.Limits.IsZero G
      J : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} J
      K : CategoryTheory.Functor J C
      c✝ : CategoryTheory.Limits.Cone K
      x✝ : CategoryTheory.Limits.IsLimit c✝
      ⊢ CategoryTheory.Limits.IsLimit (G.mapCone c✝)
    -/
    rw [Functor.isZero_iff] at hG
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      inst✝² : CategoryTheory.Limits.HasZeroObject D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
      G : CategoryTheory.Functor C D
      hG : ∀ (X : C), CategoryTheory.Limits.IsZero (G.obj X)
      J : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} J
      K : CategoryTheory.Functor J C
      c✝ : CategoryTheory.Limits.Cone K
      x✝ : CategoryTheory.Limits.IsLimit c✝
      ⊢ CategoryTheory.Limits.IsLimit (G.mapCone c✝)
    -/
    exact IsLimit.ofIsZero _ ((K ⋙ G).isZero (fun X ↦ hG _)) (hG _)⟩⟩
    /-
      🎉 no goals
    -/


/-- A zero functor preserves colimits. -/
lemma preservesColimitsOfShape_of_isZero : PreservesColimitsOfShape J G where
  preservesColimit {K} := ⟨fun _ => ⟨by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      inst✝² : CategoryTheory.Limits.HasZeroObject D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
      G : CategoryTheory.Functor C D
      hG : CategoryTheory.Limits.IsZero G
      J : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} J
      K : CategoryTheory.Functor J C
      c✝ : CategoryTheory.Limits.Cocone K
      x✝ : CategoryTheory.Limits.IsColimit c✝
      ⊢ CategoryTheory.Limits.IsColimit (G.mapCocone c✝)
    -/
    rw [Functor.isZero_iff] at hG
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      inst✝² : CategoryTheory.Limits.HasZeroObject D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
      G : CategoryTheory.Functor C D
      hG : ∀ (X : C), CategoryTheory.Limits.IsZero (G.obj X)
      J : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} J
      K : CategoryTheory.Functor J C
      c✝ : CategoryTheory.Limits.Cocone K
      x✝ : CategoryTheory.Limits.IsColimit c✝
      ⊢ CategoryTheory.Limits.IsColimit (G.mapCocone c✝)
    -/
    exact IsColimit.ofIsZero _ ((K ⋙ G).isZero (fun X ↦ hG _)) (hG _)⟩⟩
    /-
      🎉 no goals
    -/


/-- A zero functor preserves limits. -/
lemma preservesLimitsOfSize_of_isZero : PreservesLimitsOfSize.{v, u} G where
  preservesLimitsOfShape := G.preservesLimitsOfShape_of_isZero hG _


/-- A zero functor preserves colimits. -/
lemma preservesColimitsOfSize_of_isZero : PreservesColimitsOfSize.{v, u} G where
  preservesColimitsOfShape := G.preservesColimitsOfShape_of_isZero hG _



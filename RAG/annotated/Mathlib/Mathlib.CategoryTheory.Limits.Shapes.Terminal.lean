/-- A category has a terminal object if it has a limit over the empty diagram.
Use `hasTerminal_of_unique` to construct instances.
-/
abbrev HasTerminal :=
  HasLimitsOfShape (Discrete.{0} PEmpty) C


/-- A category has an initial object if it has a colimit over the empty diagram.
Use `hasInitial_of_unique` to construct instances.
-/
abbrev HasInitial :=
  HasColimitsOfShape (Discrete.{0} PEmpty) C


theorem hasTerminalChangeDiagram (h : HasLimit F₁) : HasLimit F₂ :=
                   /-
                     C : Type u₁
                     inst✝ : CategoryTheory.Category.{v₁, u₁} C
                     F₁ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w + 1}) C
                     F₂ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w' + 1}) C
                     h : CategoryTheory.Limits.HasLimit F₁
                     ⊢ (X : CategoryTheory.Discrete PEmpty.{w' + 1}) → Quiver.Hom (((CategoryTheory …
                   -/
                   /-
                     🎉 no goals
                   -/
  ⟨⟨⟨⟨limit F₁, by aesop_cat, by aesop_cat⟩,
                                 /-
                                   🎉 no goals
                                 -/
    isLimitChangeEmptyCone C (limit.isLimit F₁) _ (eqToIso rfl)⟩⟩⟩


theorem hasTerminalChangeUniverse [h : HasLimitsOfShape (Discrete.{w} PEmpty) C] :
    HasLimitsOfShape (Discrete.{w'} PEmpty) C where
  has_limit _ := hasTerminalChangeDiagram C (h.1 (Functor.empty C))


theorem hasInitialChangeDiagram (h : HasColimit F₁) : HasColimit F₂ :=
                     /-
                       C : Type u₁
                       inst✝ : CategoryTheory.Category.{v₁, u₁} C
                       F₁ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w + 1}) C
                       F₂ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w' + 1}) C
                       h : CategoryTheory.Limits.HasColimit F₁
                       ⊢ (X : CategoryTheory.Discrete PEmpty.{w' + 1}) → Quiver.Hom (F₂.obj X) (((Cat …
                     -/
                     /-
                       🎉 no goals
                     -/
  ⟨⟨⟨⟨colimit F₁, by aesop_cat, by aesop_cat⟩,
                                   /-
                                     🎉 no goals
                                   -/
    isColimitChangeEmptyCocone C (colimit.isColimit F₁) _ (eqToIso rfl)⟩⟩⟩


theorem hasInitialChangeUniverse [h : HasColimitsOfShape (Discrete.{w} PEmpty) C] :
    HasColimitsOfShape (Discrete.{w'} PEmpty) C where
  has_colimit _ := hasInitialChangeDiagram C (h.1 (Functor.empty C))


/-- An arbitrary choice of terminal object, if one exists.
You can use the notation `⊤_ C`.
This object is characterized by having a unique morphism from any object.
-/
abbrev terminal [HasTerminal C] : C :=
  limit (Functor.empty.{0} C)


/-- An arbitrary choice of initial object, if one exists.
You can use the notation `⊥_ C`.
This object is characterized by having a unique morphism to any object.
-/
abbrev initial [HasInitial C] : C :=
  colimit (Functor.empty.{0} C)


/-- Notation for the terminal object in `C` -/
notation "⊤_ " C:20 => terminal C


/-- Notation for the initial object in `C` -/
notation "⊥_ " C:20 => initial C


/-- We can more explicitly show that a category has a terminal object by specifying the object,
and showing there is a unique morphism to it from any other object. -/
theorem hasTerminal_of_unique (X : C) [∀ Y, Nonempty (Y ⟶ X)] [∀ Y, Subsingleton (Y ⟶ X)] :
    HasTerminal C where
  has_limit F := .mk ⟨_, (isTerminalEquivUnique F X).invFun fun _ ↦
    ⟨Classical.inhabited_of_nonempty', (Subsingleton.elim · _)⟩⟩


theorem IsTerminal.hasTerminal {X : C} (h : IsTerminal X) : HasTerminal C :=
                                              /-
                                                C : Type u₁
                                                inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                                X : C
                                                h : CategoryTheory.Limits.IsTerminal X
                                                F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) C
                                                ⊢ (X_1 : CategoryTheory.Discrete PEmpty.{1}) → Quiver.Hom (((CategoryTheory.Fu …
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
  { has_limit := fun F => HasLimit.mk ⟨⟨X, by aesop_cat, by aesop_cat⟩,
                                                            /-
                                                              🎉 no goals
                                                            -/
    isLimitChangeEmptyCone _ h _ (Iso.refl _)⟩ }


/-- We can more explicitly show that a category has an initial object by specifying the object,
and showing there is a unique morphism from it to any other object. -/
theorem hasInitial_of_unique (X : C) [∀ Y, Nonempty (X ⟶ Y)] [∀ Y, Subsingleton (X ⟶ Y)] :
    HasInitial C where
  has_colimit F := .mk ⟨_, (isInitialEquivUnique F X).invFun fun _ ↦
    ⟨Classical.inhabited_of_nonempty', (Subsingleton.elim · _)⟩⟩


theorem IsInitial.hasInitial {X : C} (h : IsInitial X) : HasInitial C where
  has_colimit F :=
                          /-
                            C : Type u₁
                            inst✝ : CategoryTheory.Category.{v₁, u₁} C
                            X : C
                            h : CategoryTheory.Limits.IsInitial X
                            F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) C
                            ⊢ (X_1 : CategoryTheory.Discrete PEmpty.{1}) → Quiver.Hom (F.obj X_1) (((Categ …
                          -/
                          /-
                            🎉 no goals
                          -/
    HasColimit.mk ⟨⟨X, by aesop_cat, by aesop_cat⟩, isColimitChangeEmptyCocone _ h _ (Iso.refl _)⟩
                                        /-
                                          🎉 no goals
                                        -/


/-- The map from an object to the terminal object. -/
abbrev terminal.from [HasTerminal C] (P : C) : P ⟶ ⊤_ C :=
  limit.lift (Functor.empty C) (asEmptyCone P)


/-- The map to an object from the initial object. -/
abbrev initial.to [HasInitial C] (P : C) : ⊥_ C ⟶ P :=
  colimit.desc (Functor.empty C) (asEmptyCocone P)


/-- A terminal object is terminal. -/
def terminalIsTerminal [HasTerminal C] : IsTerminal (⊤_ C) where
  lift _ := terminal.from _


/-- An initial object is initial. -/
def initialIsInitial [HasInitial C] : IsInitial (⊥_ C) where
  desc _ := initial.to _


instance uniqueToTerminal [HasTerminal C] (P : C) : Unique (P ⟶ ⊤_ C) :=
  isTerminalEquivUnique _ (⊤_ C) terminalIsTerminal P


instance uniqueFromInitial [HasInitial C] (P : C) : Unique (⊥_ C ⟶ P) :=
  isInitialEquivUnique _ (⊥_ C) initialIsInitial P


                                                                                       /-
                                                                                         C : Type u₁
                                                                                         inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                         inst✝ : CategoryTheory.Limits.HasTerminal C
                                                                                         P : C
                                                                                         f g : Quiver.Hom P (CategoryTheory.Limits.terminal C)
                                                                                         ⊢ Eq f g
                                                                                       -/
@[ext] theorem terminal.hom_ext [HasTerminal C] {P : C} (f g : P ⟶ ⊤_ C) : f = g := by ext ⟨⟨⟩⟩
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


                                                                                     /-
                                                                                       C : Type u₁
                                                                                       inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                       inst✝ : CategoryTheory.Limits.HasInitial C
                                                                                       P : C
                                                                                       f g : Quiver.Hom (CategoryTheory.Limits.initial C) P
                                                                                       ⊢ Eq f g
                                                                                     -/
@[ext] theorem initial.hom_ext [HasInitial C] {P : C} (f g : ⊥_ C ⟶ P) : f = g := by ext ⟨⟨⟩⟩
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[simp]
theorem terminal.comp_from [HasTerminal C] {P Q : C} (f : P ⟶ Q) :
    f ≫ terminal.from Q = terminal.from P := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Limits.HasTerminal C
    P Q : C
    f : Quiver.Hom P Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.terminal.fro …
  -/
  simp [eq_iff_true_of_subsingleton]
  /-
    🎉 no goals
  -/


@[simp]
theorem initial.to_comp [HasInitial C] {P Q : C} (f : P ⟶ Q) : initial.to P ≫ f = initial.to Q := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Limits.HasInitial C
    P Q : C
    f : Quiver.Hom P Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.initial.to P)  …
  -/
  simp [eq_iff_true_of_subsingleton]
  /-
    🎉 no goals
  -/


/-- The (unique) isomorphism between the chosen initial object and any other initial object. -/
@[simp]
def initialIsoIsInitial [HasInitial C] {P : C} (t : IsInitial P) : ⊥_ C ≅ P :=
  initialIsInitial.uniqueUpToIso t


/-- The (unique) isomorphism between the chosen terminal object and any other terminal object. -/
@[simp]
def terminalIsoIsTerminal [HasTerminal C] {P : C} (t : IsTerminal P) : ⊤_ C ≅ P :=
  terminalIsTerminal.uniqueUpToIso t


/-- Any morphism from a terminal object is split mono. -/
instance terminal.isSplitMono_from {Y : C} [HasTerminal C] (f : ⊤_ C ⟶ Y) : IsSplitMono f :=
  IsTerminal.isSplitMono_from terminalIsTerminal _


/-- Any morphism to an initial object is split epi. -/
instance initial.isSplitEpi_to {Y : C} [HasInitial C] (f : Y ⟶ ⊥_ C) : IsSplitEpi f :=
  IsInitial.isSplitEpi_to initialIsInitial _


instance hasInitial_op_of_hasTerminal [HasTerminal C] : HasInitial Cᵒᵖ :=
  (initialOpOfTerminal terminalIsTerminal).hasInitial


instance hasTerminal_op_of_hasInitial [HasInitial C] : HasTerminal Cᵒᵖ :=
  (terminalOpOfInitial initialIsInitial).hasTerminal


theorem hasTerminal_of_hasInitial_op [HasInitial Cᵒᵖ] : HasTerminal C :=
  (terminalUnopOfInitial initialIsInitial).hasTerminal


theorem hasInitial_of_hasTerminal_op [HasTerminal Cᵒᵖ] : HasInitial C :=
  (initialUnopOfTerminal terminalIsTerminal).hasInitial


instance {J : Type*} [Category J] {C : Type*} [Category C] [HasTerminal C] :
    HasLimit ((CategoryTheory.Functor.const J).obj (⊤_ C)) :=
  HasLimit.mk
    { cone :=
        { pt := ⊤_ C
          π := { app := fun _ => terminal.from _ } }
      isLimit := { lift := fun _ => terminal.from _ } }


/-- The limit of the constant `⊤_ C` functor is `⊤_ C`. -/
@[simps hom]
def limitConstTerminal {J : Type*} [Category J] {C : Type*} [Category C] [HasTerminal C] :
    limit ((CategoryTheory.Functor.const J).obj (⊤_ C)) ≅ ⊤_ C where
  hom := terminal.from _
  inv :=
    limit.lift ((CategoryTheory.Functor.const J).obj (⊤_ C))
      { pt := ⊤_ C
        π := { app := fun _ => terminal.from _ } }


@[reassoc (attr := simp)]
theorem limitConstTerminal_inv_π {J : Type*} [Category J] {C : Type*} [Category C] [HasTerminal C]
    {j : J} :
    limitConstTerminal.inv ≫ limit.π ((CategoryTheory.Functor.const J).obj (⊤_ C)) j =
                            /-
                              J : Type u_1
                              inst✝² : CategoryTheory.Category.{u_3, u_1} J
                              C : Type u_2
                              inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
                              inst✝ : CategoryTheory.Limits.HasTerminal C
                              j : J
                              ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.limitConstTermi …
                            -/
      terminal.from _ := by aesop_cat
                            /-
                              🎉 no goals
                            -/


instance {J : Type*} [Category J] {C : Type*} [Category C] [HasInitial C] :
    HasColimit ((CategoryTheory.Functor.const J).obj (⊥_ C)) :=
  HasColimit.mk
    { cocone :=
        { pt := ⊥_ C
          ι := { app := fun _ => initial.to _ } }
      isColimit := { desc := fun _ => initial.to _ } }


/-- The colimit of the constant `⊥_ C` functor is `⊥_ C`. -/
@[simps inv]
def colimitConstInitial {J : Type*} [Category J] {C : Type*} [Category C] [HasInitial C] :
    colimit ((CategoryTheory.Functor.const J).obj (⊥_ C)) ≅ ⊥_ C where
  hom :=
    colimit.desc ((CategoryTheory.Functor.const J).obj (⊥_ C))
      { pt := ⊥_ C
        ι := { app := fun _ => initial.to _ } }
  inv := initial.to _


@[reassoc (attr := simp)]
theorem ι_colimitConstInitial_hom {J : Type*} [Category J] {C : Type*} [Category C] [HasInitial C]
    {j : J} :
    colimit.ι ((CategoryTheory.Functor.const J).obj (⊥_ C)) j ≫ colimitConstInitial.hom =
                         /-
                           J : Type u_1
                           inst✝² : CategoryTheory.Category.{u_3, u_1} J
                           C : Type u_2
                           inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
                           inst✝ : CategoryTheory.Limits.HasInitial C
                           j : J
                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((Ca …
                         -/
      initial.to _ := by aesop_cat
                         /-
                           🎉 no goals
                         -/


instance (priority := 100) initial.mono_from [HasInitial C] [InitialMonoClass C] (X : C)
    (f : ⊥_ C ⟶ X) : Mono f :=
  initialIsInitial.mono_from f


/-- To show a category is an `InitialMonoClass` it suffices to show every morphism out of the
initial object is a monomorphism. -/
theorem InitialMonoClass.of_initial [HasInitial C] (h : ∀ X : C, Mono (initial.to X)) :
    InitialMonoClass C :=
  InitialMonoClass.of_isInitial initialIsInitial h


/-- To show a category is an `InitialMonoClass` it suffices to show the unique morphism from the
initial object to a terminal object is a monomorphism. -/
theorem InitialMonoClass.of_terminal [HasInitial C] [HasTerminal C] (h : Mono (initial.to (⊤_ C))) :
    InitialMonoClass C :=
  InitialMonoClass.of_isTerminal initialIsInitial terminalIsTerminal h


/-- The comparison morphism from the image of a terminal object to the terminal object in the target
category.
This is an isomorphism iff `G` preserves terminal objects, see
`CategoryTheory.Limits.PreservesTerminal.ofIsoComparison`.
-/
def terminalComparison [HasTerminal C] [HasTerminal D] : G.obj (⊤_ C) ⟶ ⊤_ D :=
  terminal.from _

-- TODO: Show this is an isomorphism if and only if `G` preserves initial objects.

/--
The comparison morphism from the initial object in the target category to the image of the initial
object.
-/
def initialComparison [HasInitial C] [HasInitial D] : ⊥_ D ⟶ G.obj (⊥_ C) :=
  initial.to _


instance hasLimit_of_domain_hasInitial [HasInitial J] {F : J ⥤ C} : HasLimit F :=
  HasLimit.mk { cone := _, isLimit := limitOfDiagramInitial (initialIsInitial) F }

-- See note [dsimp, simp]
-- This is reducible to allow usage of lemmas about `cone_point_unique_up_to_iso`.

/-- For a functor `F : J ⥤ C`, if `J` has an initial object then the image of it is isomorphic
to the limit of `F`. -/
abbrev limitOfInitial (F : J ⥤ C) [HasInitial J] : limit F ≅ F.obj (⊥_ J) :=
  IsLimit.conePointUniqueUpToIso (limit.isLimit _) (limitOfDiagramInitial initialIsInitial F)


instance hasLimit_of_domain_hasTerminal [HasTerminal J] {F : J ⥤ C}
    [∀ (i j : J) (f : i ⟶ j), IsIso (F.map f)] : HasLimit F :=
  HasLimit.mk { cone := _, isLimit := limitOfDiagramTerminal (terminalIsTerminal) F }

-- This is reducible to allow usage of lemmas about `cone_point_unique_up_to_iso`.

/-- For a functor `F : J ⥤ C`, if `J` has a terminal object and all the morphisms in the diagram
are isomorphisms, then the image of the terminal object is isomorphic to the limit of `F`. -/
abbrev limitOfTerminal (F : J ⥤ C) [HasTerminal J] [∀ (i j : J) (f : i ⟶ j), IsIso (F.map f)] :
    limit F ≅ F.obj (⊤_ J) :=
  IsLimit.conePointUniqueUpToIso (limit.isLimit _) (limitOfDiagramTerminal terminalIsTerminal F)


instance hasColimit_of_domain_hasTerminal [HasTerminal J] {F : J ⥤ C} : HasColimit F :=
  HasColimit.mk { cocone := _, isColimit := colimitOfDiagramTerminal (terminalIsTerminal) F }

-- This is reducible to allow usage of lemmas about `cocone_point_unique_up_to_iso`.

/-- For a functor `F : J ⥤ C`, if `J` has a terminal object then the image of it is isomorphic
to the colimit of `F`. -/
abbrev colimitOfTerminal (F : J ⥤ C) [HasTerminal J] : colimit F ≅ F.obj (⊤_ J) :=
  IsColimit.coconePointUniqueUpToIso (colimit.isColimit _)
    (colimitOfDiagramTerminal terminalIsTerminal F)


instance hasColimit_of_domain_hasInitial [HasInitial J] {F : J ⥤ C}
    [∀ (i j : J) (f : i ⟶ j), IsIso (F.map f)] : HasColimit F :=
  HasColimit.mk { cocone := _, isColimit := colimitOfDiagramInitial (initialIsInitial) F }

-- This is reducible to allow usage of lemmas about `cocone_point_unique_up_to_iso`.

/-- For a functor `F : J ⥤ C`, if `J` has an initial object and all the morphisms in the diagram
are isomorphisms, then the image of the initial object is isomorphic to the colimit of `F`. -/
abbrev colimitOfInitial (F : J ⥤ C) [HasInitial J] [∀ (i j : J) (f : i ⟶ j), IsIso (F.map f)] :
    colimit F ≅ F.obj (⊥_ J) :=
  IsColimit.coconePointUniqueUpToIso (colimit.isColimit _)
    (colimitOfDiagramInitial initialIsInitial _)


/-- If `j` is initial in the index category, then the map `limit.π F j` is an isomorphism.
-/
theorem isIso_π_of_isInitial {j : J} (I : IsInitial j) (F : J ⥤ C) [HasLimit F] :
    IsIso (limit.π F j) :=
                                                 /-
                                                   C : Type u₁
                                                   inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                   J : Type u
                                                   inst✝¹ : CategoryTheory.Category.{v, u} J
                                                   j : J
                                                   I : CategoryTheory.Limits.IsInitial j
                                                   F : CategoryTheory.Functor J C
                                                   inst✝ : CategoryTheory.Limits.HasLimit F
                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π F j) ( …
                                                 -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  ⟨⟨limit.lift _ (coneOfDiagramInitial I F), ⟨by ext; simp, by simp⟩⟩⟩
                                                               /-
                                                                 🎉 no goals
                                                               -/


instance isIso_π_initial [HasInitial J] (F : J ⥤ C) : IsIso (limit.π F (⊥_ J)) :=
  isIso_π_of_isInitial initialIsInitial F


theorem isIso_π_of_isTerminal {j : J} (I : IsTerminal j) (F : J ⥤ C) [HasLimit F]
    [∀ (i j : J) (f : i ⟶ j), IsIso (F.map f)] : IsIso (limit.π F j) :=
                                                 /-
                                                   C : Type u₁
                                                   inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                                   J : Type u
                                                   inst✝² : CategoryTheory.Category.{v, u} J
                                                   j : J
                                                   I : CategoryTheory.Limits.IsTerminal j
                                                   F : CategoryTheory.Functor J C
                                                   inst✝¹ : CategoryTheory.Limits.HasLimit F
                                                   inst✝ : ∀ (i j : J) (f : Quiver.Hom i j), CategoryTheory.IsIso (F.map f)
                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π F j) ( …
                                                 -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  ⟨⟨limit.lift _ (coneOfDiagramTerminal I F), by ext; simp, by simp⟩⟩
                                                               /-
                                                                 🎉 no goals
                                                               -/


instance isIso_π_terminal [HasTerminal J] (F : J ⥤ C) [∀ (i j : J) (f : i ⟶ j), IsIso (F.map f)] :
    IsIso (limit.π F (⊤_ J)) :=
  isIso_π_of_isTerminal terminalIsTerminal F


/-- If `j` is terminal in the index category, then the map `colimit.ι F j` is an isomorphism.
-/
theorem isIso_ι_of_isTerminal {j : J} (I : IsTerminal j) (F : J ⥤ C) [HasColimit F] :
    IsIso (colimit.ι F j) :=
                                                      /-
                                                        C : Type u₁
                                                        inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                        J : Type u
                                                        inst✝¹ : CategoryTheory.Category.{v, u} J
                                                        j : J
                                                        I : CategoryTheory.Limits.IsTerminal j
                                                        F : CategoryTheory.Functor J C
                                                        inst✝ : CategoryTheory.Limits.HasColimit F
                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι F j) …
                                                      -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  ⟨⟨colimit.desc _ (coconeOfDiagramTerminal I F), ⟨by simp, by ext; simp⟩⟩⟩
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


instance isIso_ι_terminal [HasTerminal J] (F : J ⥤ C) : IsIso (colimit.ι F (⊤_ J)) :=
  isIso_ι_of_isTerminal terminalIsTerminal F


theorem isIso_ι_of_isInitial {j : J} (I : IsInitial j) (F : J ⥤ C) [HasColimit F]
    [∀ (i j : J) (f : i ⟶ j), IsIso (F.map f)] : IsIso (colimit.ι F j) :=
  ⟨⟨colimit.desc _ (coconeOfDiagramInitial I F), by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      J : Type u
      inst✝² : CategoryTheory.Category.{v, u} J
      j : J
      I : CategoryTheory.Limits.IsInitial j
      F : CategoryTheory.Functor J C
      inst✝¹ : CategoryTheory.Limits.HasColimit F
      inst✝ : ∀ (i j : J) (f : Quiver.Hom i j), CategoryTheory.IsIso (F.map f)
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι …
    -/
    refine ⟨?_, by ext; simp⟩
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      J : Type u
      inst✝² : CategoryTheory.Category.{v, u} J
      j : J
      I : CategoryTheory.Limits.IsInitial j
      F : CategoryTheory.Functor J C
      inst✝¹ : CategoryTheory.Limits.HasColimit F
      inst✝ : ∀ (i j : J) (f : Quiver.Hom i j), CategoryTheory.IsIso (F.map f)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι F j) …
    -/
    dsimp; simp only [colimit.ι_desc, coconeOfDiagramInitial_pt, coconeOfDiagramInitial_ι_app,
      Functor.const_obj_obj, IsInitial.to_self, Functor.map_id]
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      J : Type u
      inst✝² : CategoryTheory.Category.{v, u} J
      j : J
      I : CategoryTheory.Limits.IsInitial j
      F : CategoryTheory.Functor J C
      inst✝¹ : CategoryTheory.Limits.HasColimit F
      inst✝ : ∀ (i j : J) (f : Quiver.Hom i j), CategoryTheory.IsIso (F.map f)
      ⊢ Eq (CategoryTheory.inv (CategoryTheory.CategoryStruct.id (F.obj j))) (Catego …
    -/
    dsimp [inv]; simp only [Category.id_comp, Category.comp_id, and_self]
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      J : Type u
      inst✝² : CategoryTheory.Category.{v, u} J
      j : J
      I : CategoryTheory.Limits.IsInitial j
      F : CategoryTheory.Functor J C
      inst✝¹ : CategoryTheory.Limits.HasColimit F
      inst✝ : ∀ (i j : J) (f : Quiver.Hom i j), CategoryTheory.IsIso (F.map f)
      ⊢ Eq (Classical.choose ⋯) (CategoryTheory.CategoryStruct.id (F.obj j))
    -/
    apply @Classical.choose_spec _ (fun x => x = 𝟙 F.obj j) _
    /-
      🎉 no goals
    -/
  ⟩⟩


instance isIso_ι_initial [HasInitial J] (F : J ⥤ C) [∀ (i j : J) (f : i ⟶ j), IsIso (F.map f)] :
    IsIso (colimit.ι F (⊥_ J)) :=
  isIso_ι_of_isInitial initialIsInitial F



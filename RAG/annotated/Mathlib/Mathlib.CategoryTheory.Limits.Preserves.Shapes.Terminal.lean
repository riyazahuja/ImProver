/-- The map of an empty cone is a limit iff the mapped object is terminal.
-/
def isLimitMapConeEmptyConeEquiv :
    IsLimit (G.mapCone (asEmptyCone X)) ≃ IsTerminal (G.obj X) :=
  isLimitEmptyConeEquiv D _ _ (eqToIso rfl)


/-- The property of preserving terminal objects expressed in terms of `IsTerminal`. -/
def IsTerminal.isTerminalObj [PreservesLimit (Functor.empty.{0} C) G] (l : IsTerminal X) :
    IsTerminal (G.obj X) :=
  isLimitMapConeEmptyConeEquiv G X (isLimitOfPreserves G l)


/-- The property of reflecting terminal objects expressed in terms of `IsTerminal`. -/
def IsTerminal.isTerminalOfObj [ReflectsLimit (Functor.empty.{0} C) G] (l : IsTerminal (G.obj X)) :
    IsTerminal X :=
  isLimitOfReflects G ((isLimitMapConeEmptyConeEquiv G X).symm l)


/-- A functor that preserves and reflects terminal objects induces an equivalence on
`IsTerminal`. -/
def IsTerminal.isTerminalIffObj [PreservesLimit (Functor.empty.{0} C) G]
    [ReflectsLimit (Functor.empty.{0} C) G] (X : C) :
    IsTerminal X ≃ IsTerminal (G.obj X) where
  toFun := IsTerminal.isTerminalObj G X
  invFun := IsTerminal.isTerminalOfObj G X
                 /-
                   C : Type u₁
                   inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                   D : Type u₂
                   inst✝² : CategoryTheory.Category.{v₂, u₂} D
                   G : CategoryTheory.Functor C D
                   X✝ : C
                   inst✝¹ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Functor.empty C) G
                   inst✝ : CategoryTheory.Limits.ReflectsLimit (CategoryTheory.Functor.empty C) G
                   X : C
                   ⊢ Function.LeftInverse (CategoryTheory.Limits.IsTerminal.isTerminalOfObj G X)  …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    C : Type u₁
                    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                    D : Type u₂
                    inst✝² : CategoryTheory.Category.{v₂, u₂} D
                    G : CategoryTheory.Functor C D
                    X✝ : C
                    inst✝¹ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Functor.empty C) G
                    inst✝ : CategoryTheory.Limits.ReflectsLimit (CategoryTheory.Functor.empty C) G
                    X : C
                    ⊢ Function.RightInverse (CategoryTheory.Limits.IsTerminal.isTerminalOfObj G X) …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


/-- Preserving the terminal object implies preserving all limits of the empty diagram. -/
lemma preservesLimitsOfShape_pempty_of_preservesTerminal [PreservesLimit (Functor.empty.{0} C) G] :
    PreservesLimitsOfShape (Discrete PEmpty.{1}) G where
  preservesLimit := preservesLimit_of_iso_diagram G (Functor.emptyExt (Functor.empty.{0} C) _)


/--
If `G` preserves the terminal object and `C` has a terminal object, then the image of the terminal
object is terminal.
-/
def isLimitOfHasTerminalOfPreservesLimit [PreservesLimit (Functor.empty.{0} C) G] :
    IsTerminal (G.obj (⊤_ C)) :=
  terminalIsTerminal.isTerminalObj G (⊤_ C)


/-- If `C` has a terminal object and `G` preserves terminal objects, then `D` has a terminal object
also.
Note this property is somewhat unique to (co)limits of the empty diagram: for general `J`, if `C`
has limits of shape `J` and `G` preserves them, then `D` does not necessarily have limits of shape
`J`.
-/
theorem hasTerminal_of_hasTerminal_of_preservesLimit [PreservesLimit (Functor.empty.{0} C) G] :
    HasTerminal D := ⟨fun F => by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasTerminal C
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Functor.empty C) G
    F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) D
    ⊢ CategoryTheory.Limits.HasLimit F
  -/
  haveI := HasLimit.mk ⟨_, isLimitOfHasTerminalOfPreservesLimit G⟩
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasTerminal C
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Functor.empty C) G
    F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) D
    this : CategoryTheory.Limits.HasLimit (CategoryTheory.Functor.empty D)
    ⊢ CategoryTheory.Limits.HasLimit F
  -/
  apply hasLimitOfIso F.uniqueFromEmpty.symm⟩
  /-
    🎉 no goals
  -/


/-- If the terminal comparison map for `G` is an isomorphism, then `G` preserves terminal objects.
-/
lemma PreservesTerminal.of_iso_comparison [i : IsIso (terminalComparison G)] :
    PreservesLimit (Functor.empty.{0} C) G := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasTerminal C
    inst✝ : CategoryTheory.Limits.HasTerminal D
    i : CategoryTheory.IsIso (CategoryTheory.Limits.terminalComparison G)
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Functor.empty C) G
  -/
  apply preservesLimit_of_preserves_limit_cone terminalIsTerminal
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasTerminal C
    inst✝ : CategoryTheory.Limits.HasTerminal D
    i : CategoryTheory.IsIso (CategoryTheory.Limits.terminalComparison G)
    ⊢ CategoryTheory.Limits.IsLimit (G.mapCone (CategoryTheory.Limits.asEmptyCone  …
  -/
  apply (isLimitMapConeEmptyConeEquiv _ _).symm _
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasTerminal C
    inst✝ : CategoryTheory.Limits.HasTerminal D
    i : CategoryTheory.IsIso (CategoryTheory.Limits.terminalComparison G)
    ⊢ CategoryTheory.Limits.IsTerminal (G.obj (CategoryTheory.Limits.terminal C))
  -/
  exact @IsLimit.ofPointIso _ _ _ _ _ _ _ (limit.isLimit (Functor.empty.{0} D)) i
  /-
    🎉 no goals
  -/


/-- If there is any isomorphism `G.obj ⊤ ⟶ ⊤`, then `G` preserves terminal objects. -/
lemma preservesTerminal_of_isIso (f : G.obj (⊤_ C) ⟶ ⊤_ D) [i : IsIso f] :
    PreservesLimit (Functor.empty.{0} C) G := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasTerminal C
    inst✝ : CategoryTheory.Limits.HasTerminal D
    f : Quiver.Hom (G.obj (CategoryTheory.Limits.terminal C)) (CategoryTheory.Limi …
    i : CategoryTheory.IsIso f
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Functor.empty C) G
  -/
  rw [Subsingleton.elim f (terminalComparison G)] at i
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasTerminal C
    inst✝ : CategoryTheory.Limits.HasTerminal D
    f : Quiver.Hom (G.obj (CategoryTheory.Limits.terminal C)) (CategoryTheory.Limi …
    i : CategoryTheory.IsIso (CategoryTheory.Limits.terminalComparison G)
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Functor.empty C) G
  -/
  exact PreservesTerminal.of_iso_comparison G
  /-
    🎉 no goals
  -/


/-- If there is any isomorphism `G.obj ⊤ ≅ ⊤`, then `G` preserves terminal objects. -/
lemma preservesTerminal_of_iso (f : G.obj (⊤_ C) ≅ ⊤_ D) : PreservesLimit (Functor.empty.{0} C) G :=
  preservesTerminal_of_isIso G f.hom


/-- If `G` preserves terminal objects, then the terminal comparison map for `G` is an isomorphism.
-/
def PreservesTerminal.iso : G.obj (⊤_ C) ≅ ⊤_ D :=
  (isLimitOfHasTerminalOfPreservesLimit G).conePointUniqueUpToIso (limit.isLimit _)


@[simp]
theorem PreservesTerminal.iso_hom : (PreservesTerminal.iso G).hom = terminalComparison G :=
  rfl


instance : IsIso (terminalComparison G) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X : C
    inst✝² : CategoryTheory.Limits.HasTerminal C
    inst✝¹ : CategoryTheory.Limits.HasTerminal D
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Functor.empty C) G
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.terminalComparison G)
  -/
  rw [← PreservesTerminal.iso_hom]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X : C
    inst✝² : CategoryTheory.Limits.HasTerminal C
    inst✝¹ : CategoryTheory.Limits.HasTerminal D
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Functor.empty C) G
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.PreservesTerminal.iso G).hom
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The map of an empty cocone is a colimit iff the mapped object is initial.
-/
def isColimitMapCoconeEmptyCoconeEquiv :
    IsColimit (G.mapCocone (asEmptyCocone.{v₁} X)) ≃ IsInitial (G.obj X) :=
  isColimitEmptyCoconeEquiv D _ _ (eqToIso rfl)


/-- The property of preserving initial objects expressed in terms of `IsInitial`. -/
def IsInitial.isInitialObj [PreservesColimit (Functor.empty.{0} C) G] (l : IsInitial X) :
    IsInitial (G.obj X) :=
  isColimitMapCoconeEmptyCoconeEquiv G X (isColimitOfPreserves G l)


/-- The property of reflecting initial objects expressed in terms of `IsInitial`. -/
def IsInitial.isInitialOfObj [ReflectsColimit (Functor.empty.{0} C) G] (l : IsInitial (G.obj X)) :
    IsInitial X :=
  isColimitOfReflects G ((isColimitMapCoconeEmptyCoconeEquiv G X).symm l)


/-- A functor that preserves and reflects initial objects induces an equivalence on `IsInitial`. -/
def IsInitial.isInitialIffObj [PreservesColimit (Functor.empty.{0} C) G]
    [ReflectsColimit (Functor.empty.{0} C) G] (X : C) :
    IsInitial X ≃ IsInitial (G.obj X) where
  toFun := IsInitial.isInitialObj G X
  invFun := IsInitial.isInitialOfObj G X
                 /-
                   C : Type u₁
                   inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                   D : Type u₂
                   inst✝² : CategoryTheory.Category.{v₂, u₂} D
                   G : CategoryTheory.Functor C D
                   X✝ : C
                   inst✝¹ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Functor.empty  …
                   inst✝ : CategoryTheory.Limits.ReflectsColimit (CategoryTheory.Functor.empty C) G
                   X : C
                   ⊢ Function.LeftInverse (CategoryTheory.Limits.IsInitial.isInitialOfObj G X) (C …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    C : Type u₁
                    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                    D : Type u₂
                    inst✝² : CategoryTheory.Category.{v₂, u₂} D
                    G : CategoryTheory.Functor C D
                    X✝ : C
                    inst✝¹ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Functor.empty  …
                    inst✝ : CategoryTheory.Limits.ReflectsColimit (CategoryTheory.Functor.empty C) G
                    X : C
                    ⊢ Function.RightInverse (CategoryTheory.Limits.IsInitial.isInitialOfObj G X) ( …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


/-- Preserving the initial object implies preserving all colimits of the empty diagram. -/
lemma preservesColimitsOfShape_pempty_of_preservesInitial
    [PreservesColimit (Functor.empty.{0} C) G] :
    PreservesColimitsOfShape (Discrete PEmpty.{1}) G where
  preservesColimit :=
    preservesColimit_of_iso_diagram G (Functor.emptyExt (Functor.empty.{0} C) _)


/-- If `G` preserves the initial object and `C` has an initial object, then the image of the initial
object is initial.
-/
def isColimitOfHasInitialOfPreservesColimit [PreservesColimit (Functor.empty.{0} C) G] :
    IsInitial (G.obj (⊥_ C)) :=
  initialIsInitial.isInitialObj G (⊥_ C)


/-- If `C` has an initial object and `G` preserves initial objects, then `D` has an initial object
also.
Note this property is somewhat unique to colimits of the empty diagram: for general `J`, if `C`
has colimits of shape `J` and `G` preserves them, then `D` does not necessarily have colimits of
shape `J`.
-/
theorem hasInitial_of_hasInitial_of_preservesColimit [PreservesColimit (Functor.empty.{0} C) G] :
    HasInitial D :=
  ⟨fun F => by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.HasInitial C
      inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Functor.empty C …
      F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) D
      ⊢ CategoryTheory.Limits.HasColimit F
    -/
    haveI := HasColimit.mk ⟨_, isColimitOfHasInitialOfPreservesColimit G⟩
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.HasInitial C
      inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Functor.empty C …
      F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) D
      this : CategoryTheory.Limits.HasColimit (CategoryTheory.Functor.empty D)
      ⊢ CategoryTheory.Limits.HasColimit F
    -/
    apply hasColimitOfIso F.uniqueFromEmpty⟩
    /-
      🎉 no goals
    -/


/-- If the initial comparison map for `G` is an isomorphism, then `G` preserves initial objects.
-/
lemma PreservesInitial.of_iso_comparison [i : IsIso (initialComparison G)] :
    PreservesColimit (Functor.empty.{0} C) G := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : CategoryTheory.Limits.HasInitial D
    i : CategoryTheory.IsIso (CategoryTheory.Limits.initialComparison G)
    ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Functor.empty C) G
  -/
  apply preservesColimit_of_preserves_colimit_cocone initialIsInitial
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : CategoryTheory.Limits.HasInitial D
    i : CategoryTheory.IsIso (CategoryTheory.Limits.initialComparison G)
    ⊢ CategoryTheory.Limits.IsColimit (G.mapCocone (CategoryTheory.Limits.asEmptyC …
  -/
  apply (isColimitMapCoconeEmptyCoconeEquiv _ _).symm _
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : CategoryTheory.Limits.HasInitial D
    i : CategoryTheory.IsIso (CategoryTheory.Limits.initialComparison G)
    ⊢ CategoryTheory.Limits.IsInitial (G.obj (CategoryTheory.Limits.initial C))
  -/
  exact @IsColimit.ofPointIso _ _ _ _ _ _ _ (colimit.isColimit (Functor.empty.{0} D)) i
  /-
    🎉 no goals
  -/


/-- If there is any isomorphism `⊥ ⟶ G.obj ⊥`, then `G` preserves initial objects. -/
lemma preservesInitial_of_isIso (f : ⊥_ D ⟶ G.obj (⊥_ C)) [i : IsIso f] :
    PreservesColimit (Functor.empty.{0} C) G := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : CategoryTheory.Limits.HasInitial D
    f : Quiver.Hom (CategoryTheory.Limits.initial D) (G.obj (CategoryTheory.Limits …
    i : CategoryTheory.IsIso f
    ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Functor.empty C) G
  -/
  rw [Subsingleton.elim f (initialComparison G)] at i
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : CategoryTheory.Limits.HasInitial D
    f : Quiver.Hom (CategoryTheory.Limits.initial D) (G.obj (CategoryTheory.Limits …
    i : CategoryTheory.IsIso (CategoryTheory.Limits.initialComparison G)
    ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Functor.empty C) G
  -/
  exact PreservesInitial.of_iso_comparison G
  /-
    🎉 no goals
  -/


/-- If there is any isomorphism `⊥ ≅ G.obj ⊥`, then `G` preserves initial objects. -/
lemma preservesInitial_of_iso (f : ⊥_ D ≅ G.obj (⊥_ C)) :
    PreservesColimit (Functor.empty.{0} C) G :=
  preservesInitial_of_isIso G f.hom


/-- If `G` preserves initial objects, then the initial comparison map for `G` is an isomorphism. -/
def PreservesInitial.iso : G.obj (⊥_ C) ≅ ⊥_ D :=
  (isColimitOfHasInitialOfPreservesColimit G).coconePointUniqueUpToIso (colimit.isColimit _)


@[simp]
theorem PreservesInitial.iso_hom : (PreservesInitial.iso G).inv = initialComparison G :=
  rfl


instance : IsIso (initialComparison G) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X : C
    inst✝² : CategoryTheory.Limits.HasInitial C
    inst✝¹ : CategoryTheory.Limits.HasInitial D
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Functor.empty C …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.initialComparison G)
  -/
  rw [← PreservesInitial.iso_hom]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    G : CategoryTheory.Functor C D
    X : C
    inst✝² : CategoryTheory.Limits.HasInitial C
    inst✝¹ : CategoryTheory.Limits.HasInitial D
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Functor.empty C …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.PreservesInitial.iso G).inv
  -/
  infer_instance
  /-
    🎉 no goals
  -/



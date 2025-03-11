attribute [local simp] eq_iff_true_of_subsingleton in
/-- Implementation: If each structured arrow category on `G` has an initial object, an equivalence
which is helpful for constructing a left adjoint to `G`.
-/
@[simps]
def leftAdjointOfStructuredArrowInitialsAux (A : C) (B : D) :
    ((⊥_ StructuredArrow A G).right ⟶ B) ≃ (A ⟶ G.obj B) where
  toFun g := (⊥_ StructuredArrow A G).hom ≫ G.map g
  invFun f := CommaMorphism.right (initial.to (StructuredArrow.mk f))
  left_inv g := by
    /-
      C : Type u₁
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor D C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasInitial (CategoryTheory.Structured …
      A : C
      B : D
      g : Quiver.Hom (CategoryTheory.Limits.initial (CategoryTheory.StructuredArrow  …
      ⊢ Eq ((fun f => (CategoryTheory.Limits.initial.to (CategoryTheory.StructuredAr …
    -/
    let B' : StructuredArrow A G := StructuredArrow.mk ((⊥_ StructuredArrow A G).hom ≫ G.map g)
    /-
      C : Type u₁
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor D C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasInitial (CategoryTheory.Structured …
      A : C
      B : D
      g : Quiver.Hom (CategoryTheory.Limits.initial (CategoryTheory.StructuredArrow  …
      B' : CategoryTheory.StructuredArrow A G := CategoryTheory.StructuredArrow.mk ( …
      ⊢ Eq ((fun f => (CategoryTheory.Limits.initial.to (CategoryTheory.StructuredAr …
    -/
    let g' : ⊥_ StructuredArrow A G ⟶ B' := StructuredArrow.homMk g rfl
    /-
      C : Type u₁
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor D C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasInitial (CategoryTheory.Structured …
      A : C
      B : D
      g : Quiver.Hom (CategoryTheory.Limits.initial (CategoryTheory.StructuredArrow  …
      B' : CategoryTheory.StructuredArrow A G := CategoryTheory.StructuredArrow.mk ( …
      g' : Quiver.Hom (CategoryTheory.Limits.initial (CategoryTheory.StructuredArrow …
      ⊢ Eq ((fun f => (CategoryTheory.Limits.initial.to (CategoryTheory.StructuredAr …
    -/
    have : initial.to _ = g' := by aesop_cat
    /-
      C : Type u₁
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor D C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasInitial (CategoryTheory.Structured …
      A : C
      B : D
      g : Quiver.Hom (CategoryTheory.Limits.initial (CategoryTheory.StructuredArrow  …
      B' : CategoryTheory.StructuredArrow A G := CategoryTheory.StructuredArrow.mk ( …
      g' : Quiver.Hom (CategoryTheory.Limits.initial (CategoryTheory.StructuredArrow …
      this : Eq (CategoryTheory.Limits.initial.to B') g'
      ⊢ Eq ((fun f => (CategoryTheory.Limits.initial.to (CategoryTheory.StructuredAr …
    -/
    change CommaMorphism.right (initial.to B') = _
    /-
      C : Type u₁
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor D C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasInitial (CategoryTheory.Structured …
      A : C
      B : D
      g : Quiver.Hom (CategoryTheory.Limits.initial (CategoryTheory.StructuredArrow  …
      B' : CategoryTheory.StructuredArrow A G := CategoryTheory.StructuredArrow.mk ( …
      g' : Quiver.Hom (CategoryTheory.Limits.initial (CategoryTheory.StructuredArrow …
      this : Eq (CategoryTheory.Limits.initial.to B') g'
      ⊢ Eq (CategoryTheory.Limits.initial.to B').right g
    -/
    rw [this]
    /-
      C : Type u₁
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor D C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasInitial (CategoryTheory.Structured …
      A : C
      B : D
      g : Quiver.Hom (CategoryTheory.Limits.initial (CategoryTheory.StructuredArrow  …
      B' : CategoryTheory.StructuredArrow A G := CategoryTheory.StructuredArrow.mk ( …
      g' : Quiver.Hom (CategoryTheory.Limits.initial (CategoryTheory.StructuredArrow …
      this : Eq (CategoryTheory.Limits.initial.to B') g'
      ⊢ Eq g'.right g
    -/
    rfl
    /-
      🎉 no goals
    -/
  right_inv f := by
    /-
      C : Type u₁
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor D C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasInitial (CategoryTheory.Structured …
      A : C
      B : D
      f : Quiver.Hom A (G.obj B)
      ⊢ Eq ((fun g => CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.init …
    -/
    let B' : StructuredArrow A G := StructuredArrow.mk f
    /-
      C : Type u₁
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor D C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasInitial (CategoryTheory.Structured …
      A : C
      B : D
      f : Quiver.Hom A (G.obj B)
      B' : CategoryTheory.StructuredArrow A G := CategoryTheory.StructuredArrow.mk f
      ⊢ Eq ((fun g => CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.init …
    -/
    apply (CommaMorphism.w (initial.to B')).symm.trans (Category.id_comp _)
    /-
      🎉 no goals
    -/


/--
If each structured arrow category on `G` has an initial object, construct a left adjoint to `G`. It
is shown that it is a left adjoint in `adjunctionOfStructuredArrowInitials`.
-/
def leftAdjointOfStructuredArrowInitials : C ⥤ D :=
                                                                                          /-
                                                                                            C : Type u₁
                                                                                            D : Type u₂
                                                                                            inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                                                            inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                                                            G : CategoryTheory.Functor D C
                                                                                            inst✝ : ∀ (A : C), CategoryTheory.Limits.HasInitial (CategoryTheory.Structured …
                                                                                            x✝¹ : C
                                                                                            x✝ : D
                                                                                            ⊢ ∀ (Y' : D) (g : Quiver.Hom x✝ Y') (h : Quiver.Hom (CategoryTheory.Limits.ini …
                                                                                          -/
  Adjunction.leftAdjointOfEquiv (leftAdjointOfStructuredArrowInitialsAux G) fun _ _ => by simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


/--
If each structured arrow category on `G` has an initial object, we have a constructed left adjoint
to `G`.
-/
def adjunctionOfStructuredArrowInitials : leftAdjointOfStructuredArrowInitials G ⊣ G :=
  Adjunction.adjunctionOfEquivLeft _ _


/-- If each structured arrow category on `G` has an initial object, `G` is a right adjoint. -/
lemma isRightAdjointOfStructuredArrowInitials : G.IsRightAdjoint where
  exists_leftAdjoint := ⟨_, ⟨adjunctionOfStructuredArrowInitials G⟩⟩


attribute [local simp] eq_iff_true_of_subsingleton in
/-- Implementation: If each costructured arrow category on `G` has a terminal object, an equivalence
which is helpful for constructing a right adjoint to `G`.
-/
@[simps]
def rightAdjointOfCostructuredArrowTerminalsAux (B : D) (A : C) :
    (G.obj B ⟶ A) ≃ (B ⟶ (⊤_ CostructuredArrow G A).left) where
  toFun g := CommaMorphism.left (terminal.from (CostructuredArrow.mk g))
  invFun g := G.map g ≫ (⊤_ CostructuredArrow G A).hom
                 /-
                   C : Type u₁
                   D : Type u₂
                   inst✝² : CategoryTheory.Category.{v₁, u₁} C
                   inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                   G : CategoryTheory.Functor D C
                   inst✝ : ∀ (A : C), CategoryTheory.Limits.HasTerminal (CategoryTheory.Costructu …
                   B : D
                   A : C
                   ⊢ Function.LeftInverse (fun g => CategoryTheory.CategoryStruct.comp (G.map g)  …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
  right_inv g := by
    let B' : CostructuredArrow G A :=
      CostructuredArrow.mk (G.map g ≫ (⊤_ CostructuredArrow G A).hom)
    /-
      C : Type u₁
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor D C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasTerminal (CategoryTheory.Costructu …
      B : D
      A : C
      g : Quiver.Hom B (CategoryTheory.Limits.terminal (CategoryTheory.CostructuredA …
      B' : CategoryTheory.CostructuredArrow G A := CategoryTheory.CostructuredArrow. …
      ⊢ Eq ((fun g => (CategoryTheory.Limits.terminal.from (CategoryTheory.Costructu …
    -/
    let g' : B' ⟶ ⊤_ CostructuredArrow G A := CostructuredArrow.homMk g rfl
    /-
      C : Type u₁
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor D C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasTerminal (CategoryTheory.Costructu …
      B : D
      A : C
      g : Quiver.Hom B (CategoryTheory.Limits.terminal (CategoryTheory.CostructuredA …
      B' : CategoryTheory.CostructuredArrow G A := CategoryTheory.CostructuredArrow. …
      g' : Quiver.Hom B' (CategoryTheory.Limits.terminal (CategoryTheory.Costructure …
      ⊢ Eq ((fun g => (CategoryTheory.Limits.terminal.from (CategoryTheory.Costructu …
    -/
    have : terminal.from _ = g' := by aesop_cat
    /-
      C : Type u₁
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor D C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasTerminal (CategoryTheory.Costructu …
      B : D
      A : C
      g : Quiver.Hom B (CategoryTheory.Limits.terminal (CategoryTheory.CostructuredA …
      B' : CategoryTheory.CostructuredArrow G A := CategoryTheory.CostructuredArrow. …
      g' : Quiver.Hom B' (CategoryTheory.Limits.terminal (CategoryTheory.Costructure …
      this : Eq (CategoryTheory.Limits.terminal.from B') g'
      ⊢ Eq ((fun g => (CategoryTheory.Limits.terminal.from (CategoryTheory.Costructu …
    -/
    change CommaMorphism.left (terminal.from B') = _
    /-
      C : Type u₁
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor D C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasTerminal (CategoryTheory.Costructu …
      B : D
      A : C
      g : Quiver.Hom B (CategoryTheory.Limits.terminal (CategoryTheory.CostructuredA …
      B' : CategoryTheory.CostructuredArrow G A := CategoryTheory.CostructuredArrow. …
      g' : Quiver.Hom B' (CategoryTheory.Limits.terminal (CategoryTheory.Costructure …
      this : Eq (CategoryTheory.Limits.terminal.from B') g'
      ⊢ Eq (CategoryTheory.Limits.terminal.from B').left g
    -/
    rw [this]
    /-
      C : Type u₁
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor D C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasTerminal (CategoryTheory.Costructu …
      B : D
      A : C
      g : Quiver.Hom B (CategoryTheory.Limits.terminal (CategoryTheory.CostructuredA …
      B' : CategoryTheory.CostructuredArrow G A := CategoryTheory.CostructuredArrow. …
      g' : Quiver.Hom B' (CategoryTheory.Limits.terminal (CategoryTheory.Costructure …
      this : Eq (CategoryTheory.Limits.terminal.from B') g'
      ⊢ Eq g'.left g
    -/
    rfl
    /-
      🎉 no goals
    -/


/--
If each costructured arrow category on `G` has a terminal object, construct a right adjoint to `G`.
It is shown that it is a right adjoint in `adjunctionOfStructuredArrowInitials`.
-/
def rightAdjointOfCostructuredArrowTerminals : C ⥤ D :=
  Adjunction.rightAdjointOfEquiv (rightAdjointOfCostructuredArrowTerminalsAux G)
      fun B₁ B₂ A f g => by
    /-
      C : Type u₁
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor D C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasTerminal (CategoryTheory.Costructu …
      B₁ B₂ : D
      A : C
      f : Quiver.Hom B₁ B₂
      g : Quiver.Hom (G.obj B₂) A
      ⊢ Eq ((CategoryTheory.rightAdjointOfCostructuredArrowTerminalsAux G B₁ A) (Cat …
    -/
    rw [← Equiv.eq_symm_apply]
    /-
      C : Type u₁
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor D C
      inst✝ : ∀ (A : C), CategoryTheory.Limits.HasTerminal (CategoryTheory.Costructu …
      B₁ B₂ : D
      A : C
      f : Quiver.Hom B₁ B₂
      g : Quiver.Hom (G.obj B₂) A
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) g) ((CategoryTheory.rightAd …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- If each costructured arrow category on `G` has a terminal object, we have a constructed right
adjoint to `G`.
-/
def adjunctionOfCostructuredArrowTerminals : G ⊣ rightAdjointOfCostructuredArrowTerminals G :=
  Adjunction.adjunctionOfEquivRight _ _


/-- If each costructured arrow category on `G` has a terminal object, `G` is a left adjoint. -/
lemma isLeftAdjoint_of_costructuredArrowTerminals : G.IsLeftAdjoint where
  exists_rightAdjoint :=
    ⟨rightAdjointOfCostructuredArrowTerminals G, ⟨Adjunction.adjunctionOfEquivRight _ _⟩⟩


/-- Given a left adjoint to `G`, we can construct an initial object in each structured arrow
category on `G`. -/
def mkInitialOfLeftAdjoint (h : F ⊣ G) (A : C) :
    IsInitial (StructuredArrow.mk (h.unit.app A) : StructuredArrow A G) where
            /-
              C : Type u₁
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
              inst✝ : CategoryTheory.Category.{v₂, u₂} D
              G : CategoryTheory.Functor D C
              F : CategoryTheory.Functor C D
              h : CategoryTheory.Adjunction F G
              A : C
              B : CategoryTheory.Limits.Cocone (CategoryTheory.Functor.empty (CategoryTheory …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.asEmptyCocone  …
            -/
  desc B := StructuredArrow.homMk ((h.homEquiv _ _).symm B.pt.hom)
            /-
              🎉 no goals
            -/
  uniq s m _ := by
    /-
      C : Type u₁
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor D C
      F : CategoryTheory.Functor C D
      h : CategoryTheory.Adjunction F G
      A : C
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Functor.empty (CategoryTheory …
      m : Quiver.Hom (CategoryTheory.Limits.asEmptyCocone (CategoryTheory.Structured …
      x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{1}), Eq (CategoryTheory.CategorySt …
      ⊢ Eq m ((fun B => CategoryTheory.StructuredArrow.homMk ((h.homEquiv A B.pt.rig …
    -/
    apply StructuredArrow.ext
    /-
      case a
      C : Type u₁
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor D C
      F : CategoryTheory.Functor C D
      h : CategoryTheory.Adjunction F G
      A : C
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Functor.empty (CategoryTheory …
      m : Quiver.Hom (CategoryTheory.Limits.asEmptyCocone (CategoryTheory.Structured …
      x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{1}), Eq (CategoryTheory.CategorySt …
      ⊢ Eq m.right ((fun B => CategoryTheory.StructuredArrow.homMk ((h.homEquiv A B. …
    -/
    simp [← StructuredArrow.w m]
    /-
      🎉 no goals
    -/


/-- Given a right adjoint to `F`, we can construct a terminal object in each costructured arrow
category on `F`. -/
def mkTerminalOfRightAdjoint (h : F ⊣ G) (A : D) :
    IsTerminal (CostructuredArrow.mk (h.counit.app A) : CostructuredArrow F A) where
            /-
              C : Type u₁
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
              inst✝ : CategoryTheory.Category.{v₂, u₂} D
              G : CategoryTheory.Functor D C
              F : CategoryTheory.Functor C D
              h : CategoryTheory.Adjunction F G
              A : D
              B : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty (CategoryTheory.C …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((h.homEquiv B.pt.left A) B.pt …
            -/
  lift B := CostructuredArrow.homMk (h.homEquiv _ _ B.pt.hom)
            /-
              🎉 no goals
            -/
  uniq s m _ := by
    /-
      C : Type u₁
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor D C
      F : CategoryTheory.Functor C D
      h : CategoryTheory.Adjunction F G
      A : D
      s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty (CategoryTheory.C …
      m : Quiver.Hom s.pt (CategoryTheory.Limits.asEmptyCone (CategoryTheory.Costruc …
      x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{1}), Eq (CategoryTheory.CategorySt …
      ⊢ Eq m ((fun B => CategoryTheory.CostructuredArrow.homMk ((h.homEquiv B.pt.lef …
    -/
    apply CostructuredArrow.ext
    /-
      case h
      C : Type u₁
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      G : CategoryTheory.Functor D C
      F : CategoryTheory.Functor C D
      h : CategoryTheory.Adjunction F G
      A : D
      s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty (CategoryTheory.C …
      m : Quiver.Hom s.pt (CategoryTheory.Limits.asEmptyCone (CategoryTheory.Costruc …
      x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{1}), Eq (CategoryTheory.CategorySt …
      ⊢ Eq m.left ((fun B => CategoryTheory.CostructuredArrow.homMk ((h.homEquiv B.p …
    -/
    simp [← CostructuredArrow.w m]
    /-
      🎉 no goals
    -/


theorem isRightAdjoint_iff_hasInitial_structuredArrow {G : D ⥤ C} :
    G.IsRightAdjoint ↔ ∀ A, HasInitial (StructuredArrow A G) :=
  ⟨fun _ A => (mkInitialOfLeftAdjoint _ (Adjunction.ofIsRightAdjoint G) A).hasInitial,
    fun _ => isRightAdjointOfStructuredArrowInitials _⟩


theorem isLeftAdjoint_iff_hasTerminal_costructuredArrow {F : C ⥤ D} :
    F.IsLeftAdjoint ↔ ∀ A, HasTerminal (CostructuredArrow F A) :=
  ⟨fun _ A => (mkTerminalOfRightAdjoint _ (Adjunction.ofIsLeftAdjoint F) A).hasTerminal,
    fun _ => isLeftAdjoint_of_costructuredArrowTerminals _⟩



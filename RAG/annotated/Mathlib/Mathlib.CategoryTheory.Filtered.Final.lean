/-- If `StructuredArrow d F` is filtered for any `d : D`, then `F : C ⥤ D` is final. This is
    simply because filtered categories are connected. More profoundly, the converse is also true if
    `C` is filtered, see `final_iff_isFiltered_structuredArrow`. -/
theorem Functor.final_of_isFiltered_structuredArrow [∀ d, IsFiltered (StructuredArrow d F)] :
    Final F where
  out _ := IsFiltered.isConnected _


/-- If `CostructuredArrow F d` is filtered for any `d : D`, then `F : C ⥤ D` is initial. This is
    simply because cofiltered categories are connectged. More profoundly, the converse is also true
    if `C` is cofiltered, see `initial_iff_isCofiltered_costructuredArrow`. -/
theorem Functor.initial_of_isCofiltered_costructuredArrow
    [∀ d, IsCofiltered (CostructuredArrow F d)] : Initial F where
  out _ := IsCofiltered.isConnected _


theorem isFiltered_structuredArrow_of_isFiltered_of_exists [IsFilteredOrEmpty C]
    (h₁ : ∀ d, ∃ c, Nonempty (d ⟶ F.obj c)) (h₂ : ∀ {d : D} {c : C} (s s' : d ⟶ F.obj c),
      ∃ (c' : C) (t : c ⟶ c'), s ≫ F.map t = s' ≫ F.map t) (d : D) :
    IsFiltered (StructuredArrow d F) := by
  have : Nonempty (StructuredArrow d F) := by
    obtain ⟨c, ⟨f⟩⟩ := h₁ d
    exact ⟨.mk f⟩
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.IsFilteredOrEmpty C
    h₁ : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom d (F.obj c))
    h₂ : ∀ {d : D} {c : C} (s s' : Quiver.Hom d (F.obj c)), Exists fun c' => Exist …
    d : D
    this : Nonempty (CategoryTheory.StructuredArrow d F)
    ⊢ CategoryTheory.IsFiltered (CategoryTheory.StructuredArrow d F)
  -/
  suffices IsFilteredOrEmpty (StructuredArrow d F) from IsFiltered.mk
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.IsFilteredOrEmpty C
    h₁ : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom d (F.obj c))
    h₂ : ∀ {d : D} {c : C} (s s' : Quiver.Hom d (F.obj c)), Exists fun c' => Exist …
    d : D
    this : Nonempty (CategoryTheory.StructuredArrow d F)
    ⊢ CategoryTheory.IsFilteredOrEmpty (CategoryTheory.StructuredArrow d F)
  -/
  refine ⟨fun f g => ?_, fun f g η μ => ?_⟩
  · obtain ⟨c, ⟨t, ht⟩⟩ := h₂ (f.hom ≫ F.map (IsFiltered.leftToMax f.right g.right))
        (g.hom ≫ F.map (IsFiltered.rightToMax f.right g.right))
    /-
      case refine_1.intro.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.IsFilteredOrEmpty C
      h₁ : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom d (F.obj c))
      h₂ : ∀ {d : D} {c : C} (s s' : Quiver.Hom d (F.obj c)), Exists fun c' => Exist …
      d : D
      this : Nonempty (CategoryTheory.StructuredArrow d F)
      f g : CategoryTheory.StructuredArrow d F
      c : C
      t : Quiver.Hom (CategoryTheory.IsFiltered.max f.right g.right) c
      ht : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
      ⊢ Exists fun Z => Exists fun x => Exists fun x => True
    -/
    refine ⟨.mk (f.hom ≫ F.map (IsFiltered.leftToMax f.right g.right ≫ t)), ?_, ?_, trivial⟩
      /-
        case refine_1.intro.intro.refine_1
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝ : CategoryTheory.IsFilteredOrEmpty C
        h₁ : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom d (F.obj c))
        h₂ : ∀ {d : D} {c : C} (s s' : Quiver.Hom d (F.obj c)), Exists fun c' => Exist …
        d : D
        this : Nonempty (CategoryTheory.StructuredArrow d F)
        f g : CategoryTheory.StructuredArrow d F
        c : C
        t : Quiver.Hom (CategoryTheory.IsFiltered.max f.right g.right) c
        ht : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
        ⊢ Quiver.Hom f (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStru …
      -/
    · exact StructuredArrow.homMk (IsFiltered.leftToMax _ _ ≫ t) rfl
      /-
        🎉 no goals
      -/
      /-
        case refine_1.intro.intro.refine_2
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝ : CategoryTheory.IsFilteredOrEmpty C
        h₁ : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom d (F.obj c))
        h₂ : ∀ {d : D} {c : C} (s s' : Quiver.Hom d (F.obj c)), Exists fun c' => Exist …
        d : D
        this : Nonempty (CategoryTheory.StructuredArrow d F)
        f g : CategoryTheory.StructuredArrow d F
        c : C
        t : Quiver.Hom (CategoryTheory.IsFiltered.max f.right g.right) c
        ht : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
        ⊢ Quiver.Hom g (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStru …
      -/
    · exact StructuredArrow.homMk (IsFiltered.rightToMax _ _ ≫ t) (by simpa using ht.symm)
      /-
        🎉 no goals
      -/
  · refine ⟨.mk (f.hom ≫ F.map (η.right ≫ IsFiltered.coeqHom η.right μ.right)),
      StructuredArrow.homMk (IsFiltered.coeqHom η.right μ.right) (by simp), ?_⟩
    /-
      case refine_2
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.IsFilteredOrEmpty C
      h₁ : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom d (F.obj c))
      h₂ : ∀ {d : D} {c : C} (s s' : Quiver.Hom d (F.obj c)), Exists fun c' => Exist …
      d : D
      this : Nonempty (CategoryTheory.StructuredArrow d F)
      f g : CategoryTheory.StructuredArrow d F
      η μ : Quiver.Hom f g
      ⊢ Eq (CategoryTheory.CategoryStruct.comp η (CategoryTheory.StructuredArrow.hom …
    -/
    simpa using IsFiltered.coeq_condition _ _
    /-
      🎉 no goals
    -/


theorem isCofiltered_costructuredArrow_of_isCofiltered_of_exists [IsCofilteredOrEmpty C]
    (h₁ : ∀ d, ∃ c, Nonempty (F.obj c ⟶ d)) (h₂ : ∀ {d : D} {c : C} (s s' : F.obj c ⟶ d),
      ∃ (c' : C) (t : c' ⟶ c), F.map t ≫ s = F.map t ≫ s') (d : D) :
    IsCofiltered (CostructuredArrow F d) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.IsCofilteredOrEmpty C
    h₁ : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)
    h₂ : ∀ {d : D} {c : C} (s s' : Quiver.Hom (F.obj c) d), Exists fun c' => Exist …
    d : D
    ⊢ CategoryTheory.IsCofiltered (CategoryTheory.CostructuredArrow F d)
  -/
  suffices IsFiltered (CostructuredArrow F d)ᵒᵖ from isCofiltered_of_isFiltered_op _
  suffices IsFiltered (StructuredArrow (op d) F.op) from
    IsFiltered.of_equivalence (costructuredArrowOpEquivalence _ _).symm
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.IsCofilteredOrEmpty C
    h₁ : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)
    h₂ : ∀ {d : D} {c : C} (s s' : Quiver.Hom (F.obj c) d), Exists fun c' => Exist …
    d : D
    ⊢ CategoryTheory.IsFiltered (CategoryTheory.StructuredArrow { unop := d } F.op)
  -/
  apply isFiltered_structuredArrow_of_isFiltered_of_exists
    /-
      case h₁
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.IsCofilteredOrEmpty C
      h₁ : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)
      h₂ : ∀ {d : D} {c : C} (s s' : Quiver.Hom (F.obj c) d), Exists fun c' => Exist …
      d : D
      ⊢ ∀ (d : Opposite D), Exists fun c => Nonempty (Quiver.Hom d (F.op.obj c))
    -/
  · intro d
    /-
      case h₁
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.IsCofilteredOrEmpty C
      h₁ : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)
      h₂ : ∀ {d : D} {c : C} (s s' : Quiver.Hom (F.obj c) d), Exists fun c' => Exist …
      d✝ : D
      d : Opposite D
      ⊢ Exists fun c => Nonempty (Quiver.Hom d (F.op.obj c))
    -/
    obtain ⟨c, ⟨t⟩⟩ := h₁ d.unop
    /-
      case h₁.intro.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.IsCofilteredOrEmpty C
      h₁ : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)
      h₂ : ∀ {d : D} {c : C} (s s' : Quiver.Hom (F.obj c) d), Exists fun c' => Exist …
      d✝ : D
      d : Opposite D
      c : C
      t : Quiver.Hom (F.obj c) (Opposite.unop d)
      ⊢ Exists fun c => Nonempty (Quiver.Hom d (F.op.obj c))
    -/
    exact ⟨op c, ⟨Quiver.Hom.op t⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case h₂
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.IsCofilteredOrEmpty C
      h₁ : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)
      h₂ : ∀ {d : D} {c : C} (s s' : Quiver.Hom (F.obj c) d), Exists fun c' => Exist …
      d : D
      ⊢ ∀ {d : Opposite D} {c : Opposite C} (s s' : Quiver.Hom d (F.op.obj c)), Exis …
    -/
  · intro d c s s'
    /-
      case h₂
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.IsCofilteredOrEmpty C
      h₁ : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)
      h₂ : ∀ {d : D} {c : C} (s s' : Quiver.Hom (F.obj c) d), Exists fun c' => Exist …
      d✝ : D
      d : Opposite D
      c : Opposite C
      s s' : Quiver.Hom d (F.op.obj c)
      ⊢ Exists fun c' => Exists fun t => Eq (CategoryTheory.CategoryStruct.comp s (F …
    -/
    obtain ⟨c', t, ht⟩ := h₂ s.unop s'.unop
    /-
      case h₂.intro.intro
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.IsCofilteredOrEmpty C
      h₁ : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)
      h₂ : ∀ {d : D} {c : C} (s s' : Quiver.Hom (F.obj c) d), Exists fun c' => Exist …
      d✝ : D
      d : Opposite D
      c : Opposite C
      s s' : Quiver.Hom d (F.op.obj c)
      c' : C
      t : Quiver.Hom c' (Opposite.unop c)
      ht : Eq (CategoryTheory.CategoryStruct.comp (F.map t) s.unop) (CategoryTheory. …
      ⊢ Exists fun c' => Exists fun t => Eq (CategoryTheory.CategoryStruct.comp s (F …
    -/
    exact ⟨op c', Quiver.Hom.op t, Quiver.Hom.unop_inj ht⟩
    /-
      🎉 no goals
    -/


/-- If `C` is filtered, then we can give an explicit condition for a functor `F : C ⥤ D` to
    be final. The converse is also true, see `final_iff_of_isFiltered`. -/
theorem Functor.final_of_exists_of_isFiltered [IsFilteredOrEmpty C]
    (h₁ : ∀ d, ∃ c, Nonempty (d ⟶ F.obj c)) (h₂ : ∀ {d : D} {c : C} (s s' : d ⟶ F.obj c),
      ∃ (c' : C) (t : c ⟶ c'), s ≫ F.map t = s' ≫ F.map t) : Functor.Final F := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.IsFilteredOrEmpty C
    h₁ : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom d (F.obj c))
    h₂ : ∀ {d : D} {c : C} (s s' : Quiver.Hom d (F.obj c)), Exists fun c' => Exist …
    ⊢ F.Final
  -/
  suffices ∀ d, IsFiltered (StructuredArrow d F) from final_of_isFiltered_structuredArrow F
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.IsFilteredOrEmpty C
    h₁ : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom d (F.obj c))
    h₂ : ∀ {d : D} {c : C} (s s' : Quiver.Hom d (F.obj c)), Exists fun c' => Exist …
    ⊢ ∀ (d : D), CategoryTheory.IsFiltered (CategoryTheory.StructuredArrow d F)
  -/
  exact isFiltered_structuredArrow_of_isFiltered_of_exists F h₁ h₂
  /-
    🎉 no goals
  -/


/-- The inclusion of a terminal object is final. -/
theorem Functor.final_const_of_isTerminal [IsFiltered C] {X : D} (hX : IsTerminal X) :
    ((Functor.const C).obj X).Final :=
  Functor.final_of_exists_of_isFiltered _ (fun _ => ⟨IsFiltered.nonempty.some, ⟨hX.from _⟩⟩)
    (fun {_ c} _ _ => ⟨c, 𝟙 _, hX.hom_ext _ _⟩)


/-- The inclusion of the terminal object is final. -/
theorem Functor.final_const_terminal [IsFiltered C] [HasTerminal D] :
    ((Functor.const C).obj (⊤_ D)).Final :=
  Functor.final_const_of_isTerminal terminalIsTerminal


/-- If `C` is cofiltered, then we can give an explicit condition for a functor `F : C ⥤ D` to
    be final. The converse is also true, see `initial_iff_of_isCofiltered`. -/
theorem Functor.initial_of_exists_of_isCofiltered [IsCofilteredOrEmpty C]
    (h₁ : ∀ d, ∃ c, Nonempty (F.obj c ⟶ d)) (h₂ : ∀ {d : D} {c : C} (s s' : F.obj c ⟶ d),
      ∃ (c' : C) (t : c' ⟶ c), F.map t ≫ s = F.map t ≫ s') : Functor.Initial F := by
  suffices ∀ d, IsCofiltered (CostructuredArrow F d) from
    initial_of_isCofiltered_costructuredArrow F
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.IsCofilteredOrEmpty C
    h₁ : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)
    h₂ : ∀ {d : D} {c : C} (s s' : Quiver.Hom (F.obj c) d), Exists fun c' => Exist …
    ⊢ ∀ (d : D), CategoryTheory.IsCofiltered (CategoryTheory.CostructuredArrow F d)
  -/
  exact isCofiltered_costructuredArrow_of_isCofiltered_of_exists F h₁ h₂
  /-
    🎉 no goals
  -/


/-- The inclusion of an initial object is initial. -/
theorem Functor.initial_const_of_isInitial [IsCofiltered C] {X : D} (hX : IsInitial X) :
    ((Functor.const C).obj X).Initial :=
  Functor.initial_of_exists_of_isCofiltered _ (fun _ => ⟨IsCofiltered.nonempty.some, ⟨hX.to _⟩⟩)
    (fun {_ c} _ _ => ⟨c, 𝟙 _, hX.hom_ext _ _⟩)


/-- The inclusion of the initial object is initial. -/
theorem Functor.initial_const_initial [IsCofiltered C] [HasInitial D] :
    ((Functor.const C).obj (⊥_ D)).Initial :=
  Functor.initial_const_of_isInitial initialIsInitial


/-- In this situation, `F` is also final, see
    `Functor.final_of_exists_of_isFiltered_of_fullyFaithful`. -/
theorem IsFilteredOrEmpty.of_exists_of_isFiltered_of_fullyFaithful [IsFilteredOrEmpty D] [F.Full]
    [F.Faithful] (h : ∀ d, ∃ c, Nonempty (d ⟶ F.obj c)) : IsFilteredOrEmpty C where
  cocone_objs c c' := by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.IsFilteredOrEmpty D
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      h : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom d (F.obj c))
      c c' : C
      ⊢ Exists fun Z => Exists fun x => Exists fun x => True
    -/
    obtain ⟨c₀, ⟨f⟩⟩ := h (IsFiltered.max (F.obj c) (F.obj c'))
    exact ⟨c₀, F.preimage (IsFiltered.leftToMax _ _ ≫ f),
      F.preimage (IsFiltered.rightToMax _ _ ≫ f), trivial⟩
  cocone_maps {c c'} f g := by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.IsFilteredOrEmpty D
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      h : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom d (F.obj c))
      c c' : C
      f g : Quiver.Hom c c'
      ⊢ Exists fun Z => Exists fun h => Eq (CategoryTheory.CategoryStruct.comp f h)  …
    -/
    obtain ⟨c₀, ⟨f₀⟩⟩ := h (IsFiltered.coeq (F.map f) (F.map g))
    /-
      case intro.intro
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.IsFilteredOrEmpty D
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      h : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom d (F.obj c))
      c c' : C
      f g : Quiver.Hom c c'
      c₀ : C
      f₀ : Quiver.Hom (CategoryTheory.IsFiltered.coeq (F.map f) (F.map g)) (F.obj c₀)
      ⊢ Exists fun Z => Exists fun h => Eq (CategoryTheory.CategoryStruct.comp f h)  …
    -/
    refine ⟨_, F.preimage (IsFiltered.coeqHom (F.map f) (F.map g) ≫ f₀), F.map_injective ?_⟩
    /-
      case intro.intro
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.IsFilteredOrEmpty D
      inst✝¹ : F.Full
      inst✝ : F.Faithful
      h : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom d (F.obj c))
      c c' : C
      f g : Quiver.Hom c c'
      c₀ : C
      f₀ : Quiver.Hom (CategoryTheory.IsFiltered.coeq (F.map f) (F.map g)) (F.obj c₀)
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp f (F.preimage (CategoryTheory. …
    -/
    simp [reassoc_of% (IsFiltered.coeq_condition (F.map f) (F.map g))]
    /-
      🎉 no goals
    -/


/-- In this situation, `F` is also initial, see
    `Functor.initial_of_exists_of_isCofiltered_of_fullyFaithful`. -/
theorem IsCofilteredOrEmpty.of_exists_of_isCofiltered_of_fullyFaithful [IsCofilteredOrEmpty D]
    [F.Full] [F.Faithful] (h : ∀ d, ∃ c, Nonempty (F.obj c ⟶ d)) : IsCofilteredOrEmpty C := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.IsCofilteredOrEmpty D
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    h : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)
    ⊢ CategoryTheory.IsCofilteredOrEmpty C
  -/
  suffices IsFilteredOrEmpty Cᵒᵖ from isCofilteredOrEmpty_of_isFilteredOrEmpty_op _
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.IsCofilteredOrEmpty D
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    h : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)
    ⊢ CategoryTheory.IsFilteredOrEmpty (Opposite C)
  -/
  refine IsFilteredOrEmpty.of_exists_of_isFiltered_of_fullyFaithful F.op (fun d => ?_)
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.IsCofilteredOrEmpty D
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    h : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)
    d : Opposite D
    ⊢ Exists fun c => Nonempty (Quiver.Hom d (F.op.obj c))
  -/
  obtain ⟨c, ⟨f⟩⟩ := h d.unop
  /-
    case intro.intro
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.IsCofilteredOrEmpty D
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    h : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)
    d : Opposite D
    c : C
    f : Quiver.Hom (F.obj c) (Opposite.unop d)
    ⊢ Exists fun c => Nonempty (Quiver.Hom d (F.op.obj c))
  -/
  exact ⟨op c, ⟨f.op⟩⟩
  /-
    🎉 no goals
  -/


/-- In this situation, `F` is also final, see
    `Functor.final_of_exists_of_isFiltered_of_fullyFaithful`. -/
theorem IsFiltered.of_exists_of_isFiltered_of_fullyFaithful [IsFiltered D] [F.Full] [F.Faithful]
    (h : ∀ d, ∃ c, Nonempty (d ⟶ F.obj c)) : IsFiltered C :=
  { IsFilteredOrEmpty.of_exists_of_isFiltered_of_fullyFaithful F h with
    nonempty := by
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.IsFiltered D
        inst✝¹ : F.Full
        inst✝ : F.Faithful
        h : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom d (F.obj c))
        ⊢ Nonempty C
      -/
      have : Nonempty D := IsFiltered.nonempty
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.IsFiltered D
        inst✝¹ : F.Full
        inst✝ : F.Faithful
        h : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom d (F.obj c))
        this : Nonempty D
        ⊢ Nonempty C
      -/
      obtain ⟨c, -⟩ := h (Classical.arbitrary D)
      /-
        case intro
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.IsFiltered D
        inst✝¹ : F.Full
        inst✝ : F.Faithful
        h : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom d (F.obj c))
        this : Nonempty D
        c : C
        ⊢ Nonempty C
      -/
      exact ⟨c⟩ }
      /-
        🎉 no goals
      -/


/-- In this situation, `F` is also initial, see
    `Functor.initial_of_exists_of_isCofiltered_of_fullyFaithful`. -/
theorem IsCofiltered.of_exists_of_isCofiltered_of_fullyFaithful [IsCofiltered D] [F.Full]
    [F.Faithful] (h : ∀ d, ∃ c, Nonempty (F.obj c ⟶ d)) : IsCofiltered C :=
  { IsCofilteredOrEmpty.of_exists_of_isCofiltered_of_fullyFaithful F h with
    nonempty := by
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.IsCofiltered D
        inst✝¹ : F.Full
        inst✝ : F.Faithful
        h : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)
        ⊢ Nonempty C
      -/
      have : Nonempty D := IsCofiltered.nonempty
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.IsCofiltered D
        inst✝¹ : F.Full
        inst✝ : F.Faithful
        h : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)
        this : Nonempty D
        ⊢ Nonempty C
      -/
      obtain ⟨c, -⟩ := h (Classical.arbitrary D)
      /-
        case intro
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝² : CategoryTheory.IsCofiltered D
        inst✝¹ : F.Full
        inst✝ : F.Faithful
        h : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)
        this : Nonempty D
        c : C
        ⊢ Nonempty C
      -/
      exact ⟨c⟩ }
      /-
        🎉 no goals
      -/


/-- In this situation, `C` is also filtered, see
    `IsFilteredOrEmpty.of_exists_of_isFiltered_of_fullyFaithful`. -/
theorem Functor.final_of_exists_of_isFiltered_of_fullyFaithful [IsFilteredOrEmpty D] [F.Full]
    [F.Faithful] (h : ∀ d, ∃ c, Nonempty (d ⟶ F.obj c)) : Final F := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.IsFilteredOrEmpty D
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    h : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom d (F.obj c))
    ⊢ F.Final
  -/
  have := IsFilteredOrEmpty.of_exists_of_isFiltered_of_fullyFaithful F h
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.IsFilteredOrEmpty D
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    h : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom d (F.obj c))
    this : CategoryTheory.IsFilteredOrEmpty C
    ⊢ F.Final
  -/
  refine Functor.final_of_exists_of_isFiltered F h (fun {d c} s s' => ?_)
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.IsFilteredOrEmpty D
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    h : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom d (F.obj c))
    this : CategoryTheory.IsFilteredOrEmpty C
    d : D
    c : C
    s s' : Quiver.Hom d (F.obj c)
    ⊢ Exists fun c' => Exists fun t => Eq (CategoryTheory.CategoryStruct.comp s (F …
  -/
  obtain ⟨c₀, ⟨f⟩⟩ := h (IsFiltered.coeq s s')
  /-
    case intro.intro
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.IsFilteredOrEmpty D
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    h : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom d (F.obj c))
    this : CategoryTheory.IsFilteredOrEmpty C
    d : D
    c : C
    s s' : Quiver.Hom d (F.obj c)
    c₀ : C
    f : Quiver.Hom (CategoryTheory.IsFiltered.coeq s s') (F.obj c₀)
    ⊢ Exists fun c' => Exists fun t => Eq (CategoryTheory.CategoryStruct.comp s (F …
  -/
  refine ⟨c₀, F.preimage (IsFiltered.coeqHom s s' ≫ f), ?_⟩
  /-
    case intro.intro
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.IsFilteredOrEmpty D
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    h : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom d (F.obj c))
    this : CategoryTheory.IsFilteredOrEmpty C
    d : D
    c : C
    s s' : Quiver.Hom d (F.obj c)
    c₀ : C
    f : Quiver.Hom (CategoryTheory.IsFiltered.coeq s s') (F.obj c₀)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp s (F.map (F.preimage (CategoryTheory. …
  -/
  simp [reassoc_of% (IsFiltered.coeq_condition s s')]
  /-
    🎉 no goals
  -/


/-- In this situation, `C` is also cofiltered, see
    `IsCofilteredOrEmpty.of_exists_of_isCofiltered_of_fullyFaithful`. -/
theorem Functor.initial_of_exists_of_isCofiltered_of_fullyFaithful [IsCofilteredOrEmpty D] [F.Full]
    [Faithful F] (h : ∀ d, ∃ c, Nonempty (F.obj c ⟶ d)) : Initial F := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.IsCofilteredOrEmpty D
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    h : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)
    ⊢ F.Initial
  -/
  suffices Final F.op from initial_of_final_op _
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.IsCofilteredOrEmpty D
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    h : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)
    ⊢ F.op.Final
  -/
  refine Functor.final_of_exists_of_isFiltered_of_fullyFaithful F.op (fun d => ?_)
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.IsCofilteredOrEmpty D
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    h : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)
    d : Opposite D
    ⊢ Exists fun c => Nonempty (Quiver.Hom d (F.op.obj c))
  -/
  obtain ⟨c, ⟨f⟩⟩ := h d.unop
  /-
    case intro.intro
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.IsCofilteredOrEmpty D
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    h : ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)
    d : Opposite D
    c : C
    f : Quiver.Hom (F.obj c) (Opposite.unop d)
    ⊢ Exists fun c => Nonempty (Quiver.Hom d (F.op.obj c))
  -/
  exact ⟨op c, ⟨f.op⟩⟩
  /-
    🎉 no goals
  -/


/-- Any under category on a filtered or empty category is filtered.
(Note that under categories are always cofiltered since they have an initial object.) -/
instance IsFiltered.under [IsFilteredOrEmpty C] (c : C) : IsFiltered (Under c) :=
  isFiltered_structuredArrow_of_isFiltered_of_exists _
    (fun c' => ⟨c', ⟨𝟙 _⟩⟩)
    (fun s s' => IsFilteredOrEmpty.cocone_maps s s') c


/-- Any over category on a cofiltered or empty category is cofiltered.
(Note that over categories are always filtered since they have a terminal object.) -/
instance IsCofiltered.over [IsCofilteredOrEmpty C] (c : C) : IsCofiltered (Over c) :=
  isCofiltered_costructuredArrow_of_isCofiltered_of_exists _
    (fun c' => ⟨c', ⟨𝟙 _⟩⟩)
    (fun s s' => IsCofilteredOrEmpty.cone_maps s s') c


/-- The forgetful functor of the under category on any filtered or empty category is final. -/
instance Under.final_forget [IsFilteredOrEmpty C] (c : C) : Final (Under.forget c) :=
  final_of_exists_of_isFiltered _
    (fun c' => ⟨mk (IsFiltered.leftToMax c c'), ⟨IsFiltered.rightToMax c c'⟩⟩)
    (fun {_} {x} s s' => by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝ : CategoryTheory.IsFilteredOrEmpty C
        c x✝ : C
        x : CategoryTheory.Under c
        s s' : Quiver.Hom x✝ ((CategoryTheory.Under.forget c).obj x)
        ⊢ Exists fun c' => Exists fun t => Eq (CategoryTheory.CategoryStruct.comp s (( …
      -/
      use mk (x.hom ≫ IsFiltered.coeqHom s s')
      /-
        case h
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝ : CategoryTheory.IsFilteredOrEmpty C
        c x✝ : C
        x : CategoryTheory.Under c
        s s' : Quiver.Hom x✝ ((CategoryTheory.Under.forget c).obj x)
        ⊢ Exists fun t => Eq (CategoryTheory.CategoryStruct.comp s ((CategoryTheory.Un …
      -/
      use homMk (IsFiltered.coeqHom s s') (by simp)
      /-
        case h
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝ : CategoryTheory.IsFilteredOrEmpty C
        c x✝ : C
        x : CategoryTheory.Under c
        s s' : Quiver.Hom x✝ ((CategoryTheory.Under.forget c).obj x)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp s ((CategoryTheory.Under.forget c).ma …
      -/
      simp only [forget_obj, id_obj, mk_right, const_obj_obj, forget_map, homMk_right]
      /-
        case h
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝ : CategoryTheory.IsFilteredOrEmpty C
        c x✝ : C
        x : CategoryTheory.Under c
        s s' : Quiver.Hom x✝ ((CategoryTheory.Under.forget c).obj x)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp s (CategoryTheory.IsFiltered.coeqHom  …
      -/
      rw [IsFiltered.coeq_condition])
      /-
        🎉 no goals
      -/


/-- The forgetful functor of the over category on any cofiltered or empty category is initial. -/
instance Over.initial_forget [IsCofilteredOrEmpty C] (c : C) : Initial (Over.forget c) :=
  initial_of_exists_of_isCofiltered _
    (fun c' => ⟨mk (IsCofiltered.minToLeft c c'), ⟨IsCofiltered.minToRight c c'⟩⟩)
    (fun {_} {x} s s' => by
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝ : CategoryTheory.IsCofilteredOrEmpty C
        c x✝ : C
        x : CategoryTheory.Over c
        s s' : Quiver.Hom ((CategoryTheory.Over.forget c).obj x) x✝
        ⊢ Exists fun c' => Exists fun t => Eq (CategoryTheory.CategoryStruct.comp ((Ca …
      -/
      use mk (IsCofiltered.eqHom s s' ≫ x.hom)
      /-
        case h
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝ : CategoryTheory.IsCofilteredOrEmpty C
        c x✝ : C
        x : CategoryTheory.Over c
        s s' : Quiver.Hom ((CategoryTheory.Over.forget c).obj x) x✝
        ⊢ Exists fun t => Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Over …
      -/
      use homMk (IsCofiltered.eqHom s s') (by simp)
      /-
        case h
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝ : CategoryTheory.IsCofilteredOrEmpty C
        c x✝ : C
        x : CategoryTheory.Over c
        s s' : Quiver.Hom ((CategoryTheory.Over.forget c).obj x) x✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Over.forget c).map ( …
      -/
      simp only [forget_obj, mk_left, forget_map, homMk_left]
      /-
        case h
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        inst✝ : CategoryTheory.IsCofilteredOrEmpty C
        c x✝ : C
        x : CategoryTheory.Over c
        s s' : Quiver.Hom ((CategoryTheory.Over.forget c).obj x) x✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsCofiltered.eqHom s  …
      -/
      rw [IsCofiltered.eq_condition])
      /-
        🎉 no goals
      -/


/-- If `C` is filtered, then we can give an explicit condition for a functor `F : C ⥤ D` to
    be final. -/
theorem Functor.final_iff_of_isFiltered [IsFilteredOrEmpty C] :
    Final F ↔ (∀ d, ∃ c, Nonempty (d ⟶ F.obj c)) ∧ (∀ {d : D} {c : C} (s s' : d ⟶ F.obj c),
      ∃ (c' : C) (t : c ⟶ c'), s ≫ F.map t = s' ≫ F.map t) := by
  /-
    C : Type v₁
    inst✝² : CategoryTheory.Category.{v₁, v₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.IsFilteredOrEmpty C
    ⊢ Iff F.Final (And (∀ (d : D), Exists fun c => Nonempty (Quiver.Hom d (F.obj c …
  -/
  refine ⟨fun hF => ⟨?_, ?_⟩, fun h => final_of_exists_of_isFiltered F h.1 h.2⟩
    /-
      case refine_1
      C : Type v₁
      inst✝² : CategoryTheory.Category.{v₁, v₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.IsFilteredOrEmpty C
      hF : F.Final
      ⊢ ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom d (F.obj c))
    -/
  · intro d
    /-
      case refine_1
      C : Type v₁
      inst✝² : CategoryTheory.Category.{v₁, v₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.IsFilteredOrEmpty C
      hF : F.Final
      d : D
      ⊢ Exists fun c => Nonempty (Quiver.Hom d (F.obj c))
    -/
    obtain ⟨f⟩ : Nonempty (StructuredArrow d F) := IsConnected.is_nonempty
    /-
      case refine_1.intro
      C : Type v₁
      inst✝² : CategoryTheory.Category.{v₁, v₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.IsFilteredOrEmpty C
      hF : F.Final
      d : D
      f : CategoryTheory.StructuredArrow d F
      ⊢ Exists fun c => Nonempty (Quiver.Hom d (F.obj c))
    -/
    exact ⟨_, ⟨f.hom⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type v₁
      inst✝² : CategoryTheory.Category.{v₁, v₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.IsFilteredOrEmpty C
      hF : F.Final
      ⊢ ∀ {d : D} {c : C} (s s' : Quiver.Hom d (F.obj c)), Exists fun c' => Exists f …
    -/
  · intro d c s s'
    have : colimit.ι (F ⋙ coyoneda.obj (op d)) c s = colimit.ι (F ⋙ coyoneda.obj (op d)) c s' := by
      apply (Final.colimitCompCoyonedaIso F d).toEquiv.injective
      subsingleton
    /-
      case refine_2
      C : Type v₁
      inst✝² : CategoryTheory.Category.{v₁, v₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.IsFilteredOrEmpty C
      hF : F.Final
      d : D
      c : C
      s s' : Quiver.Hom d (F.obj c)
      this : Eq (CategoryTheory.Limits.colimit.ι (F.comp (CategoryTheory.coyoneda.ob …
      ⊢ Exists fun c' => Exists fun t => Eq (CategoryTheory.CategoryStruct.comp s (F …
    -/
    obtain ⟨c', t₁, t₂, h⟩ := (Types.FilteredColimit.colimit_eq_iff.{v₁, v₁, v₁} _).mp this
    /-
      case refine_2.intro.intro.intro
      C : Type v₁
      inst✝² : CategoryTheory.Category.{v₁, v₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.IsFilteredOrEmpty C
      hF : F.Final
      d : D
      c : C
      s s' : Quiver.Hom d (F.obj c)
      this : Eq (CategoryTheory.Limits.colimit.ι (F.comp (CategoryTheory.coyoneda.ob …
      c' : C
      t₁ t₂ : Quiver.Hom c c'
      h : Eq ((F.comp (CategoryTheory.coyoneda.obj { unop := d })).map t₁ s) ((F.com …
      ⊢ Exists fun c' => Exists fun t => Eq (CategoryTheory.CategoryStruct.comp s (F …
    -/
    refine ⟨IsFiltered.coeq t₁ t₂, t₁ ≫ IsFiltered.coeqHom t₁ t₂, ?_⟩
    /-
      case refine_2.intro.intro.intro
      C : Type v₁
      inst✝² : CategoryTheory.Category.{v₁, v₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.IsFilteredOrEmpty C
      hF : F.Final
      d : D
      c : C
      s s' : Quiver.Hom d (F.obj c)
      this : Eq (CategoryTheory.Limits.colimit.ι (F.comp (CategoryTheory.coyoneda.ob …
      c' : C
      t₁ t₂ : Quiver.Hom c c'
      h : Eq ((F.comp (CategoryTheory.coyoneda.obj { unop := d })).map t₁ s) ((F.com …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp s (F.map (CategoryTheory.CategoryStru …
    -/
    conv_rhs => rw [IsFiltered.coeq_condition t₁ t₂]
    /-
      case refine_2.intro.intro.intro
      C : Type v₁
      inst✝² : CategoryTheory.Category.{v₁, v₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.IsFilteredOrEmpty C
      hF : F.Final
      d : D
      c : C
      s s' : Quiver.Hom d (F.obj c)
      this : Eq (CategoryTheory.Limits.colimit.ι (F.comp (CategoryTheory.coyoneda.ob …
      c' : C
      t₁ t₂ : Quiver.Hom c c'
      h : Eq ((F.comp (CategoryTheory.coyoneda.obj { unop := d })).map t₁ s) ((F.com …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp s (F.map (CategoryTheory.CategoryStru …
    -/
    dsimp only [comp_obj, coyoneda_obj_obj, unop_op, Functor.comp_map, coyoneda_obj_map] at h
    /-
      case refine_2.intro.intro.intro
      C : Type v₁
      inst✝² : CategoryTheory.Category.{v₁, v₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.IsFilteredOrEmpty C
      hF : F.Final
      d : D
      c : C
      s s' : Quiver.Hom d (F.obj c)
      this : Eq (CategoryTheory.Limits.colimit.ι (F.comp (CategoryTheory.coyoneda.ob …
      c' : C
      t₁ t₂ : Quiver.Hom c c'
      h : Eq (CategoryTheory.CategoryStruct.comp s (F.map t₁)) (CategoryTheory.Categ …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp s (F.map (CategoryTheory.CategoryStru …
    -/
    simp [reassoc_of% h]
    /-
      🎉 no goals
    -/


/-- If `C` is cofiltered, then we can give an explicit condition for a functor `F : C ⥤ D` to
    be initial. -/
theorem Functor.initial_iff_of_isCofiltered [IsCofilteredOrEmpty C] :
    Initial F ↔ (∀ d, ∃ c, Nonempty (F.obj c ⟶ d)) ∧ (∀ {d : D} {c : C} (s s' : F.obj c ⟶ d),
      ∃ (c' : C) (t : c' ⟶ c), F.map t ≫ s = F.map t ≫ s') := by
  /-
    C : Type v₁
    inst✝² : CategoryTheory.Category.{v₁, v₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.IsCofilteredOrEmpty C
    ⊢ Iff F.Initial (And (∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c …
  -/
  refine ⟨fun hF => ?_, fun h => initial_of_exists_of_isCofiltered F h.1 h.2⟩
  /-
    C : Type v₁
    inst✝² : CategoryTheory.Category.{v₁, v₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.IsCofilteredOrEmpty C
    hF : F.Initial
    ⊢ And (∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)) (∀ {d : D …
  -/
  obtain ⟨h₁, h₂⟩ := F.op.final_iff_of_isFiltered.mp inferInstance
  /-
    case intro
    C : Type v₁
    inst✝² : CategoryTheory.Category.{v₁, v₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.IsCofilteredOrEmpty C
    hF : F.Initial
    h₁ : ∀ (d : Opposite D), Exists fun c => Nonempty (Quiver.Hom d (F.op.obj c))
    h₂ : ∀ {d : Opposite D} {c : Opposite C} (s s' : Quiver.Hom d (F.op.obj c)), E …
    ⊢ And (∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)) (∀ {d : D …
  -/
  refine ⟨?_, ?_⟩
    /-
      case intro.refine_1
      C : Type v₁
      inst✝² : CategoryTheory.Category.{v₁, v₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.IsCofilteredOrEmpty C
      hF : F.Initial
      h₁ : ∀ (d : Opposite D), Exists fun c => Nonempty (Quiver.Hom d (F.op.obj c))
      h₂ : ∀ {d : Opposite D} {c : Opposite C} (s s' : Quiver.Hom d (F.op.obj c)), E …
      ⊢ ∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)
    -/
  · intro d
    /-
      case intro.refine_1
      C : Type v₁
      inst✝² : CategoryTheory.Category.{v₁, v₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.IsCofilteredOrEmpty C
      hF : F.Initial
      h₁ : ∀ (d : Opposite D), Exists fun c => Nonempty (Quiver.Hom d (F.op.obj c))
      h₂ : ∀ {d : Opposite D} {c : Opposite C} (s s' : Quiver.Hom d (F.op.obj c)), E …
      d : D
      ⊢ Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)
    -/
    obtain ⟨c, ⟨t⟩⟩ := h₁ (op d)
    /-
      case intro.refine_1.intro.intro
      C : Type v₁
      inst✝² : CategoryTheory.Category.{v₁, v₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.IsCofilteredOrEmpty C
      hF : F.Initial
      h₁ : ∀ (d : Opposite D), Exists fun c => Nonempty (Quiver.Hom d (F.op.obj c))
      h₂ : ∀ {d : Opposite D} {c : Opposite C} (s s' : Quiver.Hom d (F.op.obj c)), E …
      d : D
      c : Opposite C
      t : Quiver.Hom { unop := d } (F.op.obj c)
      ⊢ Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)
    -/
    exact ⟨c.unop, ⟨t.unop⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      C : Type v₁
      inst✝² : CategoryTheory.Category.{v₁, v₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.IsCofilteredOrEmpty C
      hF : F.Initial
      h₁ : ∀ (d : Opposite D), Exists fun c => Nonempty (Quiver.Hom d (F.op.obj c))
      h₂ : ∀ {d : Opposite D} {c : Opposite C} (s s' : Quiver.Hom d (F.op.obj c)), E …
      ⊢ ∀ {d : D} {c : C} (s s' : Quiver.Hom (F.obj c) d), Exists fun c' => Exists f …
    -/
  · intro d c s s'
    /-
      case intro.refine_2
      C : Type v₁
      inst✝² : CategoryTheory.Category.{v₁, v₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.IsCofilteredOrEmpty C
      hF : F.Initial
      h₁ : ∀ (d : Opposite D), Exists fun c => Nonempty (Quiver.Hom d (F.op.obj c))
      h₂ : ∀ {d : Opposite D} {c : Opposite C} (s s' : Quiver.Hom d (F.op.obj c)), E …
      d : D
      c : C
      s s' : Quiver.Hom (F.obj c) d
      ⊢ Exists fun c' => Exists fun t => Eq (CategoryTheory.CategoryStruct.comp (F.m …
    -/
    obtain ⟨c', t, ht⟩ := h₂ (Quiver.Hom.op s) (Quiver.Hom.op s')
    /-
      case intro.refine_2.intro.intro
      C : Type v₁
      inst✝² : CategoryTheory.Category.{v₁, v₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.IsCofilteredOrEmpty C
      hF : F.Initial
      h₁ : ∀ (d : Opposite D), Exists fun c => Nonempty (Quiver.Hom d (F.op.obj c))
      h₂ : ∀ {d : Opposite D} {c : Opposite C} (s s' : Quiver.Hom d (F.op.obj c)), E …
      d : D
      c : C
      s s' : Quiver.Hom (F.obj c) d
      c' : Opposite C
      t : Quiver.Hom { unop := c } c'
      ht : Eq (CategoryTheory.CategoryStruct.comp s.op (F.op.map t)) (CategoryTheory …
      ⊢ Exists fun c' => Exists fun t => Eq (CategoryTheory.CategoryStruct.comp (F.m …
    -/
    exact ⟨c'.unop, t.unop, Quiver.Hom.op_inj ht⟩
    /-
      🎉 no goals
    -/


theorem Functor.Final.exists_coeq [IsFilteredOrEmpty C] [Final F] {d : D} {c : C}
    (s s' : d ⟶ F.obj c) : ∃ (c' : C) (t : c ⟶ c'), s ≫ F.map t = s' ≫ F.map t :=
  ((final_iff_of_isFiltered F).1 inferInstance).2 s s'


theorem Functor.Initial.exists_eq [IsCofilteredOrEmpty C] [Initial F] {d : D} {c : C}
    (s s' : F.obj c ⟶ d) : ∃ (c' : C) (t : c' ⟶ c), F.map t ≫ s = F.map t ≫ s' :=
  ((initial_iff_of_isCofiltered F).1 inferInstance).2 s s'


/-- If `C` is filtered, then `F : C ⥤ D` is final if and only if `StructuredArrow d F` is filtered
    for all `d : D`. -/
theorem Functor.final_iff_isFiltered_structuredArrow [IsFilteredOrEmpty C] :
    Final F ↔ ∀ d, IsFiltered (StructuredArrow d F) := by
  /-
    C : Type v₁
    inst✝² : CategoryTheory.Category.{v₁, v₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.IsFilteredOrEmpty C
    ⊢ Iff F.Final (∀ (d : D), CategoryTheory.IsFiltered (CategoryTheory.Structured …
  -/
  refine ⟨?_, fun h => final_of_isFiltered_structuredArrow F⟩
  /-
    C : Type v₁
    inst✝² : CategoryTheory.Category.{v₁, v₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.IsFilteredOrEmpty C
    ⊢ F.Final → ∀ (d : D), CategoryTheory.IsFiltered (CategoryTheory.StructuredArr …
  -/
  rw [final_iff_of_isFiltered]
  /-
    C : Type v₁
    inst✝² : CategoryTheory.Category.{v₁, v₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.IsFilteredOrEmpty C
    ⊢ And (∀ (d : D), Exists fun c => Nonempty (Quiver.Hom d (F.obj c))) (∀ {d : D …
  -/
  exact fun h => isFiltered_structuredArrow_of_isFiltered_of_exists F h.1 h.2
  /-
    🎉 no goals
  -/


/-- If `C` is cofiltered, then `F : C ⥤ D` is initial if and only if `CostructuredArrow F d` is
    cofiltered for all `d : D`. -/
theorem Functor.initial_iff_isCofiltered_costructuredArrow [IsCofilteredOrEmpty C] :
    Initial F ↔ ∀ d, IsCofiltered (CostructuredArrow F d) := by
  /-
    C : Type v₁
    inst✝² : CategoryTheory.Category.{v₁, v₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.IsCofilteredOrEmpty C
    ⊢ Iff F.Initial (∀ (d : D), CategoryTheory.IsCofiltered (CategoryTheory.Costru …
  -/
  refine ⟨?_, fun h => initial_of_isCofiltered_costructuredArrow F⟩
  /-
    C : Type v₁
    inst✝² : CategoryTheory.Category.{v₁, v₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.IsCofilteredOrEmpty C
    ⊢ F.Initial → ∀ (d : D), CategoryTheory.IsCofiltered (CategoryTheory.Costructu …
  -/
  rw [initial_iff_of_isCofiltered]
  /-
    C : Type v₁
    inst✝² : CategoryTheory.Category.{v₁, v₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.IsCofilteredOrEmpty C
    ⊢ And (∀ (d : D), Exists fun c => Nonempty (Quiver.Hom (F.obj c) d)) (∀ {d : D …
  -/
  exact fun h => isCofiltered_costructuredArrow_of_isCofiltered_of_exists F h.1 h.2
  /-
    🎉 no goals
  -/


/-- If `C` is filtered, then the structured arrow category on the diagonal functor `C ⥤ C × C`
is filtered as well. -/
instance [IsFiltered C] (X : C × C) : IsFiltered (StructuredArrow X (diag C)) := by
  haveI : ∀ Y, IsFiltered (StructuredArrow Y (Under.forget X.1)) := by
    rw [← final_iff_isFiltered_structuredArrow (Under.forget X.1)]
    infer_instance
  /-
    C : Type v₁
    inst✝² : CategoryTheory.Category.{v₁, v₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.IsFiltered C
    X : Prod C C
    this : ∀ (Y : C), CategoryTheory.IsFiltered (CategoryTheory.StructuredArrow Y  …
    ⊢ CategoryTheory.IsFiltered (CategoryTheory.StructuredArrow X (CategoryTheory. …
  -/
  apply IsFiltered.of_equivalence (StructuredArrow.ofDiagEquivalence X).symm
  /-
    🎉 no goals
  -/


/-- The diagonal functor on any filtered category is final. -/
instance Functor.final_diag_of_isFiltered [IsFiltered C] : Final (Functor.diag C) :=
  final_of_isFiltered_structuredArrow _


/-- If `C` is cofiltered, then the costructured arrow category on the diagonal functor `C ⥤ C × C`
is cofiltered as well. -/
instance [IsCofiltered C] (X : C × C) : IsCofiltered (CostructuredArrow (diag C) X) := by
  haveI : ∀ Y, IsCofiltered (CostructuredArrow (Over.forget X.1) Y) := by
    rw [← initial_iff_isCofiltered_costructuredArrow (Over.forget X.1)]
    infer_instance
  /-
    C : Type v₁
    inst✝² : CategoryTheory.Category.{v₁, v₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.IsCofiltered C
    X : Prod C C
    this : ∀ (Y : C), CategoryTheory.IsCofiltered (CategoryTheory.CostructuredArro …
    ⊢ CategoryTheory.IsCofiltered (CategoryTheory.CostructuredArrow (CategoryTheor …
  -/
  apply IsCofiltered.of_equivalence (CostructuredArrow.ofDiagEquivalence X).symm
  /-
    🎉 no goals
  -/


/-- The diagonal functor on any cofiltered category is initial. -/
instance Functor.initial_diag_of_isFiltered [IsCofiltered C] : Initial (Functor.diag C) :=
  initial_of_isCofiltered_costructuredArrow _


/-- If `C` is filtered, then every functor `F : C ⥤ Discrete PUnit` is final. -/
theorem Functor.final_of_isFiltered_of_pUnit [IsFiltered C] (F : C ⥤ Discrete PUnit) :
    Final F := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.IsFiltered C
    F : CategoryTheory.Functor C (CategoryTheory.Discrete PUnit.{u_1 + 1})
    ⊢ F.Final
  -/
  refine final_of_exists_of_isFiltered F (fun _ => ?_) (fun {_} {c} _ _ => ?_)
    /-
      case refine_1
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.IsFiltered C
      F : CategoryTheory.Functor C (CategoryTheory.Discrete PUnit.{u_1 + 1})
      x✝ : CategoryTheory.Discrete PUnit.{u_1 + 1}
      ⊢ Exists fun c => Nonempty (Quiver.Hom x✝ (F.obj c))
    -/
  · use Classical.choice IsFiltered.nonempty
    /-
      case h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.IsFiltered C
      F : CategoryTheory.Functor C (CategoryTheory.Discrete PUnit.{u_1 + 1})
      x✝ : CategoryTheory.Discrete PUnit.{u_1 + 1}
      ⊢ Nonempty (Quiver.Hom x✝ (F.obj (Classical.choice ⋯)))
    -/
    exact ⟨Discrete.eqToHom (by simp)⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.IsFiltered C
      F : CategoryTheory.Functor C (CategoryTheory.Discrete PUnit.{u_1 + 1})
      x✝² : CategoryTheory.Discrete PUnit.{u_1 + 1}
      c : C
      x✝¹ x✝ : Quiver.Hom x✝² (F.obj c)
      ⊢ Exists fun c' => Exists fun t => Eq (CategoryTheory.CategoryStruct.comp x✝¹  …
    -/
  · use c; use 𝟙 c
    /-
      case h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.IsFiltered C
      F : CategoryTheory.Functor C (CategoryTheory.Discrete PUnit.{u_1 + 1})
      x✝² : CategoryTheory.Discrete PUnit.{u_1 + 1}
      c : C
      x✝¹ x✝ : Quiver.Hom x✝² (F.obj c)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp x✝¹ (F.map (CategoryTheory.CategorySt …
    -/
    apply Subsingleton.elim
    /-
      🎉 no goals
    -/


/-- If `C` is cofiltered, then every functor `F : C ⥤ Discrete PUnit` is initial. -/
theorem Functor.initial_of_isCofiltered_pUnit [IsCofiltered C] (F : C ⥤ Discrete PUnit) :
    Initial F := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.IsCofiltered C
    F : CategoryTheory.Functor C (CategoryTheory.Discrete PUnit.{u_1 + 1})
    ⊢ F.Initial
  -/
  refine initial_of_exists_of_isCofiltered F (fun _ => ?_) (fun {_} {c} _ _ => ?_)
    /-
      case refine_1
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.IsCofiltered C
      F : CategoryTheory.Functor C (CategoryTheory.Discrete PUnit.{u_1 + 1})
      x✝ : CategoryTheory.Discrete PUnit.{u_1 + 1}
      ⊢ Exists fun c => Nonempty (Quiver.Hom (F.obj c) x✝)
    -/
  · use Classical.choice IsCofiltered.nonempty
    /-
      case h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.IsCofiltered C
      F : CategoryTheory.Functor C (CategoryTheory.Discrete PUnit.{u_1 + 1})
      x✝ : CategoryTheory.Discrete PUnit.{u_1 + 1}
      ⊢ Nonempty (Quiver.Hom (F.obj (Classical.choice ⋯)) x✝)
    -/
    exact ⟨Discrete.eqToHom (by simp)⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.IsCofiltered C
      F : CategoryTheory.Functor C (CategoryTheory.Discrete PUnit.{u_1 + 1})
      x✝² : CategoryTheory.Discrete PUnit.{u_1 + 1}
      c : C
      x✝¹ x✝ : Quiver.Hom (F.obj c) x✝²
      ⊢ Exists fun c' => Exists fun t => Eq (CategoryTheory.CategoryStruct.comp (F.m …
    -/
  · use c; use 𝟙 c
    /-
      case h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.IsCofiltered C
      F : CategoryTheory.Functor C (CategoryTheory.Discrete PUnit.{u_1 + 1})
      x✝² : CategoryTheory.Discrete PUnit.{u_1 + 1}
      c : C
      x✝¹ x✝ : Quiver.Hom (F.obj c) x✝²
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
    -/
    apply Subsingleton.elim
    /-
      🎉 no goals
    -/


/-- The functor `StructuredArrow.proj : StructuredArrow Y T ⥤ C` is final if `T : C ⥤ D` is final
and `C` is filtered. -/
instance StructuredArrow.final_proj_of_isFiltered [IsFilteredOrEmpty C]
    (T : C ⥤ D) [Final T] (Y : D) : Final (StructuredArrow.proj Y T) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.IsFilteredOrEmpty C
    T : CategoryTheory.Functor C D
    inst✝ : T.Final
    Y : D
    ⊢ (CategoryTheory.StructuredArrow.proj Y T).Final
  -/
  refine ⟨fun X => ?_⟩
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.IsFilteredOrEmpty C
    T : CategoryTheory.Functor C D
    inst✝ : T.Final
    Y : D
    X : C
    ⊢ CategoryTheory.IsConnected (CategoryTheory.StructuredArrow X (CategoryTheory …
  -/
  rw [isConnected_iff_of_equivalence (ofStructuredArrowProjEquivalence T Y X)]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.IsFilteredOrEmpty C
    T : CategoryTheory.Functor C D
    inst✝ : T.Final
    Y : D
    X : C
    ⊢ CategoryTheory.IsConnected (CategoryTheory.StructuredArrow Y ((CategoryTheor …
  -/
  exact (final_comp (Under.forget X) T).out _
  /-
    🎉 no goals
  -/


/-- The functor `CostructuredArrow.proj : CostructuredArrow Y T ⥤ C` is initial if `T : C ⥤ D` is
initial and `C` is cofiltered. -/
instance CostructuredArrow.initial_proj_of_isCofiltered [IsCofilteredOrEmpty C]
    (T : C ⥤ D) [Initial T] (Y : D) : Initial (CostructuredArrow.proj T Y) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.IsCofilteredOrEmpty C
    T : CategoryTheory.Functor C D
    inst✝ : T.Initial
    Y : D
    ⊢ (CategoryTheory.CostructuredArrow.proj T Y).Initial
  -/
  refine ⟨fun X => ?_⟩
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.IsCofilteredOrEmpty C
    T : CategoryTheory.Functor C D
    inst✝ : T.Initial
    Y : D
    X : C
    ⊢ CategoryTheory.IsConnected (CategoryTheory.CostructuredArrow (CategoryTheory …
  -/
  rw [isConnected_iff_of_equivalence (ofCostructuredArrowProjEquivalence T Y X)]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.IsCofilteredOrEmpty C
    T : CategoryTheory.Functor C D
    inst✝ : T.Initial
    Y : D
    X : C
    ⊢ CategoryTheory.IsConnected (CategoryTheory.CostructuredArrow ((CategoryTheor …
  -/
  exact (initial_comp (Over.forget X) T).out _
  /-
    🎉 no goals
  -/


/-- The functor `StructuredArrow d T ⥤ StructuredArrow e (T ⋙ S)` that `u : e ⟶ S.obj d`
induces via `StructuredArrow.map₂` is final, if `T` and `S` are final and the domain of `T` is
filtered. -/
instance StructuredArrow.final_map₂_id {C : Type v₁} [Category.{v₁} C] [IsFiltered C] {E : Type u₃}
    [Category.{v₁} E] (T : C ⥤ D) [T.Final] (S : D ⥤ E) [S.Final] (d : D) (e : E)
    (u : e ⟶ S.obj d) : Final (map₂ (R' := T ⋙ S) (F := 𝟭 _) u (𝟙 (T ⋙ S))) := by
  /-
    C✝ : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C✝
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    C : Type v₁
    inst✝⁴ : CategoryTheory.Category.{v₁, v₁} C
    inst✝³ : CategoryTheory.IsFiltered C
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₁, u₃} E
    T : CategoryTheory.Functor C D
    inst✝¹ : T.Final
    S : CategoryTheory.Functor D E
    inst✝ : S.Final
    d : D
    e : E
    u : Quiver.Hom e (S.obj d)
    ⊢ (CategoryTheory.StructuredArrow.map₂ u (CategoryTheory.CategoryStruct.id (T. …
  -/
  have := (T ⋙ S).final_iff_isFiltered_structuredArrow.mp inferInstance e
  /-
    C✝ : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C✝
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    C : Type v₁
    inst✝⁴ : CategoryTheory.Category.{v₁, v₁} C
    inst✝³ : CategoryTheory.IsFiltered C
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₁, u₃} E
    T : CategoryTheory.Functor C D
    inst✝¹ : T.Final
    S : CategoryTheory.Functor D E
    inst✝ : S.Final
    d : D
    e : E
    u : Quiver.Hom e (S.obj d)
    this : CategoryTheory.IsFiltered (CategoryTheory.StructuredArrow e (T.comp S))
    ⊢ (CategoryTheory.StructuredArrow.map₂ u (CategoryTheory.CategoryStruct.id (T. …
  -/
  apply final_of_natIso (map₂IsoPreEquivalenceInverseCompProj T S d e u).symm
  /-
    🎉 no goals
  -/


/-- `StructuredArrow.post X T S` is final if `T` and `S` are final and the domain of `T` is
filtered. -/
instance StructuredArrow.final_post {C : Type v₁} [Category.{v₁} C] [IsFiltered C] {E : Type u₃}
    [Category.{v₁} E] (X : D) (T : C ⥤ D) [T.Final] (S : D ⥤ E) [S.Final] : Final (post X T S) := by
  /-
    C✝ : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C✝
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    C : Type v₁
    inst✝⁴ : CategoryTheory.Category.{v₁, v₁} C
    inst✝³ : CategoryTheory.IsFiltered C
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₁, u₃} E
    X : D
    T : CategoryTheory.Functor C D
    inst✝¹ : T.Final
    S : CategoryTheory.Functor D E
    inst✝ : S.Final
    ⊢ (CategoryTheory.StructuredArrow.post X T S).Final
  -/
  apply final_of_natIso (postIsoMap₂ X T S).symm
  /-
    🎉 no goals
  -/


/-- The functor `CostructuredArrow T d ⥤ CostructuredArrow (T ⋙ S) e` that `u : S.obj d ⟶ e`
induces via `CostructuredArrow.map₂` is initial, if `T` and `S` are initial and the domain of `T` is
filtered. -/
instance CostructuredArrow.initial_map₂_id {C : Type v₁} [Category.{v₁} C] [IsCofiltered C]
    {E : Type u₃} [Category.{v₁} E] (T : C ⥤ D) [T.Initial] (S : D ⥤ E) [S.Initial] (d : D) (e : E)
    (u : S.obj d ⟶ e) : Initial (map₂ (F := 𝟭 _) (U := T ⋙ S) (𝟙 (T ⋙ S)) u) := by
  /-
    C✝ : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C✝
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    C : Type v₁
    inst✝⁴ : CategoryTheory.Category.{v₁, v₁} C
    inst✝³ : CategoryTheory.IsCofiltered C
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₁, u₃} E
    T : CategoryTheory.Functor C D
    inst✝¹ : T.Initial
    S : CategoryTheory.Functor D E
    inst✝ : S.Initial
    d : D
    e : E
    u : Quiver.Hom (S.obj d) e
    ⊢ (CategoryTheory.CostructuredArrow.map₂ (CategoryTheory.CategoryStruct.id (T. …
  -/
  have := (T ⋙ S).initial_iff_isCofiltered_costructuredArrow.mp inferInstance e
  /-
    C✝ : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C✝
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    C : Type v₁
    inst✝⁴ : CategoryTheory.Category.{v₁, v₁} C
    inst✝³ : CategoryTheory.IsCofiltered C
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₁, u₃} E
    T : CategoryTheory.Functor C D
    inst✝¹ : T.Initial
    S : CategoryTheory.Functor D E
    inst✝ : S.Initial
    d : D
    e : E
    u : Quiver.Hom (S.obj d) e
    this : CategoryTheory.IsCofiltered (CategoryTheory.CostructuredArrow (T.comp S …
    ⊢ (CategoryTheory.CostructuredArrow.map₂ (CategoryTheory.CategoryStruct.id (T. …
  -/
  apply initial_of_natIso (map₂IsoPreEquivalenceInverseCompProj T S d e u).symm
  /-
    🎉 no goals
  -/


/-- `CostructuredArrow.post T S X` is initial if `T` and `S` are initial and the domain of `T` is
cofiltered. -/
instance CostructuredArrow.initial_post {C : Type v₁} [Category.{v₁} C] [IsCofiltered C]
    {E : Type u₃} [Category.{v₁} E] (X : D) (T : C ⥤ D) [T.Initial] (S : D ⥤ E) [S.Initial] :
    Initial (post T S X) := by
  /-
    C✝ : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C✝
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    C : Type v₁
    inst✝⁴ : CategoryTheory.Category.{v₁, v₁} C
    inst✝³ : CategoryTheory.IsCofiltered C
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₁, u₃} E
    X : D
    T : CategoryTheory.Functor C D
    inst✝¹ : T.Initial
    S : CategoryTheory.Functor D E
    inst✝ : S.Initial
    ⊢ (CategoryTheory.CostructuredArrow.post T S X).Initial
  -/
  apply initial_of_natIso (postIsoMap₂ X T S).symm
  /-
    🎉 no goals
  -/


open IsFiltered in
instance final_eval [∀ s, IsFiltered (I s)] (s : α) : (Pi.eval I s).Final := by
  classical
  apply Functor.final_of_exists_of_isFiltered
  · exact fun i => ⟨Function.update (fun t => nonempty.some) s i, ⟨by simpa using 𝟙 _⟩⟩
  · intro d c f g
    let c't : (∀ s, (c' : I s) × (c s ⟶ c')) := Function.update (fun t => ⟨c t, 𝟙 (c t)⟩)
      s ⟨coeq f g, coeqHom f g⟩
    refine ⟨fun t => (c't t).1, fun t => (c't t).2, ?_⟩
    dsimp only [Pi.eval_obj, Pi.eval_map, c't]
    rw [Function.update_self]
    simpa using coeq_condition _ _


open IsCofiltered in
instance initial_eval [∀ s, IsCofiltered (I s)] (s : α) : (Pi.eval I s).Initial := by
  classical
  apply Functor.initial_of_exists_of_isCofiltered
  · exact fun i => ⟨Function.update (fun t => nonempty.some) s i, ⟨by simpa using 𝟙 _⟩⟩
  · intro d c f g
    let c't : (∀ s, (c' : I s) × (c' ⟶ c s)) := Function.update (fun t => ⟨c t, 𝟙 (c t)⟩)
      s ⟨eq f g, eqHom f g⟩
    refine ⟨fun t => (c't t).1, fun t => (c't t).2, ?_⟩
    dsimp only [Pi.eval_obj, Pi.eval_map, c't]
    rw [Function.update_self]
    simpa using eq_condition _ _


open IsFiltered in
instance final_fst [IsFilteredOrEmpty C] [IsFiltered D] : (Prod.fst C D).Final := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.IsFilteredOrEmpty C
    inst✝ : CategoryTheory.IsFiltered D
    ⊢ (CategoryTheory.Prod.fst C D).Final
  -/
  apply Functor.final_of_exists_of_isFiltered
    /-
      case h₁
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.IsFilteredOrEmpty C
      inst✝ : CategoryTheory.IsFiltered D
      ⊢ ∀ (d : C), Exists fun c => Nonempty (Quiver.Hom d ((CategoryTheory.Prod.fst  …
    -/
  · exact fun c => ⟨(c, nonempty.some), ⟨𝟙 c⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case h₂
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.IsFilteredOrEmpty C
      inst✝ : CategoryTheory.IsFiltered D
      ⊢ ∀ {d : C} {c : Prod C D} (s s' : Quiver.Hom d ((CategoryTheory.Prod.fst C D) …
    -/
  · intro c ⟨c', d'⟩ f g
    /-
      case h₂
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.IsFilteredOrEmpty C
      inst✝ : CategoryTheory.IsFiltered D
      c c' : C
      d' : D
      f g : Quiver.Hom c ((CategoryTheory.Prod.fst C D).obj { fst := c', snd := d' })
      ⊢ Exists fun c'_1 => Exists fun t => Eq (CategoryTheory.CategoryStruct.comp f  …
    -/
    exact ⟨(coeq f g, d'), (coeqHom f g, 𝟙 d'), coeq_condition _ _⟩
    /-
      🎉 no goals
    -/


instance final_snd [IsFiltered C] [IsFilteredOrEmpty D] : (Prod.snd C D).Final :=
  inferInstanceAs ((Prod.braiding C D).functor ⋙ Prod.fst D C).Final


open IsCofiltered in
instance initial_fst [IsCofilteredOrEmpty C] [IsCofiltered D] : (Prod.fst C D).Initial := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.IsCofilteredOrEmpty C
    inst✝ : CategoryTheory.IsCofiltered D
    ⊢ (CategoryTheory.Prod.fst C D).Initial
  -/
  apply Functor.initial_of_exists_of_isCofiltered
    /-
      case h₁
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.IsCofilteredOrEmpty C
      inst✝ : CategoryTheory.IsCofiltered D
      ⊢ ∀ (d : C), Exists fun c => Nonempty (Quiver.Hom ((CategoryTheory.Prod.fst C  …
    -/
  · exact fun c => ⟨(c, nonempty.some), ⟨𝟙 c⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case h₂
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.IsCofilteredOrEmpty C
      inst✝ : CategoryTheory.IsCofiltered D
      ⊢ ∀ {d : C} {c : Prod C D} (s s' : Quiver.Hom ((CategoryTheory.Prod.fst C D).o …
    -/
  · intro c ⟨c', d'⟩ f g
    /-
      case h₂
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.IsCofilteredOrEmpty C
      inst✝ : CategoryTheory.IsCofiltered D
      c c' : C
      d' : D
      f g : Quiver.Hom ((CategoryTheory.Prod.fst C D).obj { fst := c', snd := d' }) c
      ⊢ Exists fun c'_1 => Exists fun t => Eq (CategoryTheory.CategoryStruct.comp (( …
    -/
    exact ⟨(eq f g, d'), (eqHom f g, 𝟙 d'), eq_condition _ _⟩
    /-
      🎉 no goals
    -/


instance initial_snd [IsCofiltered C] [IsCofilteredOrEmpty D] : (Prod.snd C D).Initial :=
  inferInstanceAs ((Prod.braiding C D).functor ⋙ Prod.fst D C).Initial



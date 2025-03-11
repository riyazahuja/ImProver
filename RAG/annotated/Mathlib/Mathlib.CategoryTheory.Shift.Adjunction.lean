/-- Given an adjunction `adj : F ⊣ G`, `a` in `A` and commutation isomorphisms
`e₁ : shiftFunctor C a ⋙ F ≅ F ⋙ shiftFunctor D a` and
`e₂ : shiftFunctor D a ⋙ G ≅ G ⋙ shiftFunctor C a`, this expresses the compatibility of
`e₁` and `e₂` with the unit of the adjunction `adj`.
-/
abbrev CompatibilityUnit :=
  ∀ (X : C), (adj.unit.app X)⟦a⟧' = adj.unit.app (X⟦a⟧) ≫ G.map (e₁.hom.app X) ≫ e₂.hom.app _


/-- Given an adjunction `adj : F ⊣ G`, `a` in `A` and commutation isomorphisms
`e₁ : shiftFunctor C a ⋙ F ≅ F ⋙ shiftFunctor D a` and
`e₂ : shiftFunctor D a ⋙ G ≅ G ⋙ shiftFunctor C a`, this expresses the compatibility of
`e₁` and `e₂` with the counit of the adjunction `adj`.
-/
abbrev CompatibilityCounit :=
  ∀ (Y : D), adj.counit.app (Y⟦a⟧) = F.map (e₂.hom.app Y) ≫ e₁.hom.app _ ≫ (adj.counit.app Y)⟦a⟧'


/-- Given an adjunction `adj : F ⊣ G`, `a` in `A` and commutation isomorphisms
`e₁ : shiftFunctor C a ⋙ F ≅ F ⋙ shiftFunctor D a` and
`e₂ : shiftFunctor D a ⋙ G ≅ G ⋙ shiftFunctor C a`, compatibility of `e₁` and `e₂` with the
unit of the adjunction `adj` implies compatibility with the counit of `adj`.
-/
lemma compatibilityCounit_of_compatibilityUnit (h : CompatibilityUnit adj e₁ e₂) :
    CompatibilityCounit adj e₁ e₂ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    ⊢ CategoryTheory.Adjunction.CommShift.CompatibilityCounit adj e₁ e₂
  -/
  intro Y
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    Y : D
    ⊢ Eq (adj.counit.app ((CategoryTheory.shiftFunctor D a).obj Y)) (CategoryTheor …
  -/
  have eq := h (G.obj Y)
  simp only [← cancel_mono (e₂.inv.app _ ≫ G.map (e₁.inv.app _)),
    assoc, Iso.hom_inv_id_app_assoc, comp_id, ← Functor.map_comp,
    Iso.hom_inv_id_app, Functor.comp_obj, Functor.map_id] at eq
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    Y : D
    eq : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C a) …
    ⊢ Eq (adj.counit.app ((CategoryTheory.shiftFunctor D a).obj Y)) (CategoryTheor …
  -/
  apply (adj.homEquiv _ _).injective
  /-
    case a
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    Y : D
    eq : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C a) …
    ⊢ Eq ((adj.homEquiv (G.obj ((CategoryTheory.shiftFunctor D a).obj Y)) ((Catego …
  -/
  dsimp
  /-
    case a
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    Y : D
    eq : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C a) …
    ⊢ Eq ((adj.homEquiv (G.obj ((CategoryTheory.shiftFunctor D a).obj Y)) ((Catego …
  -/
  rw [adj.homEquiv_unit, adj.homEquiv_unit, G.map_comp, adj.unit_naturality_assoc, ← eq]
  /-
    case a
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    Y : D
    eq : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C a) …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app (G.obj ((CategoryTheory …
  -/
  simp only [assoc, ← Functor.map_comp, Iso.inv_hom_id_app_assoc]
  /-
    case a
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    Y : D
    eq : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C a) …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app (G.obj ((CategoryTheory …
  -/
  erw [← e₂.inv.naturality]
  /-
    case a
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    Y : D
    eq : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C a) …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app (G.obj ((CategoryTheory …
  -/
  dsimp
  simp only [right_triangle_components, ← Functor.map_comp_assoc, Functor.map_id, id_comp,
    Iso.hom_inv_id_app, Functor.comp_obj]


/-- Given an adjunction `adj : F ⊣ G`, `a` in `A` and commutation isomorphisms
`e₁ : shiftFunctor C a ⋙ F ≅ F ⋙ shiftFunctor D a` and
`e₂ : shiftFunctor D a ⋙ G ≅ G ⋙ shiftFunctor C a`, if `e₁` and `e₂` are compatible with the
unit of the adjunction `adj`, then we get a formula for `e₂.inv` in terms of `e₁`.
-/
lemma compatibilityUnit_right (h : CompatibilityUnit adj e₁ e₂) (Y : D) :
    e₂.inv.app Y = adj.unit.app _ ≫ G.map (e₁.hom.app _) ≫ G.map ((adj.counit.app _)⟦a⟧') := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    Y : D
    ⊢ Eq (e₂.inv.app Y) (CategoryTheory.CategoryStruct.comp (adj.unit.app ((Catego …
  -/
  have := h (G.obj Y)
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    Y : D
    this : Eq ((CategoryTheory.shiftFunctor C a).map (adj.unit.app (G.obj Y))) (Ca …
    ⊢ Eq (e₂.inv.app Y) (CategoryTheory.CategoryStruct.comp (adj.unit.app ((Catego …
  -/
  rw [← cancel_mono (e₂.inv.app _), assoc, assoc, Iso.hom_inv_id_app] at this
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    Y : D
    this : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C  …
    ⊢ Eq (e₂.inv.app Y) (CategoryTheory.CategoryStruct.comp (adj.unit.app ((Catego …
  -/
  erw [comp_id] at this
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    Y : D
    this : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C  …
    ⊢ Eq (e₂.inv.app Y) (CategoryTheory.CategoryStruct.comp (adj.unit.app ((Catego …
  -/
  rw [← assoc, ← this, assoc]; erw [← e₂.inv.naturality]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    Y : D
    this : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C  …
    ⊢ Eq (e₂.inv.app Y) (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shift …
  -/
  rw [← cancel_mono (e₂.hom.app _)]
  simp only [Functor.comp_obj, Iso.inv_hom_id_app, Functor.id_obj, Functor.comp_map, assoc, comp_id,
    ← (shiftFunctor C a).map_comp, right_triangle_components, Functor.map_id]


/-- Given an adjunction `adj : F ⊣ G`, `a` in `A` and commutation isomorphisms
`e₁ : shiftFunctor C a ⋙ F ≅ F ⋙ shiftFunctor D a` and
`e₂ : shiftFunctor D a ⋙ G ≅ G ⋙ shiftFunctor C a`, if `e₁` and `e₂` are compatible with the
counit of the adjunction `adj`, then we get a formula for `e₁.hom` in terms of `e₂`.
-/
lemma compatibilityCounit_left (h : CompatibilityCounit adj e₁ e₂) (X : C) :
    e₁.hom.app X = F.map ((adj.unit.app X)⟦a⟧') ≫ F.map (e₂.inv.app _) ≫ adj.counit.app _ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityCounit adj e₁ e₂
    X : C
    ⊢ Eq (e₁.hom.app X) (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheor …
  -/
  have := h (F.obj X)
  rw [← cancel_epi (F.map (e₂.inv.app _)), ← assoc, ← F.map_comp, Iso.inv_hom_id_app, F.map_id,
    id_comp] at this
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityCounit adj e₁ e₂
    X : C
    this : Eq (CategoryTheory.CategoryStruct.comp (F.map (e₂.inv.app (F.obj X))) ( …
    ⊢ Eq (e₁.hom.app X) (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheor …
  -/
  rw [this]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityCounit adj e₁ e₂
    X : C
    this : Eq (CategoryTheory.CategoryStruct.comp (F.map (e₂.inv.app (F.obj X))) ( …
    ⊢ Eq (e₁.hom.app X) (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheor …
  -/
  erw [e₁.hom.naturality_assoc]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityCounit adj e₁ e₂
    X : C
    this : Eq (CategoryTheory.CategoryStruct.comp (F.map (e₂.inv.app (F.obj X))) ( …
    ⊢ Eq (e₁.hom.app X) (CategoryTheory.CategoryStruct.comp (e₁.hom.app ((Category …
  -/
  rw [Functor.comp_map, ← Functor.map_comp, left_triangle_components]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityCounit adj e₁ e₂
    X : C
    this : Eq (CategoryTheory.CategoryStruct.comp (F.map (e₂.inv.app (F.obj X))) ( …
    ⊢ Eq (e₁.hom.app X) (CategoryTheory.CategoryStruct.comp (e₁.hom.app ((Category …
  -/
  simp only [Functor.comp_obj, Functor.id_obj, Functor.map_id, comp_id]
  /-
    🎉 no goals
  -/


/-- Given an adjunction `adj : F ⊣ G`, `a` in `A` and commutation isomorphisms
`e₁ : shiftFunctor C a ⋙ F ≅ F ⋙ shiftFunctor D a` and
`e₂ : shiftFunctor D a ⋙ G ≅ G ⋙ shiftFunctor C a`, if `e₁` and `e₂` are compatible with the
unit of the adjunction `adj`, then `e₁` uniquely determines `e₂`.
-/
lemma compatibilityUnit_unique_right (h : CompatibilityUnit adj e₁ e₂)
    (h' : CompatibilityUnit adj e₁ e₂') : e₂ = e₂' := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    e₂ e₂' : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    h' : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂'
    ⊢ Eq e₂ e₂'
  -/
  rw [← Iso.symm_eq_iff]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    e₂ e₂' : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    h' : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂'
    ⊢ Eq e₂.symm e₂'.symm
  -/
  ext
  rw [Iso.symm_hom, Iso.symm_hom, compatibilityUnit_right adj e₁ e₂ h,
    compatibilityUnit_right adj e₁ e₂' h']


/-- Given an adjunction `adj : F ⊣ G`, `a` in `A` and commutation isomorphisms
`e₁ : shiftFunctor C a ⋙ F ≅ F ⋙ shiftFunctor D a` and
`e₂ : shiftFunctor D a ⋙ G ≅ G ⋙ shiftFunctor C a`, if `e₁` and `e₂` are compatible with the
unit of the adjunction `adj`, then `e₂` uniquely determines `e₁`.
-/
lemma compatibilityUnit_unique_left (h : CompatibilityUnit adj e₁ e₂)
    (h' : CompatibilityUnit adj e₁' e₂) : e₁ = e₁' := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a : A
    e₁ e₁' : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    h' : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁' e₂
    ⊢ Eq e₁ e₁'
  -/
  ext
  rw [compatibilityCounit_left adj e₁ e₂ (compatibilityCounit_of_compatibilityUnit adj _ _ h),
    compatibilityCounit_left adj e₁' e₂ (compatibilityCounit_of_compatibilityUnit adj _ _ h')]


/--
The isomorphisms `Functor.CommShift.isoZero F` and `Functor.CommShift.isoZero G` are
compatible with the unit of an adjunction `F ⊣ G`.
-/
lemma compatibilityUnit_isoZero : CompatibilityUnit adj (Functor.CommShift.isoZero F A)
    (Functor.CommShift.isoZero G A) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    ⊢ CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj (CategoryTheory.Fu …
  -/
  intro
  simp only [Functor.id_obj, Functor.comp_obj, Functor.CommShift.isoZero_hom_app,
    Functor.map_comp, assoc, unit_naturality_assoc,
    ← cancel_mono ((shiftFunctorZero C A).hom.app _), ← G.map_comp_assoc, Iso.inv_hom_id_app,
    Functor.id_obj, Functor.map_id, id_comp, NatTrans.naturality, Functor.id_map, assoc, comp_id]


/-- Given an adjunction `adj : F ⊣ G`, `a, b` in `A` and commutation isomorphisms
between shifts by `a` (resp. `b`) and `F` and `G`, if these commutation isomorphisms are
compatible with the unit of `adj`, then so are the commutation isomorphisms between shifts
by `a + b` and `F` and `G` constructed by `Functor.CommShift.isoAdd`.
-/
lemma compatibilityUnit_isoAdd (h : CompatibilityUnit adj e₁ e₂)
    (h' : CompatibilityUnit adj f₁ f₂) :
    CompatibilityUnit adj (Functor.CommShift.isoAdd e₁ f₁) (Functor.CommShift.isoAdd e₂ f₂) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a b : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    f₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C b).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    f₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D b).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    h' : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj f₁ f₂
    ⊢ CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj (CategoryTheory.Fu …
  -/
  intro X
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a b : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    f₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C b).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    f₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D b).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    h' : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj f₁ f₂
    X : C
    ⊢ Eq ((CategoryTheory.shiftFunctor C (HAdd.hAdd a b)).map (adj.unit.app X)) (C …
  -/
  have := h' (X⟦a⟧)
  simp only [← cancel_mono (f₂.inv.app _), assoc, Iso.hom_inv_id_app,
    Functor.id_obj, Functor.comp_obj, comp_id] at this
  simp only [Functor.id_obj, Functor.comp_obj, Functor.CommShift.isoAdd_hom_app,
    Functor.map_comp, assoc, unit_naturality_assoc]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a b : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    f₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C b).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    f₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D b).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    h' : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj f₁ f₂
    X : C
    this : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C  …
    ⊢ Eq ((CategoryTheory.shiftFunctor C (HAdd.hAdd a b)).map (adj.unit.app X)) (C …
  -/
  slice_rhs 5 6 => rw [← G.map_comp, Iso.inv_hom_id_app]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a b : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    f₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C b).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    f₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D b).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    h' : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj f₁ f₂
    X : C
    this : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C  …
    ⊢ Eq ((CategoryTheory.shiftFunctor C (HAdd.hAdd a b)).map (adj.unit.app X)) (C …
  -/
  simp only [Functor.comp_obj, Functor.map_id, id_comp, assoc]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a b : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    f₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C b).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    f₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D b).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    h' : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj f₁ f₂
    X : C
    this : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C  …
    ⊢ Eq ((CategoryTheory.shiftFunctor C (HAdd.hAdd a b)).map (adj.unit.app X)) (C …
  -/
  erw [f₂.hom.naturality_assoc]
  rw [← reassoc_of% this, ← cancel_mono ((shiftFunctorAdd C a b).hom.app _),
    assoc, assoc, assoc, assoc, assoc, assoc, Iso.inv_hom_id_app_assoc, Iso.inv_hom_id_app]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a b : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    f₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C b).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    f₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D b).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    h' : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj f₁ f₂
    X : C
    this : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C (HAdd …
  -/
  dsimp
  rw [← (shiftFunctor C b).map_comp_assoc, ← (shiftFunctor C b).map_comp_assoc,
    assoc, ← h X, NatTrans.naturality]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a b : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    f₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C b).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    f₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D b).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    h' : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj f₁ f₂
    X : C
    this : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorAdd C a  …
  -/
  dsimp
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.HasShift D A
    a b : A
    e₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C a).comp F) (F.comp (Ca …
    f₁ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor C b).comp F) (F.comp (Ca …
    e₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D a).comp G) (G.comp (Ca …
    f₂ : CategoryTheory.Iso ((CategoryTheory.shiftFunctor D b).comp G) (G.comp (Ca …
    h : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj e₁ e₂
    h' : CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj f₁ f₂
    X : C
    this : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorAdd C a  …
  -/
  rw [comp_id]
  /-
    🎉 no goals
  -/


/--
The property for `CommShift` structures on `F` and `G` to be compatible with an
adjunction `F ⊣ G`.
-/
class CommShift : Prop where
  commShift_unit : NatTrans.CommShift adj.unit A := by infer_instance
  commShift_counit : NatTrans.CommShift adj.counit A := by infer_instance


/-- Constructor for `Adjunction.CommShift`. -/
lemma mk' (h : NatTrans.CommShift adj.unit A) :
    adj.CommShift A where
  commShift_counit := ⟨fun a ↦ by
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁵ : CategoryTheory.Category.{u_5, u_2} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      A : Type u_3
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift C A
      inst✝² : CategoryTheory.HasShift D A
      inst✝¹ : F.CommShift A
      inst✝ : G.CommShift A
      h : CategoryTheory.NatTrans.CommShift adj.unit A
      a : A
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.comp F).commShiftIso a).hom (Cate …
    -/
    ext
    simp only [Functor.comp_obj, Functor.id_obj, NatTrans.comp_app,
      Functor.commShiftIso_comp_hom_app, whiskerRight_app, assoc, whiskerLeft_app,
      Functor.commShiftIso_id_hom_app, comp_id]
    /-
      case w.h
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁵ : CategoryTheory.Category.{u_5, u_2} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      A : Type u_3
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift C A
      inst✝² : CategoryTheory.HasShift D A
      inst✝¹ : F.CommShift A
      inst✝ : G.CommShift A
      h : CategoryTheory.NatTrans.CommShift adj.unit A
      a : A
      x✝ : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((G.commShiftIso a).hom.app x✝ …
    -/
    refine (compatibilityCounit_of_compatibilityUnit adj _ _ (fun X ↦ ?_) _).symm
    simpa only [NatTrans.comp_app,
      Functor.commShiftIso_id_hom_app, whiskerRight_app, id_comp,
      Functor.commShiftIso_comp_hom_app] using congr_app (h.shift_comm a) X⟩


@[reassoc]
lemma shift_unit_app [adj.CommShift A] (a : A) (X : C) :
    (adj.unit.app X)⟦a⟧' =
      adj.unit.app (X⟦a⟧) ≫
        G.map ((F.commShiftIso a).hom.app X) ≫
          (G.commShiftIso a).hom.app (F.obj X) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝⁵ : AddMonoid A
    inst✝⁴ : CategoryTheory.HasShift C A
    inst✝³ : CategoryTheory.HasShift D A
    inst✝² : F.CommShift A
    inst✝¹ : G.CommShift A
    inst✝ : adj.CommShift A
    a : A
    X : C
    ⊢ Eq ((CategoryTheory.shiftFunctor C a).map (adj.unit.app X)) (CategoryTheory. …
  -/
  simpa [Functor.commShiftIso_comp_hom_app] using NatTrans.shift_app_comm adj.unit a X
  /-
    🎉 no goals
  -/


@[reassoc]
lemma shift_counit_app [adj.CommShift A] (a : A) (Y : D) :
    (adj.counit.app Y)⟦a⟧' =
      (F.commShiftIso a).inv.app (G.obj Y) ≫ F.map ((G.commShiftIso a).inv.app Y) ≫
        adj.counit.app (Y⟦a⟧) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝⁵ : AddMonoid A
    inst✝⁴ : CategoryTheory.HasShift C A
    inst✝³ : CategoryTheory.HasShift D A
    inst✝² : F.CommShift A
    inst✝¹ : G.CommShift A
    inst✝ : adj.CommShift A
    a : A
    Y : D
    ⊢ Eq ((CategoryTheory.shiftFunctor D a).map (adj.counit.app Y)) (CategoryTheor …
  -/
  have eq := NatTrans.shift_app_comm adj.counit a Y
  simp only [Functor.comp_obj, Functor.id_obj, Functor.commShiftIso_comp_hom_app, assoc,
    Functor.commShiftIso_id_hom_app, comp_id] at eq
  simp only [← eq, Functor.comp_obj, Functor.id_obj, ← F.map_comp_assoc, Iso.inv_hom_id_app,
    F.map_id, id_comp, Iso.inv_hom_id_app_assoc]


/-- Auxiliary definition for `iso`. -/
noncomputable def iso' : shiftFunctor D a ⋙ G ≅ G ⋙ shiftFunctor C a :=
  (conjugateIsoEquiv (Adjunction.comp adj (shiftEquiv' D b a h).toAdjunction)
    (Adjunction.comp (shiftEquiv' C b a h).toAdjunction adj)).toFun (F.commShiftIso b)


/--
Given an adjunction `F ⊣ G` and a `CommShift` structure on `F`, these are the candidate
`CommShift.iso a` isomorphisms for a compatible `CommShift` structure on `G`.
-/
noncomputable def iso : shiftFunctor D a ⋙ G ≅ G ⋙ shiftFunctor C a :=
  iso' adj _ _ (neg_add_cancel a)


@[reassoc]
lemma iso_hom_app (X : D) :
    (iso adj a).hom.app X =
      (shiftFunctorCompIsoId C b a h).inv.app (G.obj ((shiftFunctor D a).obj X)) ≫
        (adj.unit.app ((shiftFunctor C b).obj (G.obj ((shiftFunctor D a).obj X))))⟦a⟧' ≫
          (G.map ((F.commShiftIso b).hom.app (G.obj ((shiftFunctor D a).obj X))))⟦a⟧' ≫
            (G.map ((shiftFunctor D b).map (adj.counit.app ((shiftFunctor D a).obj X))))⟦a⟧' ≫
              (G.map ((shiftFunctorCompIsoId D a b
                    /-
                      C : Type u_1
                      D : Type u_2
                      inst✝⁵ : CategoryTheory.Category.{?u.137686, u_1} C
                      inst✝⁴ : CategoryTheory.Category.{?u.137690, u_2} D
                      F : CategoryTheory.Functor C D
                      G : CategoryTheory.Functor D C
                      adj : CategoryTheory.Adjunction F G
                      A : Type u_3
                      inst✝³ : AddGroup A
                      inst✝² : CategoryTheory.HasShift C A
                      inst✝¹ : CategoryTheory.HasShift D A
                      a b : A
                      h : Eq (HAdd.hAdd b a) 0
                      inst✝ : F.CommShift A
                      X : D
                      ⊢ Eq (HAdd.hAdd a b) 0
                    -/
                (by rw [← add_left_inj a, add_assoc, h, zero_add, add_zero])).hom.app X))⟦a⟧' := by
                    /-
                      🎉 no goals
                    -/
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    a b : A
    h : Eq (HAdd.hAdd b a) 0
    inst✝ : F.CommShift A
    X : D
    ⊢ Eq ((CategoryTheory.Adjunction.RightAdjointCommShift.iso adj a).hom.app X) ( …
  -/
  obtain rfl : b = -a := by rw [← add_left_inj a, h, neg_add_cancel]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    a : A
    inst✝ : F.CommShift A
    X : D
    h : Eq (HAdd.hAdd (Neg.neg a) a) 0
    ⊢ Eq ((CategoryTheory.Adjunction.RightAdjointCommShift.iso adj a).hom.app X) ( …
  -/
  simp [iso, iso', shiftEquiv']
  /-
    🎉 no goals
  -/


@[reassoc]
lemma iso_inv_app (Y : D) :
    (iso adj a).inv.app Y =
      adj.unit.app ((shiftFunctor C a).obj (G.obj Y)) ≫
          G.map ((shiftFunctorCompIsoId D b a h).inv.app
              (F.obj ((shiftFunctor C a).obj (G.obj Y)))) ≫
            G.map ((shiftFunctor D a).map ((shiftFunctor D b).map
                ((F.commShiftIso a).hom.app (G.obj Y)))) ≫
              G.map ((shiftFunctor D a).map ((shiftFunctorCompIsoId D a b
                      /-
                        C : Type u_1
                        D : Type u_2
                        inst✝⁵ : CategoryTheory.Category.{?u.150622, u_1} C
                        inst✝⁴ : CategoryTheory.Category.{?u.150626, u_2} D
                        F : CategoryTheory.Functor C D
                        G : CategoryTheory.Functor D C
                        adj : CategoryTheory.Adjunction F G
                        A : Type u_3
                        inst✝³ : AddGroup A
                        inst✝² : CategoryTheory.HasShift C A
                        inst✝¹ : CategoryTheory.HasShift D A
                        a b : A
                        h : Eq (HAdd.hAdd b a) 0
                        inst✝ : F.CommShift A
                        Y : D
                        ⊢ Eq (HAdd.hAdd a b) 0
                      -/
                  (by rw [eq_neg_of_add_eq_zero_left h, add_neg_cancel])).hom.app
                      /-
                        🎉 no goals
                      -/
                    (F.obj (G.obj Y)))) ≫
                G.map ((shiftFunctor D a).map (adj.counit.app Y)) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    a b : A
    h : Eq (HAdd.hAdd b a) 0
    inst✝ : F.CommShift A
    Y : D
    ⊢ Eq ((CategoryTheory.Adjunction.RightAdjointCommShift.iso adj a).inv.app Y) ( …
  -/
  obtain rfl : b = -a := by rw [← add_left_inj a, h, neg_add_cancel]
  simp only [Functor.comp_obj, iso, iso', shiftEquiv', Equiv.toFun_as_coe,
    conjugateIsoEquiv_apply_inv, conjugateEquiv_apply_app, comp_unit_app, Functor.id_obj,
    Equivalence.toAdjunction_unit, Equivalence.Equivalence_mk'_unit, Iso.symm_hom, Functor.comp_map,
    comp_counit_app, Equivalence.toAdjunction_counit, Equivalence.Equivalence_mk'_counit,
    Functor.map_shiftFunctorCompIsoId_hom_app, assoc, Functor.map_comp]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    a : A
    inst✝ : F.CommShift A
    Y : D
    h : Eq (HAdd.hAdd (Neg.neg a) a) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app ((CategoryTheory.shiftF …
  -/
  slice_lhs 3 4 => rw [← Functor.map_comp, ← Functor.map_comp, Iso.inv_hom_id_app]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    a : A
    inst✝ : F.CommShift A
    Y : D
    h : Eq (HAdd.hAdd (Neg.neg a) a) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app ((CategoryTheory.shiftF …
  -/
  simp only [Functor.comp_obj, Functor.map_id, id_comp, assoc]
  /-
    🎉 no goals
  -/


/--
The commutation isomorphisms of `Adjunction.RightAdjointCommShift.iso` are compatible with
the unit of the adjunction.
-/
lemma compatibilityUnit_iso (a : A) :
    CommShift.CompatibilityUnit adj (F.commShiftIso a) (iso adj a) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : F.CommShift A
    a : A
    ⊢ CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj (F.commShiftIso a) …
  -/
  intro
  rw [← cancel_mono ((RightAdjointCommShift.iso adj a).inv.app _), assoc, assoc,
    Iso.hom_inv_id_app, RightAdjointCommShift.iso_inv_app adj _ _ (neg_add_cancel a)]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : F.CommShift A
    a : A
    X✝ : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C a).ma …
  -/
  apply (adj.homEquiv _ _).symm.injective
  /-
    case a
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : F.CommShift A
    a : A
    X✝ : C
    ⊢ Eq ((adj.homEquiv ((CategoryTheory.shiftFunctor C a).obj ((CategoryTheory.Fu …
  -/
  dsimp
  simp only [comp_id, homEquiv_counit, Functor.map_comp, assoc, counit_naturality,
    counit_naturality_assoc, left_triangle_components_assoc]
  /-
    case a
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : F.CommShift A
    a : A
    X✝ : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.shiftFunctor  …
  -/
  erw [← NatTrans.naturality_assoc]
  /-
    case a
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : F.CommShift A
    a : A
    X✝ : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.shiftFunctor  …
  -/
  dsimp
  rw [shift_shiftFunctorCompIsoId_hom_app, Iso.inv_hom_id_app_assoc,
    Functor.commShiftIso_hom_naturality_assoc, ← Functor.map_comp,
    left_triangle_components, Functor.map_id, comp_id]


open RightAdjointCommShift in
/--
Given an adjunction `F ⊣ G` and a `CommShift` structure on `F`, this constructs
the unique compatible `CommShift` structure on `G`.
-/
@[simps]
noncomputable def rightAdjointCommShift [F.CommShift A] : G.CommShift A where
  iso a := iso adj a
  zero := by
    refine CommShift.compatibilityUnit_unique_right adj (F.commShiftIso 0) _ _
      (compatibilityUnit_iso adj 0) ?_
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.177039, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.177043, u_2} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      A : Type u_3
      inst✝³ : AddGroup A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.CommShift A
      ⊢ CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj (F.commShiftIso 0) …
    -/
    rw [F.commShiftIso_zero]
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.177039, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.177043, u_2} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      A : Type u_3
      inst✝³ : AddGroup A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.CommShift A
      ⊢ CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj (CategoryTheory.Fu …
    -/
    exact CommShift.compatibilityUnit_isoZero adj
    /-
      🎉 no goals
    -/
  add a b := by
    refine CommShift.compatibilityUnit_unique_right adj (F.commShiftIso (a + b)) _ _
      (compatibilityUnit_iso adj (a + b)) ?_
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.177039, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.177043, u_2} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      A : Type u_3
      inst✝³ : AddGroup A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.CommShift A
      a b : A
      ⊢ CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj (F.commShiftIso (H …
    -/
    rw [F.commShiftIso_add]
    exact CommShift.compatibilityUnit_isoAdd adj _ _ _ _
      (compatibilityUnit_iso adj a) (compatibilityUnit_iso adj b)


lemma commShift_of_leftAdjoint [F.CommShift A] :
    letI := adj.rightAdjointCommShift A
    adj.CommShift A := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : F.CommShift A
    ⊢ adj.CommShift A
  -/
  letI := adj.rightAdjointCommShift A
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : F.CommShift A
    this : G.CommShift A := adj.rightAdjointCommShift A
    ⊢ adj.CommShift A
  -/
  refine CommShift.mk' _ _ ⟨fun a ↦ ?_⟩
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : F.CommShift A
    this : G.CommShift A := adj.rightAdjointCommShift A
    a : A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id C).commSh …
  -/
  ext X
  /-
    case w.h
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : F.CommShift A
    this : G.CommShift A := adj.rightAdjointCommShift A
    a : A
    X : C
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id C).commS …
  -/
  dsimp
  simpa only [Functor.commShiftIso_id_hom_app, Functor.comp_obj, Functor.id_obj, id_comp,
    Functor.commShiftIso_comp_hom_app] using RightAdjointCommShift.compatibilityUnit_iso adj a X


/-- Auxiliary definition for `iso`. -/
noncomputable def iso' : shiftFunctor C a ⋙ F ≅ F ⋙ shiftFunctor D a :=
  (conjugateIsoEquiv (Adjunction.comp adj (shiftEquiv' D a b h).toAdjunction)
    (Adjunction.comp (shiftEquiv' C a b h).toAdjunction adj)).invFun (G.commShiftIso b)


/--
Given an adjunction `F ⊣ G` and a `CommShift` structure on `G`, these are the candidate
`CommShift.iso a` isomorphisms for a compatible `CommShift` structure on `F`.
-/
noncomputable def iso : shiftFunctor C a ⋙ F ≅ F ⋙ shiftFunctor D a :=
  iso' adj _ _ (add_neg_cancel a)


@[reassoc]
lemma iso_hom_app (X : C) :
    (iso adj a).hom.app X = F.map ((adj.unit.app X)⟦a⟧') ≫
      F.map (G.map (((shiftFunctorCompIsoId D a b h).inv.app (F.obj X)))⟦a⟧') ≫
        F.map (((G.commShiftIso b).hom.app ((F.obj X)⟦a⟧))⟦a⟧') ≫
                                                  /-
                                                    C : Type u_1
                                                    D : Type u_2
                                                    inst✝⁵ : CategoryTheory.Category.{?u.188073, u_1} C
                                                    inst✝⁴ : CategoryTheory.Category.{?u.188077, u_2} D
                                                    F : CategoryTheory.Functor C D
                                                    G : CategoryTheory.Functor D C
                                                    adj : CategoryTheory.Adjunction F G
                                                    A : Type u_3
                                                    inst✝³ : AddGroup A
                                                    inst✝² : CategoryTheory.HasShift C A
                                                    inst✝¹ : CategoryTheory.HasShift D A
                                                    a b : A
                                                    h : Eq (HAdd.hAdd a b) 0
                                                    inst✝ : G.CommShift A
                                                    X : C
                                                    ⊢ Eq (HAdd.hAdd b a) 0
                                                  -/
          F.map ((shiftFunctorCompIsoId C b a (by simp [eq_neg_of_add_eq_zero_left h])).hom.app
                                                  /-
                                                    🎉 no goals
                                                  -/
            (G.obj ((F.obj X)⟦a⟧))) ≫ adj.counit.app ((F.obj X)⟦a⟧) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    a b : A
    h : Eq (HAdd.hAdd a b) 0
    inst✝ : G.CommShift A
    X : C
    ⊢ Eq ((CategoryTheory.Adjunction.LeftAdjointCommShift.iso adj a).hom.app X) (C …
  -/
  obtain rfl : b = -a := eq_neg_of_add_eq_zero_right h
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    a : A
    inst✝ : G.CommShift A
    X : C
    h : Eq (HAdd.hAdd a (Neg.neg a)) 0
    ⊢ Eq ((CategoryTheory.Adjunction.LeftAdjointCommShift.iso adj a).hom.app X) (C …
  -/
  simp [iso, iso', shiftEquiv']
  /-
    🎉 no goals
  -/


@[reassoc]
lemma iso_inv_app (Y : C) :
    (iso adj a).inv.app Y = (F.map ((shiftFunctorCompIsoId C a b h).inv.app Y))⟦a⟧' ≫
      (F.map ((adj.unit.app (Y⟦a⟧))⟦b⟧'))⟦a⟧' ≫ (F.map ((G.commShiftIso b).inv.app
        (F.obj (Y⟦a⟧))))⟦a⟧' ≫ (adj.counit.app ((F.obj (Y⟦a⟧))⟦b⟧))⟦a⟧' ≫
                                           /-
                                             C : Type u_1
                                             D : Type u_2
                                             inst✝⁵ : CategoryTheory.Category.{?u.206171, u_1} C
                                             inst✝⁴ : CategoryTheory.Category.{?u.206175, u_2} D
                                             F : CategoryTheory.Functor C D
                                             G : CategoryTheory.Functor D C
                                             adj : CategoryTheory.Adjunction F G
                                             A : Type u_3
                                             inst✝³ : AddGroup A
                                             inst✝² : CategoryTheory.HasShift C A
                                             inst✝¹ : CategoryTheory.HasShift D A
                                             a b : A
                                             h : Eq (HAdd.hAdd a b) 0
                                             inst✝ : G.CommShift A
                                             Y : C
                                             ⊢ Eq (HAdd.hAdd b a) 0
                                           -/
          (shiftFunctorCompIsoId D b a (by simp [eq_neg_of_add_eq_zero_left h])).hom.app
                                           /-
                                             🎉 no goals
                                           -/
            (F.obj (Y⟦a⟧)) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    a b : A
    h : Eq (HAdd.hAdd a b) 0
    inst✝ : G.CommShift A
    Y : C
    ⊢ Eq ((CategoryTheory.Adjunction.LeftAdjointCommShift.iso adj a).inv.app Y) (C …
  -/
  obtain rfl : b = -a := eq_neg_of_add_eq_zero_right h
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    a : A
    inst✝ : G.CommShift A
    Y : C
    h : Eq (HAdd.hAdd a (Neg.neg a)) 0
    ⊢ Eq ((CategoryTheory.Adjunction.LeftAdjointCommShift.iso adj a).inv.app Y) (C …
  -/
  simp [iso, iso', shiftEquiv']
  /-
    🎉 no goals
  -/


/--
The commutation isomorphisms of `Adjunction.LeftAdjointCommShift.iso` are compatible with
the unit of the adjunction.
-/
lemma compatibilityUnit_iso (a : A) :
    CommShift.CompatibilityUnit adj (iso adj a) (G.commShiftIso a) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : G.CommShift A
    a : A
    ⊢ CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj (CategoryTheory.Ad …
  -/
  intro
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : G.CommShift A
    a : A
    X✝ : C
    ⊢ Eq ((CategoryTheory.shiftFunctor C a).map (adj.unit.app X✝)) (CategoryTheory …
  -/
  rw [LeftAdjointCommShift.iso_hom_app adj _ _ (add_neg_cancel a)]
  simp only [Functor.id_obj, Functor.comp_obj, Functor.map_shiftFunctorCompIsoId_inv_app,
    Functor.map_comp, assoc, unit_naturality_assoc, right_triangle_components_assoc]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : G.CommShift A
    a : A
    X✝ : C
    ⊢ Eq ((CategoryTheory.shiftFunctor C a).map (adj.unit.app X✝)) (CategoryTheory …
  -/
  slice_rhs 4 5 => rw [← Functor.map_comp, Iso.inv_hom_id_app]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : G.CommShift A
    a : A
    X✝ : C
    ⊢ Eq ((CategoryTheory.shiftFunctor C a).map (adj.unit.app X✝)) (CategoryTheory …
  -/
  simp only [Functor.comp_obj, Functor.map_id, id_comp]
  rw [shift_shiftFunctorCompIsoId_inv_app, ← Functor.comp_map,
    (shiftFunctorCompIsoId C _ _ (neg_add_cancel a)).hom.naturality_assoc]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : G.CommShift A
    a : A
    X✝ : C
    ⊢ Eq ((CategoryTheory.shiftFunctor C a).map (adj.unit.app X✝)) (CategoryTheory …
  -/
  simp
  /-
    🎉 no goals
  -/


open LeftAdjointCommShift in
/--
Given an adjunction `F ⊣ G` and a `CommShift` structure on `G`, this constructs
the unique compatible `CommShift` structure on `F`.
-/
@[simps]
noncomputable def leftAdjointCommShift [G.CommShift A] : F.CommShift A where
  iso a := iso adj a
  zero := by
    refine CommShift.compatibilityUnit_unique_left adj _ _ (G.commShiftIso 0)
      (compatibilityUnit_iso adj 0) ?_
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.233375, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.233379, u_2} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      A : Type u_3
      inst✝³ : AddGroup A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : G.CommShift A
      ⊢ CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj (CategoryTheory.Fu …
    -/
    rw [G.commShiftIso_zero]
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.233375, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.233379, u_2} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      A : Type u_3
      inst✝³ : AddGroup A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : G.CommShift A
      ⊢ CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj (CategoryTheory.Fu …
    -/
    exact CommShift.compatibilityUnit_isoZero adj
    /-
      🎉 no goals
    -/
  add a b := by
    refine CommShift.compatibilityUnit_unique_left adj _ _ (G.commShiftIso (a + b))
      (compatibilityUnit_iso adj (a + b)) ?_
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.233375, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.233379, u_2} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      A : Type u_3
      inst✝³ : AddGroup A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : G.CommShift A
      a b : A
      ⊢ CategoryTheory.Adjunction.CommShift.CompatibilityUnit adj (CategoryTheory.Fu …
    -/
    rw [G.commShiftIso_add]
    exact CommShift.compatibilityUnit_isoAdd adj _ _ _ _
      (compatibilityUnit_iso adj a) (compatibilityUnit_iso adj b)


lemma commShift_of_rightAdjoint [G.CommShift A] :
    letI := adj.leftAdjointCommShift A
    adj.CommShift A := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : G.CommShift A
    ⊢ adj.CommShift A
  -/
  letI := adj.leftAdjointCommShift A
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : G.CommShift A
    this : F.CommShift A := adj.leftAdjointCommShift A
    ⊢ adj.CommShift A
  -/
  refine CommShift.mk' _ _ ⟨fun a ↦ ?_⟩
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : G.CommShift A
    this : F.CommShift A := adj.leftAdjointCommShift A
    a : A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id C).commSh …
  -/
  ext X
  /-
    case w.h
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : G.CommShift A
    this : F.CommShift A := adj.leftAdjointCommShift A
    a : A
    X : C
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id C).commS …
  -/
  dsimp
  simpa only [Functor.commShiftIso_id_hom_app, Functor.comp_obj, Functor.id_obj, id_comp,
    Functor.commShiftIso_comp_hom_app] using LeftAdjointCommShift.compatibilityUnit_iso adj a X


/--
If `E : C ≌ D` is an equivalence, this expresses the compatibility of `CommShift`
structures on `E.functor` and `E.inverse`.
-/
abbrev CommShift [E.functor.CommShift A] [E.inverse.CommShift A] : Prop :=
  E.toAdjunction.CommShift A


instance [E.CommShift A] : NatTrans.CommShift E.unitIso.hom A :=
  inferInstanceAs (NatTrans.CommShift E.toAdjunction.unit A)


instance [E.CommShift A] : NatTrans.CommShift E.counitIso.hom A :=
  inferInstanceAs (NatTrans.CommShift E.toAdjunction.counit A)


instance [h : E.functor.CommShift A] : E.symm.inverse.CommShift A := h

instance [h : E.inverse.CommShift A] : E.symm.functor.CommShift A := h


/-- Constructor for `Equivalence.CommShift`. -/
lemma mk' (h : NatTrans.CommShift E.unitIso.hom A) :
    E.CommShift A where
  commShift_unit := h
  commShift_counit := (Adjunction.CommShift.mk' E.toAdjunction A h).commShift_counit


/--
If `E : C ≌ D` is an equivalence and we have compatible `CommShift` structures on `E.functor`
and `E.inverse`, then we also have compatible `CommShift` structures on `E.symm.functor`
and `E.symm.inverse`.
-/
instance [E.CommShift A] : E.symm.CommShift A :=
  mk' E.symm A (inferInstanceAs (NatTrans.CommShift E.counitIso.inv A))


/-- Constructor for `Equivalence.CommShift`. -/
lemma mk'' (h : NatTrans.CommShift E.counitIso.hom A) :
    E.CommShift A :=
  have := mk' E.symm A (inferInstanceAs (NatTrans.CommShift E.counitIso.inv A))
  inferInstanceAs (E.symm.symm.CommShift A)


/--
If `E : C ≌ D` is an equivalence and we have a `CommShift` structure on `E.functor`,
this constructs the unique compatible `CommShift` structure on `E.inverse`.
-/
noncomputable def commShiftInverse [E.functor.CommShift A] : E.inverse.CommShift A :=
  E.toAdjunction.rightAdjointCommShift A


lemma commShift_of_functor [E.functor.CommShift A] :
    letI := E.commShiftInverse A
    E.CommShift A := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    E : CategoryTheory.Equivalence C D
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : E.functor.CommShift A
    ⊢ E.CommShift A
  -/
  letI := E.commShiftInverse A
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_5, u_2} D
    E : CategoryTheory.Equivalence C D
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : E.functor.CommShift A
    this : E.inverse.CommShift A := E.commShiftInverse A
    ⊢ E.CommShift A
  -/
  exact CommShift.mk' _ _ (E.toAdjunction.commShift_of_leftAdjoint A).commShift_unit
  /-
    🎉 no goals
  -/


/--
If `E : C ≌ D` is an equivalence and we have a `CommShift` structure on `E.inverse`,
this constructs the unique compatible `CommShift` structure on `E.functor`.
-/
noncomputable def commShiftFunctor [E.inverse.CommShift A] : E.functor.CommShift A :=
  E.symm.toAdjunction.rightAdjointCommShift A


lemma commShift_of_inverse [E.inverse.CommShift A] :
    letI := E.commShiftFunctor A
    E.CommShift A := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
    E : CategoryTheory.Equivalence C D
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : E.inverse.CommShift A
    ⊢ E.CommShift A
  -/
  letI := E.commShiftFunctor A
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
    E : CategoryTheory.Equivalence C D
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : E.inverse.CommShift A
    this : E.functor.CommShift A := E.commShiftFunctor A
    ⊢ E.CommShift A
  -/
  have := E.symm.commShift_of_functor A
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Category.{u_4, u_2} D
    E : CategoryTheory.Equivalence C D
    A : Type u_3
    inst✝³ : AddGroup A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.HasShift D A
    inst✝ : E.inverse.CommShift A
    this✝ : E.functor.CommShift A := E.commShiftFunctor A
    this : E.symm.CommShift A
    ⊢ E.CommShift A
  -/
  exact inferInstanceAs (E.symm.symm.CommShift A)
  /-
    🎉 no goals
  -/



/-- Suppose we have a square of functors (where the top and bottom are adjunctions `L₁ ⊣ R₁`
and `L₂ ⊣ R₂` respectively).

      C ↔ D
    G ↓   ↓ H
      E ↔ F

Then we have a bijection between natural transformations `G ⋙ L₂ ⟶ L₁ ⋙ H` and
`R₁ ⋙ G ⟶ H ⋙ R₂`. This can be seen as a bijection of the 2-cells:

         L₁                  R₁
      C --→ D             C ←-- D
    G ↓  ↗  ↓ H         G ↓  ↘  ↓ H
      E --→ F             E ←-- F
         L₂                  R₂

Note that if one of the transformations is an iso, it does not imply the other is an iso.
-/
@[simps]
def mateEquiv : (G ⋙ L₂ ⟶ L₁ ⋙ H) ≃ (R₁ ⋙ G ⟶ H ⋙ R₂) where
  toFun α :=
    whiskerLeft (R₁ ⋙ G) adj₂.unit ≫
    whiskerRight (whiskerLeft R₁ α) R₂ ≫
    whiskerRight adj₁.counit (H ⋙ R₂)
  invFun β :=
    whiskerRight adj₁.unit (G ⋙ L₂) ≫
    whiskerRight (whiskerLeft L₁ β) L₂ ≫
    whiskerLeft (L₁ ⋙ H) adj₂.counit
  left_inv α := by
    /-
      C : Type u₁
      D : Type u₂
      E : Type u₃
      F : Type u₄
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
      inst✝ : CategoryTheory.Category.{v₄, u₄} F
      G : CategoryTheory.Functor C E
      H : CategoryTheory.Functor D F
      L₁ : CategoryTheory.Functor C D
      R₁ : CategoryTheory.Functor D C
      L₂ : CategoryTheory.Functor E F
      R₂ : CategoryTheory.Functor F E
      adj₁ : CategoryTheory.Adjunction L₁ R₁
      adj₂ : CategoryTheory.Adjunction L₂ R₂
      α : Quiver.Hom (G.comp L₂) (L₁.comp H)
      ⊢ Eq ((fun β => CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRigh …
    -/
    ext
    /-
      case w.h
      C : Type u₁
      D : Type u₂
      E : Type u₃
      F : Type u₄
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
      inst✝ : CategoryTheory.Category.{v₄, u₄} F
      G : CategoryTheory.Functor C E
      H : CategoryTheory.Functor D F
      L₁ : CategoryTheory.Functor C D
      R₁ : CategoryTheory.Functor D C
      L₂ : CategoryTheory.Functor E F
      R₂ : CategoryTheory.Functor F E
      adj₁ : CategoryTheory.Adjunction L₁ R₁
      adj₂ : CategoryTheory.Adjunction L₂ R₂
      α : Quiver.Hom (G.comp L₂) (L₁.comp H)
      x✝ : C
      ⊢ Eq (((fun β => CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRig …
    -/
    unfold whiskerRight whiskerLeft
    simp only [comp_obj, id_obj, Functor.comp_map, comp_app, map_comp, assoc, counit_naturality,
      counit_naturality_assoc, left_triangle_components_assoc]
    rw [← assoc, ← Functor.comp_map, α.naturality, Functor.comp_map, assoc, ← H.map_comp,
      left_triangle_components, map_id]
    /-
      case w.h
      C : Type u₁
      D : Type u₂
      E : Type u₃
      F : Type u₄
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
      inst✝ : CategoryTheory.Category.{v₄, u₄} F
      G : CategoryTheory.Functor C E
      H : CategoryTheory.Functor D F
      L₁ : CategoryTheory.Functor C D
      R₁ : CategoryTheory.Functor D C
      L₂ : CategoryTheory.Functor E F
      R₂ : CategoryTheory.Functor F E
      adj₁ : CategoryTheory.Adjunction L₁ R₁
      adj₂ : CategoryTheory.Adjunction L₂ R₂
      α : Quiver.Hom (G.comp L₂) (L₁.comp H)
      x✝ : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.app x✝) (CategoryTheory.CategorySt …
    -/
    simp only [comp_obj, comp_id]
    /-
      🎉 no goals
    -/
  right_inv β := by
    /-
      C : Type u₁
      D : Type u₂
      E : Type u₃
      F : Type u₄
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
      inst✝ : CategoryTheory.Category.{v₄, u₄} F
      G : CategoryTheory.Functor C E
      H : CategoryTheory.Functor D F
      L₁ : CategoryTheory.Functor C D
      R₁ : CategoryTheory.Functor D C
      L₂ : CategoryTheory.Functor E F
      R₂ : CategoryTheory.Functor F E
      adj₁ : CategoryTheory.Adjunction L₁ R₁
      adj₂ : CategoryTheory.Adjunction L₂ R₂
      β : Quiver.Hom (R₁.comp G) (H.comp R₂)
      ⊢ Eq ((fun α => CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft …
    -/
    ext
    /-
      case w.h
      C : Type u₁
      D : Type u₂
      E : Type u₃
      F : Type u₄
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
      inst✝ : CategoryTheory.Category.{v₄, u₄} F
      G : CategoryTheory.Functor C E
      H : CategoryTheory.Functor D F
      L₁ : CategoryTheory.Functor C D
      R₁ : CategoryTheory.Functor D C
      L₂ : CategoryTheory.Functor E F
      R₂ : CategoryTheory.Functor F E
      adj₁ : CategoryTheory.Adjunction L₁ R₁
      adj₂ : CategoryTheory.Adjunction L₂ R₂
      β : Quiver.Hom (R₁.comp G) (H.comp R₂)
      x✝ : D
      ⊢ Eq (((fun α => CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLef …
    -/
    unfold whiskerLeft whiskerRight
    simp only [comp_obj, id_obj, Functor.comp_map, comp_app, map_comp, assoc,
      unit_naturality_assoc, right_triangle_components_assoc]
    rw [← assoc, ← Functor.comp_map, assoc, ← β.naturality, ← assoc, Functor.comp_map,
      ← G.map_comp, right_triangle_components, map_id, id_comp]


@[deprecated (since := "2024-07-09")] alias transferNatTrans := mateEquiv


/-- A component of a transposed version of the mates correspondence. -/
theorem mateEquiv_counit (α : G ⋙ L₂ ⟶ L₁ ⋙ H) (d : D) :
    L₂.map ((mateEquiv adj₁ adj₂ α).app _) ≫ adj₂.counit.app _ =
      α.app _ ≫ H.map (adj₁.counit.app d) := by
  /-
    C : Type u₁
    D : Type u₂
    E : Type u₃
    F : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    inst✝ : CategoryTheory.Category.{v₄, u₄} F
    G : CategoryTheory.Functor C E
    H : CategoryTheory.Functor D F
    L₁ : CategoryTheory.Functor C D
    R₁ : CategoryTheory.Functor D C
    L₂ : CategoryTheory.Functor E F
    R₂ : CategoryTheory.Functor F E
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom (G.comp L₂) (L₁.comp H)
    d : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (L₂.map (((CategoryTheory.mateEquiv a …
  -/
  erw [Functor.map_comp]; simp
                          /-
                            🎉 no goals
                          -/


/-- A component of a transposed version of the inverse mates correspondence. -/
theorem mateEquiv_counit_symm (α : R₁ ⋙ G ⟶ H ⋙ R₂) (d : D) :
    L₂.map (α.app _) ≫ adj₂.counit.app _ =
      ((mateEquiv adj₁ adj₂).symm α).app _ ≫ H.map (adj₁.counit.app d) := by
  /-
    C : Type u₁
    D : Type u₂
    E : Type u₃
    F : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    inst✝ : CategoryTheory.Category.{v₄, u₄} F
    G : CategoryTheory.Functor C E
    H : CategoryTheory.Functor D F
    L₁ : CategoryTheory.Functor C D
    R₁ : CategoryTheory.Functor D C
    L₂ : CategoryTheory.Functor E F
    R₂ : CategoryTheory.Functor F E
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom (R₁.comp G) (H.comp R₂)
    d : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (L₂.map (α.app d)) (adj₂.counit.app ( …
  -/
  conv_lhs => rw [← (mateEquiv adj₁ adj₂).right_inv α]
  /-
    C : Type u₁
    D : Type u₂
    E : Type u₃
    F : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    inst✝ : CategoryTheory.Category.{v₄, u₄} F
    G : CategoryTheory.Functor C E
    H : CategoryTheory.Functor D F
    L₁ : CategoryTheory.Functor C D
    R₁ : CategoryTheory.Functor D C
    L₂ : CategoryTheory.Functor E F
    R₂ : CategoryTheory.Functor F E
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom (R₁.comp G) (H.comp R₂)
    d : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (L₂.map (((CategoryTheory.mateEquiv a …
  -/
  exact (mateEquiv_counit adj₁ adj₂ ((mateEquiv adj₁ adj₂).symm α) d)
  /-
    🎉 no goals
  -/

/- A component of a transposed version of the mates correspondence. -/

theorem unit_mateEquiv (α : G ⋙ L₂ ⟶ L₁ ⋙ H) (c : C) :
    G.map (adj₁.unit.app c) ≫ (mateEquiv adj₁ adj₂ α).app _ =
      adj₂.unit.app _ ≫ R₂.map (α.app _) := by
  /-
    C : Type u₁
    D : Type u₂
    E : Type u₃
    F : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    inst✝ : CategoryTheory.Category.{v₄, u₄} F
    G : CategoryTheory.Functor C E
    H : CategoryTheory.Functor D F
    L₁ : CategoryTheory.Functor C D
    R₁ : CategoryTheory.Functor D C
    L₂ : CategoryTheory.Functor E F
    R₂ : CategoryTheory.Functor F E
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom (G.comp L₂) (L₁.comp H)
    c : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (adj₁.unit.app c)) (((Category …
  -/
  dsimp [mateEquiv]
  /-
    C : Type u₁
    D : Type u₂
    E : Type u₃
    F : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    inst✝ : CategoryTheory.Category.{v₄, u₄} F
    G : CategoryTheory.Functor C E
    H : CategoryTheory.Functor D F
    L₁ : CategoryTheory.Functor C D
    R₁ : CategoryTheory.Functor D C
    L₂ : CategoryTheory.Functor E F
    R₂ : CategoryTheory.Functor F E
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom (G.comp L₂) (L₁.comp H)
    c : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (adj₁.unit.app c)) (CategoryTh …
  -/
  rw [← adj₂.unit_naturality_assoc]
  slice_lhs 2 3 =>
    rw [← R₂.map_comp, ← Functor.comp_map G L₂, α.naturality]
  /-
    C : Type u₁
    D : Type u₂
    E : Type u₃
    F : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    inst✝ : CategoryTheory.Category.{v₄, u₄} F
    G : CategoryTheory.Functor C E
    H : CategoryTheory.Functor D F
    L₁ : CategoryTheory.Functor C D
    R₁ : CategoryTheory.Functor D C
    L₂ : CategoryTheory.Functor E F
    R₂ : CategoryTheory.Functor F E
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom (G.comp L₂) (L₁.comp H)
    c : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₂.unit.app (G.obj c)) (CategoryTh …
  -/
  rw [R₂.map_comp]
  slice_lhs 3 4 =>
    rw [← R₂.map_comp, Functor.comp_map L₁ H, ← H.map_comp, left_triangle_components]
  /-
    C : Type u₁
    D : Type u₂
    E : Type u₃
    F : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    inst✝ : CategoryTheory.Category.{v₄, u₄} F
    G : CategoryTheory.Functor C E
    H : CategoryTheory.Functor D F
    L₁ : CategoryTheory.Functor C D
    R₁ : CategoryTheory.Functor D C
    L₂ : CategoryTheory.Functor E F
    R₂ : CategoryTheory.Functor F E
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom (G.comp L₂) (L₁.comp H)
    c : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₂.unit.app (G.obj c)) (CategoryTh …
  -/
  simp only [comp_obj, map_id, comp_id]
  /-
    🎉 no goals
  -/


/-- A component of a transposed version of the inverse mates correspondence. -/
theorem unit_mateEquiv_symm (α : R₁ ⋙ G ⟶ H ⋙ R₂) (c : C) :
    G.map (adj₁.unit.app c) ≫ α.app _ =
      adj₂.unit.app _ ≫ R₂.map (((mateEquiv adj₁ adj₂).symm α).app _) := by
  /-
    C : Type u₁
    D : Type u₂
    E : Type u₃
    F : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    inst✝ : CategoryTheory.Category.{v₄, u₄} F
    G : CategoryTheory.Functor C E
    H : CategoryTheory.Functor D F
    L₁ : CategoryTheory.Functor C D
    R₁ : CategoryTheory.Functor D C
    L₂ : CategoryTheory.Functor E F
    R₂ : CategoryTheory.Functor F E
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom (R₁.comp G) (H.comp R₂)
    c : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (adj₁.unit.app c)) (α.app (L₁. …
  -/
  conv_lhs => rw [← (mateEquiv adj₁ adj₂).right_inv α]
  /-
    C : Type u₁
    D : Type u₂
    E : Type u₃
    F : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    inst✝ : CategoryTheory.Category.{v₄, u₄} F
    G : CategoryTheory.Functor C E
    H : CategoryTheory.Functor D F
    L₁ : CategoryTheory.Functor C D
    R₁ : CategoryTheory.Functor D C
    L₂ : CategoryTheory.Functor E F
    R₂ : CategoryTheory.Functor F E
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom (R₁.comp G) (H.comp R₂)
    c : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (adj₁.unit.app c)) (((Category …
  -/
  exact (unit_mateEquiv adj₁ adj₂ ((mateEquiv adj₁ adj₂).symm α) c)
  /-
    🎉 no goals
  -/


/-- Squares between left adjoints can be composed "vertically" by pasting. -/
def leftAdjointSquare.vcomp :
    (G₁ ⋙ L₂ ⟶ L₁ ⋙ H₁) → (G₂ ⋙ L₃ ⟶ L₂ ⋙ H₂) → ((G₁ ⋙ G₂) ⋙ L₃ ⟶ L₁ ⋙ (H₁ ⋙ H₂)) :=
  fun α β ↦ (whiskerLeft G₁ β) ≫ (whiskerRight α H₂)


/-- Squares between right adjoints can be composed "vertically" by pasting. -/
def rightAdjointSquare.vcomp :
    (R₁ ⋙ G₁ ⟶ H₁ ⋙ R₂) → (R₂ ⋙ G₂ ⟶ H₂ ⋙ R₃) → (R₁ ⋙ (G₁ ⋙ G₂) ⟶ (H₁ ⋙ H₂) ⋙ R₃) :=
  fun α β ↦ (whiskerRight α G₂) ≫ (whiskerLeft H₁ β)


/-- The mates equivalence commutes with vertical composition. -/
theorem mateEquiv_vcomp
    (α : G₁ ⋙ L₂ ⟶ L₁ ⋙ H₁) (β : G₂ ⋙ L₃ ⟶ L₂ ⋙ H₂) :
    (mateEquiv (G := G₁ ⋙ G₂) (H := H₁ ⋙ H₂) adj₁ adj₃) (leftAdjointSquare.vcomp α β) =
      rightAdjointSquare.vcomp (mateEquiv adj₁ adj₂ α) (mateEquiv adj₂ adj₃ β) := by
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
    inst✝³ : CategoryTheory.Category.{v₃, u₃} C
    inst✝² : CategoryTheory.Category.{v₄, u₄} D
    inst✝¹ : CategoryTheory.Category.{v₅, u₅} E
    inst✝ : CategoryTheory.Category.{v₆, u₆} F
    G₁ : CategoryTheory.Functor A C
    G₂ : CategoryTheory.Functor C E
    H₁ : CategoryTheory.Functor B D
    H₂ : CategoryTheory.Functor D F
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor C D
    R₂ : CategoryTheory.Functor D C
    L₃ : CategoryTheory.Functor E F
    R₃ : CategoryTheory.Functor F E
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom (G₁.comp L₂) (L₁.comp H₁)
    β : Quiver.Hom (G₂.comp L₃) (L₂.comp H₂)
    ⊢ Eq ((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.leftAdjointSquare.v …
  -/
  unfold leftAdjointSquare.vcomp rightAdjointSquare.vcomp mateEquiv
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
    inst✝³ : CategoryTheory.Category.{v₃, u₃} C
    inst✝² : CategoryTheory.Category.{v₄, u₄} D
    inst✝¹ : CategoryTheory.Category.{v₅, u₅} E
    inst✝ : CategoryTheory.Category.{v₆, u₆} F
    G₁ : CategoryTheory.Functor A C
    G₂ : CategoryTheory.Functor C E
    H₁ : CategoryTheory.Functor B D
    H₂ : CategoryTheory.Functor D F
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor C D
    R₂ : CategoryTheory.Functor D C
    L₃ : CategoryTheory.Functor E F
    R₃ : CategoryTheory.Functor F E
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom (G₁.comp L₂) (L₁.comp H₁)
    β : Quiver.Hom (G₂.comp L₃) (L₂.comp H₂)
    ⊢ Eq ({ toFun := fun α => CategoryTheory.CategoryStruct.comp (CategoryTheory.w …
  -/
  ext b
  simp only [comp_obj, Equiv.coe_fn_mk, whiskerLeft_comp, whiskerLeft_twice, whiskerRight_comp,
    assoc, comp_app, whiskerLeft_app, whiskerRight_app, id_obj, Functor.comp_map,
    whiskerRight_twice]
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
    inst✝³ : CategoryTheory.Category.{v₃, u₃} C
    inst✝² : CategoryTheory.Category.{v₄, u₄} D
    inst✝¹ : CategoryTheory.Category.{v₅, u₅} E
    inst✝ : CategoryTheory.Category.{v₆, u₆} F
    G₁ : CategoryTheory.Functor A C
    G₂ : CategoryTheory.Functor C E
    H₁ : CategoryTheory.Functor B D
    H₂ : CategoryTheory.Functor D F
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor C D
    R₂ : CategoryTheory.Functor D C
    L₃ : CategoryTheory.Functor E F
    R₃ : CategoryTheory.Functor F E
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom (G₁.comp L₂) (L₁.comp H₁)
    β : Quiver.Hom (G₂.comp L₃) (L₂.comp H₂)
    b : B
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₃.unit.app (G₂.obj (G₁.obj (R₁.ob …
  -/
  slice_rhs 1 4 => rw [← assoc, ← assoc, ← unit_naturality (adj₃)]
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
    inst✝³ : CategoryTheory.Category.{v₃, u₃} C
    inst✝² : CategoryTheory.Category.{v₄, u₄} D
    inst✝¹ : CategoryTheory.Category.{v₅, u₅} E
    inst✝ : CategoryTheory.Category.{v₆, u₆} F
    G₁ : CategoryTheory.Functor A C
    G₂ : CategoryTheory.Functor C E
    H₁ : CategoryTheory.Functor B D
    H₂ : CategoryTheory.Functor D F
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor C D
    R₂ : CategoryTheory.Functor D C
    L₃ : CategoryTheory.Functor E F
    R₃ : CategoryTheory.Functor F E
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom (G₁.comp L₂) (L₁.comp H₁)
    β : Quiver.Hom (G₂.comp L₃) (L₂.comp H₂)
    b : B
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₃.unit.app (G₂.obj (G₁.obj (R₁.ob …
  -/
  rw [L₃.map_comp, R₃.map_comp]
  slice_rhs 2 4 =>
    rw [← R₃.map_comp, ← R₃.map_comp, ← assoc, ← L₃.map_comp, ← G₂.map_comp, ← G₂.map_comp]
    rw [← Functor.comp_map G₂ L₃, β.naturality]
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
    inst✝³ : CategoryTheory.Category.{v₃, u₃} C
    inst✝² : CategoryTheory.Category.{v₄, u₄} D
    inst✝¹ : CategoryTheory.Category.{v₅, u₅} E
    inst✝ : CategoryTheory.Category.{v₆, u₆} F
    G₁ : CategoryTheory.Functor A C
    G₂ : CategoryTheory.Functor C E
    H₁ : CategoryTheory.Functor B D
    H₂ : CategoryTheory.Functor D F
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor C D
    R₂ : CategoryTheory.Functor D C
    L₃ : CategoryTheory.Functor E F
    R₃ : CategoryTheory.Functor F E
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom (G₁.comp L₂) (L₁.comp H₁)
    β : Quiver.Hom (G₂.comp L₃) (L₂.comp H₂)
    b : B
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₃.unit.app (G₂.obj (G₁.obj (R₁.ob …
  -/
  rw [(L₂ ⋙ H₂).map_comp, R₃.map_comp, R₃.map_comp]
  slice_rhs 4 5 =>
    rw [← R₃.map_comp, Functor.comp_map L₂ _, ← Functor.comp_map _ L₂, ← H₂.map_comp]
    rw [adj₂.counit.naturality]
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
    inst✝³ : CategoryTheory.Category.{v₃, u₃} C
    inst✝² : CategoryTheory.Category.{v₄, u₄} D
    inst✝¹ : CategoryTheory.Category.{v₅, u₅} E
    inst✝ : CategoryTheory.Category.{v₆, u₆} F
    G₁ : CategoryTheory.Functor A C
    G₂ : CategoryTheory.Functor C E
    H₁ : CategoryTheory.Functor B D
    H₂ : CategoryTheory.Functor D F
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor C D
    R₂ : CategoryTheory.Functor D C
    L₃ : CategoryTheory.Functor E F
    R₃ : CategoryTheory.Functor F E
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom (G₁.comp L₂) (L₁.comp H₁)
    β : Quiver.Hom (G₂.comp L₃) (L₂.comp H₂)
    b : B
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₃.unit.app (G₂.obj (G₁.obj (R₁.ob …
  -/
  simp only [comp_obj, Functor.comp_map, map_comp, id_obj, Functor.id_map, assoc]
  slice_rhs 4 5 =>
    rw [← R₃.map_comp, ← H₂.map_comp, ← Functor.comp_map _ L₂, adj₂.counit.naturality]
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
    inst✝³ : CategoryTheory.Category.{v₃, u₃} C
    inst✝² : CategoryTheory.Category.{v₄, u₄} D
    inst✝¹ : CategoryTheory.Category.{v₅, u₅} E
    inst✝ : CategoryTheory.Category.{v₆, u₆} F
    G₁ : CategoryTheory.Functor A C
    G₂ : CategoryTheory.Functor C E
    H₁ : CategoryTheory.Functor B D
    H₂ : CategoryTheory.Functor D F
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor C D
    R₂ : CategoryTheory.Functor D C
    L₃ : CategoryTheory.Functor E F
    R₃ : CategoryTheory.Functor F E
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom (G₁.comp L₂) (L₁.comp H₁)
    β : Quiver.Hom (G₂.comp L₃) (L₂.comp H₂)
    b : B
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₃.unit.app (G₂.obj (G₁.obj (R₁.ob …
  -/
  simp only [comp_obj, id_obj, Functor.id_map, map_comp, assoc]
  slice_rhs 3 4 =>
    rw [← R₃.map_comp, ← H₂.map_comp, left_triangle_components]
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
    inst✝³ : CategoryTheory.Category.{v₃, u₃} C
    inst✝² : CategoryTheory.Category.{v₄, u₄} D
    inst✝¹ : CategoryTheory.Category.{v₅, u₅} E
    inst✝ : CategoryTheory.Category.{v₆, u₆} F
    G₁ : CategoryTheory.Functor A C
    G₂ : CategoryTheory.Functor C E
    H₁ : CategoryTheory.Functor B D
    H₂ : CategoryTheory.Functor D F
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor C D
    R₂ : CategoryTheory.Functor D C
    L₃ : CategoryTheory.Functor E F
    R₃ : CategoryTheory.Functor F E
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom (G₁.comp L₂) (L₁.comp H₁)
    β : Quiver.Hom (G₂.comp L₃) (L₂.comp H₂)
    b : B
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₃.unit.app (G₂.obj (G₁.obj (R₁.ob …
  -/
  simp only [map_id, id_comp]
  /-
    🎉 no goals
  -/


/-- Squares between left adjoints can be composed "horizontally" by pasting. -/
def leftAdjointSquare.hcomp :
    (G ⋙ L₂ ⟶ L₁ ⋙ H) → (H ⋙ L₄ ⟶ L₃ ⋙ K) → (G ⋙ (L₂ ⋙ L₄) ⟶ (L₁ ⋙ L₃) ⋙ K) := fun α β ↦
  (whiskerRight α L₄) ≫ (whiskerLeft L₁ β)


/-- Squares between right adjoints can be composed "horizontally" by pasting. -/
def rightAdjointSquare.hcomp :
    (R₁ ⋙ G ⟶ H ⋙ R₂) → (R₃ ⋙ H ⟶ K ⋙ R₄) → ((R₃ ⋙ R₁) ⋙ G ⟶ K ⋙ (R₄ ⋙ R₂)) := fun α β ↦
  (whiskerLeft R₃ α) ≫ (whiskerRight β R₂)


/-- The mates equivalence commutes with horizontal composition of squares. -/
theorem mateEquiv_hcomp
    (α : G ⋙ L₂ ⟶ L₁ ⋙ H) (β : H ⋙ L₄ ⟶ L₃ ⋙ K) :
    (mateEquiv (adj₁.comp adj₃) (adj₂.comp adj₄)) (leftAdjointSquare.hcomp α β) =
      rightAdjointSquare.hcomp (mateEquiv adj₁ adj₂ α) (mateEquiv adj₃ adj₄ β) := by
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
    inst✝³ : CategoryTheory.Category.{v₃, u₃} C
    inst✝² : CategoryTheory.Category.{v₄, u₄} D
    inst✝¹ : CategoryTheory.Category.{v₅, u₅} E
    inst✝ : CategoryTheory.Category.{v₆, u₆} F
    G : CategoryTheory.Functor A D
    H : CategoryTheory.Functor B E
    K : CategoryTheory.Functor C F
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor D E
    R₂ : CategoryTheory.Functor E D
    L₃ : CategoryTheory.Functor B C
    R₃ : CategoryTheory.Functor C B
    L₄ : CategoryTheory.Functor E F
    R₄ : CategoryTheory.Functor F E
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    adj₄ : CategoryTheory.Adjunction L₄ R₄
    α : Quiver.Hom (G.comp L₂) (L₁.comp H)
    β : Quiver.Hom (H.comp L₄) (L₃.comp K)
    ⊢ Eq ((CategoryTheory.mateEquiv (adj₁.comp adj₃) (adj₂.comp adj₄)) (CategoryTh …
  -/
  unfold leftAdjointSquare.hcomp rightAdjointSquare.hcomp mateEquiv Adjunction.comp
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
    inst✝³ : CategoryTheory.Category.{v₃, u₃} C
    inst✝² : CategoryTheory.Category.{v₄, u₄} D
    inst✝¹ : CategoryTheory.Category.{v₅, u₅} E
    inst✝ : CategoryTheory.Category.{v₆, u₆} F
    G : CategoryTheory.Functor A D
    H : CategoryTheory.Functor B E
    K : CategoryTheory.Functor C F
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor D E
    R₂ : CategoryTheory.Functor E D
    L₃ : CategoryTheory.Functor B C
    R₃ : CategoryTheory.Functor C B
    L₄ : CategoryTheory.Functor E F
    R₄ : CategoryTheory.Functor F E
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    adj₄ : CategoryTheory.Adjunction L₄ R₄
    α : Quiver.Hom (G.comp L₂) (L₁.comp H)
    β : Quiver.Hom (H.comp L₄) (L₃.comp K)
    ⊢ Eq ({ toFun := fun α => CategoryTheory.CategoryStruct.comp (CategoryTheory.w …
  -/
  ext c
  simp only [comp_obj, mk'_unit, whiskerLeft_comp, whiskerLeft_twice, mk'_counit, whiskerRight_comp,
    assoc, Equiv.coe_fn_mk, comp_app, whiskerLeft_app, whiskerRight_app, id_obj, associator_inv_app,
    Functor.comp_map, associator_hom_app, map_id, id_comp, whiskerRight_twice]
  slice_rhs 2 4 =>
    rw [← R₂.map_comp, ← R₂.map_comp, ← assoc, ← unit_naturality (adj₄)]
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
    inst✝³ : CategoryTheory.Category.{v₃, u₃} C
    inst✝² : CategoryTheory.Category.{v₄, u₄} D
    inst✝¹ : CategoryTheory.Category.{v₅, u₅} E
    inst✝ : CategoryTheory.Category.{v₆, u₆} F
    G : CategoryTheory.Functor A D
    H : CategoryTheory.Functor B E
    K : CategoryTheory.Functor C F
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor D E
    R₂ : CategoryTheory.Functor E D
    L₃ : CategoryTheory.Functor B C
    R₃ : CategoryTheory.Functor C B
    L₄ : CategoryTheory.Functor E F
    R₄ : CategoryTheory.Functor F E
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    adj₄ : CategoryTheory.Adjunction L₄ R₄
    α : Quiver.Hom (G.comp L₂) (L₁.comp H)
    β : Quiver.Hom (H.comp L₄) (L₃.comp K)
    c : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₂.unit.app (G.obj (R₁.obj (R₃.obj …
  -/
  rw [R₂.map_comp, L₄.map_comp, R₄.map_comp, R₂.map_comp]
  slice_rhs 4 5 =>
    rw [← R₂.map_comp, ← R₄.map_comp, ← Functor.comp_map _ L₄ , β.naturality]
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
    inst✝³ : CategoryTheory.Category.{v₃, u₃} C
    inst✝² : CategoryTheory.Category.{v₄, u₄} D
    inst✝¹ : CategoryTheory.Category.{v₅, u₅} E
    inst✝ : CategoryTheory.Category.{v₆, u₆} F
    G : CategoryTheory.Functor A D
    H : CategoryTheory.Functor B E
    K : CategoryTheory.Functor C F
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor D E
    R₂ : CategoryTheory.Functor E D
    L₃ : CategoryTheory.Functor B C
    R₃ : CategoryTheory.Functor C B
    L₄ : CategoryTheory.Functor E F
    R₄ : CategoryTheory.Functor F E
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    adj₄ : CategoryTheory.Adjunction L₄ R₄
    α : Quiver.Hom (G.comp L₂) (L₁.comp H)
    β : Quiver.Hom (H.comp L₄) (L₃.comp K)
    c : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₂.unit.app (G.obj (R₁.obj (R₃.obj …
  -/
  simp only [comp_obj, Functor.comp_map, map_comp, assoc]
  /-
    🎉 no goals
  -/


/-- Squares of squares between left adjoints can be composed by iterating vertical and horizontal
composition.
-/
def leftAdjointSquare.comp
    (α : G₁ ⋙ L₃ ⟶ L₁ ⋙ H₁) (β : H₁ ⋙ L₄ ⟶ L₂ ⋙ K₁)
    (γ : G₂ ⋙ L₅ ⟶ L₃ ⋙ H₂) (δ : H₂ ⋙ L₆ ⟶ L₄ ⋙ K₂) :
    ((G₁ ⋙ G₂) ⋙ (L₅ ⋙ L₆)) ⟶ ((L₁ ⋙ L₂) ⋙ (K₁ ⋙ K₂)) :=
  leftAdjointSquare.vcomp (leftAdjointSquare.hcomp α β) (leftAdjointSquare.hcomp γ δ)


theorem leftAdjointSquare.comp_vhcomp
    (α : G₁ ⋙ L₃ ⟶ L₁ ⋙ H₁) (β : H₁ ⋙ L₄ ⟶ L₂ ⋙ K₁)
    (γ : G₂ ⋙ L₅ ⟶ L₃ ⋙ H₂) (δ : H₂ ⋙ L₆ ⟶ L₄ ⋙ K₂) :
    leftAdjointSquare.comp α β γ δ =
      leftAdjointSquare.vcomp (leftAdjointSquare.hcomp α β) (leftAdjointSquare.hcomp γ δ) := rfl


/-- Horizontal and vertical composition of squares commutes.-/
theorem leftAdjointSquare.comp_hvcomp
    (α : G₁ ⋙ L₃ ⟶ L₁ ⋙ H₁) (β : H₁ ⋙ L₄ ⟶ L₂ ⋙ K₁)
    (γ : G₂ ⋙ L₅ ⟶ L₃ ⋙ H₂) (δ : H₂ ⋙ L₆ ⟶ L₄ ⋙ K₂) :
    leftAdjointSquare.comp α β γ δ =
      leftAdjointSquare.hcomp (leftAdjointSquare.vcomp α γ) (leftAdjointSquare.vcomp β δ) := by
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    X : Type u₇
    Y : Type u₈
    Z : Type u₉
    inst✝⁸ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁷ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁶ : CategoryTheory.Category.{v₃, u₃} C
    inst✝⁵ : CategoryTheory.Category.{v₄, u₄} D
    inst✝⁴ : CategoryTheory.Category.{v₅, u₅} E
    inst✝³ : CategoryTheory.Category.{v₆, u₆} F
    inst✝² : CategoryTheory.Category.{v₇, u₇} X
    inst✝¹ : CategoryTheory.Category.{v₈, u₈} Y
    inst✝ : CategoryTheory.Category.{v₉, u₉} Z
    G₁ : CategoryTheory.Functor A D
    H₁ : CategoryTheory.Functor B E
    K₁ : CategoryTheory.Functor C F
    G₂ : CategoryTheory.Functor D X
    H₂ : CategoryTheory.Functor E Y
    K₂ : CategoryTheory.Functor F Z
    L₁ : CategoryTheory.Functor A B
    L₂ : CategoryTheory.Functor B C
    L₃ : CategoryTheory.Functor D E
    L₄ : CategoryTheory.Functor E F
    L₅ : CategoryTheory.Functor X Y
    L₆ : CategoryTheory.Functor Y Z
    α : Quiver.Hom (G₁.comp L₃) (L₁.comp H₁)
    β : Quiver.Hom (H₁.comp L₄) (L₂.comp K₁)
    γ : Quiver.Hom (G₂.comp L₅) (L₃.comp H₂)
    δ : Quiver.Hom (H₂.comp L₆) (L₄.comp K₂)
    ⊢ Eq (CategoryTheory.leftAdjointSquare.comp α β γ δ) (CategoryTheory.leftAdjoi …
  -/
  unfold leftAdjointSquare.comp leftAdjointSquare.hcomp leftAdjointSquare.vcomp
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    X : Type u₇
    Y : Type u₈
    Z : Type u₉
    inst✝⁸ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁷ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁶ : CategoryTheory.Category.{v₃, u₃} C
    inst✝⁵ : CategoryTheory.Category.{v₄, u₄} D
    inst✝⁴ : CategoryTheory.Category.{v₅, u₅} E
    inst✝³ : CategoryTheory.Category.{v₆, u₆} F
    inst✝² : CategoryTheory.Category.{v₇, u₇} X
    inst✝¹ : CategoryTheory.Category.{v₈, u₈} Y
    inst✝ : CategoryTheory.Category.{v₉, u₉} Z
    G₁ : CategoryTheory.Functor A D
    H₁ : CategoryTheory.Functor B E
    K₁ : CategoryTheory.Functor C F
    G₂ : CategoryTheory.Functor D X
    H₂ : CategoryTheory.Functor E Y
    K₂ : CategoryTheory.Functor F Z
    L₁ : CategoryTheory.Functor A B
    L₂ : CategoryTheory.Functor B C
    L₃ : CategoryTheory.Functor D E
    L₄ : CategoryTheory.Functor E F
    L₅ : CategoryTheory.Functor X Y
    L₆ : CategoryTheory.Functor Y Z
    α : Quiver.Hom (G₁.comp L₃) (L₁.comp H₁)
    β : Quiver.Hom (H₁.comp L₄) (L₂.comp K₁)
    γ : Quiver.Hom (G₂.comp L₅) (L₃.comp H₂)
    δ : Quiver.Hom (H₂.comp L₆) (L₄.comp K₂)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft G₁ (Categ …
  -/
  unfold whiskerLeft whiskerRight
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    X : Type u₇
    Y : Type u₈
    Z : Type u₉
    inst✝⁸ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁷ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁶ : CategoryTheory.Category.{v₃, u₃} C
    inst✝⁵ : CategoryTheory.Category.{v₄, u₄} D
    inst✝⁴ : CategoryTheory.Category.{v₅, u₅} E
    inst✝³ : CategoryTheory.Category.{v₆, u₆} F
    inst✝² : CategoryTheory.Category.{v₇, u₇} X
    inst✝¹ : CategoryTheory.Category.{v₈, u₈} Y
    inst✝ : CategoryTheory.Category.{v₉, u₉} Z
    G₁ : CategoryTheory.Functor A D
    H₁ : CategoryTheory.Functor B E
    K₁ : CategoryTheory.Functor C F
    G₂ : CategoryTheory.Functor D X
    H₂ : CategoryTheory.Functor E Y
    K₂ : CategoryTheory.Functor F Z
    L₁ : CategoryTheory.Functor A B
    L₂ : CategoryTheory.Functor B C
    L₃ : CategoryTheory.Functor D E
    L₄ : CategoryTheory.Functor E F
    L₅ : CategoryTheory.Functor X Y
    L₆ : CategoryTheory.Functor Y Z
    α : Quiver.Hom (G₁.comp L₃) (L₁.comp H₁)
    β : Quiver.Hom (H₁.comp L₄) (L₂.comp K₁)
    γ : Quiver.Hom (G₂.comp L₅) (L₃.comp H₂)
    δ : Quiver.Hom (H₂.comp L₆) (L₄.comp K₂)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp { app := fun X_1 => (CategoryTheory.C …
  -/
  ext a
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    X : Type u₇
    Y : Type u₈
    Z : Type u₉
    inst✝⁸ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁷ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁶ : CategoryTheory.Category.{v₃, u₃} C
    inst✝⁵ : CategoryTheory.Category.{v₄, u₄} D
    inst✝⁴ : CategoryTheory.Category.{v₅, u₅} E
    inst✝³ : CategoryTheory.Category.{v₆, u₆} F
    inst✝² : CategoryTheory.Category.{v₇, u₇} X
    inst✝¹ : CategoryTheory.Category.{v₈, u₈} Y
    inst✝ : CategoryTheory.Category.{v₉, u₉} Z
    G₁ : CategoryTheory.Functor A D
    H₁ : CategoryTheory.Functor B E
    K₁ : CategoryTheory.Functor C F
    G₂ : CategoryTheory.Functor D X
    H₂ : CategoryTheory.Functor E Y
    K₂ : CategoryTheory.Functor F Z
    L₁ : CategoryTheory.Functor A B
    L₂ : CategoryTheory.Functor B C
    L₃ : CategoryTheory.Functor D E
    L₄ : CategoryTheory.Functor E F
    L₅ : CategoryTheory.Functor X Y
    L₆ : CategoryTheory.Functor Y Z
    α : Quiver.Hom (G₁.comp L₃) (L₁.comp H₁)
    β : Quiver.Hom (H₁.comp L₄) (L₂.comp K₁)
    γ : Quiver.Hom (G₂.comp L₅) (L₃.comp H₂)
    δ : Quiver.Hom (H₂.comp L₆) (L₄.comp K₂)
    a : A
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp { app := fun X_1 => (CategoryTheory. …
  -/
  simp only [comp_obj, comp_app, map_comp, assoc]
  slice_rhs 2 3 =>
    rw [← Functor.comp_map _ L₆, δ.naturality]
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    X : Type u₇
    Y : Type u₈
    Z : Type u₉
    inst✝⁸ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁷ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁶ : CategoryTheory.Category.{v₃, u₃} C
    inst✝⁵ : CategoryTheory.Category.{v₄, u₄} D
    inst✝⁴ : CategoryTheory.Category.{v₅, u₅} E
    inst✝³ : CategoryTheory.Category.{v₆, u₆} F
    inst✝² : CategoryTheory.Category.{v₇, u₇} X
    inst✝¹ : CategoryTheory.Category.{v₈, u₈} Y
    inst✝ : CategoryTheory.Category.{v₉, u₉} Z
    G₁ : CategoryTheory.Functor A D
    H₁ : CategoryTheory.Functor B E
    K₁ : CategoryTheory.Functor C F
    G₂ : CategoryTheory.Functor D X
    H₂ : CategoryTheory.Functor E Y
    K₂ : CategoryTheory.Functor F Z
    L₁ : CategoryTheory.Functor A B
    L₂ : CategoryTheory.Functor B C
    L₃ : CategoryTheory.Functor D E
    L₄ : CategoryTheory.Functor E F
    L₅ : CategoryTheory.Functor X Y
    L₆ : CategoryTheory.Functor Y Z
    α : Quiver.Hom (G₁.comp L₃) (L₁.comp H₁)
    β : Quiver.Hom (H₁.comp L₄) (L₂.comp K₁)
    γ : Quiver.Hom (G₂.comp L₅) (L₃.comp H₂)
    δ : Quiver.Hom (H₂.comp L₆) (L₄.comp K₂)
    a : A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (L₆.map (γ.app (G₁.obj a))) (Category …
  -/
  simp only [comp_obj, Functor.comp_map, assoc]
  /-
    🎉 no goals
  -/


/-- Squares of squares between right adjoints can be composed by iterating vertical and horizontal
composition.
-/
def rightAdjointSquare.comp
    (α : R₁ ⋙ G₁ ⟶ H₁ ⋙ R₃) (β : R₂ ⋙ H₁ ⟶ K₁ ⋙ R₄)
    (γ : R₃ ⋙ G₂ ⟶ H₂ ⋙ R₅) (δ : R₄ ⋙ H₂ ⟶ K₂ ⋙ R₆) :
    ((R₂ ⋙ R₁) ⋙ (G₁ ⋙ G₂) ⟶ (K₁ ⋙ K₂) ⋙ (R₆ ⋙ R₅)) :=
  rightAdjointSquare.vcomp (rightAdjointSquare.hcomp α β) (rightAdjointSquare.hcomp γ δ)


theorem rightAdjointSquare.comp_vhcomp
    (α : R₁ ⋙ G₁ ⟶ H₁ ⋙ R₃) (β : R₂ ⋙ H₁ ⟶ K₁ ⋙ R₄)
    (γ : R₃ ⋙ G₂ ⟶ H₂ ⋙ R₅) (δ : R₄ ⋙ H₂ ⟶ K₂ ⋙ R₆) :
    rightAdjointSquare.comp α β γ δ =
    rightAdjointSquare.vcomp (rightAdjointSquare.hcomp α β) (rightAdjointSquare.hcomp γ δ) := rfl


/-- Horizontal and vertical composition of squares commutes.-/
theorem rightAdjointSquare.comp_hvcomp
    (α : R₁ ⋙ G₁ ⟶ H₁ ⋙ R₃) (β : R₂ ⋙ H₁ ⟶ K₁ ⋙ R₄)
    (γ : R₃ ⋙ G₂ ⟶ H₂ ⋙ R₅) (δ : R₄ ⋙ H₂ ⟶ K₂ ⋙ R₆) :
    rightAdjointSquare.comp α β γ δ =
    rightAdjointSquare.hcomp (rightAdjointSquare.vcomp α γ) (rightAdjointSquare.vcomp β δ) := by
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    X : Type u₇
    Y : Type u₈
    Z : Type u₉
    inst✝⁸ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁷ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁶ : CategoryTheory.Category.{v₃, u₃} C
    inst✝⁵ : CategoryTheory.Category.{v₄, u₄} D
    inst✝⁴ : CategoryTheory.Category.{v₅, u₅} E
    inst✝³ : CategoryTheory.Category.{v₆, u₆} F
    inst✝² : CategoryTheory.Category.{v₇, u₇} X
    inst✝¹ : CategoryTheory.Category.{v₈, u₈} Y
    inst✝ : CategoryTheory.Category.{v₉, u₉} Z
    G₁ : CategoryTheory.Functor A D
    H₁ : CategoryTheory.Functor B E
    K₁ : CategoryTheory.Functor C F
    G₂ : CategoryTheory.Functor D X
    H₂ : CategoryTheory.Functor E Y
    K₂ : CategoryTheory.Functor F Z
    R₁ : CategoryTheory.Functor B A
    R₂ : CategoryTheory.Functor C B
    R₃ : CategoryTheory.Functor E D
    R₄ : CategoryTheory.Functor F E
    R₅ : CategoryTheory.Functor Y X
    R₆ : CategoryTheory.Functor Z Y
    α : Quiver.Hom (R₁.comp G₁) (H₁.comp R₃)
    β : Quiver.Hom (R₂.comp H₁) (K₁.comp R₄)
    γ : Quiver.Hom (R₃.comp G₂) (H₂.comp R₅)
    δ : Quiver.Hom (R₄.comp H₂) (K₂.comp R₆)
    ⊢ Eq (CategoryTheory.rightAdjointSquare.comp α β γ δ) (CategoryTheory.rightAdj …
  -/
  unfold rightAdjointSquare.comp rightAdjointSquare.hcomp rightAdjointSquare.vcomp
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    X : Type u₇
    Y : Type u₈
    Z : Type u₉
    inst✝⁸ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁷ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁶ : CategoryTheory.Category.{v₃, u₃} C
    inst✝⁵ : CategoryTheory.Category.{v₄, u₄} D
    inst✝⁴ : CategoryTheory.Category.{v₅, u₅} E
    inst✝³ : CategoryTheory.Category.{v₆, u₆} F
    inst✝² : CategoryTheory.Category.{v₇, u₇} X
    inst✝¹ : CategoryTheory.Category.{v₈, u₈} Y
    inst✝ : CategoryTheory.Category.{v₉, u₉} Z
    G₁ : CategoryTheory.Functor A D
    H₁ : CategoryTheory.Functor B E
    K₁ : CategoryTheory.Functor C F
    G₂ : CategoryTheory.Functor D X
    H₂ : CategoryTheory.Functor E Y
    K₂ : CategoryTheory.Functor F Z
    R₁ : CategoryTheory.Functor B A
    R₂ : CategoryTheory.Functor C B
    R₃ : CategoryTheory.Functor E D
    R₄ : CategoryTheory.Functor F E
    R₅ : CategoryTheory.Functor Y X
    R₆ : CategoryTheory.Functor Z Y
    α : Quiver.Hom (R₁.comp G₁) (H₁.comp R₃)
    β : Quiver.Hom (R₂.comp H₁) (K₁.comp R₄)
    γ : Quiver.Hom (R₃.comp G₂) (H₂.comp R₅)
    δ : Quiver.Hom (R₄.comp H₂) (K₂.comp R₆)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight (Categor …
  -/
  unfold whiskerLeft whiskerRight
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    X : Type u₇
    Y : Type u₈
    Z : Type u₉
    inst✝⁸ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁷ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁶ : CategoryTheory.Category.{v₃, u₃} C
    inst✝⁵ : CategoryTheory.Category.{v₄, u₄} D
    inst✝⁴ : CategoryTheory.Category.{v₅, u₅} E
    inst✝³ : CategoryTheory.Category.{v₆, u₆} F
    inst✝² : CategoryTheory.Category.{v₇, u₇} X
    inst✝¹ : CategoryTheory.Category.{v₈, u₈} Y
    inst✝ : CategoryTheory.Category.{v₉, u₉} Z
    G₁ : CategoryTheory.Functor A D
    H₁ : CategoryTheory.Functor B E
    K₁ : CategoryTheory.Functor C F
    G₂ : CategoryTheory.Functor D X
    H₂ : CategoryTheory.Functor E Y
    K₂ : CategoryTheory.Functor F Z
    R₁ : CategoryTheory.Functor B A
    R₂ : CategoryTheory.Functor C B
    R₃ : CategoryTheory.Functor E D
    R₄ : CategoryTheory.Functor F E
    R₅ : CategoryTheory.Functor Y X
    R₆ : CategoryTheory.Functor Z Y
    α : Quiver.Hom (R₁.comp G₁) (H₁.comp R₃)
    β : Quiver.Hom (R₂.comp H₁) (K₁.comp R₄)
    γ : Quiver.Hom (R₃.comp G₂) (H₂.comp R₅)
    δ : Quiver.Hom (R₄.comp H₂) (K₂.comp R₆)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp { app := fun X_1 => G₂.map ((Category …
  -/
  ext c
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    X : Type u₇
    Y : Type u₈
    Z : Type u₉
    inst✝⁸ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁷ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁶ : CategoryTheory.Category.{v₃, u₃} C
    inst✝⁵ : CategoryTheory.Category.{v₄, u₄} D
    inst✝⁴ : CategoryTheory.Category.{v₅, u₅} E
    inst✝³ : CategoryTheory.Category.{v₆, u₆} F
    inst✝² : CategoryTheory.Category.{v₇, u₇} X
    inst✝¹ : CategoryTheory.Category.{v₈, u₈} Y
    inst✝ : CategoryTheory.Category.{v₉, u₉} Z
    G₁ : CategoryTheory.Functor A D
    H₁ : CategoryTheory.Functor B E
    K₁ : CategoryTheory.Functor C F
    G₂ : CategoryTheory.Functor D X
    H₂ : CategoryTheory.Functor E Y
    K₂ : CategoryTheory.Functor F Z
    R₁ : CategoryTheory.Functor B A
    R₂ : CategoryTheory.Functor C B
    R₃ : CategoryTheory.Functor E D
    R₄ : CategoryTheory.Functor F E
    R₅ : CategoryTheory.Functor Y X
    R₆ : CategoryTheory.Functor Z Y
    α : Quiver.Hom (R₁.comp G₁) (H₁.comp R₃)
    β : Quiver.Hom (R₂.comp H₁) (K₁.comp R₄)
    γ : Quiver.Hom (R₃.comp G₂) (H₂.comp R₅)
    δ : Quiver.Hom (R₄.comp H₂) (K₂.comp R₆)
    c : C
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp { app := fun X_1 => G₂.map ((Categor …
  -/
  simp only [comp_obj, comp_app, map_comp, assoc]
  slice_rhs 2 3 =>
    rw [← Functor.comp_map _ R₅, ← γ.naturality]
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    X : Type u₇
    Y : Type u₈
    Z : Type u₉
    inst✝⁸ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁷ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁶ : CategoryTheory.Category.{v₃, u₃} C
    inst✝⁵ : CategoryTheory.Category.{v₄, u₄} D
    inst✝⁴ : CategoryTheory.Category.{v₅, u₅} E
    inst✝³ : CategoryTheory.Category.{v₆, u₆} F
    inst✝² : CategoryTheory.Category.{v₇, u₇} X
    inst✝¹ : CategoryTheory.Category.{v₈, u₈} Y
    inst✝ : CategoryTheory.Category.{v₉, u₉} Z
    G₁ : CategoryTheory.Functor A D
    H₁ : CategoryTheory.Functor B E
    K₁ : CategoryTheory.Functor C F
    G₂ : CategoryTheory.Functor D X
    H₂ : CategoryTheory.Functor E Y
    K₂ : CategoryTheory.Functor F Z
    R₁ : CategoryTheory.Functor B A
    R₂ : CategoryTheory.Functor C B
    R₃ : CategoryTheory.Functor E D
    R₄ : CategoryTheory.Functor F E
    R₅ : CategoryTheory.Functor Y X
    R₆ : CategoryTheory.Functor Z Y
    α : Quiver.Hom (R₁.comp G₁) (H₁.comp R₃)
    β : Quiver.Hom (R₂.comp H₁) (K₁.comp R₄)
    γ : Quiver.Hom (R₃.comp G₂) (H₂.comp R₅)
    δ : Quiver.Hom (R₄.comp H₂) (K₂.comp R₆)
    c : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G₂.map (α.app (R₂.obj c))) (Category …
  -/
  simp only [comp_obj, Functor.comp_map, assoc]
  /-
    🎉 no goals
  -/


/-- The mates equivalence commutes with composition of squares of squares. These results form the
basis for an isomorphism of double categories to be proven later.
-/
theorem mateEquiv_square
    (α : G₁ ⋙ L₃ ⟶ L₁ ⋙ H₁) (β : H₁ ⋙ L₄ ⟶ L₂ ⋙ K₁)
    (γ : G₂ ⋙ L₅ ⟶ L₃ ⋙ H₂) (δ : H₂ ⋙ L₆ ⟶ L₄ ⋙ K₂) :
    (mateEquiv (G := G₁ ⋙ G₂) (H := K₁ ⋙ K₂) (adj₁.comp adj₂) (adj₅.comp adj₆))
        (leftAdjointSquare.comp α β γ δ) =
      rightAdjointSquare.comp
        (mateEquiv adj₁ adj₃ α) (mateEquiv adj₂ adj₄ β)
        (mateEquiv adj₃ adj₅ γ) (mateEquiv adj₄ adj₆ δ) := by
  have vcomp :=
    mateEquiv_vcomp (adj₁.comp adj₂) (adj₃.comp adj₄) (adj₅.comp adj₆)
      (leftAdjointSquare.hcomp α β) (leftAdjointSquare.hcomp γ δ)
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    X : Type u₇
    Y : Type u₈
    Z : Type u₉
    inst✝⁸ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁷ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁶ : CategoryTheory.Category.{v₃, u₃} C
    inst✝⁵ : CategoryTheory.Category.{v₄, u₄} D
    inst✝⁴ : CategoryTheory.Category.{v₅, u₅} E
    inst✝³ : CategoryTheory.Category.{v₆, u₆} F
    inst✝² : CategoryTheory.Category.{v₇, u₇} X
    inst✝¹ : CategoryTheory.Category.{v₈, u₈} Y
    inst✝ : CategoryTheory.Category.{v₉, u₉} Z
    G₁ : CategoryTheory.Functor A D
    H₁ : CategoryTheory.Functor B E
    K₁ : CategoryTheory.Functor C F
    G₂ : CategoryTheory.Functor D X
    H₂ : CategoryTheory.Functor E Y
    K₂ : CategoryTheory.Functor F Z
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor B C
    R₂ : CategoryTheory.Functor C B
    L₃ : CategoryTheory.Functor D E
    R₃ : CategoryTheory.Functor E D
    L₄ : CategoryTheory.Functor E F
    R₄ : CategoryTheory.Functor F E
    L₅ : CategoryTheory.Functor X Y
    R₅ : CategoryTheory.Functor Y X
    L₆ : CategoryTheory.Functor Y Z
    R₆ : CategoryTheory.Functor Z Y
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    adj₄ : CategoryTheory.Adjunction L₄ R₄
    adj₅ : CategoryTheory.Adjunction L₅ R₅
    adj₆ : CategoryTheory.Adjunction L₆ R₆
    α : Quiver.Hom (G₁.comp L₃) (L₁.comp H₁)
    β : Quiver.Hom (H₁.comp L₄) (L₂.comp K₁)
    γ : Quiver.Hom (G₂.comp L₅) (L₃.comp H₂)
    δ : Quiver.Hom (H₂.comp L₆) (L₄.comp K₂)
    vcomp : Eq ((CategoryTheory.mateEquiv (adj₁.comp adj₂) (adj₅.comp adj₆)) (Cate …
    ⊢ Eq ((CategoryTheory.mateEquiv (adj₁.comp adj₂) (adj₅.comp adj₆)) (CategoryTh …
  -/
  have hcomp1 := mateEquiv_hcomp adj₁ adj₃ adj₂ adj₄ α β
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    X : Type u₇
    Y : Type u₈
    Z : Type u₉
    inst✝⁸ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁷ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁶ : CategoryTheory.Category.{v₃, u₃} C
    inst✝⁵ : CategoryTheory.Category.{v₄, u₄} D
    inst✝⁴ : CategoryTheory.Category.{v₅, u₅} E
    inst✝³ : CategoryTheory.Category.{v₆, u₆} F
    inst✝² : CategoryTheory.Category.{v₇, u₇} X
    inst✝¹ : CategoryTheory.Category.{v₈, u₈} Y
    inst✝ : CategoryTheory.Category.{v₉, u₉} Z
    G₁ : CategoryTheory.Functor A D
    H₁ : CategoryTheory.Functor B E
    K₁ : CategoryTheory.Functor C F
    G₂ : CategoryTheory.Functor D X
    H₂ : CategoryTheory.Functor E Y
    K₂ : CategoryTheory.Functor F Z
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor B C
    R₂ : CategoryTheory.Functor C B
    L₃ : CategoryTheory.Functor D E
    R₃ : CategoryTheory.Functor E D
    L₄ : CategoryTheory.Functor E F
    R₄ : CategoryTheory.Functor F E
    L₅ : CategoryTheory.Functor X Y
    R₅ : CategoryTheory.Functor Y X
    L₆ : CategoryTheory.Functor Y Z
    R₆ : CategoryTheory.Functor Z Y
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    adj₄ : CategoryTheory.Adjunction L₄ R₄
    adj₅ : CategoryTheory.Adjunction L₅ R₅
    adj₆ : CategoryTheory.Adjunction L₆ R₆
    α : Quiver.Hom (G₁.comp L₃) (L₁.comp H₁)
    β : Quiver.Hom (H₁.comp L₄) (L₂.comp K₁)
    γ : Quiver.Hom (G₂.comp L₅) (L₃.comp H₂)
    δ : Quiver.Hom (H₂.comp L₆) (L₄.comp K₂)
    vcomp : Eq ((CategoryTheory.mateEquiv (adj₁.comp adj₂) (adj₅.comp adj₆)) (Cate …
    hcomp1 : Eq ((CategoryTheory.mateEquiv (adj₁.comp adj₂) (adj₃.comp adj₄)) (Cat …
    ⊢ Eq ((CategoryTheory.mateEquiv (adj₁.comp adj₂) (adj₅.comp adj₆)) (CategoryTh …
  -/
  have hcomp2 := mateEquiv_hcomp adj₃ adj₅ adj₄ adj₆ γ δ
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    X : Type u₇
    Y : Type u₈
    Z : Type u₉
    inst✝⁸ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁷ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁶ : CategoryTheory.Category.{v₃, u₃} C
    inst✝⁵ : CategoryTheory.Category.{v₄, u₄} D
    inst✝⁴ : CategoryTheory.Category.{v₅, u₅} E
    inst✝³ : CategoryTheory.Category.{v₆, u₆} F
    inst✝² : CategoryTheory.Category.{v₇, u₇} X
    inst✝¹ : CategoryTheory.Category.{v₈, u₈} Y
    inst✝ : CategoryTheory.Category.{v₉, u₉} Z
    G₁ : CategoryTheory.Functor A D
    H₁ : CategoryTheory.Functor B E
    K₁ : CategoryTheory.Functor C F
    G₂ : CategoryTheory.Functor D X
    H₂ : CategoryTheory.Functor E Y
    K₂ : CategoryTheory.Functor F Z
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor B C
    R₂ : CategoryTheory.Functor C B
    L₃ : CategoryTheory.Functor D E
    R₃ : CategoryTheory.Functor E D
    L₄ : CategoryTheory.Functor E F
    R₄ : CategoryTheory.Functor F E
    L₅ : CategoryTheory.Functor X Y
    R₅ : CategoryTheory.Functor Y X
    L₆ : CategoryTheory.Functor Y Z
    R₆ : CategoryTheory.Functor Z Y
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    adj₄ : CategoryTheory.Adjunction L₄ R₄
    adj₅ : CategoryTheory.Adjunction L₅ R₅
    adj₆ : CategoryTheory.Adjunction L₆ R₆
    α : Quiver.Hom (G₁.comp L₃) (L₁.comp H₁)
    β : Quiver.Hom (H₁.comp L₄) (L₂.comp K₁)
    γ : Quiver.Hom (G₂.comp L₅) (L₃.comp H₂)
    δ : Quiver.Hom (H₂.comp L₆) (L₄.comp K₂)
    vcomp : Eq ((CategoryTheory.mateEquiv (adj₁.comp adj₂) (adj₅.comp adj₆)) (Cate …
    hcomp1 : Eq ((CategoryTheory.mateEquiv (adj₁.comp adj₂) (adj₃.comp adj₄)) (Cat …
    hcomp2 : Eq ((CategoryTheory.mateEquiv (adj₃.comp adj₄) (adj₅.comp adj₆)) (Cat …
    ⊢ Eq ((CategoryTheory.mateEquiv (adj₁.comp adj₂) (adj₅.comp adj₆)) (CategoryTh …
  -/
  rw [hcomp1, hcomp2] at vcomp
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    E : Type u₅
    F : Type u₆
    X : Type u₇
    Y : Type u₈
    Z : Type u₉
    inst✝⁸ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁷ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁶ : CategoryTheory.Category.{v₃, u₃} C
    inst✝⁵ : CategoryTheory.Category.{v₄, u₄} D
    inst✝⁴ : CategoryTheory.Category.{v₅, u₅} E
    inst✝³ : CategoryTheory.Category.{v₆, u₆} F
    inst✝² : CategoryTheory.Category.{v₇, u₇} X
    inst✝¹ : CategoryTheory.Category.{v₈, u₈} Y
    inst✝ : CategoryTheory.Category.{v₉, u₉} Z
    G₁ : CategoryTheory.Functor A D
    H₁ : CategoryTheory.Functor B E
    K₁ : CategoryTheory.Functor C F
    G₂ : CategoryTheory.Functor D X
    H₂ : CategoryTheory.Functor E Y
    K₂ : CategoryTheory.Functor F Z
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor B C
    R₂ : CategoryTheory.Functor C B
    L₃ : CategoryTheory.Functor D E
    R₃ : CategoryTheory.Functor E D
    L₄ : CategoryTheory.Functor E F
    R₄ : CategoryTheory.Functor F E
    L₅ : CategoryTheory.Functor X Y
    R₅ : CategoryTheory.Functor Y X
    L₆ : CategoryTheory.Functor Y Z
    R₆ : CategoryTheory.Functor Z Y
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    adj₄ : CategoryTheory.Adjunction L₄ R₄
    adj₅ : CategoryTheory.Adjunction L₅ R₅
    adj₆ : CategoryTheory.Adjunction L₆ R₆
    α : Quiver.Hom (G₁.comp L₃) (L₁.comp H₁)
    β : Quiver.Hom (H₁.comp L₄) (L₂.comp K₁)
    γ : Quiver.Hom (G₂.comp L₅) (L₃.comp H₂)
    δ : Quiver.Hom (H₂.comp L₆) (L₄.comp K₂)
    vcomp : Eq ((CategoryTheory.mateEquiv (adj₁.comp adj₂) (adj₅.comp adj₆)) (Cate …
    hcomp1 : Eq ((CategoryTheory.mateEquiv (adj₁.comp adj₂) (adj₃.comp adj₄)) (Cat …
    hcomp2 : Eq ((CategoryTheory.mateEquiv (adj₃.comp adj₄) (adj₅.comp adj₆)) (Cat …
    ⊢ Eq ((CategoryTheory.mateEquiv (adj₁.comp adj₂) (adj₅.comp adj₆)) (CategoryTh …
  -/
  exact vcomp
  /-
    🎉 no goals
  -/


/-- Given two adjunctions `L₁ ⊣ R₁` and `L₂ ⊣ R₂` both between categories `C`, `D`, there is a
bijection between natural transformations `L₂ ⟶ L₁` and natural transformations `R₁ ⟶ R₂`. This is
defined as a special case of `mateEquiv`, where the two "vertical" functors are identity, modulo
composition with the unitors. Corresponding natural transformations are called `conjugateEquiv`.
TODO: Generalise to when the two vertical functors are equivalences rather than being exactly `𝟭`.

Furthermore, this bijection preserves (and reflects) isomorphisms, i.e. a transformation is an iso
iff its image under the bijection is an iso, see eg `CategoryTheory.conjugateIsoEquiv`.
This is in contrast to the general case `mateEquiv` which does not in general have this property.
-/
@[simps!]
def conjugateEquiv : (L₂ ⟶ L₁) ≃ (R₁ ⟶ R₂) :=
  calc
    (L₂ ⟶ L₁) ≃ _ := (Iso.homCongr L₂.leftUnitor L₁.rightUnitor).symm
    _ ≃ _ := mateEquiv adj₁ adj₂
    _ ≃ (R₁ ⟶ R₂) := R₁.rightUnitor.homCongr R₂.leftUnitor


@[deprecated (since := "2024-07-09")] alias transferNatTransSelf := conjugateEquiv


/-- A component of a transposed form of the conjugation definition. -/
theorem conjugateEquiv_counit (α : L₂ ⟶ L₁) (d : D) :
    L₂.map ((conjugateEquiv adj₁ adj₂ α).app _) ≫ adj₂.counit.app d =
      α.app _ ≫ adj₁.counit.app d := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ : CategoryTheory.Functor C D
    R₁ R₂ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom L₂ L₁
    d : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (L₂.map (((CategoryTheory.conjugateEq …
  -/
  dsimp [conjugateEquiv]
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ : CategoryTheory.Functor C D
    R₁ R₂ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom L₂ L₁
    d : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (L₂.map (CategoryTheory.CategoryStruc …
  -/
  rw [id_comp, comp_id]
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ : CategoryTheory.Functor C D
    R₁ R₂ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom L₂ L₁
    d : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (L₂.map (CategoryTheory.CategoryStruc …
  -/
  have := mateEquiv_counit adj₁ adj₂ (L₂.leftUnitor.hom ≫ α ≫ L₁.rightUnitor.inv) d
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ : CategoryTheory.Functor C D
    R₁ R₂ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom L₂ L₁
    d : D
    this : Eq (CategoryTheory.CategoryStruct.comp (L₂.map (((CategoryTheory.mateEq …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (L₂.map (CategoryTheory.CategoryStruc …
  -/
  dsimp at this
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ : CategoryTheory.Functor C D
    R₁ R₂ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom L₂ L₁
    d : D
    this : Eq (CategoryTheory.CategoryStruct.comp (L₂.map (CategoryTheory.Category …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (L₂.map (CategoryTheory.CategoryStruc …
  -/
  rw [this]
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ : CategoryTheory.Functor C D
    R₁ R₂ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom L₂ L₁
    d : D
    this : Eq (CategoryTheory.CategoryStruct.comp (L₂.map (CategoryTheory.Category …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [comp_id, id_comp]
  /-
    🎉 no goals
  -/


/-- A component of a transposed form of the inverse conjugation definition. -/
theorem conjugateEquiv_counit_symm (α : R₁ ⟶ R₂) (d : D) :
    L₂.map (α.app _) ≫ adj₂.counit.app d =
      ((conjugateEquiv adj₁ adj₂).symm α).app _ ≫ adj₁.counit.app d := by
    /-
      C : Type u₁
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L₁ L₂ : CategoryTheory.Functor C D
      R₁ R₂ : CategoryTheory.Functor D C
      adj₁ : CategoryTheory.Adjunction L₁ R₁
      adj₂ : CategoryTheory.Adjunction L₂ R₂
      α : Quiver.Hom R₁ R₂
      d : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (L₂.map (α.app d)) (adj₂.counit.app d …
    -/
    conv_lhs => rw [← (conjugateEquiv adj₁ adj₂).right_inv α]
    /-
      C : Type u₁
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L₁ L₂ : CategoryTheory.Functor C D
      R₁ R₂ : CategoryTheory.Functor D C
      adj₁ : CategoryTheory.Adjunction L₁ R₁
      adj₂ : CategoryTheory.Adjunction L₂ R₂
      α : Quiver.Hom R₁ R₂
      d : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (L₂.map (((CategoryTheory.conjugateEq …
    -/
    exact (conjugateEquiv_counit adj₁ adj₂ ((conjugateEquiv adj₁ adj₂).symm α) d)
    /-
      🎉 no goals
    -/


/-- A component of a transposed form of the conjugation definition. -/
theorem unit_conjugateEquiv (α : L₂ ⟶ L₁) (c : C) :
    adj₁.unit.app _ ≫ (conjugateEquiv adj₁ adj₂ α).app _ =
      adj₂.unit.app c ≫ R₂.map (α.app _) := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ : CategoryTheory.Functor C D
    R₁ R₂ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom L₂ L₁
    c : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₁.unit.app c) (((CategoryTheory.c …
  -/
  dsimp [conjugateEquiv]
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ : CategoryTheory.Functor C D
    R₁ R₂ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom L₂ L₁
    c : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₁.unit.app c) (CategoryTheory.Cat …
  -/
  rw [id_comp, comp_id]
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ : CategoryTheory.Functor C D
    R₁ R₂ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom L₂ L₁
    c : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₁.unit.app c) (CategoryTheory.Cat …
  -/
  have := unit_mateEquiv adj₁ adj₂ (L₂.leftUnitor.hom ≫ α ≫ L₁.rightUnitor.inv) c
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ : CategoryTheory.Functor C D
    R₁ R₂ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom L₂ L₁
    c : C
    this : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id C).m …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₁.unit.app c) (CategoryTheory.Cat …
  -/
  dsimp at this
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ : CategoryTheory.Functor C D
    R₁ R₂ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom L₂ L₁
    c : C
    this : Eq (CategoryTheory.CategoryStruct.comp (adj₁.unit.app c) (CategoryTheor …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₁.unit.app c) (CategoryTheory.Cat …
  -/
  rw [this]
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ : CategoryTheory.Functor C D
    R₁ R₂ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom L₂ L₁
    c : C
    this : Eq (CategoryTheory.CategoryStruct.comp (adj₁.unit.app c) (CategoryTheor …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₂.unit.app c) (R₂.map (CategoryTh …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- A component of a transposed form of the inverse conjugation definition. -/
theorem unit_conjugateEquiv_symm (α : R₁ ⟶ R₂) (c : C) :
    adj₁.unit.app _ ≫ α.app _ =
      adj₂.unit.app c ≫ R₂.map (((conjugateEquiv adj₁ adj₂).symm α).app _) := by
    /-
      C : Type u₁
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L₁ L₂ : CategoryTheory.Functor C D
      R₁ R₂ : CategoryTheory.Functor D C
      adj₁ : CategoryTheory.Adjunction L₁ R₁
      adj₂ : CategoryTheory.Adjunction L₂ R₂
      α : Quiver.Hom R₁ R₂
      c : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₁.unit.app c) (α.app (L₁.obj c))) …
    -/
    conv_lhs => rw [← (conjugateEquiv adj₁ adj₂).right_inv α]
    /-
      C : Type u₁
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L₁ L₂ : CategoryTheory.Functor C D
      R₁ R₂ : CategoryTheory.Functor D C
      adj₁ : CategoryTheory.Adjunction L₁ R₁
      adj₂ : CategoryTheory.Adjunction L₂ R₂
      α : Quiver.Hom R₁ R₂
      c : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₁.unit.app c) (((CategoryTheory.c …
    -/
    exact (unit_conjugateEquiv adj₁ adj₂ ((conjugateEquiv adj₁ adj₂).symm α) c)
    /-
      🎉 no goals
    -/


@[simp]
theorem conjugateEquiv_id : conjugateEquiv adj₁ adj₁ (𝟙 _) = 𝟙 _ := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ : CategoryTheory.Functor C D
    R₁ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    ⊢ Eq ((CategoryTheory.conjugateEquiv adj₁ adj₁) (CategoryTheory.CategoryStruct …
  -/
  ext
  /-
    case w.h
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ : CategoryTheory.Functor C D
    R₁ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    x✝ : D
    ⊢ Eq (((CategoryTheory.conjugateEquiv adj₁ adj₁) (CategoryTheory.CategoryStruc …
  -/
  dsimp [conjugateEquiv, mateEquiv]
  /-
    case w.h
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ : CategoryTheory.Functor C D
    R₁ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    x✝ : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (R₁ …
  -/
  simp only [comp_id, map_id, id_comp, right_triangle_components]
  /-
    🎉 no goals
  -/


@[simp]
theorem conjugateEquiv_symm_id : (conjugateEquiv adj₁ adj₁).symm (𝟙 _) = 𝟙 _ := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ : CategoryTheory.Functor C D
    R₁ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    ⊢ Eq ((CategoryTheory.conjugateEquiv adj₁ adj₁).symm (CategoryTheory.CategoryS …
  -/
  rw [Equiv.symm_apply_eq]
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ : CategoryTheory.Functor C D
    R₁ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    ⊢ Eq (CategoryTheory.CategoryStruct.id R₁) ((CategoryTheory.conjugateEquiv adj …
  -/
  simp only [conjugateEquiv_id]
  /-
    🎉 no goals
  -/


theorem conjugateEquiv_adjunction_id {L R : C ⥤ C} (adj : L ⊣ R) (α : 𝟭 C ⟶ L) (c : C) :
    (conjugateEquiv adj Adjunction.id α).app c = α.app (R.obj c) ≫ adj.counit.app c := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    L R : CategoryTheory.Functor C C
    adj : CategoryTheory.Adjunction L R
    α : Quiver.Hom (CategoryTheory.Functor.id C) L
    c : C
    ⊢ Eq (((CategoryTheory.conjugateEquiv adj CategoryTheory.Adjunction.id) α).app …
  -/
  dsimp [conjugateEquiv, mateEquiv, Adjunction.id]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    L R : CategoryTheory.Functor C C
    adj : CategoryTheory.Adjunction L R
    α : Quiver.Hom (CategoryTheory.Functor.id C) L
    c : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (R. …
  -/
  simp only [comp_id, id_comp]
  /-
    🎉 no goals
  -/


theorem conjugateEquiv_adjunction_id_symm {L R : C ⥤ C} (adj : L ⊣ R) (α : R ⟶ 𝟭 C) (c : C) :
    ((conjugateEquiv adj Adjunction.id).symm α).app c = adj.unit.app c ≫ α.app (L.obj c) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    L R : CategoryTheory.Functor C C
    adj : CategoryTheory.Adjunction L R
    α : Quiver.Hom R (CategoryTheory.Functor.id C)
    c : C
    ⊢ Eq (((CategoryTheory.conjugateEquiv adj CategoryTheory.Adjunction.id).symm α …
  -/
  dsimp [conjugateEquiv, mateEquiv, Adjunction.id]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    L R : CategoryTheory.Functor C C
    adj : CategoryTheory.Adjunction L R
    α : Quiver.Hom R (CategoryTheory.Functor.id C)
    c : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id c)  …
  -/
  simp only [comp_id, id_comp]
  /-
    🎉 no goals
  -/

@[simp]
theorem conjugateEquiv_comp (α : L₂ ⟶ L₁) (β : L₃ ⟶ L₂) :
    conjugateEquiv adj₁ adj₂ α ≫ conjugateEquiv adj₂ adj₃ β =
      conjugateEquiv adj₁ adj₃ (β ≫ α) := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ L₃ : CategoryTheory.Functor C D
    R₁ R₂ R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom L₂ L₁
    β : Quiver.Hom L₃ L₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.conjugateEquiv adj₁  …
  -/
  ext d
  /-
    case w.h
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ L₃ : CategoryTheory.Functor C D
    R₁ R₂ R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom L₂ L₁
    β : Quiver.Hom L₃ L₂
    d : D
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.conjugateEquiv adj₁ …
  -/
  dsimp [conjugateEquiv, mateEquiv]
  have vcomp := mateEquiv_vcomp adj₁ adj₂ adj₃
    (L₂.leftUnitor.hom ≫ α ≫ L₁.rightUnitor.inv)
    (L₃.leftUnitor.hom ≫ β ≫ L₂.rightUnitor.inv)
  /-
    case w.h
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ L₃ : CategoryTheory.Functor C D
    R₁ R₂ R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom L₂ L₁
    β : Quiver.Hom L₃ L₂
    d : D
    vcomp : Eq ((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.leftAdjointSq …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  have vcompd := congr_app vcomp d
  /-
    case w.h
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ L₃ : CategoryTheory.Functor C D
    R₁ R₂ R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom L₂ L₁
    β : Quiver.Hom L₃ L₂
    d : D
    vcomp : Eq ((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.leftAdjointSq …
    vcompd : Eq (((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.leftAdjoint …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp [mateEquiv, leftAdjointSquare.vcomp, rightAdjointSquare.vcomp] at vcompd
  /-
    case w.h
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ L₃ : CategoryTheory.Functor C D
    R₁ R₂ R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom L₂ L₁
    β : Quiver.Hom L₃ L₂
    d : D
    vcomp : Eq ((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.leftAdjointSq …
    vcompd : Eq (CategoryTheory.CategoryStruct.comp (adj₃.unit.app (R₁.obj d)) (Ca …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [comp_id, id_comp, assoc, map_comp] at vcompd ⊢
  /-
    case w.h
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ L₃ : CategoryTheory.Functor C D
    R₁ R₂ R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom L₂ L₁
    β : Quiver.Hom L₃ L₂
    d : D
    vcomp : Eq ((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.leftAdjointSq …
    vcompd : Eq (CategoryTheory.CategoryStruct.comp (adj₃.unit.app (R₁.obj d)) (Ca …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₂.unit.app (R₁.obj d)) (CategoryT …
  -/
  rw [vcompd]
  /-
    🎉 no goals
  -/


@[simp]
theorem conjugateEquiv_symm_comp (α : R₁ ⟶ R₂) (β : R₂ ⟶ R₃) :
    (conjugateEquiv adj₂ adj₃).symm β ≫ (conjugateEquiv adj₁ adj₂).symm α =
      (conjugateEquiv adj₁ adj₃).symm (α ≫ β) := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ L₃ : CategoryTheory.Functor C D
    R₁ R₂ R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom R₁ R₂
    β : Quiver.Hom R₂ R₃
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.conjugateEquiv adj₂  …
  -/
  rw [Equiv.eq_symm_apply, ← conjugateEquiv_comp _ adj₂]
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ L₃ : CategoryTheory.Functor C D
    R₁ R₂ R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom R₁ R₂
    β : Quiver.Hom R₂ R₃
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.conjugateEquiv adj₁  …
  -/
  simp only [Equiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


theorem conjugateEquiv_comm {α : L₂ ⟶ L₁} {β : L₁ ⟶ L₂} (βα : β ≫ α = 𝟙 _) :
    conjugateEquiv adj₁ adj₂ α ≫ conjugateEquiv adj₂ adj₁ β = 𝟙 _ := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ : CategoryTheory.Functor C D
    R₁ R₂ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom L₂ L₁
    β : Quiver.Hom L₁ L₂
    βα : Eq (CategoryTheory.CategoryStruct.comp β α) (CategoryTheory.CategoryStruc …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.conjugateEquiv adj₁  …
  -/
  rw [conjugateEquiv_comp, βα, conjugateEquiv_id]
  /-
    🎉 no goals
  -/


theorem conjugateEquiv_symm_comm {α : R₁ ⟶ R₂} {β : R₂ ⟶ R₁} (αβ : α ≫ β = 𝟙 _) :
    (conjugateEquiv adj₂ adj₁).symm β ≫ (conjugateEquiv adj₁ adj₂).symm α = 𝟙 _ := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ : CategoryTheory.Functor C D
    R₁ R₂ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom R₁ R₂
    β : Quiver.Hom R₂ R₁
    αβ : Eq (CategoryTheory.CategoryStruct.comp α β) (CategoryTheory.CategoryStruc …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.conjugateEquiv adj₂  …
  -/
  rw [conjugateEquiv_symm_comp, αβ, conjugateEquiv_symm_id]
  /-
    🎉 no goals
  -/


/-- If `α` is an isomorphism between left adjoints, then its conjugate transformation is an
isomorphism. The converse is given in `conjugateEquiv_of_iso`.
-/
instance conjugateEquiv_iso (α : L₂ ⟶ L₁) [IsIso α] :
    IsIso (conjugateEquiv adj₁ adj₂ α) :=
  ⟨⟨conjugateEquiv adj₂ adj₁ (inv α),
                                   /-
                                     C : Type u₁
                                     D : Type u₂
                                     inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                     inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                     L₁ L₂ : CategoryTheory.Functor C D
                                     R₁ R₂ : CategoryTheory.Functor D C
                                     adj₁ : CategoryTheory.Adjunction L₁ R₁
                                     adj₂ : CategoryTheory.Adjunction L₂ R₂
                                     α : Quiver.Hom L₂ L₁
                                     inst✝ : CategoryTheory.IsIso α
                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv α) α) (CategoryTh …
                                   -/
                                   /-
                                     🎉 no goals
                                   -/
      ⟨conjugateEquiv_comm _ _ (by simp), conjugateEquiv_comm _ _ (by simp)⟩⟩⟩
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


/-- If `α` is an isomorphism between right adjoints, then its conjugate transformation is an
isomorphism. The converse is given in `conjugateEquiv_symm_of_iso`.
-/
instance conjugateEquiv_symm_iso (α : R₁ ⟶ R₂) [IsIso α] :
    IsIso ((conjugateEquiv adj₁ adj₂).symm α) :=
  ⟨⟨(conjugateEquiv adj₂ adj₁).symm (inv α),
                                        /-
                                          C : Type u₁
                                          D : Type u₂
                                          inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                          L₁ L₂ : CategoryTheory.Functor C D
                                          R₁ R₂ : CategoryTheory.Functor D C
                                          adj₁ : CategoryTheory.Adjunction L₁ R₁
                                          adj₂ : CategoryTheory.Adjunction L₂ R₂
                                          α : Quiver.Hom R₁ R₂
                                          inst✝ : CategoryTheory.IsIso α
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv α) α) (CategoryTh …
                                        -/
                                        /-
                                          🎉 no goals
                                        -/
      ⟨conjugateEquiv_symm_comm _ _ (by simp), conjugateEquiv_symm_comm _ _ (by simp)⟩⟩⟩
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


/-- If `α` is a natural transformation between left adjoints whose conjugate natural transformation
is an isomorphism, then `α` is an isomorphism. The converse is given in `Conjugate_iso`.
-/
theorem conjugateEquiv_of_iso (α : L₂ ⟶ L₁) [IsIso (conjugateEquiv adj₁ adj₂ α)] :
    IsIso α := by
  suffices IsIso ((conjugateEquiv adj₁ adj₂).symm (conjugateEquiv adj₁ adj₂ α))
    by simpa using this
  /-
    C : Type u₁
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ : CategoryTheory.Functor C D
    R₁ R₂ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom L₂ L₁
    inst✝ : CategoryTheory.IsIso ((CategoryTheory.conjugateEquiv adj₁ adj₂) α)
    ⊢ CategoryTheory.IsIso ((CategoryTheory.conjugateEquiv adj₁ adj₂).symm ((Categ …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/--
If `α` is a natural transformation between right adjoints whose conjugate natural transformation is
an isomorphism, then `α` is an isomorphism. The converse is given in `conjugateEquiv_symm_iso`.
-/
theorem conjugateEquiv_symm_of_iso (α : R₁ ⟶ R₂)
    [IsIso ((conjugateEquiv adj₁ adj₂).symm α)] : IsIso α := by
  suffices IsIso ((conjugateEquiv adj₁ adj₂) ((conjugateEquiv adj₁ adj₂).symm α))
    by simpa using this
  /-
    C : Type u₁
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L₁ L₂ : CategoryTheory.Functor C D
    R₁ R₂ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    α : Quiver.Hom R₁ R₂
    inst✝ : CategoryTheory.IsIso ((CategoryTheory.conjugateEquiv adj₁ adj₂).symm α)
    ⊢ CategoryTheory.IsIso ((CategoryTheory.conjugateEquiv adj₁ adj₂) ((CategoryTh …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Thus conjugation defines an equivalence between natural isomorphisms. -/
@[simps]
def conjugateIsoEquiv : (L₂ ≅ L₁) ≃ (R₁ ≅ R₂) where
  toFun α := {
    hom := conjugateEquiv adj₁ adj₂ α.hom
    inv := conjugateEquiv adj₂ adj₁ α.inv
  }
  invFun β := {
    hom := (conjugateEquiv adj₁ adj₂).symm β.hom
    inv := (conjugateEquiv adj₂ adj₁).symm β.inv
  }
                 /-
                   C : Type u₁
                   D : Type u₂
                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                   inst✝ : CategoryTheory.Category.{v₂, u₂} D
                   L₁ L₂ : CategoryTheory.Functor C D
                   R₁ R₂ : CategoryTheory.Functor D C
                   adj₁ : CategoryTheory.Adjunction L₁ R₁
                   adj₂ : CategoryTheory.Adjunction L₂ R₂
                   ⊢ Function.LeftInverse (fun β => { hom := (CategoryTheory.conjugateEquiv adj₁  …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    C : Type u₁
                    D : Type u₂
                    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                    inst✝ : CategoryTheory.Category.{v₂, u₂} D
                    L₁ L₂ : CategoryTheory.Functor C D
                    R₁ R₂ : CategoryTheory.Functor D C
                    adj₁ : CategoryTheory.Adjunction L₁ R₁
                    adj₂ : CategoryTheory.Adjunction L₂ R₂
                    ⊢ Function.RightInverse (fun β => { hom := (CategoryTheory.conjugateEquiv adj₁ …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


/-- When all four functors in a sequare are left adjoints, the mates operation can be iterated:

         L₁                  R₁                  R₁
      C --→ D             C ←-- D             C ←-- D
   F₁ ↓  ↗  ↓  F₂      F₁ ↓  ↘  ↓ F₂       U₁ ↑  ↙  ↑ U₂
      E --→ F             E ←-- F             E ←-- F
         L₂                  R₂                  R₂

In this case the iterated mate equals the conjugate of the original transformation and is thus an
isomorphism if and only if the original transformation is. This explains why some Beck-Chevalley
natural transformations are natural isomorphisms.
-/
theorem iterated_mateEquiv_conjugateEquiv (α : F₁ ⋙ L₂ ⟶ L₁ ⋙ F₂) :
    mateEquiv adj₄ adj₃ (mateEquiv adj₁ adj₂ α) =
      conjugateEquiv (adj₁.comp adj₄) (adj₃.comp adj₂) α := by
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    F₁ : CategoryTheory.Functor A C
    U₁ : CategoryTheory.Functor C A
    F₂ : CategoryTheory.Functor B D
    U₂ : CategoryTheory.Functor D B
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor C D
    R₂ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction F₁ U₁
    adj₄ : CategoryTheory.Adjunction F₂ U₂
    α : Quiver.Hom (F₁.comp L₂) (L₁.comp F₂)
    ⊢ Eq ((CategoryTheory.mateEquiv adj₄ adj₃) ((CategoryTheory.mateEquiv adj₁ adj …
  -/
  ext d
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    F₁ : CategoryTheory.Functor A C
    U₁ : CategoryTheory.Functor C A
    F₂ : CategoryTheory.Functor B D
    U₂ : CategoryTheory.Functor D B
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor C D
    R₂ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction F₁ U₁
    adj₄ : CategoryTheory.Adjunction F₂ U₂
    α : Quiver.Hom (F₁.comp L₂) (L₁.comp F₂)
    d : D
    ⊢ Eq (((CategoryTheory.mateEquiv adj₄ adj₃) ((CategoryTheory.mateEquiv adj₁ ad …
  -/
  unfold conjugateEquiv mateEquiv Adjunction.comp
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    F₁ : CategoryTheory.Functor A C
    U₁ : CategoryTheory.Functor C A
    F₂ : CategoryTheory.Functor B D
    U₂ : CategoryTheory.Functor D B
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor C D
    R₂ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction F₁ U₁
    adj₄ : CategoryTheory.Adjunction F₂ U₂
    α : Quiver.Hom (F₁.comp L₂) (L₁.comp F₂)
    d : D
    ⊢ Eq (({ toFun := fun α => CategoryTheory.CategoryStruct.comp (CategoryTheory. …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem iterated_mateEquiv_conjugateEquiv_symm (α : U₂ ⋙ R₁ ⟶ R₂ ⋙ U₁) :
    (mateEquiv adj₁ adj₂).symm ((mateEquiv adj₄ adj₃).symm α) =
      (conjugateEquiv (adj₁.comp adj₄) (adj₃.comp adj₂)).symm α := by
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    F₁ : CategoryTheory.Functor A C
    U₁ : CategoryTheory.Functor C A
    F₂ : CategoryTheory.Functor B D
    U₂ : CategoryTheory.Functor D B
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor C D
    R₂ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction F₁ U₁
    adj₄ : CategoryTheory.Adjunction F₂ U₂
    α : Quiver.Hom (U₂.comp R₁) (R₂.comp U₁)
    ⊢ Eq ((CategoryTheory.mateEquiv adj₁ adj₂).symm ((CategoryTheory.mateEquiv adj …
  -/
  rw [Equiv.eq_symm_apply, ← iterated_mateEquiv_conjugateEquiv]
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    F₁ : CategoryTheory.Functor A C
    U₁ : CategoryTheory.Functor C A
    F₂ : CategoryTheory.Functor B D
    U₂ : CategoryTheory.Functor D B
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor C D
    R₂ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction F₁ U₁
    adj₄ : CategoryTheory.Adjunction F₂ U₂
    α : Quiver.Hom (U₂.comp R₁) (R₂.comp U₁)
    ⊢ Eq ((CategoryTheory.mateEquiv adj₄ adj₃) ((CategoryTheory.mateEquiv adj₁ adj …
  -/
  simp only [Equiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


/-- Composition of a squares between left adjoints with a conjugate square. -/
def leftAdjointSquareConjugate.vcomp :
    (G ⋙ L₂ ⟶ L₁ ⋙ H) → (L₃ ⟶ L₂) → (G ⋙ L₃ ⟶ L₁ ⋙ H) :=
  fun α β ↦ (whiskerLeft G β) ≫ α


/-- Composition of a squares between right adjoints with a conjugate square. -/
def rightAdjointSquareConjugate.vcomp :
    (R₁ ⋙ G ⟶ H ⋙ R₂) → (R₂ ⟶ R₃) → (R₁ ⋙ G ⟶ H ⋙ R₃) :=
  fun α β ↦ α ≫ (whiskerLeft H β)


/-- The mates equivalence commutes with this composition, essentially by `mateEquiv_vcomp`. -/
theorem mateEquiv_conjugateEquiv_vcomp
    (α : G ⋙ L₂ ⟶ L₁ ⋙ H) (β : L₃ ⟶ L₂) :
    (mateEquiv adj₁ adj₃) (leftAdjointSquareConjugate.vcomp α β) =
      rightAdjointSquareConjugate.vcomp (mateEquiv adj₁ adj₂ α) (conjugateEquiv adj₂ adj₃ β) := by
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    G : CategoryTheory.Functor A C
    H : CategoryTheory.Functor B D
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor C D
    R₂ : CategoryTheory.Functor D C
    L₃ : CategoryTheory.Functor C D
    R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom (G.comp L₂) (L₁.comp H)
    β : Quiver.Hom L₃ L₂
    ⊢ Eq ((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.leftAdjointSquareCo …
  -/
  ext b
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    G : CategoryTheory.Functor A C
    H : CategoryTheory.Functor B D
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor C D
    R₂ : CategoryTheory.Functor D C
    L₃ : CategoryTheory.Functor C D
    R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom (G.comp L₂) (L₁.comp H)
    β : Quiver.Hom L₃ L₂
    b : B
    ⊢ Eq (((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.leftAdjointSquareC …
  -/
  have vcomp := mateEquiv_vcomp adj₁ adj₂ adj₃ α (L₃.leftUnitor.hom ≫ β ≫ L₂.rightUnitor.inv)
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    G : CategoryTheory.Functor A C
    H : CategoryTheory.Functor B D
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor C D
    R₂ : CategoryTheory.Functor D C
    L₃ : CategoryTheory.Functor C D
    R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom (G.comp L₂) (L₁.comp H)
    β : Quiver.Hom L₃ L₂
    b : B
    vcomp : Eq ((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.leftAdjointSq …
    ⊢ Eq (((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.leftAdjointSquareC …
  -/
  unfold leftAdjointSquare.vcomp rightAdjointSquare.vcomp at vcomp
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    G : CategoryTheory.Functor A C
    H : CategoryTheory.Functor B D
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor C D
    R₂ : CategoryTheory.Functor D C
    L₃ : CategoryTheory.Functor C D
    R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom (G.comp L₂) (L₁.comp H)
    β : Quiver.Hom L₃ L₂
    b : B
    vcomp : Eq ((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.CategoryStruc …
    ⊢ Eq (((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.leftAdjointSquareC …
  -/
  unfold leftAdjointSquareConjugate.vcomp rightAdjointSquareConjugate.vcomp conjugateEquiv
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    G : CategoryTheory.Functor A C
    H : CategoryTheory.Functor B D
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor C D
    R₂ : CategoryTheory.Functor D C
    L₃ : CategoryTheory.Functor C D
    R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom (G.comp L₂) (L₁.comp H)
    β : Quiver.Hom L₃ L₂
    b : B
    vcomp : Eq ((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.CategoryStruc …
    ⊢ Eq (((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.CategoryStruct.com …
  -/
  have vcompb := congr_app vcomp b
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    G : CategoryTheory.Functor A C
    H : CategoryTheory.Functor B D
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor C D
    R₂ : CategoryTheory.Functor D C
    L₃ : CategoryTheory.Functor C D
    R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom (G.comp L₂) (L₁.comp H)
    β : Quiver.Hom L₃ L₂
    b : B
    vcomp : Eq ((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.CategoryStruc …
    vcompb : Eq (((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.CategoryStr …
    ⊢ Eq (((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.CategoryStruct.com …
  -/
  simp at vcompb
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    G : CategoryTheory.Functor A C
    H : CategoryTheory.Functor B D
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor C D
    R₂ : CategoryTheory.Functor D C
    L₃ : CategoryTheory.Functor C D
    R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom (G.comp L₂) (L₁.comp H)
    β : Quiver.Hom L₃ L₂
    b : B
    vcomp : Eq ((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.CategoryStruc …
    vcompb : Eq (CategoryTheory.CategoryStruct.comp (adj₃.unit.app (G.obj (R₁.obj  …
    ⊢ Eq (((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.CategoryStruct.com …
  -/
  unfold mateEquiv
  simp only [comp_obj, Equiv.coe_fn_mk, whiskerLeft_comp, whiskerLeft_twice, whiskerRight_comp,
    assoc, comp_app, whiskerLeft_app, whiskerRight_app, id_obj, Functor.comp_map, Iso.homCongr_symm,
    Equiv.instTrans_trans, Equiv.trans_apply, Iso.homCongr_apply, Iso.symm_inv, Iso.symm_hom,
    rightUnitor_inv_app, leftUnitor_hom_app, map_id, Functor.id_map, comp_id, id_comp]
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    G : CategoryTheory.Functor A C
    H : CategoryTheory.Functor B D
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor C D
    R₂ : CategoryTheory.Functor D C
    L₃ : CategoryTheory.Functor C D
    R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom (G.comp L₂) (L₁.comp H)
    β : Quiver.Hom L₃ L₂
    b : B
    vcomp : Eq ((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.CategoryStruc …
    vcompb : Eq (CategoryTheory.CategoryStruct.comp (adj₃.unit.app (G.obj (R₁.obj  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₃.unit.app (G.obj (R₁.obj b))) (C …
  -/
  exact vcompb
  /-
    🎉 no goals
  -/


/-- Composition of a conjugate square with a squares between left adjoints. -/
def leftAdjointConjugateSquare.vcomp :
    (L₂ ⟶ L₁) → (G ⋙ L₃ ⟶ L₂ ⋙ H) → (G ⋙ L₃ ⟶ L₁ ⋙ H) :=
  fun α β ↦ β ≫ (whiskerRight α H)


/-- Composition of a conjugate square with a squares between right adjoints. -/
def rightAdjointConjugateSquare.vcomp :
    (R₁ ⟶ R₂) → (R₂ ⋙ G ⟶ H ⋙ R₃) → (R₁ ⋙ G ⟶ H ⋙ R₃) :=
  fun α β ↦ (whiskerRight α G) ≫ β


/-- The mates equivalence commutes with this composition, essentially by `mateEquiv_vcomp`. -/
theorem conjugateEquiv_mateEquiv_vcomp
    (α : L₂ ⟶ L₁) (β : G ⋙ L₃ ⟶ L₂ ⋙ H) :
    (mateEquiv adj₁ adj₃) (leftAdjointConjugateSquare.vcomp α β) =
      rightAdjointConjugateSquare.vcomp (conjugateEquiv adj₁ adj₂ α) (mateEquiv adj₂ adj₃ β) := by
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    G : CategoryTheory.Functor A C
    H : CategoryTheory.Functor B D
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor A B
    R₂ : CategoryTheory.Functor B A
    L₃ : CategoryTheory.Functor C D
    R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom L₂ L₁
    β : Quiver.Hom (G.comp L₃) (L₂.comp H)
    ⊢ Eq ((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.leftAdjointConjugat …
  -/
  ext b
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    G : CategoryTheory.Functor A C
    H : CategoryTheory.Functor B D
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor A B
    R₂ : CategoryTheory.Functor B A
    L₃ : CategoryTheory.Functor C D
    R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom L₂ L₁
    β : Quiver.Hom (G.comp L₃) (L₂.comp H)
    b : B
    ⊢ Eq (((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.leftAdjointConjuga …
  -/
  have vcomp := mateEquiv_vcomp adj₁ adj₂ adj₃ (L₂.leftUnitor.hom ≫ α ≫ L₁.rightUnitor.inv) β
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    G : CategoryTheory.Functor A C
    H : CategoryTheory.Functor B D
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor A B
    R₂ : CategoryTheory.Functor B A
    L₃ : CategoryTheory.Functor C D
    R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom L₂ L₁
    β : Quiver.Hom (G.comp L₃) (L₂.comp H)
    b : B
    vcomp : Eq ((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.leftAdjointSq …
    ⊢ Eq (((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.leftAdjointConjuga …
  -/
  unfold leftAdjointSquare.vcomp rightAdjointSquare.vcomp at vcomp
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    G : CategoryTheory.Functor A C
    H : CategoryTheory.Functor B D
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor A B
    R₂ : CategoryTheory.Functor B A
    L₃ : CategoryTheory.Functor C D
    R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom L₂ L₁
    β : Quiver.Hom (G.comp L₃) (L₂.comp H)
    b : B
    vcomp : Eq ((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.CategoryStruc …
    ⊢ Eq (((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.leftAdjointConjuga …
  -/
  unfold leftAdjointConjugateSquare.vcomp rightAdjointConjugateSquare.vcomp conjugateEquiv
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    G : CategoryTheory.Functor A C
    H : CategoryTheory.Functor B D
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor A B
    R₂ : CategoryTheory.Functor B A
    L₃ : CategoryTheory.Functor C D
    R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom L₂ L₁
    β : Quiver.Hom (G.comp L₃) (L₂.comp H)
    b : B
    vcomp : Eq ((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.CategoryStruc …
    ⊢ Eq (((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.CategoryStruct.com …
  -/
  have vcompb := congr_app vcomp b
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    G : CategoryTheory.Functor A C
    H : CategoryTheory.Functor B D
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor A B
    R₂ : CategoryTheory.Functor B A
    L₃ : CategoryTheory.Functor C D
    R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom L₂ L₁
    β : Quiver.Hom (G.comp L₃) (L₂.comp H)
    b : B
    vcomp : Eq ((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.CategoryStruc …
    vcompb : Eq (((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.CategoryStr …
    ⊢ Eq (((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.CategoryStruct.com …
  -/
  simp at vcompb
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    G : CategoryTheory.Functor A C
    H : CategoryTheory.Functor B D
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor A B
    R₂ : CategoryTheory.Functor B A
    L₃ : CategoryTheory.Functor C D
    R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom L₂ L₁
    β : Quiver.Hom (G.comp L₃) (L₂.comp H)
    b : B
    vcomp : Eq ((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.CategoryStruc …
    vcompb : Eq (CategoryTheory.CategoryStruct.comp (adj₃.unit.app (G.obj (R₁.obj  …
    ⊢ Eq (((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.CategoryStruct.com …
  -/
  unfold mateEquiv
  simp only [comp_obj, Equiv.coe_fn_mk, whiskerLeft_comp, whiskerLeft_twice, whiskerRight_comp,
    assoc, comp_app, whiskerLeft_app, whiskerRight_app, id_obj, Functor.comp_map, Iso.homCongr_symm,
    Equiv.instTrans_trans, Equiv.trans_apply, Iso.homCongr_apply, Iso.symm_inv, Iso.symm_hom,
    rightUnitor_inv_app, leftUnitor_hom_app, map_id, Functor.id_map, comp_id, id_comp]
  /-
    case w.h
    A : Type u₁
    B : Type u₂
    C : Type u₃
    D : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    G : CategoryTheory.Functor A C
    H : CategoryTheory.Functor B D
    L₁ : CategoryTheory.Functor A B
    R₁ : CategoryTheory.Functor B A
    L₂ : CategoryTheory.Functor A B
    R₂ : CategoryTheory.Functor B A
    L₃ : CategoryTheory.Functor C D
    R₃ : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction L₁ R₁
    adj₂ : CategoryTheory.Adjunction L₂ R₂
    adj₃ : CategoryTheory.Adjunction L₃ R₃
    α : Quiver.Hom L₂ L₁
    β : Quiver.Hom (G.comp L₃) (L₂.comp H)
    b : B
    vcomp : Eq ((CategoryTheory.mateEquiv adj₁ adj₃) (CategoryTheory.CategoryStruc …
    vcompb : Eq (CategoryTheory.CategoryStruct.comp (adj₃.unit.app (G.obj (R₁.obj  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₃.unit.app (G.obj (R₁.obj b))) (C …
  -/
  exact vcompb
  /-
    🎉 no goals
  -/



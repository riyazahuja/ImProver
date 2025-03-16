/-- The equivalence of categories `(C₁ ⥤ C₂ ⥤ C₃ ⥤ E) ≌ C₁ × C₂ × C₃ ⥤ E`
given by the curryfication of functors in three variables. -/
def currying₃ : (C₁ ⥤ C₂ ⥤ C₃ ⥤ E) ≌ C₁ × C₂ × C₃ ⥤ E :=
  currying.trans (currying.trans (prod.associativity C₁ C₂ C₃).congrLeft)


/-- Uncurrying a functor in three variables. -/
abbrev uncurry₃ : (C₁ ⥤ C₂ ⥤ C₃ ⥤ E) ⥤ C₁ × C₂ × C₃ ⥤ E := currying₃.functor


/-- Currying a functor in three variables. -/
abbrev curry₃ : (C₁ × C₂ × C₃ ⥤ E) ⥤ C₁ ⥤ C₂ ⥤ C₃ ⥤ E := currying₃.inverse


/-- Uncurrying functors in three variables gives a fully faithful functor. -/
def fullyFaithfulUncurry₃ :
    (uncurry₃ : (C₁ ⥤ C₂ ⥤ C₃ ⥤ E) ⥤ (C₁ × C₂ × C₃ ⥤ E)).FullyFaithful :=
  currying₃.fullyFaithfulFunctor


@[simp]
lemma curry₃_obj_map_app_app (F : C₁ × C₂ × C₃ ⥤ E)
    {X₁ Y₁ : C₁} (f : X₁ ⟶ Y₁) (X₂ : C₂) (X₃ : C₃) :
    (((curry₃.obj F).map f).app X₂).app X₃ = F.map ⟨f, 𝟙 X₂, 𝟙 X₃⟩ := rfl


@[simp]
lemma curry₃_obj_obj_map_app (F : C₁ × C₂ × C₃ ⥤ E)
    (X₁ : C₁) {X₂ Y₂ : C₂} (f : X₂ ⟶ Y₂) (X₃ : C₃) :
    (((curry₃.obj F).obj X₁).map f).app X₃ = F.map ⟨𝟙 X₁, f, 𝟙 X₃⟩ := rfl


@[simp]
lemma curry₃_obj_obj_obj_map (F : C₁ × C₂ × C₃ ⥤ E)
    (X₁ : C₁) (X₂ : C₂) {X₃ Y₃ : C₃} (f : X₃ ⟶ Y₃) :
    (((curry₃.obj F).obj X₁).obj X₂).map f = F.map ⟨𝟙 X₁, 𝟙 X₂, f⟩ := rfl


@[simp]
lemma curry₃_map_app_app_app {F G : C₁ × C₂ × C₃ ⥤ E} (f : F ⟶ G)
    (X₁ : C₁) (X₂ : C₂) (X₃ : C₃) :
    (((curry₃.map f).app X₁).app X₂).app X₃ = f.app ⟨X₁, X₂, X₃⟩ := rfl


@[simp]
lemma currying₃_unitIso_hom_app_app_app_app (F : C₁ ⥤ C₂ ⥤ C₃ ⥤ E)
    (X₁ : C₁) (X₂ : C₂) (X₃ : C₃) :
    (((currying₃.unitIso.hom.app F).app X₁).app X₂).app X₃ = 𝟙 _ := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_4
    E : Type u_9
    inst✝³ : CategoryTheory.Category.{u_10, u_1} C₁
    inst✝² : CategoryTheory.Category.{u_13, u_2} C₂
    inst✝¹ : CategoryTheory.Category.{u_12, u_4} C₃
    inst✝ : CategoryTheory.Category.{u_11, u_9} E
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
    X₁ : C₁
    X₂ : C₂
    X₃ : C₃
    ⊢ Eq ((((CategoryTheory.currying₃.unitIso.hom.app F).app X₁).app X₂).app X₃) ( …
  -/
  simp [currying₃, Equivalence.unit]
  /-
    🎉 no goals
  -/


@[simp]
lemma currying₃_unitIso_inv_app_app_app_app (F : C₁ ⥤ C₂ ⥤ C₃ ⥤ E)
    (X₁ : C₁) (X₂ : C₂) (X₃ : C₃) :
    (((currying₃.unitIso.inv.app F).app X₁).app X₂).app X₃ = 𝟙 _ := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_4
    E : Type u_9
    inst✝³ : CategoryTheory.Category.{u_10, u_1} C₁
    inst✝² : CategoryTheory.Category.{u_13, u_2} C₂
    inst✝¹ : CategoryTheory.Category.{u_12, u_4} C₃
    inst✝ : CategoryTheory.Category.{u_11, u_9} E
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ (CategoryTheory.Funct …
    X₁ : C₁
    X₂ : C₂
    X₃ : C₃
    ⊢ Eq ((((CategoryTheory.currying₃.unitIso.inv.app F).app X₁).app X₂).app X₃) ( …
  -/
  simp [currying₃, Equivalence.unitInv]
  /-
    🎉 no goals
  -/


/-- Given functors `F₁ : C₁ ⥤ D₁`, `F₂ : C₂ ⥤ D₂`, `F₃ : C₃ ⥤ D₃`
and `G : D₁ × D₂ × D₃ ⥤ E`, this is the isomorphism between
`curry₃.obj (F₁.prod (F₂.prod F₃) ⋙ G) : C₁ ⥤ C₂ ⥤ C₃ ⥤ E`
and `F₁ ⋙ curry₃.obj G ⋙ ((whiskeringLeft₂ E).obj F₂).obj F₃`. -/
@[simps!]
def curry₃ObjProdComp (F₁ : C₁ ⥤ D₁) (F₂ : C₂ ⥤ D₂) (F₃ : C₃ ⥤ D₃) (G : D₁ × D₂ × D₃ ⥤ E) :
    curry₃.obj (F₁.prod (F₂.prod F₃) ⋙ G) ≅
      F₁ ⋙ curry₃.obj G ⋙ ((whiskeringLeft₂ E).obj F₂).obj F₃ :=
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_4
    C₂₃ : Type u_5
    D₁ : Type u_6
    D₂ : Type u_7
    D₃ : Type u_8
    E : Type u_9
    inst✝⁸ : CategoryTheory.Category.{?u.38608, u_1} C₁
    inst✝⁷ : CategoryTheory.Category.{?u.38612, u_2} C₂
    inst✝⁶ : CategoryTheory.Category.{?u.38616, u_4} C₃
    inst✝⁵ : CategoryTheory.Category.{?u.38620, u_3} C₁₂
    inst✝⁴ : CategoryTheory.Category.{?u.38624, u_5} C₂₃
    inst✝³ : CategoryTheory.Category.{?u.38628, u_6} D₁
    inst✝² : CategoryTheory.Category.{?u.38632, u_7} D₂
    inst✝¹ : CategoryTheory.Category.{?u.38636, u_8} D₃
    inst✝ : CategoryTheory.Category.{?u.38640, u_9} E
    F₁ : CategoryTheory.Functor C₁ D₁
    F₂ : CategoryTheory.Functor C₂ D₂
    F₃ : CategoryTheory.Functor C₃ D₃
    G : CategoryTheory.Functor (Prod D₁ (Prod D₂ D₃)) E
    ⊢ ∀ {X Y : C₁} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp (( …
  -/
              /-
                C₁ : Type u_1
                C₂ : Type u_2
                C₁₂ : Type u_3
                C₃ : Type u_4
                C₂₃ : Type u_5
                D₁ : Type u_6
                D₂ : Type u_7
                D₃ : Type u_8
                E : Type u_9
                inst✝⁸ : CategoryTheory.Category.{?u.38608, u_1} C₁
                inst✝⁷ : CategoryTheory.Category.{?u.38612, u_2} C₂
                inst✝⁶ : CategoryTheory.Category.{?u.38616, u_4} C₃
                inst✝⁵ : CategoryTheory.Category.{?u.38620, u_3} C₁₂
                inst✝⁴ : CategoryTheory.Category.{?u.38624, u_5} C₂₃
                inst✝³ : CategoryTheory.Category.{?u.38628, u_6} D₁
                inst✝² : CategoryTheory.Category.{?u.38632, u_7} D₂
                inst✝¹ : CategoryTheory.Category.{?u.38636, u_8} D₃
                inst✝ : CategoryTheory.Category.{?u.38640, u_9} E
                F₁ : CategoryTheory.Functor C₁ D₁
                F₂ : CategoryTheory.Functor C₂ D₂
                F₃ : CategoryTheory.Functor C₃ D₃
                G : CategoryTheory.Functor (Prod D₁ (Prod D₂ D₃)) E
                X₁ : C₁
                ⊢ ∀ {X Y : C₂} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp (( …
              -/
                /-
                  C₁ : Type u_1
                  C₂ : Type u_2
                  C₁₂ : Type u_3
                  C₃ : Type u_4
                  C₂₃ : Type u_5
                  D₁ : Type u_6
                  D₂ : Type u_7
                  D₃ : Type u_8
                  E : Type u_9
                  inst✝⁸ : CategoryTheory.Category.{?u.38608, u_1} C₁
                  inst✝⁷ : CategoryTheory.Category.{?u.38612, u_2} C₂
                  inst✝⁶ : CategoryTheory.Category.{?u.38616, u_4} C₃
                  inst✝⁵ : CategoryTheory.Category.{?u.38620, u_3} C₁₂
                  inst✝⁴ : CategoryTheory.Category.{?u.38624, u_5} C₂₃
                  inst✝³ : CategoryTheory.Category.{?u.38628, u_6} D₁
                  inst✝² : CategoryTheory.Category.{?u.38632, u_7} D₂
                  inst✝¹ : CategoryTheory.Category.{?u.38636, u_8} D₃
                  inst✝ : CategoryTheory.Category.{?u.38640, u_9} E
                  F₁ : CategoryTheory.Functor C₁ D₁
                  F₂ : CategoryTheory.Functor C₂ D₂
                  F₃ : CategoryTheory.Functor C₃ D₃
                  G : CategoryTheory.Functor (Prod D₁ (Prod D₂ D₃)) E
                  X₁ : C₁
                  X₂ : C₂
                  ⊢ ∀ {X Y : C₃} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp (( …
                -/
  NatIso.ofComponents
                /-
                  🎉 no goals
                -/
              /-
                🎉 no goals
              -/
  /-
    🎉 no goals
  -/
    (fun X₁ ↦ NatIso.ofComponents
      (fun X₂ ↦ NatIso.ofComponents (fun X₃ ↦ Iso.refl _)))


/-- `bifunctorComp₁₂` can be described in terms of the curryfication of functors. -/
@[simps!]
def bifunctorComp₁₂Iso (F₁₂ : C₁ ⥤ C₂ ⥤ C₁₂) (G : C₁₂ ⥤ C₃ ⥤ E) :
    bifunctorComp₁₂ F₁₂ G ≅ curry.obj (uncurry.obj F₁₂ ⋙ G) :=
                                /-
                                  C₁ : Type u_1
                                  C₂ : Type u_2
                                  C₁₂ : Type u_3
                                  C₃ : Type u_4
                                  C₂₃ : Type u_5
                                  D₁ : Type u_6
                                  D₂ : Type u_7
                                  D₃ : Type u_8
                                  E : Type u_9
                                  inst✝⁸ : CategoryTheory.Category.{?u.58256, u_1} C₁
                                  inst✝⁷ : CategoryTheory.Category.{?u.58260, u_2} C₂
                                  inst✝⁶ : CategoryTheory.Category.{?u.58264, u_4} C₃
                                  inst✝⁵ : CategoryTheory.Category.{?u.58268, u_3} C₁₂
                                  inst✝⁴ : CategoryTheory.Category.{?u.58272, u_5} C₂₃
                                  inst✝³ : CategoryTheory.Category.{?u.58276, u_6} D₁
                                  inst✝² : CategoryTheory.Category.{?u.58280, u_7} D₂
                                  inst✝¹ : CategoryTheory.Category.{?u.58284, u_8} D₃
                                  inst✝ : CategoryTheory.Category.{?u.58288, u_9} E
                                  F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
                                  G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ E)
                                  x✝ : C₁
                                  ⊢ ∀ {X Y : C₂} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp (( …
                                -/
                                /-
                                  🎉 no goals
                                -/
  NatIso.ofComponents (fun _ => NatIso.ofComponents (fun _ => Iso.refl _))
  /-
    🎉 no goals
  -/


/-- `bifunctorComp₂₃` can be described in terms of the curryfication of functors. -/
@[simps!]
def bifunctorComp₂₃Iso (F : C₁ ⥤ C₂₃ ⥤ E) (G₂₃ : C₂ ⥤ C₃ ⥤ C₂₃) :
    bifunctorComp₂₃ F G₂₃ ≅
    curry.obj (curry.obj (prod.associator _ _ _ ⋙
      uncurry.obj (uncurry.obj G₂₃ ⋙ F.flip).flip)) :=
                               /-
                                 C₁ : Type u_1
                                 C₂ : Type u_2
                                 C₁₂ : Type u_3
                                 C₃ : Type u_4
                                 C₂₃ : Type u_5
                                 D₁ : Type u_6
                                 D₂ : Type u_7
                                 D₃ : Type u_8
                                 E : Type u_9
                                 inst✝⁸ : CategoryTheory.Category.{?u.66843, u_1} C₁
                                 inst✝⁷ : CategoryTheory.Category.{?u.66847, u_2} C₂
                                 inst✝⁶ : CategoryTheory.Category.{?u.66851, u_4} C₃
                                 inst✝⁵ : CategoryTheory.Category.{?u.66855, u_3} C₁₂
                                 inst✝⁴ : CategoryTheory.Category.{?u.66859, u_5} C₂₃
                                 inst✝³ : CategoryTheory.Category.{?u.66863, u_6} D₁
                                 inst✝² : CategoryTheory.Category.{?u.66867, u_7} D₂
                                 inst✝¹ : CategoryTheory.Category.{?u.66871, u_8} D₃
                                 inst✝ : CategoryTheory.Category.{?u.66875, u_9} E
                                 F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ E)
                                 G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
                                 x✝ : C₁
                                 ⊢ ∀ {X Y : C₂} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp (( …
                               -/
    /-
      C₁ : Type u_1
      C₂ : Type u_2
      C₁₂ : Type u_3
      C₃ : Type u_4
      C₂₃ : Type u_5
      D₁ : Type u_6
      D₂ : Type u_7
      D₃ : Type u_8
      E : Type u_9
      inst✝⁸ : CategoryTheory.Category.{?u.66843, u_1} C₁
      inst✝⁷ : CategoryTheory.Category.{?u.66847, u_2} C₂
      inst✝⁶ : CategoryTheory.Category.{?u.66851, u_4} C₃
      inst✝⁵ : CategoryTheory.Category.{?u.66855, u_3} C₁₂
      inst✝⁴ : CategoryTheory.Category.{?u.66859, u_5} C₂₃
      inst✝³ : CategoryTheory.Category.{?u.66863, u_6} D₁
      inst✝² : CategoryTheory.Category.{?u.66867, u_7} D₂
      inst✝¹ : CategoryTheory.Category.{?u.66871, u_8} D₃
      inst✝ : CategoryTheory.Category.{?u.66875, u_9} E
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ E)
      G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
      x✝¹ : C₁
      x✝ : C₂
      ⊢ ∀ {X Y : C₃} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp (( …
    -/
                               /-
                                 🎉 no goals
                               -/
    /-
      🎉 no goals
    -/
  NatIso.ofComponents (fun _ ↦ NatIso.ofComponents (fun _ ↦
  /-
    🎉 no goals
  -/
    NatIso.ofComponents (fun _ ↦ Iso.refl _)))



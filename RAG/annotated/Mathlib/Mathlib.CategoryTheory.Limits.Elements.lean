/-- (implementation) A system `(Fi, fi)_i` of elements induces an element in `lim_i A(Fi)`. -/
noncomputable def liftedConeElement' : limit ((F ⋙ π A) ⋙ A) :=
                                              /-
                                                C : Type u
                                                inst✝² : CategoryTheory.Category.{v, u} C
                                                A : CategoryTheory.Functor C (Type w)
                                                I : Type u₁
                                                inst✝¹ : CategoryTheory.Category.{v₁, u₁} I
                                                inst✝ : Small.{w, u₁} I
                                                F : CategoryTheory.Functor I A.Elements
                                                ⊢ ∀ (j j' : I) (f : Quiver.Hom j j'), Eq (((F.comp (CategoryTheory.CategoryOfE …
                                              -/
  Types.Limit.mk _ (fun i => (F.obj i).2) (by simp)
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
lemma π_liftedConeElement' (i : I) :
    limit.π ((F ⋙ π A) ⋙ A) i (liftedConeElement' F) = (F.obj i).2 :=
  Types.Limit.π_mk _ _ _ _


/-- (implementation) A system `(Fi, fi)_i` of elements induces an element in `A(lim_i Fi)`. -/
noncomputable def liftedConeElement : A.obj (limit (F ⋙ π A)) :=
  (preservesLimitIso A (F ⋙ π A)).inv (liftedConeElement' F)


@[simp]
lemma map_lift_mapCone (c : Cone F) :
    A.map (limit.lift (F ⋙ π A) ((π A).mapCone c)) c.pt.snd = liftedConeElement F := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor C (Type w)
    I : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} I
    inst✝² : Small.{w, u₁} I
    F : CategoryTheory.Functor I A.Elements
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape I C
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape I A
    c : CategoryTheory.Limits.Cone F
    ⊢ Eq (A.map (CategoryTheory.Limits.limit.lift (F.comp (CategoryTheory.Category …
  -/
  apply (preservesLimitIso A (F ⋙ π A)).toEquiv.injective
  /-
    case a
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor C (Type w)
    I : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} I
    inst✝² : Small.{w, u₁} I
    F : CategoryTheory.Functor I A.Elements
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape I C
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape I A
    c : CategoryTheory.Limits.Cone F
    ⊢ Eq ((CategoryTheory.preservesLimitIso A (F.comp (CategoryTheory.CategoryOfEl …
  -/
  ext i
  have h₁ := congrFun (preservesLimitIso_hom_π A (F ⋙ π A) i)
    (A.map (limit.lift (F ⋙ π A) ((π A).mapCone c)) c.pt.snd)
  /-
    case a.w
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor C (Type w)
    I : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} I
    inst✝² : Small.{w, u₁} I
    F : CategoryTheory.Functor I A.Elements
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape I C
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape I A
    c : CategoryTheory.Limits.Cone F
    i : I
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.preservesLimitIso  …
    ⊢ Eq (CategoryTheory.Limits.limit.π ((F.comp (CategoryTheory.CategoryOfElement …
  -/
  have h₂ := (c.π.app i).property
  /-
    case a.w
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor C (Type w)
    I : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} I
    inst✝² : Small.{w, u₁} I
    F : CategoryTheory.Functor I A.Elements
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape I C
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape I A
    c : CategoryTheory.Limits.Cone F
    i : I
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.preservesLimitIso  …
    h₂ : Eq (A.map (↑(c.π.app i)) (((CategoryTheory.Functor.const I).obj c.pt).obj …
    ⊢ Eq (CategoryTheory.Limits.limit.π ((F.comp (CategoryTheory.CategoryOfElement …
  -/
  simp_all [← FunctorToTypes.map_comp_apply, liftedConeElement]
  /-
    🎉 no goals
  -/


@[simp]
lemma map_π_liftedConeElement (i : I) :
    A.map (limit.π (F ⋙ π A) i) (liftedConeElement F) = (F.obj i).snd := by
  have := congrFun
    (preservesLimitIso_inv_π A (F ⋙ π A) i) (liftedConeElement' F)
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    A : CategoryTheory.Functor C (Type w)
    I : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} I
    inst✝² : Small.{w, u₁} I
    F : CategoryTheory.Functor I A.Elements
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape I C
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape I A
    i : I
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.preservesLimitIs …
    ⊢ Eq (A.map (CategoryTheory.Limits.limit.π (F.comp (CategoryTheory.CategoryOfE …
  -/
  simp_all [liftedConeElement]
  /-
    🎉 no goals
  -/


/-- (implementation) The constructured limit cone. -/
@[simps]
noncomputable def liftedCone : Cone F where
  pt := ⟨_, liftedConeElement F⟩
  π :=
                                               /-
                                                 C : Type u
                                                 inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                 A : CategoryTheory.Functor C (Type w)
                                                 I : Type u₁
                                                 inst✝³ : CategoryTheory.Category.{v₁, u₁} I
                                                 inst✝² : Small.{w, u₁} I
                                                 F : CategoryTheory.Functor I A.Elements
                                                 inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape I C
                                                 inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape I A
                                                 i : I
                                                 ⊢ Eq (A.map (CategoryTheory.Limits.limit.π (F.comp (CategoryTheory.CategoryOfE …
                                               -/
    { app := fun i => ⟨limit.π (F ⋙ π A) i, by simp⟩
                                               /-
                                                 🎉 no goals
                                               -/
                                     /-
                                       C : Type u
                                       inst✝⁴ : CategoryTheory.Category.{v, u} C
                                       A : CategoryTheory.Functor C (Type w)
                                       I : Type u₁
                                       inst✝³ : CategoryTheory.Category.{v₁, u₁} I
                                       inst✝² : Small.{w, u₁} I
                                       F : CategoryTheory.Functor I A.Elements
                                       inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape I C
                                       inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape I A
                                       i i' : I
                                       f : Quiver.Hom i i'
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const I).ob …
                                     -/
      naturality := fun i i' f => by ext; simpa using (limit.w _ _).symm }
                                          /-
                                            🎉 no goals
                                          -/


/-- (implementation) The constructed limit cone is a lift of the limit cone in `C`. -/
noncomputable def isValidLift : (π A).mapCone (liftedCone F) ≅ limit.cone (F ⋙ π A) :=
  Iso.refl _


/-- (implementation) The constuctured limit cone is a limit cone. -/
noncomputable def isLimit : IsLimit (liftedCone F) where
                                                        /-
                                                          C : Type u
                                                          inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                          A : CategoryTheory.Functor C (Type w)
                                                          I : Type u₁
                                                          inst✝³ : CategoryTheory.Category.{v₁, u₁} I
                                                          inst✝² : Small.{w, u₁} I
                                                          F : CategoryTheory.Functor I A.Elements
                                                          inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape I C
                                                          inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape I A
                                                          s : CategoryTheory.Limits.Cone F
                                                          ⊢ Eq (A.map (CategoryTheory.Limits.limit.lift (F.comp (CategoryTheory.Category …
                                                        -/
  lift s := ⟨limit.lift (F ⋙ π A) ((π A).mapCone s), by simp⟩
                                                        /-
                                                          🎉 no goals
                                                        -/
  uniq s m h := ext _ _ _ <| limit.hom_ext
                /-
                  C : Type u
                  inst✝⁴ : CategoryTheory.Category.{v, u} C
                  A : CategoryTheory.Functor C (Type w)
                  I : Type u₁
                  inst✝³ : CategoryTheory.Category.{v₁, u₁} I
                  inst✝² : Small.{w, u₁} I
                  F : CategoryTheory.Functor I A.Elements
                  inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape I C
                  inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape I A
                  s : CategoryTheory.Limits.Cone F
                  m : Quiver.Hom s.pt (CategoryTheory.CategoryOfElements.CreatesLimitsAux.lifted …
                  h : ∀ (j : I), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Categ …
                  i : I
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (↑m) (CategoryTheory.Limits.limit.π ( …
                -/
    fun i => by simpa using congrArg Subtype.val (h i)
                /-
                  🎉 no goals
                -/


noncomputable instance (F : I ⥤ A.Elements) : CreatesLimit F (π A) :=
  createsLimitOfReflectsIso' (limit.isLimit _) ⟨⟨liftedCone F, isValidLift F⟩, isLimit F⟩


noncomputable instance : CreatesLimitsOfShape I (π A) where


instance : HasLimitsOfShape I A.Elements :=
  hasLimitsOfShape_of_hasLimitsOfShape_createsLimitsOfShape (π A)



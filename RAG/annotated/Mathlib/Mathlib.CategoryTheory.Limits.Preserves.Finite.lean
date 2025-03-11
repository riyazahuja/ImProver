/-- A functor is said to preserve finite limits, if it preserves all limits of shape `J`,
where `J : Type` is a finite category.
-/
class PreservesFiniteLimits (F : C ⥤ D) : Prop where
  preservesFiniteLimits :
    ∀ (J : Type) [SmallCategory J] [FinCategory J], PreservesLimitsOfShape J F := by infer_instance


instance (F : C ⥤ D) : Subsingleton (PreservesFiniteLimits F) :=
                     /-
                       C : Type u₁
                       inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                       D : Type u₂
                       inst✝² : CategoryTheory.Category.{v₂, u₂} D
                       E : Type u₃
                       inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
                       J : Type w
                       inst✝ : CategoryTheory.SmallCategory J
                       K : CategoryTheory.Functor J C
                       F : CategoryTheory.Functor C D
                       x✝¹ x✝ : CategoryTheory.Limits.PreservesFiniteLimits F
                       a b : ∀ (J : Type) [inst : CategoryTheory.SmallCategory J] [inst_1 : CategoryT …
                       ⊢ Eq ⋯ ⋯
                     -/
  ⟨fun ⟨a⟩ ⟨b⟩ => by congr⟩
                     /-
                       🎉 no goals
                     -/


/-- Preserving finite limits also implies preserving limits over finite shapes in higher universes,
though through a noncomputable instance. -/
instance (priority := 100) preservesLimitsOfShapeOfPreservesFiniteLimits (F : C ⥤ D)
    [PreservesFiniteLimits F] (J : Type w) [SmallCategory J] [FinCategory J] :
    PreservesLimitsOfShape J F := by
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
    J✝ : Type w
    inst✝³ : CategoryTheory.SmallCategory J✝
    K : CategoryTheory.Functor J✝ C
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.Limits.PreservesFiniteLimits F
    J : Type w
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.FinCategory J
    ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J F
  -/
  apply preservesLimitsOfShape_of_equiv (FinCategory.equivAsType J)
  /-
    🎉 no goals
  -/

-- This is a dangerous instance as it has unbound universe variables.

/-- If we preserve limits of some arbitrary size, then we preserve all finite limits. -/
lemma PreservesLimitsOfSize.preservesFiniteLimits (F : C ⥤ D)
    [PreservesLimitsOfSize.{w, w₂} F] : PreservesFiniteLimits F where
  preservesFiniteLimits J (sJ : SmallCategory J) fJ := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.Limits.PreservesLimitsOfSize.{w, w₂, v₁, v₂, u₁, u₂} F
      J : Type
      sJ : CategoryTheory.SmallCategory J
      fJ : CategoryTheory.FinCategory J
      ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J F
    -/
    haveI := preservesSmallestLimits_of_preservesLimits F
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.Limits.PreservesLimitsOfSize.{w, w₂, v₁, v₂, u₁, u₂} F
      J : Type
      sJ : CategoryTheory.SmallCategory J
      fJ : CategoryTheory.FinCategory J
      this : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, v₁, v₂, u₁, u₂} F
      ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J F
    -/
    exact preservesLimitsOfShape_of_equiv (FinCategory.equivAsType J) F
    /-
      🎉 no goals
    -/

-- Added as a specialization of the dangerous instance above, for limits indexed in Type 0.

instance (priority := 120) PreservesLimitsOfSize0.preservesFiniteLimits
    (F : C ⥤ D) [PreservesLimitsOfSize.{0, 0} F] : PreservesFiniteLimits F :=
  PreservesLimitsOfSize.preservesFiniteLimits F

-- An alternative specialization of the dangerous instance for small limits.

instance (priority := 120) PreservesLimits.preservesFiniteLimits (F : C ⥤ D)
    [PreservesLimits F] : PreservesFiniteLimits F :=
  PreservesLimitsOfSize.preservesFiniteLimits F


/-- We can always derive `PreservesFiniteLimits C` by showing that we are preserving limits at an
arbitrary universe. -/
lemma preservesFiniteLimits_of_preservesFiniteLimitsOfSize (F : C ⥤ D)
    (h :
      ∀ (J : Type w) {𝒥 : SmallCategory J} (_ : @FinCategory J 𝒥), PreservesLimitsOfShape J F) :
    PreservesFiniteLimits F where
      preservesFiniteLimits J (_ : SmallCategory J) _ := by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          h : ∀ (J : Type w) {𝒥 : CategoryTheory.SmallCategory J}, CategoryTheory.FinCat …
          J : Type
          x✝¹ : CategoryTheory.SmallCategory J
          x✝ : CategoryTheory.FinCategory J
          ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J F
        -/
        letI : Category (ULiftHom (ULift J)) := ULiftHom.category
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          h : ∀ (J : Type w) {𝒥 : CategoryTheory.SmallCategory J}, CategoryTheory.FinCat …
          J : Type
          x✝¹ : CategoryTheory.SmallCategory J
          x✝ : CategoryTheory.FinCategory J
          this : CategoryTheory.Category.{?u.5708, ?u.5706} (CategoryTheory.ULiftHom (UL …
          ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J F
        -/
        haveI := h (ULiftHom (ULift J)) CategoryTheory.finCategoryUlift
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          h : ∀ (J : Type w) {𝒥 : CategoryTheory.SmallCategory J}, CategoryTheory.FinCat …
          J : Type
          x✝¹ : CategoryTheory.SmallCategory J
          x✝ : CategoryTheory.FinCategory J
          this✝ : CategoryTheory.Category.{?u.5708, ?u.5706} (CategoryTheory.ULiftHom (U …
          this : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.ULiftHom ( …
          ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J F
        -/
        exact preservesLimitsOfShape_of_equiv (ULiftHomULiftCategory.equiv J).symm F
        /-
          🎉 no goals
        -/


/-- The composition of two left exact functors is left exact. -/
lemma comp_preservesFiniteLimits (F : C ⥤ D) (G : D ⥤ E) [PreservesFiniteLimits F]
    [PreservesFiniteLimits G] : PreservesFiniteLimits (F ⋙ G) :=
  ⟨fun _ _ _ => inferInstance⟩


/-- Transfer preservation of finite limits along a natural isomorphism in the functor. -/
lemma preservesFiniteLimits_of_natIso {F G : C ⥤ D} (h : F ≅ G) [PreservesFiniteLimits F] :
    PreservesFiniteLimits G where
  preservesFiniteLimits _ _ _ := preservesLimitsOfShape_of_natIso h

/- Porting note: adding this class because quantified classes don't behave well
https://github.com/leanprover-community/mathlib4/pull/2764 -/

/-- A functor `F` preserves finite products if it preserves all from `Discrete J`
for `Fintype J` -/
class PreservesFiniteProducts (F : C ⥤ D) : Prop where
  preserves : ∀ (J : Type) [Fintype J], PreservesLimitsOfShape (Discrete J) F


instance (F : C ⥤ D) : Subsingleton (PreservesFiniteProducts F) :=
                     /-
                       C : Type u₁
                       inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                       D : Type u₂
                       inst✝² : CategoryTheory.Category.{v₂, u₂} D
                       E : Type u₃
                       inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
                       J : Type w
                       inst✝ : CategoryTheory.SmallCategory J
                       K : CategoryTheory.Functor J C
                       F : CategoryTheory.Functor C D
                       x✝¹ x✝ : CategoryTheory.Limits.PreservesFiniteProducts F
                       a b : ∀ (J : Type) [inst : Fintype J], CategoryTheory.Limits.PreservesLimitsOf …
                       ⊢ Eq ⋯ ⋯
                     -/
  ⟨fun ⟨a⟩ ⟨b⟩ => by congr⟩
                     /-
                       🎉 no goals
                     -/


instance (priority := 100) (F : C ⥤ D) (J : Type u) [Finite J]
    [PreservesFiniteProducts F] : PreservesLimitsOfShape (Discrete J) F := by
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    J✝ : Type w
    inst✝² : CategoryTheory.SmallCategory J✝
    K : CategoryTheory.Functor J✝ C
    F : CategoryTheory.Functor C D
    J : Type u
    inst✝¹ : Finite J
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts F
    ⊢ CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete J) F
  -/
  apply Nonempty.some
  /-
    case h
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    J✝ : Type w
    inst✝² : CategoryTheory.SmallCategory J✝
    K : CategoryTheory.Functor J✝ C
    F : CategoryTheory.Functor C D
    J : Type u
    inst✝¹ : Finite J
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts F
    ⊢ Nonempty (CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discr …
  -/
  obtain ⟨n, ⟨e⟩⟩ := Finite.exists_equiv_fin J
  /-
    case h.intro.intro
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    J✝ : Type w
    inst✝² : CategoryTheory.SmallCategory J✝
    K : CategoryTheory.Functor J✝ C
    F : CategoryTheory.Functor C D
    J : Type u
    inst✝¹ : Finite J
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts F
    n : Nat
    e : Equiv J (Fin n)
    ⊢ Nonempty (CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discr …
  -/
  exact ⟨preservesLimitsOfShape_of_equiv (Discrete.equivalence e.symm) F⟩
  /-
    🎉 no goals
  -/


instance comp_preservesFiniteProducts (F : C ⥤ D) (G : D ⥤ E)
    [PreservesFiniteProducts F] [PreservesFiniteProducts G] :
    PreservesFiniteProducts (F ⋙ G) where
  preserves _ _ := inferInstance


instance (F : C ⥤ D) [PreservesFiniteLimits F] : PreservesFiniteProducts F where
  preserves _ _ := inferInstance


/--
A functor is said to reflect finite limits, if it reflects all limits of shape `J`,
where `J : Type` is a finite category.
-/
class ReflectsFiniteLimits (F : C ⥤ D) : Prop where
  reflects : ∀ (J : Type) [SmallCategory J] [FinCategory J], ReflectsLimitsOfShape J F := by
    infer_instance


instance (F : C ⥤ D) : Subsingleton (ReflectsFiniteLimits F) :=
                     /-
                       C : Type u₁
                       inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                       D : Type u₂
                       inst✝² : CategoryTheory.Category.{v₂, u₂} D
                       E : Type u₃
                       inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
                       J : Type w
                       inst✝ : CategoryTheory.SmallCategory J
                       K : CategoryTheory.Functor J C
                       F : CategoryTheory.Functor C D
                       x✝¹ x✝ : CategoryTheory.Limits.ReflectsFiniteLimits F
                       a b : ∀ (J : Type) [inst : CategoryTheory.SmallCategory J] [inst_1 : CategoryT …
                       ⊢ Eq ⋯ ⋯
                     -/
  ⟨fun ⟨a⟩ ⟨b⟩ => by congr⟩
                     /-
                       🎉 no goals
                     -/

/- Similarly to preserving finite products, quantified classes don't behave well. -/

/--
A functor `F` preserves finite products if it reflects limits of shape `Discrete J` for finite `J`
-/
class ReflectsFiniteProducts (F : C ⥤ D) : Prop where
  reflects : ∀ (J : Type) [Fintype J], ReflectsLimitsOfShape (Discrete J) F


instance (F : C ⥤ D) : Subsingleton (ReflectsFiniteProducts F) :=
                     /-
                       C : Type u₁
                       inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                       D : Type u₂
                       inst✝² : CategoryTheory.Category.{v₂, u₂} D
                       E : Type u₃
                       inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
                       J : Type w
                       inst✝ : CategoryTheory.SmallCategory J
                       K : CategoryTheory.Functor J C
                       F : CategoryTheory.Functor C D
                       x✝¹ x✝ : CategoryTheory.Limits.ReflectsFiniteProducts F
                       a b : ∀ (J : Type) [inst : Fintype J], CategoryTheory.Limits.ReflectsLimitsOfS …
                       ⊢ Eq ⋯ ⋯
                     -/
  ⟨fun ⟨a⟩ ⟨b⟩ => by congr⟩
                     /-
                       🎉 no goals
                     -/

-- This is a dangerous instance as it has unbound universe variables.

/-- If we reflect limits of some arbitrary size, then we reflect all finite limits. -/
lemma ReflectsLimitsOfSize.reflectsFiniteLimits
    (F : C ⥤ D) [ReflectsLimitsOfSize.{w, w₂} F] : ReflectsFiniteLimits F where
  reflects J (sJ : SmallCategory J) fJ := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.Limits.ReflectsLimitsOfSize.{w, w₂, v₁, v₂, u₁, u₂} F
      J : Type
      sJ : CategoryTheory.SmallCategory J
      fJ : CategoryTheory.FinCategory J
      ⊢ CategoryTheory.Limits.ReflectsLimitsOfShape J F
    -/
    haveI := reflectsSmallestLimits_of_reflectsLimits F
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.Limits.ReflectsLimitsOfSize.{w, w₂, v₁, v₂, u₁, u₂} F
      J : Type
      sJ : CategoryTheory.SmallCategory J
      fJ : CategoryTheory.FinCategory J
      this : CategoryTheory.Limits.ReflectsLimitsOfSize.{0, 0, v₁, v₂, u₁, u₂} F
      ⊢ CategoryTheory.Limits.ReflectsLimitsOfShape J F
    -/
    exact reflectsLimitsOfShape_of_equiv (FinCategory.equivAsType J) F
    /-
      🎉 no goals
    -/

-- Added as a specialization of the dangerous instance above, for colimits indexed in Type 0.

instance (priority := 120) (F : C ⥤ D) [ReflectsLimitsOfSize.{0, 0} F] :
    ReflectsFiniteLimits F :=
  ReflectsLimitsOfSize.reflectsFiniteLimits F

-- An alternative specialization of the dangerous instance for small colimits.

instance (priority := 120) (F : C ⥤ D)
    [ReflectsLimits F] : ReflectsFiniteLimits F :=
  ReflectsLimitsOfSize.reflectsFiniteLimits F


/--
If `F ⋙ G` preserves finite limits and `G` reflects finite limits, then `F` preserves
finite limits.
-/
lemma preservesFiniteLimits_of_reflects_of_preserves (F : C ⥤ D) (G : D ⥤ E)
    [PreservesFiniteLimits (F ⋙ G)] [ReflectsFiniteLimits G] : PreservesFiniteLimits F where
  preservesFiniteLimits _ _ _ := preservesLimitsOfShape_of_reflects_of_preserves F G


/--
If `F ⋙ G` preserves finite products and `G` reflects finite products, then `F` preserves
finite products.
-/
lemma preservesFiniteProducts_of_reflects_of_preserves (F : C ⥤ D) (G : D ⥤ E)
    [PreservesFiniteProducts (F ⋙ G)] [ReflectsFiniteProducts G] : PreservesFiniteProducts F where
  preserves _ _ := preservesLimitsOfShape_of_reflects_of_preserves F G


instance reflectsFiniteLimits_of_reflectsIsomorphisms (F : C ⥤ D)
    [F.ReflectsIsomorphisms] [HasFiniteLimits C] [PreservesFiniteLimits F] :
      ReflectsFiniteLimits F where
  reflects _ _ _ := reflectsLimitsOfShape_of_reflectsIsomorphisms


instance reflectsFiniteProducts_of_reflectsIsomorphisms (F : C ⥤ D)
    [F.ReflectsIsomorphisms] [HasFiniteProducts C] [PreservesFiniteProducts F] :
      ReflectsFiniteProducts F where
  reflects _ _ := reflectsLimitsOfShape_of_reflectsIsomorphisms


instance comp_reflectsFiniteProducts (F : C ⥤ D) (G : D ⥤ E)
    [ReflectsFiniteProducts F] [ReflectsFiniteProducts G] :
    ReflectsFiniteProducts (F ⋙ G) where
  reflects _ _ := inferInstance


instance (F : C ⥤ D) [ReflectsFiniteLimits F] : ReflectsFiniteProducts F where
  reflects _ _ := inferInstance


/-- A functor is said to preserve finite colimits, if it preserves all colimits of
shape `J`, where `J : Type` is a finite category.
-/
class PreservesFiniteColimits (F : C ⥤ D) : Prop where
  preservesFiniteColimits :
    ∀ (J : Type) [SmallCategory J] [FinCategory J], PreservesColimitsOfShape J F := by
      infer_instance


instance (F : C ⥤ D) : Subsingleton (PreservesFiniteColimits F) :=
                     /-
                       C : Type u₁
                       inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                       D : Type u₂
                       inst✝² : CategoryTheory.Category.{v₂, u₂} D
                       E : Type u₃
                       inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
                       J : Type w
                       inst✝ : CategoryTheory.SmallCategory J
                       K : CategoryTheory.Functor J C
                       F : CategoryTheory.Functor C D
                       x✝¹ x✝ : CategoryTheory.Limits.PreservesFiniteColimits F
                       a b : ∀ (J : Type) [inst : CategoryTheory.SmallCategory J] [inst_1 : CategoryT …
                       ⊢ Eq ⋯ ⋯
                     -/
  ⟨fun ⟨a⟩ ⟨b⟩ => by congr⟩
                     /-
                       🎉 no goals
                     -/


/--
Preserving finite colimits also implies preserving colimits over finite shapes in higher
universes.
-/
instance (priority := 100) preservesColimitsOfShapeOfPreservesFiniteColimits
    (F : C ⥤ D) [PreservesFiniteColimits F] (J : Type w) [SmallCategory J] [FinCategory J] :
    PreservesColimitsOfShape J F := by
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} E
    J✝ : Type w
    inst✝³ : CategoryTheory.SmallCategory J✝
    K : CategoryTheory.Functor J✝ C
    F : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.Limits.PreservesFiniteColimits F
    J : Type w
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.FinCategory J
    ⊢ CategoryTheory.Limits.PreservesColimitsOfShape J F
  -/
  apply preservesColimitsOfShape_of_equiv (FinCategory.equivAsType J)
  /-
    🎉 no goals
  -/

-- This is a dangerous instance as it has unbound universe variables.

/-- If we preserve colimits of some arbitrary size, then we preserve all finite colimits. -/
lemma PreservesColimitsOfSize.preservesFiniteColimits (F : C ⥤ D)
    [PreservesColimitsOfSize.{w, w₂} F] : PreservesFiniteColimits F where
  preservesFiniteColimits J (sJ : SmallCategory J) fJ := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.Limits.PreservesColimitsOfSize.{w, w₂, v₁, v₂, u₁, u₂} F
      J : Type
      sJ : CategoryTheory.SmallCategory J
      fJ : CategoryTheory.FinCategory J
      ⊢ CategoryTheory.Limits.PreservesColimitsOfShape J F
    -/
    haveI := preservesSmallestColimits_of_preservesColimits F
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.Limits.PreservesColimitsOfSize.{w, w₂, v₁, v₂, u₁, u₂} F
      J : Type
      sJ : CategoryTheory.SmallCategory J
      fJ : CategoryTheory.FinCategory J
      this : CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₂, u₁, u₂} F
      ⊢ CategoryTheory.Limits.PreservesColimitsOfShape J F
    -/
    exact preservesColimitsOfShape_of_equiv (FinCategory.equivAsType J) F
    /-
      🎉 no goals
    -/

-- Added as a specialization of the dangerous instance above, for colimits indexed in Type 0.

instance (priority := 120) PreservesColimitsOfSize0.preservesFiniteColimits
    (F : C ⥤ D) [PreservesColimitsOfSize.{0, 0} F] : PreservesFiniteColimits F :=
  PreservesColimitsOfSize.preservesFiniteColimits F

-- An alternative specialization of the dangerous instance for small colimits.

instance (priority := 120) PreservesColimits.preservesFiniteColimits (F : C ⥤ D)
    [PreservesColimits F] : PreservesFiniteColimits F :=
  PreservesColimitsOfSize.preservesFiniteColimits F


/-- We can always derive `PreservesFiniteColimits C`
by showing that we are preserving colimits at an arbitrary universe. -/
lemma preservesFiniteColimits_of_preservesFiniteColimitsOfSize (F : C ⥤ D)
    (h :
      ∀ (J : Type w) {𝒥 : SmallCategory J} (_ : @FinCategory J 𝒥), PreservesColimitsOfShape J F) :
    PreservesFiniteColimits F where
      preservesFiniteColimits J (_ : SmallCategory J) _ := by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          h : ∀ (J : Type w) {𝒥 : CategoryTheory.SmallCategory J}, CategoryTheory.FinCat …
          J : Type
          x✝¹ : CategoryTheory.SmallCategory J
          x✝ : CategoryTheory.FinCategory J
          ⊢ CategoryTheory.Limits.PreservesColimitsOfShape J F
        -/
        letI : Category (ULiftHom (ULift J)) := ULiftHom.category
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          h : ∀ (J : Type w) {𝒥 : CategoryTheory.SmallCategory J}, CategoryTheory.FinCat …
          J : Type
          x✝¹ : CategoryTheory.SmallCategory J
          x✝ : CategoryTheory.FinCategory J
          this : CategoryTheory.Category.{?u.33281, ?u.33279} (CategoryTheory.ULiftHom ( …
          ⊢ CategoryTheory.Limits.PreservesColimitsOfShape J F
        -/
        haveI := h (ULiftHom (ULift J)) CategoryTheory.finCategoryUlift
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          h : ∀ (J : Type w) {𝒥 : CategoryTheory.SmallCategory J}, CategoryTheory.FinCat …
          J : Type
          x✝¹ : CategoryTheory.SmallCategory J
          x✝ : CategoryTheory.FinCategory J
          this✝ : CategoryTheory.Category.{?u.33281, ?u.33279} (CategoryTheory.ULiftHom  …
          this : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.ULiftHom …
          ⊢ CategoryTheory.Limits.PreservesColimitsOfShape J F
        -/
        exact preservesColimitsOfShape_of_equiv (ULiftHomULiftCategory.equiv J).symm F
        /-
          🎉 no goals
        -/


/-- The composition of two right exact functors is right exact. -/
lemma comp_preservesFiniteColimits (F : C ⥤ D) (G : D ⥤ E) [PreservesFiniteColimits F]
    [PreservesFiniteColimits G] : PreservesFiniteColimits (F ⋙ G) :=
  ⟨fun _ _ _ => inferInstance⟩


/-- Transfer preservation of finite colimits along a natural isomorphism in the functor. -/
lemma preservesFiniteColimits_of_natIso {F G : C ⥤ D} (h : F ≅ G) [PreservesFiniteColimits F] :
    PreservesFiniteColimits G where
  preservesFiniteColimits _ _ _ := preservesColimitsOfShape_of_natIso h

/- Porting note: adding this class because quantified classes don't behave well
https://github.com/leanprover-community/mathlib4/pull/2764 -/

/-- A functor `F` preserves finite products if it preserves all from `Discrete J`
for `Fintype J` -/
class PreservesFiniteCoproducts (F : C ⥤ D) : Prop where
  /-- preservation of colimits indexed by `Discrete J` when `[Fintype J]` -/
  preserves : ∀ (J : Type) [Fintype J], PreservesColimitsOfShape (Discrete J) F


instance (F : C ⥤ D) : Subsingleton (PreservesFiniteCoproducts F) :=
                     /-
                       C : Type u₁
                       inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                       D : Type u₂
                       inst✝² : CategoryTheory.Category.{v₂, u₂} D
                       E : Type u₃
                       inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
                       J : Type w
                       inst✝ : CategoryTheory.SmallCategory J
                       K : CategoryTheory.Functor J C
                       F : CategoryTheory.Functor C D
                       x✝¹ x✝ : CategoryTheory.Limits.PreservesFiniteCoproducts F
                       a b : ∀ (J : Type) [inst : Fintype J], CategoryTheory.Limits.PreservesColimits …
                       ⊢ Eq ⋯ ⋯
                     -/
  ⟨fun ⟨a⟩ ⟨b⟩ => by congr⟩
                     /-
                       🎉 no goals
                     -/


instance (priority := 100) (F : C ⥤ D) (J : Type u) [Finite J]
    [PreservesFiniteCoproducts F] : PreservesColimitsOfShape (Discrete J) F := by
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    J✝ : Type w
    inst✝² : CategoryTheory.SmallCategory J✝
    K : CategoryTheory.Functor J✝ C
    F : CategoryTheory.Functor C D
    J : Type u
    inst✝¹ : Finite J
    inst✝ : CategoryTheory.Limits.PreservesFiniteCoproducts F
    ⊢ CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discrete J) F
  -/
  apply Nonempty.some
  /-
    case h
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    J✝ : Type w
    inst✝² : CategoryTheory.SmallCategory J✝
    K : CategoryTheory.Functor J✝ C
    F : CategoryTheory.Functor C D
    J : Type u
    inst✝¹ : Finite J
    inst✝ : CategoryTheory.Limits.PreservesFiniteCoproducts F
    ⊢ Nonempty (CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Dis …
  -/
  obtain ⟨n, ⟨e⟩⟩ := Finite.exists_equiv_fin J
  /-
    case h.intro.intro
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} E
    J✝ : Type w
    inst✝² : CategoryTheory.SmallCategory J✝
    K : CategoryTheory.Functor J✝ C
    F : CategoryTheory.Functor C D
    J : Type u
    inst✝¹ : Finite J
    inst✝ : CategoryTheory.Limits.PreservesFiniteCoproducts F
    n : Nat
    e : Equiv J (Fin n)
    ⊢ Nonempty (CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Dis …
  -/
  exact ⟨preservesColimitsOfShape_of_equiv (Discrete.equivalence e.symm) F⟩
  /-
    🎉 no goals
  -/


instance comp_preservesFiniteCoproducts (F : C ⥤ D) (G : D ⥤ E)
    [PreservesFiniteCoproducts F] [PreservesFiniteCoproducts G] :
    PreservesFiniteCoproducts (F ⋙ G) where
  preserves _ _ := inferInstance


instance (F : C ⥤ D) [PreservesFiniteColimits F] : PreservesFiniteCoproducts F where
  preserves _ _ := inferInstance


/--
A functor is said to reflect finite colimits, if it reflects all colimits of shape `J`,
where `J : Type` is a finite category.
-/
class ReflectsFiniteColimits (F : C ⥤ D) : Prop where
  reflects : ∀ (J : Type) [SmallCategory J] [FinCategory J], ReflectsColimitsOfShape J F := by
    infer_instance


instance (F : C ⥤ D) : Subsingleton (ReflectsFiniteColimits F) :=
                     /-
                       C : Type u₁
                       inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                       D : Type u₂
                       inst✝² : CategoryTheory.Category.{v₂, u₂} D
                       E : Type u₃
                       inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
                       J : Type w
                       inst✝ : CategoryTheory.SmallCategory J
                       K : CategoryTheory.Functor J C
                       F : CategoryTheory.Functor C D
                       x✝¹ x✝ : CategoryTheory.Limits.ReflectsFiniteColimits F
                       a b : ∀ (J : Type) [inst : CategoryTheory.SmallCategory J] [inst_1 : CategoryT …
                       ⊢ Eq ⋯ ⋯
                     -/
  ⟨fun ⟨a⟩ ⟨b⟩ => by congr⟩
                     /-
                       🎉 no goals
                     -/

-- This is a dangerous instance as it has unbound universe variables.

/-- If we reflect colimits of some arbitrary size, then we reflect all finite colimits. -/
lemma ReflectsColimitsOfSize.reflectsFiniteColimits
    (F : C ⥤ D) [ReflectsColimitsOfSize.{w, w₂} F] : ReflectsFiniteColimits F where
  reflects J (sJ : SmallCategory J) fJ := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.Limits.ReflectsColimitsOfSize.{w, w₂, v₁, v₂, u₁, u₂} F
      J : Type
      sJ : CategoryTheory.SmallCategory J
      fJ : CategoryTheory.FinCategory J
      ⊢ CategoryTheory.Limits.ReflectsColimitsOfShape J F
    -/
    haveI := reflectsSmallestColimits_of_reflectsColimits F
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝ : CategoryTheory.Limits.ReflectsColimitsOfSize.{w, w₂, v₁, v₂, u₁, u₂} F
      J : Type
      sJ : CategoryTheory.SmallCategory J
      fJ : CategoryTheory.FinCategory J
      this : CategoryTheory.Limits.ReflectsColimitsOfSize.{0, 0, v₁, v₂, u₁, u₂} F
      ⊢ CategoryTheory.Limits.ReflectsColimitsOfShape J F
    -/
    exact reflectsColimitsOfShape_of_equiv (FinCategory.equivAsType J) F
    /-
      🎉 no goals
    -/

-- Added as a specialization of the dangerous instance above, for colimits indexed in Type 0.

instance (priority := 120) (F : C ⥤ D) [ReflectsColimitsOfSize.{0, 0} F] :
    ReflectsFiniteColimits F :=
  ReflectsColimitsOfSize.reflectsFiniteColimits F

-- An alternative specialization of the dangerous instance for small colimits.

instance (priority := 120) (F : C ⥤ D)
    [ReflectsColimits F] : ReflectsFiniteColimits F :=
  ReflectsColimitsOfSize.reflectsFiniteColimits F

/- Similarly to preserving finite coproducts, quantified classes don't behave well. -/

/--
A functor `F` preserves finite coproducts if it reflects colimits of shape `Discrete J` for
finite `J`
-/
class ReflectsFiniteCoproducts (F : C ⥤ D) : Prop where
  reflects : ∀ (J : Type) [Fintype J], ReflectsColimitsOfShape (Discrete J) F


instance (F : C ⥤ D) : Subsingleton (ReflectsFiniteCoproducts F) :=
                     /-
                       C : Type u₁
                       inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                       D : Type u₂
                       inst✝² : CategoryTheory.Category.{v₂, u₂} D
                       E : Type u₃
                       inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
                       J : Type w
                       inst✝ : CategoryTheory.SmallCategory J
                       K : CategoryTheory.Functor J C
                       F : CategoryTheory.Functor C D
                       x✝¹ x✝ : CategoryTheory.Limits.ReflectsFiniteCoproducts F
                       a b : ∀ (J : Type) [inst : Fintype J], CategoryTheory.Limits.ReflectsColimitsO …
                       ⊢ Eq ⋯ ⋯
                     -/
  ⟨fun ⟨a⟩ ⟨b⟩ => by congr⟩
                     /-
                       🎉 no goals
                     -/


/--
If `F ⋙ G` preserves finite colimits and `G` reflects finite colimits, then `F` preserves finite
colimits.
-/
lemma preservesFiniteColimits_of_reflects_of_preserves (F : C ⥤ D) (G : D ⥤ E)
    [PreservesFiniteColimits (F ⋙ G)] [ReflectsFiniteColimits G] : PreservesFiniteColimits F where
  preservesFiniteColimits _ _ _ := preservesColimitsOfShape_of_reflects_of_preserves F G


/--
If `F ⋙ G` preserves finite coproducts and `G` reflects finite coproducts, then `F` preserves
finite coproducts.
-/
lemma preservesFiniteCoproducts_of_reflects_of_preserves (F : C ⥤ D) (G : D ⥤ E)
    [PreservesFiniteCoproducts (F ⋙ G)] [ReflectsFiniteCoproducts G] :
    PreservesFiniteCoproducts F where
  preserves _ _ := preservesColimitsOfShape_of_reflects_of_preserves F G


instance reflectsFiniteColimitsOfReflectsIsomorphisms (F : C ⥤ D)
    [F.ReflectsIsomorphisms] [HasFiniteColimits C] [PreservesFiniteColimits F] :
      ReflectsFiniteColimits F where
  reflects _ _ _ := reflectsColimitsOfShape_of_reflectsIsomorphisms


instance reflectsFiniteCoproductsOfReflectsIsomorphisms (F : C ⥤ D)
    [F.ReflectsIsomorphisms] [HasFiniteCoproducts C] [PreservesFiniteCoproducts F] :
      ReflectsFiniteCoproducts F where
  reflects _ _ := reflectsColimitsOfShape_of_reflectsIsomorphisms


instance comp_reflectsFiniteCoproducts (F : C ⥤ D) (G : D ⥤ E)
    [ReflectsFiniteCoproducts F] [ReflectsFiniteCoproducts G] :
    ReflectsFiniteCoproducts (F ⋙ G) where
  reflects _ _ := inferInstance


instance (F : C ⥤ D) [ReflectsFiniteColimits F] : ReflectsFiniteCoproducts F where
  reflects _ _ := inferInstance



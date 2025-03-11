/-- The right adjoint of `Cocones.functoriality K F : Cocone K ⥤ Cocone (K ⋙ F)`.

Auxiliary definition for `functorialityIsLeftAdjoint`.
-/
def functorialityRightAdjoint : Cocone (K ⋙ F) ⥤ Cocone K :=
  Cocones.functoriality _ G ⋙
    Cocones.precompose (K.rightUnitor.inv ≫ whiskerLeft K adj.unit ≫ (associator _ _ _).inv)


attribute [local simp] functorialityRightAdjoint


/-- The unit for the adjunction for `Cocones.functoriality K F : Cocone K ⥤ Cocone (K ⋙ F)`.

Auxiliary definition for `functorialityIsLeftAdjoint`.
-/
@[simps]
def functorialityUnit :
    𝟭 (Cocone K) ⟶ Cocones.functoriality _ F ⋙ functorialityRightAdjoint adj K where
  app c := { hom := adj.unit.app c.pt }


/-- The counit for the adjunction for `Cocones.functoriality K F : Cocone K ⥤ Cocone (K ⋙ F)`.

Auxiliary definition for `functorialityIsLeftAdjoint`.
-/
@[simps]
def functorialityCounit :
    functorialityRightAdjoint adj K ⋙ Cocones.functoriality _ F ⟶ 𝟭 (Cocone (K ⋙ F)) where
  app c := { hom := adj.counit.app c.pt }


/-- The functor `Cocones.functoriality K F : Cocone K ⥤ Cocone (K ⋙ F)` is a left adjoint. -/
def functorialityAdjunction : Cocones.functoriality K F ⊣ functorialityRightAdjoint adj K where
  unit := functorialityUnit adj K
  counit := functorialityCounit adj K


include adj in
/-- A left adjoint preserves colimits.

See <https://stacks.math.columbia.edu/tag/0038>.
-/
lemma leftAdjoint_preservesColimits : PreservesColimitsOfSize.{v, u} F where
  preservesColimitsOfShape :=
    { preservesColimit :=
        { preserves := fun hc =>
            ⟨IsColimit.isoUniqueCoconeMorphism.inv fun _ =>
              @Equiv.unique _ _ (IsColimit.isoUniqueCoconeMorphism.hom hc _)
                ((adj.functorialityAdjunction _).homEquiv _ _)⟩ } }


include adj in
@[deprecated "No deprecation message was provided." (since := "2024-11-19")]
lemma leftAdjointPreservesColimits : PreservesColimitsOfSize.{v, u} F :=
  adj.leftAdjoint_preservesColimits


noncomputable
instance colim_preservesColimits [HasColimitsOfShape J C] :
    PreservesColimits (colim (J := J) (C := C)) :=
  colimConstAdj.leftAdjoint_preservesColimits

-- see Note [lower instance priority]

noncomputable instance (priority := 100) isEquivalence_preservesColimits
    (E : C ⥤ D) [E.IsEquivalence] :
    PreservesColimitsOfSize.{v, u} E :=
  leftAdjoint_preservesColimits E.adjunction

-- see Note [lower instance priority]

noncomputable instance (priority := 100)
    _root_.CategoryTheory.Functor.reflectsColimits_of_isEquivalence
    (E : D ⥤ C) [E.IsEquivalence] :
    ReflectsColimitsOfSize.{v, u} E where
  reflectsColimitsOfShape :=
    { reflectsColimit :=
        { reflects := fun t =>
          ⟨(isColimitOfPreserves E.inv t).mapCoconeEquiv E.asEquivalence.unitIso.symm⟩ } }


@[deprecated "No deprecation message was provided." (since := "2024-11-18")]
lemma isEquivalenceReflectsColimits (E : D ⥤ C) [E.IsEquivalence] :
    ReflectsColimitsOfSize.{v, u} E :=
  Functor.reflectsColimits_of_isEquivalence E

-- see Note [lower instance priority]

noncomputable instance (priority := 100)
    _root_.CategoryTheory.Functor.createsColimitsOfIsEquivalence (H : D ⥤ C)
    [H.IsEquivalence] :
    CreatesColimitsOfSize.{v, u} H where
  CreatesColimitsOfShape :=
    { CreatesColimit :=
        { lifts := fun c _ =>
            { liftedCocone := mapCoconeInv H c
              validLift := mapCoconeMapCoconeInv H c } } }


@[deprecated (since := "2024-11-18")] alias isEquivalenceCreatesColimits :=
  Functor.createsColimitsOfIsEquivalence

-- verify the preserve_colimits instance works as expected:

theorem hasColimit_comp_equivalence (E : C ⥤ D) [E.IsEquivalence] [HasColimit K] :
    HasColimit (K ⋙ E) :=
  HasColimit.mk
    { cocone := E.mapCocone (colimit.cocone K)
      isColimit := isColimitOfPreserves _ (colimit.isColimit K) }


theorem hasColimit_of_comp_equivalence (E : C ⥤ D) [E.IsEquivalence] [HasColimit (K ⋙ E)] :
    HasColimit K :=
  @hasColimitOfIso _ _ _ _ (K ⋙ E ⋙ E.inv) K
    (@hasColimit_comp_equivalence _ _ _ _ _ _ (K ⋙ E) E.inv _ _)
    ((Functor.rightUnitor _).symm ≪≫ isoWhiskerLeft K E.asEquivalence.unitIso)


/-- Transport a `HasColimitsOfShape` instance across an equivalence. -/
theorem hasColimitsOfShape_of_equivalence (E : C ⥤ D) [E.IsEquivalence] [HasColimitsOfShape J D] :
    HasColimitsOfShape J C :=
  ⟨fun F => hasColimit_of_comp_equivalence F E⟩


/-- Transport a `HasColimitsOfSize` instance across an equivalence. -/
theorem has_colimits_of_equivalence (E : C ⥤ D) [E.IsEquivalence] [HasColimitsOfSize.{v, u} D] :
    HasColimitsOfSize.{v, u} C :=
  ⟨fun _ _ => hasColimitsOfShape_of_equivalence E⟩


/-- The left adjoint of `Cones.functoriality K G : Cone K ⥤ Cone (K ⋙ G)`.

Auxiliary definition for `functorialityIsRightAdjoint`.
-/
def functorialityLeftAdjoint : Cone (K ⋙ G) ⥤ Cone K :=
  Cones.functoriality _ F ⋙
    Cones.postcompose ((associator _ _ _).hom ≫ whiskerLeft K adj.counit ≫ K.rightUnitor.hom)


attribute [local simp] functorialityLeftAdjoint


/-- The unit for the adjunction for `Cones.functoriality K G : Cone K ⥤ Cone (K ⋙ G)`.

Auxiliary definition for `functorialityIsRightAdjoint`.
-/
@[simps]
def functorialityUnit' :
    𝟭 (Cone (K ⋙ G)) ⟶ functorialityLeftAdjoint adj K ⋙ Cones.functoriality _ G where
  app c := { hom := adj.unit.app c.pt }


/-- The counit for the adjunction for `Cones.functoriality K G : Cone K ⥤ Cone (K ⋙ G)`.

Auxiliary definition for `functorialityIsRightAdjoint`.
-/
@[simps]
def functorialityCounit' :
    Cones.functoriality _ G ⋙ functorialityLeftAdjoint adj K ⟶ 𝟭 (Cone K) where
  app c := { hom := adj.counit.app c.pt }


/-- The functor `Cones.functoriality K G : Cone K ⥤ Cone (K ⋙ G)` is a right adjoint. -/
def functorialityAdjunction' : functorialityLeftAdjoint adj K ⊣ Cones.functoriality K G where
  unit := functorialityUnit' adj K
  counit := functorialityCounit' adj K


include adj in
/-- A right adjoint preserves limits.

See <https://stacks.math.columbia.edu/tag/0038>.
-/
lemma rightAdjoint_preservesLimits : PreservesLimitsOfSize.{v, u} G where
  preservesLimitsOfShape :=
    { preservesLimit :=
        { preserves := fun hc =>
            ⟨IsLimit.isoUniqueConeMorphism.inv fun _ =>
              @Equiv.unique _ _ (IsLimit.isoUniqueConeMorphism.hom hc _)
                ((adj.functorialityAdjunction' _).homEquiv _ _).symm⟩ } }


include adj in
@[deprecated "No deprecation message was provided." (since := "2024-11-19")]
lemma rightAdjointPreservesLimits : PreservesLimitsOfSize.{v, u} G :=
  adj.rightAdjoint_preservesLimits


instance lim_preservesLimits [HasLimitsOfShape J C] :
    PreservesLimits (lim (J := J) (C := C)) :=
  constLimAdj.rightAdjoint_preservesLimits

-- see Note [lower instance priority]

instance (priority := 100) isEquivalencePreservesLimits
    (E : D ⥤ C) [E.IsEquivalence] :
    PreservesLimitsOfSize.{v, u} E :=
  rightAdjoint_preservesLimits E.asEquivalence.symm.toAdjunction

-- see Note [lower instance priority]

noncomputable instance (priority := 100)
    _root_.CategoryTheory.Functor.reflectsLimits_of_isEquivalence
    (E : D ⥤ C) [E.IsEquivalence] :
    ReflectsLimitsOfSize.{v, u} E where
  reflectsLimitsOfShape :=
    { reflectsLimit :=
        { reflects := fun t =>
            ⟨(isLimitOfPreserves E.inv t).mapConeEquiv E.asEquivalence.unitIso.symm⟩ } }


@[deprecated "No deprecation message was provided." (since := "2024-11-18")]
lemma isEquivalenceReflectsLimits (E : D ⥤ C) [E.IsEquivalence] :
    ReflectsLimitsOfSize.{v, u} E :=
  Functor.reflectsLimits_of_isEquivalence E

-- see Note [lower instance priority]

noncomputable instance (priority := 100)
    _root_.CategoryTheory.Functor.createsLimitsOfIsEquivalence (H : D ⥤ C) [H.IsEquivalence] :
    CreatesLimitsOfSize.{v, u} H where
  CreatesLimitsOfShape :=
    { CreatesLimit :=
        { lifts := fun c _ =>
            { liftedCone := mapConeInv H c
              validLift := mapConeMapConeInv H c } } }


@[deprecated (since := "2024-11-18")] alias isEquivalenceCreatesLimits :=
  Functor.createsLimitsOfIsEquivalence

-- verify the preserve_limits instance works as expected:

theorem hasLimit_comp_equivalence (E : D ⥤ C) [E.IsEquivalence] [HasLimit K] : HasLimit (K ⋙ E) :=
  HasLimit.mk
    { cone := E.mapCone (limit.cone K)
      isLimit := isLimitOfPreserves _ (limit.isLimit K) }


theorem hasLimit_of_comp_equivalence (E : D ⥤ C) [E.IsEquivalence] [HasLimit (K ⋙ E)] :
    HasLimit K :=
  @hasLimitOfIso _ _ _ _ (K ⋙ E ⋙ E.inv) K
    (@hasLimit_comp_equivalence _ _ _ _ _ _ (K ⋙ E) E.inv _ _)
    (isoWhiskerLeft K E.asEquivalence.unitIso.symm ≪≫ Functor.rightUnitor _)


/-- Transport a `HasLimitsOfShape` instance across an equivalence. -/
theorem hasLimitsOfShape_of_equivalence (E : D ⥤ C) [E.IsEquivalence] [HasLimitsOfShape J C] :
    HasLimitsOfShape J D :=
  ⟨fun F => hasLimit_of_comp_equivalence F E⟩


/-- Transport a `HasLimitsOfSize` instance across an equivalence. -/
theorem has_limits_of_equivalence (E : D ⥤ C) [E.IsEquivalence] [HasLimitsOfSize.{v, u} C] :
    HasLimitsOfSize.{v, u} D :=
  ⟨fun _ _ => hasLimitsOfShape_of_equivalence E⟩


/-- auxiliary construction for `coconesIso` -/
@[simp]
def coconesIsoComponentHom {J : Type u} [Category.{v} J] {K : J ⥤ C} (Y : D)
    (t : ((cocones J D).obj (op (K ⋙ F))).obj Y) : (G ⋙ (cocones J C).obj (op K)).obj Y where
  app j := (adj.homEquiv (K.obj j) Y) (t.app j)
  naturality j j' f := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      J : Type u
      inst✝ : CategoryTheory.Category.{v, u} J
      K : CategoryTheory.Functor J C
      Y : D
      t : ((CategoryTheory.cocones J D).obj { unop := K.comp F }).obj Y
      j j' : J
      f : Quiver.Hom j j'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Opposite.unop { unop := Opposite.un …
    -/
    erw [← adj.homEquiv_naturality_left, t.naturality]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      J : Type u
      inst✝ : CategoryTheory.Category.{v, u} J
      K : CategoryTheory.Functor J C
      Y : D
      t : ((CategoryTheory.cocones J D).obj { unop := K.comp F }).obj Y
      j j' : J
      f : Quiver.Hom j j'
      ⊢ Eq ((adj.homEquiv ((Opposite.unop { unop := Opposite.unop { unop := K } }).o …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      J : Type u
      inst✝ : CategoryTheory.Category.{v, u} J
      K : CategoryTheory.Functor J C
      Y : D
      t : ((CategoryTheory.cocones J D).obj { unop := K.comp F }).obj Y
      j j' : J
      f : Quiver.Hom j j'
      ⊢ Eq ((adj.homEquiv (K.obj j) Y) (CategoryTheory.CategoryStruct.comp (t.app j) …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- auxiliary construction for `coconesIso` -/
@[simp]
def coconesIsoComponentInv {J : Type u} [Category.{v} J] {K : J ⥤ C} (Y : D)
    (t : (G ⋙ (cocones J C).obj (op K)).obj Y) : ((cocones J D).obj (op (K ⋙ F))).obj Y where
  app j := (adj.homEquiv (K.obj j) Y).symm (t.app j)
  naturality j j' f := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      J : Type u
      inst✝ : CategoryTheory.Category.{v, u} J
      K : CategoryTheory.Functor J C
      Y : D
      t : (G.comp ((CategoryTheory.cocones J C).obj { unop := K })).obj Y
      j j' : J
      f : Quiver.Hom j j'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Opposite.unop { unop := Opposite.un …
    -/
    erw [← adj.homEquiv_naturality_left_symm, ← adj.homEquiv_naturality_right_symm, t.naturality]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      J : Type u
      inst✝ : CategoryTheory.Category.{v, u} J
      K : CategoryTheory.Functor J C
      Y : D
      t : (G.comp ((CategoryTheory.cocones J C).obj { unop := K })).obj Y
      j j' : J
      f : Quiver.Hom j j'
      ⊢ Eq ((adj.homEquiv (K.obj j) (((CategoryTheory.Functor.const J).obj Y).obj j' …
    -/
    dsimp; simp
           /-
             🎉 no goals
           -/


/-- auxiliary construction for `conesIso` -/
@[simp]
def conesIsoComponentHom {J : Type u} [Category.{v} J] {K : J ⥤ D} (X : Cᵒᵖ)
    (t : (Functor.op F ⋙ (cones J D).obj K).obj X) : ((cones J C).obj (K ⋙ G)).obj X where
  app j := (adj.homEquiv (unop X) (K.obj j)) (t.app j)
  naturality j j' f := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      J : Type u
      inst✝ : CategoryTheory.Category.{v, u} J
      K : CategoryTheory.Functor J D
      X : Opposite C
      t : (F.op.comp ((CategoryTheory.cones J D).obj K)).obj X
      j j' : J
      f : Quiver.Hom j j'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Opposite.unop ((CategoryTheory.Func …
    -/
    erw [← adj.homEquiv_naturality_right, ← t.naturality, Category.id_comp, Category.id_comp]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      J : Type u
      inst✝ : CategoryTheory.Category.{v, u} J
      K : CategoryTheory.Functor J D
      X : Opposite C
      t : (F.op.comp ((CategoryTheory.cones J D).obj K)).obj X
      j j' : J
      f : Quiver.Hom j j'
      ⊢ Eq ((fun j => (adj.homEquiv (Opposite.unop X) (K.obj j)) (t.app j)) j') ((ad …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- auxiliary construction for `conesIso` -/
@[simp]
def conesIsoComponentInv {J : Type u} [Category.{v} J] {K : J ⥤ D} (X : Cᵒᵖ)
    (t : ((cones J C).obj (K ⋙ G)).obj X) : (Functor.op F ⋙ (cones J D).obj K).obj X where
  app j := (adj.homEquiv (unop X) (K.obj j)).symm (t.app j)
  naturality j j' f := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      J : Type u
      inst✝ : CategoryTheory.Category.{v, u} J
      K : CategoryTheory.Functor J D
      X : Opposite C
      t : ((CategoryTheory.cones J C).obj (K.comp G)).obj X
      j j' : J
      f : Quiver.Hom j j'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Opposite.unop ((CategoryTheory.Func …
    -/
    erw [← adj.homEquiv_naturality_right_symm, ← t.naturality, Category.id_comp, Category.id_comp]
    /-
      🎉 no goals
    -/


/-- When `F ⊣ G`,
the functor associating to each `Y` the cocones over `K ⋙ F` with cone point `Y`
is naturally isomorphic to
the functor associating to each `Y` the cocones over `K` with cone point `G.obj Y`.
-/
def coconesIso {J : Type u} [Category.{v} J] {K : J ⥤ C} :
    (cocones J D).obj (op (K ⋙ F)) ≅ G ⋙ (cocones J C).obj (op K) :=
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₀, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₀, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    J : Type u
    inst✝ : CategoryTheory.Category.{v, u} J
    K : CategoryTheory.Functor J C
    ⊢ ∀ {X Y : D} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
  -/
  NatIso.ofComponents fun Y =>
  /-
    🎉 no goals
  -/
    { hom := coconesIsoComponentHom adj Y
      inv := coconesIsoComponentInv adj Y }

-- Note: this is natural in K, but we do not yet have the tools to formulate that.

/-- When `F ⊣ G`,
the functor associating to each `X` the cones over `K` with cone point `F.op.obj X`
is naturally isomorphic to
the functor associating to each `X` the cones over `K ⋙ G` with cone point `X`.
-/
def conesIso {J : Type u} [Category.{v} J] {K : J ⥤ D} :
    F.op ⋙ (cones J D).obj K ≅ (cones J C).obj (K ⋙ G) :=
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₀, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₀, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    J : Type u
    inst✝ : CategoryTheory.Category.{v, u} J
    K : CategoryTheory.Functor J D
    ⊢ ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
  -/
  NatIso.ofComponents fun X =>
  /-
    🎉 no goals
  -/
    { hom := conesIsoComponentHom adj X
      inv := conesIsoComponentInv adj X }


noncomputable instance [IsLeftAdjoint F] : PreservesColimitsOfShape J F :=
  (Adjunction.ofIsLeftAdjoint F).leftAdjoint_preservesColimits.preservesColimitsOfShape


noncomputable instance [IsLeftAdjoint F] : PreservesColimitsOfSize.{v, u} F where


noncomputable instance [IsRightAdjoint F] : PreservesLimitsOfShape J F :=
  (Adjunction.ofIsRightAdjoint F).rightAdjoint_preservesLimits.preservesLimitsOfShape


noncomputable instance [IsRightAdjoint F] : PreservesLimitsOfSize.{v, u} F where



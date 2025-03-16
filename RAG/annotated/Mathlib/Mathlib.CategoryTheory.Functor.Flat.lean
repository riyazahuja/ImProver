/-- A functor `F : C ⥤ D` is representably flat if the comma category `(X/F)` is cofiltered for
each `X : D`.
-/
class RepresentablyFlat (F : C ⥤ D) : Prop where
  cofiltered : ∀ X : D, IsCofiltered (StructuredArrow X F)


/-- A functor `F : C ⥤ D` is representably coflat if the comma category `(F/X)` is filtered for
each `X : D`. -/
class RepresentablyCoflat (F : C ⥤ D) : Prop where
  filtered : ∀ X : D, IsFiltered (CostructuredArrow F X)


instance RepresentablyFlat.of_isRightAdjoint [F.IsRightAdjoint] : RepresentablyFlat F where
  cofiltered _ := IsCofiltered.of_isInitial _ (mkInitialOfLeftAdjoint _ (.ofIsRightAdjoint F) _)


instance RepresentablyCoflat.of_isLeftAdjoint [F.IsLeftAdjoint] : RepresentablyCoflat F where
  filtered _ := IsFiltered.of_isTerminal _ (mkTerminalOfRightAdjoint _ (.ofIsLeftAdjoint F) _)


theorem RepresentablyFlat.id : RepresentablyFlat (𝟭 C) := inferInstance


theorem RepresentablyCoflat.id : RepresentablyCoflat (𝟭 C) := inferInstance


set_option maxHeartbeats 400000 in
instance RepresentablyFlat.comp (G : D ⥤ E) [RepresentablyFlat F]
    [RepresentablyFlat G] : RepresentablyFlat (F ⋙ G) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    inst✝¹ : CategoryTheory.RepresentablyFlat F
    inst✝ : CategoryTheory.RepresentablyFlat G
    ⊢ CategoryTheory.RepresentablyFlat (F.comp G)
  -/
  refine ⟨fun X => IsCofiltered.of_cone_nonempty.{0} _ (fun {J} _ _ H => ?_)⟩
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    inst✝¹ : CategoryTheory.RepresentablyFlat F
    inst✝ : CategoryTheory.RepresentablyFlat G
    X : E
    J : Type
    x✝¹ : CategoryTheory.SmallCategory J
    x✝ : CategoryTheory.FinCategory J
    H : CategoryTheory.Functor J (CategoryTheory.StructuredArrow X (F.comp G))
    ⊢ Nonempty (CategoryTheory.Limits.Cone H)
  -/
  obtain ⟨c₁⟩ := IsCofiltered.cone_nonempty (H ⋙ StructuredArrow.pre X F G)
  let H₂ : J ⥤ StructuredArrow c₁.pt.right F :=
    { obj := fun j => StructuredArrow.mk (c₁.π.app j).right
      map := fun {j j'} f =>
        StructuredArrow.homMk (H.map f).right (congrArg CommaMorphism.right (c₁.w f)) }
  /-
    case intro
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    inst✝¹ : CategoryTheory.RepresentablyFlat F
    inst✝ : CategoryTheory.RepresentablyFlat G
    X : E
    J : Type
    x✝¹ : CategoryTheory.SmallCategory J
    x✝ : CategoryTheory.FinCategory J
    H : CategoryTheory.Functor J (CategoryTheory.StructuredArrow X (F.comp G))
    c₁ : CategoryTheory.Limits.Cone (H.comp (CategoryTheory.StructuredArrow.pre X  …
    H₂ : CategoryTheory.Functor J (CategoryTheory.StructuredArrow c₁.pt.right F) : …
    ⊢ Nonempty (CategoryTheory.Limits.Cone H)
  -/
  obtain ⟨c₂⟩ := IsCofiltered.cone_nonempty H₂
  /-
    case intro.intro
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    inst✝¹ : CategoryTheory.RepresentablyFlat F
    inst✝ : CategoryTheory.RepresentablyFlat G
    X : E
    J : Type
    x✝¹ : CategoryTheory.SmallCategory J
    x✝ : CategoryTheory.FinCategory J
    H : CategoryTheory.Functor J (CategoryTheory.StructuredArrow X (F.comp G))
    c₁ : CategoryTheory.Limits.Cone (H.comp (CategoryTheory.StructuredArrow.pre X  …
    H₂ : CategoryTheory.Functor J (CategoryTheory.StructuredArrow c₁.pt.right F) : …
    c₂ : CategoryTheory.Limits.Cone H₂
    ⊢ Nonempty (CategoryTheory.Limits.Cone H)
  -/
  simp only [H₂] at c₂
  exact ⟨⟨StructuredArrow.mk (c₁.pt.hom ≫ G.map c₂.pt.hom),
    ⟨fun j => StructuredArrow.homMk (c₂.π.app j).right (by simp [← G.map_comp, (c₂.π.app j).w]),
     fun j j' f => by simpa using (c₂.w f).symm⟩⟩⟩


/-- Being a representably flat functor is closed under natural isomorphisms. -/
theorem RepresentablyFlat.of_iso [RepresentablyFlat F] {G : C ⥤ D} (α : F ≅ G) :
    RepresentablyFlat G where
  cofiltered _ := IsCofiltered.of_equivalence (StructuredArrow.mapNatIso α)


theorem RepresentablyCoflat.of_iso [RepresentablyCoflat F] {G : C ⥤ D} (α : F ≅ G) :
    RepresentablyCoflat G where
  filtered _ := IsFiltered.of_equivalence (CostructuredArrow.mapNatIso α)


theorem representablyCoflat_op_iff : RepresentablyCoflat F.op ↔ RepresentablyFlat F := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    ⊢ Iff (CategoryTheory.RepresentablyCoflat F.op) (CategoryTheory.RepresentablyF …
  -/
  refine ⟨fun _ => ⟨fun X => ?_⟩, fun _ => ⟨fun ⟨X⟩ => ?_⟩⟩
    /-
      case refine_1
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      x✝ : CategoryTheory.RepresentablyCoflat F.op
      X : D
      ⊢ CategoryTheory.IsCofiltered (CategoryTheory.StructuredArrow X F)
    -/
  · suffices IsFiltered (StructuredArrow X F)ᵒᵖ from isCofiltered_of_isFiltered_op _
    /-
      case refine_1
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      x✝ : CategoryTheory.RepresentablyCoflat F.op
      X : D
      ⊢ CategoryTheory.IsFiltered (Opposite (CategoryTheory.StructuredArrow X F))
    -/
    apply IsFiltered.of_equivalence (structuredArrowOpEquivalence _ _).symm
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      x✝¹ : CategoryTheory.RepresentablyFlat F
      x✝ : Opposite D
      X : D
      ⊢ CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow F.op { unop := X …
    -/
  · suffices IsCofiltered (CostructuredArrow F.op (op X))ᵒᵖ from isFiltered_of_isCofiltered_op _
    suffices IsCofiltered (StructuredArrow X F)ᵒᵖᵒᵖ from
      IsCofiltered.of_equivalence (structuredArrowOpEquivalence _ _).op
    /-
      case refine_2
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      x✝¹ : CategoryTheory.RepresentablyFlat F
      x✝ : Opposite D
      X : D
      ⊢ CategoryTheory.IsCofiltered (Opposite (Opposite (CategoryTheory.StructuredAr …
    -/
    apply IsCofiltered.of_equivalence (opOpEquivalence _)
    /-
      🎉 no goals
    -/


theorem representablyFlat_op_iff : RepresentablyFlat F.op ↔ RepresentablyCoflat F := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    ⊢ Iff (CategoryTheory.RepresentablyFlat F.op) (CategoryTheory.RepresentablyCof …
  -/
  refine ⟨fun _ => ⟨fun X => ?_⟩, fun _ => ⟨fun ⟨X⟩ => ?_⟩⟩
    /-
      case refine_1
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      x✝ : CategoryTheory.RepresentablyFlat F.op
      X : D
      ⊢ CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow F X)
    -/
  · suffices IsCofiltered (CostructuredArrow F X)ᵒᵖ from isFiltered_of_isCofiltered_op _
    /-
      case refine_1
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      x✝ : CategoryTheory.RepresentablyFlat F.op
      X : D
      ⊢ CategoryTheory.IsCofiltered (Opposite (CategoryTheory.CostructuredArrow F X))
    -/
    apply IsCofiltered.of_equivalence (costructuredArrowOpEquivalence _ _).symm
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      x✝¹ : CategoryTheory.RepresentablyCoflat F
      x✝ : Opposite D
      X : D
      ⊢ CategoryTheory.IsCofiltered (CategoryTheory.StructuredArrow { unop := X } F. …
    -/
  · suffices IsFiltered (StructuredArrow (op X) F.op)ᵒᵖ from isCofiltered_of_isFiltered_op _
    suffices IsFiltered (CostructuredArrow F X)ᵒᵖᵒᵖ from
      IsFiltered.of_equivalence (costructuredArrowOpEquivalence _ _).op
    /-
      case refine_2
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      x✝¹ : CategoryTheory.RepresentablyCoflat F
      x✝ : Opposite D
      X : D
      ⊢ CategoryTheory.IsFiltered (Opposite (Opposite (CategoryTheory.CostructuredAr …
    -/
    apply IsFiltered.of_equivalence (opOpEquivalence _)
    /-
      🎉 no goals
    -/


instance [RepresentablyFlat F] : RepresentablyCoflat F.op :=
  (representablyCoflat_op_iff F).2 inferInstance


instance [RepresentablyCoflat F] : RepresentablyFlat F.op :=
  (representablyFlat_op_iff F).2 inferInstance


instance RepresentablyCoflat.comp (G : D ⥤ E) [RepresentablyCoflat F] [RepresentablyCoflat G] :
    RepresentablyCoflat (F ⋙ G) :=
  (representablyFlat_op_iff _).1 <| inferInstanceAs <| RepresentablyFlat (F.op ⋙ G.op)


theorem flat_of_preservesFiniteLimits [HasFiniteLimits C] (F : C ⥤ D) [PreservesFiniteLimits F] :
    RepresentablyFlat F :=
  ⟨fun X =>
    haveI : HasFiniteLimits (StructuredArrow X F) := by
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
        F : CategoryTheory.Functor C D
        inst✝ : CategoryTheory.Limits.PreservesFiniteLimits F
        X : D
        ⊢ CategoryTheory.Limits.HasFiniteLimits (CategoryTheory.StructuredArrow X F)
      -/
      apply hasFiniteLimits_of_hasFiniteLimits_of_size.{v₁} (StructuredArrow X F)
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
        F : CategoryTheory.Functor C D
        inst✝ : CategoryTheory.Limits.PreservesFiniteLimits F
        X : D
        ⊢ ∀ (J : Type v₁) {𝒥 : CategoryTheory.SmallCategory J}, CategoryTheory.FinCate …
      -/
      intro J sJ fJ
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
        F : CategoryTheory.Functor C D
        inst✝ : CategoryTheory.Limits.PreservesFiniteLimits F
        X : D
        J : Type v₁
        sJ : CategoryTheory.SmallCategory J
        fJ : CategoryTheory.FinCategory J
        ⊢ CategoryTheory.Limits.HasLimitsOfShape J (CategoryTheory.StructuredArrow X F)
      -/
      constructor
      -- Porting note: instance was inferred automatically in Lean 3
      /-
        case has_limit
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
        F : CategoryTheory.Functor C D
        inst✝ : CategoryTheory.Limits.PreservesFiniteLimits F
        X : D
        J : Type v₁
        sJ : CategoryTheory.SmallCategory J
        fJ : CategoryTheory.FinCategory J
        ⊢ autoParam (∀ (F_1 : CategoryTheory.Functor J (CategoryTheory.StructuredArrow …
      -/
      infer_instance
      /-
        🎉 no goals
      -/
    IsCofiltered.of_hasFiniteLimits _⟩


theorem coflat_of_preservesFiniteColimits [HasFiniteColimits C] (F : C ⥤ D)
    [PreservesFiniteColimits F] : RepresentablyCoflat F :=
  let _ := preservesFiniteLimits_op F
  (representablyFlat_op_iff _).1 (flat_of_preservesFiniteLimits _)


/-- (Implementation).
Given a limit cone `c : cone K` and a cone `s : cone (K ⋙ F)` with `F` representably flat,
`s` can factor through `F.mapCone c`.
-/
noncomputable def lift : s.pt ⟶ F.obj c.pt :=
  let s' := IsCofiltered.cone (s.toStructuredArrow ⋙ StructuredArrow.pre _ K F)
  s'.pt.hom ≫
    (F.map <|
      hc.lift <|
        (Cones.postcompose
              ({ app := fun _ => 𝟙 _ } :
                (s.toStructuredArrow ⋙ pre s.pt K F) ⋙ proj s.pt F ⟶ K)).obj <|
          (StructuredArrow.proj s.pt F).mapCone s')


theorem fac (x : J) : lift F hc s ≫ (F.mapCone c).π.app x = s.π.app x := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    J : Type v₁
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.FinCategory J
    K : CategoryTheory.Functor J C
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat F
    c : CategoryTheory.Limits.Cone K
    hc : CategoryTheory.Limits.IsLimit c
    s : CategoryTheory.Limits.Cone (K.comp F)
    x : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.PreservesFiniteLimits …
  -/
  simp [lift, ← Functor.map_comp]
  /-
    🎉 no goals
  -/


theorem uniq {K : J ⥤ C} {c : Cone K} (hc : IsLimit c) (s : Cone (K ⋙ F))
    (f₁ f₂ : s.pt ⟶ F.obj c.pt) (h₁ : ∀ j : J, f₁ ≫ (F.mapCone c).π.app j = s.π.app j)
    (h₂ : ∀ j : J, f₂ ≫ (F.mapCone c).π.app j = s.π.app j) : f₁ = f₂ := by
  -- We can make two cones over the diagram of `s` via `f₁` and `f₂`.
  let α₁ : (F.mapCone c).toStructuredArrow ⋙ map f₁ ⟶ s.toStructuredArrow :=
    { app := fun X => eqToHom (by simp [← h₁]) }
  let α₂ : (F.mapCone c).toStructuredArrow ⋙ map f₂ ⟶ s.toStructuredArrow :=
    { app := fun X => eqToHom (by simp [← h₂]) }
  let c₁ : Cone (s.toStructuredArrow ⋙ pre s.pt K F) :=
    (Cones.postcompose (whiskerRight α₁ (pre s.pt K F) : _)).obj (c.toStructuredArrowCone F f₁)
  let c₂ : Cone (s.toStructuredArrow ⋙ pre s.pt K F) :=
    (Cones.postcompose (whiskerRight α₂ (pre s.pt K F) : _)).obj (c.toStructuredArrowCone F f₂)
  -- The two cones can then be combined and we may obtain a cone over the two cones since
  -- `StructuredArrow s.pt F` is cofiltered.
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    J : Type v₁
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.FinCategory J
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat F
    K : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cone K
    hc : CategoryTheory.Limits.IsLimit c
    s : CategoryTheory.Limits.Cone (K.comp F)
    f₁ f₂ : Quiver.Hom s.pt (F.obj c.pt)
    h₁ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp f₁ ((F.mapCone c).π.app …
    h₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp f₂ ((F.mapCone c).π.app …
    α₁ : Quiver.Hom ((F.mapCone c).toStructuredArrow.comp (CategoryTheory.Structur …
    α₂ : Quiver.Hom ((F.mapCone c).toStructuredArrow.comp (CategoryTheory.Structur …
    c₁ : CategoryTheory.Limits.Cone (s.toStructuredArrow.comp (CategoryTheory.Stru …
    c₂ : CategoryTheory.Limits.Cone (s.toStructuredArrow.comp (CategoryTheory.Stru …
    ⊢ Eq f₁ f₂
  -/
  let c₀ := IsCofiltered.cone (biconeMk _ c₁ c₂)
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    J : Type v₁
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.FinCategory J
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat F
    K : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cone K
    hc : CategoryTheory.Limits.IsLimit c
    s : CategoryTheory.Limits.Cone (K.comp F)
    f₁ f₂ : Quiver.Hom s.pt (F.obj c.pt)
    h₁ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp f₁ ((F.mapCone c).π.app …
    h₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp f₂ ((F.mapCone c).π.app …
    α₁ : Quiver.Hom ((F.mapCone c).toStructuredArrow.comp (CategoryTheory.Structur …
    α₂ : Quiver.Hom ((F.mapCone c).toStructuredArrow.comp (CategoryTheory.Structur …
    c₁ : CategoryTheory.Limits.Cone (s.toStructuredArrow.comp (CategoryTheory.Stru …
    c₂ : CategoryTheory.Limits.Cone (s.toStructuredArrow.comp (CategoryTheory.Stru …
    c₀ : CategoryTheory.Limits.Cone (CategoryTheory.biconeMk J c₁ c₂) := CategoryT …
    ⊢ Eq f₁ f₂
  -/
  let g₁ : c₀.pt ⟶ c₁.pt := c₀.π.app Bicone.left
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    J : Type v₁
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.FinCategory J
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat F
    K : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cone K
    hc : CategoryTheory.Limits.IsLimit c
    s : CategoryTheory.Limits.Cone (K.comp F)
    f₁ f₂ : Quiver.Hom s.pt (F.obj c.pt)
    h₁ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp f₁ ((F.mapCone c).π.app …
    h₂ : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp f₂ ((F.mapCone c).π.app …
    α₁ : Quiver.Hom ((F.mapCone c).toStructuredArrow.comp (CategoryTheory.Structur …
    α₂ : Quiver.Hom ((F.mapCone c).toStructuredArrow.comp (CategoryTheory.Structur …
    c₁ : CategoryTheory.Limits.Cone (s.toStructuredArrow.comp (CategoryTheory.Stru …
    c₂ : CategoryTheory.Limits.Cone (s.toStructuredArrow.comp (CategoryTheory.Stru …
    c₀ : CategoryTheory.Limits.Cone (CategoryTheory.biconeMk J c₁ c₂) := CategoryT …
    g₁ : Quiver.Hom c₀.pt c₁.pt := c₀.π.app CategoryTheory.Bicone.left
    ⊢ Eq f₁ f₂
  -/
  let g₂ : c₀.pt ⟶ c₂.pt := c₀.π.app Bicone.right
  -- Then `g₁.right` and `g₂.right` are two maps from the same cone into the `c`.
  have : ∀ j : J, g₁.right ≫ c.π.app j = g₂.right ≫ c.π.app j := by
    intro j
    injection c₀.π.naturality (BiconeHom.left j) with _ e₁
    injection c₀.π.naturality (BiconeHom.right j) with _ e₂
    convert e₁.symm.trans e₂ <;> simp [c₁, c₂]
  have : c.extend g₁.right = c.extend g₂.right := by
    unfold Cone.extend
    congr 1
    ext x
    apply this
  -- And thus they are equal as `c` is the limit.
  have : g₁.right = g₂.right := calc
    g₁.right = hc.lift (c.extend g₁.right) := by
      apply hc.uniq (c.extend _)
      -- Porting note: was `by tidy`, but `aesop` only works if max heartbeats
      -- is increased, so we replace it by the output of `tidy?`
      intro j; rfl
    _ = hc.lift (c.extend g₂.right) := by
      congr
    _ = g₂.right := by
      symm
      apply hc.uniq (c.extend _)
      -- Porting note: was `by tidy`, but `aesop` only works if max heartbeats
      -- is increased, so we replace it by the output of `tidy?`
      intro _; rfl

  -- Finally, since `fᵢ` factors through `F(gᵢ)`, the result follows.
  calc
    f₁ = 𝟙 _ ≫ f₁ := by simp
    _ = c₀.pt.hom ≫ F.map g₁.right := g₁.w
    _ = c₀.pt.hom ≫ F.map g₂.right := by rw [this]
    _ = 𝟙 _ ≫ f₂ := g₂.w.symm
    _ = f₂ := by simp


/-- Representably flat functors preserve finite limits. -/
lemma preservesFiniteLimits_of_flat (F : C ⥤ D) [RepresentablyFlat F] :
    PreservesFiniteLimits F := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat F
    ⊢ CategoryTheory.Limits.PreservesFiniteLimits F
  -/
  apply preservesFiniteLimits_of_preservesFiniteLimitsOfSize
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat F
    ⊢ ∀ (J : Type ?u.258451) {𝒥 : CategoryTheory.SmallCategory J}, CategoryTheory. …
  -/
  intro J _ _; constructor
  /-
    case h.preservesLimit
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat F
    J : Type ?u.258451
    𝒥✝ : CategoryTheory.SmallCategory J
    x✝ : CategoryTheory.FinCategory J
    ⊢ autoParam (∀ {K : CategoryTheory.Functor J C}, CategoryTheory.Limits.Preserv …
  -/
  intro K; constructor
  /-
    case h.preservesLimit.preserves
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat F
    J : Type ?u.258451
    𝒥✝ : CategoryTheory.SmallCategory J
    x✝ : CategoryTheory.FinCategory J
    K : CategoryTheory.Functor J C
    ⊢ ∀ {c : CategoryTheory.Limits.Cone K}, CategoryTheory.Limits.IsLimit c → None …
  -/
  intro c hc
  /-
    case h.preservesLimit.preserves
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat F
    J : Type ?u.258451
    𝒥✝ : CategoryTheory.SmallCategory J
    x✝ : CategoryTheory.FinCategory J
    K : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cone K
    hc : CategoryTheory.Limits.IsLimit c
    ⊢ Nonempty (CategoryTheory.Limits.IsLimit (F.mapCone c))
  -/
  constructor
  exact
    { lift := PreservesFiniteLimitsOfFlat.lift F hc
      fac := PreservesFiniteLimitsOfFlat.fac F hc
      uniq := fun s m h => by
        apply PreservesFiniteLimitsOfFlat.uniq F hc
        · exact h
        · exact PreservesFiniteLimitsOfFlat.fac F hc s }


/-- Representably coflat functors preserve finite colimits. -/
lemma preservesFiniteColimits_of_coflat (F : C ⥤ D) [RepresentablyCoflat F] :
    PreservesFiniteColimits F :=
  letI _ := preservesFiniteLimits_of_flat F.op
  preservesFiniteColimits_of_op _


/-- If `C` is finitely complete, then `F : C ⥤ D` is representably flat iff it preserves
finite limits.
-/
lemma preservesFiniteLimits_iff_flat [HasFiniteLimits C] (F : C ⥤ D) :
    RepresentablyFlat F ↔ PreservesFiniteLimits F :=
  ⟨fun _ ↦ preservesFiniteLimits_of_flat F, fun _ ↦ flat_of_preservesFiniteLimits F⟩


/-- If `C` is finitely cocomplete, then `F : C ⥤ D` is representably coflat iff it preserves
finite colmits. -/
lemma preservesFiniteColimits_iff_coflat [HasFiniteColimits C] (F : C ⥤ D) :
    RepresentablyCoflat F ↔ PreservesFiniteColimits F :=
  ⟨fun _ => preservesFiniteColimits_of_coflat F, fun _ => coflat_of_preservesFiniteColimits F⟩


/-- (Implementation)
The evaluation of `F.lan` at `X` is the colimit over the costructured arrows over `X`.
-/
noncomputable def lanEvaluationIsoColim (F : C ⥤ D) (X : D)
    [∀ X : D, HasColimitsOfShape (CostructuredArrow F X) E] :
    F.lan ⋙ (evaluation D E).obj X ≅
      (whiskeringLeft _ _ E).obj (CostructuredArrow.proj F X) ⋙ colim :=
  NatIso.ofComponents (fun G =>
    IsColimit.coconePointUniqueUpToIso
    (Functor.isPointwiseLeftKanExtensionLeftKanExtensionUnit F G X)
    (colimit.isColimit _)) (fun {G₁ G₂} φ => by
      /-
        C D : Type u₁
        inst✝³ : CategoryTheory.SmallCategory C
        inst✝² : CategoryTheory.SmallCategory D
        E : Type u₂
        inst✝¹ : CategoryTheory.Category.{u₁, u₂} E
        F : CategoryTheory.Functor C D
        X : D
        inst✝ : ∀ (X : D), CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Co …
        G₁ G₂ : CategoryTheory.Functor C E
        φ : Quiver.Hom G₁ G₂
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.lan.comp ((CategoryTheory.evaluat …
      -/
      apply (Functor.isPointwiseLeftKanExtensionLeftKanExtensionUnit F G₁ X).hom_ext
      /-
        C D : Type u₁
        inst✝³ : CategoryTheory.SmallCategory C
        inst✝² : CategoryTheory.SmallCategory D
        E : Type u₂
        inst✝¹ : CategoryTheory.Category.{u₁, u₂} E
        F : CategoryTheory.Functor C D
        X : D
        inst✝ : ∀ (X : D), CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Co …
        G₁ G₂ : CategoryTheory.Functor C E
        φ : Quiver.Hom G₁ G₂
        ⊢ ∀ (j : CategoryTheory.CostructuredArrow F X), Eq (CategoryTheory.CategoryStr …
      -/
      intro T
      have h₁ := fun (G : C ⥤ E) => IsColimit.comp_coconePointUniqueUpToIso_hom
        (Functor.isPointwiseLeftKanExtensionLeftKanExtensionUnit F G X) (colimit.isColimit _) T
      /-
        C D : Type u₁
        inst✝³ : CategoryTheory.SmallCategory C
        inst✝² : CategoryTheory.SmallCategory D
        E : Type u₂
        inst✝¹ : CategoryTheory.Category.{u₁, u₂} E
        F : CategoryTheory.Functor C D
        X : D
        inst✝ : ∀ (X : D), CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Co …
        G₁ G₂ : CategoryTheory.Functor C E
        φ : Quiver.Hom G₁ G₂
        T : CategoryTheory.CostructuredArrow F X
        h₁ : ∀ (G : CategoryTheory.Functor C E), Eq (CategoryTheory.CategoryStruct.com …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.LeftExtensi …
      -/
      have h₂ := congr_app (F.lanUnit.naturality φ) T.left
      /-
        C D : Type u₁
        inst✝³ : CategoryTheory.SmallCategory C
        inst✝² : CategoryTheory.SmallCategory D
        E : Type u₂
        inst✝¹ : CategoryTheory.Category.{u₁, u₂} E
        F : CategoryTheory.Functor C D
        X : D
        inst✝ : ∀ (X : D), CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Co …
        G₁ G₂ : CategoryTheory.Functor C E
        φ : Quiver.Hom G₁ G₂
        T : CategoryTheory.CostructuredArrow F X
        h₁ : ∀ (G : CategoryTheory.Functor C E), Eq (CategoryTheory.CategoryStruct.com …
        h₂ : Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Cate …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.LeftExtensi …
      -/
      dsimp at h₁ h₂ ⊢
      /-
        C D : Type u₁
        inst✝³ : CategoryTheory.SmallCategory C
        inst✝² : CategoryTheory.SmallCategory D
        E : Type u₂
        inst✝¹ : CategoryTheory.Category.{u₁, u₂} E
        F : CategoryTheory.Functor C D
        X : D
        inst✝ : ∀ (X : D), CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Co …
        G₁ G₂ : CategoryTheory.Functor C E
        φ : Quiver.Hom G₁ G₂
        T : CategoryTheory.CostructuredArrow F X
        h₁ : ∀ (G : CategoryTheory.Functor C E), Eq (CategoryTheory.CategoryStruct.com …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp (φ.app T.left) ((F.lanUnit.app G₂) …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      simp only [Category.assoc] at h₁ ⊢
      /-
        C D : Type u₁
        inst✝³ : CategoryTheory.SmallCategory C
        inst✝² : CategoryTheory.SmallCategory D
        E : Type u₂
        inst✝¹ : CategoryTheory.Category.{u₁, u₂} E
        F : CategoryTheory.Functor C D
        X : D
        inst✝ : ∀ (X : D), CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Co …
        G₁ G₂ : CategoryTheory.Functor C E
        φ : Quiver.Hom G₁ G₂
        T : CategoryTheory.CostructuredArrow F X
        h₂ : Eq (CategoryTheory.CategoryStruct.comp (φ.app T.left) ((F.lanUnit.app G₂) …
        h₁ : ∀ (G : CategoryTheory.Functor C E), Eq (CategoryTheory.CategoryStruct.com …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.leftKanExtensionUnit G₁).app T.le …
      -/
      simp only [Functor.lan, Functor.lanUnit] at h₂ ⊢
      rw [reassoc_of% h₁, NatTrans.naturality_assoc, ← reassoc_of% h₂, h₁,
        ι_colimMap, whiskerLeft_app]
      /-
        C D : Type u₁
        inst✝³ : CategoryTheory.SmallCategory C
        inst✝² : CategoryTheory.SmallCategory D
        E : Type u₂
        inst✝¹ : CategoryTheory.Category.{u₁, u₂} E
        F : CategoryTheory.Functor C D
        X : D
        inst✝ : ∀ (X : D), CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Co …
        G₁ G₂ : CategoryTheory.Functor C E
        φ : Quiver.Hom G₁ G₂
        T : CategoryTheory.CostructuredArrow F X
        h₂ : Eq (CategoryTheory.CategoryStruct.comp (φ.app T.left) ((F.leftKanExtensio …
        h₁ : ∀ (G : CategoryTheory.Functor C E), Eq (CategoryTheory.CategoryStruct.com …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (φ.app T.left) (CategoryTheory.Limits …
      -/
      rfl)
      /-
        🎉 no goals
      -/


/-- If `F : C ⥤ D` is a representably flat functor between small categories, then the functor
`Lan F.op` that takes presheaves over `C` to presheaves over `D` preserves finite limits.
-/
noncomputable instance lan_preservesFiniteLimits_of_flat (F : C ⥤ D) [RepresentablyFlat F] :
    PreservesFiniteLimits (F.op.lan : _ ⥤ Dᵒᵖ ⥤ E) := by
  /-
    C D : Type u₁
    inst✝⁹ : CategoryTheory.SmallCategory C
    inst✝⁸ : CategoryTheory.SmallCategory D
    E : Type u₂
    inst✝⁷ : CategoryTheory.Category.{u₁, u₂} E
    inst✝⁶ : CategoryTheory.ConcreteCategory E
    inst✝⁵ : CategoryTheory.Limits.HasLimits E
    inst✝⁴ : CategoryTheory.Limits.HasColimits E
    inst✝³ : CategoryTheory.Limits.ReflectsLimits (CategoryTheory.forget E)
    inst✝² : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget E)
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat F
    ⊢ CategoryTheory.Limits.PreservesFiniteLimits F.op.lan
  -/
  apply preservesFiniteLimits_of_preservesFiniteLimitsOfSize.{u₁}
  /-
    case h
    C D : Type u₁
    inst✝⁹ : CategoryTheory.SmallCategory C
    inst✝⁸ : CategoryTheory.SmallCategory D
    E : Type u₂
    inst✝⁷ : CategoryTheory.Category.{u₁, u₂} E
    inst✝⁶ : CategoryTheory.ConcreteCategory E
    inst✝⁵ : CategoryTheory.Limits.HasLimits E
    inst✝⁴ : CategoryTheory.Limits.HasColimits E
    inst✝³ : CategoryTheory.Limits.ReflectsLimits (CategoryTheory.forget E)
    inst✝² : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget E)
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat F
    ⊢ ∀ (J : Type u₁) {𝒥 : CategoryTheory.SmallCategory J}, CategoryTheory.FinCate …
  -/
  intro J _ _
  /-
    case h
    C D : Type u₁
    inst✝⁹ : CategoryTheory.SmallCategory C
    inst✝⁸ : CategoryTheory.SmallCategory D
    E : Type u₂
    inst✝⁷ : CategoryTheory.Category.{u₁, u₂} E
    inst✝⁶ : CategoryTheory.ConcreteCategory E
    inst✝⁵ : CategoryTheory.Limits.HasLimits E
    inst✝⁴ : CategoryTheory.Limits.HasColimits E
    inst✝³ : CategoryTheory.Limits.ReflectsLimits (CategoryTheory.forget E)
    inst✝² : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget E)
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat F
    J : Type u₁
    𝒥✝ : CategoryTheory.SmallCategory J
    x✝ : CategoryTheory.FinCategory J
    ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J F.op.lan
  -/
  apply preservesLimitsOfShape_of_evaluation (F.op.lan : (Cᵒᵖ ⥤ E) ⥤ Dᵒᵖ ⥤ E) J
  /-
    case h
    C D : Type u₁
    inst✝⁹ : CategoryTheory.SmallCategory C
    inst✝⁸ : CategoryTheory.SmallCategory D
    E : Type u₂
    inst✝⁷ : CategoryTheory.Category.{u₁, u₂} E
    inst✝⁶ : CategoryTheory.ConcreteCategory E
    inst✝⁵ : CategoryTheory.Limits.HasLimits E
    inst✝⁴ : CategoryTheory.Limits.HasColimits E
    inst✝³ : CategoryTheory.Limits.ReflectsLimits (CategoryTheory.forget E)
    inst✝² : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget E)
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat F
    J : Type u₁
    𝒥✝ : CategoryTheory.SmallCategory J
    x✝ : CategoryTheory.FinCategory J
    ⊢ ∀ (k : Opposite D), CategoryTheory.Limits.PreservesLimitsOfShape J (F.op.lan …
  -/
  intro K
  haveI : IsFiltered (CostructuredArrow F.op K) :=
    IsFiltered.of_equivalence (structuredArrowOpEquivalence F (unop K))
  /-
    case h
    C D : Type u₁
    inst✝⁹ : CategoryTheory.SmallCategory C
    inst✝⁸ : CategoryTheory.SmallCategory D
    E : Type u₂
    inst✝⁷ : CategoryTheory.Category.{u₁, u₂} E
    inst✝⁶ : CategoryTheory.ConcreteCategory E
    inst✝⁵ : CategoryTheory.Limits.HasLimits E
    inst✝⁴ : CategoryTheory.Limits.HasColimits E
    inst✝³ : CategoryTheory.Limits.ReflectsLimits (CategoryTheory.forget E)
    inst✝² : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝¹ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget E)
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat F
    J : Type u₁
    𝒥✝ : CategoryTheory.SmallCategory J
    x✝ : CategoryTheory.FinCategory J
    K : Opposite D
    this : CategoryTheory.IsFiltered (CategoryTheory.CostructuredArrow F.op K)
    ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J (F.op.lan.comp ((CategoryTheo …
  -/
  exact preservesLimitsOfShape_of_natIso (lanEvaluationIsoColim _ _ _).symm
  /-
    🎉 no goals
  -/


instance lan_flat_of_flat (F : C ⥤ D) [RepresentablyFlat F] :
    RepresentablyFlat (F.op.lan : _ ⥤ Dᵒᵖ ⥤ E) :=
  flat_of_preservesFiniteLimits _


instance lan_preservesFiniteLimits_of_preservesFiniteLimits (F : C ⥤ D)
    [PreservesFiniteLimits F] : PreservesFiniteLimits (F.op.lan : _ ⥤ Dᵒᵖ ⥤ E) := by
  /-
    C D : Type u₁
    inst✝¹⁰ : CategoryTheory.SmallCategory C
    inst✝⁹ : CategoryTheory.SmallCategory D
    E : Type u₂
    inst✝⁸ : CategoryTheory.Category.{u₁, u₂} E
    inst✝⁷ : CategoryTheory.ConcreteCategory E
    inst✝⁶ : CategoryTheory.Limits.HasLimits E
    inst✝⁵ : CategoryTheory.Limits.HasColimits E
    inst✝⁴ : CategoryTheory.Limits.ReflectsLimits (CategoryTheory.forget E)
    inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget E)
    inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.Limits.PreservesFiniteLimits F
    ⊢ CategoryTheory.Limits.PreservesFiniteLimits F.op.lan
  -/
  haveI := flat_of_preservesFiniteLimits F
  /-
    C D : Type u₁
    inst✝¹⁰ : CategoryTheory.SmallCategory C
    inst✝⁹ : CategoryTheory.SmallCategory D
    E : Type u₂
    inst✝⁸ : CategoryTheory.Category.{u₁, u₂} E
    inst✝⁷ : CategoryTheory.ConcreteCategory E
    inst✝⁶ : CategoryTheory.Limits.HasLimits E
    inst✝⁵ : CategoryTheory.Limits.HasColimits E
    inst✝⁴ : CategoryTheory.Limits.ReflectsLimits (CategoryTheory.forget E)
    inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget E)
    inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.Limits.PreservesFiniteLimits F
    this : CategoryTheory.RepresentablyFlat F
    ⊢ CategoryTheory.Limits.PreservesFiniteLimits F.op.lan
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem flat_iff_lan_flat (F : C ⥤ D) :
    RepresentablyFlat F ↔ RepresentablyFlat (F.op.lan : _ ⥤ Dᵒᵖ ⥤ Type u₁) :=
  ⟨fun _ => inferInstance, fun H => by
    /-
      C D : Type u₁
      inst✝² : CategoryTheory.SmallCategory C
      inst✝¹ : CategoryTheory.SmallCategory D
      inst✝ : CategoryTheory.Limits.HasFiniteLimits C
      F : CategoryTheory.Functor C D
      H : CategoryTheory.RepresentablyFlat F.op.lan
      ⊢ CategoryTheory.RepresentablyFlat F
    -/
    haveI := preservesFiniteLimits_of_flat (F.op.lan : _ ⥤ Dᵒᵖ ⥤ Type u₁)
    haveI : PreservesFiniteLimits F := by
      apply preservesFiniteLimits_of_preservesFiniteLimitsOfSize.{u₁}
      intros; apply preservesLimit_of_lan_preservesLimit
    /-
      C D : Type u₁
      inst✝² : CategoryTheory.SmallCategory C
      inst✝¹ : CategoryTheory.SmallCategory D
      inst✝ : CategoryTheory.Limits.HasFiniteLimits C
      F : CategoryTheory.Functor C D
      H : CategoryTheory.RepresentablyFlat F.op.lan
      this✝ : CategoryTheory.Limits.PreservesFiniteLimits F.op.lan
      this : CategoryTheory.Limits.PreservesFiniteLimits F
      ⊢ CategoryTheory.RepresentablyFlat F
    -/
    apply flat_of_preservesFiniteLimits⟩
    /-
      🎉 no goals
    -/


/-- If `C` is finitely complete, then `F : C ⥤ D` preserves finite limits iff
`Lan F.op : (Cᵒᵖ ⥤ Type*) ⥤ (Dᵒᵖ ⥤ Type*)` preserves finite limits.
-/
lemma preservesFiniteLimits_iff_lan_preservesFiniteLimits (F : C ⥤ D) :
    PreservesFiniteLimits F ↔ PreservesFiniteLimits (F.op.lan : _ ⥤ Dᵒᵖ ⥤ Type u₁) :=
  ⟨fun _ ↦ inferInstance,
    fun _ ↦ preservesFiniteLimits_of_preservesFiniteLimitsOfSize.{u₁} _
      (fun _ _ _ ↦ preservesLimit_of_lan_preservesLimit _ _)⟩



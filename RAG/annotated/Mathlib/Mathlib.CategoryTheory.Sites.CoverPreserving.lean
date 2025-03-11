/-- A functor `G : (C, J) ⥤ (D, K)` between sites is *cover-preserving*
if for all covering sieves `R` in `C`, `R.functorPushforward G` is a covering sieve in `D`.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed `@[nolint has_nonempty_instance]`
structure CoverPreserving (G : C ⥤ D) : Prop where
  cover_preserve : ∀ {U : C} {S : Sieve U} (_ : S ∈ J U), S.functorPushforward G ∈ K (G.obj U)


/-- The identity functor on a site is cover-preserving. -/
theorem idCoverPreserving : CoverPreserving J J (𝟭 _) :=
                /-
                  C : Type u₁
                  inst✝ : CategoryTheory.Category.{v₁, u₁} C
                  J : CategoryTheory.GrothendieckTopology C
                  U✝ : C
                  S✝ : CategoryTheory.Sieve U✝
                  hS : Membership.mem (J U✝) S✝
                  ⊢ Membership.mem (J ((CategoryTheory.Functor.id C).obj U✝)) (CategoryTheory.Si …
                -/
  ⟨fun hS => by simpa using hS⟩
                /-
                  🎉 no goals
                -/


/-- The composition of two cover-preserving functors is cover-preserving. -/
theorem CoverPreserving.comp {F} (hF : CoverPreserving J K F) {G} (hG : CoverPreserving K L G) :
    CoverPreserving J L (F ⋙ G) :=
  ⟨fun hS => by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      A : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} A
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology A
      F : CategoryTheory.Functor C D
      hF : CategoryTheory.CoverPreserving J K F
      G : CategoryTheory.Functor D A
      hG : CategoryTheory.CoverPreserving K L G
      U✝ : C
      S✝ : CategoryTheory.Sieve U✝
      hS : Membership.mem (J U✝) S✝
      ⊢ Membership.mem (L ((F.comp G).obj U✝)) (CategoryTheory.Sieve.functorPushforw …
    -/
    rw [Sieve.functorPushforward_comp]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      A : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} A
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology A
      F : CategoryTheory.Functor C D
      hF : CategoryTheory.CoverPreserving J K F
      G : CategoryTheory.Functor D A
      hG : CategoryTheory.CoverPreserving K L G
      U✝ : C
      S✝ : CategoryTheory.Sieve U✝
      hS : Membership.mem (J U✝) S✝
      ⊢ Membership.mem (L ((F.comp G).obj U✝)) (CategoryTheory.Sieve.functorPushforw …
    -/
    exact hG.cover_preserve (hF.cover_preserve hS)⟩
    /-
      🎉 no goals
    -/


/-- A functor `G : (C, J) ⥤ (D, K)` between sites is called compatible preserving if for each
compatible family of elements at `C` and valued in `G.op ⋙ ℱ`, and each commuting diagram
`f₁ ≫ G.map g₁ = f₂ ≫ G.map g₂`, `x g₁` and `x g₂` coincide when restricted via `fᵢ`.
This is actually stronger than merely preserving compatible families because of the definition of
`functorPushforward` used.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): linter not ported yet @[nolint has_nonempty_instance]
structure CompatiblePreserving (K : GrothendieckTopology D) (G : C ⥤ D) : Prop where
  compatible :
    ∀ (ℱ : Sheaf K (Type w)) {Z} {T : Presieve Z} {x : FamilyOfElements (G.op ⋙ ℱ.val) T}
      (_ : x.Compatible) {Y₁ Y₂} {X} (f₁ : X ⟶ G.obj Y₁) (f₂ : X ⟶ G.obj Y₂) {g₁ : Y₁ ⟶ Z}
      {g₂ : Y₂ ⟶ Z} (hg₁ : T g₁) (hg₂ : T g₂) (_ : f₁ ≫ G.map g₁ = f₂ ≫ G.map g₂),
      ℱ.val.map f₁.op (x g₁ hg₁) = ℱ.val.map f₂.op (x g₂ hg₂)


/-- `CompatiblePreserving` functors indeed preserve compatible families. -/
theorem Presieve.FamilyOfElements.Compatible.functorPushforward :
    (x.functorPushforward G).Compatible := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    hG : CategoryTheory.CompatiblePreserving K G
    ℱ : CategoryTheory.Sheaf K (Type w)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    h : x.Compatible
    ⊢ (CategoryTheory.Presieve.FamilyOfElements.functorPushforward G x).Compatible
  -/
  rintro Z₁ Z₂ W g₁ g₂ f₁' f₂' H₁ H₂ eq
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    hG : CategoryTheory.CompatiblePreserving K G
    ℱ : CategoryTheory.Sheaf K (Type w)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    h : x.Compatible
    Z₁ Z₂ W : D
    g₁ : Quiver.Hom W Z₁
    g₂ : Quiver.Hom W Z₂
    f₁' : Quiver.Hom Z₁ (G.obj Z)
    f₂' : Quiver.Hom Z₂ (G.obj Z)
    H₁ : CategoryTheory.Presieve.functorPushforward G T f₁'
    H₂ : CategoryTheory.Presieve.functorPushforward G T f₂'
    eq : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁') (CategoryTheory.CategorySt …
    ⊢ Eq (ℱ.val.map g₁.op (CategoryTheory.Presieve.FamilyOfElements.functorPushfor …
  -/
  unfold FamilyOfElements.functorPushforward
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    hG : CategoryTheory.CompatiblePreserving K G
    ℱ : CategoryTheory.Sheaf K (Type w)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    h : x.Compatible
    Z₁ Z₂ W : D
    g₁ : Quiver.Hom W Z₁
    g₂ : Quiver.Hom W Z₂
    f₁' : Quiver.Hom Z₁ (G.obj Z)
    f₂' : Quiver.Hom Z₂ (G.obj Z)
    H₁ : CategoryTheory.Presieve.functorPushforward G T f₁'
    H₂ : CategoryTheory.Presieve.functorPushforward G T f₂'
    eq : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁') (CategoryTheory.CategorySt …
    ⊢ Eq (ℱ.val.map g₁.op (CategoryTheory.Presieve.FunctorPushforwardStructure.cas …
  -/
  rcases getFunctorPushforwardStructure H₁ with ⟨X₁, f₁, h₁, hf₁, rfl⟩
  /-
    case mk
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    hG : CategoryTheory.CompatiblePreserving K G
    ℱ : CategoryTheory.Sheaf K (Type w)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    h : x.Compatible
    Z₁ Z₂ W : D
    g₁ : Quiver.Hom W Z₁
    g₂ : Quiver.Hom W Z₂
    f₂' : Quiver.Hom Z₂ (G.obj Z)
    H₂ : CategoryTheory.Presieve.functorPushforward G T f₂'
    X₁ : C
    f₁ : Quiver.Hom X₁ Z
    h₁ : Quiver.Hom Z₁ (G.obj X₁)
    hf₁ : T f₁
    H₁ : CategoryTheory.Presieve.functorPushforward G T (CategoryTheory.CategorySt …
    eq : Eq (CategoryTheory.CategoryStruct.comp g₁ (CategoryTheory.CategoryStruct. …
    ⊢ Eq (ℱ.val.map g₁.op (CategoryTheory.Presieve.FunctorPushforwardStructure.cas …
  -/
  rcases getFunctorPushforwardStructure H₂ with ⟨X₂, f₂, h₂, hf₂, rfl⟩
  suffices ℱ.val.map (g₁ ≫ h₁).op (x f₁ hf₁) = ℱ.val.map (g₂ ≫ h₂).op (x f₂ hf₂) by
    simpa using this
  /-
    case mk.mk
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    hG : CategoryTheory.CompatiblePreserving K G
    ℱ : CategoryTheory.Sheaf K (Type w)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    h : x.Compatible
    Z₁ Z₂ W : D
    g₁ : Quiver.Hom W Z₁
    g₂ : Quiver.Hom W Z₂
    X₁ : C
    f₁ : Quiver.Hom X₁ Z
    h₁ : Quiver.Hom Z₁ (G.obj X₁)
    hf₁ : T f₁
    H₁ : CategoryTheory.Presieve.functorPushforward G T (CategoryTheory.CategorySt …
    X₂ : C
    f₂ : Quiver.Hom X₂ Z
    h₂ : Quiver.Hom Z₂ (G.obj X₂)
    hf₂ : T f₂
    H₂ : CategoryTheory.Presieve.functorPushforward G T (CategoryTheory.CategorySt …
    eq : Eq (CategoryTheory.CategoryStruct.comp g₁ (CategoryTheory.CategoryStruct. …
    ⊢ Eq (ℱ.val.map (CategoryTheory.CategoryStruct.comp g₁ h₁).op (x f₁ hf₁)) (ℱ.v …
  -/
  apply hG.compatible ℱ h _ _ hf₁ hf₂
  /-
    case mk.mk
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    hG : CategoryTheory.CompatiblePreserving K G
    ℱ : CategoryTheory.Sheaf K (Type w)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    h : x.Compatible
    Z₁ Z₂ W : D
    g₁ : Quiver.Hom W Z₁
    g₂ : Quiver.Hom W Z₂
    X₁ : C
    f₁ : Quiver.Hom X₁ Z
    h₁ : Quiver.Hom Z₁ (G.obj X₁)
    hf₁ : T f₁
    H₁ : CategoryTheory.Presieve.functorPushforward G T (CategoryTheory.CategorySt …
    X₂ : C
    f₂ : Quiver.Hom X₂ Z
    h₂ : Quiver.Hom Z₂ (G.obj X₂)
    hf₂ : T f₂
    H₂ : CategoryTheory.Presieve.functorPushforward G T (CategoryTheory.CategorySt …
    eq : Eq (CategoryTheory.CategoryStruct.comp g₁ (CategoryTheory.CategoryStruct. …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp g …
  -/
  simpa using eq
  /-
    🎉 no goals
  -/


@[simp]
theorem CompatiblePreserving.apply_map {Y : C} {f : Y ⟶ Z} (hf : T f) :
    x.functorPushforward G (G.map f) (image_mem_functorPushforward G T hf) = x f hf := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    hG : CategoryTheory.CompatiblePreserving K G
    ℱ : CategoryTheory.Sheaf K (Type w)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    h : x.Compatible
    Y : C
    f : Quiver.Hom Y Z
    hf : T f
    ⊢ Eq (CategoryTheory.Presieve.FamilyOfElements.functorPushforward G x (G.map f …
  -/
  unfold FamilyOfElements.functorPushforward
  rcases getFunctorPushforwardStructure (image_mem_functorPushforward G T hf) with
    ⟨X, g, f', hg, eq⟩
  /-
    case mk
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    hG : CategoryTheory.CompatiblePreserving K G
    ℱ : CategoryTheory.Sheaf K (Type w)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    h : x.Compatible
    Y : C
    f : Quiver.Hom Y Z
    hf : T f
    X : C
    g : Quiver.Hom X Z
    f' : Quiver.Hom (G.obj Y) (G.obj X)
    hg : T g
    eq : Eq (G.map f) (CategoryTheory.CategoryStruct.comp f' (G.map g))
    ⊢ Eq (CategoryTheory.Presieve.FunctorPushforwardStructure.casesOn { preobj :=  …
  -/
  simpa using hG.compatible ℱ h f' (𝟙 _) hg hf (by simp [eq])
  /-
    🎉 no goals
  -/


theorem compatiblePreservingOfFlat {C : Type u₁} [Category.{v₁} C] {D : Type u₁} [Category.{v₁} D]
    (K : GrothendieckTopology D) (G : C ⥤ D) [RepresentablyFlat G] : CompatiblePreserving K G := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat G
    ⊢ CategoryTheory.CompatiblePreserving K G
  -/
  constructor
  /-
    case compatible
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat G
    ⊢ ∀ (ℱ : CategoryTheory.Sheaf K (Type u_1)) {Z : C} {T : CategoryTheory.Presie …
  -/
  intro ℱ Z T x hx Y₁ Y₂ X f₁ f₂ g₁ g₂ hg₁ hg₂ e
  -- First, `f₁` and `f₂` form a cone over `cospan g₁ g₂ ⋙ u`.
  let c : Cone (cospan g₁ g₂ ⋙ G) :=
    (Cones.postcompose (diagramIsoCospan (cospan g₁ g₂ ⋙ G)).inv).obj (PullbackCone.mk f₁ f₂ e)
  /-
    This can then be viewed as a cospan of structured arrows, and we may obtain an arbitrary cone
    over it since `StructuredArrow W u` is cofiltered.
    Then, it suffices to prove that it is compatible when restricted onto `u(c'.X.right)`.
    -/
  /-
    case compatible
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat G
    ℱ : CategoryTheory.Sheaf K (Type u_1)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    hx : x.Compatible
    Y₁ Y₂ : C
    X : D
    f₁ : Quiver.Hom X (G.obj Y₁)
    f₂ : Quiver.Hom X (G.obj Y₂)
    g₁ : Quiver.Hom Y₁ Z
    g₂ : Quiver.Hom Y₂ Z
    hg₁ : T g₁
    hg₂ : T g₂
    e : Eq (CategoryTheory.CategoryStruct.comp f₁ (G.map g₁)) (CategoryTheory.Cate …
    c : CategoryTheory.Limits.Cone ((CategoryTheory.Limits.cospan g₁ g₂).comp G) : …
    ⊢ Eq (ℱ.val.map f₁.op (x g₁ hg₁)) (ℱ.val.map f₂.op (x g₂ hg₂))
  -/
  let c' := IsCofiltered.cone (c.toStructuredArrow ⋙ StructuredArrow.pre _ _ _)
  have eq₁ : f₁ = (c'.pt.hom ≫ G.map (c'.π.app left).right) ≫ eqToHom (by simp) := by
    erw [← (c'.π.app left).w]
    dsimp [c]
    simp
  have eq₂ : f₂ = (c'.pt.hom ≫ G.map (c'.π.app right).right) ≫ eqToHom (by simp) := by
    erw [← (c'.π.app right).w]
    dsimp [c]
    simp
  /-
    case compatible
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat G
    ℱ : CategoryTheory.Sheaf K (Type u_1)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    hx : x.Compatible
    Y₁ Y₂ : C
    X : D
    f₁ : Quiver.Hom X (G.obj Y₁)
    f₂ : Quiver.Hom X (G.obj Y₂)
    g₁ : Quiver.Hom Y₁ Z
    g₂ : Quiver.Hom Y₂ Z
    hg₁ : T g₁
    hg₂ : T g₂
    e : Eq (CategoryTheory.CategoryStruct.comp f₁ (G.map g₁)) (CategoryTheory.Cate …
    c : CategoryTheory.Limits.Cone ((CategoryTheory.Limits.cospan g₁ g₂).comp G) : …
    c' : CategoryTheory.Limits.Cone (c.toStructuredArrow.comp (CategoryTheory.Stru …
    eq₁ : Eq f₁ (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct …
    eq₂ : Eq f₂ (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct …
    ⊢ Eq (ℱ.val.map f₁.op (x g₁ hg₁)) (ℱ.val.map f₂.op (x g₂ hg₂))
  -/
  conv_lhs => rw [eq₁]
  /-
    case compatible
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat G
    ℱ : CategoryTheory.Sheaf K (Type u_1)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    hx : x.Compatible
    Y₁ Y₂ : C
    X : D
    f₁ : Quiver.Hom X (G.obj Y₁)
    f₂ : Quiver.Hom X (G.obj Y₂)
    g₁ : Quiver.Hom Y₁ Z
    g₂ : Quiver.Hom Y₂ Z
    hg₁ : T g₁
    hg₂ : T g₂
    e : Eq (CategoryTheory.CategoryStruct.comp f₁ (G.map g₁)) (CategoryTheory.Cate …
    c : CategoryTheory.Limits.Cone ((CategoryTheory.Limits.cospan g₁ g₂).comp G) : …
    c' : CategoryTheory.Limits.Cone (c.toStructuredArrow.comp (CategoryTheory.Stru …
    eq₁ : Eq f₁ (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct …
    eq₂ : Eq f₂ (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct …
    ⊢ Eq (ℱ.val.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategorySt …
  -/
  conv_rhs => rw [eq₂]
  /-
    case compatible
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat G
    ℱ : CategoryTheory.Sheaf K (Type u_1)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    hx : x.Compatible
    Y₁ Y₂ : C
    X : D
    f₁ : Quiver.Hom X (G.obj Y₁)
    f₂ : Quiver.Hom X (G.obj Y₂)
    g₁ : Quiver.Hom Y₁ Z
    g₂ : Quiver.Hom Y₂ Z
    hg₁ : T g₁
    hg₂ : T g₂
    e : Eq (CategoryTheory.CategoryStruct.comp f₁ (G.map g₁)) (CategoryTheory.Cate …
    c : CategoryTheory.Limits.Cone ((CategoryTheory.Limits.cospan g₁ g₂).comp G) : …
    c' : CategoryTheory.Limits.Cone (c.toStructuredArrow.comp (CategoryTheory.Stru …
    eq₁ : Eq f₁ (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct …
    eq₂ : Eq f₂ (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct …
    ⊢ Eq (ℱ.val.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategorySt …
  -/
  simp only [c, op_comp, Functor.map_comp, types_comp_apply, eqToHom_op, eqToHom_map]
  /-
    case compatible
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat G
    ℱ : CategoryTheory.Sheaf K (Type u_1)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    hx : x.Compatible
    Y₁ Y₂ : C
    X : D
    f₁ : Quiver.Hom X (G.obj Y₁)
    f₂ : Quiver.Hom X (G.obj Y₂)
    g₁ : Quiver.Hom Y₁ Z
    g₂ : Quiver.Hom Y₂ Z
    hg₁ : T g₁
    hg₂ : T g₂
    e : Eq (CategoryTheory.CategoryStruct.comp f₁ (G.map g₁)) (CategoryTheory.Cate …
    c : CategoryTheory.Limits.Cone ((CategoryTheory.Limits.cospan g₁ g₂).comp G) : …
    c' : CategoryTheory.Limits.Cone (c.toStructuredArrow.comp (CategoryTheory.Stru …
    eq₁ : Eq f₁ (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct …
    eq₂ : Eq f₂ (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct …
    ⊢ Eq (ℱ.val.map c'.pt.hom.op (ℱ.val.map (G.map (c'.π.app CategoryTheory.Limits …
  -/
  apply congr_arg -- Porting note: was `congr 1` which for some reason doesn't do anything here
  -- despite goal being of the form f a = f b, with f=`ℱ.val.map (Quiver.Hom.op c'.pt.hom)`
  /-
    Since everything now falls in the image of `u`,
    the result follows from the compatibility of `x` in the image of `u`.
    -/
  /-
    case compatible.h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat G
    ℱ : CategoryTheory.Sheaf K (Type u_1)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    hx : x.Compatible
    Y₁ Y₂ : C
    X : D
    f₁ : Quiver.Hom X (G.obj Y₁)
    f₂ : Quiver.Hom X (G.obj Y₂)
    g₁ : Quiver.Hom Y₁ Z
    g₂ : Quiver.Hom Y₂ Z
    hg₁ : T g₁
    hg₂ : T g₂
    e : Eq (CategoryTheory.CategoryStruct.comp f₁ (G.map g₁)) (CategoryTheory.Cate …
    c : CategoryTheory.Limits.Cone ((CategoryTheory.Limits.cospan g₁ g₂).comp G) : …
    c' : CategoryTheory.Limits.Cone (c.toStructuredArrow.comp (CategoryTheory.Stru …
    eq₁ : Eq f₁ (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct …
    eq₂ : Eq f₂ (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct …
    ⊢ Eq (ℱ.val.map (G.map (c'.π.app CategoryTheory.Limits.WalkingCospan.left).rig …
  -/
  injection c'.π.naturality WalkingCospan.Hom.inl with _ e₁
  /-
    case compatible.h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat G
    ℱ : CategoryTheory.Sheaf K (Type u_1)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    hx : x.Compatible
    Y₁ Y₂ : C
    X : D
    f₁ : Quiver.Hom X (G.obj Y₁)
    f₂ : Quiver.Hom X (G.obj Y₂)
    g₁ : Quiver.Hom Y₁ Z
    g₂ : Quiver.Hom Y₂ Z
    hg₁ : T g₁
    hg₂ : T g₂
    e : Eq (CategoryTheory.CategoryStruct.comp f₁ (G.map g₁)) (CategoryTheory.Cate …
    c : CategoryTheory.Limits.Cone ((CategoryTheory.Limits.cospan g₁ g₂).comp G) : …
    c' : CategoryTheory.Limits.Cone (c.toStructuredArrow.comp (CategoryTheory.Stru …
    eq₁ : Eq f₁ (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct …
    eq₂ : Eq f₂ (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct …
    left_eq✝ : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.co …
    e₁ : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const Ca …
    ⊢ Eq (ℱ.val.map (G.map (c'.π.app CategoryTheory.Limits.WalkingCospan.left).rig …
  -/
  injection c'.π.naturality WalkingCospan.Hom.inr with _ e₂
  /-
    case compatible.h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.RepresentablyFlat G
    ℱ : CategoryTheory.Sheaf K (Type u_1)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    hx : x.Compatible
    Y₁ Y₂ : C
    X : D
    f₁ : Quiver.Hom X (G.obj Y₁)
    f₂ : Quiver.Hom X (G.obj Y₂)
    g₁ : Quiver.Hom Y₁ Z
    g₂ : Quiver.Hom Y₂ Z
    hg₁ : T g₁
    hg₂ : T g₂
    e : Eq (CategoryTheory.CategoryStruct.comp f₁ (G.map g₁)) (CategoryTheory.Cate …
    c : CategoryTheory.Limits.Cone ((CategoryTheory.Limits.cospan g₁ g₂).comp G) : …
    c' : CategoryTheory.Limits.Cone (c.toStructuredArrow.comp (CategoryTheory.Stru …
    eq₁ : Eq f₁ (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct …
    eq₂ : Eq f₂ (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct …
    left_eq✝¹ : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.c …
    e₁ : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const Ca …
    left_eq✝ : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.co …
    e₂ : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const Ca …
    ⊢ Eq (ℱ.val.map (G.map (c'.π.app CategoryTheory.Limits.WalkingCospan.left).rig …
  -/
  exact hx (c'.π.app left).right (c'.π.app right).right hg₁ hg₂ (e₁.symm.trans e₂)
  /-
    🎉 no goals
  -/


theorem compatiblePreservingOfDownwardsClosed (F : C ⥤ D) [F.Full] [F.Faithful]
    (hF : ∀ {c : C} {d : D} (_ : d ⟶ F.obj c), Σc', F.obj c' ≅ d) : CompatiblePreserving K F := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    hF : {c : C} → {d : D} → Quiver.Hom d (F.obj c) → Sigma fun c' => CategoryTheo …
    ⊢ CategoryTheory.CompatiblePreserving K F
  -/
  constructor
  /-
    case compatible
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    hF : {c : C} → {d : D} → Quiver.Hom d (F.obj c) → Sigma fun c' => CategoryTheo …
    ⊢ ∀ (ℱ : CategoryTheory.Sheaf K (Type u_1)) {Z : C} {T : CategoryTheory.Presie …
  -/
  introv hx he
  /-
    case compatible
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    hF : {c : C} → {d : D} → Quiver.Hom d (F.obj c) → Sigma fun c' => CategoryTheo …
    ℱ : CategoryTheory.Sheaf K (Type u_1)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (F.op.comp ℱ.val) T
    hx : x.Compatible
    Y₁ Y₂ : C
    X : D
    f₁ : Quiver.Hom X (F.obj Y₁)
    f₂ : Quiver.Hom X (F.obj Y₂)
    g₁ : Quiver.Hom Y₁ Z
    g₂ : Quiver.Hom Y₂ Z
    hg₁ : T g₁
    hg₂ : T g₂
    he : Eq (CategoryTheory.CategoryStruct.comp f₁ (F.map g₁)) (CategoryTheory.Cat …
    ⊢ Eq (ℱ.val.map f₁.op (x g₁ hg₁)) (ℱ.val.map f₂.op (x g₂ hg₂))
  -/
  obtain ⟨X', e⟩ := hF f₁
  /-
    case compatible.mk
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    hF : {c : C} → {d : D} → Quiver.Hom d (F.obj c) → Sigma fun c' => CategoryTheo …
    ℱ : CategoryTheory.Sheaf K (Type u_1)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (F.op.comp ℱ.val) T
    hx : x.Compatible
    Y₁ Y₂ : C
    X : D
    f₁ : Quiver.Hom X (F.obj Y₁)
    f₂ : Quiver.Hom X (F.obj Y₂)
    g₁ : Quiver.Hom Y₁ Z
    g₂ : Quiver.Hom Y₂ Z
    hg₁ : T g₁
    hg₂ : T g₂
    he : Eq (CategoryTheory.CategoryStruct.comp f₁ (F.map g₁)) (CategoryTheory.Cat …
    X' : C
    e : CategoryTheory.Iso (F.obj X') X
    ⊢ Eq (ℱ.val.map f₁.op (x g₁ hg₁)) (ℱ.val.map f₂.op (x g₂ hg₂))
  -/
  apply (ℱ.1.mapIso e.op).toEquiv.injective
  /-
    case compatible.mk.a
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    hF : {c : C} → {d : D} → Quiver.Hom d (F.obj c) → Sigma fun c' => CategoryTheo …
    ℱ : CategoryTheory.Sheaf K (Type u_1)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (F.op.comp ℱ.val) T
    hx : x.Compatible
    Y₁ Y₂ : C
    X : D
    f₁ : Quiver.Hom X (F.obj Y₁)
    f₂ : Quiver.Hom X (F.obj Y₂)
    g₁ : Quiver.Hom Y₁ Z
    g₂ : Quiver.Hom Y₂ Z
    hg₁ : T g₁
    hg₂ : T g₂
    he : Eq (CategoryTheory.CategoryStruct.comp f₁ (F.map g₁)) (CategoryTheory.Cat …
    X' : C
    e : CategoryTheory.Iso (F.obj X') X
    ⊢ Eq ((ℱ.val.mapIso e.op).toEquiv (ℱ.val.map f₁.op (x g₁ hg₁))) ((ℱ.val.mapIso …
  -/
  simp only [Iso.op_hom, Iso.toEquiv_fun, ℱ.1.mapIso_hom, ← FunctorToTypes.map_comp_apply]
  simpa using
    hx (F.preimage <| e.hom ≫ f₁) (F.preimage <| e.hom ≫ f₂) hg₁ hg₂
      (F.map_injective <| by simpa using he)


/-- If `F` is cover-preserving and compatible-preserving,
then `F` is a continuous functor.

This result is basically <https://stacks.math.columbia.edu/tag/00WW>.
-/
lemma Functor.isContinuous_of_coverPreserving (hF₁ : CompatiblePreserving.{w} K F)
    (hF₂ : CoverPreserving J K F) : Functor.IsContinuous.{w} F J K where
  op_comp_isSheaf_of_types G X S hS x hx := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      hF₁ : CategoryTheory.CompatiblePreserving K F
      hF₂ : CategoryTheory.CoverPreserving J K F
      G : CategoryTheory.Sheaf K (Type w)
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J X) S
      x : CategoryTheory.Presieve.FamilyOfElements (F.op.comp G.val) S.arrows
      hx : x.Compatible
      ⊢ ExistsUnique fun t => x.IsAmalgamation t
    -/
    apply existsUnique_of_exists_of_unique
      /-
        case hex
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        hF₁ : CategoryTheory.CompatiblePreserving K F
        hF₂ : CategoryTheory.CoverPreserving J K F
        G : CategoryTheory.Sheaf K (Type w)
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem (J X) S
        x : CategoryTheory.Presieve.FamilyOfElements (F.op.comp G.val) S.arrows
        hx : x.Compatible
        ⊢ Exists fun x_1 => x.IsAmalgamation x_1
      -/
    · have H := (isSheaf_iff_isSheaf_of_type _ _).1 G.2 _ (hF₂.cover_preserve hS)
      exact ⟨H.amalgamate (x.functorPushforward F) (hx.functorPushforward hF₁),
        fun V f hf => (H.isAmalgamation (hx.functorPushforward hF₁) (F.map f) _).trans
          (hF₁.apply_map _ hx hf)⟩
      /-
        case hunique
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        hF₁ : CategoryTheory.CompatiblePreserving K F
        hF₂ : CategoryTheory.CoverPreserving J K F
        G : CategoryTheory.Sheaf K (Type w)
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem (J X) S
        x : CategoryTheory.Presieve.FamilyOfElements (F.op.comp G.val) S.arrows
        hx : x.Compatible
        ⊢ ∀ (y₁ y₂ : (F.op.comp G.val).obj { unop := X }), x.IsAmalgamation y₁ → x.IsA …
      -/
    · intro y₁ y₂ hy₁ hy₂
      apply (Presieve.isSeparated_of_isSheaf _ _ ((isSheaf_iff_isSheaf_of_type _ _).1 G.2) _
        (hF₂.cover_preserve hS)).ext
      /-
        case hunique
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        hF₁ : CategoryTheory.CompatiblePreserving K F
        hF₂ : CategoryTheory.CoverPreserving J K F
        G : CategoryTheory.Sheaf K (Type w)
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem (J X) S
        x : CategoryTheory.Presieve.FamilyOfElements (F.op.comp G.val) S.arrows
        hx : x.Compatible
        y₁ y₂ : (F.op.comp G.val).obj { unop := X }
        hy₁ : x.IsAmalgamation y₁
        hy₂ : x.IsAmalgamation y₂
        ⊢ ∀ ⦃Y : D⦄ ⦃f : Quiver.Hom Y (F.obj X)⦄, (CategoryTheory.Sieve.functorPushfor …
      -/
      rintro Y _ ⟨Z, g, h, hg, rfl⟩
      /-
        case hunique.intro.intro.intro.intro
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        hF₁ : CategoryTheory.CompatiblePreserving K F
        hF₂ : CategoryTheory.CoverPreserving J K F
        G : CategoryTheory.Sheaf K (Type w)
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem (J X) S
        x : CategoryTheory.Presieve.FamilyOfElements (F.op.comp G.val) S.arrows
        hx : x.Compatible
        y₁ y₂ : (F.op.comp G.val).obj { unop := X }
        hy₁ : x.IsAmalgamation y₁
        hy₂ : x.IsAmalgamation y₂
        Y : D
        Z : C
        g : Quiver.Hom Z X
        h : Quiver.Hom Y (F.obj Z)
        hg : S.arrows g
        ⊢ Eq (G.val.map (CategoryTheory.CategoryStruct.comp h (F.map g)).op y₁) (G.val …
      -/
      dsimp
      /-
        case hunique.intro.intro.intro.intro
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        hF₁ : CategoryTheory.CompatiblePreserving K F
        hF₂ : CategoryTheory.CoverPreserving J K F
        G : CategoryTheory.Sheaf K (Type w)
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem (J X) S
        x : CategoryTheory.Presieve.FamilyOfElements (F.op.comp G.val) S.arrows
        hx : x.Compatible
        y₁ y₂ : (F.op.comp G.val).obj { unop := X }
        hy₁ : x.IsAmalgamation y₁
        hy₂ : x.IsAmalgamation y₂
        Y : D
        Z : C
        g : Quiver.Hom Z X
        h : Quiver.Hom Y (F.obj Z)
        hg : S.arrows g
        ⊢ Eq (G.val.map (CategoryTheory.CategoryStruct.comp (F.map g).op h.op) y₁) (G. …
      -/
      simp only [Functor.map_comp, types_comp_apply]
      /-
        case hunique.intro.intro.intro.intro
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        hF₁ : CategoryTheory.CompatiblePreserving K F
        hF₂ : CategoryTheory.CoverPreserving J K F
        G : CategoryTheory.Sheaf K (Type w)
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem (J X) S
        x : CategoryTheory.Presieve.FamilyOfElements (F.op.comp G.val) S.arrows
        hx : x.Compatible
        y₁ y₂ : (F.op.comp G.val).obj { unop := X }
        hy₁ : x.IsAmalgamation y₁
        hy₂ : x.IsAmalgamation y₂
        Y : D
        Z : C
        g : Quiver.Hom Z X
        h : Quiver.Hom Y (F.obj Z)
        hg : S.arrows g
        ⊢ Eq (G.val.map h.op (G.val.map (F.map g).op y₁)) (G.val.map h.op (G.val.map ( …
      -/
      have H := (hy₁ g hg).trans (hy₂ g hg).symm
      /-
        case hunique.intro.intro.intro.intro
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        hF₁ : CategoryTheory.CompatiblePreserving K F
        hF₂ : CategoryTheory.CoverPreserving J K F
        G : CategoryTheory.Sheaf K (Type w)
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem (J X) S
        x : CategoryTheory.Presieve.FamilyOfElements (F.op.comp G.val) S.arrows
        hx : x.Compatible
        y₁ y₂ : (F.op.comp G.val).obj { unop := X }
        hy₁ : x.IsAmalgamation y₁
        hy₂ : x.IsAmalgamation y₂
        Y : D
        Z : C
        g : Quiver.Hom Z X
        h : Quiver.Hom Y (F.obj Z)
        hg : S.arrows g
        H : Eq ((F.op.comp G.val).map g.op y₁) ((F.op.comp G.val).map g.op y₂)
        ⊢ Eq (G.val.map h.op (G.val.map (F.map g).op y₁)) (G.val.map h.op (G.val.map ( …
      -/
      dsimp at H
      /-
        case hunique.intro.intro.intro.intro
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        F : CategoryTheory.Functor C D
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        hF₁ : CategoryTheory.CompatiblePreserving K F
        hF₂ : CategoryTheory.CoverPreserving J K F
        G : CategoryTheory.Sheaf K (Type w)
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem (J X) S
        x : CategoryTheory.Presieve.FamilyOfElements (F.op.comp G.val) S.arrows
        hx : x.Compatible
        y₁ y₂ : (F.op.comp G.val).obj { unop := X }
        hy₁ : x.IsAmalgamation y₁
        hy₂ : x.IsAmalgamation y₂
        Y : D
        Z : C
        g : Quiver.Hom Z X
        h : Quiver.Hom Y (F.obj Z)
        hg : S.arrows g
        H : Eq (G.val.map (F.map g).op y₁) (G.val.map (F.map g).op y₂)
        ⊢ Eq (G.val.map h.op (G.val.map (F.map g).op y₁)) (G.val.map h.op (G.val.map ( …
      -/
      rw [H]
      /-
        🎉 no goals
      -/



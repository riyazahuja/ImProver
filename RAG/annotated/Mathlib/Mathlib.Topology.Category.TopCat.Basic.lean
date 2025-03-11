/-- The category of topological spaces and continuous maps. -/
@[to_additive existing TopCat]
def TopCat : Type (u + 1) :=
  Bundled TopologicalSpace


instance bundledHom : BundledHom @ContinuousMap where
  toFun := @ContinuousMap.toFun
  id := @ContinuousMap.id
  comp := @ContinuousMap.comp


deriving instance LargeCategory for TopCat

-- Porting note: currently no derive handler for ConcreteCategory
-- see https://github.com/leanprover-community/mathlib4/issues/5020

instance concreteCategory : ConcreteCategory TopCat :=
  inferInstanceAs <| ConcreteCategory (Bundled TopologicalSpace)


instance : CoeSort TopCat Type* where
  coe X := X.α


instance topologicalSpaceUnbundled (X : TopCat) : TopologicalSpace X :=
  X.str


instance instFunLike (X Y : TopCat) : FunLike (X ⟶ Y) X Y :=
  inferInstanceAs <| FunLike C(X, Y) X Y


instance instContinuousMapClass (X Y : TopCat) : ContinuousMapClass (X ⟶ Y) X Y :=
  inferInstanceAs <| ContinuousMapClass C(X, Y) X Y


@[simp]
theorem id_app (X : TopCat.{u}) (x : ↑X) : (𝟙 X : X ⟶ X) x = x := rfl


@[simp]
theorem comp_app {X Y Z : TopCat.{u}} (f : X ⟶ Y) (g : Y ⟶ Z) (x : X) :
    (f ≫ g : X → Z) x = g (f x) := rfl


@[simp] theorem coe_id (X : TopCat.{u}) : (𝟙 X : X → X) = id := rfl


@[simp] theorem coe_comp {X Y Z : TopCat.{u}} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g : X → Z) = g ∘ f := rfl


@[simp]
lemma hom_inv_id_apply {X Y : TopCat} (f : X ≅ Y) (x : X) : f.inv (f.hom x) = x :=
  DFunLike.congr_fun f.hom_inv_id x


@[simp]
lemma inv_hom_id_apply {X Y : TopCat} (f : X ≅ Y) (y : Y) : f.hom (f.inv y) = y :=
  DFunLike.congr_fun f.inv_hom_id y


/-- Construct a bundled `Top` from the underlying type and the typeclass. -/
def of (X : Type u) [TopologicalSpace X] : TopCat :=
  -- Porting note: needed to call inferInstance
  ⟨X, inferInstance⟩


instance topologicalSpace_coe (X : TopCat) : TopologicalSpace X :=
  X.str

-- Porting note: cannot see through forget; made reducible to get closer to Lean 3 behavior

@[instance] abbrev topologicalSpace_forget
    (X : TopCat) : TopologicalSpace <| (forget TopCat).obj X :=
  X.str


@[simp]
theorem coe_of (X : Type u) [TopologicalSpace X] : (of X : Type u) = X := rfl


/--
Replace a function coercion for a morphism `TopCat.of X ⟶ TopCat.of Y` with the definitionally
equal function coercion for a continuous map `C(X, Y)`.
-/
@[simp] theorem coe_of_of {X Y : Type u} [TopologicalSpace X] [TopologicalSpace Y]
    {f : C(X, Y)} {x} :
    @DFunLike.coe (TopCat.of X ⟶ TopCat.of Y) ((CategoryTheory.forget TopCat).obj (TopCat.of X))
      (fun _ ↦ (CategoryTheory.forget TopCat).obj (TopCat.of Y)) ConcreteCategory.instFunLike
      f x =
    @DFunLike.coe C(X, Y) X
      (fun _ ↦ Y) _
      f x :=
  rfl


instance inhabited : Inhabited TopCat :=
  ⟨TopCat.of Empty⟩

-- Porting note: added to ease the port of `AlgebraicTopology.TopologicalSimplex`

lemma hom_apply {X Y : TopCat} (f : X ⟶ Y) (x : X) : f x = ContinuousMap.toFun f x := rfl


/-- The discrete topology on any type. -/
def discrete : Type u ⥤ TopCat.{u} where
  obj X := ⟨X , ⊥⟩
  map f := @ContinuousMap.mk _ _ ⊥ ⊥ f continuous_bot


instance {X : Type u} : DiscreteTopology (discrete.obj X) :=
  ⟨rfl⟩


/-- The trivial topology on any type. -/
def trivial : Type u ⥤ TopCat.{u} where
  obj X := ⟨X, ⊤⟩
  map f := @ContinuousMap.mk _ _ ⊤ ⊤ f continuous_top


/-- Any homeomorphisms induces an isomorphism in `Top`. -/
@[simps]
def isoOfHomeo {X Y : TopCat.{u}} (f : X ≃ₜ Y) : X ≅ Y where
  -- Porting note: previously ⟨f⟩ for hom (inv) and tidy closed proofs
  hom := (f : C(X, Y))
  inv := (f.symm : C(Y, X))
                   /-
                     X Y : TopCat
                     f : Homeomorph ↑X ↑Y
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp ↑f ↑f.symm) (CategoryTheory.CategoryS …
                   -/
  hom_inv_id := by ext; exact f.symm_apply_apply _
                        /-
                          🎉 no goals
                        -/
                   /-
                     X Y : TopCat
                     f : Homeomorph ↑X ↑Y
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp ↑f.symm ↑f) (CategoryTheory.CategoryS …
                   -/
  inv_hom_id := by ext; exact f.apply_symm_apply _
                        /-
                          🎉 no goals
                        -/


/-- Any isomorphism in `Top` induces a homeomorphism. -/
@[simps]
def homeoOfIso {X Y : TopCat.{u}} (f : X ≅ Y) : X ≃ₜ Y where
  toFun := f.hom
  invFun := f.inv
                   /-
                     X Y : TopCat
                     f : CategoryTheory.Iso X Y
                     x : ↑X
                     ⊢ Eq (f.inv (f.hom x)) x
                   -/
  left_inv x := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      X Y : TopCat
                      f : CategoryTheory.Iso X Y
                      x : ↑Y
                      ⊢ Eq (f.hom (f.inv x)) x
                    -/
  right_inv x := by simp
                    /-
                      🎉 no goals
                    -/
  continuous_toFun := f.hom.continuous
  continuous_invFun := f.inv.continuous


@[simp]
theorem of_isoOfHomeo {X Y : TopCat.{u}} (f : X ≃ₜ Y) : homeoOfIso (isoOfHomeo f) = f := by
  -- Porting note: unfold some defs now
  /-
    X Y : TopCat
    f : Homeomorph ↑X ↑Y
    ⊢ Eq (TopCat.homeoOfIso (TopCat.isoOfHomeo f)) f
  -/
  dsimp [homeoOfIso, isoOfHomeo]
  /-
    X Y : TopCat
    f : Homeomorph ↑X ↑Y
    ⊢ Eq { toFun := ⇑↑f, invFun := ⇑↑f.symm, left_inv := ⋯, right_inv := ⋯, contin …
  -/
  ext
  /-
    case H
    X Y : TopCat
    f : Homeomorph ↑X ↑Y
    x✝ : ↑X
    ⊢ Eq ({ toFun := ⇑↑f, invFun := ⇑↑f.symm, left_inv := ⋯, right_inv := ⋯, conti …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem of_homeoOfIso {X Y : TopCat.{u}} (f : X ≅ Y) : isoOfHomeo (homeoOfIso f) = f := by
  -- Porting note: unfold some defs now
  /-
    X Y : TopCat
    f : CategoryTheory.Iso X Y
    ⊢ Eq (TopCat.isoOfHomeo (TopCat.homeoOfIso f)) f
  -/
  dsimp [homeoOfIso, isoOfHomeo]
  /-
    X Y : TopCat
    f : CategoryTheory.Iso X Y
    ⊢ Eq { hom := ↑{ toFun := ⇑f.hom, invFun := ⇑f.inv, left_inv := ⋯, right_inv : …
  -/
  ext
  /-
    case w.w
    X Y : TopCat
    f : CategoryTheory.Iso X Y
    x✝ : (CategoryTheory.forget TopCat).obj X
    ⊢ Eq ({ hom := ↑{ toFun := ⇑f.hom, invFun := ⇑f.inv, left_inv := ⋯, right_inv  …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma isIso_of_bijective_of_isOpenMap {X Y : TopCat.{u}} (f : X ⟶ Y)
    (hfbij : Function.Bijective f) (hfcl : IsOpenMap f) : IsIso f :=
  let e : X ≃ₜ Y := Homeomorph.homeomorphOfContinuousOpen
    (Equiv.ofBijective f hfbij) f.continuous hfcl
  inferInstanceAs <| IsIso (TopCat.isoOfHomeo e).hom


lemma isIso_of_bijective_of_isClosedMap {X Y : TopCat.{u}} (f : X ⟶ Y)
    (hfbij : Function.Bijective f) (hfcl : IsClosedMap f) : IsIso f :=
  let e : X ≃ₜ Y := Homeomorph.homeomorphOfContinuousClosed
    (Equiv.ofBijective f hfbij) f.continuous hfcl
  inferInstanceAs <| IsIso (TopCat.isoOfHomeo e).hom

-- Porting note: simpNF requested partially simped version below

theorem isOpenEmbedding_iff_comp_isIso {X Y Z : TopCat} (f : X ⟶ Y) (g : Y ⟶ Z) [IsIso g] :
    IsOpenEmbedding (f ≫ g) ↔ IsOpenEmbedding f :=
  (TopCat.homeoOfIso (asIso g)).isOpenEmbedding.of_comp_iff f


@[deprecated (since := "2024-10-18")]
alias openEmbedding_iff_comp_isIso := isOpenEmbedding_iff_comp_isIso


@[simp]
theorem isOpenEmbedding_iff_comp_isIso' {X Y Z : TopCat} (f : X ⟶ Y) (g : Y ⟶ Z) [IsIso g] :
    IsOpenEmbedding ((forget TopCat).map f ≫ (forget TopCat).map g) ↔ IsOpenEmbedding f := by
  /-
    X Y Z : TopCat
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : CategoryTheory.IsIso g
    ⊢ Iff (Topology.IsOpenEmbedding (CategoryTheory.CategoryStruct.comp ((Category …
  -/
  simp only [← Functor.map_comp]
  /-
    X Y Z : TopCat
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : CategoryTheory.IsIso g
    ⊢ Iff (Topology.IsOpenEmbedding ((CategoryTheory.forget TopCat).map (CategoryT …
  -/
  exact isOpenEmbedding_iff_comp_isIso f g
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-18")]
alias openEmbedding_iff_comp_isIso' := isOpenEmbedding_iff_comp_isIso'


theorem isOpenEmbedding_iff_isIso_comp {X Y Z : TopCat} (f : X ⟶ Y) (g : Y ⟶ Z) [IsIso f] :
    IsOpenEmbedding (f ≫ g) ↔ IsOpenEmbedding g := by
  /-
    X Y Z : TopCat
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : CategoryTheory.IsIso f
    ⊢ Iff (Topology.IsOpenEmbedding ⇑(CategoryTheory.CategoryStruct.comp f g)) (To …
  -/
  constructor
    /-
      case mp
      X Y Z : TopCat
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      inst✝ : CategoryTheory.IsIso f
      ⊢ Topology.IsOpenEmbedding ⇑(CategoryTheory.CategoryStruct.comp f g) → Topolog …
    -/
  · intro h
    /-
      case mp
      X Y Z : TopCat
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      inst✝ : CategoryTheory.IsIso f
      h : Topology.IsOpenEmbedding ⇑(CategoryTheory.CategoryStruct.comp f g)
      ⊢ Topology.IsOpenEmbedding ⇑g
    -/
    convert h.comp (TopCat.homeoOfIso (asIso f).symm).isOpenEmbedding
    /-
      case h.e'_5
      X Y Z : TopCat
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      inst✝ : CategoryTheory.IsIso f
      h : Topology.IsOpenEmbedding ⇑(CategoryTheory.CategoryStruct.comp f g)
      ⊢ Eq (⇑g) (Function.comp ⇑(CategoryTheory.CategoryStruct.comp f g) ⇑(TopCat.ho …
    -/
    exact congrArg _ (IsIso.inv_hom_id_assoc f g).symm
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X Y Z : TopCat
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      inst✝ : CategoryTheory.IsIso f
      ⊢ Topology.IsOpenEmbedding ⇑g → Topology.IsOpenEmbedding ⇑(CategoryTheory.Cate …
    -/
  · exact fun h => h.comp (TopCat.homeoOfIso (asIso f)).isOpenEmbedding
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-18")]
alias openEmbedding_iff_isIso_comp := isOpenEmbedding_iff_isIso_comp


@[simp]
theorem isOpenEmbedding_iff_isIso_comp' {X Y Z : TopCat} (f : X ⟶ Y) (g : Y ⟶ Z) [IsIso f] :
    IsOpenEmbedding ((forget TopCat).map f ≫ (forget TopCat).map g) ↔ IsOpenEmbedding g := by
  /-
    X Y Z : TopCat
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : CategoryTheory.IsIso f
    ⊢ Iff (Topology.IsOpenEmbedding (CategoryTheory.CategoryStruct.comp ((Category …
  -/
  simp only [← Functor.map_comp]
  /-
    X Y Z : TopCat
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : CategoryTheory.IsIso f
    ⊢ Iff (Topology.IsOpenEmbedding ((CategoryTheory.forget TopCat).map (CategoryT …
  -/
  exact isOpenEmbedding_iff_isIso_comp f g
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-18")]
alias openEmbedding_iff_isIso_comp' := isOpenEmbedding_iff_isIso_comp'



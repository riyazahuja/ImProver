/-- The category of groups with zero. -/
def GrpWithZero :=
  Bundled GroupWithZero


instance : CoeSort GrpWithZero Type* :=
  Bundled.coeSort


instance (X : GrpWithZero) : GroupWithZero X :=
  X.str


/-- Construct a bundled `GrpWithZero` from a `GroupWithZero`. -/
def of (α : Type*) [GroupWithZero α] : GrpWithZero :=
  Bundled.of α


instance : Inhabited GrpWithZero :=
  ⟨of (WithZero PUnit)⟩


instance : LargeCategory.{u} GrpWithZero where
  Hom X Y := MonoidWithZeroHom X Y
  id X := MonoidWithZeroHom.id X
  comp f g := g.comp f
  id_comp := MonoidWithZeroHom.comp_id
  comp_id := MonoidWithZeroHom.id_comp
  assoc _ _ _ := MonoidWithZeroHom.comp_assoc _ _ _


instance {M N : GrpWithZero} : FunLike (M ⟶ N) M N :=
  ⟨fun f => f.toFun, fun f g h => by
    /-
      M N : GrpWithZero
      f g : Quiver.Hom M N
      h : Eq ((fun f => (↑f).toFun) f) ((fun f => (↑f).toFun) g)
      ⊢ Eq f g
    -/
    cases f
    /-
      case mk
      M N : GrpWithZero
      g : Quiver.Hom M N
      toZeroHom✝ : ZeroHom ↑M ↑N
      map_one'✝ : Eq (toZeroHom✝.toFun 1) 1
      map_mul'✝ : ∀ (x y : ↑M), Eq (toZeroHom✝.toFun (HMul.hMul x y)) (HMul.hMul (to …
      h : Eq ((fun f => (↑f).toFun) { toZeroHom := toZeroHom✝, map_one' := map_one'✝ …
      ⊢ Eq { toZeroHom := toZeroHom✝, map_one' := map_one'✝, map_mul' := map_mul'✝ } g
    -/
    cases g
    /-
      case mk.mk
      M N : GrpWithZero
      toZeroHom✝¹ : ZeroHom ↑M ↑N
      map_one'✝¹ : Eq (toZeroHom✝¹.toFun 1) 1
      map_mul'✝¹ : ∀ (x y : ↑M), Eq (toZeroHom✝¹.toFun (HMul.hMul x y)) (HMul.hMul ( …
      toZeroHom✝ : ZeroHom ↑M ↑N
      map_one'✝ : Eq (toZeroHom✝.toFun 1) 1
      map_mul'✝ : ∀ (x y : ↑M), Eq (toZeroHom✝.toFun (HMul.hMul x y)) (HMul.hMul (to …
      h : Eq ((fun f => (↑f).toFun) { toZeroHom := toZeroHom✝¹, map_one' := map_one' …
      ⊢ Eq { toZeroHom := toZeroHom✝¹, map_one' := map_one'✝¹, map_mul' := map_mul'✝ …
    -/
    congr
    /-
      case mk.mk.e_toZeroHom
      M N : GrpWithZero
      toZeroHom✝¹ : ZeroHom ↑M ↑N
      map_one'✝¹ : Eq (toZeroHom✝¹.toFun 1) 1
      map_mul'✝¹ : ∀ (x y : ↑M), Eq (toZeroHom✝¹.toFun (HMul.hMul x y)) (HMul.hMul ( …
      toZeroHom✝ : ZeroHom ↑M ↑N
      map_one'✝ : Eq (toZeroHom✝.toFun 1) 1
      map_mul'✝ : ∀ (x y : ↑M), Eq (toZeroHom✝.toFun (HMul.hMul x y)) (HMul.hMul (to …
      h : Eq ((fun f => (↑f).toFun) { toZeroHom := toZeroHom✝¹, map_one' := map_one' …
      ⊢ Eq toZeroHom✝¹ toZeroHom✝
    -/
    apply DFunLike.coe_injective'
    /-
      case mk.mk.e_toZeroHom.a
      M N : GrpWithZero
      toZeroHom✝¹ : ZeroHom ↑M ↑N
      map_one'✝¹ : Eq (toZeroHom✝¹.toFun 1) 1
      map_mul'✝¹ : ∀ (x y : ↑M), Eq (toZeroHom✝¹.toFun (HMul.hMul x y)) (HMul.hMul ( …
      toZeroHom✝ : ZeroHom ↑M ↑N
      map_one'✝ : Eq (toZeroHom✝.toFun 1) 1
      map_mul'✝ : ∀ (x y : ↑M), Eq (toZeroHom✝.toFun (HMul.hMul x y)) (HMul.hMul (to …
      h : Eq ((fun f => (↑f).toFun) { toZeroHom := toZeroHom✝¹, map_one' := map_one' …
      ⊢ Eq ⇑toZeroHom✝¹ ⇑toZeroHom✝
    -/
    exact h⟩
    /-
      🎉 no goals
    -/


lemma coe_id {X : GrpWithZero} : (𝟙 X : X → X) = id := rfl


lemma coe_comp {X Y Z : GrpWithZero} {f : X ⟶ Y} {g : Y ⟶ Z} : (f ≫ g : X → Z) = g ∘ f := rfl


instance groupWithZeroConcreteCategory : ConcreteCategory GrpWithZero where
  forget :=
  { obj := fun G => G
    map := fun f => f.toFun }
  forget_faithful := ⟨fun h => DFunLike.coe_injective h⟩


@[simp] lemma forget_map {X Y : GrpWithZero} (f : X ⟶ Y) :
  (forget GrpWithZero).map f = f := rfl


instance hasForgetToBipointed : HasForget₂ GrpWithZero Bipointed where
  forget₂ :=
      { obj := fun X => ⟨X, 0, 1⟩
        map := fun f => ⟨f, f.map_zero', f.map_one'⟩ }


instance hasForgetToMon : HasForget₂ GrpWithZero MonCat where
  forget₂ :=
      { obj := fun X => ⟨ X , _ ⟩
        map := fun f => f.toMonoidHom }


/-- Constructs an isomorphism of groups with zero from a group isomorphism between them. -/
@[simps]
def Iso.mk {α β : GrpWithZero.{u}} (e : α ≃* β) : α ≅ β where
  hom := (e : α →*₀ β)
  inv := (e.symm : β →*₀ α)
  hom_inv_id := by
    /-
      α β : GrpWithZero
      e : MulEquiv ↑α ↑β
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ↑e ↑e.symm) (CategoryTheory.CategoryS …
    -/
    ext
    /-
      case w
      α β : GrpWithZero
      e : MulEquiv ↑α ↑β
      x✝ : (CategoryTheory.forget GrpWithZero).obj α
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ↑e ↑e.symm) x✝) ((CategoryTheory.Cat …
    -/
    exact e.symm_apply_apply _
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      α β : GrpWithZero
      e : MulEquiv ↑α ↑β
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ↑e.symm ↑e) (CategoryTheory.CategoryS …
    -/
    ext
    /-
      case w
      α β : GrpWithZero
      e : MulEquiv ↑α ↑β
      x✝ : (CategoryTheory.forget GrpWithZero).obj β
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ↑e.symm ↑e) x✝) ((CategoryTheory.Cat …
    -/
    exact e.apply_symm_apply _
    /-
      🎉 no goals
    -/



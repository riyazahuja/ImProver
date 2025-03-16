/-- The category of finite types. -/
def FintypeCat :=
  Bundled Fintype


instance : CoeSort FintypeCat Type* :=
  Bundled.coeSort


/-- Construct a bundled `FintypeCat` from the underlying type and typeclass. -/
def of (X : Type*) [Fintype X] : FintypeCat :=
  Bundled.of X


instance : Inhabited FintypeCat :=
  ⟨of PEmpty⟩


instance {X : FintypeCat} : Fintype X :=
  X.2


instance : Category FintypeCat :=
  InducedCategory.category Bundled.α


/-- The fully faithful embedding of `FintypeCat` into the category of types. -/
@[simps!]
def incl : FintypeCat ⥤ Type* :=
  inducedFunctor _


instance : incl.Full := InducedCategory.full _

instance : incl.Faithful := InducedCategory.faithful _


instance concreteCategoryFintype : ConcreteCategory FintypeCat :=
  ⟨incl⟩

/- Help typeclass inference infer fullness of forgetful functor. -/

instance : (forget FintypeCat).Full := inferInstanceAs <| FintypeCat.incl.Full


@[simp]
theorem id_apply (X : FintypeCat) (x : X) : (𝟙 X : X → X) x = x :=
  rfl


@[simp]
theorem comp_apply {X Y Z : FintypeCat} (f : X ⟶ Y) (g : Y ⟶ Z) (x : X) : (f ≫ g) x = g (f x) :=
  rfl


@[simp]
lemma hom_inv_id_apply {X Y : FintypeCat} (f : X ≅ Y) (x : X) : f.inv (f.hom x) = x :=
  congr_fun f.hom_inv_id x


@[simp]
lemma inv_hom_id_apply {X Y : FintypeCat} (f : X ≅ Y) (y : Y) : f.hom (f.inv y) = y :=
  congr_fun f.inv_hom_id y

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10688): added to ease automation

@[ext]
lemma hom_ext {X Y : FintypeCat} (f g : X ⟶ Y) (h : ∀ x, f x = g x) : f = g := by
  /-
    X Y : FintypeCat
    f g : Quiver.Hom X Y
    h : ∀ (x : ↑X), Eq (f x) (g x)
    ⊢ Eq f g
  -/
  funext
  /-
    case h
    X Y : FintypeCat
    f g : Quiver.Hom X Y
    h : ∀ (x : ↑X), Eq (f x) (g x)
    x✝ : ↑X
    ⊢ Eq (f x✝) (g x✝)
  -/
  apply h
  /-
    🎉 no goals
  -/

-- See `equivEquivIso` in the root namespace for the analogue in `Type`.

/-- Equivalences between finite types are the same as isomorphisms in `FintypeCat`. -/
@[simps]
def equivEquivIso {A B : FintypeCat} : A ≃ B ≃ (A ≅ B) where
  toFun e :=
    { hom := e
      inv := e.symm }
  invFun i :=
    { toFun := i.hom
      invFun := i.inv
      left_inv := congr_fun i.hom_inv_id
      right_inv := congr_fun i.inv_hom_id }
                 /-
                   A B : FintypeCat
                   ⊢ Function.LeftInverse (fun i => { toFun := i.hom, invFun := i.inv, left_inv : …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    A B : FintypeCat
                    ⊢ Function.RightInverse (fun i => { toFun := i.hom, invFun := i.inv, left_inv  …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


instance (X Y : FintypeCat) : Finite (X ⟶ Y) :=
  inferInstanceAs <| Finite (X → Y)


instance (X Y : FintypeCat) : Finite (X ≅ Y) :=
  Finite.of_injective _ (fun _ _ h ↦ Iso.ext h)


instance (X : FintypeCat) : Finite (Aut X) :=
  inferInstanceAs <| Finite (X ≅ X)


/--
The "standard" skeleton for `FintypeCat`. This is the full subcategory of `FintypeCat`
spanned by objects of the form `ULift (Fin n)` for `n : ℕ`. We parameterize the objects
of `Fintype.Skeleton` directly as `ULift ℕ`, as the type `ULift (Fin m) ≃ ULift (Fin n)`
is nonempty if and only if `n = m`. Specifying universes, `Skeleton : Type u` is a small
skeletal category equivalent to `Fintype.{u}`.
-/
def Skeleton : Type u :=
  ULift ℕ


/-- Given any natural number `n`, this creates the associated object of `Fintype.Skeleton`. -/
def mk : ℕ → Skeleton :=
  ULift.up


instance : Inhabited Skeleton :=
  ⟨mk 0⟩


/-- Given any object of `Fintype.Skeleton`, this returns the associated natural number. -/
def len : Skeleton → ℕ :=
  ULift.down


@[ext]
theorem ext (X Y : Skeleton) : X.len = Y.len → X = Y :=
  ULift.ext _ _


instance : SmallCategory Skeleton.{u} where
  Hom X Y := ULift.{u} (Fin X.len) → ULift.{u} (Fin Y.len)
  id _ := id
  comp f g := g ∘ f


theorem is_skeletal : Skeletal Skeleton.{u} := fun X Y ⟨h⟩ =>
  ext _ _ <|
    Fin.equiv_iff_eq.mp <|
      Nonempty.intro <|
        { toFun := fun x => (h.hom ⟨x⟩).down
          invFun := fun x => (h.inv ⟨x⟩).down
          left_inv := by
            /-
              X Y : FintypeCat.Skeleton
              x✝ : CategoryTheory.IsIsomorphic X Y
              h : CategoryTheory.Iso X Y
              ⊢ Function.LeftInverse (fun x => (h.inv { down := x }).down) fun x => (h.hom { …
            -/
            intro a
            /-
              X Y : FintypeCat.Skeleton
              x✝ : CategoryTheory.IsIsomorphic X Y
              h : CategoryTheory.Iso X Y
              a : Fin X.len
              ⊢ Eq ((fun x => (h.inv { down := x }).down) ((fun x => (h.hom { down := x }).d …
            -/
            change ULift.down _ = _
            /-
              X Y : FintypeCat.Skeleton
              x✝ : CategoryTheory.IsIsomorphic X Y
              h : CategoryTheory.Iso X Y
              a : Fin X.len
              ⊢ Eq (h.inv { down := (fun x => (h.hom { down := x }).down) a }).down a
            -/
            rw [ULift.up_down]
            /-
              X Y : FintypeCat.Skeleton
              x✝ : CategoryTheory.IsIsomorphic X Y
              h : CategoryTheory.Iso X Y
              a : Fin X.len
              ⊢ Eq (h.inv (h.hom { down := a })).down a
            -/
            change ((h.hom ≫ h.inv) _).down = _
            /-
              X Y : FintypeCat.Skeleton
              x✝ : CategoryTheory.IsIsomorphic X Y
              h : CategoryTheory.Iso X Y
              a : Fin X.len
              ⊢ Eq (CategoryTheory.CategoryStruct.comp h.hom h.inv { down := a }).down a
            -/
            simp
            /-
              X Y : FintypeCat.Skeleton
              x✝ : CategoryTheory.IsIsomorphic X Y
              h : CategoryTheory.Iso X Y
              a : Fin X.len
              ⊢ Eq (CategoryTheory.CategoryStruct.id X { down := a }).down a
            -/
            rfl
            /-
              🎉 no goals
            -/
          right_inv := by
            /-
              X Y : FintypeCat.Skeleton
              x✝ : CategoryTheory.IsIsomorphic X Y
              h : CategoryTheory.Iso X Y
              ⊢ Function.RightInverse (fun x => (h.inv { down := x }).down) fun x => (h.hom  …
            -/
            intro a
            /-
              X Y : FintypeCat.Skeleton
              x✝ : CategoryTheory.IsIsomorphic X Y
              h : CategoryTheory.Iso X Y
              a : Fin Y.len
              ⊢ Eq ((fun x => (h.hom { down := x }).down) ((fun x => (h.inv { down := x }).d …
            -/
            change ULift.down _ = _
            /-
              X Y : FintypeCat.Skeleton
              x✝ : CategoryTheory.IsIsomorphic X Y
              h : CategoryTheory.Iso X Y
              a : Fin Y.len
              ⊢ Eq (h.hom { down := (fun x => (h.inv { down := x }).down) a }).down a
            -/
            rw [ULift.up_down]
            /-
              X Y : FintypeCat.Skeleton
              x✝ : CategoryTheory.IsIsomorphic X Y
              h : CategoryTheory.Iso X Y
              a : Fin Y.len
              ⊢ Eq (h.hom (h.inv { down := a })).down a
            -/
            change ((h.inv ≫ h.hom) _).down = _
            /-
              X Y : FintypeCat.Skeleton
              x✝ : CategoryTheory.IsIsomorphic X Y
              h : CategoryTheory.Iso X Y
              a : Fin Y.len
              ⊢ Eq (CategoryTheory.CategoryStruct.comp h.inv h.hom { down := a }).down a
            -/
            simp
            /-
              X Y : FintypeCat.Skeleton
              x✝ : CategoryTheory.IsIsomorphic X Y
              h : CategoryTheory.Iso X Y
              a : Fin Y.len
              ⊢ Eq (CategoryTheory.CategoryStruct.id Y { down := a }).down a
            -/
            rfl }
            /-
              🎉 no goals
            -/


/-- The canonical fully faithful embedding of `Fintype.Skeleton` into `FintypeCat`. -/
def incl : Skeleton.{u} ⥤ FintypeCat.{u} where
  obj X := FintypeCat.of (ULift (Fin X.len))
  map f := f


instance : incl.Full where map_surjective f := ⟨f, rfl⟩


instance : incl.Faithful where


instance : incl.EssSurj :=
  Functor.EssSurj.mk fun X =>
    let F := Fintype.equivFin X
    ⟨mk (Fintype.card X),
      Nonempty.intro
        { hom := F.symm ∘ ULift.down
          inv := ULift.up ∘ F }⟩


noncomputable instance : incl.IsEquivalence where


/-- The equivalence between `Fintype.Skeleton` and `Fintype`. -/
noncomputable def equivalence : Skeleton ≌ FintypeCat :=
  incl.asEquivalence


@[simp]
theorem incl_mk_nat_card (n : ℕ) : Fintype.card (incl.obj (mk n)) = n := by
  /-
    n : Nat
    ⊢ Eq (Fintype.card ↑(FintypeCat.Skeleton.incl.obj (FintypeCat.Skeleton.mk n))) n
  -/
  convert Finset.card_fin n
  /-
    case h.e'_2
    n : Nat
    ⊢ Eq (Fintype.card ↑(FintypeCat.Skeleton.incl.obj (FintypeCat.Skeleton.mk n))) …
  -/
  apply Fintype.ofEquiv_card
  /-
    🎉 no goals
  -/


/-- `Fintype.Skeleton` is a skeleton of `Fintype`. -/
lemma isSkeleton : IsSkeletonOf FintypeCat Skeleton Skeleton.incl where
  skel := Skeleton.is_skeletal
            /-
              ⊢ FintypeCat.Skeleton.incl.IsEquivalence
            -/
  eqv := by infer_instance
            /-
              🎉 no goals
            -/


/-- If `u` and `v` are two arbitrary universes, we may construct a functor
`uSwitch.{u, v} : FintypeCat.{u} ⥤ FintypeCat.{v}` by sending
`X : FintypeCat.{u}` to `ULift.{v} (Fin (Fintype.card X))`. -/
noncomputable def uSwitch : FintypeCat.{u} ⥤ FintypeCat.{v} where
  obj X := FintypeCat.of <| ULift.{v} (Fin (Fintype.card X))
  map {X Y} f x := ULift.up <| (Fintype.equivFin Y) (f ((Fintype.equivFin X).symm x.down))
                             /-
                               X Y Z : FintypeCat
                               f : Quiver.Hom X Y
                               g : Quiver.Hom Y Z
                               ⊢ Eq ({ obj := fun X => FintypeCat.of (ULift.{v, 0} (Fin (Fintype.card ↑X))),  …
                             -/
  map_comp {X Y Z} f g := by ext; simp
                                  /-
                                    🎉 no goals
                                  -/


/-- Switching the universe of an object `X : FintypeCat.{u}` does not change `X` up to equivalence
of types. This is natural in the sense that it commutes with `uSwitch.map f` for
any `f : X ⟶ Y` in `FintypeCat.{u}`. -/
noncomputable def uSwitchEquiv (X : FintypeCat.{u}) :
    uSwitch.{u, v}.obj X ≃ X :=
  Equiv.ulift.trans (Fintype.equivFin X).symm


lemma uSwitchEquiv_naturality {X Y : FintypeCat.{u}} (f : X ⟶ Y)
    (x : uSwitch.{u, v}.obj X) :
    f (X.uSwitchEquiv x) = Y.uSwitchEquiv (uSwitch.map f x) := by
  /-
    X Y : FintypeCat
    f : Quiver.Hom X Y
    x : ↑(FintypeCat.uSwitch.obj X)
    ⊢ Eq (f (X.uSwitchEquiv x)) (Y.uSwitchEquiv (FintypeCat.uSwitch.map f x))
  -/
  simp only [uSwitch, uSwitchEquiv, Equiv.trans_apply]
  /-
    X Y : FintypeCat
    f : Quiver.Hom X Y
    x : ↑(FintypeCat.uSwitch.obj X)
    ⊢ Eq (f ((Fintype.equivFin ↑X).symm (Equiv.ulift x))) ((Fintype.equivFin ↑Y).s …
  -/
  erw [Equiv.ulift_apply, Equiv.ulift_apply]
  /-
    X Y : FintypeCat
    f : Quiver.Hom X Y
    x : ↑(FintypeCat.uSwitch.obj X)
    ⊢ Eq (f ((Fintype.equivFin ↑X).symm x.down)) ((Fintype.equivFin ↑Y).symm { dow …
  -/
  simp only [Equiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


lemma uSwitchEquiv_symm_naturality {X Y : FintypeCat.{u}} (f : X ⟶ Y) (x : X) :
    uSwitch.map f (X.uSwitchEquiv.symm x) = Y.uSwitchEquiv.symm (f x) := by
  rw [← Equiv.apply_eq_iff_eq_symm_apply, ← uSwitchEquiv_naturality f,
    Equiv.apply_symm_apply]


lemma uSwitch_map_uSwitch_map {X Y : FintypeCat.{u}} (f : X ⟶ Y) :
    uSwitch.map (uSwitch.map f) =
    (equivEquivIso ((uSwitch.obj X).uSwitchEquiv.trans X.uSwitchEquiv)).hom ≫
      f ≫ (equivEquivIso ((uSwitch.obj Y).uSwitchEquiv.trans
      Y.uSwitchEquiv)).inv := by
  /-
    X Y : FintypeCat
    f : Quiver.Hom X Y
    ⊢ Eq (FintypeCat.uSwitch.map (FintypeCat.uSwitch.map f)) (CategoryTheory.Categ …
  -/
  ext x
  /-
    case h
    X Y : FintypeCat
    f : Quiver.Hom X Y
    x : ↑(FintypeCat.uSwitch.obj (FintypeCat.uSwitch.obj X))
    ⊢ Eq (FintypeCat.uSwitch.map (FintypeCat.uSwitch.map f) x) (CategoryTheory.Cat …
  -/
  simp only [comp_apply, equivEquivIso_apply_hom, Equiv.trans_apply]
  /-
    case h
    X Y : FintypeCat
    f : Quiver.Hom X Y
    x : ↑(FintypeCat.uSwitch.obj (FintypeCat.uSwitch.obj X))
    ⊢ Eq (FintypeCat.uSwitch.map (FintypeCat.uSwitch.map f) x) ((FintypeCat.equivE …
  -/
  rw [uSwitchEquiv_naturality f, ← uSwitchEquiv_naturality]
  /-
    case h
    X Y : FintypeCat
    f : Quiver.Hom X Y
    x : ↑(FintypeCat.uSwitch.obj (FintypeCat.uSwitch.obj X))
    ⊢ Eq (FintypeCat.uSwitch.map (FintypeCat.uSwitch.map f) x) ((FintypeCat.equivE …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `uSwitch.{u, v}` is an equivalence of categories with quasi-inverse `uSwitch.{v, u}`. -/
noncomputable def uSwitchEquivalence : FintypeCat.{u} ≌ FintypeCat.{v} where
  functor := uSwitch
  inverse := uSwitch
  unitIso := NatIso.ofComponents (fun X ↦ (equivEquivIso <|
    (uSwitch.obj X).uSwitchEquiv.trans X.uSwitchEquiv).symm) <| by
    /-
      ⊢ ∀ {X Y : FintypeCat} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
    -/
    simp [uSwitch_map_uSwitch_map]
    /-
      🎉 no goals
    -/
  counitIso := NatIso.ofComponents (fun X ↦ equivEquivIso <|
    (uSwitch.obj X).uSwitchEquiv.trans X.uSwitchEquiv) <| by
    /-
      ⊢ ∀ {X Y : FintypeCat} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
    -/
    simp [uSwitch_map_uSwitch_map]
    /-
      🎉 no goals
    -/
  functor_unitIso_comp X := by
    /-
      X : FintypeCat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (FintypeCat.uSwitch.map ((CategoryThe …
    -/
    ext x
    /-
      case h
      X : FintypeCat
      x : ↑(FintypeCat.uSwitch.obj ((CategoryTheory.Functor.id FintypeCat).obj X))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (FintypeCat.uSwitch.map ((CategoryThe …
    -/
    simp [← uSwitchEquiv_naturality]
    /-
      🎉 no goals
    -/


instance : uSwitch.IsEquivalence :=
  uSwitchEquivalence.isEquivalence_functor


lemma naturality (σ : F ⟶ G) (f : X ⟶ Y) (x : F.obj X) :
    σ.app Y (F.map f x) = G.map f (σ.app X x) :=
  congr_fun (σ.naturality f) x



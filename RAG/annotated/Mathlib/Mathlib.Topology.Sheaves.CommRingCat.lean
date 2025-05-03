/--
Specialize `restrictOpen` to `CommRingCat` because inferring `C := CommRingCat` isn't reliable.
Instead of unfolding the definition, rewrite with `restrictOpenCommRingCat_apply` to ensure the
correct coercion to functions is taken.

(The correct fix in the longer term is to redesign concrete categories so we don't use `forget`
everywhere, but the correct `FunLike` instance for the morphisms of those categories.)
-/
abbrev restrictOpenCommRingCat {X : TopCat}
    {F : Presheaf CommRingCat X} {V : Opens ↑X} (f : CommRingCat.carrier (F.obj (op V)))
    (U : Opens ↑X) (e : U ≤ V := by restrict_tac) :
    CommRingCat.carrier (F.obj (op U)) :=
  TopCat.Presheaf.restrictOpen (C := CommRingCat) f U e


/-- Notation for `TopCat.Presheaf.restrictOpenCommRingCat`. -/
scoped[AlgebraicGeometry] infixl:80 " |_ᵣ " => TopCat.Presheaf.restrictOpenCommRingCat


open AlgebraicGeometry in
lemma restrictOpenCommRingCat_apply {X : TopCat}
    {F : Presheaf CommRingCat X} {V : Opens ↑X} (f : CommRingCat.carrier (F.obj (op V)))
    (U : Opens ↑X) (e : U ≤ V := by restrict_tac) :
    f |_ᵣ U = F.map (homOfLE e).op f :=
  rfl


open AlgebraicGeometry in
lemma _root_.CommRingCat.presheaf_restrict_restrict (X : TopCat)
    {F : TopCat.Presheaf CommRingCat X}
    {U V W : Opens ↑X} (e₁ : U ≤ V := by restrict_tac) (e₂ : V ≤ W := by restrict_tac)
    (f : CommRingCat.carrier (F.obj (op W))) :
    f |_ᵣ V |_ᵣ U = f |_ᵣ U :=
  TopCat.Presheaf.restrict_restrict (C := CommRingCat) e₁ e₂ f


/-- A subpresheaf with a submonoid structure on each of the components. -/
structure SubmonoidPresheaf (F : X.Presheaf CommRingCat) where
  obj : ∀ U, Submonoid (F.obj U)
  map : ∀ {U V : (Opens X)ᵒᵖ} (i : U ⟶ V), obj U ≤ (obj V).comap (F.map i).hom


/-- The localization of a presheaf of `CommRing`s with respect to a `SubmonoidPresheaf`. -/
protected noncomputable def SubmonoidPresheaf.localizationPresheaf : X.Presheaf CommRingCat where
  obj U := CommRingCat.of <| Localization (G.obj U)
  map {_ _} i := CommRingCat.ofHom <| IsLocalization.map _ (F.map i).hom (G.map i)
  map_id U := by
    /-
      X : TopCat
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      F : TopCat.Presheaf CommRingCat X
      G : F.SubmonoidPresheaf
      U : Opposite (TopologicalSpace.Opens ↑X)
      ⊢ Eq ({ obj := fun U => CommRingCat.of (Localization (G.obj U)), map := fun {x …
    -/
    simp_rw [F.map_id]
    /-
      X : TopCat
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      F : TopCat.Presheaf CommRingCat X
      G : F.SubmonoidPresheaf
      U : Opposite (TopologicalSpace.Opens ↑X)
      ⊢ Eq (CommRingCat.ofHom (IsLocalization.map (Localization (G.obj U)) (Category …
    -/
    ext x
    -- Porting note: `M` and `S` needs to be specified manually
    /-
      case hf.a
      X : TopCat
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      F : TopCat.Presheaf CommRingCat X
      G : F.SubmonoidPresheaf
      U : Opposite (TopologicalSpace.Opens ↑X)
      x : ↑(CommRingCat.of (Localization (G.obj U)))
      ⊢ Eq ((CommRingCat.ofHom (IsLocalization.map (Localization (G.obj U)) (Categor …
    -/
    exact IsLocalization.map_id (M := G.obj U) (S := Localization (G.obj U)) x
    /-
      🎉 no goals
    -/
  map_comp {U V W} i j := by
    /-
      X : TopCat
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      F : TopCat.Presheaf CommRingCat X
      G : F.SubmonoidPresheaf
      U V W : Opposite (TopologicalSpace.Opens ↑X)
      i : Quiver.Hom U V
      j : Quiver.Hom V W
      ⊢ Eq ({ obj := fun U => CommRingCat.of (Localization (G.obj U)), map := fun {x …
    -/
    delta CommRingCat.ofHom CommRingCat.of Bundled.of
    /-
      X : TopCat
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      F : TopCat.Presheaf CommRingCat X
      G : F.SubmonoidPresheaf
      U V W : Opposite (TopologicalSpace.Opens ↑X)
      i : Quiver.Hom U V
      j : Quiver.Hom V W
      ⊢ Eq ({ obj := fun U => CommRingCat.mk✝ (Localization (G.obj U)), map := fun { …
    -/
    simp_rw [F.map_comp]
    /-
      X : TopCat
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      F : TopCat.Presheaf CommRingCat X
      G : F.SubmonoidPresheaf
      U V W : Opposite (TopologicalSpace.Opens ↑X)
      i : Quiver.Hom U V
      j : Quiver.Hom V W
      ⊢ Eq { hom := IsLocalization.map (Localization (G.obj W)) (CategoryTheory.Cate …
    -/
    ext : 1
    /-
      case hf
      X : TopCat
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      F : TopCat.Presheaf CommRingCat X
      G : F.SubmonoidPresheaf
      U V W : Opposite (TopologicalSpace.Opens ↑X)
      i : Quiver.Hom U V
      j : Quiver.Hom V W
      ⊢ Eq { hom := IsLocalization.map (Localization (G.obj W)) (CategoryTheory.Cate …
    -/
    dsimp
    /-
      case hf
      X : TopCat
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      F : TopCat.Presheaf CommRingCat X
      G : F.SubmonoidPresheaf
      U V W : Opposite (TopologicalSpace.Opens ↑X)
      i : Quiver.Hom U V
      j : Quiver.Hom V W
      ⊢ Eq (IsLocalization.map (Localization (G.obj W)) ((F.map j).hom.comp (F.map i …
    -/
    rw [IsLocalization.map_comp_map]
    /-
      🎉 no goals
    -/

-- Porting note: this instance can't be synthesized

instance (U) : Algebra (F.obj U) (G.localizationPresheaf.obj U) :=
  show Algebra _ (Localization (G.obj U)) from inferInstance

-- Porting note: this instance can't be synthesized

instance (U) : IsLocalization (G.obj U) (G.localizationPresheaf.obj U) :=
  show IsLocalization (G.obj U) (Localization (G.obj U)) from inferInstance


/-- The map into the localization presheaf. -/
@[simps app]
def SubmonoidPresheaf.toLocalizationPresheaf : F ⟶ G.localizationPresheaf where
  app U := CommRingCat.ofHom <| algebraMap (F.obj U) (Localization <| G.obj U)
  naturality {_ _} i := CommRingCat.hom_ext <| (IsLocalization.map_comp (G.map i)).symm


instance epi_toLocalizationPresheaf : Epi G.toLocalizationPresheaf :=
  @NatTrans.epi_of_epi_app _ _ _ _ _ _ G.toLocalizationPresheaf fun U => Localization.epi' (G.obj U)


/-- Given a submonoid at each of the stalks, we may define a submonoid presheaf consisting of
sections whose restriction onto each stalk falls in the given submonoid. -/
@[simps]
noncomputable def submonoidPresheafOfStalk (S : ∀ x : X, Submonoid (F.stalk x)) :
    F.SubmonoidPresheaf where
  obj U := ⨅ x : U.unop, Submonoid.comap (F.germ U.unop x.1 x.2).hom (S x)
  map {U V} i := by
    /-
      X : TopCat
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      F : TopCat.Presheaf CommRingCat X
      G : F.SubmonoidPresheaf
      S : (x : ↑X) → Submonoid ↑(F.stalk x)
      U V : Opposite (TopologicalSpace.Opens ↑X)
      i : Quiver.Hom U V
      ⊢ LE.le ((fun U => iInf fun x => Submonoid.comap (F.germ (Opposite.unop U) ↑x  …
    -/
    intro s hs
    /-
      X : TopCat
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      F : TopCat.Presheaf CommRingCat X
      G : F.SubmonoidPresheaf
      S : (x : ↑X) → Submonoid ↑(F.stalk x)
      U V : Opposite (TopologicalSpace.Opens ↑X)
      i : Quiver.Hom U V
      s : ↑(F.obj U)
      hs : Membership.mem ((fun U => iInf fun x => Submonoid.comap (F.germ (Opposite …
      ⊢ Membership.mem (Submonoid.comap (F.map i).hom ((fun U => iInf fun x => Submo …
    -/
    simp only [Submonoid.mem_comap, Submonoid.mem_iInf] at hs ⊢
    /-
      X : TopCat
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      F : TopCat.Presheaf CommRingCat X
      G : F.SubmonoidPresheaf
      S : (x : ↑X) → Submonoid ↑(F.stalk x)
      U V : Opposite (TopologicalSpace.Opens ↑X)
      i : Quiver.Hom U V
      s : ↑(F.obj U)
      hs : ∀ (i : Subtype fun x => Membership.mem (Opposite.unop U) x), Membership.m …
      ⊢ ∀ (i_1 : Subtype fun x => Membership.mem (Opposite.unop V) x), Membership.me …
    -/
    intro x
    /-
      X : TopCat
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      F : TopCat.Presheaf CommRingCat X
      G : F.SubmonoidPresheaf
      S : (x : ↑X) → Submonoid ↑(F.stalk x)
      U V : Opposite (TopologicalSpace.Opens ↑X)
      i : Quiver.Hom U V
      s : ↑(F.obj U)
      hs : ∀ (i : Subtype fun x => Membership.mem (Opposite.unop U) x), Membership.m …
      x : Subtype fun x => Membership.mem (Opposite.unop V) x
      ⊢ Membership.mem (S ↑x) ((F.germ (Opposite.unop V) ↑x ⋯).hom ((F.map i).hom s))
    -/
    change (F.map i.unop.op ≫ F.germ V.unop x.1 x.2) s ∈ _
    /-
      X : TopCat
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      F : TopCat.Presheaf CommRingCat X
      G : F.SubmonoidPresheaf
      S : (x : ↑X) → Submonoid ↑(F.stalk x)
      U V : Opposite (TopologicalSpace.Opens ↑X)
      i : Quiver.Hom U V
      s : ↑(F.obj U)
      hs : ∀ (i : Subtype fun x => Membership.mem (Opposite.unop U) x), Membership.m …
      x : Subtype fun x => Membership.mem (Opposite.unop V) x
      ⊢ Membership.mem (S ↑x) ((CategoryTheory.CategoryStruct.comp (F.map i.unop.op) …
    -/
    rw [F.germ_res]
    /-
      X : TopCat
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.ConcreteCategory C
      F : TopCat.Presheaf CommRingCat X
      G : F.SubmonoidPresheaf
      S : (x : ↑X) → Submonoid ↑(F.stalk x)
      U V : Opposite (TopologicalSpace.Opens ↑X)
      i : Quiver.Hom U V
      s : ↑(F.obj U)
      hs : ∀ (i : Subtype fun x => Membership.mem (Opposite.unop U) x), Membership.m …
      x : Subtype fun x => Membership.mem (Opposite.unop V) x
      ⊢ Membership.mem (S ↑x) ((F.germ (Opposite.unop U) ↑x ⋯).hom s)
    -/
    exact hs ⟨_, i.unop.le x.2⟩
    /-
      🎉 no goals
    -/


noncomputable instance : Inhabited F.SubmonoidPresheaf :=
  ⟨F.submonoidPresheafOfStalk fun _ => ⊥⟩


/-- The localization of a presheaf of `CommRing`s at locally non-zero-divisor sections. -/
noncomputable def totalQuotientPresheaf : X.Presheaf CommRingCat.{w} :=
  (F.submonoidPresheafOfStalk fun x => (F.stalk x)⁰).localizationPresheaf


/-- The map into the presheaf of total quotient rings -/
noncomputable def toTotalQuotientPresheaf : F ⟶ F.totalQuotientPresheaf :=
  SubmonoidPresheaf.toLocalizationPresheaf _

-- Porting note: deriving `Epi` failed

instance : Epi (toTotalQuotientPresheaf F) := epi_toLocalizationPresheaf _


instance (F : X.Sheaf CommRingCat.{w}) : Mono F.presheaf.toTotalQuotientPresheaf := by
  -- Porting note: was an `apply (config := { instances := false })`
  -- See https://github.com/leanprover/lean4/issues/2273
  suffices ∀ (U : (Opens ↑X)ᵒᵖ), Mono (F.presheaf.toTotalQuotientPresheaf.app U) from
    NatTrans.mono_of_mono_app _
  /-
    X : TopCat
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    F✝ : TopCat.Presheaf CommRingCat X
    G : F✝.SubmonoidPresheaf
    F : TopCat.Sheaf CommRingCat X
    ⊢ ∀ (U : Opposite (TopologicalSpace.Opens ↑X)), CategoryTheory.Mono (F.preshea …
  -/
  intro U
  /-
    X : TopCat
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    F✝ : TopCat.Presheaf CommRingCat X
    G : F✝.SubmonoidPresheaf
    F : TopCat.Sheaf CommRingCat X
    U : Opposite (TopologicalSpace.Opens ↑X)
    ⊢ CategoryTheory.Mono (F.presheaf.toTotalQuotientPresheaf.app U)
  -/
  apply ConcreteCategory.mono_of_injective
  /-
    case i
    X : TopCat
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    F✝ : TopCat.Presheaf CommRingCat X
    G : F✝.SubmonoidPresheaf
    F : TopCat.Sheaf CommRingCat X
    U : Opposite (TopologicalSpace.Opens ↑X)
    ⊢ Function.Injective ⇑(F.presheaf.toTotalQuotientPresheaf.app U)
  -/
  dsimp [toTotalQuotientPresheaf, CommRingCat.ofHom]
  -- Porting note: this is a hack to make the `refine` below works
  /-
    case i
    X : TopCat
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    F✝ : TopCat.Presheaf CommRingCat X
    G : F✝.SubmonoidPresheaf
    F : TopCat.Sheaf CommRingCat X
    U : Opposite (TopologicalSpace.Opens ↑X)
    ⊢ Function.Injective ⇑{ hom := algebraMap (↑(F.presheaf.obj U)) (Localization  …
  -/
  set m := _
  /-
    case i
    X : TopCat
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    F✝ : TopCat.Presheaf CommRingCat X
    G : F✝.SubmonoidPresheaf
    F : TopCat.Sheaf CommRingCat X
    U : Opposite (TopologicalSpace.Opens ↑X)
    m : ?m.38151 := ?m.38152
    ⊢ Function.Injective ⇑{ hom := algebraMap (↑(F.presheaf.obj U)) (Localization  …
  -/
  change Function.Injective (algebraMap _ (Localization m))
  /-
    case i
    X : TopCat
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    F✝ : TopCat.Presheaf CommRingCat X
    G : F✝.SubmonoidPresheaf
    F : TopCat.Sheaf CommRingCat X
    U : Opposite (TopologicalSpace.Opens ↑X)
    m : Submonoid ↑(F.presheaf.obj U) := iInf fun x => Submonoid.comap (F.presheaf …
    ⊢ Function.Injective ⇑(algebraMap ((CategoryTheory.forget CommRingCat).obj (F. …
  -/
  change Function.Injective (algebraMap (F.presheaf.obj U) _)
  /-
    case i
    X : TopCat
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    F✝ : TopCat.Presheaf CommRingCat X
    G : F✝.SubmonoidPresheaf
    F : TopCat.Sheaf CommRingCat X
    U : Opposite (TopologicalSpace.Opens ↑X)
    m : Submonoid ↑(F.presheaf.obj U) := iInf fun x => Submonoid.comap (F.presheaf …
    ⊢ Function.Injective ⇑(algebraMap (↑(F.presheaf.obj U)) (Localization m))
  -/
  haveI : IsLocalization _ (Localization m) := Localization.isLocalization
  -- Porting note: `M` and `S` need to be specified manually, so used a hack to save some typing
  /-
    case i
    X : TopCat
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    F✝ : TopCat.Presheaf CommRingCat X
    G : F✝.SubmonoidPresheaf
    F : TopCat.Sheaf CommRingCat X
    U : Opposite (TopologicalSpace.Opens ↑X)
    m : Submonoid ↑(F.presheaf.obj U) := iInf fun x => Submonoid.comap (F.presheaf …
    this : IsLocalization m (Localization m)
    ⊢ Function.Injective ⇑(algebraMap (↑(F.presheaf.obj U)) (Localization m))
  -/
  refine IsLocalization.injective (M := m) (S := Localization m) ?_
  /-
    case i
    X : TopCat
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    F✝ : TopCat.Presheaf CommRingCat X
    G : F✝.SubmonoidPresheaf
    F : TopCat.Sheaf CommRingCat X
    U : Opposite (TopologicalSpace.Opens ↑X)
    m : Submonoid ↑(F.presheaf.obj U) := iInf fun x => Submonoid.comap (F.presheaf …
    this : IsLocalization m (Localization m)
    ⊢ LE.le m (nonZeroDivisors ↑(F.presheaf.obj U))
  -/
  intro s hs t e
  /-
    case i
    X : TopCat
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    F✝ : TopCat.Presheaf CommRingCat X
    G : F✝.SubmonoidPresheaf
    F : TopCat.Sheaf CommRingCat X
    U : Opposite (TopologicalSpace.Opens ↑X)
    m : Submonoid ↑(F.presheaf.obj U) := iInf fun x => Submonoid.comap (F.presheaf …
    this : IsLocalization m (Localization m)
    s : ↑(F.presheaf.obj U)
    hs : Membership.mem m s
    t : ↑(F.presheaf.obj U)
    e : Eq (HMul.hMul t s) 0
    ⊢ Eq t 0
  -/
  apply section_ext F (unop U)
  /-
    case i.h
    X : TopCat
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    F✝ : TopCat.Presheaf CommRingCat X
    G : F✝.SubmonoidPresheaf
    F : TopCat.Sheaf CommRingCat X
    U : Opposite (TopologicalSpace.Opens ↑X)
    m : Submonoid ↑(F.presheaf.obj U) := iInf fun x => Submonoid.comap (F.presheaf …
    this : IsLocalization m (Localization m)
    s : ↑(F.presheaf.obj U)
    hs : Membership.mem m s
    t : ↑(F.presheaf.obj U)
    e : Eq (HMul.hMul t s) 0
    ⊢ ∀ (x : ↑X) (hx : Membership.mem (Opposite.unop U) x), Eq ((F.presheaf.germ ( …
  -/
  intro x hx
  /-
    case i.h
    X : TopCat
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    F✝ : TopCat.Presheaf CommRingCat X
    G : F✝.SubmonoidPresheaf
    F : TopCat.Sheaf CommRingCat X
    U : Opposite (TopologicalSpace.Opens ↑X)
    m : Submonoid ↑(F.presheaf.obj U) := iInf fun x => Submonoid.comap (F.presheaf …
    this : IsLocalization m (Localization m)
    s : ↑(F.presheaf.obj U)
    hs : Membership.mem m s
    t : ↑(F.presheaf.obj U)
    e : Eq (HMul.hMul t s) 0
    x : ↑X
    hx : Membership.mem (Opposite.unop U) x
    ⊢ Eq ((F.presheaf.germ (Opposite.unop U) x hx) t) ((F.presheaf.germ (Opposite. …
  -/
  show (F.presheaf.germ (unop U) x hx) t = (F.presheaf.germ (unop U) x hx) 0
  /-
    case i.h
    X : TopCat
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    F✝ : TopCat.Presheaf CommRingCat X
    G : F✝.SubmonoidPresheaf
    F : TopCat.Sheaf CommRingCat X
    U : Opposite (TopologicalSpace.Opens ↑X)
    m : Submonoid ↑(F.presheaf.obj U) := iInf fun x => Submonoid.comap (F.presheaf …
    this : IsLocalization m (Localization m)
    s : ↑(F.presheaf.obj U)
    hs : Membership.mem m s
    t : ↑(F.presheaf.obj U)
    e : Eq (HMul.hMul t s) 0
    x : ↑X
    hx : Membership.mem (Opposite.unop U) x
    ⊢ Eq ((F.presheaf.germ (Opposite.unop U) x hx).hom t) ((F.presheaf.germ (Oppos …
  -/
  rw [RingHom.map_zero]
  /-
    case i.h
    X : TopCat
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    F✝ : TopCat.Presheaf CommRingCat X
    G : F✝.SubmonoidPresheaf
    F : TopCat.Sheaf CommRingCat X
    U : Opposite (TopologicalSpace.Opens ↑X)
    m : Submonoid ↑(F.presheaf.obj U) := iInf fun x => Submonoid.comap (F.presheaf …
    this : IsLocalization m (Localization m)
    s : ↑(F.presheaf.obj U)
    hs : Membership.mem m s
    t : ↑(F.presheaf.obj U)
    e : Eq (HMul.hMul t s) 0
    x : ↑X
    hx : Membership.mem (Opposite.unop U) x
    ⊢ Eq ((F.presheaf.germ (Opposite.unop U) x hx).hom t) 0
  -/
  apply Submonoid.mem_iInf.mp hs ⟨x, hx⟩
  /-
    case i.h.a
    X : TopCat
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ConcreteCategory C
    F✝ : TopCat.Presheaf CommRingCat X
    G : F✝.SubmonoidPresheaf
    F : TopCat.Sheaf CommRingCat X
    U : Opposite (TopologicalSpace.Opens ↑X)
    m : Submonoid ↑(F.presheaf.obj U) := iInf fun x => Submonoid.comap (F.presheaf …
    this : IsLocalization m (Localization m)
    s : ↑(F.presheaf.obj U)
    hs : Membership.mem m s
    t : ↑(F.presheaf.obj U)
    e : Eq (HMul.hMul t s) 0
    x : ↑X
    hx : Membership.mem (Opposite.unop U) x
    ⊢ Eq (HMul.hMul ((F.presheaf.germ (Opposite.unop U) x hx).hom t) ((F.presheaf. …
  -/
  rw [← map_mul, e, map_zero]
  /-
    🎉 no goals
  -/


/-- The (bundled) commutative ring of continuous functions from a topological space
to a topological commutative ring, with pointwise multiplication. -/
def continuousFunctions (X : TopCat.{v}ᵒᵖ) (R : TopCommRingCat.{v}) : CommRingCat.{v} :=
  -- Porting note: Lean did not see through that `X.unop ⟶ R` is just continuous functions
  -- hence forms a ring
  @CommRingCat.of (X.unop ⟶ (forget₂ TopCommRingCat TopCat).obj R) <|
    inferInstanceAs (CommRing (ContinuousMap _ _))


instance (X : TopCat.{v}ᵒᵖ) (R : TopCommRingCat.{v}) :
    CommRing (unop X ⟶ (forget₂ TopCommRingCat TopCat).obj R) :=
  inferInstanceAs (CommRing (ContinuousMap _ _))


/-- Pulling back functions into a topological ring along a continuous map is a ring homomorphism. -/
def pullback {X Y : TopCatᵒᵖ} (f : X ⟶ Y) (R : TopCommRingCat) :
    continuousFunctions X R ⟶ continuousFunctions Y R := CommRingCat.ofHom
  { toFun g := f.unop ≫ g
    map_one' := rfl
    map_zero' := rfl
                   /-
                     X✝ : TopCat
                     X Y : Opposite TopCat
                     f : Quiver.Hom X Y
                     R : TopCommRingCat
                     ⊢ ∀ (x y : Quiver.Hom (Opposite.unop X) ((CategoryTheory.forget₂ TopCommRingCa …
                   -/
                   /-
                     X✝ : TopCat
                     X Y : Opposite TopCat
                     f : Quiver.Hom X Y
                     R : TopCommRingCat
                     ⊢ ∀ (x y : Quiver.Hom (Opposite.unop X) ((CategoryTheory.forget₂ TopCommRingCa …
                   -/
    map_add' := by aesop_cat
                   /-
                     🎉 no goals
                   -/
                   /-
                     🎉 no goals
                   -/
    map_mul' := by aesop_cat }


/-- A homomorphism of topological rings can be postcomposed with functions from a source space `X`;
this is a ring homomorphism (with respect to the pointwise ring operations on functions). -/
def map (X : TopCat.{u}ᵒᵖ) {R S : TopCommRingCat.{u}} (φ : R ⟶ S) :
    continuousFunctions X R ⟶ continuousFunctions X S := CommRingCat.ofHom
  { toFun g := g ≫ (forget₂ TopCommRingCat TopCat).map φ
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` tactic does not work, since Lean can't see through `R ⟶ S` is just
    -- continuous ring homomorphism
    map_one' := ContinuousMap.ext fun _ => φ.1.map_one
    map_zero' := ContinuousMap.ext fun _ => φ.1.map_zero
    map_add' := fun _ _ => ContinuousMap.ext fun _ => φ.1.map_add _ _
    map_mul' := fun _ _ => ContinuousMap.ext fun _ => φ.1.map_mul _ _ }


/-- An upgraded version of the Yoneda embedding, observing that the continuous maps
from `X : TopCat` to `R : TopCommRingCat` form a commutative ring, functorial in both `X` and
`R`. -/
def commRingYoneda : TopCommRingCat.{u} ⥤ TopCat.{u}ᵒᵖ ⥤ CommRingCat.{u} where
  obj R :=
    { obj := fun X => continuousFunctions X R
      map := fun {_ _} f => continuousFunctions.pullback f R
      map_id := fun X => by
        /-
          X✝ : TopCat
          R : TopCommRingCat
          X : Opposite TopCat
          ⊢ Eq ({ obj := fun X => TopCat.continuousFunctions X R, map := fun {x x_1} f = …
        -/
        ext
        /-
          case hf.a
          X✝ : TopCat
          R : TopCommRingCat
          X : Opposite TopCat
          x✝ : ↑({ obj := fun X => TopCat.continuousFunctions X R, map := fun {x x_1} f  …
          ⊢ Eq (({ obj := fun X => TopCat.continuousFunctions X R, map := fun {x x_1} f  …
        -/
        rfl
        /-
          🎉 no goals
        -/
      map_comp := fun {_ _ _} _ _ => rfl }
  map {_ _} φ :=
    { app := fun X => continuousFunctions.map X φ
      naturality := fun _ _ _ => rfl }
  map_id X := by
    /-
      X✝ : TopCat
      X : TopCommRingCat
      ⊢ Eq ({ obj := fun R => { obj := fun X => TopCat.continuousFunctions X R, map  …
    -/
    ext
    /-
      case w.h.hf.a
      X✝ : TopCat
      X : TopCommRingCat
      x✝¹ : Opposite TopCat
      x✝ : ↑(({ obj := fun R => { obj := fun X => TopCat.continuousFunctions X R, ma …
      ⊢ Eq ((({ obj := fun R => { obj := fun X => TopCat.continuousFunctions X R, ma …
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_comp {_ _ _} _ _ := rfl


/-- The presheaf (of commutative rings), consisting of functions on an open set `U ⊆ X` with
values in some topological commutative ring `T`.

For example, we could construct the presheaf of continuous complex valued functions of `X` as
```
presheafToTopCommRing X (TopCommRingCat.of ℂ)
```
(this requires `import Topology.Instances.Complex`).
-/
def presheafToTopCommRing (T : TopCommRingCat.{v}) : X.Presheaf CommRingCat.{v} :=
  (Opens.toTopCat X).op ⋙ commRingYoneda.obj T


instance algebra_section_stalk (F : X.Presheaf CommRingCat) {U : Opens X} (x : U) :
    Algebra (F.obj <| op U) (F.stalk x) :=
  (F.germ U x.1 x.2).hom.toAlgebra


@[simp]
theorem stalk_open_algebraMap {X : TopCat} (F : X.Presheaf CommRingCat) {U : Opens X} (x : U) :
    algebraMap (F.obj <| op U) (F.stalk x) = (F.germ U x.1 x.2).hom :=
  rfl


/-- `F(U ⊔ V)` is isomorphic to the `eq_locus` of the two maps `F(U) × F(V) ⟶ F(U ⊓ V)`. -/
def objSupIsoProdEqLocus {X : TopCat} (F : X.Sheaf CommRingCat) (U V : Opens X) :
    F.1.obj (op <| U ⊔ V) ≅ CommRingCat.of <|
    -- Porting note: Lean 3 is able to figure out the ring homomorphism automatically
    RingHom.eqLocus
      (RingHom.comp (F.val.map (homOfLE inf_le_left : U ⊓ V ⟶ U).op).hom
        (RingHom.fst (F.val.obj <| op U) (F.val.obj <| op V)))
      (RingHom.comp (F.val.map (homOfLE inf_le_right : U ⊓ V ⟶ V).op).hom
        (RingHom.snd (F.val.obj <| op U) (F.val.obj <| op V))) :=
  (F.isLimitPullbackCone U V).conePointUniqueUpToIso (CommRingCat.pullbackConeIsLimit _ _)


theorem objSupIsoProdEqLocus_hom_fst {X : TopCat} (F : X.Sheaf CommRingCat) (U V : Opens X) (x) :
    ((F.objSupIsoProdEqLocus U V).hom x).1.fst = F.1.map (homOfLE le_sup_left).op x :=
  ConcreteCategory.congr_hom
    ((F.isLimitPullbackCone U V).conePointUniqueUpToIso_hom_comp
      (CommRingCat.pullbackConeIsLimit _ _) WalkingCospan.left)
    x


theorem objSupIsoProdEqLocus_hom_snd {X : TopCat} (F : X.Sheaf CommRingCat) (U V : Opens X) (x) :
    ((F.objSupIsoProdEqLocus U V).hom x).1.snd = F.1.map (homOfLE le_sup_right).op x :=
  ConcreteCategory.congr_hom
    ((F.isLimitPullbackCone U V).conePointUniqueUpToIso_hom_comp
      (CommRingCat.pullbackConeIsLimit _ _) WalkingCospan.right)
    x


theorem objSupIsoProdEqLocus_inv_fst {X : TopCat} (F : X.Sheaf CommRingCat) (U V : Opens X) (x) :
    F.1.map (homOfLE le_sup_left).op ((F.objSupIsoProdEqLocus U V).inv x) = x.1.1 :=
  ConcreteCategory.congr_hom
    ((F.isLimitPullbackCone U V).conePointUniqueUpToIso_inv_comp
      (CommRingCat.pullbackConeIsLimit _ _) WalkingCospan.left)
    x


theorem objSupIsoProdEqLocus_inv_snd {X : TopCat} (F : X.Sheaf CommRingCat) (U V : Opens X) (x) :
    F.1.map (homOfLE le_sup_right).op ((F.objSupIsoProdEqLocus U V).inv x) = x.1.2 :=
  ConcreteCategory.congr_hom
    ((F.isLimitPullbackCone U V).conePointUniqueUpToIso_inv_comp
      (CommRingCat.pullbackConeIsLimit _ _) WalkingCospan.right)
    x


theorem objSupIsoProdEqLocus_inv_eq_iff {X : TopCat.{u}} (F : X.Sheaf CommRingCat.{u})
    {U V W UW VW : Opens X} (e : W ≤ U ⊔ V) (x) (y : F.1.obj (op W))
    (h₁ : UW = U ⊓ W) (h₂ : VW = V ⊓ W) :
    F.1.map (homOfLE e).op ((F.objSupIsoProdEqLocus U V).inv x) = y ↔
    F.1.map (homOfLE (h₁ ▸ inf_le_left : UW ≤ U)).op x.1.1 =
      F.1.map (homOfLE <| h₁ ▸ inf_le_right).op y ∧
    F.1.map (homOfLE (h₂ ▸ inf_le_left : VW ≤ V)).op x.1.2 =
      F.1.map (homOfLE <| h₂ ▸ inf_le_right).op y := by
  /-
    X : TopCat
    F : TopCat.Sheaf CommRingCat X
    U V W UW VW : TopologicalSpace.Opens ↑X
    e : LE.le W (Max.max U V)
    x : ↑(CommRingCat.of (Subtype fun x => Membership.mem (((F.val.map (CategoryTh …
    y : ↑(F.val.obj { unop := W })
    h₁ : Eq UW (Min.min U W)
    h₂ : Eq VW (Min.min V W)
    ⊢ Iff (Eq ((F.val.map (CategoryTheory.homOfLE e).op).hom ((F.objSupIsoProdEqLo …
  -/
  subst h₁ h₂
  /-
    X : TopCat
    F : TopCat.Sheaf CommRingCat X
    U V W : TopologicalSpace.Opens ↑X
    e : LE.le W (Max.max U V)
    x : ↑(CommRingCat.of (Subtype fun x => Membership.mem (((F.val.map (CategoryTh …
    y : ↑(F.val.obj { unop := W })
    ⊢ Iff (Eq ((F.val.map (CategoryTheory.homOfLE e).op).hom ((F.objSupIsoProdEqLo …
  -/
  constructor
    /-
      case mp
      X : TopCat
      F : TopCat.Sheaf CommRingCat X
      U V W : TopologicalSpace.Opens ↑X
      e : LE.le W (Max.max U V)
      x : ↑(CommRingCat.of (Subtype fun x => Membership.mem (((F.val.map (CategoryTh …
      y : ↑(F.val.obj { unop := W })
      ⊢ Eq ((F.val.map (CategoryTheory.homOfLE e).op).hom ((F.objSupIsoProdEqLocus U …
    -/
  · rintro rfl
    /-
      case mp
      X : TopCat
      F : TopCat.Sheaf CommRingCat X
      U V W : TopologicalSpace.Opens ↑X
      e : LE.le W (Max.max U V)
      x : ↑(CommRingCat.of (Subtype fun x => Membership.mem (((F.val.map (CategoryTh …
      ⊢ And (Eq ((F.val.map (CategoryTheory.homOfLE ⋯).op).hom (↑x).1) ((F.val.map ( …
    -/
    rw [← TopCat.Sheaf.objSupIsoProdEqLocus_inv_fst, ← TopCat.Sheaf.objSupIsoProdEqLocus_inv_snd]
    -- `simp` doesn't see through the type equality of objects in `CommRingCat`, so use `rw` https://github.com/leanprover-community/mathlib4/pull/8386
    /-
      case mp
      X : TopCat
      F : TopCat.Sheaf CommRingCat X
      U V W : TopologicalSpace.Opens ↑X
      e : LE.le W (Max.max U V)
      x : ↑(CommRingCat.of (Subtype fun x => Membership.mem (((F.val.map (CategoryTh …
      ⊢ And (Eq ((F.val.map (CategoryTheory.homOfLE ⋯).op).hom ((F.val.map (Category …
    -/
    repeat rw [← CommRingCat.comp_apply]
    /-
      case mp
      X : TopCat
      F : TopCat.Sheaf CommRingCat X
      U V W : TopologicalSpace.Opens ↑X
      e : LE.le W (Max.max U V)
      x : ↑(CommRingCat.of (Subtype fun x => Membership.mem (((F.val.map (CategoryTh …
      ⊢ And (Eq ((CategoryTheory.CategoryStruct.comp (F.objSupIsoProdEqLocus U V).in …
    -/
    simp only [← Functor.map_comp, ← op_comp, Category.assoc, homOfLE_comp, and_self]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : TopCat
      F : TopCat.Sheaf CommRingCat X
      U V W : TopologicalSpace.Opens ↑X
      e : LE.le W (Max.max U V)
      x : ↑(CommRingCat.of (Subtype fun x => Membership.mem (((F.val.map (CategoryTh …
      y : ↑(F.val.obj { unop := W })
      ⊢ And (Eq ((F.val.map (CategoryTheory.homOfLE ⋯).op).hom (↑x).1) ((F.val.map ( …
    -/
  · rintro ⟨e₁, e₂⟩
    refine F.eq_of_locally_eq₂
      (homOfLE (inf_le_right : U ⊓ W ≤ W)) (homOfLE (inf_le_right : V ⊓ W ≤ W)) ?_ _ _ ?_ ?_
      /-
        case mpr.intro.refine_1
        X : TopCat
        F : TopCat.Sheaf CommRingCat X
        U V W : TopologicalSpace.Opens ↑X
        e : LE.le W (Max.max U V)
        x : ↑(CommRingCat.of (Subtype fun x => Membership.mem (((F.val.map (CategoryTh …
        y : ↑(F.val.obj { unop := W })
        e₁ : Eq ((F.val.map (CategoryTheory.homOfLE ⋯).op).hom (↑x).1) ((F.val.map (Ca …
        e₂ : Eq ((F.val.map (CategoryTheory.homOfLE ⋯).op).hom (↑x).2) ((F.val.map (Ca …
        ⊢ LE.le W (Max.max (Min.min U W) (Min.min V W))
      -/
    · rw [← inf_sup_right]
      /-
        case mpr.intro.refine_1
        X : TopCat
        F : TopCat.Sheaf CommRingCat X
        U V W : TopologicalSpace.Opens ↑X
        e : LE.le W (Max.max U V)
        x : ↑(CommRingCat.of (Subtype fun x => Membership.mem (((F.val.map (CategoryTh …
        y : ↑(F.val.obj { unop := W })
        e₁ : Eq ((F.val.map (CategoryTheory.homOfLE ⋯).op).hom (↑x).1) ((F.val.map (Ca …
        e₂ : Eq ((F.val.map (CategoryTheory.homOfLE ⋯).op).hom (↑x).2) ((F.val.map (Ca …
        ⊢ LE.le W (Min.min (Max.max U V) W)
      -/
      exact le_inf e le_rfl
      /-
        🎉 no goals
      -/
    · change (F.val.map _)
        ((F.val.map (homOfLE e).op).hom ((F.objSupIsoProdEqLocus U V).inv.hom x)) = (F.val.map _) y
      /-
        case mpr.intro.refine_2
        X : TopCat
        F : TopCat.Sheaf CommRingCat X
        U V W : TopologicalSpace.Opens ↑X
        e : LE.le W (Max.max U V)
        x : ↑(CommRingCat.of (Subtype fun x => Membership.mem (((F.val.map (CategoryTh …
        y : ↑(F.val.obj { unop := W })
        e₁ : Eq ((F.val.map (CategoryTheory.homOfLE ⋯).op).hom (↑x).1) ((F.val.map (Ca …
        e₂ : Eq ((F.val.map (CategoryTheory.homOfLE ⋯).op).hom (↑x).2) ((F.val.map (Ca …
        ⊢ Eq ((F.val.map (CategoryTheory.homOfLE ⋯).op).hom ((F.val.map (CategoryTheor …
      -/
      rw [← e₁, ← TopCat.Sheaf.objSupIsoProdEqLocus_inv_fst]
      -- `simp` doesn't see through the type equality of objects in `CommRingCat`, so use `rw` https://github.com/leanprover-community/mathlib4/pull/8386
      /-
        case mpr.intro.refine_2
        X : TopCat
        F : TopCat.Sheaf CommRingCat X
        U V W : TopologicalSpace.Opens ↑X
        e : LE.le W (Max.max U V)
        x : ↑(CommRingCat.of (Subtype fun x => Membership.mem (((F.val.map (CategoryTh …
        y : ↑(F.val.obj { unop := W })
        e₁ : Eq ((F.val.map (CategoryTheory.homOfLE ⋯).op).hom (↑x).1) ((F.val.map (Ca …
        e₂ : Eq ((F.val.map (CategoryTheory.homOfLE ⋯).op).hom (↑x).2) ((F.val.map (Ca …
        ⊢ Eq ((F.val.map (CategoryTheory.homOfLE ⋯).op).hom ((F.val.map (CategoryTheor …
      -/
      repeat rw [← CommRingCat.comp_apply]
      /-
        case mpr.intro.refine_2
        X : TopCat
        F : TopCat.Sheaf CommRingCat X
        U V W : TopologicalSpace.Opens ↑X
        e : LE.le W (Max.max U V)
        x : ↑(CommRingCat.of (Subtype fun x => Membership.mem (((F.val.map (CategoryTh …
        y : ↑(F.val.obj { unop := W })
        e₁ : Eq ((F.val.map (CategoryTheory.homOfLE ⋯).op).hom (↑x).1) ((F.val.map (Ca …
        e₂ : Eq ((F.val.map (CategoryTheory.homOfLE ⋯).op).hom (↑x).2) ((F.val.map (Ca …
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (F.objSupIsoProdEqLocus U V).inv (Ca …
      -/
      simp only [← Functor.map_comp, ← op_comp, Category.assoc, homOfLE_comp]
      /-
        🎉 no goals
      -/
    · show (F.val.map _)
        ((F.val.map (homOfLE e).op).hom ((F.objSupIsoProdEqLocus U V).inv.hom x)) = (F.val.map _) y
      /-
        case mpr.intro.refine_3
        X : TopCat
        F : TopCat.Sheaf CommRingCat X
        U V W : TopologicalSpace.Opens ↑X
        e : LE.le W (Max.max U V)
        x : ↑(CommRingCat.of (Subtype fun x => Membership.mem (((F.val.map (CategoryTh …
        y : ↑(F.val.obj { unop := W })
        e₁ : Eq ((F.val.map (CategoryTheory.homOfLE ⋯).op).hom (↑x).1) ((F.val.map (Ca …
        e₂ : Eq ((F.val.map (CategoryTheory.homOfLE ⋯).op).hom (↑x).2) ((F.val.map (Ca …
        ⊢ Eq ((F.val.map (CategoryTheory.homOfLE ⋯).op).hom ((F.val.map (CategoryTheor …
      -/
      rw [← e₂, ← TopCat.Sheaf.objSupIsoProdEqLocus_inv_snd]
      -- `simp` doesn't see through the type equality of objects in `CommRingCat`, so use `rw` https://github.com/leanprover-community/mathlib4/pull/8386
      /-
        case mpr.intro.refine_3
        X : TopCat
        F : TopCat.Sheaf CommRingCat X
        U V W : TopologicalSpace.Opens ↑X
        e : LE.le W (Max.max U V)
        x : ↑(CommRingCat.of (Subtype fun x => Membership.mem (((F.val.map (CategoryTh …
        y : ↑(F.val.obj { unop := W })
        e₁ : Eq ((F.val.map (CategoryTheory.homOfLE ⋯).op).hom (↑x).1) ((F.val.map (Ca …
        e₂ : Eq ((F.val.map (CategoryTheory.homOfLE ⋯).op).hom (↑x).2) ((F.val.map (Ca …
        ⊢ Eq ((F.val.map (CategoryTheory.homOfLE ⋯).op).hom ((F.val.map (CategoryTheor …
      -/
      repeat rw [← CommRingCat.comp_apply]
      /-
        case mpr.intro.refine_3
        X : TopCat
        F : TopCat.Sheaf CommRingCat X
        U V W : TopologicalSpace.Opens ↑X
        e : LE.le W (Max.max U V)
        x : ↑(CommRingCat.of (Subtype fun x => Membership.mem (((F.val.map (CategoryTh …
        y : ↑(F.val.obj { unop := W })
        e₁ : Eq ((F.val.map (CategoryTheory.homOfLE ⋯).op).hom (↑x).1) ((F.val.map (Ca …
        e₂ : Eq ((F.val.map (CategoryTheory.homOfLE ⋯).op).hom (↑x).2) ((F.val.map (Ca …
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (F.objSupIsoProdEqLocus U V).inv (Ca …
      -/
      simp only [← Functor.map_comp, ← op_comp, Category.assoc, homOfLE_comp]
      /-
        🎉 no goals
      -/



instance opensHomHasCoeToFun {U V : Opens X} : CoeFun (U ⟶ V) fun _ => U → V :=
  ⟨fun f x => ⟨x, f.le x.2⟩⟩


/-- The inclusion `U ⊓ V ⟶ U` as a morphism in the category of open sets.
-/
noncomputable def infLELeft (U V : Opens X) : U ⊓ V ⟶ U :=
  inf_le_left.hom


/-- The inclusion `U ⊓ V ⟶ V` as a morphism in the category of open sets.
-/
noncomputable def infLERight (U V : Opens X) : U ⊓ V ⟶ V :=
  inf_le_right.hom


/-- The inclusion `U i ⟶ iSup U` as a morphism in the category of open sets.
-/
noncomputable def leSupr {ι : Type*} (U : ι → Opens X) (i : ι) : U i ⟶ iSup U :=
  (le_iSup U i).hom


/-- The inclusion `⊥ ⟶ U` as a morphism in the category of open sets.
-/
noncomputable def botLE (U : Opens X) : ⊥ ⟶ U :=
  bot_le.hom


/-- The inclusion `U ⟶ ⊤` as a morphism in the category of open sets.
-/
noncomputable def leTop (U : Opens X) : U ⟶ ⊤ :=
  le_top.hom

-- We do not mark this as a simp lemma because it breaks open `x`.
-- Nevertheless, it is useful in `SheafOfFunctions`.

theorem infLELeft_apply (U V : Opens X) (x) :
    (infLELeft U V) x = ⟨x.1, (@inf_le_left _ _ U V : _ ≤ _) x.2⟩ :=
  rfl


@[simp]
theorem infLELeft_apply_mk (U V : Opens X) (x) (m) :
    (infLELeft U V) ⟨x, m⟩ = ⟨x, (@inf_le_left _ _ U V : _ ≤ _) m⟩ :=
  rfl


@[simp]
theorem leSupr_apply_mk {ι : Type*} (U : ι → Opens X) (i : ι) (x) (m) :
    (leSupr U i) ⟨x, m⟩ = ⟨x, (le_iSup U i : _) m⟩ :=
  rfl


/-- The functor from open sets in `X` to `TopCat`,
realising each open set as a topological space itself.
-/
def toTopCat (X : TopCat.{u}) : Opens X ⥤ TopCat where
  obj U := ⟨U, inferInstance⟩
  map i := ⟨fun x ↦ ⟨x.1, i.le x.2⟩, IsEmbedding.subtypeVal.continuous_iff.2 continuous_induced_dom⟩


@[simp]
theorem toTopCat_map (X : TopCat.{u}) {U V : Opens X} {f : U ⟶ V} {x} {h} :
    ((toTopCat X).map f) ⟨x, h⟩ = ⟨x, f.le h⟩ :=
  rfl


/-- The inclusion map from an open subset to the whole space, as a morphism in `TopCat`.
-/
@[simps (config := .asFn)]
def inclusion' {X : TopCat.{u}} (U : Opens X) : (toTopCat X).obj U ⟶ X where
  toFun := _
  continuous_toFun := continuous_subtype_val


@[simp]
theorem coe_inclusion' {X : TopCat} {U : Opens X} :
    (inclusion' U : U → X) = Subtype.val := rfl


theorem isOpenEmbedding {X : TopCat.{u}} (U : Opens X) : IsOpenEmbedding (inclusion' U) :=
  U.2.isOpenEmbedding_subtypeVal


@[deprecated (since := "2024-10-18")]
alias openEmbedding := isOpenEmbedding


/-- The inclusion of the top open subset (i.e. the whole space) is an isomorphism.
-/
def inclusionTopIso (X : TopCat.{u}) : (toTopCat X).obj ⊤ ≅ X where
  hom := inclusion' ⊤
  inv := ⟨fun x => ⟨x, trivial⟩, continuous_def.2 fun _ ⟨_, hS, hSU⟩ => hSU ▸ hS⟩


/-- `Opens.map f` gives the functor from open sets in Y to open set in X,
    given by taking preimages under f. -/
def map (f : X ⟶ Y) : Opens Y ⥤ Opens X where
  obj U := ⟨f ⁻¹' (U : Set Y), U.isOpen.preimage f.continuous⟩
  map i := ⟨⟨fun _ h => i.le h⟩⟩


@[simp]
theorem map_coe (f : X ⟶ Y) (U : Opens Y) : ((map f).obj U : Set X) = f ⁻¹' (U : Set Y) :=
  rfl


@[simp]
theorem map_obj (f : X ⟶ Y) (U) (p) : (map f).obj ⟨U, p⟩ = ⟨f ⁻¹' U, p.preimage f.continuous⟩ :=
  rfl


@[simp]
lemma map_homOfLE (f : X ⟶ Y) {U V : Opens Y} (e : U ≤ V) :
    (TopologicalSpace.Opens.map f).map (homOfLE e) =
      homOfLE (show (Opens.map f).obj U ≤ (Opens.map f).obj V from fun _ hx ↦ e hx) :=
  rfl


@[simp]
theorem map_id_obj (U : Opens X) : (map (𝟙 X)).obj U = U :=
  let ⟨_, _⟩ := U
  rfl


@[simp 1100]
theorem map_id_obj' (U) (p) : (map (𝟙 X)).obj ⟨U, p⟩ = ⟨U, p⟩ :=
  rfl


@[simp 1100]
theorem map_id_obj_unop (U : (Opens X)ᵒᵖ) : (map (𝟙 X)).obj (unop U) = unop U :=
  let ⟨_, _⟩ := U.unop
  rfl


@[simp 1100]
                                                                         /-
                                                                           X : TopCat
                                                                           U : Opposite (TopologicalSpace.Opens ↑X)
                                                                           ⊢ Eq ((TopologicalSpace.Opens.map (CategoryTheory.CategoryStruct.id X)).op.obj …
                                                                         -/
theorem op_map_id_obj (U : (Opens X)ᵒᵖ) : (map (𝟙 X)).op.obj U = U := by simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp]
lemma map_top (f : X ⟶ Y) : (Opens.map f).obj ⊤ = ⊤ := rfl


/-- The inclusion `U ⟶ (map f).obj ⊤` as a morphism in the category of open sets.
-/
noncomputable def leMapTop (f : X ⟶ Y) (U : Opens X) : U ⟶ (map f).obj ⊤ :=
  leTop U


@[simp]
theorem map_comp_obj (f : X ⟶ Y) (g : Y ⟶ Z) (U) :
    (map (f ≫ g)).obj U = (map f).obj ((map g).obj U) :=
  rfl


@[simp]
theorem map_comp_obj' (f : X ⟶ Y) (g : Y ⟶ Z) (U) (p) :
    (map (f ≫ g)).obj ⟨U, p⟩ = (map f).obj ((map g).obj ⟨U, p⟩) :=
  rfl


@[simp]
theorem map_comp_map (f : X ⟶ Y) (g : Y ⟶ Z) {U V} (i : U ⟶ V) :
    (map (f ≫ g)).map i = (map f).map ((map g).map i) :=
  rfl


@[simp]
theorem map_comp_obj_unop (f : X ⟶ Y) (g : Y ⟶ Z) (U) :
    (map (f ≫ g)).obj (unop U) = (map f).obj ((map g).obj (unop U)) :=
  rfl


@[simp]
theorem op_map_comp_obj (f : X ⟶ Y) (g : Y ⟶ Z) (U) :
    (map (f ≫ g)).op.obj U = (map f).op.obj ((map g).op.obj U) :=
  rfl


theorem map_iSup (f : X ⟶ Y) {ι : Type*} (U : ι → Opens Y) :
    (map f).obj (iSup U) = iSup ((map f).obj ∘ U) := by
  /-
    X Y : TopCat
    f : Quiver.Hom X Y
    ι : Type u_1
    U : ι → TopologicalSpace.Opens ↑Y
    ⊢ Eq ((TopologicalSpace.Opens.map f).obj (iSup U)) (iSup (Function.comp (Topol …
  -/
  ext1; rw [iSup_def, iSup_def, map_obj]
  /-
    case h
    X Y : TopCat
    f : Quiver.Hom X Y
    ι : Type u_1
    U : ι → TopologicalSpace.Opens ↑Y
    ⊢ Eq ↑{ carrier := Set.preimage (⇑f) (Set.iUnion fun i => ↑(U i)), is_open' := …
  -/
  dsimp; rw [Set.preimage_iUnion]
         /-
           🎉 no goals
         -/


/-- The functor `Opens X ⥤ Opens X` given by taking preimages under the identity function
is naturally isomorphic to the identity functor.
-/
@[simps]
def mapId : map (𝟙 X) ≅ 𝟭 (Opens X) where
  hom := { app := fun U => eqToHom (map_id_obj U) }
  inv := { app := fun U => eqToHom (map_id_obj U).symm }


theorem map_id_eq : map (𝟙 X) = 𝟭 (Opens X) := by
  /-
    X : TopCat
    ⊢ Eq (TopologicalSpace.Opens.map (CategoryTheory.CategoryStruct.id X)) (Catego …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The natural isomorphism between taking preimages under `f ≫ g`, and the composite
of taking preimages under `g`, then preimages under `f`.
-/
@[simps]
def mapComp (f : X ⟶ Y) (g : Y ⟶ Z) : map (f ≫ g) ≅ map g ⋙ map f where
  hom := { app := fun U => eqToHom (map_comp_obj f g U) }
  inv := { app := fun U => eqToHom (map_comp_obj f g U).symm }


theorem map_comp_eq (f : X ⟶ Y) (g : Y ⟶ Z) : map (f ≫ g) = map g ⋙ map f :=
  rfl

-- We could make `f g` implicit here, but it's nice to be able to see when
-- they are the identity (often!)

/-- If two continuous maps `f g : X ⟶ Y` are equal,
then the functors `Opens Y ⥤ Opens X` they induce are isomorphic.
-/
def mapIso (f g : X ⟶ Y) (h : f = g) : map f ≅ map g :=
                                           /-
                                             X Y Z : TopCat
                                             f g : Quiver.Hom X Y
                                             h : Eq f g
                                             U : TopologicalSpace.Opens ↑Y
                                             ⊢ Eq ((TopologicalSpace.Opens.map f).obj U) ((TopologicalSpace.Opens.map g).ob …
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
  NatIso.ofComponents fun U => eqToIso (by rw [congr_arg map h])
  /-
    🎉 no goals
  -/


theorem map_eq (f g : X ⟶ Y) (h : f = g) : map f = map g := by
  /-
    X Y : TopCat
    f g : Quiver.Hom X Y
    h : Eq f g
    ⊢ Eq (TopologicalSpace.Opens.map f) (TopologicalSpace.Opens.map g)
  -/
  subst h
  /-
    X Y : TopCat
    f : Quiver.Hom X Y
    ⊢ Eq (TopologicalSpace.Opens.map f) (TopologicalSpace.Opens.map f)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem mapIso_refl (f : X ⟶ Y) (h) : mapIso f f h = Iso.refl (map _) :=
  rfl


@[simp]
theorem mapIso_hom_app (f g : X ⟶ Y) (h : f = g) (U : Opens Y) :
                                           /-
                                             X Y Z : TopCat
                                             f g : Quiver.Hom X Y
                                             h : Eq f g
                                             U : TopologicalSpace.Opens ↑Y
                                             ⊢ Eq ((TopologicalSpace.Opens.map f).obj U) ((TopologicalSpace.Opens.map g).ob …
                                           -/
    (mapIso f g h).hom.app U = eqToHom (by rw [h]) :=
                                           /-
                                             🎉 no goals
                                           -/
  rfl


@[simp]
theorem mapIso_inv_app (f g : X ⟶ Y) (h : f = g) (U : Opens Y) :
                                           /-
                                             X Y Z : TopCat
                                             f g : Quiver.Hom X Y
                                             h : Eq f g
                                             U : TopologicalSpace.Opens ↑Y
                                             ⊢ Eq ((TopologicalSpace.Opens.map g).obj U) ((TopologicalSpace.Opens.map f).ob …
                                           -/
    (mapIso f g h).inv.app U = eqToHom (by rw [h]) :=
                                           /-
                                             🎉 no goals
                                           -/
  rfl


/-- A homeomorphism of spaces gives an equivalence of categories of open sets.

TODO: define `OrderIso.equivalence`, use it.
-/
@[simps]
def mapMapIso {X Y : TopCat.{u}} (H : X ≅ Y) : Opens Y ≌ Opens X where
  functor := map H.hom
  inverse := map H.inv
                                                      /-
                                                        X✝ Y✝ Z X Y : TopCat
                                                        H : CategoryTheory.Iso X Y
                                                        U : TopologicalSpace.Opens ↑Y
                                                        ⊢ Eq ((CategoryTheory.Functor.id (TopologicalSpace.Opens ↑Y)).obj U) (((Topolo …
                                                      -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  unitIso := NatIso.ofComponents fun U => eqToIso (by simp [map, Set.preimage_preimage])
             /-
               🎉 no goals
             -/
                                                        /-
                                                          X✝ Y✝ Z X Y : TopCat
                                                          H : CategoryTheory.Iso X Y
                                                          U : TopologicalSpace.Opens ↑X
                                                          ⊢ Eq (((TopologicalSpace.Opens.map H.inv).comp (TopologicalSpace.Opens.map H.h …
                                                        -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  counitIso := NatIso.ofComponents fun U => eqToIso (by simp [map, Set.preimage_preimage])
               /-
                 🎉 no goals
               -/


/-- An open map `f : X ⟶ Y` induces a functor `Opens X ⥤ Opens Y`.
-/
@[simps obj_coe]
def IsOpenMap.functor {X Y : TopCat} {f : X ⟶ Y} (hf : IsOpenMap f) : Opens X ⥤ Opens Y where
  obj U := ⟨f '' (U : Set X), hf (U : Set X) U.2⟩
  map h := ⟨⟨Set.image_subset _ h.down.down⟩⟩


/-- An open map `f : X ⟶ Y` induces an adjunction between `Opens X` and `Opens Y`.
-/
def IsOpenMap.adjunction {X Y : TopCat} {f : X ⟶ Y} (hf : IsOpenMap f) :
    hf.functor ⊣ Opens.map f where
  unit := { app := fun _ => homOfLE fun x hxU => ⟨x, hxU, rfl⟩ }
  counit := { app := fun _ => homOfLE fun _ ⟨_, hfxV, hxy⟩ => hxy ▸ hfxV }


instance IsOpenMap.functorFullOfMono {X Y : TopCat} {f : X ⟶ Y} (hf : IsOpenMap f) [H : Mono f] :
    hf.functor.Full where
  map_surjective i :=
    ⟨homOfLE fun x hx => by
      /-
        X Y : TopCat
        f : Quiver.Hom X Y
        hf : IsOpenMap ⇑f
        H : CategoryTheory.Mono f
        X✝ Y✝ : TopologicalSpace.Opens ↑X
        i : Quiver.Hom (hf.functor.obj X✝) (hf.functor.obj Y✝)
        x : ↑X
        hx : Membership.mem (↑X✝) x
        ⊢ Membership.mem (↑Y✝) x
      -/
      obtain ⟨y, hy, eq⟩ := i.le ⟨x, hx, rfl⟩
      /-
        case intro.intro
        X Y : TopCat
        f : Quiver.Hom X Y
        hf : IsOpenMap ⇑f
        H : CategoryTheory.Mono f
        X✝ Y✝ : TopologicalSpace.Opens ↑X
        i : Quiver.Hom (hf.functor.obj X✝) (hf.functor.obj Y✝)
        x : ↑X
        hx : Membership.mem (↑X✝) x
        y : ↑X
        hy : Membership.mem (↑Y✝) y
        eq : Eq (f y) (f x)
        ⊢ Membership.mem (↑Y✝) x
      -/
      exact (TopCat.mono_iff_injective f).mp H eq ▸ hy, rfl⟩
      /-
        🎉 no goals
      -/


instance IsOpenMap.functor_faithful {X Y : TopCat} {f : X ⟶ Y} (hf : IsOpenMap f) :
    hf.functor.Faithful where


lemma Topology.IsOpenEmbedding.functor_obj_injective {X Y : TopCat} {f : X ⟶ Y}
    (hf : IsOpenEmbedding f) : Function.Injective hf.isOpenMap.functor.obj :=
  fun _ _ e ↦ Opens.ext (Set.image_injective.mpr hf.injective (congr_arg (↑· : Opens Y → Set Y) e))


@[deprecated (since := "2024-10-18")]
alias OpenEmbedding.functor_obj_injective := IsOpenEmbedding.functor_obj_injective


/-- Given an inducing map `X ⟶ Y` and some `U : Opens X`, this is the union of all open sets
whose preimage is `U`. This is right adjoint to `Opens.map`. -/
@[nolint unusedArguments]
def functorObj {X Y : TopCat} {f : X ⟶ Y} (_ : IsInducing f) (U : Opens X) : Opens Y :=
  sSup { s : Opens Y | (Opens.map f).obj s = U }


lemma map_functorObj {X Y : TopCat} {f : X ⟶ Y} (hf : IsInducing f)
    (U : Opens X) :
    (Opens.map f).obj (hf.functorObj U) = U := by
  /-
    X Y : TopCat
    f : Quiver.Hom X Y
    hf : Topology.IsInducing ⇑f
    U : TopologicalSpace.Opens ↑X
    ⊢ Eq ((TopologicalSpace.Opens.map f).obj (hf.functorObj U)) U
  -/
  apply le_antisymm
    /-
      case a
      X Y : TopCat
      f : Quiver.Hom X Y
      hf : Topology.IsInducing ⇑f
      U : TopologicalSpace.Opens ↑X
      ⊢ LE.le ((TopologicalSpace.Opens.map f).obj (hf.functorObj U)) U
    -/
  · rintro x ⟨_, ⟨s, rfl⟩, _, ⟨rfl : _ = U, rfl⟩, hx : f x ∈ s⟩; exact hx
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
    /-
      case a
      X Y : TopCat
      f : Quiver.Hom X Y
      hf : Topology.IsInducing ⇑f
      U : TopologicalSpace.Opens ↑X
      ⊢ LE.le U ((TopologicalSpace.Opens.map f).obj (hf.functorObj U))
    -/
  · intros x hx
    /-
      case a
      X Y : TopCat
      f : Quiver.Hom X Y
      hf : Topology.IsInducing ⇑f
      U : TopologicalSpace.Opens ↑X
      x : ↑X
      hx : Membership.mem (↑U) x
      ⊢ Membership.mem (↑((TopologicalSpace.Opens.map f).obj (hf.functorObj U))) x
    -/
    obtain ⟨U, hU⟩ := U
    /-
      case a.mk
      X Y : TopCat
      f : Quiver.Hom X Y
      hf : Topology.IsInducing ⇑f
      x : ↑X
      U : Set ↑X
      hU : IsOpen U
      hx : Membership.mem (↑{ carrier := U, is_open' := hU }) x
      ⊢ Membership.mem (↑((TopologicalSpace.Opens.map f).obj (hf.functorObj { carrie …
    -/
    obtain ⟨t, ht, rfl⟩ := hf.isOpen_iff.mp hU
    /-
      case a.mk.intro.intro
      X Y : TopCat
      f : Quiver.Hom X Y
      hf : Topology.IsInducing ⇑f
      x : ↑X
      t : Set ↑Y
      ht : IsOpen t
      hU : IsOpen (Set.preimage (⇑f) t)
      hx : Membership.mem (↑{ carrier := Set.preimage (⇑f) t, is_open' := hU }) x
      ⊢ Membership.mem (↑((TopologicalSpace.Opens.map f).obj (hf.functorObj { carrie …
    -/
    exact Opens.mem_sSup.mpr ⟨⟨_, ht⟩, rfl, hx⟩
    /-
      🎉 no goals
    -/


lemma mem_functorObj_iff {X Y : TopCat} {f : X ⟶ Y} (hf : IsInducing f) (U : Opens X)
    {x : X} : f x ∈ hf.functorObj U ↔ x ∈ U := by
  /-
    X Y : TopCat
    f : Quiver.Hom X Y
    hf : Topology.IsInducing ⇑f
    U : TopologicalSpace.Opens ↑X
    x : ↑X
    ⊢ Iff (Membership.mem (hf.functorObj U) (f x)) (Membership.mem U x)
  -/
  conv_rhs => rw [← hf.map_functorObj U]
  /-
    X Y : TopCat
    f : Quiver.Hom X Y
    hf : Topology.IsInducing ⇑f
    U : TopologicalSpace.Opens ↑X
    x : ↑X
    ⊢ Iff (Membership.mem (hf.functorObj U) (f x)) (Membership.mem ((TopologicalSp …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma le_functorObj_iff {X Y : TopCat} {f : X ⟶ Y} (hf : IsInducing f) {U : Opens X}
    {V : Opens Y} : V ≤ hf.functorObj U ↔ (Opens.map f).obj V ≤ U := by
  /-
    X Y : TopCat
    f : Quiver.Hom X Y
    hf : Topology.IsInducing ⇑f
    U : TopologicalSpace.Opens ↑X
    V : TopologicalSpace.Opens ↑Y
    ⊢ Iff (LE.le V (hf.functorObj U)) (LE.le ((TopologicalSpace.Opens.map f).obj V …
  -/
  obtain ⟨U, hU⟩ := U
  /-
    case mk
    X Y : TopCat
    f : Quiver.Hom X Y
    hf : Topology.IsInducing ⇑f
    V : TopologicalSpace.Opens ↑Y
    U : Set ↑X
    hU : IsOpen U
    ⊢ Iff (LE.le V (hf.functorObj { carrier := U, is_open' := hU })) (LE.le ((Topo …
  -/
  obtain ⟨t, ht, rfl⟩ := hf.isOpen_iff.mp hU
  /-
    case mk.intro.intro
    X Y : TopCat
    f : Quiver.Hom X Y
    hf : Topology.IsInducing ⇑f
    V : TopologicalSpace.Opens ↑Y
    t : Set ↑Y
    ht : IsOpen t
    hU : IsOpen (Set.preimage (⇑f) t)
    ⊢ Iff (LE.le V (hf.functorObj { carrier := Set.preimage (⇑f) t, is_open' := hU …
  -/
  constructor
    /-
      case mk.intro.intro.mp
      X Y : TopCat
      f : Quiver.Hom X Y
      hf : Topology.IsInducing ⇑f
      V : TopologicalSpace.Opens ↑Y
      t : Set ↑Y
      ht : IsOpen t
      hU : IsOpen (Set.preimage (⇑f) t)
      ⊢ LE.le V (hf.functorObj { carrier := Set.preimage (⇑f) t, is_open' := hU }) → …
    -/
  · exact fun i x hx ↦ (hf.mem_functorObj_iff ((Opens.map f).obj ⟨t, ht⟩)).mp (i hx)
    /-
      🎉 no goals
    -/
    /-
      case mk.intro.intro.mpr
      X Y : TopCat
      f : Quiver.Hom X Y
      hf : Topology.IsInducing ⇑f
      V : TopologicalSpace.Opens ↑Y
      t : Set ↑Y
      ht : IsOpen t
      hU : IsOpen (Set.preimage (⇑f) t)
      ⊢ LE.le ((TopologicalSpace.Opens.map f).obj V) { carrier := Set.preimage (⇑f)  …
    -/
  · intros h x hx
    /-
      case mk.intro.intro.mpr
      X Y : TopCat
      f : Quiver.Hom X Y
      hf : Topology.IsInducing ⇑f
      V : TopologicalSpace.Opens ↑Y
      t : Set ↑Y
      ht : IsOpen t
      hU : IsOpen (Set.preimage (⇑f) t)
      h : LE.le ((TopologicalSpace.Opens.map f).obj V) { carrier := Set.preimage (⇑f …
      x : ↑Y
      hx : Membership.mem (↑V) x
      ⊢ Membership.mem (↑(hf.functorObj { carrier := Set.preimage (⇑f) t, is_open' : …
    -/
    refine Opens.mem_sSup.mpr ⟨⟨_, V.2.union ht⟩, Opens.ext ?_, Set.mem_union_left t hx⟩
    /-
      case mk.intro.intro.mpr
      X Y : TopCat
      f : Quiver.Hom X Y
      hf : Topology.IsInducing ⇑f
      V : TopologicalSpace.Opens ↑Y
      t : Set ↑Y
      ht : IsOpen t
      hU : IsOpen (Set.preimage (⇑f) t)
      h : LE.le ((TopologicalSpace.Opens.map f).obj V) { carrier := Set.preimage (⇑f …
      x : ↑Y
      hx : Membership.mem (↑V) x
      ⊢ Eq ↑((TopologicalSpace.Opens.map f).obj { carrier := Union.union V.carrier t …
    -/
    dsimp
    /-
      case mk.intro.intro.mpr
      X Y : TopCat
      f : Quiver.Hom X Y
      hf : Topology.IsInducing ⇑f
      V : TopologicalSpace.Opens ↑Y
      t : Set ↑Y
      ht : IsOpen t
      hU : IsOpen (Set.preimage (⇑f) t)
      h : LE.le ((TopologicalSpace.Opens.map f).obj V) { carrier := Set.preimage (⇑f …
      x : ↑Y
      hx : Membership.mem (↑V) x
      ⊢ Eq (Union.union (Set.preimage ⇑f ↑V) (Set.preimage (⇑f) t)) (Set.preimage (⇑ …
    -/
    rwa [Set.union_eq_right]
    /-
      🎉 no goals
    -/


/-- An inducing map `f : X ⟶ Y` induces a Galois insertion between `Opens Y` and `Opens X`. -/
def opensGI {X Y : TopCat} {f : X ⟶ Y} (hf : IsInducing f) :
    GaloisInsertion (Opens.map f).obj hf.functorObj :=
  ⟨_, fun _ _ ↦ hf.le_functorObj_iff.symm, fun U ↦ (hf.map_functorObj U).ge, fun _ _ ↦ rfl⟩


/-- An inducing map `f : X ⟶ Y` induces a functor `Opens X ⥤ Opens Y`. -/
@[simps]
def functor {X Y : TopCat} {f : X ⟶ Y} (hf : IsInducing f) :
    Opens X ⥤ Opens Y where
  obj := hf.functorObj
  map {U V} h := homOfLE (hf.le_functorObj_iff.mpr ((hf.map_functorObj U).trans_le h.le))


/-- An inducing map `f : X ⟶ Y` induces an adjunction between `Opens Y` and `Opens X`. -/
def adjunction {X Y : TopCat} {f : X ⟶ Y} (hf : IsInducing f) :
    Opens.map f ⊣ hf.functor :=
  hf.opensGI.gc.adjunction


@[simp]
theorem isOpenEmbedding_obj_top {X : TopCat} (U : Opens X) :
    U.isOpenEmbedding.isOpenMap.functor.obj ⊤ = U := by
  /-
    X : TopCat
    U : TopologicalSpace.Opens ↑X
    ⊢ Eq (⋯.functor.obj Top.top) U
  -/
  ext1
  /-
    case h
    X : TopCat
    U : TopologicalSpace.Opens ↑X
    ⊢ Eq ↑(⋯.functor.obj Top.top) ↑U
  -/
  exact Set.image_univ.trans Subtype.range_coe
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-18")]
alias openEmbedding_obj_top := isOpenEmbedding_obj_top


@[simp]
theorem inclusion'_map_eq_top {X : TopCat} (U : Opens X) : (Opens.map U.inclusion').obj U = ⊤ := by
  /-
    X : TopCat
    U : TopologicalSpace.Opens ↑X
    ⊢ Eq ((TopologicalSpace.Opens.map U.inclusion').obj U) Top.top
  -/
  ext1
  /-
    case h
    X : TopCat
    U : TopologicalSpace.Opens ↑X
    ⊢ Eq ↑((TopologicalSpace.Opens.map U.inclusion').obj U) ↑Top.top
  -/
  exact Subtype.coe_preimage_self _
  /-
    🎉 no goals
  -/


@[simp]
theorem adjunction_counit_app_self {X : TopCat} (U : Opens X) :
                                                                      /-
                                                                        X : TopCat
                                                                        U : TopologicalSpace.Opens ↑X
                                                                        ⊢ Eq (((TopologicalSpace.Opens.map U.inclusion').comp ⋯.functor).obj U) ((Cate …
                                                                      -/
    U.isOpenEmbedding.isOpenMap.adjunction.counit.app U = eqToHom (by simp) := Subsingleton.elim _ _
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem inclusion'_top_functor (X : TopCat) :
    (@Opens.isOpenEmbedding X ⊤).isOpenMap.functor = map (inclusionTopIso X).inv := by
  /-
    X : TopCat
    ⊢ Eq ⋯.functor (TopologicalSpace.Opens.map (TopologicalSpace.Opens.inclusionTo …
  -/
  refine CategoryTheory.Functor.ext ?_ ?_
    /-
      case refine_1
      X : TopCat
      ⊢ ∀ (X_1 : TopologicalSpace.Opens ↑((TopologicalSpace.Opens.toTopCat X).obj To …
    -/
  · intro U
    /-
      case refine_1
      X : TopCat
      U : TopologicalSpace.Opens ↑((TopologicalSpace.Opens.toTopCat X).obj Top.top)
      ⊢ Eq (⋯.functor.obj U) ((TopologicalSpace.Opens.map (TopologicalSpace.Opens.in …
    -/
    ext x
    /-
      case refine_1.h.h
      X : TopCat
      U : TopologicalSpace.Opens ↑((TopologicalSpace.Opens.toTopCat X).obj Top.top)
      x : ↑X
      ⊢ Iff (Membership.mem (↑(⋯.functor.obj U)) x) (Membership.mem (↑((TopologicalS …
    -/
    exact ⟨fun ⟨⟨_, _⟩, h, rfl⟩ => h, fun h => ⟨⟨x, trivial⟩, h, rfl⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : TopCat
      ⊢ ∀ (X_1 Y : TopologicalSpace.Opens ↑((TopologicalSpace.Opens.toTopCat X).obj  …
    -/
  · subsingleton
    /-
      🎉 no goals
    -/


theorem functor_obj_map_obj {X Y : TopCat} {f : X ⟶ Y} (hf : IsOpenMap f) (U : Opens Y) :
    hf.functor.obj ((Opens.map f).obj U) = hf.functor.obj ⊤ ⊓ U := by
  /-
    X Y : TopCat
    f : Quiver.Hom X Y
    hf : IsOpenMap ⇑f
    U : TopologicalSpace.Opens ↑Y
    ⊢ Eq (hf.functor.obj ((TopologicalSpace.Opens.map f).obj U)) (Min.min (hf.func …
  -/
  ext
  /-
    case h.h
    X Y : TopCat
    f : Quiver.Hom X Y
    hf : IsOpenMap ⇑f
    U : TopologicalSpace.Opens ↑Y
    x✝ : ↑Y
    ⊢ Iff (Membership.mem (↑(hf.functor.obj ((TopologicalSpace.Opens.map f).obj U) …
  -/
  constructor
    /-
      case h.h.mp
      X Y : TopCat
      f : Quiver.Hom X Y
      hf : IsOpenMap ⇑f
      U : TopologicalSpace.Opens ↑Y
      x✝ : ↑Y
      ⊢ Membership.mem (↑(hf.functor.obj ((TopologicalSpace.Opens.map f).obj U))) x✝ …
    -/
  · rintro ⟨x, hx, rfl⟩
    /-
      case h.h.mp.intro.intro
      X Y : TopCat
      f : Quiver.Hom X Y
      hf : IsOpenMap ⇑f
      U : TopologicalSpace.Opens ↑Y
      x : ↑X
      hx : Membership.mem (↑((TopologicalSpace.Opens.map f).obj U)) x
      ⊢ Membership.mem (↑(Min.min (hf.functor.obj Top.top) U)) (f x)
    -/
    exact ⟨⟨x, trivial, rfl⟩, hx⟩
    /-
      🎉 no goals
    -/
    /-
      case h.h.mpr
      X Y : TopCat
      f : Quiver.Hom X Y
      hf : IsOpenMap ⇑f
      U : TopologicalSpace.Opens ↑Y
      x✝ : ↑Y
      ⊢ Membership.mem (↑(Min.min (hf.functor.obj Top.top) U)) x✝ → Membership.mem ( …
    -/
  · rintro ⟨⟨x, -, rfl⟩, hx⟩
    /-
      case h.h.mpr.intro.intro.intro
      X Y : TopCat
      f : Quiver.Hom X Y
      hf : IsOpenMap ⇑f
      U : TopologicalSpace.Opens ↑Y
      x : ↑X
      hx : Membership.mem (↑U) (f x)
      ⊢ Membership.mem (↑(hf.functor.obj ((TopologicalSpace.Opens.map f).obj U))) (f …
    -/
    exact ⟨x, hx, rfl⟩
    /-
      🎉 no goals
    -/

-- Porting note: added to ease the proof of `functor_map_eq_inf`

lemma set_range_inclusion' {X : TopCat} (U : Opens X) :
    Set.range (inclusion' U) = (U : Set X) := by
  /-
    X : TopCat
    U : TopologicalSpace.Opens ↑X
    ⊢ Eq (Set.range ⇑U.inclusion') ↑U
  -/
  ext x
  /-
    case h
    X : TopCat
    U : TopologicalSpace.Opens ↑X
    x : ↑X
    ⊢ Iff (Membership.mem (Set.range ⇑U.inclusion') x) (Membership.mem (↑U) x)
  -/
  constructor
    /-
      case h.mp
      X : TopCat
      U : TopologicalSpace.Opens ↑X
      x : ↑X
      ⊢ Membership.mem (Set.range ⇑U.inclusion') x → Membership.mem (↑U) x
    -/
  · rintro ⟨x, rfl⟩
    /-
      case h.mp.intro
      X : TopCat
      U : TopologicalSpace.Opens ↑X
      x : ↑((TopologicalSpace.Opens.toTopCat X).obj U)
      ⊢ Membership.mem (↑U) (U.inclusion' x)
    -/
    exact x.2
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      X : TopCat
      U : TopologicalSpace.Opens ↑X
      x : ↑X
      ⊢ Membership.mem (↑U) x → Membership.mem (Set.range ⇑U.inclusion') x
    -/
  · intro h
    /-
      case h.mpr
      X : TopCat
      U : TopologicalSpace.Opens ↑X
      x : ↑X
      h : Membership.mem (↑U) x
      ⊢ Membership.mem (Set.range ⇑U.inclusion') x
    -/
    exact ⟨⟨x, h⟩, rfl⟩
    /-
      🎉 no goals
    -/

@[deprecated (since := "2024-09-07")] alias set_range_forget_map_inclusion' := set_range_inclusion'


@[simp]
theorem functor_map_eq_inf {X : TopCat} (U V : Opens X) :
    U.isOpenEmbedding.isOpenMap.functor.obj ((Opens.map U.inclusion').obj V) = V ⊓ U := by
  /-
    X : TopCat
    U V : TopologicalSpace.Opens ↑X
    ⊢ Eq (⋯.functor.obj ((TopologicalSpace.Opens.map U.inclusion').obj V)) (Min.mi …
  -/
  ext1
  simp only [IsOpenMap.coe_functor_obj, map_coe, coe_inf,
    Set.image_preimage_eq_inter_range, set_range_inclusion' U]


theorem map_functor_eq' {X U : TopCat} (f : U ⟶ X) (hf : IsOpenEmbedding f) (V) :
    ((Opens.map f).obj <| hf.isOpenMap.functor.obj V) = V :=
  Opens.ext <| Set.preimage_image_eq _ hf.injective


@[simp]
theorem map_functor_eq {X : TopCat} {U : Opens X} (V : Opens U) :
    ((Opens.map U.inclusion').obj <| U.isOpenEmbedding.isOpenMap.functor.obj V) = V :=
  TopologicalSpace.Opens.map_functor_eq' _ U.isOpenEmbedding V


@[simp]
theorem adjunction_counit_map_functor {X : TopCat} {U : Opens X} (V : Opens U) :
    U.isOpenEmbedding.isOpenMap.adjunction.counit.app (U.isOpenEmbedding.isOpenMap.functor.obj V) =
                  /-
                    X : TopCat
                    U : TopologicalSpace.Opens ↑X
                    V : TopologicalSpace.Opens (Subtype fun x => Membership.mem U x)
                    ⊢ Eq (((TopologicalSpace.Opens.map U.inclusion').comp ⋯.functor).obj (⋯.functo …
                  -/
      eqToHom (by dsimp; rw [map_functor_eq V]) := by
                         /-
                           🎉 no goals
                         -/
  /-
    X : TopCat
    U : TopologicalSpace.Opens ↑X
    V : TopologicalSpace.Opens (Subtype fun x => Membership.mem U x)
    ⊢ Eq (⋯.adjunction.counit.app (⋯.functor.obj V)) (CategoryTheory.eqToHom ⋯)
  -/
  subsingleton
  /-
    🎉 no goals
  -/



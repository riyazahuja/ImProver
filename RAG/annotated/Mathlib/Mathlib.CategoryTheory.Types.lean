@[to_additive existing CategoryTheory.types]
instance types : LargeCategory (Type u) where
  Hom a b := a → b
  id _ := id
  comp f g := g ∘ f


theorem types_hom {α β : Type u} : (α ⟶ β) = (α → β) :=
  rfl

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10688): this lemma was not here in Lean 3. Lean 3 `ext` would solve this goal
-- because of its "if all else fails, apply all `ext` lemmas" policy,
-- which apparently we want to move away from.

@[ext] theorem types_ext {α β : Type u} (f g : α ⟶ β) (h : ∀ a : α, f a = g a) : f = g := by
  /-
    α β : Type u
    f g : Quiver.Hom α β
    h : ∀ (a : α), Eq (f a) (g a)
    ⊢ Eq f g
  -/
  funext x
  /-
    case h
    α β : Type u
    f g : Quiver.Hom α β
    h : ∀ (a : α), Eq (f a) (g a)
    x : α
    ⊢ Eq (f x) (g x)
  -/
  exact h x
  /-
    🎉 no goals
  -/


theorem types_id (X : Type u) : 𝟙 X = id :=
  rfl


theorem types_comp {X Y Z : Type u} (f : X ⟶ Y) (g : Y ⟶ Z) : f ≫ g = g ∘ f :=
  rfl


@[simp]
theorem types_id_apply (X : Type u) (x : X) : (𝟙 X : X → X) x = x :=
  rfl


@[simp]
theorem types_comp_apply {X Y Z : Type u} (f : X ⟶ Y) (g : Y ⟶ Z) (x : X) : (f ≫ g) x = g (f x) :=
  rfl


@[simp]
theorem hom_inv_id_apply {X Y : Type u} (f : X ≅ Y) (x : X) : f.inv (f.hom x) = x :=
  congr_fun f.hom_inv_id x


@[simp]
theorem inv_hom_id_apply {X Y : Type u} (f : X ≅ Y) (y : Y) : f.hom (f.inv y) = y :=
  congr_fun f.inv_hom_id y

-- Unfortunately without this wrapper we can't use `CategoryTheory` idioms, such as `IsIso f`.

/-- `asHom f` helps Lean type check a function as a morphism in the category `Type`. -/
abbrev asHom {α β : Type u} (f : α → β) : α ⟶ β :=
  f


@[inherit_doc]
scoped notation "↾" f:200 => CategoryTheory.asHom f


/-- The sections of a functor `F : J ⥤ Type` are
the choices of a point `u j : F.obj j` for each `j`,
such that `F.map f (u j) = u j'` for every morphism `f : j ⟶ j'`.

We later use these to define limits in `Type` and in many concrete categories.
-/
def sections (F : J ⥤ Type w) : Set (∀ j, F.obj j) :=
  { u | ∀ {j j'} (f : j ⟶ j'), F.map f (u j) = u j' }


@[simp]
lemma sections_property {F : J ⥤ Type w} (s : (F.sections : Type _))
    {j j' : J} (f : j ⟶ j') : F.map f (s.val j) = s.val j' :=
  s.property f


lemma sections_ext_iff {F : J ⥤ Type w} {x y : F.sections} : x = y ↔ ∀ j, x.val j = y.val j :=
  Subtype.ext_iff.trans funext_iff


/-- The functor which sends a functor to types to its sections. -/
@[simps]
def sectionsFunctor : (J ⥤ Type w) ⥤ Type max u w where
  obj F := F.sections
  map {F G} φ x := ⟨fun j => φ.app j (x.1 j), fun {j j'} f =>
                                                        /-
                                                          J : Type u
                                                          inst✝ : CategoryTheory.Category.{v, u} J
                                                          F G : CategoryTheory.Functor J (Type w)
                                                          φ : Quiver.Hom F G
                                                          x : (fun F => ↑F.sections) F
                                                          j j' : J
                                                          f : Quiver.Hom j j'
                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) (φ.app j') (↑x j)) ((fun j  …
                                                        -/
    (congr_fun (φ.naturality f) (x.1 j)).symm.trans (by simp [x.2 f])⟩
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem map_comp_apply (f : X ⟶ Y) (g : Y ⟶ Z) (a : F.obj X) :
                                                      /-
                                                        C : Type u
                                                        inst✝ : CategoryTheory.Category.{v, u} C
                                                        F : CategoryTheory.Functor C (Type w)
                                                        X Y Z : C
                                                        f : Quiver.Hom X Y
                                                        g : Quiver.Hom Y Z
                                                        a : F.obj X
                                                        ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp f g) a) (F.map g (F.map f a))
                                                      -/
    (F.map (f ≫ g)) a = (F.map g) ((F.map f) a) := by simp [types_comp]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
                                                               /-
                                                                 C : Type u
                                                                 inst✝ : CategoryTheory.Category.{v, u} C
                                                                 F : CategoryTheory.Functor C (Type w)
                                                                 X : C
                                                                 a : F.obj X
                                                                 ⊢ Eq (F.map (CategoryTheory.CategoryStruct.id X) a) a
                                                               -/
theorem map_id_apply (a : F.obj X) : (F.map (𝟙 X)) a = a := by simp [types_id]
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem naturality (f : X ⟶ Y) (x : F.obj X) : σ.app Y ((F.map f) x) = (G.map f) (σ.app X x) :=
  congr_fun (σ.naturality f) x


@[simp]
theorem comp (x : F.obj X) : (σ ≫ τ).app X x = τ.app X (σ.app X x) :=
  rfl


@[simp]
theorem eqToHom_map_comp_apply (p : X = Y) (q : Y = Z) (x : F.obj X) :
    F.map (eqToHom q) (F.map (eqToHom p) x) = F.map (eqToHom <| p.trans q) x := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C (Type w)
    X Y Z : C
    p : Eq X Y
    q : Eq Y Z
    x : F.obj X
    ⊢ Eq (F.map (CategoryTheory.eqToHom q) (F.map (CategoryTheory.eqToHom p) x)) ( …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


@[simp]
theorem hcomp (x : (I ⋙ F).obj W) : (ρ ◫ σ).app W x = (G.map (ρ.app W)) (σ.app (I.obj W) x) :=
  rfl


@[simp]
theorem map_inv_map_hom_apply (f : X ≅ Y) (x : F.obj X) : F.map f.inv (F.map f.hom x) = x :=
  congr_fun (F.mapIso f).hom_inv_id x


@[simp]
theorem map_hom_map_inv_apply (f : X ≅ Y) (y : F.obj Y) : F.map f.hom (F.map f.inv y) = y :=
  congr_fun (F.mapIso f).inv_hom_id y


@[simp]
theorem hom_inv_id_app_apply (α : F ≅ G) (X) (x) : α.inv.app X (α.hom.app X x) = x :=
  congr_fun (α.hom_inv_id_app X) x


@[simp]
theorem inv_hom_id_app_apply (α : F ≅ G) (X) (x) : α.hom.app X (α.inv.app X x) = x :=
  congr_fun (α.inv_hom_id_app X) x


/-- The isomorphism between a `Type` which has been `ULift`ed to the same universe,
and the original type.
-/
def uliftTrivial (V : Type u) : ULift.{u} V ≅ V where
  hom a := a.1
  inv a := .up a


/-- The functor embedding `Type u` into `Type (max u v)`.
Write this as `uliftFunctor.{5, 2}` to get `Type 2 ⥤ Type 5`.
-/
@[pp_with_univ]
def uliftFunctor : Type u ⥤ Type max u v where
  obj X := ULift.{v} X
  map {X} {_} f := fun x : ULift.{v} X => ULift.up (f x.down)


@[simp]
theorem uliftFunctor_obj {X : Type u} : uliftFunctor.obj.{v} X = ULift.{v} X :=
  rfl


@[simp]
theorem uliftFunctor_map {X Y : Type u} (f : X ⟶ Y) (x : ULift.{v} X) :
    uliftFunctor.map f x = ULift.up (f x.down) :=
  rfl


instance uliftFunctor_full : Functor.Full.{u} uliftFunctor where
  map_surjective f := ⟨fun x => (f (ULift.up x)).down, rfl⟩


instance uliftFunctor_faithful : uliftFunctor.Faithful where
  map_injective {_X} {_Y} f g p :=
    funext fun x =>
      congr_arg ULift.down (congr_fun p (ULift.up x) : ULift.up (f x) = ULift.up (g x))


/-- The functor embedding `Type u` into `Type u` via `ULift` is isomorphic to the identity functor.
 -/
def uliftFunctorTrivial : uliftFunctor.{u, u} ≅ 𝟭 _ :=
  /-
    ⊢ ∀ {X Y : Type u} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.com …
  -/
  NatIso.ofComponents uliftTrivial
  /-
    🎉 no goals
  -/

-- TODO We should connect this to a general story about concrete categories
-- whose forgetful functor is representable.

/-- Any term `x` of a type `X` corresponds to a morphism `PUnit ⟶ X`. -/
def homOfElement {X : Type u} (x : X) : PUnit ⟶ X := fun _ => x


theorem homOfElement_eq_iff {X : Type u} (x y : X) : homOfElement x = homOfElement y ↔ x = y :=
                                       /-
                                         X : Type u
                                         x y : X
                                         ⊢ Eq x y → Eq (CategoryTheory.homOfElement x) (CategoryTheory.homOfElement y)
                                       -/
  ⟨fun H => congr_fun H PUnit.unit, by aesop⟩
                                       /-
                                         🎉 no goals
                                       -/


/-- A morphism in `Type` is a monomorphism if and only if it is injective.

See <https://stacks.math.columbia.edu/tag/003C>.
-/
theorem mono_iff_injective {X Y : Type u} (f : X ⟶ Y) : Mono f ↔ Function.Injective f := by
  /-
    X Y : Type u
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Mono f) (Function.Injective f)
  -/
  constructor
    /-
      case mp
      X Y : Type u
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.Mono f → Function.Injective f
    -/
  · intro H x x' h
    /-
      case mp
      X Y : Type u
      f : Quiver.Hom X Y
      H : CategoryTheory.Mono f
      x x' : X
      h : Eq (f x) (f x')
      ⊢ Eq x x'
    -/
    rw [← homOfElement_eq_iff] at h ⊢
    /-
      case mp
      X Y : Type u
      f : Quiver.Hom X Y
      H : CategoryTheory.Mono f
      x x' : X
      h : Eq (CategoryTheory.homOfElement (f x)) (CategoryTheory.homOfElement (f x'))
      ⊢ Eq (CategoryTheory.homOfElement x) (CategoryTheory.homOfElement x')
    -/
    exact (cancel_mono f).mp h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X Y : Type u
      f : Quiver.Hom X Y
      ⊢ Function.Injective f → CategoryTheory.Mono f
    -/
  · exact fun H => ⟨fun g g' h => H.comp_left h⟩
    /-
      🎉 no goals
    -/


theorem injective_of_mono {X Y : Type u} (f : X ⟶ Y) [hf : Mono f] : Function.Injective f :=
  (mono_iff_injective f).1 hf


/-- A morphism in `Type` is an epimorphism if and only if it is surjective.

See <https://stacks.math.columbia.edu/tag/003C>.
-/
theorem epi_iff_surjective {X Y : Type u} (f : X ⟶ Y) : Epi f ↔ Function.Surjective f := by
  /-
    X Y : Type u
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Epi f) (Function.Surjective f)
  -/
  constructor
    /-
      case mp
      X Y : Type u
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.Epi f → Function.Surjective f
    -/
  · rintro ⟨H⟩
    /-
      case mp.mk
      X Y : Type u
      f : Quiver.Hom X Y
      H : ∀ {Z : Type u} (g h : Quiver.Hom Y Z), Eq (CategoryTheory.CategoryStruct.c …
      ⊢ Function.Surjective f
    -/
    refine Function.surjective_of_right_cancellable_Prop fun g₁ g₂ hg => ?_
    /-
      case mp.mk
      X Y : Type u
      f : Quiver.Hom X Y
      H : ∀ {Z : Type u} (g h : Quiver.Hom Y Z), Eq (CategoryTheory.CategoryStruct.c …
      g₁ g₂ : Y → Prop
      hg : Eq (Function.comp g₁ f) (Function.comp g₂ f)
      ⊢ Eq g₁ g₂
    -/
    rw [← Equiv.ulift.symm.injective.comp_left.eq_iff]
    /-
      case mp.mk
      X Y : Type u
      f : Quiver.Hom X Y
      H : ∀ {Z : Type u} (g h : Quiver.Hom Y Z), Eq (CategoryTheory.CategoryStruct.c …
      g₁ g₂ : Y → Prop
      hg : Eq (Function.comp g₁ f) (Function.comp g₂ f)
      ⊢ Eq (Function.comp (⇑Equiv.ulift.symm) g₁) (Function.comp (⇑Equiv.ulift.symm) …
    -/
    apply H
    /-
      case mp.mk.a
      X Y : Type u
      f : Quiver.Hom X Y
      H : ∀ {Z : Type u} (g h : Quiver.Hom Y Z), Eq (CategoryTheory.CategoryStruct.c …
      g₁ g₂ : Y → Prop
      hg : Eq (Function.comp g₁ f) (Function.comp g₂ f)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (Function.comp (⇑Equiv.ulift.symm)  …
    -/
    change ULift.up ∘ g₁ ∘ f = ULift.up ∘ g₂ ∘ f
    /-
      case mp.mk.a
      X Y : Type u
      f : Quiver.Hom X Y
      H : ∀ {Z : Type u} (g h : Quiver.Hom Y Z), Eq (CategoryTheory.CategoryStruct.c …
      g₁ g₂ : Y → Prop
      hg : Eq (Function.comp g₁ f) (Function.comp g₂ f)
      ⊢ Eq (Function.comp ULift.up (Function.comp g₁ f)) (Function.comp ULift.up (Fu …
    -/
    rw [hg]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X Y : Type u
      f : Quiver.Hom X Y
      ⊢ Function.Surjective f → CategoryTheory.Epi f
    -/
  · exact fun H => ⟨fun g g' h => H.injective_comp_right h⟩
    /-
      🎉 no goals
    -/


theorem surjective_of_epi {X Y : Type u} (f : X ⟶ Y) [hf : Epi f] : Function.Surjective f :=
  (epi_iff_surjective f).1 hf


/-- `ofTypeFunctor m` converts from Lean's `Type`-based `Category` to `CategoryTheory`. This
allows us to use these functors in category theory. -/
def ofTypeFunctor (m : Type u → Type v) [_root_.Functor m] [LawfulFunctor m] : Type u ⥤ Type v where
  obj := m
  map f := Functor.map f
                        /-
                          m : Type u → Type v
                          inst✝¹ : _root_.Functor m
                          inst✝ : LawfulFunctor m
                          α : Type u
                          ⊢ Eq ({ obj := m, map := fun {X Y} f => Functor.map f }.map (CategoryTheory.Ca …
                        -/
  map_id := fun α => by funext X; apply id_map  /- Porting note: original proof is via
                                  /-
                                    🎉 no goals
                                  -/
  `fun α => _root_.Functor.map_id` but I cannot get Lean to find this. Reproduced its
  original proof -/
  map_comp f g := funext fun _ => LawfulFunctor.comp_map f g _


@[simp]
theorem ofTypeFunctor_obj : (ofTypeFunctor m).obj = m :=
  rfl


@[simp]
theorem ofTypeFunctor_map {α β} (f : α → β) :
    (ofTypeFunctor m).map f = (Functor.map f : m α → m β) :=
  rfl


/-- Any equivalence between types in the same universe gives
a categorical isomorphism between those types.
-/
def toIso (e : X ≃ Y) : X ≅ Y where
  hom := e.toFun
  inv := e.invFun
  hom_inv_id := funext e.left_inv
  inv_hom_id := funext e.right_inv


@[simp]
theorem toIso_hom {e : X ≃ Y} : e.toIso.hom = e :=
  rfl


@[simp]
theorem toIso_inv {e : X ≃ Y} : e.toIso.inv = e.symm :=
  rfl


/-- Any isomorphism between types gives an equivalence. -/
def toEquiv (i : X ≅ Y) : X ≃ Y where
  toFun := i.hom
  invFun := i.inv
  left_inv x := congr_fun i.hom_inv_id x
  right_inv y := congr_fun i.inv_hom_id y


@[simp]
theorem toEquiv_fun (i : X ≅ Y) : (i.toEquiv : X → Y) = i.hom :=
  rfl


@[simp]
theorem toEquiv_symm_fun (i : X ≅ Y) : (i.toEquiv.symm : Y → X) = i.inv :=
  rfl


@[simp]
theorem toEquiv_id (X : Type u) : (Iso.refl X).toEquiv = Equiv.refl X :=
  rfl


@[simp]
theorem toEquiv_comp {X Y Z : Type u} (f : X ≅ Y) (g : Y ≅ Z) :
    (f ≪≫ g).toEquiv = f.toEquiv.trans g.toEquiv :=
  rfl


/-- A morphism in `Type u` is an isomorphism if and only if it is bijective. -/
theorem isIso_iff_bijective {X Y : Type u} (f : X ⟶ Y) : IsIso f ↔ Function.Bijective f :=
  Iff.intro (fun _ => (asIso f : X ≅ Y).toEquiv.bijective) fun b =>
    (Equiv.ofBijective f b).toIso.isIso_hom


instance : SplitEpiCategory (Type u) where
  isSplitEpi_of_epi f hf :=
    IsSplitEpi.mk' <|
      { section_ := Function.surjInv <| (epi_iff_surjective f).1 hf
        id := funext <| Function.rightInverse_surjInv <| (epi_iff_surjective f).1 hf }


/-- Equivalences (between types in the same universe) are the same as (isomorphic to) isomorphisms
of types. -/
@[simps]
def equivIsoIso {X Y : Type u} : X ≃ Y ≅ X ≅ Y where
  hom e := e.toIso
  inv i := i.toEquiv


/-- Equivalences (between types in the same universe) are the same as (equivalent to) isomorphisms
of types. -/
def equivEquivIso {X Y : Type u} : X ≃ Y ≃ (X ≅ Y) :=
  equivIsoIso.toEquiv


@[simp]
theorem equivEquivIso_hom {X Y : Type u} (e : X ≃ Y) : equivEquivIso e = e.toIso :=
  rfl


@[simp]
theorem equivEquivIso_inv {X Y : Type u} (e : X ≅ Y) : equivEquivIso.symm e = e.toEquiv :=
  rfl


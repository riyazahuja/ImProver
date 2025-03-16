/-- The over category has as objects arrows in `T` with codomain `X` and as morphisms commutative
triangles.

See <https://stacks.math.columbia.edu/tag/001G>.
-/
def Over (X : T) :=
  CostructuredArrow (𝟭 T) X


instance (X : T) : Category (Over X) := commaCategory

-- Satisfying the inhabited linter

instance Over.inhabited [Inhabited T] : Inhabited (Over (default : T)) where
  default :=
    { left := default
      right := default
      hom := 𝟙 _ }


@[ext]
theorem OverMorphism.ext {X : T} {U V : Over X} {f g : U ⟶ V} (h : f.left = g.left) : f = g := by
  /-
    T : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} T
    X : T
    U V : CategoryTheory.Over X
    f g : Quiver.Hom U V
    h : Eq f.left g.left
    ⊢ Eq f g
  -/
  let ⟨_,b,_⟩ := f
  /-
    T : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} T
    X : T
    U V : CategoryTheory.Over X
    f g : Quiver.Hom U V
    left✝ : Quiver.Hom U.left V.left
    b : Quiver.Hom U.right V.right
    w✝ : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id T).map …
    h : Eq { left := left✝, right := b, w := w✝ }.left g.left
    ⊢ Eq { left := left✝, right := b, w := w✝ } g
  -/
  let ⟨_,e,_⟩ := g
  /-
    T : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} T
    X : T
    U V : CategoryTheory.Over X
    f g : Quiver.Hom U V
    left✝¹ : Quiver.Hom U.left V.left
    b : Quiver.Hom U.right V.right
    w✝¹ : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id T).ma …
    left✝ : Quiver.Hom U.left V.left
    e : Quiver.Hom U.right V.right
    w✝ : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id T).map …
    h : Eq { left := left✝¹, right := b, w := w✝¹ }.left { left := left✝, right := …
    ⊢ Eq { left := left✝¹, right := b, w := w✝¹ } { left := left✝, right := e, w : …
  -/
  congr
  /-
    case e_right
    T : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} T
    X : T
    U V : CategoryTheory.Over X
    f g : Quiver.Hom U V
    left✝¹ : Quiver.Hom U.left V.left
    b : Quiver.Hom U.right V.right
    w✝¹ : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id T).ma …
    left✝ : Quiver.Hom U.left V.left
    e : Quiver.Hom U.right V.right
    w✝ : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id T).map …
    h : Eq { left := left✝¹, right := b, w := w✝¹ }.left { left := left✝, right := …
    ⊢ Eq b e
  -/
  simp only [eq_iff_true_of_subsingleton]
  /-
    🎉 no goals
  -/


@[simp]
                                                       /-
                                                         T : Type u₁
                                                         inst✝ : CategoryTheory.Category.{v₁, u₁} T
                                                         X : T
                                                         U : CategoryTheory.Over X
                                                         ⊢ Eq U.right { as := PUnit.unit }
                                                       -/
theorem over_right (U : Over X) : U.right = ⟨⟨⟩⟩ := by simp only
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
theorem id_left (U : Over X) : CommaMorphism.left (𝟙 U) = 𝟙 U.left :=
  rfl


@[simp, reassoc]
theorem comp_left (a b c : Over X) (f : a ⟶ b) (g : b ⟶ c) : (f ≫ g).left = f.left ≫ g.left :=
  rfl


@[reassoc (attr := simp)]
                                                                    /-
                                                                      T : Type u₁
                                                                      inst✝ : CategoryTheory.Category.{v₁, u₁} T
                                                                      X : T
                                                                      A B : CategoryTheory.Over X
                                                                      f : Quiver.Hom A B
                                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp f.left B.hom) A.hom
                                                                    -/
theorem w {A B : Over X} (f : A ⟶ B) : f.left ≫ B.hom = A.hom := by have := f.w; aesop_cat
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


/-- To give an object in the over category, it suffices to give a morphism with codomain `X`. -/
@[simps! left hom]
def mk {X Y : T} (f : Y ⟶ X) : Over X :=
  CostructuredArrow.mk f


/-- We can set up a coercion from arrows with codomain `X` to `over X`. This most likely should not
    be a global instance, but it is sometimes useful. -/
def coeFromHom {X Y : T} : CoeOut (Y ⟶ X) (Over X) where coe := mk


@[simp]
theorem coe_hom {X Y : T} (f : Y ⟶ X) : (f : Over X).hom = f :=
  rfl


/-- To give a morphism in the over category, it suffices to give an arrow fitting in a commutative
    triangle. -/
@[simps!]
def homMk {U V : Over X} (f : U.left ⟶ V.left) (w : f ≫ V.hom = U.hom := by aesop_cat) : U ⟶ V :=
  CostructuredArrow.homMk f w

-- Porting note: simp solves this; simpNF still sees them after `-simp` (?)

/-- Construct an isomorphism in the over category given isomorphisms of the objects whose forward
direction gives a commutative triangle.
-/
@[simps!]
def isoMk {f g : Over X} (hl : f.left ≅ g.left) (hw : hl.hom ≫ g.hom = f.hom := by aesop_cat) :
    f ≅ g :=
  CostructuredArrow.isoMk hl hw

-- Porting note: simp solves this; simpNF still sees them after `-simp` (?)

@[reassoc (attr := simp)]
lemma hom_left_inv_left {f g : Over X} (e : f ≅ g) :
    e.hom.left ≫ e.inv.left = 𝟙 f.left := by
  /-
    T : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} T
    X : T
    f g : CategoryTheory.Over X
    e : CategoryTheory.Iso f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp e.hom.left e.inv.left) (CategoryTheor …
  -/
  simp [← Over.comp_left]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma inv_left_hom_left {f g : Over X} (e : f ≅ g) :
    e.inv.left ≫ e.hom.left = 𝟙 g.left := by
  /-
    T : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} T
    X : T
    f g : CategoryTheory.Over X
    e : CategoryTheory.Iso f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp e.inv.left e.hom.left) (CategoryTheor …
  -/
  simp [← Over.comp_left]
  /-
    🎉 no goals
  -/


/-- The forgetful functor mapping an arrow to its domain.

See <https://stacks.math.columbia.edu/tag/001G>.
-/
def forget : Over X ⥤ T :=
  Comma.fst _ _


@[simp]
theorem forget_obj {U : Over X} : (forget X).obj U = U.left :=
  rfl


@[simp]
theorem forget_map {U V : Over X} {f : U ⟶ V} : (forget X).map f = f.left :=
  rfl


/-- The natural cocone over the forgetful functor `Over X ⥤ T` with cocone point `X`. -/
@[simps]
def forgetCocone (X : T) : Limits.Cocone (forget X) :=
  { pt := X
    ι := { app := Comma.hom } }


/-- A morphism `f : X ⟶ Y` induces a functor `Over X ⥤ Over Y` in the obvious way.

See <https://stacks.math.columbia.edu/tag/001G>.
-/
def map {Y : T} (f : X ⟶ Y) : Over X ⥤ Over Y :=
  Comma.mapRight _ <| Discrete.natTrans fun _ => f


@[simp]
theorem map_obj_left : ((map f).obj U).left = U.left :=
  rfl


@[simp]
theorem map_obj_hom : ((map f).obj U).hom = U.hom ≫ f :=
  rfl


@[simp]
theorem map_map_left : ((map f).map g).left = g.left :=
  rfl

/-- If `f` is an isomorphism, `map f` is an equivalence of categories. -/
def mapIso {Y : T} (f : X ≅ Y) : Over X ≌ Over Y :=
  Comma.mapRightIso _ <| Discrete.natIso fun _ ↦ f


@[simp] lemma mapIso_functor {Y : T} (f : X ≅ Y) : (mapIso f).functor = map f.hom := rfl

@[simp] lemma mapIso_inverse {Y : T} (f : X ≅ Y) : (mapIso f).inverse = map f.inv := rfl


/-- Mapping by the identity morphism is just the identity functor. -/
theorem mapId_eq (Y : T) : map (𝟙 Y) = 𝟭 _ := by
  /-
    T : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} T
    Y : T
    ⊢ Eq (CategoryTheory.Over.map (CategoryTheory.CategoryStruct.id Y)) (CategoryT …
  -/
  fapply Functor.ext
    /-
      case h_obj
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      Y : T
      ⊢ ∀ (X : CategoryTheory.Over Y), Eq ((CategoryTheory.Over.map (CategoryTheory. …
    -/
  · intro x
    /-
      case h_obj
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      Y : T
      x : CategoryTheory.Over Y
      ⊢ Eq ((CategoryTheory.Over.map (CategoryTheory.CategoryStruct.id Y)).obj x) (( …
    -/
    dsimp [Over, Over.map, Comma.mapRight]
    /-
      case h_obj
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      Y : T
      x : CategoryTheory.Over Y
      ⊢ Eq { left := x.left, right := x.right, hom := CategoryTheory.CategoryStruct. …
    -/
    simp only [Category.comp_id]
    /-
      case h_obj
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      Y : T
      x : CategoryTheory.Over Y
      ⊢ Eq { left := x.left, right := x.right, hom := x.hom } x
    -/
    exact rfl
    /-
      🎉 no goals
    -/
    /-
      case h_map
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      Y : T
      ⊢ autoParam (∀ (X Y_1 : CategoryTheory.Over Y) (f : Quiver.Hom X Y_1), Eq ((Ca …
    -/
  · intros x y u
    /-
      case h_map
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      Y : T
      x y : CategoryTheory.Over Y
      u : Quiver.Hom x y
      ⊢ Eq ((CategoryTheory.Over.map (CategoryTheory.CategoryStruct.id Y)).map u) (C …
    -/
    dsimp [Over, Over.map, Comma.mapRight]
    /-
      case h_map
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      Y : T
      x y : CategoryTheory.Over Y
      u : Quiver.Hom x y
      ⊢ Eq { left := u.left, right := CategoryTheory.CategoryStruct.id x.right, w := …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The natural isomorphism arising from `mapForget_eq`. -/
@[simps!]
def mapId (Y : T) : map (𝟙 Y) ≅ 𝟭 _ := eqToIso (mapId_eq Y)
--  NatIso.ofComponents fun X => isoMk (Iso.refl _)


/-- Mapping by `f` and then forgetting is the same as forgetting. -/
theorem mapForget_eq {X Y : T} (f : X ⟶ Y) :
    (map f) ⋙ (forget Y) = (forget X) := by
  /-
    T : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} T
    X Y : T
    f : Quiver.Hom X Y
    ⊢ Eq ((CategoryTheory.Over.map f).comp (CategoryTheory.Over.forget Y)) (Catego …
  -/
  fapply Functor.ext
    /-
      case h_obj
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      X Y : T
      f : Quiver.Hom X Y
      ⊢ ∀ (X_1 : CategoryTheory.Over X), Eq (((CategoryTheory.Over.map f).comp (Cate …
    -/
  · dsimp [Over, Over.map]; intro x; exact rfl
                                     /-
                                       🎉 no goals
                                     -/
    /-
      case h_map
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      X Y : T
      f : Quiver.Hom X Y
      ⊢ autoParam (∀ (X_1 Y_1 : CategoryTheory.Over X) (f_1 : Quiver.Hom X_1 Y_1), E …
    -/
  · intros x y u; simp
                  /-
                    🎉 no goals
                  -/


/-- The natural isomorphism arising from `mapForget_eq`. -/
def mapForget {X Y : T} (f : X ⟶ Y) :
    (map f) ⋙ (forget Y) ≅ (forget X) := eqToIso (mapForget_eq f)


@[simp]
theorem eqToHom_left {X : T} {U V : Over X} (e : U = V) :
    (eqToHom e).left = eqToHom (e ▸ rfl : U.left = V.left) := by
  /-
    T : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} T
    X : T
    U V : CategoryTheory.Over X
    e : Eq U V
    ⊢ Eq (CategoryTheory.eqToHom e).left (CategoryTheory.eqToHom ⋯)
  -/
  subst e; rfl
           /-
             🎉 no goals
           -/


/-- Mapping by the composite morphism `f ≫ g` is the same as mapping by `f` then by `g`. -/
theorem mapComp_eq {X Y Z : T} (f : X ⟶ Y) (g : Y ⟶ Z) :
    map (f ≫ g) = (map f) ⋙ (map g) := by
  /-
    T : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} T
    X Y Z : T
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.Over.map (CategoryTheory.CategoryStruct.comp f g)) ((Cate …
  -/
  fapply Functor.ext
    /-
      case h_obj
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      X Y Z : T
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ ∀ (X_1 : CategoryTheory.Over X), Eq ((CategoryTheory.Over.map (CategoryTheor …
    -/
  · simp [Over.map, Comma.mapRight]
    /-
      🎉 no goals
    -/
    /-
      case h_map
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      X Y Z : T
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ autoParam (∀ (X_1 Y_1 : CategoryTheory.Over X) (f_1 : Quiver.Hom X_1 Y_1), E …
    -/
  · intro U V k
    /-
      case h_map
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      X Y Z : T
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      U V : CategoryTheory.Over X
      k : Quiver.Hom U V
      ⊢ Eq ((CategoryTheory.Over.map (CategoryTheory.CategoryStruct.comp f g)).map k …
    -/
    ext
    /-
      case h_map.h
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      X Y Z : T
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      U V : CategoryTheory.Over X
      k : Quiver.Hom U V
      ⊢ Eq ((CategoryTheory.Over.map (CategoryTheory.CategoryStruct.comp f g)).map k …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The natural isomorphism arising from `mapComp_eq`. -/
@[simps!]
def mapComp {X Y Z : T} (f : X ⟶ Y) (g : Y ⟶ Z) :
    map (f ≫ g) ≅ (map f) ⋙ (map g) := eqToIso (mapComp_eq f g)


/-- If `f = g`, then `map f` is naturally isomorphic to `map g`. -/
@[simps!]
def mapCongr {X Y : T} (f g : X ⟶ Y) (h : f = g) :
    map f ≅ map g :=
                                           /-
                                             T : Type u₁
                                             inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                             D : Type u₂
                                             inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                             X✝ X Y : T
                                             f g : Quiver.Hom X Y
                                             h : Eq f g
                                             A : CategoryTheory.Over X
                                             ⊢ Eq ((CategoryTheory.Over.map f).obj A) ((CategoryTheory.Over.map g).obj A)
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
  NatIso.ofComponents (fun A ↦ eqToIso (by rw [h]))
  /-
    🎉 no goals
  -/


variable (T) in
/-- The functor defined by the over categories.-/
@[simps] def mapFunctor : T ⥤ Cat where
  obj X := Cat.of (Over X)
  map := map
  map_id := mapId_eq
  map_comp := mapComp_eq


instance forget_reflects_iso : (forget X).ReflectsIsomorphisms where
  reflects {Y Z} f t := by
    let g : Z ⟶ Y := Over.homMk (inv ((forget X).map f))
      ((asIso ((forget X).map f)).inv_comp_eq.2 (Over.w f).symm)
    /-
      T : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      X : T
      Y Z : CategoryTheory.Over X
      f : Quiver.Hom Y Z
      t : CategoryTheory.IsIso ((CategoryTheory.Over.forget X).map f)
      g : Quiver.Hom Z Y := CategoryTheory.Over.homMk (CategoryTheory.inv ((Category …
      ⊢ CategoryTheory.IsIso f
    -/
    dsimp [forget] at t
    /-
      T : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      X : T
      Y Z : CategoryTheory.Over X
      f : Quiver.Hom Y Z
      t : CategoryTheory.IsIso f.left
      g : Quiver.Hom Z Y := CategoryTheory.Over.homMk (CategoryTheory.inv ((Category …
      ⊢ CategoryTheory.IsIso f
    -/
    refine ⟨⟨g, ⟨?_,?_⟩⟩⟩
    /-
      case refine_1
      T : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      X : T
      Y Z : CategoryTheory.Over X
      f : Quiver.Hom Y Z
      t : CategoryTheory.IsIso f.left
      g : Quiver.Hom Z Y := CategoryTheory.Over.homMk (CategoryTheory.inv ((Category …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct.i …
    -/
    repeat (ext; simp [g])
    /-
      🎉 no goals
    -/


/-- The identity over `X` is terminal. -/
noncomputable def mkIdTerminal : Limits.IsTerminal (mk (𝟙 X)) :=
  CostructuredArrow.mkIdTerminal


instance forget_faithful : (forget X).Faithful where

-- TODO: Show the converse holds if `T` has binary products.

/--
If `k.left` is an epimorphism, then `k` is an epimorphism. In other words, `Over.forget X` reflects
epimorphisms.
The converse does not hold without additional assumptions on the underlying category, see
`CategoryTheory.Over.epi_left_of_epi`.
-/
theorem epi_of_epi_left {f g : Over X} (k : f ⟶ g) [hk : Epi k.left] : Epi k :=
  (forget X).epi_of_epi_map hk


/--
If `k.left` is a monomorphism, then `k` is a monomorphism. In other words, `Over.forget X` reflects
monomorphisms.
The converse of `CategoryTheory.Over.mono_left_of_mono`.

This lemma is not an instance, to avoid loops in type class inference.
-/
theorem mono_of_mono_left {f g : Over X} (k : f ⟶ g) [hk : Mono k.left] : Mono k :=
  (forget X).mono_of_mono_map hk


/--
If `k` is a monomorphism, then `k.left` is a monomorphism. In other words, `Over.forget X` preserves
monomorphisms.
The converse of `CategoryTheory.Over.mono_of_mono_left`.
-/
instance mono_left_of_mono {f g : Over X} (k : f ⟶ g) [Mono k] : Mono k.left := by
  /-
    T : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} T
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    X : T
    f g : CategoryTheory.Over X
    k : Quiver.Hom f g
    inst✝ : CategoryTheory.Mono k
    ⊢ CategoryTheory.Mono k.left
  -/
  refine ⟨fun {Y : T} l m a => ?_⟩
  let l' : mk (m ≫ f.hom) ⟶ f := homMk l (by
        dsimp; rw [← Over.w k, ← Category.assoc, congrArg (· ≫ g.hom) a, Category.assoc])
  /-
    T : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} T
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    X : T
    f g : CategoryTheory.Over X
    k : Quiver.Hom f g
    inst✝ : CategoryTheory.Mono k
    Y : T
    l m : Quiver.Hom Y f.left
    a : Eq (CategoryTheory.CategoryStruct.comp l k.left) (CategoryTheory.CategoryS …
    l' : Quiver.Hom (CategoryTheory.Over.mk (CategoryTheory.CategoryStruct.comp m  …
    ⊢ Eq l m
  -/
  suffices l' = (homMk m : mk (m ≫ f.hom) ⟶ f) by apply congrArg CommaMorphism.left this
  /-
    T : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} T
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    X : T
    f g : CategoryTheory.Over X
    k : Quiver.Hom f g
    inst✝ : CategoryTheory.Mono k
    Y : T
    l m : Quiver.Hom Y f.left
    a : Eq (CategoryTheory.CategoryStruct.comp l k.left) (CategoryTheory.CategoryS …
    l' : Quiver.Hom (CategoryTheory.Over.mk (CategoryTheory.CategoryStruct.comp m  …
    ⊢ Eq l' (CategoryTheory.Over.homMk m ⋯)
  -/
  rw [← cancel_mono k]
  /-
    T : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} T
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    X : T
    f g : CategoryTheory.Over X
    k : Quiver.Hom f g
    inst✝ : CategoryTheory.Mono k
    Y : T
    l m : Quiver.Hom Y f.left
    a : Eq (CategoryTheory.CategoryStruct.comp l k.left) (CategoryTheory.CategoryS …
    l' : Quiver.Hom (CategoryTheory.Over.mk (CategoryTheory.CategoryStruct.comp m  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp l' k) (CategoryTheory.CategoryStruct. …
  -/
  ext
  /-
    case h
    T : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} T
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    X : T
    f g : CategoryTheory.Over X
    k : Quiver.Hom f g
    inst✝ : CategoryTheory.Mono k
    Y : T
    l m : Quiver.Hom Y f.left
    a : Eq (CategoryTheory.CategoryStruct.comp l k.left) (CategoryTheory.CategoryS …
    l' : Quiver.Hom (CategoryTheory.Over.mk (CategoryTheory.CategoryStruct.comp m  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp l' k).left (CategoryTheory.CategorySt …
  -/
  apply a
  /-
    🎉 no goals
  -/


/-- Given f : Y ⟶ X, this is the obvious functor from (T/X)/f to T/Y -/
@[simps]
def iteratedSliceForward : Over f ⥤ Over f.left where
  obj α := Over.mk α.hom.left
                                      /-
                                        T : Type u₁
                                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                        D : Type u₂
                                        inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                        X : T
                                        f : CategoryTheory.Over X
                                        X✝ Y✝ : CategoryTheory.Over f
                                        κ : Quiver.Hom X✝ Y✝
                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp κ.left.left ((fun α => CategoryTheory …
                                      -/
  map κ := Over.homMk κ.left.left (by dsimp; rw [← Over.w κ]; rfl)
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- Given f : Y ⟶ X, this is the obvious functor from T/Y to (T/X)/f -/
@[simps]
def iteratedSliceBackward : Over f.left ⥤ Over f where
               /-
                 T : Type u₁
                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                 D : Type u₂
                 inst✝ : CategoryTheory.Category.{v₂, u₂} D
                 X : T
                 f : CategoryTheory.Over X
                 g : CategoryTheory.Over f.left
                 ⊢ Eq (CategoryTheory.CategoryStruct.comp g.hom f.hom) (CategoryTheory.Over.mk  …
               -/
  obj g := mk (homMk g.hom : mk (g.hom ≫ f.hom) ⟶ f)
               /-
                 🎉 no goals
               -/
  map α := homMk (homMk α.left (w_assoc α f.hom)) (OverMorphism.ext (w α))


/-- Given f : Y ⟶ X, we have an equivalence between (T/X)/f and T/Y -/
@[simps]
def iteratedSliceEquiv : Over f ≌ Over f.left where
  functor := iteratedSliceForward f
  inverse := iteratedSliceBackward f
                                                       /-
                                                         T : Type u₁
                                                         inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                                         D : Type u₂
                                                         inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                         X : T
                                                         f : CategoryTheory.Over X
                                                         g : CategoryTheory.Over f
                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl ((CategoryTh …
                                                       -/
                                                       /-
                                                         🎉 no goals
                                                       -/
                                           /-
                                             🎉 no goals
                                           -/
  unitIso := NatIso.ofComponents (fun g => Over.isoMk (Over.isoMk (Iso.refl _)))
             /-
               🎉 no goals
             -/
                                             /-
                                               T : Type u₁
                                               inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                               D : Type u₂
                                               inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                               X : T
                                               f : CategoryTheory.Over X
                                               g : CategoryTheory.Over f.left
                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl ((f.iterated …
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
  counitIso := NatIso.ofComponents (fun g => Over.isoMk (Iso.refl _))
               /-
                 🎉 no goals
               -/


theorem iteratedSliceForward_forget :
    iteratedSliceForward f ⋙ forget f.left = forget f ⋙ forget X :=
  rfl


theorem iteratedSliceBackward_forget_forget :
    iteratedSliceBackward f ⋙ forget f ⋙ forget X = forget f.left :=
  rfl


/-- A functor `F : T ⥤ D` induces a functor `Over X ⥤ Over (F.obj X)` in the obvious way. -/
@[simps]
def post (F : T ⥤ D) : Over X ⥤ Over (F.obj X) where
  obj Y := mk <| F.map Y.hom
  map f := Over.homMk (F.map f.left)
        /-
          T : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          X : T
          F : CategoryTheory.Functor T D
          X✝ Y✝ : CategoryTheory.Over X
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f.left) ((fun Y => CategoryThe …
        -/
    (by simp only [Functor.id_obj, mk_left, Functor.const_obj_obj, mk_hom, ← F.map_comp, w])
        /-
          🎉 no goals
        -/


lemma post_comp {E : Type*} [Category E] (F : T ⥤ D) (G : D ⥤ E) :
    post (X := X) (F ⋙ G) = post (X := X) F ⋙ post G :=
  rfl


/-- `post (F ⋙ G)` is isomorphic (actually equal) to `post F ⋙ post G`. -/
@[simps!]
def postComp {E : Type*} [Category E] (F : T ⥤ D) (G : D ⥤ E) :
    post (X := X) (F ⋙ G) ≅ post F ⋙ post G :=
  /-
    T : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} T
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    X : T
    E : Type u_1
    inst✝ : CategoryTheory.Category.{?u.119595, u_1} E
    F : CategoryTheory.Functor T D
    G : CategoryTheory.Functor D E
    ⊢ ∀ {X_1 Y : CategoryTheory.Over X} (f : Quiver.Hom X_1 Y), Eq (CategoryTheory …
  -/
  NatIso.ofComponents (fun X ↦ Iso.refl _)
  /-
    🎉 no goals
  -/


/-- A natural transformation `F ⟶ G` induces a natural transformation on
`Over X` up to `Under.map`. -/
@[simps]
def postMap {F G : T ⥤ D} (e : F ⟶ G) : post F ⋙ map (e.app X) ⟶ post G where
           /-
             T : Type u₁
             inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
             D : Type u₂
             inst✝ : CategoryTheory.Category.{v₂, u₂} D
             X : T
             F G : CategoryTheory.Functor T D
             e : Quiver.Hom F G
             Y : CategoryTheory.Over X
             ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.app Y.left) ((CategoryTheory.Over. …
           -/
  app Y := Over.homMk (e.app Y.left)
           /-
             🎉 no goals
           -/


/-- If `F` and `G` are naturally isomorphic, then `Over.post F` and `Over.post G` are also naturally
isomorphic up to `Over.map` -/
@[simps!]
def postCongr {F G : T ⥤ D} (e : F ≅ G) : post F ⋙ map (e.hom.app X) ≅ post G :=
                               /-
                                 T : Type u₁
                                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                 D : Type u₂
                                 inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                 X : T
                                 F G : CategoryTheory.Functor T D
                                 e : CategoryTheory.Iso F G
                                 A : CategoryTheory.Over X
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.app A.left).hom ((CategoryTheory.O …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents (fun A ↦ Over.isoMk (e.app A.left))
  /-
    🎉 no goals
  -/


instance [F.Faithful] : (Over.post (X := X) F).Faithful where
  map_injective {A B} f g h := by
    /-
      T : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} T
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      X : T
      F : CategoryTheory.Functor T D
      inst✝ : F.Faithful
      A B : CategoryTheory.Over X
      f g : Quiver.Hom A B
      h : Eq ((CategoryTheory.Over.post F).map f) ((CategoryTheory.Over.post F).map g)
      ⊢ Eq f g
    -/
    ext
    /-
      case h
      T : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} T
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      X : T
      F : CategoryTheory.Functor T D
      inst✝ : F.Faithful
      A B : CategoryTheory.Over X
      f g : Quiver.Hom A B
      h : Eq ((CategoryTheory.Over.post F).map f) ((CategoryTheory.Over.post F).map g)
      ⊢ Eq f.left g.left
    -/
    exact F.map_injective (congrArg CommaMorphism.left h)
    /-
      🎉 no goals
    -/


instance [F.Faithful] [F.Full] : (Over.post (X := X) F).Full where
  map_surjective {A B} f := by
    /-
      T : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} T
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      X : T
      F : CategoryTheory.Functor T D
      inst✝¹ : F.Faithful
      inst✝ : F.Full
      A B : CategoryTheory.Over X
      f : Quiver.Hom ((CategoryTheory.Over.post F).obj A) ((CategoryTheory.Over.post …
      ⊢ Exists fun a => Eq ((CategoryTheory.Over.post F).map a) f
    -/
    obtain ⟨a, ha⟩ := F.map_surjective f.left
    /-
      case intro
      T : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} T
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      X : T
      F : CategoryTheory.Functor T D
      inst✝¹ : F.Faithful
      inst✝ : F.Full
      A B : CategoryTheory.Over X
      f : Quiver.Hom ((CategoryTheory.Over.post F).obj A) ((CategoryTheory.Over.post …
      a : Quiver.Hom ((CategoryTheory.Functor.id T).obj A.left) ((CategoryTheory.Fun …
      ha : Eq (F.map a) f.left
      ⊢ Exists fun a => Eq ((CategoryTheory.Over.post F).map a) f
    -/
    have w : a ≫ B.hom = A.hom := F.map_injective <| by simpa [ha] using Over.w _
    /-
      case intro
      T : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} T
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      X : T
      F : CategoryTheory.Functor T D
      inst✝¹ : F.Faithful
      inst✝ : F.Full
      A B : CategoryTheory.Over X
      f : Quiver.Hom ((CategoryTheory.Over.post F).obj A) ((CategoryTheory.Over.post …
      a : Quiver.Hom ((CategoryTheory.Functor.id T).obj A.left) ((CategoryTheory.Fun …
      ha : Eq (F.map a) f.left
      w : Eq (CategoryTheory.CategoryStruct.comp a B.hom) A.hom
      ⊢ Exists fun a => Eq ((CategoryTheory.Over.post F).map a) f
    -/
    exact ⟨Over.homMk a, by ext; simpa⟩
    /-
      🎉 no goals
    -/


instance [F.Full] [F.EssSurj] : (Over.post (X := X) F).EssSurj where
  mem_essImage B := by
    /-
      T : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} T
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      X : T
      F : CategoryTheory.Functor T D
      inst✝¹ : F.Full
      inst✝ : F.EssSurj
      B : CategoryTheory.Over (F.obj X)
      ⊢ Membership.mem (CategoryTheory.Over.post F).essImage B
    -/
    obtain ⟨A', ⟨e⟩⟩ := Functor.EssSurj.mem_essImage (F := F) B.left
    /-
      case intro.intro
      T : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} T
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      X : T
      F : CategoryTheory.Functor T D
      inst✝¹ : F.Full
      inst✝ : F.EssSurj
      B : CategoryTheory.Over (F.obj X)
      A' : T
      e : CategoryTheory.Iso (F.obj A') B.left
      ⊢ Membership.mem (CategoryTheory.Over.post F).essImage B
    -/
    obtain ⟨f, hf⟩ := F.map_surjective (e.hom ≫ B.hom)
    /-
      case intro.intro.intro
      T : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} T
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      X : T
      F : CategoryTheory.Functor T D
      inst✝¹ : F.Full
      inst✝ : F.EssSurj
      B : CategoryTheory.Over (F.obj X)
      A' : T
      e : CategoryTheory.Iso (F.obj A') B.left
      f : Quiver.Hom A' X
      hf : Eq (F.map f) (CategoryTheory.CategoryStruct.comp e.hom B.hom)
      ⊢ Membership.mem (CategoryTheory.Over.post F).essImage B
    -/
    exact ⟨Over.mk f, ⟨Over.isoMk e⟩⟩
    /-
      🎉 no goals
    -/


instance [F.IsEquivalence] : (Over.post (X := X) F).IsEquivalence where


/-- An equivalence of categories induces an equivalence on over categories. -/
@[simps]
def postEquiv (F : T ≌ D) : Over X ≌ Over (F.functor.obj X) where
  functor := Over.post F.functor
  inverse := Over.post (X := F.functor.obj X) F.inverse ⋙ Over.map (F.unitIso.inv.app X)
                                          /-
                                            T : Type u₁
                                            inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                            D : Type u₂
                                            inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                            X : T
                                            F✝ : CategoryTheory.Functor T D
                                            F : CategoryTheory.Equivalence T D
                                            A : CategoryTheory.Over X
                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.unitIso.app A.left).hom (((Categor …
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
  unitIso := NatIso.ofComponents (fun A ↦ Over.isoMk (F.unitIso.app A.left))
             /-
               🎉 no goals
             -/
                                            /-
                                              T : Type u₁
                                              inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                              D : Type u₂
                                              inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                              X : T
                                              F✝ : CategoryTheory.Functor T D
                                              F : CategoryTheory.Equivalence T D
                                              A : CategoryTheory.Over (F.functor.obj X)
                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.counitIso.app A.left).hom ((Catego …
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
  counitIso := NatIso.ofComponents (fun A ↦ Over.isoMk (F.counitIso.app A.left))
               /-
                 🎉 no goals
               -/


/-- Reinterpreting an `F`-costructured arrow `F.obj d ⟶ X` as an arrow over `X` induces a functor
    `CostructuredArrow F X ⥤ Over X`. -/
@[simps!]
def toOver (F : D ⥤ T) (X : T) : CostructuredArrow F X ⥤ Over X :=
  CostructuredArrow.pre F (𝟭 T) X


instance (F : D ⥤ T) (X : T) [F.Faithful] : (toOver F X).Faithful :=
  show (CostructuredArrow.pre _ _ _).Faithful from inferInstance


instance (F : D ⥤ T) (X : T) [F.Full] : (toOver F X).Full :=
  show (CostructuredArrow.pre _ _ _).Full from inferInstance


instance (F : D ⥤ T) (X : T) [F.EssSurj] : (toOver F X).EssSurj :=
  show (CostructuredArrow.pre _ _ _).EssSurj from inferInstance


/-- An equivalence `F` induces an equivalence `CostructuredArrow F X ≌ Over X`. -/
instance isEquivalence_toOver (F : D ⥤ T) (X : T) [F.IsEquivalence] :
    (toOver F X).IsEquivalence :=
  CostructuredArrow.isEquivalence_pre _ _ _


/-- The under category has as objects arrows with domain `X` and as morphisms commutative
    triangles. -/
def Under (X : T) :=
  StructuredArrow X (𝟭 T)


instance (X : T) : Category (Under X) := commaCategory

-- Satisfying the inhabited linter

instance Under.inhabited [Inhabited T] : Inhabited (Under (default : T)) where
  default :=
    { left := default
      right := default
      hom := 𝟙 _ }


@[ext]
theorem UnderMorphism.ext {X : T} {U V : Under X} {f g : U ⟶ V} (h : f.right = g.right) :
    f = g := by
  /-
    T : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} T
    X : T
    U V : CategoryTheory.Under X
    f g : Quiver.Hom U V
    h : Eq f.right g.right
    ⊢ Eq f g
  -/
  let ⟨_,b,_⟩ := f; let ⟨_,e,_⟩ := g
  /-
    T : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} T
    X : T
    U V : CategoryTheory.Under X
    f g : Quiver.Hom U V
    left✝¹ : Quiver.Hom U.left V.left
    b : Quiver.Hom U.right V.right
    w✝¹ : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.fromPUni …
    left✝ : Quiver.Hom U.left V.left
    e : Quiver.Hom U.right V.right
    w✝ : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.fromPUnit …
    h : Eq { left := left✝¹, right := b, w := w✝¹ }.right { left := left✝, right : …
    ⊢ Eq { left := left✝¹, right := b, w := w✝¹ } { left := left✝, right := e, w : …
  -/
  congr; simp only [eq_iff_true_of_subsingleton]
         /-
           🎉 no goals
         -/


@[simp]
                                                       /-
                                                         T : Type u₁
                                                         inst✝ : CategoryTheory.Category.{v₁, u₁} T
                                                         X : T
                                                         U : CategoryTheory.Under X
                                                         ⊢ Eq U.left { as := PUnit.unit }
                                                       -/
theorem under_left (U : Under X) : U.left = ⟨⟨⟩⟩ := by simp only
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
theorem id_right (U : Under X) : CommaMorphism.right (𝟙 U) = 𝟙 U.right :=
  rfl


@[simp]
theorem comp_right (a b c : Under X) (f : a ⟶ b) (g : b ⟶ c) : (f ≫ g).right = f.right ≫ g.right :=
  rfl


@[reassoc (attr := simp)]
                                                                      /-
                                                                        T : Type u₁
                                                                        inst✝ : CategoryTheory.Category.{v₁, u₁} T
                                                                        X : T
                                                                        A B : CategoryTheory.Under X
                                                                        f : Quiver.Hom A B
                                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp A.hom f.right) B.hom
                                                                      -/
theorem w {A B : Under X} (f : A ⟶ B) : A.hom ≫ f.right = B.hom := by have := f.w; aesop_cat
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


/-- To give an object in the under category, it suffices to give an arrow with domain `X`. -/
@[simps! right hom]
def mk {X Y : T} (f : X ⟶ Y) : Under X :=
  StructuredArrow.mk f


/-- To give a morphism in the under category, it suffices to give a morphism fitting in a
    commutative triangle. -/
@[simps!]
def homMk {U V : Under X} (f : U.right ⟶ V.right) (w : U.hom ≫ f = V.hom := by aesop_cat) : U ⟶ V :=
  StructuredArrow.homMk f w

-- Porting note: simp solves this; simpNF still sees them after `-simp` (?)

/-- Construct an isomorphism in the over category given isomorphisms of the objects whose forward
direction gives a commutative triangle.
-/
def isoMk {f g : Under X} (hr : f.right ≅ g.right)
    (hw : f.hom ≫ hr.hom = g.hom := by aesop_cat) : f ≅ g :=
  StructuredArrow.isoMk hr hw


@[simp]
theorem isoMk_hom_right {f g : Under X} (hr : f.right ≅ g.right) (hw : f.hom ≫ hr.hom = g.hom) :
    (isoMk hr hw).hom.right = hr.hom :=
  rfl


@[simp]
theorem isoMk_inv_right {f g : Under X} (hr : f.right ≅ g.right) (hw : f.hom ≫ hr.hom = g.hom) :
    (isoMk hr hw).inv.right = hr.inv :=
  rfl


@[reassoc (attr := simp)]
lemma hom_right_inv_right {f g : Under X} (e : f ≅ g) :
    e.hom.right ≫ e.inv.right = 𝟙 f.right := by
  /-
    T : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} T
    X : T
    f g : CategoryTheory.Under X
    e : CategoryTheory.Iso f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp e.hom.right e.inv.right) (CategoryThe …
  -/
  simp [← Under.comp_right]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma inv_right_hom_right {f g : Under X} (e : f ≅ g) :
    e.inv.right ≫ e.hom.right = 𝟙 g.right := by
  /-
    T : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} T
    X : T
    f g : CategoryTheory.Under X
    e : CategoryTheory.Iso f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp e.inv.right e.hom.right) (CategoryThe …
  -/
  simp [← Under.comp_right]
  /-
    🎉 no goals
  -/


/-- The forgetful functor mapping an arrow to its domain. -/
def forget : Under X ⥤ T :=
  Comma.snd _ _


@[simp]
theorem forget_obj {U : Under X} : (forget X).obj U = U.right :=
  rfl


@[simp]
theorem forget_map {U V : Under X} {f : U ⟶ V} : (forget X).map f = f.right :=
  rfl


/-- The natural cone over the forgetful functor `Under X ⥤ T` with cone point `X`. -/
@[simps]
def forgetCone (X : T) : Limits.Cone (forget X) :=
  { pt := X
    π := { app := Comma.hom } }


/-- A morphism `X ⟶ Y` induces a functor `Under Y ⥤ Under X` in the obvious way. -/
def map {Y : T} (f : X ⟶ Y) : Under Y ⥤ Under X :=
  Comma.mapLeft _ <| Discrete.natTrans fun _ => f


@[simp]
theorem map_obj_right : ((map f).obj U).right = U.right :=
  rfl


@[simp]
theorem map_obj_hom : ((map f).obj U).hom = f ≫ U.hom :=
  rfl


@[simp]
theorem map_map_right : ((map f).map g).right = g.right :=
  rfl

/-- If `f` is an isomorphism, `map f` is an equivalence of categories. -/
def mapIso {Y : T} (f : X ≅ Y) : Under Y ≌ Under X :=
  Comma.mapLeftIso _ <| Discrete.natIso fun _ ↦ f.symm


/-- Mapping by the identity morphism is just the identity functor. -/
theorem mapId_eq (Y : T) : map (𝟙 Y) = 𝟭 _ := by
  /-
    T : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} T
    Y : T
    ⊢ Eq (CategoryTheory.Under.map (CategoryTheory.CategoryStruct.id Y)) (Category …
  -/
  fapply Functor.ext
    /-
      case h_obj
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      Y : T
      ⊢ ∀ (X : CategoryTheory.Under Y), Eq ((CategoryTheory.Under.map (CategoryTheor …
    -/
  · intro x
    /-
      case h_obj
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      Y : T
      x : CategoryTheory.Under Y
      ⊢ Eq ((CategoryTheory.Under.map (CategoryTheory.CategoryStruct.id Y)).obj x) ( …
    -/
    dsimp [Under, Under.map, Comma.mapLeft]
    /-
      case h_obj
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      Y : T
      x : CategoryTheory.Under Y
      ⊢ Eq { left := x.left, right := x.right, hom := CategoryTheory.CategoryStruct. …
    -/
    simp only [Category.id_comp]
    /-
      case h_obj
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      Y : T
      x : CategoryTheory.Under Y
      ⊢ Eq { left := x.left, right := x.right, hom := x.hom } x
    -/
    exact rfl
    /-
      🎉 no goals
    -/
    /-
      case h_map
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      Y : T
      ⊢ autoParam (∀ (X Y_1 : CategoryTheory.Under Y) (f : Quiver.Hom X Y_1), Eq ((C …
    -/
  · intros x y u
    /-
      case h_map
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      Y : T
      x y : CategoryTheory.Under Y
      u : Quiver.Hom x y
      ⊢ Eq ((CategoryTheory.Under.map (CategoryTheory.CategoryStruct.id Y)).map u) ( …
    -/
    dsimp [Under, Under.map, Comma.mapLeft]
    /-
      case h_map
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      Y : T
      x y : CategoryTheory.Under Y
      u : Quiver.Hom x y
      ⊢ Eq { left := CategoryTheory.CategoryStruct.id x.left, right := u.right, w := …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Mapping by the identity morphism is just the identity functor. -/
@[simps!]
def mapId (Y : T) : map (𝟙 Y) ≅ 𝟭 _ := eqToIso (mapId_eq Y)


/-- Mapping by `f` and then forgetting is the same as forgetting. -/
theorem mapForget_eq {X Y : T} (f : X ⟶ Y) :
    (map f) ⋙ (forget X) = (forget Y) := by
  /-
    T : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} T
    X Y : T
    f : Quiver.Hom X Y
    ⊢ Eq ((CategoryTheory.Under.map f).comp (CategoryTheory.Under.forget X)) (Cate …
  -/
  fapply Functor.ext
    /-
      case h_obj
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      X Y : T
      f : Quiver.Hom X Y
      ⊢ ∀ (X_1 : CategoryTheory.Under Y), Eq (((CategoryTheory.Under.map f).comp (Ca …
    -/
  · dsimp [Under, Under.map]; intro x; exact rfl
                                       /-
                                         🎉 no goals
                                       -/
    /-
      case h_map
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      X Y : T
      f : Quiver.Hom X Y
      ⊢ autoParam (∀ (X_1 Y_1 : CategoryTheory.Under Y) (f_1 : Quiver.Hom X_1 Y_1),  …
    -/
  · intros x y u; simp
                  /-
                    🎉 no goals
                  -/


/-- The natural isomorphism arising from `mapForget_eq`. -/
def mapForget {X Y : T} (f : X ⟶ Y) :
    (map f) ⋙ (forget X) ≅ (forget Y) := eqToIso (mapForget_eq f)


@[simp]
theorem eqToHom_right {X : T} {U V : Under X} (e : U = V) :
    (eqToHom e).right = eqToHom (e ▸ rfl : U.right = V.right) := by
  /-
    T : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} T
    X : T
    U V : CategoryTheory.Under X
    e : Eq U V
    ⊢ Eq (CategoryTheory.eqToHom e).right (CategoryTheory.eqToHom ⋯)
  -/
  subst e; rfl
           /-
             🎉 no goals
           -/


/-- Mapping by the composite morphism `f ≫ g` is the same as mapping by `f` then by `g`. -/
theorem mapComp_eq {X Y Z : T} (f : X ⟶ Y) (g : Y ⟶ Z) :
    map (f ≫ g) = (map g) ⋙ (map f) := by
  /-
    T : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} T
    X Y Z : T
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.Under.map (CategoryTheory.CategoryStruct.comp f g)) ((Cat …
  -/
  fapply Functor.ext
    /-
      case h_obj
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      X Y Z : T
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ ∀ (X_1 : CategoryTheory.Under Z), Eq ((CategoryTheory.Under.map (CategoryThe …
    -/
  · simp [Under.map, Comma.mapLeft]
    /-
      🎉 no goals
    -/
    /-
      case h_map
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      X Y Z : T
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ autoParam (∀ (X_1 Y_1 : CategoryTheory.Under Z) (f_1 : Quiver.Hom X_1 Y_1),  …
    -/
  · intro U V k
    /-
      case h_map
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      X Y Z : T
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      U V : CategoryTheory.Under Z
      k : Quiver.Hom U V
      ⊢ Eq ((CategoryTheory.Under.map (CategoryTheory.CategoryStruct.comp f g)).map  …
    -/
    ext
    /-
      case h_map.h
      T : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} T
      X Y Z : T
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      U V : CategoryTheory.Under Z
      k : Quiver.Hom U V
      ⊢ Eq ((CategoryTheory.Under.map (CategoryTheory.CategoryStruct.comp f g)).map  …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The natural isomorphism arising from `mapComp_eq`. -/
@[simps!]
def mapComp {Y Z : T} (f : X ⟶ Y) (g : Y ⟶ Z) : map (f ≫ g) ≅ map g ⋙ map f :=
  eqToIso (mapComp_eq f g)


/-- If `f = g`, then `map f` is naturally isomorphic to `map g`. -/
@[simps!]
def mapCongr {X Y : T} (f g : X ⟶ Y) (h : f = g) :
    map f ≅ map g :=
                                           /-
                                             T : Type u₁
                                             inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                             D : Type u₂
                                             inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                             X✝ X Y : T
                                             f g : Quiver.Hom X Y
                                             h : Eq f g
                                             A : CategoryTheory.Under Y
                                             ⊢ Eq ((CategoryTheory.Under.map f).obj A) ((CategoryTheory.Under.map g).obj A)
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
  NatIso.ofComponents (fun A ↦ eqToIso (by rw [h]))
  /-
    🎉 no goals
  -/


variable (T) in
/-- The functor defined by the under categories.-/
@[simps] def mapFunctor : Tᵒᵖ  ⥤ Cat where
  obj X := Cat.of (Under X.unop)
  map f := map f.unop
  map_id X := mapId_eq X.unop
  map_comp f g := mapComp_eq (g.unop) (f.unop)


instance forget_reflects_iso : (forget X).ReflectsIsomorphisms where
  reflects {Y Z} f t := by
    let g : Z ⟶ Y := Under.homMk (inv ((Under.forget X).map f))
      ((IsIso.comp_inv_eq _).2 (Under.w f).symm)
    /-
      T : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      X : T
      Y Z : CategoryTheory.Under X
      f : Quiver.Hom Y Z
      t : CategoryTheory.IsIso ((CategoryTheory.Under.forget X).map f)
      g : Quiver.Hom Z Y := CategoryTheory.Under.homMk (CategoryTheory.inv ((Categor …
      ⊢ CategoryTheory.IsIso f
    -/
    dsimp [forget] at t
    /-
      T : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      X : T
      Y Z : CategoryTheory.Under X
      f : Quiver.Hom Y Z
      t : CategoryTheory.IsIso f.right
      g : Quiver.Hom Z Y := CategoryTheory.Under.homMk (CategoryTheory.inv ((Categor …
      ⊢ CategoryTheory.IsIso f
    -/
    refine ⟨⟨g, ⟨?_,?_⟩⟩⟩
    /-
      case refine_1
      T : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      X : T
      Y Z : CategoryTheory.Under X
      f : Quiver.Hom Y Z
      t : CategoryTheory.IsIso f.right
      g : Quiver.Hom Z Y := CategoryTheory.Under.homMk (CategoryTheory.inv ((Categor …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct.i …
    -/
    repeat (ext; simp [g])
    /-
      🎉 no goals
    -/


/-- The identity under `X` is initial. -/
noncomputable def mkIdInitial : Limits.IsInitial (mk (𝟙 X)) :=
  StructuredArrow.mkIdInitial


instance forget_faithful : (forget X).Faithful where

-- TODO: Show the converse holds if `T` has binary coproducts.

/-- If `k.right` is a monomorphism, then `k` is a monomorphism. In other words, `Under.forget X`
reflects epimorphisms.
The converse does not hold without additional assumptions on the underlying category, see
`CategoryTheory.Under.mono_right_of_mono`.
-/
theorem mono_of_mono_right {f g : Under X} (k : f ⟶ g) [hk : Mono k.right] : Mono k :=
  (forget X).mono_of_mono_map hk


/--
If `k.right` is an epimorphism, then `k` is an epimorphism. In other words, `Under.forget X`
reflects epimorphisms.
The converse of `CategoryTheory.Under.epi_right_of_epi`.

This lemma is not an instance, to avoid loops in type class inference.
-/
theorem epi_of_epi_right {f g : Under X} (k : f ⟶ g) [hk : Epi k.right] : Epi k :=
  (forget X).epi_of_epi_map hk


/--
If `k` is an epimorphism, then `k.right` is an epimorphism. In other words, `Under.forget X`
preserves epimorphisms.
The converse of `CategoryTheory.under.epi_of_epi_right`.
-/
instance epi_right_of_epi {f g : Under X} (k : f ⟶ g) [Epi k] : Epi k.right := by
  /-
    T : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} T
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    X : T
    f g : CategoryTheory.Under X
    k : Quiver.Hom f g
    inst✝ : CategoryTheory.Epi k
    ⊢ CategoryTheory.Epi k.right
  -/
  refine ⟨fun {Y : T} l m a => ?_⟩
  let l' : g ⟶ mk (g.hom ≫ m) := homMk l (by
    dsimp; rw [← Under.w k, Category.assoc, a, Category.assoc])
  -- Porting note: add type ascription here to `homMk m`
  /-
    T : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} T
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    X : T
    f g : CategoryTheory.Under X
    k : Quiver.Hom f g
    inst✝ : CategoryTheory.Epi k
    Y : T
    l m : Quiver.Hom g.right Y
    a : Eq (CategoryTheory.CategoryStruct.comp k.right l) (CategoryTheory.Category …
    l' : Quiver.Hom g (CategoryTheory.Under.mk (CategoryTheory.CategoryStruct.comp …
    ⊢ Eq l m
  -/
  suffices l' = (homMk m : g ⟶ mk (g.hom ≫ m)) by apply congrArg CommaMorphism.right this
  /-
    T : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} T
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    X : T
    f g : CategoryTheory.Under X
    k : Quiver.Hom f g
    inst✝ : CategoryTheory.Epi k
    Y : T
    l m : Quiver.Hom g.right Y
    a : Eq (CategoryTheory.CategoryStruct.comp k.right l) (CategoryTheory.Category …
    l' : Quiver.Hom g (CategoryTheory.Under.mk (CategoryTheory.CategoryStruct.comp …
    ⊢ Eq l' (CategoryTheory.Under.homMk m ⋯)
  -/
  rw [← cancel_epi k]; ext; apply a
                            /-
                              🎉 no goals
                            -/


/-- A functor `F : T ⥤ D` induces a functor `Under X ⥤ Under (F.obj X)` in the obvious way. -/
@[simps]
def post {X : T} (F : T ⥤ D) : Under X ⥤ Under (F.obj X) where
  obj Y := mk <| F.map Y.hom
  map f := Under.homMk (F.map f.right)
        /-
          T : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          X✝¹ X : T
          F : CategoryTheory.Functor T D
          X✝ Y✝ : CategoryTheory.Under X
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun Y => CategoryTheory.Under.mk (F …
        -/
    (by simp only [Functor.id_obj, Functor.const_obj_obj, mk_right, mk_hom, ← F.map_comp, w])
        /-
          🎉 no goals
        -/


/-- `post (F ⋙ G)` is isomorphic (actually equal) to `post F ⋙ post G`. -/
@[simps!]
def postComp {E : Type*} [Category E] (F : T ⥤ D) (G : D ⥤ E) :
    post (X := X) (F ⋙ G) ≅ post F ⋙ post G :=
  /-
    T : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} T
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    X : T
    E : Type u_1
    inst✝ : CategoryTheory.Category.{?u.249786, u_1} E
    F : CategoryTheory.Functor T D
    G : CategoryTheory.Functor D E
    ⊢ ∀ {X_1 Y : CategoryTheory.Under X} (f : Quiver.Hom X_1 Y), Eq (CategoryTheor …
  -/
  NatIso.ofComponents (fun X ↦ Iso.refl _)
  /-
    🎉 no goals
  -/


/-- A natural transformation `F ⟶ G` induces a natural transformation on
`Under X` up to `Under.map`. -/
@[simps]
def postMap {F G : T ⥤ D} (e : F ⟶ G) : post (X := X) F ⟶ post G ⋙ map (e.app X) where
           /-
             T : Type u₁
             inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
             D : Type u₂
             inst✝ : CategoryTheory.Category.{v₂, u₂} D
             X : T
             F G : CategoryTheory.Functor T D
             e : Quiver.Hom F G
             Y : CategoryTheory.Under X
             ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Under.post F).obj Y) …
           -/
  app Y := Under.homMk (e.app Y.right)
           /-
             🎉 no goals
           -/


/-- If `F` and `G` are naturally isomorphic, then `Under.post F` and `Under.post G` are also
naturally isomorphic up to `Under.map` -/
@[simps!]
def postCongr {F G : T ⥤ D} (e : F ≅ G) : post F ≅ post G ⋙ map (e.hom.app X) :=
                               /-
                                 T : Type u₁
                                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                 D : Type u₂
                                 inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                 X : T
                                 F G : CategoryTheory.Functor T D
                                 e : CategoryTheory.Iso F G
                                 A : CategoryTheory.Under X
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Under.post F).obj A) …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents (fun A ↦ Under.isoMk (e.app A.right))
  /-
    🎉 no goals
  -/


instance [F.Faithful] : (Under.post (X := X) F).Faithful where
  map_injective {A B} f g h := by
    /-
      T : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} T
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      X : T
      F : CategoryTheory.Functor T D
      inst✝ : F.Faithful
      A B : CategoryTheory.Under X
      f g : Quiver.Hom A B
      h : Eq ((CategoryTheory.Under.post F).map f) ((CategoryTheory.Under.post F).ma …
      ⊢ Eq f g
    -/
    ext
    /-
      case h
      T : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} T
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      X : T
      F : CategoryTheory.Functor T D
      inst✝ : F.Faithful
      A B : CategoryTheory.Under X
      f g : Quiver.Hom A B
      h : Eq ((CategoryTheory.Under.post F).map f) ((CategoryTheory.Under.post F).ma …
      ⊢ Eq f.right g.right
    -/
    exact F.map_injective (congrArg CommaMorphism.right h)
    /-
      🎉 no goals
    -/


instance [F.Faithful] [F.Full] : (Under.post (X := X) F).Full where
  map_surjective {A B} f := by
    /-
      T : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} T
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      X : T
      F : CategoryTheory.Functor T D
      inst✝¹ : F.Faithful
      inst✝ : F.Full
      A B : CategoryTheory.Under X
      f : Quiver.Hom ((CategoryTheory.Under.post F).obj A) ((CategoryTheory.Under.po …
      ⊢ Exists fun a => Eq ((CategoryTheory.Under.post F).map a) f
    -/
    obtain ⟨a, ha⟩ := F.map_surjective f.right
    /-
      case intro
      T : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} T
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      X : T
      F : CategoryTheory.Functor T D
      inst✝¹ : F.Faithful
      inst✝ : F.Full
      A B : CategoryTheory.Under X
      f : Quiver.Hom ((CategoryTheory.Under.post F).obj A) ((CategoryTheory.Under.po …
      a : Quiver.Hom ((CategoryTheory.Functor.id T).obj A.right) ((CategoryTheory.Fu …
      ha : Eq (F.map a) f.right
      ⊢ Exists fun a => Eq ((CategoryTheory.Under.post F).map a) f
    -/
    dsimp at a
    /-
      case intro
      T : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} T
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      X : T
      F : CategoryTheory.Functor T D
      inst✝¹ : F.Faithful
      inst✝ : F.Full
      A B : CategoryTheory.Under X
      f : Quiver.Hom ((CategoryTheory.Under.post F).obj A) ((CategoryTheory.Under.po …
      a : Quiver.Hom A.right B.right
      ha : Eq (F.map a) f.right
      ⊢ Exists fun a => Eq ((CategoryTheory.Under.post F).map a) f
    -/
    have w : A.hom ≫ a = B.hom := F.map_injective <| by simpa [ha] using Under.w f
    /-
      case intro
      T : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} T
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      X : T
      F : CategoryTheory.Functor T D
      inst✝¹ : F.Faithful
      inst✝ : F.Full
      A B : CategoryTheory.Under X
      f : Quiver.Hom ((CategoryTheory.Under.post F).obj A) ((CategoryTheory.Under.po …
      a : Quiver.Hom A.right B.right
      ha : Eq (F.map a) f.right
      w : Eq (CategoryTheory.CategoryStruct.comp A.hom a) B.hom
      ⊢ Exists fun a => Eq ((CategoryTheory.Under.post F).map a) f
    -/
    exact ⟨Under.homMk a, by ext; simpa⟩
    /-
      🎉 no goals
    -/


instance [F.Full] [F.EssSurj] : (Under.post (X := X) F).EssSurj where
  mem_essImage B := by
    /-
      T : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} T
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      X : T
      F : CategoryTheory.Functor T D
      inst✝¹ : F.Full
      inst✝ : F.EssSurj
      B : CategoryTheory.Under (F.obj X)
      ⊢ Membership.mem (CategoryTheory.Under.post F).essImage B
    -/
    obtain ⟨B', ⟨e⟩⟩ := Functor.EssSurj.mem_essImage (F := F) B.right
    /-
      case intro.intro
      T : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} T
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      X : T
      F : CategoryTheory.Functor T D
      inst✝¹ : F.Full
      inst✝ : F.EssSurj
      B : CategoryTheory.Under (F.obj X)
      B' : T
      e : CategoryTheory.Iso (F.obj B') B.right
      ⊢ Membership.mem (CategoryTheory.Under.post F).essImage B
    -/
    obtain ⟨f, hf⟩ := F.map_surjective (B.hom ≫ e.inv)
    /-
      case intro.intro.intro
      T : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} T
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      X : T
      F : CategoryTheory.Functor T D
      inst✝¹ : F.Full
      inst✝ : F.EssSurj
      B : CategoryTheory.Under (F.obj X)
      B' : T
      e : CategoryTheory.Iso (F.obj B') B.right
      f : Quiver.Hom X B'
      hf : Eq (F.map f) (CategoryTheory.CategoryStruct.comp B.hom e.inv)
      ⊢ Membership.mem (CategoryTheory.Under.post F).essImage B
    -/
    exact ⟨Under.mk f, ⟨Under.isoMk e⟩⟩
    /-
      🎉 no goals
    -/


instance [F.IsEquivalence] : (Under.post (X := X) F).IsEquivalence where


/-- An equivalence of categories induces an equivalence on under categories. -/
@[simps]
def postEquiv (F : T ≌ D) : Under X ≌ Under (F.functor.obj X) where
  functor := post F.functor
  inverse := post (X := F.functor.obj X) F.inverse ⋙ Under.map (F.unitIso.hom.app X)
                                          /-
                                            T : Type u₁
                                            inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                            D : Type u₂
                                            inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                            X : T
                                            F✝ : CategoryTheory.Functor T D
                                            F : CategoryTheory.Equivalence T D
                                            A : CategoryTheory.Under X
                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Category …
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
  unitIso := NatIso.ofComponents (fun A ↦ Under.isoMk (F.unitIso.app A.right))
             /-
               🎉 no goals
             -/
                                            /-
                                              T : Type u₁
                                              inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                              D : Type u₂
                                              inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                              X : T
                                              F✝ : CategoryTheory.Functor T D
                                              F : CategoryTheory.Equivalence T D
                                              A : CategoryTheory.Under (F.functor.obj X)
                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.Under.post F.inver …
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
  counitIso := NatIso.ofComponents (fun A ↦ Under.isoMk (F.counitIso.app A.right))
               /-
                 🎉 no goals
               -/


/-- Reinterpreting an `F`-structured arrow `X ⟶ F.obj d` as an arrow under `X` induces a functor
    `StructuredArrow X F ⥤ Under X`. -/
@[simps!]
def toUnder (X : T) (F : D ⥤ T) : StructuredArrow X F ⥤ Under X :=
  StructuredArrow.pre X F (𝟭 T)


instance (X : T) (F : D ⥤ T) [F.Faithful] : (toUnder X F).Faithful :=
  show (StructuredArrow.pre _ _ _).Faithful from inferInstance


instance (X : T) (F : D ⥤ T) [F.Full] : (toUnder X F).Full :=
  show (StructuredArrow.pre _ _ _).Full from inferInstance


instance (X : T) (F : D ⥤ T) [F.EssSurj] : (toUnder X F).EssSurj :=
  show (StructuredArrow.pre _ _ _).EssSurj from inferInstance


/-- An equivalence `F` induces an equivalence `StructuredArrow X F ≌ Under X`. -/
instance isEquivalence_toUnder (X : T) (F : D ⥤ T) [F.IsEquivalence] :
    (toUnder X F).IsEquivalence :=
  StructuredArrow.isEquivalence_pre _ _ _


/-- Given `X : T`, to upgrade a functor `F : S ⥤ T` to a functor `S ⥤ Over X`, it suffices to
    provide maps `F.obj Y ⟶ X` for all `Y` making the obvious triangles involving all `F.map g`
    commute. -/
@[simps! obj_left map_left]
def toOver (F : S ⥤ T) (X : T) (f : (Y : S) → F.obj Y ⟶ X)
    (h : ∀ {Y Z : S} (g : Y ⟶ Z), F.map g ≫ f Z = f Y) : S ⥤ Over X :=
  F.toCostructuredArrow (𝟭 _) X f h


/-- Upgrading a functor `S ⥤ T` to a functor `S ⥤ Over X` and composing with the forgetful functor
    `Over X ⥤ T` recovers the original functor. -/
def toOverCompForget (F : S ⥤ T) (X : T) (f : (Y : S) → F.obj Y ⟶ X)
    (h : ∀ {Y Z : S} (g : Y ⟶ Z), F.map g ≫ f Z = f Y) : F.toOver X f h ⋙ Over.forget _ ≅ F :=
  Iso.refl _


@[simp]
lemma toOver_comp_forget (F : S ⥤ T) (X : T) (f : (Y : S) → F.obj Y ⟶ X)
    (h : ∀ {Y Z : S} (g : Y ⟶ Z), F.map g ≫ f Z = f Y) : F.toOver X f h ⋙ Over.forget _ = F :=
  rfl


/-- Given `X : T`, to upgrade a functor `F : S ⥤ T` to a functor `S ⥤ Under X`, it suffices to
    provide maps `X ⟶ F.obj Y` for all `Y` making the obvious triangles involving all `F.map g`
    commute. -/
@[simps! obj_right map_right]
def toUnder (F : S ⥤ T) (X : T) (f : (Y : S) → X ⟶ F.obj Y)
    (h : ∀ {Y Z : S} (g : Y ⟶ Z), f Y ≫ F.map g = f Z) : S ⥤ Under X :=
  F.toStructuredArrow X (𝟭 _) f h


/-- Upgrading a functor `S ⥤ T` to a functor `S ⥤ Under X` and composing with the forgetful functor
    `Under X ⥤ T` recovers the original functor. -/
def toUnderCompForget (F : S ⥤ T) (X : T) (f : (Y : S) → X ⟶ F.obj Y)
    (h : ∀ {Y Z : S} (g : Y ⟶ Z), f Y ≫ F.map g = f Z) : F.toUnder X f h ⋙ Under.forget _ ≅ F :=
  Iso.refl _


@[simp]
lemma toUnder_comp_forget (F : S ⥤ T) (X : T) (f : (Y : S) → X ⟶ F.obj Y)
    (h : ∀ {Y Z : S} (g : Y ⟶ Z), f Y ≫ F.map g = f Z) : F.toUnder X f h ⋙ Under.forget _ = F :=
  rfl


/-- A functor from the structured arrow category on the projection functor for any structured
arrow category. -/
@[simps!]
def ofStructuredArrowProjEquivalence.functor (F : D ⥤ T) (Y : T) (X : D) :
    StructuredArrow X (StructuredArrow.proj Y F) ⥤ StructuredArrow Y (Under.forget X ⋙ F) :=
  Functor.toStructuredArrow
    (Functor.toUnder (StructuredArrow.proj X _ ⋙ StructuredArrow.proj Y _) _
                   /-
                     T : Type u₁
                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                     D : Type u₂
                     inst✝ : CategoryTheory.Category.{v₂, u₂} D
                     F : CategoryTheory.Functor D T
                     Y : T
                     X : D
                     g : CategoryTheory.StructuredArrow X (CategoryTheory.StructuredArrow.proj Y F)
                     ⊢ Quiver.Hom X (((CategoryTheory.StructuredArrow.proj X (CategoryTheory.Struct …
                   -/
                   /-
                     🎉 no goals
                   -/
      (fun g => by exact g.hom) (fun m => by have := m.w; aesop_cat)) _ _
                                                          /-
                                                            🎉 no goals
                                                          -/
                               /-
                                 T : Type u₁
                                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                 D : Type u₂
                                 inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                 F : CategoryTheory.Functor D T
                                 Y : T
                                 X : D
                                 ⊢ ∀ {Y_1 Z : CategoryTheory.StructuredArrow X (CategoryTheory.StructuredArrow. …
                               -/
    (fun f => f.right.hom) (by simp)
                               /-
                                 🎉 no goals
                               -/


/-- The inverse functor of `ofStructuredArrowProjEquivalence.functor`. -/
@[simps!]
def ofStructuredArrowProjEquivalence.inverse (F : D ⥤ T) (Y : T) (X : D) :
    StructuredArrow Y (Under.forget X ⋙ F) ⥤ StructuredArrow X (StructuredArrow.proj Y F) :=
  Functor.toStructuredArrow
    (Functor.toStructuredArrow (StructuredArrow.proj Y _ ⋙ Under.forget X) _ _
                   /-
                     T : Type u₁
                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                     D : Type u₂
                     inst✝ : CategoryTheory.Category.{v₂, u₂} D
                     F : CategoryTheory.Functor D T
                     Y : T
                     X : D
                     g : CategoryTheory.StructuredArrow Y ((CategoryTheory.Under.forget X).comp F)
                     ⊢ Quiver.Hom Y (F.obj (((CategoryTheory.StructuredArrow.proj Y ((CategoryTheor …
                   -/
                   /-
                     🎉 no goals
                   -/
      (fun g => by exact g.hom) (fun m => by have := m.w; aesop_cat)) _ _
                                                          /-
                                                            🎉 no goals
                                                          -/
                               /-
                                 T : Type u₁
                                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                 D : Type u₂
                                 inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                 F : CategoryTheory.Functor D T
                                 Y : T
                                 X : D
                                 ⊢ ∀ {Y_1 Z : CategoryTheory.StructuredArrow Y ((CategoryTheory.Under.forget X) …
                               -/
    (fun f => f.right.hom) (by simp)
                               /-
                                 🎉 no goals
                               -/


/-- Characterization of the structured arrow category on the projection functor of any
structured arrow category. -/
def ofStructuredArrowProjEquivalence (F : D ⥤ T) (Y : T) (X : D) :
    StructuredArrow X (StructuredArrow.proj Y F) ≌ StructuredArrow Y (Under.forget X ⋙ F) where
  functor := ofStructuredArrowProjEquivalence.functor F Y X
  inverse := ofStructuredArrowProjEquivalence.inverse F Y X
                                                           /-
                                                             T : Type u₁
                                                             inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                                             D : Type u₂
                                                             inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                             F : CategoryTheory.Functor D T
                                                             Y : T
                                                             X : D
                                                             ⊢ ∀ {X_1 Y_1 : CategoryTheory.StructuredArrow X (CategoryTheory.StructuredArro …
                                                           -/
  unitIso := NatIso.ofComponents (fun _ => Iso.refl _) (by simp)
                                                           /-
                                                             🎉 no goals
                                                           -/
                                                             /-
                                                               T : Type u₁
                                                               inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                                               D : Type u₂
                                                               inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                               F : CategoryTheory.Functor D T
                                                               Y : T
                                                               X : D
                                                               ⊢ ∀ {X_1 Y_1 : CategoryTheory.StructuredArrow Y ((CategoryTheory.Under.forget  …
                                                             -/
  counitIso := NatIso.ofComponents (fun _ => Iso.refl _) (by aesop_cat)
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- The canonical functor from the structured arrow category on the diagonal functor
`T ⥤ T × T` to the structured arrow category on `Under.forget`. -/
@[simps!]
def ofDiagEquivalence.functor (X : T × T) :
    StructuredArrow X (Functor.diag _) ⥤ StructuredArrow X.2 (Under.forget X.1) :=
  Functor.toStructuredArrow
    (Functor.toUnder (StructuredArrow.proj X _) _
                   /-
                     T : Type u₁
                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                     D : Type u₂
                     inst✝ : CategoryTheory.Category.{v₂, u₂} D
                     X : Prod T T
                     f : CategoryTheory.StructuredArrow X (CategoryTheory.Functor.diag T)
                     ⊢ Quiver.Hom X.1 ((CategoryTheory.StructuredArrow.proj X (CategoryTheory.Funct …
                   -/
                   /-
                     🎉 no goals
                   -/
      (fun f => by exact f.hom.1) (fun m => by have := m.w; aesop_cat)) _ _
                                                            /-
                                                              🎉 no goals
                                                            -/
                                    /-
                                      T : Type u₁
                                      inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                      D : Type u₂
                                      inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                      X : Prod T T
                                      Y✝ Z✝ : CategoryTheory.StructuredArrow X (CategoryTheory.Functor.diag T)
                                      m : Quiver.Hom Y✝ Z✝
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun f => f.hom.2) Y✝) ((CategoryThe …
                                    -/
    (fun f => f.hom.2) (fun m => by have := m.w; aesop_cat)
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- The inverse functor of `ofDiagEquivalence.functor`. -/
@[simps!]
def ofDiagEquivalence.inverse (X : T × T) :
    StructuredArrow X.2 (Under.forget X.1) ⥤ StructuredArrow X (Functor.diag _) :=
  Functor.toStructuredArrow (StructuredArrow.proj _ _ ⋙ Under.forget _) _ _
                                                 /-
                                                   T : Type u₁
                                                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                                   D : Type u₂
                                                   inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                   X : Prod T T
                                                   Y✝ Z✝ : CategoryTheory.StructuredArrow X.2 (CategoryTheory.Under.forget X.1)
                                                   m : Quiver.Hom Y✝ Z✝
                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun f => { fst := f.right.hom, snd  …
                                                 -/
    (fun f => (f.right.hom, f.hom)) (fun m => by have := m.w; aesop_cat)
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- Characterization of the structured arrow category on the diagonal functor `T ⥤ T × T`. -/
def ofDiagEquivalence (X : T × T) :
    StructuredArrow X (Functor.diag _) ≌ StructuredArrow X.2 (Under.forget X.1) where
  functor := ofDiagEquivalence.functor X
  inverse := ofDiagEquivalence.inverse X
                                                           /-
                                                             T : Type u₁
                                                             inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                                             D : Type u₂
                                                             inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                             X : Prod T T
                                                             ⊢ ∀ {X_1 Y : CategoryTheory.StructuredArrow X (CategoryTheory.Functor.diag T)} …
                                                           -/
  unitIso := NatIso.ofComponents (fun _ => Iso.refl _) (by simp)
                                                           /-
                                                             🎉 no goals
                                                           -/
                                                             /-
                                                               T : Type u₁
                                                               inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                                               D : Type u₂
                                                               inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                               X : Prod T T
                                                               ⊢ ∀ {X_1 Y : CategoryTheory.StructuredArrow X.2 (CategoryTheory.Under.forget X …
                                                             -/
  counitIso := NatIso.ofComponents (fun _ => Iso.refl _) (by aesop_cat)
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- A version of `StructuredArrow.ofDiagEquivalence` with the roles of the first and second
projection swapped. -/
-- noncomputability is only for performance
noncomputable def ofDiagEquivalence' (X : T × T) :
    StructuredArrow X (Functor.diag _) ≌ StructuredArrow X.1 (Under.forget X.2) :=
  (ofDiagEquivalence X).trans <|
    (ofStructuredArrowProjEquivalence (𝟭 T) X.1 X.2).trans <|
    StructuredArrow.mapNatIso (Under.forget X.2).rightUnitor


/-- The functor used to define the equivalence `ofCommaSndEquivalence`. -/
@[simps]
def ofCommaSndEquivalenceFunctor (c : C) :
    StructuredArrow c (Comma.fst F G) ⥤ Comma (Under.forget c ⋙ F) G where
  obj X := ⟨Under.mk X.hom, X.right.right, X.right.hom⟩
                                         /-
                                           T : Type u₁
                                           inst✝² : CategoryTheory.Category.{v₁, u₁} T
                                           D : Type u₂
                                           inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                           C : Type u₃
                                           inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                           F : CategoryTheory.Functor C T
                                           G : CategoryTheory.Functor D T
                                           c : C
                                           X✝ Y✝ : CategoryTheory.StructuredArrow c (CategoryTheory.Comma.fst F G)
                                           f : Quiver.Hom X✝ Y✝
                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X => { left := CategoryTheory.U …
                                         -/
                                         /-
                                           🎉 no goals
                                         -/
  map f := ⟨Under.homMk f.right.left (by simpa using f.w.symm), f.right.right, by simp⟩
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


/-- The inverse functor used to define the equivalence `ofCommaSndEquivalence`. -/
@[simps!]
def ofCommaSndEquivalenceInverse (c : C) :
    Comma (Under.forget c ⋙ F) G ⥤ StructuredArrow c (Comma.fst F G) :=
  Functor.toStructuredArrow (Comma.preLeft (Under.forget c) F G) _ _
                                       /-
                                         T : Type u₁
                                         inst✝² : CategoryTheory.Category.{v₁, u₁} T
                                         D : Type u₂
                                         inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                         C : Type u₃
                                         inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                         F : CategoryTheory.Functor C T
                                         G : CategoryTheory.Functor D T
                                         c : C
                                         Y✝ Z✝ : CategoryTheory.Comma ((CategoryTheory.Under.forget c).comp F) G
                                         x✝ : Quiver.Hom Y✝ Z✝
                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun Y => Y.left.hom) Y✝) ((Category …
                                       -/
    (fun Y => Y.left.hom) (fun _ => by simp)
                                       /-
                                         🎉 no goals
                                       -/


/-- There is a canonical equivalence between the structured arrow category with domain `c` on
the functor `Comma.fst F G : Comma F G ⥤ F` and the comma category over
`Under.forget c ⋙ F : Under c ⥤ T` and `G`. -/
@[simps]
def ofCommaSndEquivalence (c : C) :
    StructuredArrow c (Comma.fst F G) ≌ Comma (Under.forget c ⋙ F) G where
  functor := ofCommaSndEquivalenceFunctor F G c
  inverse := ofCommaSndEquivalenceInverse F G c
             /-
               T : Type u₁
               inst✝² : CategoryTheory.Category.{v₁, u₁} T
               D : Type u₂
               inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
               C : Type u₃
               inst✝ : CategoryTheory.Category.{v₃, u₃} C
               F : CategoryTheory.Functor C T
               G : CategoryTheory.Functor D T
               c : C
               ⊢ ∀ {X Y : CategoryTheory.StructuredArrow c (CategoryTheory.Comma.fst F G)} (f …
             -/
  unitIso := NatIso.ofComponents (fun _ => Iso.refl _)
             /-
               🎉 no goals
             -/
               /-
                 T : Type u₁
                 inst✝² : CategoryTheory.Category.{v₁, u₁} T
                 D : Type u₂
                 inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                 C : Type u₃
                 inst✝ : CategoryTheory.Category.{v₃, u₃} C
                 F : CategoryTheory.Functor C T
                 G : CategoryTheory.Functor D T
                 c : C
                 ⊢ ∀ {X Y : CategoryTheory.Comma ((CategoryTheory.Under.forget c).comp F) G} (f …
               -/
  counitIso := NatIso.ofComponents (fun _ => Iso.refl _)
               /-
                 🎉 no goals
               -/


/-- A functor from the costructured arrow category on the projection functor for any costructured
arrow category. -/
@[simps!]
def ofCostructuredArrowProjEquivalence.functor (F : T ⥤ D) (Y : D) (X : T) :
    CostructuredArrow (CostructuredArrow.proj F Y) X ⥤ CostructuredArrow (Over.forget X ⋙ F) Y :=
  Functor.toCostructuredArrow
    (Functor.toOver (CostructuredArrow.proj _ X ⋙ CostructuredArrow.proj F Y) _
                   /-
                     T : Type u₁
                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                     D : Type u₂
                     inst✝ : CategoryTheory.Category.{v₂, u₂} D
                     F : CategoryTheory.Functor T D
                     Y : D
                     X : T
                     g : CategoryTheory.CostructuredArrow (CategoryTheory.CostructuredArrow.proj F  …
                     ⊢ Quiver.Hom (((CategoryTheory.CostructuredArrow.proj (CategoryTheory.Costruct …
                   -/
                   /-
                     🎉 no goals
                   -/
      (fun g => by exact g.hom) (fun m => by have := m.w; aesop_cat)) _ _
                                                          /-
                                                            🎉 no goals
                                                          -/
                              /-
                                T : Type u₁
                                inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                D : Type u₂
                                inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                F : CategoryTheory.Functor T D
                                Y : D
                                X : T
                                ⊢ ∀ {Y_1 Z : CategoryTheory.CostructuredArrow (CategoryTheory.CostructuredArro …
                              -/
    (fun f => f.left.hom) (by simp)
                              /-
                                🎉 no goals
                              -/


/-- The inverse functor of `ofCostructuredArrowProjEquivalence.functor`. -/
@[simps!]
def ofCostructuredArrowProjEquivalence.inverse (F : T ⥤ D) (Y : D) (X : T) :
    CostructuredArrow (Over.forget X ⋙ F) Y ⥤ CostructuredArrow (CostructuredArrow.proj F Y) X :=
  Functor.toCostructuredArrow
    (Functor.toCostructuredArrow (CostructuredArrow.proj _ Y ⋙ Over.forget X) _ _
                   /-
                     T : Type u₁
                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                     D : Type u₂
                     inst✝ : CategoryTheory.Category.{v₂, u₂} D
                     F : CategoryTheory.Functor T D
                     Y : D
                     X : T
                     g : CategoryTheory.CostructuredArrow ((CategoryTheory.Over.forget X).comp F) Y
                     ⊢ Quiver.Hom (F.obj (((CategoryTheory.CostructuredArrow.proj ((CategoryTheory. …
                   -/
                   /-
                     🎉 no goals
                   -/
      (fun g => by exact g.hom) (fun m => by have := m.w; aesop_cat)) _ _
                                                          /-
                                                            🎉 no goals
                                                          -/
                              /-
                                T : Type u₁
                                inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                D : Type u₂
                                inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                F : CategoryTheory.Functor T D
                                Y : D
                                X : T
                                ⊢ ∀ {Y_1 Z : CategoryTheory.CostructuredArrow ((CategoryTheory.Over.forget X). …
                              -/
    (fun f => f.left.hom) (by simp)
                              /-
                                🎉 no goals
                              -/


/-- Characterization of the costructured arrow category on the projection functor of any
costructured arrow category. -/
def ofCostructuredArrowProjEquivalence (F : T ⥤ D) (Y : D) (X : T) :
    CostructuredArrow (CostructuredArrow.proj F Y) X
      ≌ CostructuredArrow (Over.forget X ⋙ F) Y where
  functor := ofCostructuredArrowProjEquivalence.functor F Y X
  inverse := ofCostructuredArrowProjEquivalence.inverse F Y X
                                                           /-
                                                             T : Type u₁
                                                             inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                                             D : Type u₂
                                                             inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                             F : CategoryTheory.Functor T D
                                                             Y : D
                                                             X : T
                                                             ⊢ ∀ {X_1 Y_1 : CategoryTheory.CostructuredArrow (CategoryTheory.CostructuredAr …
                                                           -/
  unitIso := NatIso.ofComponents (fun _ => Iso.refl _) (by simp)
                                                           /-
                                                             🎉 no goals
                                                           -/
                                                             /-
                                                               T : Type u₁
                                                               inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                                               D : Type u₂
                                                               inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                               F : CategoryTheory.Functor T D
                                                               Y : D
                                                               X : T
                                                               ⊢ ∀ {X_1 Y_1 : CategoryTheory.CostructuredArrow ((CategoryTheory.Over.forget X …
                                                             -/
  counitIso := NatIso.ofComponents (fun _ => Iso.refl _) (by aesop_cat)
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- The canonical functor from the costructured arrow category on the diagonal functor
`T ⥤ T × T` to the costructured arrow category on `Under.forget`. -/
@[simps!]
def ofDiagEquivalence.functor (X : T × T) :
    CostructuredArrow (Functor.diag _) X ⥤ CostructuredArrow (Over.forget X.1) X.2 :=
  Functor.toCostructuredArrow
    (Functor.toOver (CostructuredArrow.proj _ X) _
                   /-
                     T : Type u₁
                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                     D : Type u₂
                     inst✝ : CategoryTheory.Category.{v₂, u₂} D
                     X : Prod T T
                     g : CategoryTheory.CostructuredArrow (CategoryTheory.Functor.diag T) X
                     ⊢ Quiver.Hom ((CategoryTheory.CostructuredArrow.proj (CategoryTheory.Functor.d …
                   -/
                   /-
                     🎉 no goals
                   -/
      (fun g => by exact g.hom.1) (fun m => by have := congrArg (·.1) m.w; aesop_cat))
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
    _ _
                                    /-
                                      T : Type u₁
                                      inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                      D : Type u₂
                                      inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                      X : Prod T T
                                      Y✝ Z✝ : CategoryTheory.CostructuredArrow (CategoryTheory.Functor.diag T) X
                                      m : Quiver.Hom Y✝ Z✝
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Over.forget X.1).map …
                                    -/
    (fun f => f.hom.2) (fun m => by have := congrArg (·.2) m.w; aesop_cat)
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- The inverse functor of `ofDiagEquivalence.functor`. -/
@[simps!]
def ofDiagEquivalence.inverse (X : T × T) :
    CostructuredArrow (Over.forget X.1) X.2 ⥤ CostructuredArrow (Functor.diag _) X :=
  Functor.toCostructuredArrow (CostructuredArrow.proj _ _ ⋙ Over.forget _) _ X
                                                /-
                                                  T : Type u₁
                                                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                                  D : Type u₂
                                                  inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                  X : Prod T T
                                                  Y✝ Z✝ : CategoryTheory.CostructuredArrow (CategoryTheory.Over.forget X.1) X.2
                                                  m : Quiver.Hom Y✝ Z✝
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.diag T).map  …
                                                -/
    (fun f => (f.left.hom, f.hom)) (fun m => by have := m.w; aesop_cat)
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- Characterization of the costructured arrow category on the diagonal functor `T ⥤ T × T`. -/
def ofDiagEquivalence (X : T × T) :
    CostructuredArrow (Functor.diag _) X ≌ CostructuredArrow (Over.forget X.1) X.2 where
  functor := ofDiagEquivalence.functor X
  inverse := ofDiagEquivalence.inverse X
                                                           /-
                                                             T : Type u₁
                                                             inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                                             D : Type u₂
                                                             inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                             X : Prod T T
                                                             ⊢ ∀ {X_1 Y : CategoryTheory.CostructuredArrow (CategoryTheory.Functor.diag T)  …
                                                           -/
  unitIso := NatIso.ofComponents (fun _ => Iso.refl _) (by simp)
                                                           /-
                                                             🎉 no goals
                                                           -/
                                                             /-
                                                               T : Type u₁
                                                               inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                                               D : Type u₂
                                                               inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                               X : Prod T T
                                                               ⊢ ∀ {X_1 Y : CategoryTheory.CostructuredArrow (CategoryTheory.Over.forget X.1) …
                                                             -/
  counitIso := NatIso.ofComponents (fun _ => Iso.refl _) (by aesop_cat)
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- A version of `CostructuredArrow.ofDiagEquivalence` with the roles of the first and second
projection swapped. -/
-- noncomputability is only for performance
noncomputable def ofDiagEquivalence' (X : T × T) :
    CostructuredArrow (Functor.diag _) X ≌ CostructuredArrow (Over.forget X.2) X.1 :=
  (ofDiagEquivalence X).trans <|
    (ofCostructuredArrowProjEquivalence (𝟭 T) X.1 X.2).trans <|
    CostructuredArrow.mapNatIso (Over.forget X.2).rightUnitor


/-- The functor used to define the equivalence `ofCommaFstEquivalence`. -/
@[simps]
def ofCommaFstEquivalenceFunctor (c : C) :
    CostructuredArrow (Comma.fst F G) c ⥤ Comma (Over.forget c ⋙ F) G where
  obj X := ⟨Over.mk X.hom, X.left.right, X.left.hom⟩
                                       /-
                                         T : Type u₁
                                         inst✝² : CategoryTheory.Category.{v₁, u₁} T
                                         D : Type u₂
                                         inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                         C : Type u₃
                                         inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                         F : CategoryTheory.Functor C T
                                         G : CategoryTheory.Functor D T
                                         c : C
                                         X✝ Y✝ : CategoryTheory.CostructuredArrow (CategoryTheory.Comma.fst F G) c
                                         f : Quiver.Hom X✝ Y✝
                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp f.left.left ((fun X => { left := Cate …
                                       -/
                                       /-
                                         🎉 no goals
                                       -/
  map f := ⟨Over.homMk f.left.left (by simpa using f.w), f.left.right, by simp⟩
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


/-- The inverse functor used to define the equivalence `ofCommaFstEquivalence`. -/
@[simps!]
def ofCommaFstEquivalenceInverse (c : C) :
    Comma (Over.forget c ⋙ F) G ⥤ CostructuredArrow (Comma.fst F G) c :=
  Functor.toCostructuredArrow (Comma.preLeft (Over.forget c) F G) _ _
                                       /-
                                         T : Type u₁
                                         inst✝² : CategoryTheory.Category.{v₁, u₁} T
                                         D : Type u₂
                                         inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                         C : Type u₃
                                         inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                         F : CategoryTheory.Functor C T
                                         G : CategoryTheory.Functor D T
                                         c : C
                                         Y✝ Z✝ : CategoryTheory.Comma ((CategoryTheory.Over.forget c).comp F) G
                                         x✝ : Quiver.Hom Y✝ Z✝
                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Comma.fst F G).map ( …
                                       -/
    (fun Y => Y.left.hom) (fun _ => by simp)
                                       /-
                                         🎉 no goals
                                       -/


/-- There is a canonical equivalence between the costructured arrow category with codomain `c` on
the functor `Comma.fst F G : Comma F G ⥤ F` and the comma category over
`Over.forget c ⋙ F : Over c ⥤ T` and `G`. -/
@[simps]
def ofCommaFstEquivalence (c : C) :
    CostructuredArrow (Comma.fst F G) c ≌ Comma (Over.forget c ⋙ F) G where
  functor := ofCommaFstEquivalenceFunctor F G c
  inverse := ofCommaFstEquivalenceInverse F G c
             /-
               T : Type u₁
               inst✝² : CategoryTheory.Category.{v₁, u₁} T
               D : Type u₂
               inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
               C : Type u₃
               inst✝ : CategoryTheory.Category.{v₃, u₃} C
               F : CategoryTheory.Functor C T
               G : CategoryTheory.Functor D T
               c : C
               ⊢ ∀ {X Y : CategoryTheory.CostructuredArrow (CategoryTheory.Comma.fst F G) c}  …
             -/
  unitIso := NatIso.ofComponents (fun _ => Iso.refl _)
             /-
               🎉 no goals
             -/
               /-
                 T : Type u₁
                 inst✝² : CategoryTheory.Category.{v₁, u₁} T
                 D : Type u₂
                 inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                 C : Type u₃
                 inst✝ : CategoryTheory.Category.{v₃, u₃} C
                 F : CategoryTheory.Functor C T
                 G : CategoryTheory.Functor D T
                 c : C
                 ⊢ ∀ {X Y : CategoryTheory.Comma ((CategoryTheory.Over.forget c).comp F) G} (f  …
               -/
  counitIso := NatIso.ofComponents (fun _ => Iso.refl _)
               /-
                 🎉 no goals
               -/


/-- The canonical functor by reversing structure arrows. -/
@[simps]
def Over.opToOpUnder : Over (op X) ⥤ (Under X)ᵒᵖ where
  obj Y := ⟨Under.mk Y.hom.unop⟩
                                                /-
                                                  T : Type u₁
                                                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                                  D : Type u₂
                                                  inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                  X : T
                                                  Z Y : CategoryTheory.Over { unop := X }
                                                  f : Quiver.Hom Z Y
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (Opposite.unop ((fun Y => { unop := C …
                                                -/
  map {Z Y} f := ⟨Under.homMk (f.left.unop) (by dsimp; rw [← unop_comp, Over.w])⟩
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- The canonical functor by reversing structure arrows. -/
@[simps]
def Under.opToOverOp : (Under X)ᵒᵖ ⥤ Over (op X) where
  obj Y := Over.mk (Y.unop.hom.op)
                                                  /-
                                                    T : Type u₁
                                                    inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                                    D : Type u₂
                                                    inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                    X : T
                                                    Z Y : Opposite (CategoryTheory.Under X)
                                                    f : Quiver.Hom Z Y
                                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp f.unop.right.op ((fun Y => CategoryTh …
                                                  -/
  map {Z Y} f := Over.homMk f.unop.right.op <| by dsimp; rw [← Under.w f.unop, op_comp]
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- `Over.opToOpUnder` is an equivalence of categories. -/
@[simps]
def Over.opEquivOpUnder : Over (op X) ≌ (Under X)ᵒᵖ where
  functor := Over.opToOpUnder X
  inverse := Under.opToOverOp X
  unitIso := Iso.refl _
  counitIso := Iso.refl _


/-- The canonical functor by reversing structure arrows. -/
@[simps]
def Under.opToOpOver : Under (op X) ⥤ (Over X)ᵒᵖ where
  obj Y := ⟨Over.mk Y.hom.unop⟩
                                                /-
                                                  T : Type u₁
                                                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                                  D : Type u₂
                                                  inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                  X : T
                                                  Z Y : CategoryTheory.Under { unop := X }
                                                  f : Quiver.Hom Z Y
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp f.right.unop (Opposite.unop ((fun Y = …
                                                -/
  map {Z Y} f := ⟨Over.homMk (f.right.unop) (by dsimp; rw [← unop_comp, Under.w])⟩
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- The canonical functor by reversing structure arrows. -/
@[simps]
def Over.opToUnderOp : (Over X)ᵒᵖ ⥤ Under (op X) where
  obj Y := Under.mk (Y.unop.hom.op)
                                                  /-
                                                    T : Type u₁
                                                    inst✝¹ : CategoryTheory.Category.{v₁, u₁} T
                                                    D : Type u₂
                                                    inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                    X : T
                                                    Z Y : Opposite (CategoryTheory.Over X)
                                                    f : Quiver.Hom Z Y
                                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun Y => CategoryTheory.Under.mk (O …
                                                  -/
  map {Z Y} f := Under.homMk f.unop.left.op <| by dsimp; rw [← Over.w f.unop, op_comp]
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- `Under.opToOpOver` is an equivalence of categories. -/
@[simps]
def Under.opEquivOpOver : Under (op X) ≌ (Over X)ᵒᵖ where
  functor := Under.opToOpOver X
  inverse := Over.opToUnderOp X
  unitIso := Iso.refl _
  counitIso := Iso.refl _



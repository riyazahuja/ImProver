/-- Abbreviation that allows writing `CategoryTheory.SingleObj` rather than `Quiver.SingleObj`.
-/
abbrev SingleObj :=
  Quiver.SingleObj


/-- One and `flip (*)` become `id` and `comp` for morphisms of the single object category. -/
instance categoryStruct [One M] [Mul M] : CategoryStruct (SingleObj M) where
  Hom _ _ := M
  comp x y := y * x
  id _ := 1


/-- Monoid laws become category laws for the single object category. -/
instance category : Category (SingleObj M) where
  comp_id := one_mul
  id_comp := mul_one
  assoc x y z := (mul_assoc z y x).symm


theorem id_as_one (x : SingleObj M) : 𝟙 x = 1 :=
  rfl


theorem comp_as_mul {x y z : SingleObj M} (f : x ⟶ y) (g : y ⟶ z) : f ≫ g = g * f :=
  rfl


/-- If `M` is finite and in universe zero, then `SingleObj M` is a `FinCategory`. -/
instance finCategoryOfFintype (M : Type) [Fintype M] [Monoid M] : FinCategory (SingleObj M) where


/-- Groupoid structure on `SingleObj M`.

See <https://stacks.math.columbia.edu/tag/0019>.
-/
instance groupoid : Groupoid (SingleObj G) where
  inv x := x⁻¹
  inv_comp := mul_inv_cancel
  comp_inv := inv_mul_cancel


theorem inv_as_inv {x y : SingleObj G} (f : x ⟶ y) : inv f = f⁻¹ := by
  /-
    G : Type u
    inst✝ : Group G
    x y : CategoryTheory.SingleObj G
    f : Quiver.Hom x y
    ⊢ Eq (CategoryTheory.inv f) (Inv.inv f)
  -/
  apply IsIso.inv_eq_of_hom_inv_id
  /-
    case hom_inv_id
    G : Type u
    inst✝ : Group G
    x y : CategoryTheory.SingleObj G
    f : Quiver.Hom x y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (Inv.inv f)) (CategoryTheory.Catego …
  -/
  rw [comp_as_mul, inv_mul_cancel, id_as_one]
  /-
    🎉 no goals
  -/


/-- Abbreviation that allows writing `CategoryTheory.SingleObj.star` rather than
`Quiver.SingleObj.star`.
-/
abbrev star : SingleObj M :=
  Quiver.SingleObj.star M


/-- The endomorphisms monoid of the only object in `SingleObj M` is equivalent to the original
     monoid M. -/
def toEnd : M ≃* End (SingleObj.star M) :=
  { Equiv.refl M with map_mul' := fun _ _ => rfl }


theorem toEnd_def (x : M) : toEnd M x = x :=
  rfl


/-- There is a 1-1 correspondence between monoid homomorphisms `M → N` and functors between the
    corresponding single-object categories. It means that `SingleObj` is a fully faithful
    functor.

See <https://stacks.math.columbia.edu/tag/001F> --
although we do not characterize when the functor is full or faithful.
-/
def mapHom : (M →* N) ≃ SingleObj M ⥤ SingleObj N where
  toFun f :=
    { obj := id
      map := ⇑f
      map_id := fun _ => f.map_one
      map_comp := fun x y => f.map_mul y x }
  invFun f :=
    { toFun := fun x => f.map ((toEnd M) x)
      map_one' := f.map_id _
      map_mul' := fun x y => f.map_comp y x }
                 /-
                   M G : Type u
                   inst✝² : Monoid M
                   inst✝¹ : Group G
                   N : Type v
                   inst✝ : Monoid N
                   ⊢ Function.LeftInverse (fun f => { toFun := fun x => f.map ((CategoryTheory.Si …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    M G : Type u
                    inst✝² : Monoid M
                    inst✝¹ : Group G
                    N : Type v
                    inst✝ : Monoid N
                    ⊢ Function.RightInverse (fun f => { toFun := fun x => f.map ((CategoryTheory.S …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


theorem mapHom_id : mapHom M M (MonoidHom.id M) = 𝟭 _ :=
  rfl


theorem mapHom_comp (f : M →* N) {P : Type w} [Monoid P] (g : N →* P) :
    mapHom M P (g.comp f) = mapHom M N f ⋙ mapHom N P g :=
  rfl


/-- Given a function `f : C → G` from a category to a group, we get a functor
    `C ⥤ G` sending any morphism `x ⟶ y` to `f y * (f x)⁻¹`. -/
@[simps]
def differenceFunctor (f : C → G) : C ⥤ SingleObj G where
  obj _ := ()
  map {x y} _ := f y * (f x)⁻¹
  map_id := by
    /-
      M G : Type u
      inst✝³ : Monoid M
      inst✝² : Group G
      N : Type v
      inst✝¹ : Monoid N
      C : Type v
      inst✝ : CategoryTheory.Category.{w, v} C
      f : C → G
      ⊢ ∀ (X : C), Eq ({ obj := fun x => Unit.unit, map := fun {x y} x_1 => HMul.hMu …
    -/
    intro
    /-
      M G : Type u
      inst✝³ : Monoid M
      inst✝² : Group G
      N : Type v
      inst✝¹ : Monoid N
      C : Type v
      inst✝ : CategoryTheory.Category.{w, v} C
      f : C → G
      X✝ : C
      ⊢ Eq ({ obj := fun x => Unit.unit, map := fun {x y} x_1 => HMul.hMul (f y) (In …
    -/
    simp only [SingleObj.id_as_one, mul_inv_cancel]
    /-
      🎉 no goals
    -/
  map_comp := by
    /-
      M G : Type u
      inst✝³ : Monoid M
      inst✝² : Group G
      N : Type v
      inst✝¹ : Monoid N
      C : Type v
      inst✝ : CategoryTheory.Category.{w, v} C
      f : C → G
      ⊢ ∀ {X Y Z : C} (f_1 : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ obj := fun  …
    -/
    intros
    /-
      M G : Type u
      inst✝³ : Monoid M
      inst✝² : Group G
      N : Type v
      inst✝¹ : Monoid N
      C : Type v
      inst✝ : CategoryTheory.Category.{w, v} C
      f : C → G
      X✝ Y✝ Z✝ : C
      f✝ : Quiver.Hom X✝ Y✝
      g✝ : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun x => Unit.unit, map := fun {x y} x_1 => HMul.hMul (f y) (In …
    -/
    dsimp
    /-
      M G : Type u
      inst✝³ : Monoid M
      inst✝² : Group G
      N : Type v
      inst✝¹ : Monoid N
      C : Type v
      inst✝ : CategoryTheory.Category.{w, v} C
      f : C → G
      X✝ Y✝ Z✝ : C
      f✝ : Quiver.Hom X✝ Y✝
      g✝ : Quiver.Hom Y✝ Z✝
      ⊢ Eq (HMul.hMul (f Z✝) (Inv.inv (f X✝))) (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [SingleObj.comp_as_mul, ← mul_assoc, mul_left_inj, mul_assoc, inv_mul_cancel, mul_one]
    /-
      🎉 no goals
    -/


/-- A monoid homomorphism `f: M → End X` into the endomorphisms of an object `X` of a category `C`
induces a functor `SingleObj M ⥤ C`. -/
@[simps]
def functor {X : C} (f : M →* End X) : SingleObj M ⥤ C where
  obj _ := X
  map a := f a
  map_id _ := MonoidHom.map_one f
  map_comp a b := MonoidHom.map_mul f b a


/-- Construct a natural transformation between functors `SingleObj M ⥤ C` by
giving a compatible morphism `SingleObj.star M`. -/
@[simps]
def natTrans {F G : SingleObj M ⥤ C} (u : F.obj (SingleObj.star M) ⟶ G.obj (SingleObj.star M))
    (h : ∀ a : M, F.map a ≫ u = u ≫ G.map a) : F ⟶ G where
  app _ := u
  naturality _ _ a := h a


/-- Reinterpret a monoid homomorphism `f : M → N` as a functor `(single_obj M) ⥤ (single_obj N)`.
See also `CategoryTheory.SingleObj.mapHom` for an equivalence between these types. -/
abbrev toFunctor (f : M →* N) : SingleObj M ⥤ SingleObj N :=
  SingleObj.mapHom M N f


@[simp]
theorem comp_toFunctor (f : M →* N) {P : Type w} [Monoid P] (g : N →* P) :
    (g.comp f).toFunctor = f.toFunctor ⋙ g.toFunctor :=
  rfl


@[simp]
theorem id_toFunctor : (id M).toFunctor = 𝟭 _ :=
  rfl


/-- Reinterpret a monoid isomorphism `f : M ≃* N` as an equivalence `SingleObj M ≌ SingleObj N`. -/
@[simps!]
def toSingleObjEquiv (e : M ≃* N) : SingleObj M ≌ SingleObj N where
  functor := e.toMonoidHom.toFunctor
  inverse := e.symm.toMonoidHom.toFunctor
  unitIso := eqToIso (by
    /-
      M : Type u
      N : Type v
      inst✝¹ : Monoid M
      inst✝ : Monoid N
      e : MulEquiv M N
      ⊢ Eq (CategoryTheory.Functor.id (CategoryTheory.SingleObj M)) (e.toMonoidHom.t …
    -/
    rw [← MonoidHom.comp_toFunctor, ← MonoidHom.id_toFunctor]
    /-
      M : Type u
      N : Type v
      inst✝¹ : Monoid M
      inst✝ : Monoid N
      e : MulEquiv M N
      ⊢ Eq (MonoidHom.id M).toFunctor (e.symm.toMonoidHom.comp e.toMonoidHom).toFunc …
    -/
    congr 1
    /-
      case e_f
      M : Type u
      N : Type v
      inst✝¹ : Monoid M
      inst✝ : Monoid N
      e : MulEquiv M N
      ⊢ Eq (MonoidHom.id M) (e.symm.toMonoidHom.comp e.toMonoidHom)
    -/
    aesop_cat)
    /-
      🎉 no goals
    -/
  counitIso := eqToIso (by
    /-
      M : Type u
      N : Type v
      inst✝¹ : Monoid M
      inst✝ : Monoid N
      e : MulEquiv M N
      ⊢ Eq (e.symm.toMonoidHom.toFunctor.comp e.toMonoidHom.toFunctor) (CategoryTheo …
    -/
    rw [← MonoidHom.comp_toFunctor, ← MonoidHom.id_toFunctor]
    /-
      M : Type u
      N : Type v
      inst✝¹ : Monoid M
      inst✝ : Monoid N
      e : MulEquiv M N
      ⊢ Eq (e.toMonoidHom.comp e.symm.toMonoidHom).toFunctor (MonoidHom.id N).toFunc …
    -/
    congr 1
    /-
      case e_f
      M : Type u
      N : Type v
      inst✝¹ : Monoid M
      inst✝ : Monoid N
      e : MulEquiv M N
      ⊢ Eq (e.toMonoidHom.comp e.symm.toMonoidHom) (MonoidHom.id N)
    -/
    aesop_cat)
    /-
      🎉 no goals
    -/


/-- The units in a monoid are (multiplicatively) equivalent to
the automorphisms of `star` when we think of the monoid as a single-object category. -/
def toAut : Mˣ ≃* Aut (SingleObj.star M) :=
  MulEquiv.trans (Units.mapEquiv (SingleObj.toEnd M))
    (Aut.unitsEndEquivAut (SingleObj.star M))


@[simp]
theorem toAut_hom (x : Mˣ) : (toAut M x).hom = SingleObj.toEnd M x :=
  rfl


@[simp]
theorem toAut_inv (x : Mˣ) : (toAut M x).inv = SingleObj.toEnd M (x⁻¹ : Mˣ) :=
  rfl


/-- The fully faithful functor from `MonCat` to `Cat`. -/
def toCat : MonCat ⥤ Cat where
  obj x := Cat.of (SingleObj x)
  map {x y} f := SingleObj.mapHom x y f


instance toCat_full : toCat.Full where
  map_surjective := (SingleObj.mapHom _ _).surjective


instance toCat_faithful : toCat.Faithful where
                        /-
                          X✝ Y✝ : MonCat
                          a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
                          h : Eq (MonCat.toCat.map a₁✝) (MonCat.toCat.map a₂✝)
                          ⊢ Eq a₁✝ a₂✝
                        -/
  map_injective h := by rwa [toCat, (SingleObj.mapHom _ _).apply_eq_iff_eq] at h
                        /-
                          🎉 no goals
                        -/



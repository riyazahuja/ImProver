/-- An `Action V G` represents a bundled action of
the monoid `G` on an object of some category `V`.

As an example, when `V = ModuleCat R`, this is an `R`-linear representation of `G`,
while when `V = Type` this is a `G`-action.
-/
structure Action (G : MonCat.{u}) where
  V : V
  ρ : G ⟶ MonCat.of (End V)


@[simp 1100]
                                                                      /-
                                                                        V : Type (u + 1)
                                                                        inst✝ : CategoryTheory.LargeCategory V
                                                                        G : MonCat
                                                                        A : Action V G
                                                                        ⊢ Eq (A.ρ 1) (CategoryTheory.CategoryStruct.id A.V)
                                                                      -/
theorem ρ_one {G : MonCat.{u}} (A : Action V G) : A.ρ 1 = 𝟙 A.V := by rw [MonoidHom.map_one]; rfl
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/


/-- When a group acts, we can lift the action to the group of automorphisms. -/
@[simps]
def ρAut {G : Grp.{u}} (A : Action V (MonCat.of G)) : G ⟶ Grp.of (Aut A.V) where
  toFun g :=
    { hom := A.ρ g
      inv := A.ρ (g⁻¹ : G)
                                                             /-
                                                               V : Type (u + 1)
                                                               inst✝ : CategoryTheory.LargeCategory V
                                                               G : Grp
                                                               A : Action V (MonCat.of ↑G)
                                                               g : ↑G
                                                               ⊢ Eq (A.ρ (HMul.hMul (Inv.inv g) g)) (CategoryTheory.CategoryStruct.id A.V)
                                                             -/
      hom_inv_id := (A.ρ.map_mul (g⁻¹ : G) g).symm.trans (by rw [inv_mul_cancel, ρ_one])
                                                             /-
                                                               🎉 no goals
                                                             -/
                                                             /-
                                                               V : Type (u + 1)
                                                               inst✝ : CategoryTheory.LargeCategory V
                                                               G : Grp
                                                               A : Action V (MonCat.of ↑G)
                                                               g : ↑G
                                                               ⊢ Eq (A.ρ (HMul.hMul g (Inv.inv g))) (CategoryTheory.CategoryStruct.id A.V)
                                                             -/
      inv_hom_id := (A.ρ.map_mul g (g⁻¹ : G)).symm.trans (by rw [mul_inv_cancel, ρ_one]) }
                                                             /-
                                                               🎉 no goals
                                                             -/
  map_one' := Aut.ext A.ρ.map_one
  map_mul' x y := Aut.ext (A.ρ.map_mul x y)

-- These lemmas have always been bad (https://github.com/leanprover-community/mathlib4/issues/7657),
-- but https://github.com/leanprover/lean4/pull/2644 made `simp` start noticing
-- It would be worth fixing these, as `ρAut_apply_inv` is used in `erw` later.

instance inhabited' : Inhabited (Action (Type u) G) :=
  ⟨⟨PUnit, 1⟩⟩


/-- The trivial representation of a group. -/
def trivial : Action AddCommGrp G where
  V := AddCommGrp.of PUnit
  ρ := 1


instance : Inhabited (Action AddCommGrp G) :=
  ⟨trivial G⟩


/-- A homomorphism of `Action V G`s is a morphism between the underlying objects,
commuting with the action of `G`.
-/
@[ext]
structure Hom (M N : Action V G) where
  hom : M.V ⟶ N.V
  comm : ∀ g : G, M.ρ g ≫ hom = hom ≫ N.ρ g := by aesop_cat


attribute [reassoc] comm

/-- The identity morphism on an `Action V G`. -/
@[simps]
def id (M : Action V G) : Action.Hom M M where hom := 𝟙 M.V


instance (M : Action V G) : Inhabited (Action.Hom M M) :=
  ⟨id M⟩


/-- The composition of two `Action V G` homomorphisms is the composition of the underlying maps.
-/
@[simps]
def comp {M N K : Action V G} (p : Action.Hom M N) (q : Action.Hom N K) : Action.Hom M K where
  hom := p.hom ≫ q.hom


instance : Category (Action V G) where
  Hom M N := Hom M N
  id M := Hom.id M
  comp f g := Hom.comp f g


@[ext]
lemma hom_ext {M N : Action V G} (φ₁ φ₂ : M ⟶ N) (h : φ₁.hom = φ₂.hom) : φ₁ = φ₂ :=
  Hom.ext h


@[simp]
theorem id_hom (M : Action V G) : (𝟙 M : Hom M M).hom = 𝟙 M.V :=
  rfl


@[simp]
theorem comp_hom {M N K : Action V G} (f : M ⟶ N) (g : N ⟶ K) :
    (f ≫ g : Hom M K).hom = f.hom ≫ g.hom :=
  rfl


@[simp]
theorem hom_inv_hom {M N : Action V G} (f : M ≅ N) :
    f.hom.hom ≫ f.inv.hom = 𝟙 M.V := by
  /-
    V : Type (u + 1)
    inst✝ : CategoryTheory.LargeCategory V
    G : MonCat
    M N : Action V G
    f : CategoryTheory.Iso M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom.hom f.inv.hom) (CategoryTheory. …
  -/
  rw [← comp_hom, Iso.hom_inv_id, id_hom]
  /-
    🎉 no goals
  -/


@[simp]
theorem inv_hom_hom {M N : Action V G} (f : M ≅ N) :
    f.inv.hom ≫ f.hom.hom = 𝟙 N.V := by
  /-
    V : Type (u + 1)
    inst✝ : CategoryTheory.LargeCategory V
    G : MonCat
    M N : Action V G
    f : CategoryTheory.Iso M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f.inv.hom f.hom.hom) (CategoryTheory. …
  -/
  rw [← comp_hom, Iso.inv_hom_id, id_hom]
  /-
    🎉 no goals
  -/


/-- Construct an isomorphism of `G` actions/representations
from an isomorphism of the underlying objects,
where the forward direction commutes with the group action. -/
@[simps]
def mkIso {M N : Action V G} (f : M.V ≅ N.V)
    (comm : ∀ g : G, M.ρ g ≫ f.hom = f.hom ≫ N.ρ g := by aesop_cat) : M ≅ N where
  hom :=
    { hom := f.hom
      comm := comm }
  inv :=
    { hom := f.inv
                          /-
                            V : Type (u + 1)
                            inst✝ : CategoryTheory.LargeCategory V
                            G : MonCat
                            M N : Action V G
                            f : CategoryTheory.Iso M.V N.V
                            comm : autoParam (∀ (g : ↑G), Eq (CategoryTheory.CategoryStruct.comp (M.ρ g) f …
                            g : ↑G
                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (N.ρ g) f.inv) (CategoryTheory.Catego …
                          -/
      comm := fun g => by have w := comm g =≫ f.inv; simp at w; simp [w] }
                                                                /-
                                                                  🎉 no goals
                                                                -/


instance (priority := 100) isIso_of_hom_isIso {M N : Action V G} (f : M ⟶ N) [IsIso f.hom] :
    IsIso f := (mkIso (asIso f.hom) f.comm).isIso_hom


instance isIso_hom_mk {M N : Action V G} (f : M.V ⟶ N.V) [IsIso f] (w) :
    @IsIso _ _ M N (Hom.mk f w) :=
  (mkIso (asIso f) w).isIso_hom


instance {M N : Action V G} (f : M ≅ N) : IsIso f.hom.hom where
                        /-
                          V : Type (u + 1)
                          inst✝ : CategoryTheory.LargeCategory V
                          G : MonCat
                          M N : Action V G
                          f : CategoryTheory.Iso M N
                          ⊢ And (Eq (CategoryTheory.CategoryStruct.comp f.hom.hom f.inv.hom) (CategoryTh …
                        -/
  out := ⟨f.inv.hom, by simp⟩
                        /-
                          🎉 no goals
                        -/


instance {M N : Action V G} (f : M ≅ N) : IsIso f.inv.hom where
                        /-
                          V : Type (u + 1)
                          inst✝ : CategoryTheory.LargeCategory V
                          G : MonCat
                          M N : Action V G
                          f : CategoryTheory.Iso M N
                          ⊢ And (Eq (CategoryTheory.CategoryStruct.comp f.inv.hom f.hom.hom) (CategoryTh …
                        -/
  out := ⟨f.hom.hom, by simp⟩
                        /-
                          🎉 no goals
                        -/


/-- Auxiliary definition for `functorCategoryEquivalence`. -/
@[simps]
def functor : Action V G ⥤ SingleObj G ⥤ V where
  obj M :=
    { obj := fun _ => M.V
      map := fun g => M.ρ g
      map_id := fun _ => M.ρ.map_one
      map_comp := fun g h => M.ρ.map_mul h g }
  map f :=
    { app := fun _ => f.hom
      naturality := fun _ _ g => f.comm g }


/-- Auxiliary definition for `functorCategoryEquivalence`. -/
@[simps]
def inverse : (SingleObj G ⥤ V) ⥤ Action V G where
  obj F :=
    { V := F.obj PUnit.unit
      ρ :=
        { toFun := fun g => F.map g
          map_one' := F.map_id PUnit.unit
          map_mul' := fun g h => F.map_comp h g } }
  map f :=
    { hom := f.app PUnit.unit
      comm := fun g => f.naturality g }


/-- Auxiliary definition for `functorCategoryEquivalence`. -/
@[simps!]
def unitIso : 𝟭 (Action V G) ≅ functor ⋙ inverse :=
                               /-
                                 V : Type (u + 1)
                                 inst✝ : CategoryTheory.LargeCategory V
                                 G : MonCat
                                 M : Action V G
                                 ⊢ ∀ (g : ↑G), Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun M => mkIso (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `functorCategoryEquivalence`. -/
@[simps!]
def counitIso : inverse ⋙ functor ≅ 𝟭 (SingleObj G ⥤ V) :=
                               /-
                                 V : Type (u + 1)
                                 inst✝ : CategoryTheory.LargeCategory V
                                 G : MonCat
                                 M : CategoryTheory.Functor (CategoryTheory.SingleObj ↑G) V
                                 ⊢ ∀ {X Y : CategoryTheory.SingleObj ↑G} (f : Quiver.Hom X Y), Eq (CategoryTheo …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun M => NatIso.ofComponents fun _ => Iso.refl _
  /-
    🎉 no goals
  -/


/-- The category of actions of `G` in the category `V`
is equivalent to the functor category `singleObj G ⥤ V`.
-/
@[simps]
def functorCategoryEquivalence : Action V G ≌ SingleObj G ⥤ V where
  functor := functor
  inverse := inverse
  unitIso := unitIso
  counitIso := counitIso


instance : (FunctorCategoryEquivalence.functor (V := V) (G := G)).IsEquivalence :=
  (functorCategoryEquivalence V G).isEquivalence_functor


instance : (FunctorCategoryEquivalence.inverse (V := V) (G := G)).IsEquivalence :=
  (functorCategoryEquivalence V G).isEquivalence_inverse

/-
porting note: these two lemmas are redundant with the projections created by the @[simps]
attribute above

theorem functorCategoryEquivalence.functor_def :
    (functorCategoryEquivalence V G).functor = FunctorCategoryEquivalence.functor :=
  rfl

theorem functorCategoryEquivalence.inverse_def :
    (functorCategoryEquivalence V G).inverse = FunctorCategoryEquivalence.inverse :=
  rfl
-/


/-- (implementation) The forgetful functor from bundled actions to the underlying objects.

Use the `CategoryTheory.forget` API provided by the `ConcreteCategory` instance below,
rather than using this directly.
-/
@[simps]
def forget : Action V G ⥤ V where
  obj M := M.V
  map f := f.hom


instance : (forget V G).Faithful where map_injective w := Hom.ext w


instance [ConcreteCategory V] : ConcreteCategory (Action V G) where
  forget := forget V G ⋙ ConcreteCategory.forget


instance hasForgetToV [ConcreteCategory V] : HasForget₂ (Action V G) V where forget₂ := forget V G


/-- The forgetful functor is intertwined by `functorCategoryEquivalence` with
evaluation at `PUnit.star`. -/
def functorCategoryEquivalenceCompEvaluation :
    (functorCategoryEquivalence V G).functor ⋙ (evaluation _ _).obj PUnit.unit ≅ forget V G :=
  Iso.refl _


noncomputable instance preservesLimits_forget [HasLimits V] :
    PreservesLimits (forget V G) :=
  Limits.preservesLimits_of_natIso (Action.functorCategoryEquivalenceCompEvaluation V G)


noncomputable instance preservesColimits_forget [HasColimits V] :
    PreservesColimits (forget V G) :=
  preservesColimits_of_natIso (Action.functorCategoryEquivalenceCompEvaluation V G)

-- TODO construct categorical images?

theorem Iso.conj_ρ {M N : Action V G} (f : M ≅ N) (g : G) :
    N.ρ g = ((forget V G).mapIso f).conj (M.ρ g) := by
      /-
        V : Type (u + 1)
        inst✝ : CategoryTheory.LargeCategory V
        G : MonCat
        M N : Action V G
        f : CategoryTheory.Iso M N
        g : ↑G
        ⊢ Eq (N.ρ g) (((Action.forget V G).mapIso f).conj (M.ρ g))
      -/
      rw [Iso.conj_apply, Iso.eq_inv_comp]; simp [f.hom.comm]
                                            /-
                                              🎉 no goals
                                            -/


/-- Actions/representations of the trivial group are just objects in the ambient category. -/
def actionPunitEquivalence : Action V (MonCat.of PUnit) ≌ V where
  functor := forget V _
  inverse :=
    { obj := fun X => ⟨X, 1⟩
                                       /-
                                         V : Type (u + 1)
                                         inst✝ : CategoryTheory.LargeCategory V
                                         G : MonCat
                                         X✝ Y✝ : V
                                         f : Quiver.Hom X✝ Y✝
                                         x✝ : ↑(MonCat.of PUnit.{u + 1})
                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun X => { V := X, ρ := 1 }) X✝).ρ …
                                       -/
      map := fun f => ⟨f, fun ⟨⟩ => by simp⟩ }
                                       /-
                                         🎉 no goals
                                       -/
  unitIso :=
    /-
      V : Type (u + 1)
      inst✝ : CategoryTheory.LargeCategory V
      G : MonCat
      ⊢ ∀ {X Y : Action V (MonCat.of PUnit.{u + 1})} (f : Quiver.Hom X Y), Eq (Categ …
    -/
    NatIso.ofComponents fun X => mkIso (Iso.refl _) fun ⟨⟩ => by
    /-
      🎉 no goals
    -/
      /-
        V : Type (u + 1)
        inst✝ : CategoryTheory.LargeCategory V
        G : MonCat
        X : Action V (MonCat.of PUnit.{u + 1})
        x✝ : ↑(MonCat.of PUnit.{u + 1})
        ⊢ Eq (((CategoryTheory.Functor.id (Action V (MonCat.of PUnit.{u + 1}))).obj X) …
      -/
      simp only [MonCat.oneHom_apply, MonCat.one_of, End.one_def, id_eq, Functor.comp_obj,
      /-
        🎉 no goals
      -/
        forget_obj, Iso.refl_hom, Category.comp_id]
      exact ρ_one X
               /-
                 V : Type (u + 1)
                 inst✝ : CategoryTheory.LargeCategory V
                 G : MonCat
                 ⊢ ∀ {X Y : V} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp (({ …
               -/
  counitIso := NatIso.ofComponents fun _ => Iso.refl _
               /-
                 🎉 no goals
               -/


/-- The "restriction" functor along a monoid homomorphism `f : G ⟶ H`,
taking actions of `H` to actions of `G`.

(This makes sense for any homomorphism, but the name is natural when `f` is a monomorphism.)
-/
@[simps]
def res {G H : MonCat} (f : G ⟶ H) : Action V H ⥤ Action V G where
  obj M :=
    { V := M.V
      ρ := f ≫ M.ρ }
  map p :=
    { hom := p.hom
      comm := fun g => p.comm (f g) }


/-- The natural isomorphism from restriction along the identity homomorphism to
the identity functor on `Action V G`.
-/
@[simps!]
def resId {G : MonCat} : res V (𝟙 G) ≅ 𝟭 (Action V G) :=
                               /-
                                 V : Type (u + 1)
                                 inst✝ : CategoryTheory.LargeCategory V
                                 G✝ G : MonCat
                                 M : Action V G
                                 ⊢ ∀ (g : ↑G), Eq (CategoryTheory.CategoryStruct.comp (((Action.res V (Category …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun M => mkIso (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- The natural isomorphism from the composition of restrictions along homomorphisms
to the restriction along the composition of homomorphism.
-/
@[simps!]
def resComp {G H K : MonCat} (f : G ⟶ H) (g : H ⟶ K) : res V g ⋙ res V f ≅ res V (f ≫ g) :=
                               /-
                                 V : Type (u + 1)
                                 inst✝ : CategoryTheory.LargeCategory V
                                 G✝ G H K : MonCat
                                 f : Quiver.Hom G H
                                 g : Quiver.Hom H K
                                 M : Action V K
                                 ⊢ ∀ (g_1 : ↑G), Eq (CategoryTheory.CategoryStruct.comp ((((Action.res V g).com …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun M => mkIso (Iso.refl _)
  /-
    🎉 no goals
  -/

-- TODO promote `res` to a pseudofunctor from
-- the locally discrete bicategory constructed from `Monᵒᵖ` to `Cat`, sending `G` to `Action V G`.


/-- A functor between categories induces a functor between
the categories of `G`-actions within those categories. -/
@[simps]
def mapAction (F : V ⥤ W) (G : MonCat.{u}) : Action V G ⥤ Action W G where
  obj M :=
    { V := F.obj M.V
      ρ :=
        { toFun := fun g => F.map (M.ρ g)
                         /-
                           V : Type (u + 1)
                           inst✝¹ : CategoryTheory.LargeCategory V
                           W : Type (u + 1)
                           inst✝ : CategoryTheory.LargeCategory W
                           F : CategoryTheory.Functor V W
                           G : MonCat
                           M : Action V G
                           ⊢ Eq ((fun g => F.map (M.ρ g)) 1) 1
                         -/
          map_one' := by simp only [End.one_def, Action.ρ_one, F.map_id, MonCat.one_of]
                         /-
                           🎉 no goals
                         -/
          map_mul' := fun g h => by
            /-
              V : Type (u + 1)
              inst✝¹ : CategoryTheory.LargeCategory V
              W : Type (u + 1)
              inst✝ : CategoryTheory.LargeCategory W
              F : CategoryTheory.Functor V W
              G : MonCat
              M : Action V G
              g h : ↑G
              ⊢ Eq ({ toFun := fun g => F.map (M.ρ g), map_one' := ⋯ }.toFun (HMul.hMul g h) …
            -/
            dsimp
            /-
              V : Type (u + 1)
              inst✝¹ : CategoryTheory.LargeCategory V
              W : Type (u + 1)
              inst✝ : CategoryTheory.LargeCategory W
              F : CategoryTheory.Functor V W
              G : MonCat
              M : Action V G
              g h : ↑G
              ⊢ Eq (F.map (M.ρ (HMul.hMul g h))) (HMul.hMul (F.map (M.ρ g)) (F.map (M.ρ h)))
            -/
            rw [map_mul, MonCat.mul_of, End.mul_def, End.mul_def, F.map_comp] } }
            /-
              🎉 no goals
            -/
  map f :=
    { hom := F.map f.hom
                          /-
                            V : Type (u + 1)
                            inst✝¹ : CategoryTheory.LargeCategory V
                            W : Type (u + 1)
                            inst✝ : CategoryTheory.LargeCategory W
                            F : CategoryTheory.Functor V W
                            G : MonCat
                            X✝ Y✝ : Action V G
                            f : Quiver.Hom X✝ Y✝
                            g : ↑G
                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun M => { V := F.obj M.V, ρ := {  …
                          -/
      comm := fun g => by dsimp; rw [← F.map_comp, f.comm, F.map_comp] }
                                 /-
                                   🎉 no goals
                                 -/
                 /-
                   V : Type (u + 1)
                   inst✝¹ : CategoryTheory.LargeCategory V
                   W : Type (u + 1)
                   inst✝ : CategoryTheory.LargeCategory W
                   F : CategoryTheory.Functor V W
                   G : MonCat
                   M : Action V G
                   ⊢ Eq ({ obj := fun M => { V := F.obj M.V, ρ := { toFun := fun g => F.map (M.ρ  …
                 -/
  map_id M := by ext; simp only [Action.id_hom, F.map_id]
                      /-
                        🎉 no goals
                      -/
                     /-
                       V : Type (u + 1)
                       inst✝¹ : CategoryTheory.LargeCategory V
                       W : Type (u + 1)
                       inst✝ : CategoryTheory.LargeCategory W
                       F : CategoryTheory.Functor V W
                       G : MonCat
                       X✝ Y✝ Z✝ : Action V G
                       f : Quiver.Hom X✝ Y✝
                       g : Quiver.Hom Y✝ Z✝
                       ⊢ Eq ({ obj := fun M => { V := F.obj M.V, ρ := { toFun := fun g => F.map (M.ρ  …
                     -/
  map_comp f g := by ext; simp only [Action.comp_hom, F.map_comp]
                          /-
                            🎉 no goals
                          -/



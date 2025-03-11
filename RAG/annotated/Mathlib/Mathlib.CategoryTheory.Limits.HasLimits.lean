/-- `LimitCone F` contains a cone over `F` together with the information that it is a limit. -/
-- @[nolint has_nonempty_instance] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed; linter not ported yet
structure LimitCone (F : J ⥤ C) where
  /-- The cone itself -/
  cone : Cone F
  /-- The proof that is the limit cone -/
  isLimit : IsLimit cone


/-- `HasLimit F` represents the mere existence of a limit for `F`. -/
class HasLimit (F : J ⥤ C) : Prop where mk' ::
  /-- There is some limit cone for `F` -/
  exists_limit : Nonempty (LimitCone F)


theorem HasLimit.mk {F : J ⥤ C} (d : LimitCone F) : HasLimit F :=
  ⟨Nonempty.intro d⟩


/-- Use the axiom of choice to extract explicit `LimitCone F` from `HasLimit F`. -/
def getLimitCone (F : J ⥤ C) [HasLimit F] : LimitCone F :=
  Classical.choice <| HasLimit.exists_limit


/-- `C` has limits of shape `J` if there exists a limit for every functor `F : J ⥤ C`. -/
class HasLimitsOfShape : Prop where
  /-- All functors `F : J ⥤ C` from `J` have limits -/
  has_limit : ∀ F : J ⥤ C, HasLimit F := by infer_instance


/-- `C` has all limits of size `v₁ u₁` (`HasLimitsOfSize.{v₁ u₁} C`)
if it has limits of every shape `J : Type u₁` with `[Category.{v₁} J]`.
-/
@[pp_with_univ]
class HasLimitsOfSize (C : Type u) [Category.{v} C] : Prop where
  /-- All functors `F : J ⥤ C` from all small `J` have limits -/
  has_limits_of_shape : ∀ (J : Type u₁) [Category.{v₁} J], HasLimitsOfShape J C := by
    infer_instance


/-- `C` has all (small) limits if it has limits of every shape that is as big as its hom-sets. -/
abbrev HasLimits (C : Type u) [Category.{v} C] : Prop :=
  HasLimitsOfSize.{v, v} C


theorem HasLimits.has_limits_of_shape {C : Type u} [Category.{v} C] [HasLimits C] (J : Type v)
    [Category.{v} J] : HasLimitsOfShape J C :=
  HasLimitsOfSize.has_limits_of_shape J


instance (priority := 100) hasLimitOfHasLimitsOfShape {J : Type u₁} [Category.{v₁} J]
    [HasLimitsOfShape J C] (F : J ⥤ C) : HasLimit F :=
  HasLimitsOfShape.has_limit F

-- see Note [lower instance priority]

instance (priority := 100) hasLimitsOfShapeOfHasLimits {J : Type u₁} [Category.{v₁} J]
    [HasLimitsOfSize.{v₁, u₁} C] : HasLimitsOfShape J C :=
  HasLimitsOfSize.has_limits_of_shape J

-- Interface to the `HasLimit` class.

/-- An arbitrary choice of limit cone for a functor. -/
def limit.cone (F : J ⥤ C) [HasLimit F] : Cone F :=
  (getLimitCone F).cone


/-- An arbitrary choice of limit object of a functor. -/
def limit (F : J ⥤ C) [HasLimit F] :=
  (limit.cone F).pt


/-- The projection from the limit object to a value of the functor. -/
def limit.π (F : J ⥤ C) [HasLimit F] (j : J) : limit F ⟶ F.obj j :=
  (limit.cone F).π.app j


@[reassoc]
theorem limit.π_comp_eqToHom (F : J ⥤ C) [HasLimit F] {j j' : J} (hj : j = j') :
                              /-
                                J : Type u₁
                                inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                K : Type u₂
                                inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                C : Type u
                                inst✝¹ : CategoryTheory.Category.{v, u} C
                                F✝ F : CategoryTheory.Functor J C
                                inst✝ : CategoryTheory.Limits.HasLimit F
                                j j' : J
                                hj : Eq j j'
                                ⊢ Eq (F.obj j) (F.obj j')
                              -/
    limit.π F j ≫ eqToHom (by subst hj; rfl) = limit.π F j' := by
                                        /-
                                          🎉 no goals
                                        -/
  /-
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.HasLimit F
    j j' : J
    hj : Eq j j'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π F j) ( …
  -/
  subst hj
  /-
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.HasLimit F
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π F j) ( …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem limit.cone_x {F : J ⥤ C} [HasLimit F] : (limit.cone F).pt = limit F :=
  rfl


@[simp]
theorem limit.cone_π {F : J ⥤ C} [HasLimit F] : (limit.cone F).π.app = limit.π _ :=
  rfl


@[reassoc (attr := simp)]
theorem limit.w (F : J ⥤ C) [HasLimit F] {j j' : J} (f : j ⟶ j') :
    limit.π F j ≫ F.map f = limit.π F j' :=
  (limit.cone F).w f


/-- Evidence that the arbitrary choice of cone provided by `limit.cone F` is a limit cone. -/
def limit.isLimit (F : J ⥤ C) [HasLimit F] : IsLimit (limit.cone F) :=
  (getLimitCone F).isLimit


/-- The morphism from the cone point of any other cone to the limit object. -/
def limit.lift (F : J ⥤ C) [HasLimit F] (c : Cone F) : c.pt ⟶ limit F :=
  (limit.isLimit F).lift c


@[simp]
theorem limit.isLimit_lift {F : J ⥤ C} [HasLimit F] (c : Cone F) :
    (limit.isLimit F).lift c = limit.lift F c :=
  rfl


@[reassoc (attr := simp)]
theorem limit.lift_π {F : J ⥤ C} [HasLimit F] (c : Cone F) (j : J) :
    limit.lift F c ≫ limit.π F j = c.π.app j :=
  IsLimit.fac _ c j


/-- Functoriality of limits.

Usually this morphism should be accessed through `lim.map`,
but may be needed separately when you have specified limits for the source and target functors,
but not necessarily for all functors of shape `J`.
-/
def limMap {F G : J ⥤ C} [HasLimit F] [HasLimit G] (α : F ⟶ G) : limit F ⟶ limit G :=
  IsLimit.map _ (limit.isLimit G) α


@[reassoc (attr := simp)]
theorem limMap_π {F G : J ⥤ C} [HasLimit F] [HasLimit G] (α : F ⟶ G) (j : J) :
    limMap α ≫ limit.π G j = limit.π F j ≫ α.app j :=
  limit.lift_π _ j


/-- The cone morphism from any cone to the arbitrary choice of limit cone. -/
def limit.coneMorphism {F : J ⥤ C} [HasLimit F] (c : Cone F) : c ⟶ limit.cone F :=
  (limit.isLimit F).liftConeMorphism c


@[simp]
theorem limit.coneMorphism_hom {F : J ⥤ C} [HasLimit F] (c : Cone F) :
    (limit.coneMorphism c).hom = limit.lift F c :=
  rfl


theorem limit.coneMorphism_π {F : J ⥤ C} [HasLimit F] (c : Cone F) (j : J) :
                                                               /-
                                                                 J : Type u₁
                                                                 inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                                                 C : Type u
                                                                 inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                 F : CategoryTheory.Functor J C
                                                                 inst✝ : CategoryTheory.Limits.HasLimit F
                                                                 c : CategoryTheory.Limits.Cone F
                                                                 j : J
                                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.coneMorp …
                                                               -/
    (limit.coneMorphism c).hom ≫ limit.π F j = c.π.app j := by simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[reassoc (attr := simp)]
theorem limit.conePointUniqueUpToIso_hom_comp {F : J ⥤ C} [HasLimit F] {c : Cone F} (hc : IsLimit c)
    (j : J) : (IsLimit.conePointUniqueUpToIso hc (limit.isLimit _)).hom ≫ limit.π F j = c.π.app j :=
  IsLimit.conePointUniqueUpToIso_hom_comp _ _ _


@[reassoc (attr := simp)]
theorem limit.conePointUniqueUpToIso_inv_comp {F : J ⥤ C} [HasLimit F] {c : Cone F} (hc : IsLimit c)
    (j : J) : (IsLimit.conePointUniqueUpToIso (limit.isLimit _) hc).inv ≫ limit.π F j = c.π.app j :=
  IsLimit.conePointUniqueUpToIso_inv_comp _ _ _


theorem limit.existsUnique {F : J ⥤ C} [HasLimit F] (t : Cone F) :
    ∃! l : t.pt ⟶ limit F, ∀ j, l ≫ limit.π F j = t.π.app j :=
  (limit.isLimit F).existsUnique _


/-- Given any other limit cone for `F`, the chosen `limit F` is isomorphic to the cone point.
-/
def limit.isoLimitCone {F : J ⥤ C} [HasLimit F] (t : LimitCone F) : limit F ≅ t.cone.pt :=
  IsLimit.conePointUniqueUpToIso (limit.isLimit F) t.isLimit


@[reassoc (attr := simp)]
theorem limit.isoLimitCone_hom_π {F : J ⥤ C} [HasLimit F] (t : LimitCone F) (j : J) :
    (limit.isoLimitCone t).hom ≫ t.cone.π.app j = limit.π F j := by
  /-
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.HasLimit F
    t : CategoryTheory.Limits.LimitCone F
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.isoLimit …
  -/
  dsimp [limit.isoLimitCone, IsLimit.conePointUniqueUpToIso]
  /-
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.HasLimit F
    t : CategoryTheory.Limits.LimitCone F
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (t.isLimit.lift (CategoryTheory.Limit …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem limit.isoLimitCone_inv_π {F : J ⥤ C} [HasLimit F] (t : LimitCone F) (j : J) :
    (limit.isoLimitCone t).inv ≫ limit.π F j = t.cone.π.app j := by
  /-
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.HasLimit F
    t : CategoryTheory.Limits.LimitCone F
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.isoLimit …
  -/
  dsimp [limit.isoLimitCone, IsLimit.conePointUniqueUpToIso]
  /-
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.HasLimit F
    t : CategoryTheory.Limits.LimitCone F
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.lift F t …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


@[ext]
theorem limit.hom_ext {F : J ⥤ C} [HasLimit F] {X : C} {f f' : X ⟶ limit F}
    (w : ∀ j, f ≫ limit.π F j = f' ≫ limit.π F j) : f = f' :=
  (limit.isLimit F).hom_ext w


@[reassoc (attr := simp)]
theorem limit.lift_map {F G : J ⥤ C} [HasLimit F] [HasLimit G] (c : Cone F) (α : F ⟶ G) :
    limit.lift F c ≫ limMap α = limit.lift G ((Cones.postcompose α).obj c) := by
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasLimit F
    inst✝ : CategoryTheory.Limits.HasLimit G
    c : CategoryTheory.Limits.Cone F
    α : Quiver.Hom F G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.lift F c …
  -/
  ext
  /-
    case w
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasLimit F
    inst✝ : CategoryTheory.Limits.HasLimit G
    c : CategoryTheory.Limits.Cone F
    α : Quiver.Hom F G
    j✝ : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [assoc, limMap_π, limit.lift_π_assoc, limit.lift_π]
  /-
    case w
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasLimit F
    inst✝ : CategoryTheory.Limits.HasLimit G
    c : CategoryTheory.Limits.Cone F
    α : Quiver.Hom F G
    j✝ : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app j✝) (α.app j✝)) (((CategoryT …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem limit.lift_cone {F : J ⥤ C} [HasLimit F] : limit.lift F (limit.cone F) = 𝟙 (limit F) :=
  (limit.isLimit _).lift_self


/-- The isomorphism (in `Type`) between
morphisms from a specified object `W` to the limit object,
and cones with cone point `W`.
-/
def limit.homIso (F : J ⥤ C) [HasLimit F] (W : C) :
    ULift.{u₁} (W ⟶ limit F : Type v) ≅ F.cones.obj (op W) :=
  (limit.isLimit F).homIso W


@[simp]
theorem limit.homIso_hom (F : J ⥤ C) [HasLimit F] {W : C} (f : ULift (W ⟶ limit F)) :
    (limit.homIso F W).hom f = (const J).map f.down ≫ (limit.cone F).π :=
  (limit.isLimit F).homIso_hom f


/-- The isomorphism (in `Type`) between
morphisms from a specified object `W` to the limit object,
and an explicit componentwise description of cones with cone point `W`.
-/
def limit.homIso' (F : J ⥤ C) [HasLimit F] (W : C) :
    ULift.{u₁} (W ⟶ limit F : Type v) ≅
      { p : ∀ j, W ⟶ F.obj j // ∀ {j j' : J} (f : j ⟶ j'), p j ≫ F.map f = p j' } :=
  (limit.isLimit F).homIso' W


theorem limit.lift_extend {F : J ⥤ C} [HasLimit F] (c : Cone F) {X : C} (f : X ⟶ c.pt) :
                                                         /-
                                                           J : Type u₁
                                                           inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                                           C : Type u
                                                           inst✝¹ : CategoryTheory.Category.{v, u} C
                                                           F : CategoryTheory.Functor J C
                                                           inst✝ : CategoryTheory.Limits.HasLimit F
                                                           c : CategoryTheory.Limits.Cone F
                                                           X : C
                                                           f : Quiver.Hom X c.pt
                                                           ⊢ Eq (CategoryTheory.Limits.limit.lift F (c.extend f)) (CategoryTheory.Categor …
                                                         -/
    limit.lift F (c.extend f) = f ≫ limit.lift F c := by aesop_cat
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- If a functor `F` has a limit, so does any naturally isomorphic functor.
-/
theorem hasLimitOfIso {F G : J ⥤ C} [HasLimit F] (α : F ≅ G) : HasLimit G :=
  HasLimit.mk
    { cone := (Cones.postcompose α.hom).obj (limit.cone F)
      isLimit := (IsLimit.postcomposeHomEquiv _ _).symm (limit.isLimit F) }

-- See the construction of limits from products and equalizers
-- for an example usage.

/-- If a functor `G` has the same collection of cones as a functor `F`
which has a limit, then `G` also has a limit. -/
theorem HasLimit.ofConesIso {J K : Type u₁} [Category.{v₁} J] [Category.{v₂} K] (F : J ⥤ C)
    (G : K ⥤ C) (h : F.cones ≅ G.cones) [HasLimit F] : HasLimit G :=
  HasLimit.mk ⟨_, IsLimit.ofNatIso (IsLimit.natIso (limit.isLimit F) ≪≫ h)⟩


/-- The limits of `F : J ⥤ C` and `G : J ⥤ C` are isomorphic,
if the functors are naturally isomorphic.
-/
def HasLimit.isoOfNatIso {F G : J ⥤ C} [HasLimit F] [HasLimit G] (w : F ≅ G) : limit F ≅ limit G :=
  IsLimit.conePointsIsoOfNatIso (limit.isLimit F) (limit.isLimit G) w


@[reassoc (attr := simp)]
theorem HasLimit.isoOfNatIso_hom_π {F G : J ⥤ C} [HasLimit F] [HasLimit G] (w : F ≅ G) (j : J) :
    (HasLimit.isoOfNatIso w).hom ≫ limit.π G j = limit.π F j ≫ w.hom.app j :=
  IsLimit.conePointsIsoOfNatIso_hom_comp _ _ _ _


@[reassoc (attr := simp)]
theorem HasLimit.isoOfNatIso_inv_π {F G : J ⥤ C} [HasLimit F] [HasLimit G] (w : F ≅ G) (j : J) :
    (HasLimit.isoOfNatIso w).inv ≫ limit.π F j = limit.π G j ≫ w.inv.app j :=
  IsLimit.conePointsIsoOfNatIso_inv_comp _ _ _ _


@[reassoc (attr := simp)]
theorem HasLimit.lift_isoOfNatIso_hom {F G : J ⥤ C} [HasLimit F] [HasLimit G] (t : Cone F)
    (w : F ≅ G) :
    limit.lift F t ≫ (HasLimit.isoOfNatIso w).hom =
      limit.lift G ((Cones.postcompose w.hom).obj _) :=
  IsLimit.lift_comp_conePointsIsoOfNatIso_hom _ _ _


@[reassoc (attr := simp)]
theorem HasLimit.lift_isoOfNatIso_inv {F G : J ⥤ C} [HasLimit F] [HasLimit G] (t : Cone G)
    (w : F ≅ G) :
    limit.lift G t ≫ (HasLimit.isoOfNatIso w).inv =
      limit.lift F ((Cones.postcompose w.inv).obj _) :=
  IsLimit.lift_comp_conePointsIsoOfNatIso_inv _ _ _


/-- The limits of `F : J ⥤ C` and `G : K ⥤ C` are isomorphic,
if there is an equivalence `e : J ≌ K` making the triangle commute up to natural isomorphism.
-/
def HasLimit.isoOfEquivalence {F : J ⥤ C} [HasLimit F] {G : K ⥤ C} [HasLimit G] (e : J ≌ K)
    (w : e.functor ⋙ G ≅ F) : limit F ≅ limit G :=
  IsLimit.conePointsIsoOfEquivalence (limit.isLimit F) (limit.isLimit G) e w


@[simp]
theorem HasLimit.isoOfEquivalence_hom_π {F : J ⥤ C} [HasLimit F] {G : K ⥤ C} [HasLimit G]
    (e : J ≌ K) (w : e.functor ⋙ G ≅ F) (k : K) :
    (HasLimit.isoOfEquivalence e w).hom ≫ limit.π G k =
      limit.π F (e.inverse.obj k) ≫ w.inv.app (e.inverse.obj k) ≫ G.map (e.counit.app k) := by
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasLimit F
    G : CategoryTheory.Functor K C
    inst✝ : CategoryTheory.Limits.HasLimit G
    e : CategoryTheory.Equivalence J K
    w : CategoryTheory.Iso (e.functor.comp G) F
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.HasLimit.isoOf …
  -/
  simp only [HasLimit.isoOfEquivalence, IsLimit.conePointsIsoOfEquivalence_hom]
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasLimit F
    G : CategoryTheory.Functor K C
    inst✝ : CategoryTheory.Limits.HasLimit G
    e : CategoryTheory.Equivalence J K
    w : CategoryTheory.Iso (e.functor.comp G) F
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.limit.isLimit …
  -/
  dsimp
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasLimit F
    G : CategoryTheory.Functor K C
    inst✝ : CategoryTheory.Limits.HasLimit G
    e : CategoryTheory.Equivalence J K
    w : CategoryTheory.Iso (e.functor.comp G) F
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.lift G ( …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem HasLimit.isoOfEquivalence_inv_π {F : J ⥤ C} [HasLimit F] {G : K ⥤ C} [HasLimit G]
    (e : J ≌ K) (w : e.functor ⋙ G ≅ F) (j : J) :
    (HasLimit.isoOfEquivalence e w).inv ≫ limit.π F j =
    limit.π G (e.functor.obj j) ≫ w.hom.app j := by
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasLimit F
    G : CategoryTheory.Functor K C
    inst✝ : CategoryTheory.Limits.HasLimit G
    e : CategoryTheory.Equivalence J K
    w : CategoryTheory.Iso (e.functor.comp G) F
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.HasLimit.isoOf …
  -/
  simp only [HasLimit.isoOfEquivalence, IsLimit.conePointsIsoOfEquivalence_hom]
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasLimit F
    G : CategoryTheory.Functor K C
    inst✝ : CategoryTheory.Limits.HasLimit G
    e : CategoryTheory.Equivalence J K
    w : CategoryTheory.Iso (e.functor.comp G) F
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.limit.isLimit …
  -/
  dsimp
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasLimit F
    G : CategoryTheory.Functor K C
    inst✝ : CategoryTheory.Limits.HasLimit G
    e : CategoryTheory.Equivalence J K
    w : CategoryTheory.Iso (e.functor.comp G) F
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.lift F ( …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The canonical morphism from the limit of `F` to the limit of `E ⋙ F`.
-/
def limit.pre : limit F ⟶ limit (E ⋙ F) :=
  limit.lift (E ⋙ F) ((limit.cone F).whisker E)


@[reassoc (attr := simp)]
theorem limit.pre_π (k : K) : limit.pre F E ≫ limit.π (E ⋙ F) k = limit.π F (E.obj k) := by
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasLimit F
    E : CategoryTheory.Functor K J
    inst✝ : CategoryTheory.Limits.HasLimit (E.comp F)
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.pre F E) …
  -/
  erw [IsLimit.fac]
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasLimit F
    E : CategoryTheory.Functor K J
    inst✝ : CategoryTheory.Limits.HasLimit (E.comp F)
    k : K
    ⊢ Eq ((CategoryTheory.Limits.Cone.whisker E (CategoryTheory.Limits.limit.cone  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem limit.lift_pre (c : Cone F) :
                                                                            /-
                                                                              J : Type u₁
                                                                              inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
                                                                              K : Type u₂
                                                                              inst✝³ : CategoryTheory.Category.{v₂, u₂} K
                                                                              C : Type u
                                                                              inst✝² : CategoryTheory.Category.{v, u} C
                                                                              F : CategoryTheory.Functor J C
                                                                              inst✝¹ : CategoryTheory.Limits.HasLimit F
                                                                              E : CategoryTheory.Functor K J
                                                                              inst✝ : CategoryTheory.Limits.HasLimit (E.comp F)
                                                                              c : CategoryTheory.Limits.Cone F
                                                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.lift F c …
                                                                            -/
    limit.lift F c ≫ limit.pre F E = limit.lift (E ⋙ F) (c.whisker E) := by ext; simp
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


@[simp]
theorem limit.pre_pre [h : HasLimit (D ⋙ E ⋙ F)] : haveI : HasLimit ((D ⋙ E) ⋙ F) := h
    limit.pre F E ≫ limit.pre (E ⋙ F) D = limit.pre F (D ⋙ E) := by
  /-
    J : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝² : CategoryTheory.Limits.HasLimit F
    E : CategoryTheory.Functor K J
    inst✝¹ : CategoryTheory.Limits.HasLimit (E.comp F)
    L : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} L
    D : CategoryTheory.Functor L K
    h : CategoryTheory.Limits.HasLimit (D.comp (E.comp F))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.pre F E) …
  -/
  haveI : HasLimit ((D ⋙ E) ⋙ F) := h
  /-
    J : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝² : CategoryTheory.Limits.HasLimit F
    E : CategoryTheory.Functor K J
    inst✝¹ : CategoryTheory.Limits.HasLimit (E.comp F)
    L : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} L
    D : CategoryTheory.Functor L K
    h : CategoryTheory.Limits.HasLimit (D.comp (E.comp F))
    this : CategoryTheory.Limits.HasLimit ((D.comp E).comp F)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.pre F E) …
  -/
  ext j; erw [assoc, limit.pre_π, limit.pre_π, limit.pre_π]; rfl
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- -
If we have particular limit cones available for `E ⋙ F` and for `F`,
we obtain a formula for `limit.pre F E`.
-/
theorem limit.pre_eq (s : LimitCone (E ⋙ F)) (t : LimitCone F) :
    limit.pre F E = (limit.isoLimitCone t).hom ≫ s.isLimit.lift (t.cone.whisker E) ≫
                                       /-
                                         J : Type u₁
                                         inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
                                         K : Type u₂
                                         inst✝³ : CategoryTheory.Category.{v₂, u₂} K
                                         C : Type u
                                         inst✝² : CategoryTheory.Category.{v, u} C
                                         F : CategoryTheory.Functor J C
                                         inst✝¹ : CategoryTheory.Limits.HasLimit F
                                         E : CategoryTheory.Functor K J
                                         inst✝ : CategoryTheory.Limits.HasLimit (E.comp F)
                                         s : CategoryTheory.Limits.LimitCone (E.comp F)
                                         t : CategoryTheory.Limits.LimitCone F
                                         ⊢ Eq (CategoryTheory.Limits.limit.pre F E) (CategoryTheory.CategoryStruct.comp …
                                       -/
      (limit.isoLimitCone s).inv := by aesop_cat
                                       /-
                                         🎉 no goals
                                       -/


/-- The canonical morphism from `G` applied to the limit of `F` to the limit of `F ⋙ G`.
-/
def limit.post : G.obj (limit F) ⟶ limit (F ⋙ G) :=
  limit.lift (F ⋙ G) (G.mapCone (limit.cone F))


@[reassoc (attr := simp)]
theorem limit.post_π (j : J) : limit.post F G ≫ limit.π (F ⋙ G) j = G.map (limit.π F j) := by
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasLimit F
    G : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.Limits.HasLimit (F.comp G)
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.post F G …
  -/
  erw [IsLimit.fac]
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasLimit F
    G : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.Limits.HasLimit (F.comp G)
    j : J
    ⊢ Eq ((G.mapCone (CategoryTheory.Limits.limit.cone F)).π.app j) (G.map (Catego …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem limit.lift_post (c : Cone F) :
    G.map (limit.lift F c) ≫ limit.post F G = limit.lift (F ⋙ G) (G.mapCone c) := by
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasLimit F
    G : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.Limits.HasLimit (F.comp G)
    c : CategoryTheory.Limits.Cone F
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.limit.l …
  -/
  ext
  /-
    case w
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasLimit F
    G : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.Limits.HasLimit (F.comp G)
    c : CategoryTheory.Limits.Cone F
    j✝ : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [assoc, limit.post_π, ← G.map_comp, limit.lift_π, limit.lift_π]
  /-
    case w
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasLimit F
    G : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.Limits.HasLimit (F.comp G)
    c : CategoryTheory.Limits.Cone F
    j✝ : J
    ⊢ Eq (G.map (c.π.app j✝)) ((G.mapCone c).π.app j✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem limit.post_post {E : Type u''} [Category.{v''} E] (H : D ⥤ E) [h : HasLimit ((F ⋙ G) ⋙ H)] :
    -- H G (limit F) ⟶ H (limit (F ⋙ G)) ⟶ limit ((F ⋙ G) ⋙ H) equals
    -- H G (limit F) ⟶ limit (F ⋙ (G ⋙ H))
    haveI : HasLimit (F ⋙ G ⋙ H) := h
    H.map (limit.post F G) ≫ limit.post (F ⋙ G) H = limit.post F (G ⋙ H) := by
  /-
    J : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor J C
    inst✝² : CategoryTheory.Limits.HasLimit F
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasLimit (F.comp G)
    E : Type u''
    inst✝ : CategoryTheory.Category.{v'', u''} E
    H : CategoryTheory.Functor D E
    h : CategoryTheory.Limits.HasLimit ((F.comp G).comp H)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (H.map (CategoryTheory.Limits.limit.p …
  -/
  haveI : HasLimit (F ⋙ G ⋙ H) := h
  /-
    J : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor J C
    inst✝² : CategoryTheory.Limits.HasLimit F
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasLimit (F.comp G)
    E : Type u''
    inst✝ : CategoryTheory.Category.{v'', u''} E
    H : CategoryTheory.Functor D E
    h : CategoryTheory.Limits.HasLimit ((F.comp G).comp H)
    this : CategoryTheory.Limits.HasLimit (F.comp (G.comp H))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (H.map (CategoryTheory.Limits.limit.p …
  -/
  ext; erw [assoc, limit.post_π, ← H.map_comp, limit.post_π, limit.post_π]; rfl
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem limit.pre_post {D : Type u'} [Category.{v'} D] (E : K ⥤ J) (F : J ⥤ C) (G : C ⥤ D)
    [HasLimit F] [HasLimit (E ⋙ F)] [HasLimit (F ⋙ G)]
    [h : HasLimit ((E ⋙ F) ⋙ G)] :-- G (limit F) ⟶ G (limit (E ⋙ F)) ⟶ limit ((E ⋙ F) ⋙ G) vs
            -- G (limit F) ⟶ limit F ⋙ G ⟶ limit (E ⋙ (F ⋙ G)) or
    haveI : HasLimit (E ⋙ F ⋙ G) := h
    G.map (limit.pre F E) ≫ limit.post (E ⋙ F) G = limit.post F G ≫ limit.pre (F ⋙ G) E := by
  /-
    J : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    E : CategoryTheory.Functor K J
    F : CategoryTheory.Functor J C
    G : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.Limits.HasLimit F
    inst✝¹ : CategoryTheory.Limits.HasLimit (E.comp F)
    inst✝ : CategoryTheory.Limits.HasLimit (F.comp G)
    h : CategoryTheory.Limits.HasLimit ((E.comp F).comp G)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.limit.p …
  -/
  haveI : HasLimit (E ⋙ F ⋙ G) := h
  /-
    J : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    E : CategoryTheory.Functor K J
    F : CategoryTheory.Functor J C
    G : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.Limits.HasLimit F
    inst✝¹ : CategoryTheory.Limits.HasLimit (E.comp F)
    inst✝ : CategoryTheory.Limits.HasLimit (F.comp G)
    h : CategoryTheory.Limits.HasLimit ((E.comp F).comp G)
    this : CategoryTheory.Limits.HasLimit (E.comp (F.comp G))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.limit.p …
  -/
  ext; erw [assoc, limit.post_π, ← G.map_comp, limit.pre_π, assoc, limit.pre_π, limit.post_π]
       /-
         🎉 no goals
       -/


instance hasLimitEquivalenceComp (e : K ≌ J) [HasLimit F] : HasLimit (e.functor ⋙ F) :=
  HasLimit.mk
    { cone := Cone.whisker e.functor (limit.cone F)
      isLimit := IsLimit.whiskerEquivalence (limit.isLimit F) e }

-- Porting note: testing whether this still needed
-- attribute [local elab_without_expected_type] inv_fun_id_assoc

-- not entirely sure why this is needed

/-- If a `E ⋙ F` has a limit, and `E` is an equivalence, we can construct a limit of `F`.
-/
theorem hasLimitOfEquivalenceComp (e : K ≌ J) [HasLimit (e.functor ⋙ F)] : HasLimit F := by
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    e : CategoryTheory.Equivalence K J
    inst✝ : CategoryTheory.Limits.HasLimit (e.functor.comp F)
    ⊢ CategoryTheory.Limits.HasLimit F
  -/
  haveI : HasLimit (e.inverse ⋙ e.functor ⋙ F) := Limits.hasLimitEquivalenceComp e.symm
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    e : CategoryTheory.Equivalence K J
    inst✝ : CategoryTheory.Limits.HasLimit (e.functor.comp F)
    this : CategoryTheory.Limits.HasLimit (e.inverse.comp (e.functor.comp F))
    ⊢ CategoryTheory.Limits.HasLimit F
  -/
  apply hasLimitOfIso (e.invFunIdAssoc F)
  /-
    🎉 no goals
  -/

-- `hasLimitCompEquivalence` and `hasLimitOfCompEquivalence`
-- are proved in `CategoryTheory/Adjunction/Limits.lean`.

/-- `limit F` is functorial in `F`, when `C` has all limits of shape `J`. -/
@[simps]
def lim : (J ⥤ C) ⥤ C where
  obj F := limit F
  map α := limMap α
  map_id F := by
    /-
      J : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} K
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      F✝ : CategoryTheory.Functor J C
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
      F : CategoryTheory.Functor J C
      ⊢ Eq ({ obj := fun F => CategoryTheory.Limits.limit F, map := fun {X Y} α => C …
    -/
    apply Limits.limit.hom_ext; intro j
    /-
      case w
      J : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} K
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      F✝ : CategoryTheory.Functor J C
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
      F : CategoryTheory.Functor J C
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ({ obj := fun F => CategoryTheory.Lim …
    -/
    erw [limMap_π, Category.id_comp, Category.comp_id]
    /-
      🎉 no goals
    -/
  map_comp α β := by
    /-
      J : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} K
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J C
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
      X✝ Y✝ Z✝ : CategoryTheory.Functor J C
      α : Quiver.Hom X✝ Y✝
      β : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun F => CategoryTheory.Limits.limit F, map := fun {X Y} α => C …
    -/
    apply Limits.limit.hom_ext; intro j
    /-
      case w
      J : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} K
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J C
      inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
      X✝ Y✝ Z✝ : CategoryTheory.Functor J C
      α : Quiver.Hom X✝ Y✝
      β : Quiver.Hom Y✝ Z✝
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ({ obj := fun F => CategoryTheory.Lim …
    -/
    erw [assoc, IsLimit.fac, IsLimit.fac, ← assoc, IsLimit.fac, assoc]; rfl
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem limit.map_pre [HasLimitsOfShape K C] (E : K ⥤ J) :
    lim.map α ≫ limit.pre G E = limit.pre F E ≫ lim.map (whiskerLeft E α) := by
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
    G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape K C
    E : CategoryTheory.Functor K J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.lim.map α) (Ca …
  -/
  ext
  /-
    case w
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
    G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape K C
    E : CategoryTheory.Functor K J
    j✝ : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem limit.map_pre' [HasLimitsOfShape K C] (F : J ⥤ C) {E₁ E₂ : K ⥤ J} (α : E₁ ⟶ E₂) :
    limit.pre F E₂ = limit.pre F E₁ ≫ lim.map (whiskerRight α F) := by
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape J C
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape K C
    F : CategoryTheory.Functor J C
    E₁ E₂ : CategoryTheory.Functor K J
    α : Quiver.Hom E₁ E₂
    ⊢ Eq (CategoryTheory.Limits.limit.pre F E₂) (CategoryTheory.CategoryStruct.com …
  -/
  ext1; simp [← category.assoc]
        /-
          🎉 no goals
        -/


theorem limit.id_pre (F : J ⥤ C) : limit.pre F (𝟭 _) = lim.map (Functor.leftUnitor F).inv := by
  /-
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    F : CategoryTheory.Functor J C
    ⊢ Eq (CategoryTheory.Limits.limit.pre F (CategoryTheory.Functor.id J)) (Catego …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


theorem limit.map_post {D : Type u'} [Category.{v'} D] [HasLimitsOfShape J D] (H : C ⥤ D) :
    /- H (limit F) ⟶ H (limit G) ⟶ limit (G ⋙ H) vs
     H (limit F) ⟶ limit (F ⋙ H) ⟶ limit (G ⋙ H) -/
    H.map (limMap α) ≫ limit.post G H = limit.post F H ≫ limMap (whiskerRight α H) := by
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape J C
    G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    D : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} D
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J D
    H : CategoryTheory.Functor C D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (H.map (CategoryTheory.Limits.limMap  …
  -/
  ext
  /-
    case w
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape J C
    G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    D : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} D
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J D
    H : CategoryTheory.Functor C D
    j✝ : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [whiskerRight_app, limMap_π, assoc, limit.post_π_assoc, limit.post_π, ← H.map_comp]
  /-
    🎉 no goals
  -/


/-- The isomorphism between
morphisms from `W` to the cone point of the limit cone for `F`
and cones over `F` with cone point `W`
is natural in `F`.
-/
def limYoneda :
    lim ⋙ yoneda ⋙ (whiskeringRight _ _ _).obj uliftFunctor.{u₁} ≅ CategoryTheory.cones J C :=
                               /-
                                 J : Type u₁
                                 inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                 K : Type u₂
                                 inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                 C : Type u
                                 inst✝¹ : CategoryTheory.Category.{v, u} C
                                 F✝ : CategoryTheory.Functor J C
                                 inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
                                 G : CategoryTheory.Functor J C
                                 α : Quiver.Hom F✝ G
                                 F : CategoryTheory.Functor J C
                                 ⊢ ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun F => NatIso.ofComponents fun W => limit.homIso F (unop W)
  /-
    🎉 no goals
  -/


/-- The constant functor and limit functor are adjoint to each other -/
def constLimAdj : (const J : C ⥤ J ⥤ C) ⊣ lim := Adjunction.mk' {
  homEquiv := fun c g ↦
    { toFun := fun f => limit.lift _ ⟨c, f⟩
      invFun := fun f =>
        { app := fun _ => f ≫ limit.π _ _ }
                     /-
                       J : Type u₁
                       inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                       K : Type u₂
                       inst✝² : CategoryTheory.Category.{v₂, u₂} K
                       C : Type u
                       inst✝¹ : CategoryTheory.Category.{v, u} C
                       F : CategoryTheory.Functor J C
                       inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
                       G : CategoryTheory.Functor J C
                       α : Quiver.Hom F G
                       c : C
                       g : CategoryTheory.Functor J C
                       ⊢ Function.LeftInverse (fun f => { app := fun x => CategoryTheory.CategoryStru …
                     -/
      left_inv := by aesop_cat
                     /-
                       🎉 no goals
                     -/
                      /-
                        J : Type u₁
                        inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                        K : Type u₂
                        inst✝² : CategoryTheory.Category.{v₂, u₂} K
                        C : Type u
                        inst✝¹ : CategoryTheory.Category.{v, u} C
                        F : CategoryTheory.Functor J C
                        inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
                        G : CategoryTheory.Functor J C
                        α : Quiver.Hom F G
                        c : C
                        g : CategoryTheory.Functor J C
                        ⊢ Function.RightInverse (fun f => { app := fun x => CategoryTheory.CategoryStr …
                      -/
      right_inv := by aesop_cat }
                      /-
                        🎉 no goals
                      -/
  unit := { app := fun _ => limit.lift _ ⟨_, 𝟙 _⟩ }
  counit := { app := fun g => { app := limit.π _ } } }


instance : IsRightAdjoint (lim : (J ⥤ C) ⥤ C) :=
  ⟨_, ⟨constLimAdj⟩⟩


instance limMap_mono' {F G : J ⥤ C} [HasLimitsOfShape J C] (α : F ⟶ G) [Mono α] : Mono (limMap α) :=
  (lim : (J ⥤ C) ⥤ C).map_mono α


instance limMap_mono {F G : J ⥤ C} [HasLimit F] [HasLimit G] (α : F ⟶ G) [∀ j, Mono (α.app j)] :
    Mono (limMap α) :=
  ⟨fun {Z} u v h =>
                                                           /-
                                                             J : Type u₁
                                                             inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
                                                             K : Type u₂
                                                             inst✝⁴ : CategoryTheory.Category.{v₂, u₂} K
                                                             C : Type u
                                                             inst✝³ : CategoryTheory.Category.{v, u} C
                                                             F✝ F G : CategoryTheory.Functor J C
                                                             inst✝² : CategoryTheory.Limits.HasLimit F
                                                             inst✝¹ : CategoryTheory.Limits.HasLimit G
                                                             α : Quiver.Hom F G
                                                             inst✝ : ∀ (j : J), CategoryTheory.Mono (α.app j)
                                                             Z : C
                                                             u v : Quiver.Hom Z (CategoryTheory.Limits.limit F)
                                                             h : Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.Limits.limMap α)) …
                                                             j : J
                                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp u …
                                                           -/
    limit.hom_ext fun j => (cancel_mono (α.app j)).1 <| by simpa using h =≫ limit.π _ j⟩
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- The limit cone obtained from a right adjoint of the constant functor. -/
@[simps]
noncomputable def coneOfAdj (F : J ⥤ C) : Cone F where
  pt := L.obj F
  π := adj.counit.app F


/-- The cones defined by `coneOfAdj` are limit cones. -/
@[simps]
def isLimitConeOfAdj (F : J ⥤ C) :
    IsLimit (coneOfAdj adj F) where
  lift s := adj.homEquiv _ _ s.π
  fac s j := by
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F✝ : CategoryTheory.Functor J C
      L : CategoryTheory.Functor (CategoryTheory.Functor J C) C
      adj : CategoryTheory.Adjunction (CategoryTheory.Functor.const J) L
      F : CategoryTheory.Functor J C
      s : CategoryTheory.Limits.Cone F
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => (adj.homEquiv s.pt F) s.π) …
    -/
    have eq := NatTrans.congr_app (adj.counit.naturality s.π) j
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F✝ : CategoryTheory.Functor J C
      L : CategoryTheory.Functor (CategoryTheory.Functor J C) C
      adj : CategoryTheory.Adjunction (CategoryTheory.Functor.const J) L
      F : CategoryTheory.Functor J C
      s : CategoryTheory.Limits.Cone F
      j : J
      eq : Eq ((CategoryTheory.CategoryStruct.comp ((L.comp (CategoryTheory.Functor. …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => (adj.homEquiv s.pt F) s.π) …
    -/
    have eq' := NatTrans.congr_app (adj.left_triangle_components s.pt) j
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F✝ : CategoryTheory.Functor J C
      L : CategoryTheory.Functor (CategoryTheory.Functor J C) C
      adj : CategoryTheory.Adjunction (CategoryTheory.Functor.const J) L
      F : CategoryTheory.Functor J C
      s : CategoryTheory.Limits.Cone F
      j : J
      eq : Eq ((CategoryTheory.CategoryStruct.comp ((L.comp (CategoryTheory.Functor. …
      eq' : Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.const J …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => (adj.homEquiv s.pt F) s.π) …
    -/
    dsimp at eq eq' ⊢
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F✝ : CategoryTheory.Functor J C
      L : CategoryTheory.Functor (CategoryTheory.Functor J C) C
      adj : CategoryTheory.Adjunction (CategoryTheory.Functor.const J) L
      F : CategoryTheory.Functor J C
      s : CategoryTheory.Limits.Cone F
      j : J
      eq : Eq (CategoryTheory.CategoryStruct.comp (L.map s.π) ((adj.counit.app F).ap …
      eq' : Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app s.pt) ((adj.counit. …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((adj.homEquiv s.pt F) s.π) ((adj.cou …
    -/
    rw [adj.homEquiv_unit, assoc, eq, reassoc_of% eq']
    /-
      🎉 no goals
    -/
                                                       /-
                                                         J : Type u₁
                                                         inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                                         K : Type u₂
                                                         inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
                                                         C : Type u
                                                         inst✝ : CategoryTheory.Category.{v, u} C
                                                         F✝ : CategoryTheory.Functor J C
                                                         L : CategoryTheory.Functor (CategoryTheory.Functor J C) C
                                                         adj : CategoryTheory.Adjunction (CategoryTheory.Functor.const J) L
                                                         F : CategoryTheory.Functor J C
                                                         s : CategoryTheory.Limits.Cone F
                                                         m : Quiver.Hom s.pt (CategoryTheory.Limits.coneOfAdj adj F).pt
                                                         hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Limi …
                                                         ⊢ Eq ((adj.homEquiv s.pt F).symm m) ((adj.homEquiv s.pt F).symm ((fun s => (ad …
                                                       -/
  uniq s m hm := (adj.homEquiv _ _).symm.injective (by ext j; simpa using hm j)
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- We can transport limits of shape `J` along an equivalence `J ≌ J'`.
-/
theorem hasLimitsOfShape_of_equivalence {J' : Type u₂} [Category.{v₂} J'] (e : J ≌ J')
    [HasLimitsOfShape J C] : HasLimitsOfShape J' C := by
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J' : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} J'
    e : CategoryTheory.Equivalence J J'
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    ⊢ CategoryTheory.Limits.HasLimitsOfShape J' C
  -/
  constructor
  /-
    case has_limit
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J' : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} J'
    e : CategoryTheory.Equivalence J J'
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    ⊢ autoParam (∀ (F : CategoryTheory.Functor J' C), CategoryTheory.Limits.HasLim …
  -/
  intro F
  /-
    case has_limit
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J' : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} J'
    e : CategoryTheory.Equivalence J J'
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    F : CategoryTheory.Functor J' C
    ⊢ CategoryTheory.Limits.HasLimit F
  -/
  apply hasLimitOfEquivalenceComp e
  /-
    🎉 no goals
  -/


/-- A category that has larger limits also has smaller limits. -/
theorem hasLimitsOfSizeOfUnivLE [UnivLE.{v₂, v₁}] [UnivLE.{u₂, u₁}]
    [HasLimitsOfSize.{v₁, u₁} C] : HasLimitsOfSize.{v₂, u₂} C where
  has_limits_of_shape J {_} := hasLimitsOfShape_of_equivalence
    ((ShrinkHoms.equivalence J).trans <| Shrink.equivalence _).symm


/-- `hasLimitsOfSizeShrink.{v u} C` tries to obtain `HasLimitsOfSize.{v u} C`
from some other `HasLimitsOfSize C`.
-/
theorem hasLimitsOfSizeShrink [HasLimitsOfSize.{max v₁ v₂, max u₁ u₂} C] :
    HasLimitsOfSize.{v₁, u₁} C := hasLimitsOfSizeOfUnivLE.{max v₁ v₂, max u₁ u₂} C


instance (priority := 100) hasSmallestLimitsOfHasLimits [HasLimits C] : HasLimitsOfSize.{0, 0} C :=
  hasLimitsOfSizeShrink.{0, 0} C


/-- `ColimitCocone F` contains a cocone over `F` together with the information that it is a
    colimit. -/
-- @[nolint has_nonempty_instance] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed; linter not ported yet
structure ColimitCocone (F : J ⥤ C) where
  /-- The cocone itself -/
  cocone : Cocone F
  /-- The proof that it is the colimit cocone -/
  isColimit : IsColimit cocone


/-- `HasColimit F` represents the mere existence of a colimit for `F`. -/
class HasColimit (F : J ⥤ C) : Prop where mk' ::
  /-- There exists a colimit for `F` -/
  exists_colimit : Nonempty (ColimitCocone F)


theorem HasColimit.mk {F : J ⥤ C} (d : ColimitCocone F) : HasColimit F :=
  ⟨Nonempty.intro d⟩


/-- Use the axiom of choice to extract explicit `ColimitCocone F` from `HasColimit F`. -/
def getColimitCocone (F : J ⥤ C) [HasColimit F] : ColimitCocone F :=
  Classical.choice <| HasColimit.exists_colimit


/-- `C` has colimits of shape `J` if there exists a colimit for every functor `F : J ⥤ C`. -/
class HasColimitsOfShape : Prop where
  /-- All `F : J ⥤ C` have colimits for a fixed `J` -/
  has_colimit : ∀ F : J ⥤ C, HasColimit F := by infer_instance


/-- `C` has all colimits of size `v₁ u₁` (`HasColimitsOfSize.{v₁ u₁} C`)
if it has colimits of every shape `J : Type u₁` with `[Category.{v₁} J]`.
-/
@[pp_with_univ]
class HasColimitsOfSize (C : Type u) [Category.{v} C] : Prop where
  /-- All `F : J ⥤ C` have colimits for all small `J` -/
  has_colimits_of_shape : ∀ (J : Type u₁) [Category.{v₁} J], HasColimitsOfShape J C := by
    infer_instance


/-- `C` has all (small) colimits if it has colimits of every shape that is as big as its hom-sets.
-/
abbrev HasColimits (C : Type u) [Category.{v} C] : Prop :=
  HasColimitsOfSize.{v, v} C


theorem HasColimits.hasColimitsOfShape {C : Type u} [Category.{v} C] [HasColimits C] (J : Type v)
    [Category.{v} J] : HasColimitsOfShape J C :=
  HasColimitsOfSize.has_colimits_of_shape J


instance (priority := 100) hasColimitOfHasColimitsOfShape {J : Type u₁} [Category.{v₁} J]
    [HasColimitsOfShape J C] (F : J ⥤ C) : HasColimit F :=
  HasColimitsOfShape.has_colimit F

-- see Note [lower instance priority]

instance (priority := 100) hasColimitsOfShapeOfHasColimitsOfSize {J : Type u₁} [Category.{v₁} J]
    [HasColimitsOfSize.{v₁, u₁} C] : HasColimitsOfShape J C :=
  HasColimitsOfSize.has_colimits_of_shape J

-- Interface to the `HasColimit` class.

/-- An arbitrary choice of colimit cocone of a functor. -/
def colimit.cocone (F : J ⥤ C) [HasColimit F] : Cocone F :=
  (getColimitCocone F).cocone


/-- An arbitrary choice of colimit object of a functor. -/
def colimit (F : J ⥤ C) [HasColimit F] :=
  (colimit.cocone F).pt


/-- The coprojection from a value of the functor to the colimit object. -/
def colimit.ι (F : J ⥤ C) [HasColimit F] (j : J) : F.obj j ⟶ colimit F :=
  (colimit.cocone F).ι.app j


@[reassoc]
theorem colimit.eqToHom_comp_ι (F : J ⥤ C) [HasColimit F] {j j' : J} (hj : j = j') :
                /-
                  J : Type u₁
                  inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                  K : Type u₂
                  inst✝² : CategoryTheory.Category.{v₂, u₂} K
                  C : Type u
                  inst✝¹ : CategoryTheory.Category.{v, u} C
                  F✝ F : CategoryTheory.Functor J C
                  inst✝ : CategoryTheory.Limits.HasColimit F
                  j j' : J
                  hj : Eq j j'
                  ⊢ Eq (F.obj j') (F.obj j)
                -/
    eqToHom (by subst hj; rfl) ≫ colimit.ι F j = colimit.ι F j'  := by
                          /-
                            🎉 no goals
                          -/
  /-
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.HasColimit F
    j j' : J
    hj : Eq j j'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
  -/
  subst hj
  /-
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.HasColimit F
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem colimit.cocone_ι {F : J ⥤ C} [HasColimit F] (j : J) :
    (colimit.cocone F).ι.app j = colimit.ι _ j :=
  rfl


@[simp]
theorem colimit.cocone_x {F : J ⥤ C} [HasColimit F] : (colimit.cocone F).pt = colimit F :=
  rfl


@[reassoc (attr := simp)]
theorem colimit.w (F : J ⥤ C) [HasColimit F] {j j' : J} (f : j ⟶ j') :
    F.map f ≫ colimit.ι F j' = colimit.ι F j :=
  (colimit.cocone F).w f


/-- Evidence that the arbitrary choice of cocone is a colimit cocone. -/
def colimit.isColimit (F : J ⥤ C) [HasColimit F] : IsColimit (colimit.cocone F) :=
  (getColimitCocone F).isColimit


/-- The morphism from the colimit object to the cone point of any other cocone. -/
def colimit.desc (F : J ⥤ C) [HasColimit F] (c : Cocone F) : colimit F ⟶ c.pt :=
  (colimit.isColimit F).desc c


@[simp]
theorem colimit.isColimit_desc {F : J ⥤ C} [HasColimit F] (c : Cocone F) :
    (colimit.isColimit F).desc c = colimit.desc F c :=
  rfl


/-- We have lots of lemmas describing how to simplify `colimit.ι F j ≫ _`,
and combined with `colimit.ext` we rely on these lemmas for many calculations.

However, since `Category.assoc` is a `@[simp]` lemma, often expressions are
right associated, and it's hard to apply these lemmas about `colimit.ι`.

We thus use `reassoc` to define additional `@[simp]` lemmas, with an arbitrary extra morphism.
(see `Tactic/reassoc_axiom.lean`)
 -/
@[reassoc (attr := simp)]
theorem colimit.ι_desc {F : J ⥤ C} [HasColimit F] (c : Cocone F) (j : J) :
    colimit.ι F j ≫ colimit.desc F c = c.ι.app j :=
  IsColimit.fac _ c j


/-- Functoriality of colimits.

Usually this morphism should be accessed through `colim.map`,
but may be needed separately when you have specified colimits for the source and target functors,
but not necessarily for all functors of shape `J`.
-/
def colimMap {F G : J ⥤ C} [HasColimit F] [HasColimit G] (α : F ⟶ G) : colimit F ⟶ colimit G :=
  IsColimit.map (colimit.isColimit F) _ α


@[reassoc (attr := simp)]
theorem ι_colimMap {F G : J ⥤ C} [HasColimit F] [HasColimit G] (α : F ⟶ G) (j : J) :
    colimit.ι F j ≫ colimMap α = α.app j ≫ colimit.ι G j :=
  colimit.ι_desc _ j


/-- The cocone morphism from the arbitrary choice of colimit cocone to any cocone. -/
def colimit.coconeMorphism {F : J ⥤ C} [HasColimit F] (c : Cocone F) : colimit.cocone F ⟶ c :=
  (colimit.isColimit F).descCoconeMorphism c


@[simp]
theorem colimit.coconeMorphism_hom {F : J ⥤ C} [HasColimit F] (c : Cocone F) :
    (colimit.coconeMorphism c).hom = colimit.desc F c :=
  rfl


theorem colimit.ι_coconeMorphism {F : J ⥤ C} [HasColimit F] (c : Cocone F) (j : J) :
                                                                     /-
                                                                       J : Type u₁
                                                                       inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                                                       C : Type u
                                                                       inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                       F : CategoryTheory.Functor J C
                                                                       inst✝ : CategoryTheory.Limits.HasColimit F
                                                                       c : CategoryTheory.Limits.Cocone F
                                                                       j : J
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι F j) …
                                                                     -/
    colimit.ι F j ≫ (colimit.coconeMorphism c).hom = c.ι.app j := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[reassoc (attr := simp)]
theorem colimit.comp_coconePointUniqueUpToIso_hom {F : J ⥤ C} [HasColimit F] {c : Cocone F}
    (hc : IsColimit c) (j : J) :
    colimit.ι F j ≫ (IsColimit.coconePointUniqueUpToIso (colimit.isColimit _) hc).hom = c.ι.app j :=
  IsColimit.comp_coconePointUniqueUpToIso_hom _ _ _


@[reassoc (attr := simp)]
theorem colimit.comp_coconePointUniqueUpToIso_inv {F : J ⥤ C} [HasColimit F] {c : Cocone F}
    (hc : IsColimit c) (j : J) :
    colimit.ι F j ≫ (IsColimit.coconePointUniqueUpToIso hc (colimit.isColimit _)).inv = c.ι.app j :=
  IsColimit.comp_coconePointUniqueUpToIso_inv _ _ _


theorem colimit.existsUnique {F : J ⥤ C} [HasColimit F] (t : Cocone F) :
    ∃! d : colimit F ⟶ t.pt, ∀ j, colimit.ι F j ≫ d = t.ι.app j :=
  (colimit.isColimit F).existsUnique _


/--
Given any other colimit cocone for `F`, the chosen `colimit F` is isomorphic to the cocone point.
-/
def colimit.isoColimitCocone {F : J ⥤ C} [HasColimit F] (t : ColimitCocone F) :
    colimit F ≅ t.cocone.pt :=
  IsColimit.coconePointUniqueUpToIso (colimit.isColimit F) t.isColimit


@[reassoc (attr := simp)]
theorem colimit.isoColimitCocone_ι_hom {F : J ⥤ C} [HasColimit F] (t : ColimitCocone F) (j : J) :
    colimit.ι F j ≫ (colimit.isoColimitCocone t).hom = t.cocone.ι.app j := by
  /-
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.HasColimit F
    t : CategoryTheory.Limits.ColimitCocone F
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι F j) …
  -/
  dsimp [colimit.isoColimitCocone, IsColimit.coconePointUniqueUpToIso]
  /-
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.HasColimit F
    t : CategoryTheory.Limits.ColimitCocone F
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι F j) …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem colimit.isoColimitCocone_ι_inv {F : J ⥤ C} [HasColimit F] (t : ColimitCocone F) (j : J) :
    t.cocone.ι.app j ≫ (colimit.isoColimitCocone t).inv = colimit.ι F j := by
  /-
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.HasColimit F
    t : CategoryTheory.Limits.ColimitCocone F
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (t.cocone.ι.app j) (CategoryTheory.Li …
  -/
  dsimp [colimit.isoColimitCocone, IsColimit.coconePointUniqueUpToIso]
  /-
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.HasColimit F
    t : CategoryTheory.Limits.ColimitCocone F
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (t.cocone.ι.app j) (t.isColimit.desc  …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


@[ext]
theorem colimit.hom_ext {F : J ⥤ C} [HasColimit F] {X : C} {f f' : colimit F ⟶ X}
    (w : ∀ j, colimit.ι F j ≫ f = colimit.ι F j ≫ f') : f = f' :=
  (colimit.isColimit F).hom_ext w


@[simp]
theorem colimit.desc_cocone {F : J ⥤ C} [HasColimit F] :
    colimit.desc F (colimit.cocone F) = 𝟙 (colimit F) :=
  (colimit.isColimit _).desc_self


/-- The isomorphism (in `Type`) between
morphisms from the colimit object to a specified object `W`,
and cocones with cone point `W`.
-/
def colimit.homIso (F : J ⥤ C) [HasColimit F] (W : C) :
    ULift.{u₁} (colimit F ⟶ W : Type v) ≅ F.cocones.obj W :=
  (colimit.isColimit F).homIso W


@[simp]
theorem colimit.homIso_hom (F : J ⥤ C) [HasColimit F] {W : C} (f : ULift (colimit F ⟶ W)) :
    (colimit.homIso F W).hom f = (colimit.cocone F).ι ≫ (const J).map f.down :=
  (colimit.isColimit F).homIso_hom f


/-- The isomorphism (in `Type`) between
morphisms from the colimit object to a specified object `W`,
and an explicit componentwise description of cocones with cone point `W`.
-/
def colimit.homIso' (F : J ⥤ C) [HasColimit F] (W : C) :
    ULift.{u₁} (colimit F ⟶ W : Type v) ≅
      { p : ∀ j, F.obj j ⟶ W // ∀ {j j'} (f : j ⟶ j'), F.map f ≫ p j' = p j } :=
  (colimit.isColimit F).homIso' W


theorem colimit.desc_extend (F : J ⥤ C) [HasColimit F] (c : Cocone F) {X : C} (f : c.pt ⟶ X) :
                                                             /-
                                                               J : Type u₁
                                                               inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                                               C : Type u
                                                               inst✝¹ : CategoryTheory.Category.{v, u} C
                                                               F : CategoryTheory.Functor J C
                                                               inst✝ : CategoryTheory.Limits.HasColimit F
                                                               c : CategoryTheory.Limits.Cocone F
                                                               X : C
                                                               f : Quiver.Hom c.pt X
                                                               ⊢ Eq (CategoryTheory.Limits.colimit.desc F (c.extend f)) (CategoryTheory.Categ …
                                                             -/
    colimit.desc F (c.extend f) = colimit.desc F c ≫ f := by ext1; rw [← Category.assoc]; simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/

-- This has the isomorphism pointing in the opposite direction than in `has_limit_of_iso`.
-- This is intentional; it seems to help with elaboration.

/-- If `F` has a colimit, so does any naturally isomorphic functor.
-/
theorem hasColimitOfIso {F G : J ⥤ C} [HasColimit F] (α : G ≅ F) : HasColimit G :=
  HasColimit.mk
    { cocone := (Cocones.precompose α.hom).obj (colimit.cocone F)
      isColimit := (IsColimit.precomposeHomEquiv _ _).symm (colimit.isColimit F) }


/-- If a functor `G` has the same collection of cocones as a functor `F`
which has a colimit, then `G` also has a colimit. -/
theorem HasColimit.ofCoconesIso {K : Type u₁} [Category.{v₂} K] (F : J ⥤ C) (G : K ⥤ C)
    (h : F.cocones ≅ G.cocones) [HasColimit F] : HasColimit G :=
  HasColimit.mk ⟨_, IsColimit.ofNatIso (IsColimit.natIso (colimit.isColimit F) ≪≫ h)⟩


/-- The colimits of `F : J ⥤ C` and `G : J ⥤ C` are isomorphic,
if the functors are naturally isomorphic.
-/
def HasColimit.isoOfNatIso {F G : J ⥤ C} [HasColimit F] [HasColimit G] (w : F ≅ G) :
    colimit F ≅ colimit G :=
  IsColimit.coconePointsIsoOfNatIso (colimit.isColimit F) (colimit.isColimit G) w


@[reassoc (attr := simp)]
theorem HasColimit.isoOfNatIso_ι_hom {F G : J ⥤ C} [HasColimit F] [HasColimit G] (w : F ≅ G)
    (j : J) : colimit.ι F j ≫ (HasColimit.isoOfNatIso w).hom = w.hom.app j ≫ colimit.ι G j :=
  IsColimit.comp_coconePointsIsoOfNatIso_hom _ _ _ _


@[reassoc (attr := simp)]
theorem HasColimit.isoOfNatIso_ι_inv {F G : J ⥤ C} [HasColimit F] [HasColimit G] (w : F ≅ G)
    (j : J) : colimit.ι G j ≫ (HasColimit.isoOfNatIso w).inv = w.inv.app j ≫ colimit.ι F j :=
  IsColimit.comp_coconePointsIsoOfNatIso_inv _ _ _ _


@[reassoc (attr := simp)]
theorem HasColimit.isoOfNatIso_hom_desc {F G : J ⥤ C} [HasColimit F] [HasColimit G] (t : Cocone G)
    (w : F ≅ G) :
    (HasColimit.isoOfNatIso w).hom ≫ colimit.desc G t =
      colimit.desc F ((Cocones.precompose w.hom).obj _) :=
  IsColimit.coconePointsIsoOfNatIso_hom_desc _ _ _


@[reassoc (attr := simp)]
theorem HasColimit.isoOfNatIso_inv_desc {F G : J ⥤ C} [HasColimit F] [HasColimit G] (t : Cocone F)
    (w : F ≅ G) :
    (HasColimit.isoOfNatIso w).inv ≫ colimit.desc F t =
      colimit.desc G ((Cocones.precompose w.inv).obj _) :=
  IsColimit.coconePointsIsoOfNatIso_inv_desc _ _ _


/-- The colimits of `F : J ⥤ C` and `G : K ⥤ C` are isomorphic,
if there is an equivalence `e : J ≌ K` making the triangle commute up to natural isomorphism.
-/
def HasColimit.isoOfEquivalence {F : J ⥤ C} [HasColimit F] {G : K ⥤ C} [HasColimit G] (e : J ≌ K)
    (w : e.functor ⋙ G ≅ F) : colimit F ≅ colimit G :=
  IsColimit.coconePointsIsoOfEquivalence (colimit.isColimit F) (colimit.isColimit G) e w


@[simp]
theorem HasColimit.isoOfEquivalence_hom_π {F : J ⥤ C} [HasColimit F] {G : K ⥤ C} [HasColimit G]
    (e : J ≌ K) (w : e.functor ⋙ G ≅ F) (j : J) :
    colimit.ι F j ≫ (HasColimit.isoOfEquivalence e w).hom =
      F.map (e.unit.app j) ≫ w.inv.app _ ≫ colimit.ι G _ := by
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasColimit F
    G : CategoryTheory.Functor K C
    inst✝ : CategoryTheory.Limits.HasColimit G
    e : CategoryTheory.Equivalence J K
    w : CategoryTheory.Iso (e.functor.comp G) F
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι F j) …
  -/
  simp [HasColimit.isoOfEquivalence, IsColimit.coconePointsIsoOfEquivalence_inv]
  /-
    🎉 no goals
  -/


@[simp]
theorem HasColimit.isoOfEquivalence_inv_π {F : J ⥤ C} [HasColimit F] {G : K ⥤ C} [HasColimit G]
    (e : J ≌ K) (w : e.functor ⋙ G ≅ F) (k : K) :
    colimit.ι G k ≫ (HasColimit.isoOfEquivalence e w).inv =
      G.map (e.counitInv.app k) ≫ w.hom.app (e.inverse.obj k) ≫ colimit.ι F (e.inverse.obj k) := by
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasColimit F
    G : CategoryTheory.Functor K C
    inst✝ : CategoryTheory.Limits.HasColimit G
    e : CategoryTheory.Equivalence J K
    w : CategoryTheory.Iso (e.functor.comp G) F
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι G k) …
  -/
  simp [HasColimit.isoOfEquivalence, IsColimit.coconePointsIsoOfEquivalence_inv]
  /-
    🎉 no goals
  -/


/-- The canonical morphism from the colimit of `E ⋙ F` to the colimit of `F`.
-/
def colimit.pre : colimit (E ⋙ F) ⟶ colimit F :=
  colimit.desc (E ⋙ F) ((colimit.cocone F).whisker E)


@[reassoc (attr := simp)]
theorem colimit.ι_pre (k : K) : colimit.ι (E ⋙ F) k ≫ colimit.pre F E = colimit.ι F (E.obj k) := by
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasColimit F
    E : CategoryTheory.Functor K J
    inst✝ : CategoryTheory.Limits.HasColimit (E.comp F)
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (E.c …
  -/
  erw [IsColimit.fac]
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasColimit F
    E : CategoryTheory.Functor K J
    inst✝ : CategoryTheory.Limits.HasColimit (E.comp F)
    k : K
    ⊢ Eq ((CategoryTheory.Limits.Cocone.whisker E (CategoryTheory.Limits.colimit.c …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem colimit.ι_inv_pre [IsIso (pre F E)] (k : K) :
    colimit.ι F (E.obj k) ≫ inv (colimit.pre F E) = colimit.ι (E ⋙ F) k := by
  /-
    J : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝² : CategoryTheory.Limits.HasColimit F
    E : CategoryTheory.Functor K J
    inst✝¹ : CategoryTheory.Limits.HasColimit (E.comp F)
    inst✝ : CategoryTheory.IsIso (CategoryTheory.Limits.colimit.pre F E)
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι F (E …
  -/
  simp [IsIso.comp_inv_eq]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem colimit.pre_desc (c : Cocone F) :
    colimit.pre F E ≫ colimit.desc F c = colimit.desc (E ⋙ F) (c.whisker E) := by
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasColimit F
    E : CategoryTheory.Functor K J
    inst✝ : CategoryTheory.Limits.HasColimit (E.comp F)
    c : CategoryTheory.Limits.Cocone F
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.pre F  …
  -/
  ext; rw [← assoc, colimit.ι_pre]; simp
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem colimit.pre_pre [h : HasColimit (D ⋙ E ⋙ F)] :
    haveI : HasColimit ((D ⋙ E) ⋙ F) := h
    colimit.pre (E ⋙ F) D ≫ colimit.pre F E = colimit.pre F (D ⋙ E) := by
  /-
    J : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝² : CategoryTheory.Limits.HasColimit F
    E : CategoryTheory.Functor K J
    inst✝¹ : CategoryTheory.Limits.HasColimit (E.comp F)
    L : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} L
    D : CategoryTheory.Functor L K
    h : CategoryTheory.Limits.HasColimit (D.comp (E.comp F))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.pre (E …
  -/
  ext j
  /-
    case w
    J : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝² : CategoryTheory.Limits.HasColimit F
    E : CategoryTheory.Functor K J
    inst✝¹ : CategoryTheory.Limits.HasColimit (E.comp F)
    L : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} L
    D : CategoryTheory.Functor L K
    h : CategoryTheory.Limits.HasColimit (D.comp (E.comp F))
    j : L
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (D.c …
  -/
  rw [← assoc, colimit.ι_pre, colimit.ι_pre]
  /-
    case w
    J : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝² : CategoryTheory.Limits.HasColimit F
    E : CategoryTheory.Functor K J
    inst✝¹ : CategoryTheory.Limits.HasColimit (E.comp F)
    L : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} L
    D : CategoryTheory.Functor L K
    h : CategoryTheory.Limits.HasColimit (D.comp (E.comp F))
    j : L
    ⊢ Eq (CategoryTheory.Limits.colimit.ι F (E.obj (D.obj j))) (CategoryTheory.Cat …
  -/
  haveI : HasColimit ((D ⋙ E) ⋙ F) := h
  /-
    case w
    J : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝² : CategoryTheory.Limits.HasColimit F
    E : CategoryTheory.Functor K J
    inst✝¹ : CategoryTheory.Limits.HasColimit (E.comp F)
    L : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} L
    D : CategoryTheory.Functor L K
    h : CategoryTheory.Limits.HasColimit (D.comp (E.comp F))
    j : L
    this : CategoryTheory.Limits.HasColimit ((D.comp E).comp F)
    ⊢ Eq (CategoryTheory.Limits.colimit.ι F (E.obj (D.obj j))) (CategoryTheory.Cat …
  -/
  exact (colimit.ι_pre F (D ⋙ E) j).symm
  /-
    🎉 no goals
  -/


/-- -
If we have particular colimit cocones available for `E ⋙ F` and for `F`,
we obtain a formula for `colimit.pre F E`.
-/
theorem colimit.pre_eq (s : ColimitCocone (E ⋙ F)) (t : ColimitCocone F) :
    colimit.pre F E =
      (colimit.isoColimitCocone s).hom ≫
        s.isColimit.desc (t.cocone.whisker E) ≫ (colimit.isoColimitCocone t).inv := by
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasColimit F
    E : CategoryTheory.Functor K J
    inst✝ : CategoryTheory.Limits.HasColimit (E.comp F)
    s : CategoryTheory.Limits.ColimitCocone (E.comp F)
    t : CategoryTheory.Limits.ColimitCocone F
    ⊢ Eq (CategoryTheory.Limits.colimit.pre F E) (CategoryTheory.CategoryStruct.co …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


/-- The canonical morphism from `G` applied to the colimit of `F ⋙ G`
to `G` applied to the colimit of `F`.
-/
def colimit.post : colimit (F ⋙ G) ⟶ G.obj (colimit F) :=
  colimit.desc (F ⋙ G) (G.mapCocone (colimit.cocone F))


@[reassoc (attr := simp)]
theorem colimit.ι_post (j : J) :
    colimit.ι (F ⋙ G) j ≫ colimit.post F G = G.map (colimit.ι F j) := by
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    inst✝¹ : CategoryTheory.Limits.HasColimit F
    G : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.Limits.HasColimit (F.comp G)
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
  -/
  erw [IsColimit.fac]
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    inst✝¹ : CategoryTheory.Limits.HasColimit F
    G : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.Limits.HasColimit (F.comp G)
    j : J
    ⊢ Eq ((G.mapCocone (CategoryTheory.Limits.colimit.cocone F)).ι.app j) (G.map ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem colimit.post_desc (c : Cocone F) :
    colimit.post F G ≫ G.map (colimit.desc F c) = colimit.desc (F ⋙ G) (G.mapCocone c) := by
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    inst✝¹ : CategoryTheory.Limits.HasColimit F
    G : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.Limits.HasColimit (F.comp G)
    c : CategoryTheory.Limits.Cocone F
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.post F …
  -/
  ext
  /-
    case w
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    inst✝¹ : CategoryTheory.Limits.HasColimit F
    G : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.Limits.HasColimit (F.comp G)
    c : CategoryTheory.Limits.Cocone F
    j✝ : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
  -/
  rw [← assoc, colimit.ι_post, ← G.map_comp, colimit.ι_desc, colimit.ι_desc]
  /-
    case w
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    inst✝¹ : CategoryTheory.Limits.HasColimit F
    G : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.Limits.HasColimit (F.comp G)
    c : CategoryTheory.Limits.Cocone F
    j✝ : J
    ⊢ Eq (G.map (c.ι.app j✝)) ((G.mapCocone c).ι.app j✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem colimit.post_post {E : Type u''} [Category.{v''} E] (H : D ⥤ E)
    -- H G (colimit F) ⟶ H (colimit (F ⋙ G)) ⟶ colimit ((F ⋙ G) ⋙ H) equals
    -- H G (colimit F) ⟶ colimit (F ⋙ (G ⋙ H))
    [h : HasColimit ((F ⋙ G) ⋙ H)] : haveI : HasColimit (F ⋙ G ⋙ H) := h
    colimit.post (F ⋙ G) H ≫ H.map (colimit.post F G) = colimit.post F (G ⋙ H) := by
  /-
    J : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    inst✝² : CategoryTheory.Limits.HasColimit F
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasColimit (F.comp G)
    E : Type u''
    inst✝ : CategoryTheory.Category.{v'', u''} E
    H : CategoryTheory.Functor D E
    h : CategoryTheory.Limits.HasColimit ((F.comp G).comp H)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.post ( …
  -/
  ext j
  /-
    case w
    J : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    inst✝² : CategoryTheory.Limits.HasColimit F
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasColimit (F.comp G)
    E : Type u''
    inst✝ : CategoryTheory.Category.{v'', u''} E
    H : CategoryTheory.Functor D E
    h : CategoryTheory.Limits.HasColimit ((F.comp G).comp H)
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((F. …
  -/
  rw [← assoc, colimit.ι_post, ← H.map_comp, colimit.ι_post]
  /-
    case w
    J : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    inst✝² : CategoryTheory.Limits.HasColimit F
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasColimit (F.comp G)
    E : Type u''
    inst✝ : CategoryTheory.Category.{v'', u''} E
    H : CategoryTheory.Functor D E
    h : CategoryTheory.Limits.HasColimit ((F.comp G).comp H)
    j : J
    ⊢ Eq (H.map (G.map (CategoryTheory.Limits.colimit.ι F j))) (CategoryTheory.Cat …
  -/
  haveI : HasColimit (F ⋙ G ⋙ H) := h
  /-
    case w
    J : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    inst✝² : CategoryTheory.Limits.HasColimit F
    G : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasColimit (F.comp G)
    E : Type u''
    inst✝ : CategoryTheory.Category.{v'', u''} E
    H : CategoryTheory.Functor D E
    h : CategoryTheory.Limits.HasColimit ((F.comp G).comp H)
    j : J
    this : CategoryTheory.Limits.HasColimit (F.comp (G.comp H))
    ⊢ Eq (H.map (G.map (CategoryTheory.Limits.colimit.ι F j))) (CategoryTheory.Cat …
  -/
  exact (colimit.ι_post F (G ⋙ H) j).symm
  /-
    🎉 no goals
  -/


theorem colimit.pre_post {D : Type u'} [Category.{v'} D] (E : K ⥤ J) (F : J ⥤ C) (G : C ⥤ D)
    [HasColimit F] [HasColimit (E ⋙ F)] [HasColimit (F ⋙ G)] [h : HasColimit ((E ⋙ F) ⋙ G)] :
    -- G (colimit F) ⟶ G (colimit (E ⋙ F)) ⟶ colimit ((E ⋙ F) ⋙ G) vs
    -- G (colimit F) ⟶ colimit F ⋙ G ⟶ colimit (E ⋙ (F ⋙ G)) or
    haveI : HasColimit (E ⋙ F ⋙ G) := h
    colimit.post (E ⋙ F) G ≫ G.map (colimit.pre F E) =
      colimit.pre (F ⋙ G) E ≫ colimit.post F G := by
  /-
    J : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    E : CategoryTheory.Functor K J
    F : CategoryTheory.Functor J C
    G : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.Limits.HasColimit F
    inst✝¹ : CategoryTheory.Limits.HasColimit (E.comp F)
    inst✝ : CategoryTheory.Limits.HasColimit (F.comp G)
    h : CategoryTheory.Limits.HasColimit ((E.comp F).comp G)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.post ( …
  -/
  ext j
  /-
    case w
    J : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    E : CategoryTheory.Functor K J
    F : CategoryTheory.Functor J C
    G : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.Limits.HasColimit F
    inst✝¹ : CategoryTheory.Limits.HasColimit (E.comp F)
    inst✝ : CategoryTheory.Limits.HasColimit (F.comp G)
    h : CategoryTheory.Limits.HasColimit ((E.comp F).comp G)
    j : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((E. …
  -/
  rw [← assoc, colimit.ι_post, ← G.map_comp, colimit.ι_pre, ← assoc]
  /-
    case w
    J : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    E : CategoryTheory.Functor K J
    F : CategoryTheory.Functor J C
    G : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.Limits.HasColimit F
    inst✝¹ : CategoryTheory.Limits.HasColimit (E.comp F)
    inst✝ : CategoryTheory.Limits.HasColimit (F.comp G)
    h : CategoryTheory.Limits.HasColimit ((E.comp F).comp G)
    j : K
    ⊢ Eq (G.map (CategoryTheory.Limits.colimit.ι F (E.obj j))) (CategoryTheory.Cat …
  -/
  haveI : HasColimit (E ⋙ F ⋙ G) := h
  /-
    case w
    J : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    E : CategoryTheory.Functor K J
    F : CategoryTheory.Functor J C
    G : CategoryTheory.Functor C D
    inst✝² : CategoryTheory.Limits.HasColimit F
    inst✝¹ : CategoryTheory.Limits.HasColimit (E.comp F)
    inst✝ : CategoryTheory.Limits.HasColimit (F.comp G)
    h : CategoryTheory.Limits.HasColimit ((E.comp F).comp G)
    j : K
    this : CategoryTheory.Limits.HasColimit (E.comp (F.comp G))
    ⊢ Eq (G.map (CategoryTheory.Limits.colimit.ι F (E.obj j))) (CategoryTheory.Cat …
  -/
  erw [colimit.ι_pre (F ⋙ G) E j, colimit.ι_post]
  /-
    🎉 no goals
  -/


instance hasColimit_equivalence_comp (e : K ≌ J) [HasColimit F] : HasColimit (e.functor ⋙ F) :=
  HasColimit.mk
    { cocone := Cocone.whisker e.functor (colimit.cocone F)
      isColimit := IsColimit.whiskerEquivalence (colimit.isColimit F) e }


/-- If a `E ⋙ F` has a colimit, and `E` is an equivalence, we can construct a colimit of `F`.
-/
theorem hasColimit_of_equivalence_comp (e : K ≌ J) [HasColimit (e.functor ⋙ F)] : HasColimit F := by
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    e : CategoryTheory.Equivalence K J
    inst✝ : CategoryTheory.Limits.HasColimit (e.functor.comp F)
    ⊢ CategoryTheory.Limits.HasColimit F
  -/
  haveI : HasColimit (e.inverse ⋙ e.functor ⋙ F) := Limits.hasColimit_equivalence_comp e.symm
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    e : CategoryTheory.Equivalence K J
    inst✝ : CategoryTheory.Limits.HasColimit (e.functor.comp F)
    this : CategoryTheory.Limits.HasColimit (e.inverse.comp (e.functor.comp F))
    ⊢ CategoryTheory.Limits.HasColimit F
  -/
  apply hasColimitOfIso (e.invFunIdAssoc F).symm
  /-
    🎉 no goals
  -/


/-- `colimit F` is functorial in `F`, when `C` has all colimits of shape `J`. -/
@[simps] -- Porting note: simps on all fields now
def colim : (J ⥤ C) ⥤ C where
  obj F := colimit F
  map α := colimMap α


@[reassoc]
                                                                                            /-
                                                                                              J : Type u₁
                                                                                              inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                                                                              C : Type u
                                                                                              inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                                              F : CategoryTheory.Functor J C
                                                                                              inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
                                                                                              G : CategoryTheory.Functor J C
                                                                                              α : Quiver.Hom F G
                                                                                              j : J
                                                                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι F j) …
                                                                                            -/
theorem colimit.ι_map (j : J) : colimit.ι F j ≫ colim.map α = α.app j ≫ colimit.ι G j := by simp
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


@[reassoc (attr := simp)]
theorem colimit.map_desc (c : Cocone G) :
    colimMap α ≫ colimit.desc G c = colimit.desc F ((Cocones.precompose α).obj c) := by
  /-
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    c : CategoryTheory.Limits.Cocone G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimMap α) (C …
  -/
  ext j
  /-
    case w
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    c : CategoryTheory.Limits.Cocone G
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι F j) …
  -/
  simp [← assoc, colimit.ι_map, assoc, colimit.ι_desc, colimit.ι_desc]
  /-
    🎉 no goals
  -/


theorem colimit.pre_map [HasColimitsOfShape K C] (E : K ⥤ J) :
    colimit.pre F E ≫ colim.map α = colim.map (whiskerLeft E α) ≫ colimit.pre G E := by
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J C
    G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape K C
    E : CategoryTheory.Functor K J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.pre F  …
  -/
  ext
  /-
    case w
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J C
    G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape K C
    E : CategoryTheory.Functor K J
    j✝ : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (E.c …
  -/
  rw [← assoc, colimit.ι_pre, colimit.ι_map, ← assoc, colimit.ι_map, assoc, colimit.ι_pre]
  /-
    case w
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J C
    G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape K C
    E : CategoryTheory.Functor K J
    j✝ : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.app (E.obj j✝)) (CategoryTheory.Li …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem colimit.pre_map' [HasColimitsOfShape K C] (F : J ⥤ C) {E₁ E₂ : K ⥤ J} (α : E₁ ⟶ E₂) :
    colimit.pre F E₁ = colim.map (whiskerRight α F) ≫ colimit.pre F E₂ := by
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J C
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape K C
    F : CategoryTheory.Functor J C
    E₁ E₂ : CategoryTheory.Functor K J
    α : Quiver.Hom E₁ E₂
    ⊢ Eq (CategoryTheory.Limits.colimit.pre F E₁) (CategoryTheory.CategoryStruct.c …
  -/
  ext1
  /-
    case w
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J C
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape K C
    F : CategoryTheory.Functor J C
    E₁ E₂ : CategoryTheory.Functor K J
    α : Quiver.Hom E₁ E₂
    j✝ : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (E₁. …
  -/
  simp [← assoc, assoc]
  /-
    🎉 no goals
  -/


theorem colimit.pre_id (F : J ⥤ C) :
                                                                     /-
                                                                       J : Type u₁
                                                                       inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                                                       C : Type u
                                                                       inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                       inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
                                                                       F : CategoryTheory.Functor J C
                                                                       ⊢ Eq (CategoryTheory.Limits.colimit.pre F (CategoryTheory.Functor.id J)) (Cate …
                                                                     -/
    colimit.pre F (𝟭 _) = colim.map (Functor.leftUnitor F).hom := by aesop_cat
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem colimit.map_post {D : Type u'} [Category.{v'} D] [HasColimitsOfShape J D]
    (H : C ⥤ D) :/- H (colimit F) ⟶ H (colimit G) ⟶ colimit (G ⋙ H) vs
             H (colimit F) ⟶ colimit (F ⋙ H) ⟶ colimit (G ⋙ H) -/
          colimit.post
          F H ≫
        H.map (colim.map α) =
      colim.map (whiskerRight α H) ≫ colimit.post G H := by
  /-
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape J C
    G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    D : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} D
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J D
    H : CategoryTheory.Functor C D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.post F …
  -/
  ext
  /-
    case w
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape J C
    G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    D : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} D
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J D
    H : CategoryTheory.Functor C D
    j✝ : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
  -/
  rw [← assoc, colimit.ι_post, ← H.map_comp, colimit.ι_map, H.map_comp]
  /-
    case w
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape J C
    G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    D : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} D
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J D
    H : CategoryTheory.Functor C D
    j✝ : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (H.map (α.app j✝)) (H.map (CategoryTh …
  -/
  rw [← assoc, colimit.ι_map, assoc, colimit.ι_post]
  /-
    case w
    J : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape J C
    G : CategoryTheory.Functor J C
    α : Quiver.Hom F G
    D : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} D
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J D
    H : CategoryTheory.Functor C D
    j✝ : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (H.map (α.app j✝)) (H.map (CategoryTh …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The isomorphism between
morphisms from the cone point of the colimit cocone for `F` to `W`
and cocones over `F` with cone point `W`
is natural in `F`.
-/
def colimCoyoneda : colim.op ⋙ coyoneda ⋙ (whiskeringRight _ _ _).obj uliftFunctor.{u₁}
    ≅ CategoryTheory.cocones J C :=
                               /-
                                 J : Type u₁
                                 inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                 K : Type u₂
                                 inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                 C : Type u
                                 inst✝¹ : CategoryTheory.Category.{v, u} C
                                 F✝ : CategoryTheory.Functor J C
                                 inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
                                 G : CategoryTheory.Functor J C
                                 α : Quiver.Hom F✝ G
                                 F : Opposite (CategoryTheory.Functor J C)
                                 ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun F => NatIso.ofComponents fun W => colimit.homIso (unop F) W
  /-
    🎉 no goals
  -/


/-- The colimit functor and constant functor are adjoint to each other
-/
def colimConstAdj : (colim : (J ⥤ C) ⥤ C) ⊣ const J := Adjunction.mk' {
  homEquiv := fun f c ↦
    { toFun := fun g =>
        { app := fun _ => colimit.ι _ _ ≫ g }
      invFun := fun g => colimit.desc _ ⟨_, g⟩
                     /-
                       J : Type u₁
                       inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                       K : Type u₂
                       inst✝² : CategoryTheory.Category.{v₂, u₂} K
                       C : Type u
                       inst✝¹ : CategoryTheory.Category.{v, u} C
                       F : CategoryTheory.Functor J C
                       inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
                       G : CategoryTheory.Functor J C
                       α : Quiver.Hom F G
                       f : CategoryTheory.Functor J C
                       c : C
                       ⊢ Function.LeftInverse (fun g => CategoryTheory.Limits.colimit.desc f { pt :=  …
                     -/
      left_inv := by aesop_cat
                     /-
                       🎉 no goals
                     -/
                      /-
                        J : Type u₁
                        inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                        K : Type u₂
                        inst✝² : CategoryTheory.Category.{v₂, u₂} K
                        C : Type u
                        inst✝¹ : CategoryTheory.Category.{v, u} C
                        F : CategoryTheory.Functor J C
                        inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
                        G : CategoryTheory.Functor J C
                        α : Quiver.Hom F G
                        f : CategoryTheory.Functor J C
                        c : C
                        ⊢ Function.RightInverse (fun g => CategoryTheory.Limits.colimit.desc f { pt := …
                      -/
      right_inv := by aesop_cat }
                      /-
                        🎉 no goals
                      -/
  unit := { app := fun g => { app := colimit.ι _ } }
  counit := { app := fun _ => colimit.desc _ ⟨_, 𝟙 _⟩ } }


instance : IsLeftAdjoint (colim : (J ⥤ C) ⥤ C) :=
  ⟨_, ⟨colimConstAdj⟩⟩


instance colimMap_epi' {F G : J ⥤ C} [HasColimitsOfShape J C] (α : F ⟶ G) [Epi α] :
    Epi (colimMap α) :=
  (colim : (J ⥤ C) ⥤ C).map_epi α


instance colimMap_epi {F G : J ⥤ C} [HasColimit F] [HasColimit G] (α : F ⟶ G) [∀ j, Epi (α.app j)] :
    Epi (colimMap α) :=
  ⟨fun {Z} u v h =>
                                                            /-
                                                              J : Type u₁
                                                              inst✝⁵ : CategoryTheory.Category.{v₁, u₁} J
                                                              K : Type u₂
                                                              inst✝⁴ : CategoryTheory.Category.{v₂, u₂} K
                                                              C : Type u
                                                              inst✝³ : CategoryTheory.Category.{v, u} C
                                                              F✝ F G : CategoryTheory.Functor J C
                                                              inst✝² : CategoryTheory.Limits.HasColimit F
                                                              inst✝¹ : CategoryTheory.Limits.HasColimit G
                                                              α : Quiver.Hom F G
                                                              inst✝ : ∀ (j : J), CategoryTheory.Epi (α.app j)
                                                              Z : C
                                                              u v : Quiver.Hom (CategoryTheory.Limits.colimit G) Z
                                                              h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimMap α)  …
                                                              j : J
                                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.app j) (CategoryTheory.CategoryStr …
                                                            -/
    colimit.hom_ext fun j => (cancel_epi (α.app j)).1 <| by simpa using colimit.ι _ j ≫= h⟩
                                                            /-
                                                              🎉 no goals
                                                            -/


/-- We can transport colimits of shape `J` along an equivalence `J ≌ J'`.
-/
theorem hasColimitsOfShape_of_equivalence {J' : Type u₂} [Category.{v₂} J'] (e : J ≌ J')
    [HasColimitsOfShape J C] : HasColimitsOfShape J' C := by
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J' : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} J'
    e : CategoryTheory.Equivalence J J'
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    ⊢ CategoryTheory.Limits.HasColimitsOfShape J' C
  -/
  constructor
  /-
    case has_colimit
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J' : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} J'
    e : CategoryTheory.Equivalence J J'
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    ⊢ autoParam (∀ (F : CategoryTheory.Functor J' C), CategoryTheory.Limits.HasCol …
  -/
  intro F
  /-
    case has_colimit
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J' : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} J'
    e : CategoryTheory.Equivalence J J'
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    F : CategoryTheory.Functor J' C
    ⊢ CategoryTheory.Limits.HasColimit F
  -/
  apply hasColimit_of_equivalence_comp e
  /-
    🎉 no goals
  -/


/-- A category that has larger colimits also has smaller colimits. -/
theorem hasColimitsOfSizeOfUnivLE [UnivLE.{v₂, v₁}] [UnivLE.{u₂, u₁}]
    [HasColimitsOfSize.{v₁, u₁} C] : HasColimitsOfSize.{v₂, u₂} C where
  has_colimits_of_shape J {_} := hasColimitsOfShape_of_equivalence
    ((ShrinkHoms.equivalence J).trans <| Shrink.equivalence _).symm


/-- `hasColimitsOfSizeShrink.{v u} C` tries to obtain `HasColimitsOfSize.{v u} C`
from some other `HasColimitsOfSize C`.
-/
theorem hasColimitsOfSizeShrink [HasColimitsOfSize.{max v₁ v₂, max u₁ u₂} C] :
    HasColimitsOfSize.{v₁, u₁} C := hasColimitsOfSizeOfUnivLE.{max v₁ v₂, max u₁ u₂} C


instance (priority := 100) hasSmallestColimitsOfHasColimits [HasColimits C] :
    HasColimitsOfSize.{0, 0} C :=
  hasColimitsOfSizeShrink.{0, 0} C


/-- If `t : Cone F` is a limit cone, then `t.op : Cocone F.op` is a colimit cocone.
-/
def IsLimit.op {t : Cone F} (P : IsLimit t) : IsColimit t.op where
  desc s := (P.lift s.unop).op
  fac s j := congrArg Quiver.Hom.op (P.fac s.unop (unop j))
  uniq s m w := by
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cone F
      P : CategoryTheory.Limits.IsLimit t
      s : CategoryTheory.Limits.Cocone F.op
      m : Quiver.Hom t.op.pt s.pt
      w : ∀ (j : Opposite J), Eq (CategoryTheory.CategoryStruct.comp (t.op.ι.app j)  …
      ⊢ Eq m ((fun s => (P.lift s.unop).op) s)
    -/
    dsimp
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cone F
      P : CategoryTheory.Limits.IsLimit t
      s : CategoryTheory.Limits.Cocone F.op
      m : Quiver.Hom t.op.pt s.pt
      w : ∀ (j : Opposite J), Eq (CategoryTheory.CategoryStruct.comp (t.op.ι.app j)  …
      ⊢ Eq m (P.lift s.unop).op
    -/
    rw [← P.uniq s.unop m.unop]
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor J C
        t : CategoryTheory.Limits.Cone F
        P : CategoryTheory.Limits.IsLimit t
        s : CategoryTheory.Limits.Cocone F.op
        m : Quiver.Hom t.op.pt s.pt
        w : ∀ (j : Opposite J), Eq (CategoryTheory.CategoryStruct.comp (t.op.ι.app j)  …
        ⊢ Eq m m.unop.op
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor J C
        t : CategoryTheory.Limits.Cone F
        P : CategoryTheory.Limits.IsLimit t
        s : CategoryTheory.Limits.Cocone F.op
        m : Quiver.Hom t.op.pt s.pt
        w : ∀ (j : Opposite J), Eq (CategoryTheory.CategoryStruct.comp (t.op.ι.app j)  …
        ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m.unop (t.π.app j)) (s.uno …
      -/
    · dsimp
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor J C
        t : CategoryTheory.Limits.Cone F
        P : CategoryTheory.Limits.IsLimit t
        s : CategoryTheory.Limits.Cocone F.op
        m : Quiver.Hom t.op.pt s.pt
        w : ∀ (j : Opposite J), Eq (CategoryTheory.CategoryStruct.comp (t.op.ι.app j)  …
        ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m.unop (t.π.app j)) (s.ι.a …
      -/
      intro j
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor J C
        t : CategoryTheory.Limits.Cone F
        P : CategoryTheory.Limits.IsLimit t
        s : CategoryTheory.Limits.Cocone F.op
        m : Quiver.Hom t.op.pt s.pt
        w : ∀ (j : Opposite J), Eq (CategoryTheory.CategoryStruct.comp (t.op.ι.app j)  …
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m.unop (t.π.app j)) (s.ι.app { unop : …
      -/
      rw [← w]
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor J C
        t : CategoryTheory.Limits.Cone F
        P : CategoryTheory.Limits.IsLimit t
        s : CategoryTheory.Limits.Cocone F.op
        m : Quiver.Hom t.op.pt s.pt
        w : ∀ (j : Opposite J), Eq (CategoryTheory.CategoryStruct.comp (t.op.ι.app j)  …
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m.unop (t.π.app j)) (CategoryTheory.C …
      -/
      rfl
      /-
        🎉 no goals
      -/


/-- If `t : Cocone F` is a colimit cocone, then `t.op : Cone F.op` is a limit cone.
-/
def IsColimit.op {t : Cocone F} (P : IsColimit t) : IsLimit t.op where
  lift s := (P.desc s.unop).op
  fac s j := congrArg Quiver.Hom.op (P.fac s.unop (unop j))
  uniq s m w := by
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cocone F
      P : CategoryTheory.Limits.IsColimit t
      s : CategoryTheory.Limits.Cone F.op
      m : Quiver.Hom s.pt t.op.pt
      w : ∀ (j : Opposite J), Eq (CategoryTheory.CategoryStruct.comp m (t.op.π.app j …
      ⊢ Eq m ((fun s => (P.desc s.unop).op) s)
    -/
    dsimp
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cocone F
      P : CategoryTheory.Limits.IsColimit t
      s : CategoryTheory.Limits.Cone F.op
      m : Quiver.Hom s.pt t.op.pt
      w : ∀ (j : Opposite J), Eq (CategoryTheory.CategoryStruct.comp m (t.op.π.app j …
      ⊢ Eq m (P.desc s.unop).op
    -/
    rw [← P.uniq s.unop m.unop]
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor J C
        t : CategoryTheory.Limits.Cocone F
        P : CategoryTheory.Limits.IsColimit t
        s : CategoryTheory.Limits.Cone F.op
        m : Quiver.Hom s.pt t.op.pt
        w : ∀ (j : Opposite J), Eq (CategoryTheory.CategoryStruct.comp m (t.op.π.app j …
        ⊢ Eq m m.unop.op
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor J C
        t : CategoryTheory.Limits.Cocone F
        P : CategoryTheory.Limits.IsColimit t
        s : CategoryTheory.Limits.Cone F.op
        m : Quiver.Hom s.pt t.op.pt
        w : ∀ (j : Opposite J), Eq (CategoryTheory.CategoryStruct.comp m (t.op.π.app j …
        ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (t.ι.app j) m.unop) (s.uno …
      -/
    · dsimp
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor J C
        t : CategoryTheory.Limits.Cocone F
        P : CategoryTheory.Limits.IsColimit t
        s : CategoryTheory.Limits.Cone F.op
        m : Quiver.Hom s.pt t.op.pt
        w : ∀ (j : Opposite J), Eq (CategoryTheory.CategoryStruct.comp m (t.op.π.app j …
        ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (t.ι.app j) m.unop) (s.π.a …
      -/
      intro j
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor J C
        t : CategoryTheory.Limits.Cocone F
        P : CategoryTheory.Limits.IsColimit t
        s : CategoryTheory.Limits.Cone F.op
        m : Quiver.Hom s.pt t.op.pt
        w : ∀ (j : Opposite J), Eq (CategoryTheory.CategoryStruct.comp m (t.op.π.app j …
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (t.ι.app j) m.unop) (s.π.app { unop : …
      -/
      rw [← w]
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor J C
        t : CategoryTheory.Limits.Cocone F
        P : CategoryTheory.Limits.IsColimit t
        s : CategoryTheory.Limits.Cone F.op
        m : Quiver.Hom s.pt t.op.pt
        w : ∀ (j : Opposite J), Eq (CategoryTheory.CategoryStruct.comp m (t.op.π.app j …
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (t.ι.app j) m.unop) (CategoryTheory.C …
      -/
      rfl
      /-
        🎉 no goals
      -/


/-- If `t : Cone F.op` is a limit cone, then `t.unop : Cocone F` is a colimit cocone.
-/
def IsLimit.unop {t : Cone F.op} (P : IsLimit t) : IsColimit t.unop where
  desc s := (P.lift s.op).unop
  fac s j := congrArg Quiver.Hom.unop (P.fac s.op (.op j))
  uniq s m w := by
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cone F.op
      P : CategoryTheory.Limits.IsLimit t
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom t.unop.pt s.pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (t.unop.ι.app j) m) (s.ι …
      ⊢ Eq m ((fun s => (P.lift s.op).unop) s)
    -/
    dsimp
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cone F.op
      P : CategoryTheory.Limits.IsLimit t
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom t.unop.pt s.pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (t.unop.ι.app j) m) (s.ι …
      ⊢ Eq m (P.lift s.op).unop
    -/
    rw [← P.uniq s.op m.op]
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor J C
        t : CategoryTheory.Limits.Cone F.op
        P : CategoryTheory.Limits.IsLimit t
        s : CategoryTheory.Limits.Cocone F
        m : Quiver.Hom t.unop.pt s.pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (t.unop.ι.app j) m) (s.ι …
        ⊢ Eq m m.op.unop
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor J C
        t : CategoryTheory.Limits.Cone F.op
        P : CategoryTheory.Limits.IsLimit t
        s : CategoryTheory.Limits.Cocone F
        m : Quiver.Hom t.unop.pt s.pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (t.unop.ι.app j) m) (s.ι …
        ⊢ ∀ (j : Opposite J), Eq (CategoryTheory.CategoryStruct.comp m.op (t.π.app j)) …
      -/
    · dsimp
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor J C
        t : CategoryTheory.Limits.Cone F.op
        P : CategoryTheory.Limits.IsLimit t
        s : CategoryTheory.Limits.Cocone F
        m : Quiver.Hom t.unop.pt s.pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (t.unop.ι.app j) m) (s.ι …
        ⊢ ∀ (j : Opposite J), Eq (CategoryTheory.CategoryStruct.comp m.op (t.π.app j)) …
      -/
      intro j
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor J C
        t : CategoryTheory.Limits.Cone F.op
        P : CategoryTheory.Limits.IsLimit t
        s : CategoryTheory.Limits.Cocone F
        m : Quiver.Hom t.unop.pt s.pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (t.unop.ι.app j) m) (s.ι …
        j : Opposite J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m.op (t.π.app j)) (s.ι.app (Opposite. …
      -/
      rw [← w]
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor J C
        t : CategoryTheory.Limits.Cone F.op
        P : CategoryTheory.Limits.IsLimit t
        s : CategoryTheory.Limits.Cocone F
        m : Quiver.Hom t.unop.pt s.pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (t.unop.ι.app j) m) (s.ι …
        j : Opposite J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m.op (t.π.app j)) (CategoryTheory.Cat …
      -/
      rfl
      /-
        🎉 no goals
      -/


/-- If `t : Cocone F.op` is a colimit cocone, then `t.unop : Cone F` is a limit cone.
-/
def IsColimit.unop {t : Cocone F.op} (P : IsColimit t) : IsLimit t.unop where
  lift s := (P.desc s.op).unop
  fac s j := congrArg Quiver.Hom.unop (P.fac s.op (.op j))
  uniq s m w := by
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cocone F.op
      P : CategoryTheory.Limits.IsColimit t
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt t.unop.pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (t.unop.π.app j)) (s.π …
      ⊢ Eq m ((fun s => (P.desc s.op).unop) s)
    -/
    dsimp
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cocone F.op
      P : CategoryTheory.Limits.IsColimit t
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt t.unop.pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (t.unop.π.app j)) (s.π …
      ⊢ Eq m (P.desc s.op).unop
    -/
    rw [← P.uniq s.op m.op]
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor J C
        t : CategoryTheory.Limits.Cocone F.op
        P : CategoryTheory.Limits.IsColimit t
        s : CategoryTheory.Limits.Cone F
        m : Quiver.Hom s.pt t.unop.pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (t.unop.π.app j)) (s.π …
        ⊢ Eq m m.op.unop
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor J C
        t : CategoryTheory.Limits.Cocone F.op
        P : CategoryTheory.Limits.IsColimit t
        s : CategoryTheory.Limits.Cone F
        m : Quiver.Hom s.pt t.unop.pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (t.unop.π.app j)) (s.π …
        ⊢ ∀ (j : Opposite J), Eq (CategoryTheory.CategoryStruct.comp (t.ι.app j) m.op) …
      -/
    · dsimp
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor J C
        t : CategoryTheory.Limits.Cocone F.op
        P : CategoryTheory.Limits.IsColimit t
        s : CategoryTheory.Limits.Cone F
        m : Quiver.Hom s.pt t.unop.pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (t.unop.π.app j)) (s.π …
        ⊢ ∀ (j : Opposite J), Eq (CategoryTheory.CategoryStruct.comp (t.ι.app j) m.op) …
      -/
      intro j
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor J C
        t : CategoryTheory.Limits.Cocone F.op
        P : CategoryTheory.Limits.IsColimit t
        s : CategoryTheory.Limits.Cone F
        m : Quiver.Hom s.pt t.unop.pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (t.unop.π.app j)) (s.π …
        j : Opposite J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (t.ι.app j) m.op) (s.π.app (Opposite. …
      -/
      rw [← w]
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F : CategoryTheory.Functor J C
        t : CategoryTheory.Limits.Cocone F.op
        P : CategoryTheory.Limits.IsColimit t
        s : CategoryTheory.Limits.Cone F
        m : Quiver.Hom s.pt t.unop.pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (t.unop.π.app j)) (s.π …
        j : Opposite J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (t.ι.app j) m.op) (CategoryTheory.Cat …
      -/
      rfl
      /-
        🎉 no goals
      -/


/-- If `t.op : Cocone F.op` is a colimit cocone, then `t : Cone F` is a limit cone. -/
def isLimitOfOp {t : Cone F} (P : IsColimit t.op) : IsLimit t :=
  P.unop


/-- If `t.op : Cone F.op` is a limit cone, then `t : Cocone F` is a colimit cocone. -/
def isColimitOfOp {t : Cocone F} (P : IsLimit t.op) : IsColimit t :=
  P.unop


/-- If `t.unop : Cocone F` is a colimit cocone, then `t : Cone F.op` is a limit cone.-/
def isLimitOfUnop {t : Cone F.op} (P : IsColimit t.unop) : IsLimit t :=
  P.op


/-- If `t.unop : Cone F` is a limit cone, then `t : Cocone F.op` is a colimit cocone. -/
def isColimitOfUnop {t : Cocone F.op} (P : IsLimit t.unop) : IsColimit t :=
  P.op


/-- `t : Cone F` is a limit cone if and only if `t.op : Cocone F.op` is a colimit cocone.
-/
def isLimitEquivIsColimitOp {t : Cone F} : IsLimit t ≃ IsColimit t.op :=
  equivOfSubsingletonOfSubsingleton IsLimit.op isLimitOfOp


/-- `t : Cocone F` is a colimit cocone if and only if `t.op : Cone F.op` is a limit cone.
-/
def isColimitEquivIsLimitOp {t : Cocone F} : IsColimit t ≃ IsLimit t.op :=
  equivOfSubsingletonOfSubsingleton IsColimit.op isColimitOfOp



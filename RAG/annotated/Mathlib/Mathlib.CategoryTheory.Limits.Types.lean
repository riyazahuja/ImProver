/-- Given a section of a functor F into `Type*`,
  construct a cone over F with `PUnit` as the cone point. -/
def coneOfSection {s} (hs : s ∈ F.sections) : Cone F where
  pt := PUnit
  π :=
  { app := fun j _ ↦ s j,
                                 /-
                                   J : Type v
                                   inst✝ : CategoryTheory.Category.{w, v} J
                                   F : CategoryTheory.Functor J (Type u)
                                   s : (j : J) → F.obj j
                                   hs : Membership.mem F.sections s
                                   i j : J
                                   f : Quiver.Hom i j
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
                                 -/
    naturality := fun i j f ↦ by ext; exact (hs f).symm }
                                      /-
                                        🎉 no goals
                                      -/


/-- Given a cone over a functor F into `Type*` and an element in the cone point,
  construct a section of F. -/
def sectionOfCone (c : Cone F) (x : c.pt) : F.sections :=
  ⟨fun j ↦ c.π.app j x, fun f ↦ congr_fun (c.π.naturality f).symm x⟩


theorem isLimit_iff (c : Cone F) :
    Nonempty (IsLimit c) ↔ ∀ s ∈ F.sections, ∃! x : c.pt, ∀ j, c.π.app j x = s j := by
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    c : CategoryTheory.Limits.Cone F
    ⊢ Iff (Nonempty (CategoryTheory.Limits.IsLimit c)) (∀ (s : (j : J) → F.obj j), …
  -/
  refine ⟨fun ⟨t⟩ s hs ↦ ?_, fun h ↦ ⟨?_⟩⟩
    /-
      case refine_1
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J (Type u)
      c : CategoryTheory.Limits.Cone F
      x✝ : Nonempty (CategoryTheory.Limits.IsLimit c)
      s : (j : J) → F.obj j
      hs : Membership.mem F.sections s
      t : CategoryTheory.Limits.IsLimit c
      ⊢ ExistsUnique fun x => ∀ (j : J), Eq (c.π.app j x) (s j)
    -/
  · let cs := coneOfSection hs
    exact ⟨t.lift cs ⟨⟩, fun j ↦ congr_fun (t.fac cs j) ⟨⟩,
      fun x hx ↦ congr_fun (t.uniq cs (fun _ ↦ x) fun j ↦ funext fun _ ↦ hx j) ⟨⟩⟩
    /-
      case refine_2
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J (Type u)
      c : CategoryTheory.Limits.Cone F
      h : ∀ (s : (j : J) → F.obj j), Membership.mem F.sections s → ExistsUnique fun  …
      ⊢ CategoryTheory.Limits.IsLimit c
    -/
  · choose x hx using fun c y ↦ h _ (sectionOfCone c y).2
    exact ⟨x, fun c j ↦ funext fun y ↦ (hx c y).1 j,
      fun c f hf ↦ funext fun y ↦ (hx c y).2 (f y) (fun j ↦ congr_fun (hf j) y)⟩


theorem isLimit_iff_bijective_sectionOfCone (c : Cone F) :
    Nonempty (IsLimit c) ↔ (Types.sectionOfCone c).Bijective := by
  simp_rw [isLimit_iff, Function.bijective_iff_existsUnique, Subtype.forall, F.sections_ext_iff,
    sectionOfCone]


/-- The equivalence between a limiting cone of `F` in `Type u` and the "concrete" definition as the
  sections of `F`. -/
noncomputable def isLimitEquivSections {c : Cone F} (t : IsLimit c) :
    c.pt ≃ F.sections where
  toFun := sectionOfCone c
  invFun s := t.lift (coneOfSection s.2) ⟨⟩
  left_inv x := (congr_fun (t.uniq (coneOfSection _) (fun _ ↦ x) fun _ ↦ rfl) ⟨⟩).symm
  right_inv s := Subtype.ext (funext fun j ↦ congr_fun (t.fac (coneOfSection s.2) j) ⟨⟩)


@[simp]
theorem isLimitEquivSections_apply {c : Cone F} (t : IsLimit c) (j : J)
    (x : c.pt) : (isLimitEquivSections t x : ∀ j, F.obj j) j = c.π.app j x := rfl


@[simp]
theorem isLimitEquivSections_symm_apply {c : Cone F} (t : IsLimit c)
    (x : F.sections) (j : J) :
    c.π.app j ((isLimitEquivSections t).symm x) = (x : ∀ j, F.obj j) j := by
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    c : CategoryTheory.Limits.Cone F
    t : CategoryTheory.Limits.IsLimit c
    x : ↑F.sections
    j : J
    ⊢ Eq (c.π.app j ((CategoryTheory.Limits.Types.isLimitEquivSections t).symm x)) …
  -/
  conv_rhs => rw [← (isLimitEquivSections t).right_inv x]
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    c : CategoryTheory.Limits.Cone F
    t : CategoryTheory.Limits.IsLimit c
    x : ↑F.sections
    j : J
    ⊢ Eq (c.π.app j ((CategoryTheory.Limits.Types.isLimitEquivSections t).symm x)) …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- (internal implementation) the limit cone of a functor,
implemented as flat sections of a pi type
-/
@[simps]
noncomputable def limitCone : Cone F where
  pt := Shrink F.sections
  π :=
    { app := fun j u => ((equivShrink F.sections).symm u).val j
      naturality := fun j j' f => by
        /-
          J : Type v
          inst✝¹ : CategoryTheory.Category.{w, v} J
          F : CategoryTheory.Functor J (Type u)
          inst✝ : Small.{u, max u v} ↑F.sections
          j j' : J
          f : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        funext x
        /-
          case h
          J : Type v
          inst✝¹ : CategoryTheory.Category.{w, v} J
          F : CategoryTheory.Functor J (Type u)
          inst✝ : Small.{u, max u v} ↑F.sections
          j j' : J
          f : Quiver.Hom j j'
          x : ((CategoryTheory.Functor.const J).obj (Shrink.{u, max u v} ↑F.sections)).o …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        simp }
        /-
          🎉 no goals
        -/


@[ext]
lemma limitCone_pt_ext {x y : (limitCone F).pt}
    (w : (equivShrink F.sections).symm x = (equivShrink F.sections).symm y) : x = y := by
  /-
    J : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    inst✝ : Small.{u, max u v} ↑F.sections
    x y : (CategoryTheory.Limits.Types.Small.limitCone F).pt
    w : Eq ((equivShrink ↑F.sections).symm x) ((equivShrink ↑F.sections).symm y)
    ⊢ Eq x y
  -/
  aesop
  /-
    🎉 no goals
  -/


/-- (internal implementation) the fact that the proposed limit cone is the limit -/
@[simps]
noncomputable def limitConeIsLimit : IsLimit (limitCone.{v, u} F) where
  lift s v := equivShrink F.sections
    { val := fun j => s.π.app j v
      property := fun f => congr_fun (Cone.w s f) _ }
  uniq := fun _ _ w => by
    /-
      J : Type v
      inst✝¹ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J (Type u)
      inst✝ : Small.{u, max u v} ↑F.sections
      x✝¹ : CategoryTheory.Limits.Cone F
      x✝ : Quiver.Hom x✝¹.pt (CategoryTheory.Limits.Types.Small.limitCone F).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp x✝ ((CategoryTheory.Limi …
      ⊢ Eq x✝ ((fun s v => (equivShrink ↑F.sections) ⟨fun j => s.π.app j v, ⋯⟩) x✝¹)
    -/
    ext x j
    /-
      case h.w.a.h
      J : Type v
      inst✝¹ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J (Type u)
      inst✝ : Small.{u, max u v} ↑F.sections
      x✝¹ : CategoryTheory.Limits.Cone F
      x✝ : Quiver.Hom x✝¹.pt (CategoryTheory.Limits.Types.Small.limitCone F).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp x✝ ((CategoryTheory.Limi …
      x : x✝¹.pt
      j : J
      ⊢ Eq (↑((equivShrink ↑F.sections).symm (x✝ x)) j) (↑((equivShrink ↑F.sections) …
    -/
    simpa using congr_fun (w j) x
    /-
      🎉 no goals
    -/


theorem hasLimit_iff_small_sections (F : J ⥤ Type u) : HasLimit F ↔ Small.{u} F.sections :=
  ⟨fun _ => .mk ⟨_, ⟨(Equiv.ofBijective _
    ((isLimit_iff_bijective_sectionOfCone (limit.cone F)).mp ⟨limit.isLimit _⟩)).symm⟩⟩,
   fun _ => ⟨_, Small.limitConeIsLimit F⟩⟩

-- TODO: If `UnivLE` works out well, we will eventually want to deprecate these
-- definitions, and probably as a first step put them in namespace or otherwise rename them.

/-- (internal implementation) the limit cone of a functor,
implemented as flat sections of a pi type
-/
@[simps]
noncomputable def limitCone (F : J ⥤ TypeMax.{v, u}) : Cone F where
  pt := F.sections
  π :=
    { app := fun j u => u.val j
      naturality := fun j j' f => by
        /-
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          F : CategoryTheory.Functor J TypeMax
          j j' : J
          f : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        funext x
        /-
          case h
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          F : CategoryTheory.Functor J TypeMax
          j j' : J
          f : Quiver.Hom j j'
          x : ((CategoryTheory.Functor.const J).obj ↑F.sections).obj j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        simp }
        /-
          🎉 no goals
        -/


/-- (internal implementation) the fact that the proposed limit cone is the limit -/
@[simps]
noncomputable def limitConeIsLimit (F : J ⥤ TypeMax.{v, u}) : IsLimit (limitCone F) where
  lift s v :=
    { val := fun j => s.π.app j v
      property := fun f => congr_fun (Cone.w s f) _ }
  uniq := fun _ _ w => by
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J TypeMax
      x✝¹ : CategoryTheory.Limits.Cone F
      x✝ : Quiver.Hom x✝¹.pt (CategoryTheory.Limits.Types.limitCone F).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp x✝ ((CategoryTheory.Limi …
      ⊢ Eq x✝ ((fun s v => ⟨fun j => s.π.app j v, ⋯⟩) x✝¹)
    -/
    funext x
    /-
      case h
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J TypeMax
      x✝¹ : CategoryTheory.Limits.Cone F
      x✝ : Quiver.Hom x✝¹.pt (CategoryTheory.Limits.Types.limitCone F).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp x✝ ((CategoryTheory.Limi …
      x : x✝¹.pt
      ⊢ Eq (x✝ x) ((fun s v => ⟨fun j => s.π.app j v, ⋯⟩) x✝¹ x)
    -/
    apply Subtype.ext
    /-
      case h.a
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J TypeMax
      x✝¹ : CategoryTheory.Limits.Cone F
      x✝ : Quiver.Hom x✝¹.pt (CategoryTheory.Limits.Types.limitCone F).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp x✝ ((CategoryTheory.Limi …
      x : x✝¹.pt
      ⊢ Eq ↑(x✝ x) ↑((fun s v => ⟨fun j => s.π.app j v, ⋯⟩) x✝¹ x)
    -/
    funext j
    /-
      case h.a.h
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J TypeMax
      x✝¹ : CategoryTheory.Limits.Cone F
      x✝ : Quiver.Hom x✝¹.pt (CategoryTheory.Limits.Types.limitCone F).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp x✝ ((CategoryTheory.Limi …
      x : x✝¹.pt
      j : J
      ⊢ Eq (↑(x✝ x) j) (↑((fun s v => ⟨fun j => s.π.app j v, ⋯⟩) x✝¹ x) j)
    -/
    exact congr_fun (w j) x
    /-
      🎉 no goals
    -/


instance hasLimit [Small.{u} J] (F : J ⥤ Type u) : HasLimit F :=
  (hasLimit_iff_small_sections F).mpr inferInstance


instance hasLimitsOfShape [Small.{u} J] : HasLimitsOfShape J (Type u) where


/--
The category of types has all limits.

More specifically, when `UnivLE.{v, u}`, the category `Type u` has all `v`-small limits.

See <https://stacks.math.columbia.edu/tag/002U>.
-/
instance (priority := 1300) hasLimitsOfSize [UnivLE.{v, u}] : HasLimitsOfSize.{w, v} (Type u) where
  has_limits_of_shape _ := { }


/-- The equivalence between the abstract limit of `F` in `TypeMax.{v, u}`
and the "concrete" definition as the sections of `F`.
-/
noncomputable def limitEquivSections : limit F ≃ F.sections :=
  isLimitEquivSections (limit.isLimit F)


@[simp]
theorem limitEquivSections_apply (x : limit F) (j : J) :
    ((limitEquivSections F) x : ∀ j, F.obj j) j = limit.π F j x :=
  isLimitEquivSections_apply _ _ _


@[simp]
theorem limitEquivSections_symm_apply (x : F.sections) (j : J) :
    limit.π F j ((limitEquivSections F).symm x) = (x : ∀ j, F.obj j) j :=
  isLimitEquivSections_symm_apply _ _ _

-- Porting note: `limitEquivSections_symm_apply'` was removed because the linter
--   complains it is unnecessary
--@[simp]
--theorem limitEquivSections_symm_apply' (F : J ⥤ Type v) (x : F.sections) (j : J) :
--    limit.π F j ((limitEquivSections.{v, v} F).symm x) = (x : ∀ j, F.obj j) j :=
--  isLimitEquivSections_symm_apply _ _ _

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11182): removed @[ext]

/-- Construct a term of `limit F : Type u` from a family of terms `x : Π j, F.obj j`
which are "coherent": `∀ (j j') (f : j ⟶ j'), F.map f (x j) = x j'`.
-/
noncomputable def Limit.mk (x : ∀ j, F.obj j) (h : ∀ (j j') (f : j ⟶ j'), F.map f (x j) = x j') :
    limit F :=
  (limitEquivSections F).symm ⟨x, h _ _⟩


@[simp]
theorem Limit.π_mk (x : ∀ j, F.obj j) (h : ∀ (j j') (f : j ⟶ j'), F.map f (x j) = x j') (j) :
    limit.π F j (Limit.mk F x h) = x j := by
  /-
    J : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    inst✝ : CategoryTheory.Limits.HasLimit F
    x : (j : J) → F.obj j
    h : ∀ (j j' : J) (f : Quiver.Hom j j'), Eq (F.map f (x j)) (x j')
    j : J
    ⊢ Eq (CategoryTheory.Limits.limit.π F j (CategoryTheory.Limits.Types.Limit.mk  …
  -/
  dsimp [Limit.mk]
  /-
    J : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    inst✝ : CategoryTheory.Limits.HasLimit F
    x : (j : J) → F.obj j
    h : ∀ (j j' : J) (f : Quiver.Hom j j'), Eq (F.map f (x j)) (x j')
    j : J
    ⊢ Eq (CategoryTheory.Limits.limit.π F j ((CategoryTheory.Limits.Types.limitEqu …
  -/
  simp
  /-
    🎉 no goals
  -/

-- Porting note: `Limit.π_mk'` was removed because the linter complains it is unnecessary
--@[simp]
--theorem Limit.π_mk' (F : J ⥤ Type v) (x : ∀ j, F.obj j)
--    (h : ∀ (j j') (f : j ⟶ j'), F.map f (x j) = x j') (j) :
--    limit.π F j (Limit.mk.{v, v} F x h) = x j := by
--  dsimp [Limit.mk]
--  simp

-- PROJECT: prove this for concrete categories where the forgetful functor preserves limits

@[ext]
theorem limit_ext (x y : limit F) (w : ∀ j, limit.π F j x = limit.π F j y) : x = y := by
  /-
    J : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    inst✝ : CategoryTheory.Limits.HasLimit F
    x y : CategoryTheory.Limits.limit F
    w : ∀ (j : J), Eq (CategoryTheory.Limits.limit.π F j x) (CategoryTheory.Limits …
    ⊢ Eq x y
  -/
  apply (limitEquivSections F).injective
  /-
    case a
    J : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    inst✝ : CategoryTheory.Limits.HasLimit F
    x y : CategoryTheory.Limits.limit F
    w : ∀ (j : J), Eq (CategoryTheory.Limits.limit.π F j x) (CategoryTheory.Limits …
    ⊢ Eq ((CategoryTheory.Limits.Types.limitEquivSections F) x) ((CategoryTheory.L …
  -/
  ext j
  /-
    case a.a.h
    J : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    inst✝ : CategoryTheory.Limits.HasLimit F
    x y : CategoryTheory.Limits.limit F
    w : ∀ (j : J), Eq (CategoryTheory.Limits.limit.π F j x) (CategoryTheory.Limits …
    j : J
    ⊢ Eq (↑((CategoryTheory.Limits.Types.limitEquivSections F) x) j) (↑((CategoryT …
  -/
  simp [w j]
  /-
    🎉 no goals
  -/


@[ext]
theorem limit_ext' (F : J ⥤ Type v) (x y : limit F) (w : ∀ j, limit.π F j x = limit.π F j y) :
    x = y :=
  limit_ext F x y w


theorem limit_ext_iff' (F : J ⥤ Type v) (x y : limit F) :
    x = y ↔ ∀ j, limit.π F j x = limit.π F j y :=
  ⟨fun t _ => t ▸ rfl, limit_ext' _ _ _⟩

-- TODO: are there other limits lemmas that should have `_apply` versions?
-- Can we generate these like with `@[reassoc]`?
-- PROJECT: prove these for any concrete category where the forgetful functor preserves limits?
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): @[simp] was removed because the linter said it was useless
--@[simp]

variable {F} in
theorem Limit.w_apply {j j' : J} {x : limit F} (f : j ⟶ j') :
    F.map f (limit.π F j x) = limit.π F j' x :=
  congr_fun (limit.w F f) x

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): @[simp] was removed because the linter said it was useless

theorem Limit.lift_π_apply (s : Cone F) (j : J) (x : s.pt) :
    limit.π F j (limit.lift F s x) = s.π.app j x :=
  congr_fun (limit.lift_π s j) x

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): @[simp] was removed because the linter said it was useless

theorem Limit.map_π_apply {F G : J ⥤ Type u} [HasLimit F] [HasLimit G] (α : F ⟶ G) (j : J)
    (x : limit F) : limit.π G j (limMap α x) = α.app j (limit.π F j x) :=
  congr_fun (limMap_π α j) x


@[simp]
theorem Limit.w_apply' {F : J ⥤ Type v} {j j' : J} {x : limit F} (f : j ⟶ j') :
    F.map f (limit.π F j x) = limit.π F j' x :=
  congr_fun (limit.w F f) x


@[simp]
theorem Limit.lift_π_apply' (F : J ⥤ Type v) (s : Cone F) (j : J) (x : s.pt) :
    limit.π F j (limit.lift F s x) = s.π.app j x :=
  congr_fun (limit.lift_π s j) x


@[simp]
theorem Limit.map_π_apply' {F G : J ⥤ Type v} (α : F ⟶ G) (j : J) (x : limit F) :
    limit.π G j (limMap α x) = α.app j (limit.π F j x) :=
  congr_fun (limMap_π α j) x


/--
The relation defining the quotient type which implements the colimit of a functor `F : J ⥤ Type u`.
See `CategoryTheory.Limits.Types.Quot`.
-/
def Quot.Rel (F : J ⥤ Type u) : (Σ j, F.obj j) → (Σ j, F.obj j) → Prop := fun p p' =>
  ∃ f : p.1 ⟶ p'.1, p'.2 = F.map f p.2

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed @[nolint has_nonempty_instance]

/-- A quotient type implementing the colimit of a functor `F : J ⥤ Type u`,
as pairs `⟨j, x⟩` where `x : F.obj j`, modulo the equivalence relation generated by
`⟨j, x⟩ ~ ⟨j', x'⟩` whenever there is a morphism `f : j ⟶ j'` so `F.map f x = x'`.
-/
def Quot (F : J ⥤ Type u) : Type (max v u) :=
  _root_.Quot (Quot.Rel F)


instance [Small.{u} J] (F : J ⥤ Type u) : Small.{u} (Quot F) :=
  small_of_surjective Quot.mk_surjective


/-- Inclusion into the quotient type implementing the colimit. -/
def Quot.ι (F : J ⥤ Type u) (j : J) : F.obj j → Quot F :=
  fun x => Quot.mk _ ⟨j, x⟩


lemma Quot.jointly_surjective {F : J ⥤ Type u} (x : Quot F) : ∃ j y, x = Quot.ι F j y :=
  Quot.ind (β := fun x => ∃ j y, x = Quot.ι F j y) (fun ⟨j, y⟩ => ⟨j, y, rfl⟩) x


/-- (implementation detail) Part of the universal property of the colimit cocone, but without
    assuming that `Quot F` lives in the correct universe. -/
def Quot.desc : Quot F → c.pt :=
  Quot.lift (fun x => c.ι.app x.1 x.2) <| by
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J (Type u)
      c : CategoryTheory.Limits.Cocone F
      ⊢ ∀ (a b : Sigma fun j => F.obj j), CategoryTheory.Limits.Types.Quot.Rel F a b …
    -/
    rintro ⟨j, x⟩ ⟨j', _⟩ ⟨φ : j ⟶ j', rfl : _ = F.map φ x⟩
    /-
      case mk.mk.intro
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J (Type u)
      c : CategoryTheory.Limits.Cocone F
      j : J
      x : F.obj j
      j' : J
      φ : Quiver.Hom j j'
      ⊢ Eq ((fun x => c.ι.app x.fst x.snd) ⟨j, x⟩) ((fun x => c.ι.app x.fst x.snd) ⟨ …
    -/
    exact congr_fun (c.ι.naturality φ).symm x
    /-
      🎉 no goals
    -/


@[simp]
lemma Quot.ι_desc (j : J) (x : F.obj j) : Quot.desc c (Quot.ι F j x) = c.ι.app j x := rfl


@[simp]
lemma Quot.map_ι {j j' : J} {f : j ⟶ j'} (x : F.obj j) : Quot.ι F j' (F.map f x) = Quot.ι F j x :=
  (Quot.sound ⟨f, rfl⟩).symm


/-- (implementation detail) A function `Quot F → α` induces a cocone on `F` as long as the universes
    work out. -/
@[simps]
def toCocone {α : Type u} (f : Quot F → α) : Cocone F where
  pt := α
  ι := { app := fun j => f ∘ Quot.ι F j }


lemma Quot.desc_toCocone_desc {α : Type u} (f : Quot F → α) (hc : IsColimit c) (x : Quot F) :
    hc.desc (toCocone f) (Quot.desc c x) = f x := by
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    c : CategoryTheory.Limits.Cocone F
    α : Type u
    f : CategoryTheory.Limits.Types.Quot F → α
    hc : CategoryTheory.Limits.IsColimit c
    x : CategoryTheory.Limits.Types.Quot F
    ⊢ Eq (hc.desc (CategoryTheory.Limits.Types.toCocone f) (CategoryTheory.Limits. …
  -/
  obtain ⟨j, y, rfl⟩ := Quot.jointly_surjective x
  /-
    case intro.intro
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    c : CategoryTheory.Limits.Cocone F
    α : Type u
    f : CategoryTheory.Limits.Types.Quot F → α
    hc : CategoryTheory.Limits.IsColimit c
    j : J
    y : F.obj j
    ⊢ Eq (hc.desc (CategoryTheory.Limits.Types.toCocone f) (CategoryTheory.Limits. …
  -/
  simpa using congrFun (hc.fac _ j) y
  /-
    🎉 no goals
  -/


theorem isColimit_iff_bijective_desc : Nonempty (IsColimit c) ↔ (Quot.desc c).Bijective := by
  classical
  refine ⟨?_, ?_⟩
  · refine fun ⟨hc⟩ => ⟨fun x y h => ?_, fun x => ?_⟩
    · let f : Quot F → ULift.{u} Bool := fun z => ULift.up (x = z)
      suffices f x = f y by simpa [f] using this
      rw [← Quot.desc_toCocone_desc c f hc x, h, Quot.desc_toCocone_desc]
    · let f₁ : c.pt ⟶ ULift.{u} Bool := fun _ => ULift.up true
      let f₂ : c.pt ⟶ ULift.{u} Bool := fun x => ULift.up (∃ a, Quot.desc c a = x)
      suffices f₁ = f₂ by simpa [f₁, f₂] using congrFun this x
      refine hc.hom_ext fun j => funext fun x => ?_
      simpa [f₁, f₂] using ⟨Quot.ι F j x, by simp⟩
  · refine fun h => ⟨?_⟩
    let e := Equiv.ofBijective _ h
    have h : ∀ j x, e.symm (c.ι.app j x) = Quot.ι F j x :=
      fun j x => e.injective (Equiv.ofBijective_apply_symm_apply _ _ _)
    exact
      { desc := fun s => Quot.desc s ∘ e.symm
        fac := fun s j => by
          ext x
          simp [h]
        uniq := fun s m hm => by
          ext x
          obtain ⟨x, rfl⟩ := e.surjective x
          obtain ⟨j, x, rfl⟩ := Quot.jointly_surjective x
          rw [← h, Equiv.apply_symm_apply]
          simpa [h] using congrFun (hm j) x }


/-- (internal implementation) the colimit cocone of a functor,
implemented as a quotient of a sigma type
-/
@[simps]
noncomputable def colimitCocone (F : J ⥤ Type u) [Small.{u} (Quot F)] : Cocone F where
  pt := Shrink (Quot F)
  ι :=
    { app := fun j x => equivShrink.{u} _ (Quot.mk _ ⟨j, x⟩)
      naturality := fun _ _ f => funext fun _ => congrArg _ (Quot.sound ⟨f, rfl⟩).symm }


@[simp]
theorem Quot.desc_colimitCocone (F : J ⥤ Type u) [Small.{u} (Quot F)] :
    Quot.desc (colimitCocone F) = equivShrink.{u} (Quot F) := by
  /-
    J : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    inst✝ : Small.{u, max u v} (CategoryTheory.Limits.Types.Quot F)
    ⊢ Eq (CategoryTheory.Limits.Types.Quot.desc (CategoryTheory.Limits.Types.colim …
  -/
  ext ⟨j, x⟩
  /-
    case h.mk.mk
    J : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    inst✝ : Small.{u, max u v} (CategoryTheory.Limits.Types.Quot F)
    x✝ : CategoryTheory.Limits.Types.Quot F
    j : J
    x : F.obj j
    ⊢ Eq (CategoryTheory.Limits.Types.Quot.desc (CategoryTheory.Limits.Types.colim …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- (internal implementation) the fact that the proposed colimit cocone is the colimit -/
noncomputable def colimitCoconeIsColimit (F : J ⥤ Type u) [Small.{u} (Quot F)] :
    IsColimit (colimitCocone F) :=
  Nonempty.some <| by
    /-
      J : Type v
      inst✝¹ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J (Type u)
      inst✝ : Small.{u, max u v} (CategoryTheory.Limits.Types.Quot F)
      ⊢ Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Types.colim …
    -/
    rw [isColimit_iff_bijective_desc, Quot.desc_colimitCocone]
    /-
      J : Type v
      inst✝¹ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J (Type u)
      inst✝ : Small.{u, max u v} (CategoryTheory.Limits.Types.Quot F)
      ⊢ Function.Bijective ⇑(equivShrink (CategoryTheory.Limits.Types.Quot F))
    -/
    exact (equivShrink _).bijective
    /-
      🎉 no goals
    -/


theorem hasColimit_iff_small_quot (F : J ⥤ Type u) : HasColimit F ↔ Small.{u} (Quot F) :=
  ⟨fun _ => .mk ⟨_, ⟨(Equiv.ofBijective _
    ((isColimit_iff_bijective_desc (colimit.cocone F)).mp ⟨colimit.isColimit _⟩))⟩⟩,
   fun _ => ⟨_, colimitCoconeIsColimit F⟩⟩


theorem small_quot_of_hasColimit (F : J ⥤ Type u) [HasColimit F] : Small.{u} (Quot F) :=
  (hasColimit_iff_small_quot F).mp inferInstance


instance hasColimit [Small.{u} J] (F : J ⥤ Type u) : HasColimit F :=
  (hasColimit_iff_small_quot F).mpr inferInstance


instance hasColimitsOfShape [Small.{u} J] : HasColimitsOfShape J (Type u) where


/-- The category of types has all colimits.

See <https://stacks.math.columbia.edu/tag/002U>.
-/
instance (priority := 1300) hasColimitsOfSize [UnivLE.{v, u}] :
    HasColimitsOfSize.{w, v} (Type u) where


/-- (internal implementation) the colimit cocone of a functor,
implemented as a quotient of a sigma type
-/
@[simps]
def colimitCocone (F : J ⥤ TypeMax.{v, u}) : Cocone F where
  pt := Quot F
  ι :=
    { app := fun j x => Quot.mk (Quot.Rel F) ⟨j, x⟩
      naturality := fun _ _ f => funext fun _ => (Quot.sound ⟨f, rfl⟩).symm }


/-- (internal implementation) the fact that the proposed colimit cocone is the colimit -/
def colimitCoconeIsColimit (F : J ⥤ TypeMax.{v, u}) : IsColimit (colimitCocone F) where
  desc s :=
    Quot.lift (fun p : Σj, F.obj j => s.ι.app p.1 p.2) fun ⟨j, x⟩ ⟨j', x'⟩ ⟨f, hf⟩ => by
      /-
        J : Type v
        inst✝ : CategoryTheory.Category.{w, v} J
        F : CategoryTheory.Functor J TypeMax
        s : CategoryTheory.Limits.Cocone F
        x✝² x✝¹ : Sigma fun j => F.obj j
        j : J
        x : F.obj j
        j' : J
        x' : F.obj j'
        x✝ : CategoryTheory.Limits.Types.Quot.Rel F ⟨j, x⟩ ⟨j', x'⟩
        f : Quiver.Hom ⟨j, x⟩.fst ⟨j', x'⟩.fst
        hf : Eq ⟨j', x'⟩.snd (F.map f ⟨j, x⟩.snd)
        ⊢ Eq ((fun p => s.ι.app p.fst p.snd) ⟨j, x⟩) ((fun p => s.ι.app p.fst p.snd) ⟨ …
      -/
      dsimp at hf
      /-
        J : Type v
        inst✝ : CategoryTheory.Category.{w, v} J
        F : CategoryTheory.Functor J TypeMax
        s : CategoryTheory.Limits.Cocone F
        x✝² x✝¹ : Sigma fun j => F.obj j
        j : J
        x : F.obj j
        j' : J
        x' : F.obj j'
        x✝ : CategoryTheory.Limits.Types.Quot.Rel F ⟨j, x⟩ ⟨j', x'⟩
        f : Quiver.Hom ⟨j, x⟩.fst ⟨j', x'⟩.fst
        hf : Eq x' (F.map f x)
        ⊢ Eq ((fun p => s.ι.app p.fst p.snd) ⟨j, x⟩) ((fun p => s.ι.app p.fst p.snd) ⟨ …
      -/
      rw [hf]
      /-
        J : Type v
        inst✝ : CategoryTheory.Category.{w, v} J
        F : CategoryTheory.Functor J TypeMax
        s : CategoryTheory.Limits.Cocone F
        x✝² x✝¹ : Sigma fun j => F.obj j
        j : J
        x : F.obj j
        j' : J
        x' : F.obj j'
        x✝ : CategoryTheory.Limits.Types.Quot.Rel F ⟨j, x⟩ ⟨j', x'⟩
        f : Quiver.Hom ⟨j, x⟩.fst ⟨j', x'⟩.fst
        hf : Eq x' (F.map f x)
        ⊢ Eq ((fun p => s.ι.app p.fst p.snd) ⟨j, x⟩) ((fun p => s.ι.app p.fst p.snd) ⟨ …
      -/
      exact (congr_fun (Cocone.w s f) x).symm
      /-
        🎉 no goals
      -/
  uniq s m hm := by
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J TypeMax
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (CategoryTheory.Limits.Types.TypeMax.colimitCocone F).pt s.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits …
      ⊢ Eq m ((fun s => Quot.lift (fun p => s.ι.app p.fst p.snd) ⋯) s)
    -/
    funext x
    /-
      case h
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J TypeMax
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (CategoryTheory.Limits.Types.TypeMax.colimitCocone F).pt s.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits …
      x : (CategoryTheory.Limits.Types.TypeMax.colimitCocone F).pt
      ⊢ Eq (m x) ((fun s => Quot.lift (fun p => s.ι.app p.fst p.snd) ⋯) s x)
    -/
    induction' x using Quot.ind with x
    /-
      case h.mk
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J TypeMax
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (CategoryTheory.Limits.Types.TypeMax.colimitCocone F).pt s.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits …
      x : Sigma fun j => F.obj j
      ⊢ Eq (m (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel F) x)) ((fun s => Quot. …
    -/
    exact congr_fun (hm x.1) x.2
    /-
      🎉 no goals
    -/


/-- The equivalence between the abstract colimit of `F` in `Type u`
and the "concrete" definition as a quotient.
-/
noncomputable def colimitEquivQuot : colimit F ≃ Quot F :=
  (IsColimit.coconePointUniqueUpToIso
    (colimit.isColimit F) (colimitCoconeIsColimit F)).toEquiv.trans (equivShrink _).symm


@[simp]
theorem colimitEquivQuot_symm_apply (j : J) (x : F.obj j) :
    (colimitEquivQuot F).symm (Quot.mk _ ⟨j, x⟩) = colimit.ι F j x :=
  congrFun (IsColimit.comp_coconePointUniqueUpToIso_inv (colimit.isColimit F) _ _) x


@[simp]
theorem colimitEquivQuot_apply (j : J) (x : F.obj j) :
    (colimitEquivQuot F) (colimit.ι F j x) = Quot.mk _ ⟨j, x⟩ := by
  /-
    J : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    inst✝ : CategoryTheory.Limits.HasColimit F
    j : J
    x : F.obj j
    ⊢ Eq ((CategoryTheory.Limits.Types.colimitEquivQuot F) (CategoryTheory.Limits. …
  -/
  apply (colimitEquivQuot F).symm.injective
  /-
    case a
    J : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    inst✝ : CategoryTheory.Limits.HasColimit F
    j : J
    x : F.obj j
    ⊢ Eq ((CategoryTheory.Limits.Types.colimitEquivQuot F).symm ((CategoryTheory.L …
  -/
  simp
  /-
    🎉 no goals
  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): @[simp] was removed because the linter said it was useless

variable {F} in
theorem Colimit.w_apply {j j' : J} {x : F.obj j} (f : j ⟶ j') :
    colimit.ι F j' (F.map f x) = colimit.ι F j x :=
  congr_fun (colimit.w F f) x

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): @[simp] was removed because the linter said it was useless

theorem Colimit.ι_desc_apply (s : Cocone F) (j : J) (x : F.obj j) :
    colimit.desc F s (colimit.ι F j x) = s.ι.app j x :=
   congr_fun (colimit.ι_desc s j) x

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): @[simp] was removed because the linter said it was useless

theorem Colimit.ι_map_apply {F G : J ⥤ Type u} [HasColimitsOfShape J (Type u)] (α : F ⟶ G) (j : J)
    (x : F.obj j) : colim.map α (colimit.ι F j x) = colimit.ι G j (α.app j x) :=
  congr_fun (colimit.ι_map α j) x


@[simp]
theorem Colimit.w_apply' {F : J ⥤ Type v} {j j' : J} {x : F.obj j} (f : j ⟶ j') :
    colimit.ι F j' (F.map f x) = colimit.ι F j x :=
  congr_fun (colimit.w F f) x


@[simp]
theorem Colimit.ι_desc_apply' (F : J ⥤ Type v) (s : Cocone F) (j : J) (x : F.obj j) :
    colimit.desc F s (colimit.ι F j x) = s.ι.app j x :=
  congr_fun (colimit.ι_desc s j) x


@[simp]
theorem Colimit.ι_map_apply' {F G : J ⥤ Type v} (α : F ⟶ G) (j : J) (x) :
    colim.map α (colimit.ι F j x) = colimit.ι G j (α.app j x) :=
  congr_fun (colimit.ι_map α j) x


variable {F} in
theorem colimit_sound {j j' : J} {x : F.obj j} {x' : F.obj j'} (f : j ⟶ j')
    (w : F.map f x = x') : colimit.ι F j x = colimit.ι F j' x' := by
  /-
    J : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    inst✝ : CategoryTheory.Limits.HasColimit F
    j j' : J
    x : F.obj j
    x' : F.obj j'
    f : Quiver.Hom j j'
    w : Eq (F.map f x) x'
    ⊢ Eq (CategoryTheory.Limits.colimit.ι F j x) (CategoryTheory.Limits.colimit.ι  …
  -/
  rw [← w, Colimit.w_apply]
  /-
    🎉 no goals
  -/


variable {F} in
theorem colimit_sound' {j j' : J} {x : F.obj j} {x' : F.obj j'} {j'' : J}
    (f : j ⟶ j'') (f' : j' ⟶ j'') (w : F.map f x = F.map f' x') :
    colimit.ι F j x = colimit.ι F j' x' := by
  /-
    J : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    inst✝ : CategoryTheory.Limits.HasColimit F
    j j' : J
    x : F.obj j
    x' : F.obj j'
    j'' : J
    f : Quiver.Hom j j''
    f' : Quiver.Hom j' j''
    w : Eq (F.map f x) (F.map f' x')
    ⊢ Eq (CategoryTheory.Limits.colimit.ι F j x) (CategoryTheory.Limits.colimit.ι  …
  -/
  rw [← colimit.w _ f, ← colimit.w _ f']
  /-
    J : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    inst✝ : CategoryTheory.Limits.HasColimit F
    j j' : J
    x : F.obj j
    x' : F.obj j'
    j'' : J
    f : Quiver.Hom j j''
    f' : Quiver.Hom j' j''
    w : Eq (F.map f x) (F.map f' x')
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) (CategoryTheory.Limits.coli …
  -/
  rw [types_comp_apply, types_comp_apply, w]
  /-
    🎉 no goals
  -/


variable {F} in
theorem colimit_eq {j j' : J} {x : F.obj j} {x' : F.obj j'}
    (w : colimit.ι F j x = colimit.ι F j' x') :
      Relation.EqvGen (Quot.Rel F) ⟨j, x⟩ ⟨j', x'⟩ := by
  /-
    J : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    inst✝ : CategoryTheory.Limits.HasColimit F
    j j' : J
    x : F.obj j
    x' : F.obj j'
    w : Eq (CategoryTheory.Limits.colimit.ι F j x) (CategoryTheory.Limits.colimit. …
    ⊢ Relation.EqvGen (CategoryTheory.Limits.Types.Quot.Rel F) ⟨j, x⟩ ⟨j', x'⟩
  -/
  apply Quot.eq.1
  /-
    J : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    inst✝ : CategoryTheory.Limits.HasColimit F
    j j' : J
    x : F.obj j
    x' : F.obj j'
    w : Eq (CategoryTheory.Limits.colimit.ι F j x) (CategoryTheory.Limits.colimit. …
    ⊢ Eq (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel F) ⟨j, x⟩) (Quot.mk (Categ …
  -/
  simpa using congr_arg (colimitEquivQuot F) w
  /-
    🎉 no goals
  -/


theorem jointly_surjective_of_isColimit {F : J ⥤ Type u} {t : Cocone F} (h : IsColimit t)
    (x : t.pt) : ∃ j y, t.ι.app j y = x := by
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    t : CategoryTheory.Limits.Cocone F
    h : CategoryTheory.Limits.IsColimit t
    x : t.pt
    ⊢ Exists fun j => Exists fun y => Eq (t.ι.app j y) x
  -/
  by_contra hx
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    t : CategoryTheory.Limits.Cocone F
    h : CategoryTheory.Limits.IsColimit t
    x : t.pt
    hx : Not (Exists fun j => Exists fun y => Eq (t.ι.app j y) x)
    ⊢ False
  -/
  simp_rw [not_exists] at hx
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J (Type u)
    t : CategoryTheory.Limits.Cocone F
    h : CategoryTheory.Limits.IsColimit t
    x : t.pt
    hx : ∀ (x_1 : J) (x_2 : F.obj x_1), Not (Eq (t.ι.app x_1 x_2) x)
    ⊢ False
  -/
  apply (_ : (fun _ ↦ ULift.up True) ≠ (⟨· ≠ x⟩))
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J (Type u)
      t : CategoryTheory.Limits.Cocone F
      h : CategoryTheory.Limits.IsColimit t
      x : t.pt
      hx : ∀ (x_1 : J) (x_2 : F.obj x_1), Not (Eq (t.ι.app x_1 x_2) x)
      ⊢ Eq (fun x => { down := True }) fun x_1 => { down := Ne x_1 x }
    -/
  · refine h.hom_ext fun j ↦ ?_
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J (Type u)
      t : CategoryTheory.Limits.Cocone F
      h : CategoryTheory.Limits.IsColimit t
      x : t.pt
      hx : ∀ (x_1 : J) (x_2 : F.obj x_1), Not (Eq (t.ι.app x_1 x_2) x)
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (t.ι.app j) fun x => { down := True } …
    -/
    ext y
    /-
      case h.h.a
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J (Type u)
      t : CategoryTheory.Limits.Cocone F
      h : CategoryTheory.Limits.IsColimit t
      x : t.pt
      hx : ∀ (x_1 : J) (x_2 : F.obj x_1), Not (Eq (t.ι.app x_1 x_2) x)
      j : J
      y : F.obj j
      ⊢ Iff (CategoryTheory.CategoryStruct.comp (t.ι.app j) (fun x => { down := True …
    -/
    exact (true_iff _).mpr (hx j y)
    /-
      🎉 no goals
    -/
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J (Type u)
      t : CategoryTheory.Limits.Cocone F
      h : CategoryTheory.Limits.IsColimit t
      x : t.pt
      hx : ∀ (x_1 : J) (x_2 : F.obj x_1), Not (Eq (t.ι.app x_1 x_2) x)
      ⊢ Ne (fun x => { down := True }) fun x_1 => { down := Ne x_1 x }
    -/
  · exact fun he ↦ of_eq_true (congr_arg ULift.down <| congr_fun he x).symm rfl
    /-
      🎉 no goals
    -/


theorem jointly_surjective (F : J ⥤ Type u) {t : Cocone F} (h : IsColimit t) (x : t.pt) :
    ∃ j y, t.ι.app j y = x := jointly_surjective_of_isColimit h x


variable {F} in
/-- A variant of `jointly_surjective` for `x : colimit F`. -/
theorem jointly_surjective' (x : colimit F) :
    ∃ j y, colimit.ι F j y = x :=
  jointly_surjective F (colimit.isColimit F) x


/-- If a colimit is nonempty, also its index category is nonempty. -/
theorem nonempty_of_nonempty_colimit {F : J ⥤ Type u} [HasColimit F] :
    Nonempty (colimit F) → Nonempty J :=
  Nonempty.map <| Sigma.fst ∘ Quot.out ∘ (colimitEquivQuot F).toFun


/-- the image of a morphism in Type is just `Set.range f` -/
def Image : Type u :=
  Set.range f


instance [Inhabited α] : Inhabited (Image f) where default := ⟨f default, ⟨_, rfl⟩⟩


/-- the inclusion of `Image f` into the target -/
def Image.ι : Image f ⟶ β :=
  Subtype.val


instance : Mono (Image.ι f) :=
  (mono_iff_injective _).2 Subtype.val_injective


/-- the universal property for the image factorisation -/
noncomputable def Image.lift (F' : MonoFactorisation f) : Image f ⟶ F'.I :=
  (fun x => F'.e (Classical.indefiniteDescription _ x.2).1 : Image f → F'.I)


theorem Image.lift_fac (F' : MonoFactorisation f) : Image.lift F' ≫ F'.m = Image.ι f := by
  /-
    α β : Type u
    f : Quiver.Hom α β
    F' : CategoryTheory.Limits.MonoFactorisation f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Types.Image.li …
  -/
  funext x
  /-
    case h
    α β : Type u
    f : Quiver.Hom α β
    F' : CategoryTheory.Limits.MonoFactorisation f
    x : CategoryTheory.Limits.Types.Image f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Types.Image.li …
  -/
  change (F'.e ≫ F'.m) _ = _
  /-
    case h
    α β : Type u
    f : Quiver.Hom α β
    F' : CategoryTheory.Limits.MonoFactorisation f
    x : CategoryTheory.Limits.Types.Image f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp F'.e F'.m ↑(Classical.indefiniteDescr …
  -/
  rw [F'.fac, (Classical.indefiniteDescription _ x.2).2]
  /-
    case h
    α β : Type u
    f : Quiver.Hom α β
    F' : CategoryTheory.Limits.MonoFactorisation f
    x : CategoryTheory.Limits.Types.Image f
    ⊢ Eq (↑x) (CategoryTheory.Limits.Types.Image.ι f x)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- the factorisation of any morphism in Type through a mono. -/
def monoFactorisation : MonoFactorisation f where
  I := Image f
  m := Image.ι f
  e := Set.rangeFactorization f


/-- the factorisation through a mono has the universal property of the image. -/
noncomputable def isImage : IsImage (monoFactorisation f) where
  lift := Image.lift
  lift_fac := Image.lift_fac


instance : HasImage f :=
  HasImage.mk ⟨_, isImage f⟩


instance : HasImages (Type u) where
                  /-
                    J : Type v
                    inst✝¹ : CategoryTheory.Category.{w, v} J
                    F : CategoryTheory.Functor J (Type u)
                    inst✝ : CategoryTheory.Limits.HasColimit F
                    α β : Type u
                    f : Quiver.Hom α β
                    ⊢ ∀ {X Y : Type u} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasImage f
                  -/
  has_image := by infer_instance
                  /-
                    🎉 no goals
                  -/


instance : HasImageMaps (Type u) where
  has_image_map {f g} st :=
    HasImageMap.transport st (monoFactorisation f.hom) (isImage g.hom)
      (fun x => ⟨st.right x.1, ⟨st.left (Classical.choose x.2), by
        /-
          J : Type v
          inst✝¹ : CategoryTheory.Category.{w, v} J
          F : CategoryTheory.Functor J (Type u)
          inst✝ : CategoryTheory.Limits.HasColimit F
          α β : Type u
          f✝ : Quiver.Hom α β
          f g : CategoryTheory.Arrow (Type u)
          st : Quiver.Hom f g
          x : (CategoryTheory.Limits.Types.monoFactorisation f.hom).I
          ⊢ Eq (g.hom (st.left (Classical.choose ⋯))) (st.right ↑x)
        -/
        have p := st.w
        /-
          J : Type v
          inst✝¹ : CategoryTheory.Category.{w, v} J
          F : CategoryTheory.Functor J (Type u)
          inst✝ : CategoryTheory.Limits.HasColimit F
          α β : Type u
          f✝ : Quiver.Hom α β
          f g : CategoryTheory.Arrow (Type u)
          st : Quiver.Hom f g
          x : (CategoryTheory.Limits.Types.monoFactorisation f.hom).I
          p : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Type u …
          ⊢ Eq (g.hom (st.left (Classical.choose ⋯))) (st.right ↑x)
        -/
        replace p := congr_fun p (Classical.choose x.2)
        /-
          J : Type v
          inst✝¹ : CategoryTheory.Category.{w, v} J
          F : CategoryTheory.Functor J (Type u)
          inst✝ : CategoryTheory.Limits.HasColimit F
          α β : Type u
          f✝ : Quiver.Hom α β
          f g : CategoryTheory.Arrow (Type u)
          st : Quiver.Hom f g
          x : (CategoryTheory.Limits.Types.monoFactorisation f.hom).I
          p : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Type u …
          ⊢ Eq (g.hom (st.left (Classical.choose ⋯))) (st.right ↑x)
        -/
        simp only [Functor.id_obj, Functor.id_map, types_comp_apply] at p
        /-
          J : Type v
          inst✝¹ : CategoryTheory.Category.{w, v} J
          F : CategoryTheory.Functor J (Type u)
          inst✝ : CategoryTheory.Limits.HasColimit F
          α β : Type u
          f✝ : Quiver.Hom α β
          f g : CategoryTheory.Arrow (Type u)
          st : Quiver.Hom f g
          x : (CategoryTheory.Limits.Types.monoFactorisation f.hom).I
          p : Eq (g.hom (st.left (Classical.choose ⋯))) (st.right (f.hom (Classical.choo …
          ⊢ Eq (g.hom (st.left (Classical.choose ⋯))) (st.right ↑x)
        -/
        rw [p, Classical.choose_spec x.2]⟩⟩) rfl
        /-
          🎉 no goals
        -/


private noncomputable def limitOfSurjectionsSurjective.preimage
    (a : F.obj ⟨0⟩) : (n : ℕ) → F.obj ⟨n⟩
    | 0 => a
    | n+1 => (hF n (preimage a n)).choose


include hF in
open limitOfSurjectionsSurjective in
/-- Auxiliary lemma. Use `limit_of_surjections_surjective` instead. -/
lemma surjective_π_app_zero_of_surjective_map_aux :
    Function.Surjective ((limitCone F).π.app ⟨0⟩) := by
  /-
    F : CategoryTheory.Functor (Opposite Nat) (Type u)
    hF : ∀ (n : Nat), Function.Surjective (F.map (CategoryTheory.homOfLE ⋯).op)
    ⊢ Function.Surjective ((CategoryTheory.Limits.Types.limitCone F).π.app { unop  …
  -/
  intro a
  /-
    F : CategoryTheory.Functor (Opposite Nat) (Type u)
    hF : ∀ (n : Nat), Function.Surjective (F.map (CategoryTheory.homOfLE ⋯).op)
    a : F.obj { unop := 0 }
    ⊢ Exists fun a_1 => Eq ((CategoryTheory.Limits.Types.limitCone F).π.app { unop …
  -/
  refine ⟨⟨fun ⟨n⟩ ↦ preimage hF a n, ?_⟩, rfl⟩
  /-
    F : CategoryTheory.Functor (Opposite Nat) (Type u)
    hF : ∀ (n : Nat), Function.Surjective (F.map (CategoryTheory.homOfLE ⋯).op)
    a : F.obj { unop := 0 }
    ⊢ Membership.mem F.sections fun x => CategoryTheory.Limits.Types.surjective_π_ …
  -/
  intro ⟨n⟩ ⟨m⟩ ⟨⟨⟨(h : m ≤ n)⟩⟩⟩
  induction h with
  | refl =>
    erw [CategoryTheory.Functor.map_id, types_id_apply]
  | @step p h ih =>
    rw [← ih]
    have h' : m ≤ p := h
    erw [CategoryTheory.Functor.map_comp (f := (homOfLE (Nat.le_succ p)).op) (g := (homOfLE h').op),
      types_comp_apply, (hF p _).choose_spec]
    rfl


/--
Given surjections `⋯ ⟶ Xₙ₊₁ ⟶ Xₙ ⟶ ⋯ ⟶ X₀`, the projection map `lim Xₙ ⟶ X₀` is surjective.
-/
lemma surjective_π_app_zero_of_surjective_map
    (hc : IsLimit c)
    (hF : ∀ n, Function.Surjective (F.map (homOfLE (Nat.le_succ n)).op)) :
    Function.Surjective (c.π.app ⟨0⟩) := by
  /-
    F : CategoryTheory.Functor (Opposite Nat) (Type u)
    c : CategoryTheory.Limits.Cone F
    hc : CategoryTheory.Limits.IsLimit c
    hF : ∀ (n : Nat), Function.Surjective (F.map (CategoryTheory.homOfLE ⋯).op)
    ⊢ Function.Surjective (c.π.app { unop := 0 })
  -/
  let i := hc.conePointUniqueUpToIso (limitConeIsLimit F)
  /-
    F : CategoryTheory.Functor (Opposite Nat) (Type u)
    c : CategoryTheory.Limits.Cone F
    hc : CategoryTheory.Limits.IsLimit c
    hF : ∀ (n : Nat), Function.Surjective (F.map (CategoryTheory.homOfLE ⋯).op)
    i : CategoryTheory.Iso c.pt (CategoryTheory.Limits.Types.limitCone F).pt := hc …
    ⊢ Function.Surjective (c.π.app { unop := 0 })
  -/
  have : c.π.app ⟨0⟩ = i.hom ≫ (limitCone F).π.app ⟨0⟩ := by simp [i]
  /-
    F : CategoryTheory.Functor (Opposite Nat) (Type u)
    c : CategoryTheory.Limits.Cone F
    hc : CategoryTheory.Limits.IsLimit c
    hF : ∀ (n : Nat), Function.Surjective (F.map (CategoryTheory.homOfLE ⋯).op)
    i : CategoryTheory.Iso c.pt (CategoryTheory.Limits.Types.limitCone F).pt := hc …
    this : Eq (c.π.app { unop := 0 }) (CategoryTheory.CategoryStruct.comp i.hom (( …
    ⊢ Function.Surjective (c.π.app { unop := 0 })
  -/
  rw [this]
  /-
    F : CategoryTheory.Functor (Opposite Nat) (Type u)
    c : CategoryTheory.Limits.Cone F
    hc : CategoryTheory.Limits.IsLimit c
    hF : ∀ (n : Nat), Function.Surjective (F.map (CategoryTheory.homOfLE ⋯).op)
    i : CategoryTheory.Iso c.pt (CategoryTheory.Limits.Types.limitCone F).pt := hc …
    this : Eq (c.π.app { unop := 0 }) (CategoryTheory.CategoryStruct.comp i.hom (( …
    ⊢ Function.Surjective (CategoryTheory.CategoryStruct.comp i.hom ((CategoryTheo …
  -/
  apply Function.Surjective.comp
    /-
      case hg
      F : CategoryTheory.Functor (Opposite Nat) (Type u)
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit c
      hF : ∀ (n : Nat), Function.Surjective (F.map (CategoryTheory.homOfLE ⋯).op)
      i : CategoryTheory.Iso c.pt (CategoryTheory.Limits.Types.limitCone F).pt := hc …
      this : Eq (c.π.app { unop := 0 }) (CategoryTheory.CategoryStruct.comp i.hom (( …
      ⊢ Function.Surjective ((CategoryTheory.Limits.Types.limitCone F).π.app { unop  …
    -/
  · exact surjective_π_app_zero_of_surjective_map_aux hF
    /-
      🎉 no goals
    -/
    /-
      case hf
      F : CategoryTheory.Functor (Opposite Nat) (Type u)
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit c
      hF : ∀ (n : Nat), Function.Surjective (F.map (CategoryTheory.homOfLE ⋯).op)
      i : CategoryTheory.Iso c.pt (CategoryTheory.Limits.Types.limitCone F).pt := hc …
      this : Eq (c.π.app { unop := 0 }) (CategoryTheory.CategoryStruct.comp i.hom (( …
      ⊢ Function.Surjective i.hom
    -/
  · rw [← epi_iff_surjective]
    /-
      case hf
      F : CategoryTheory.Functor (Opposite Nat) (Type u)
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit c
      hF : ∀ (n : Nat), Function.Surjective (F.map (CategoryTheory.homOfLE ⋯).op)
      i : CategoryTheory.Iso c.pt (CategoryTheory.Limits.Types.limitCone F).pt := hc …
      this : Eq (c.π.app { unop := 0 }) (CategoryTheory.CategoryStruct.comp i.hom (( …
      ⊢ CategoryTheory.Epi i.hom
    -/
    infer_instance
    /-
      🎉 no goals
    -/


/-- Sections of `F ⋙ coyoneda.obj (op X)` identify to natural
transformations `(const J).obj X ⟶ F`. -/
@[simps]
def compCoyonedaSectionsEquiv (F : J ⥤ C) (X : C) :
    (F ⋙ coyoneda.obj (op X)).sections ≃ ((const J).obj X ⟶ F) where
  toFun s :=
    { app := fun j => s.val j
      naturality := fun j j' f => by
        /-
          J : Type u_1
          C : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.85686, u_1} J
          inst✝ : CategoryTheory.Category.{?u.86167, u_2} C
          F : CategoryTheory.Functor J C
          X : C
          s : ↑(F.comp (CategoryTheory.coyoneda.obj { unop := X })).sections
          j j' : J
          f : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        dsimp
        /-
          J : Type u_1
          C : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.85686, u_1} J
          inst✝ : CategoryTheory.Category.{?u.86167, u_2} C
          F : CategoryTheory.Functor J C
          X : C
          s : ↑(F.comp (CategoryTheory.coyoneda.obj { unop := X })).sections
          j j' : J
          f : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X)  …
        -/
        rw [Category.id_comp]
        /-
          J : Type u_1
          C : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.85686, u_1} J
          inst✝ : CategoryTheory.Category.{?u.86167, u_2} C
          F : CategoryTheory.Functor J C
          X : C
          s : ↑(F.comp (CategoryTheory.coyoneda.obj { unop := X })).sections
          j j' : J
          f : Quiver.Hom j j'
          ⊢ Eq (↑s j') (CategoryTheory.CategoryStruct.comp (↑s j) (F.map f))
        -/
        exact (s.property f).symm }
        /-
          🎉 no goals
        -/
                                         /-
                                           J : Type u_1
                                           C : Type u_2
                                           inst✝¹ : CategoryTheory.Category.{?u.85686, u_1} J
                                           inst✝ : CategoryTheory.Category.{?u.86167, u_2} C
                                           F : CategoryTheory.Functor J C
                                           X : C
                                           τ : Quiver.Hom ((CategoryTheory.Functor.const J).obj X) F
                                           j j' : J
                                           f : Quiver.Hom j j'
                                           ⊢ Eq ((F.comp (CategoryTheory.coyoneda.obj { unop := X })).map f (τ.app j)) (τ …
                                         -/
  invFun τ := ⟨τ.app, fun {j j'} f => by simpa using (τ.naturality f).symm⟩
                                         /-
                                           🎉 no goals
                                         -/
  left_inv _ := rfl
  right_inv _ := rfl


/-- Sections of `F.op ⋙ yoneda.obj X` identify to natural
transformations `F ⟶ (const J).obj X`. -/
@[simps]
def opCompYonedaSectionsEquiv (F : J ⥤ C) (X : C) :
    (F.op ⋙ yoneda.obj X).sections ≃ (F ⟶ (const J).obj X) where
  toFun s :=
    { app := fun j => s.val (op j)
      naturality := fun j j' f => by
        /-
          J : Type u_1
          C : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.88354, u_1} J
          inst✝ : CategoryTheory.Category.{?u.88869, u_2} C
          F : CategoryTheory.Functor J C
          X : C
          s : ↑(F.op.comp (CategoryTheory.yoneda.obj X)).sections
          j j' : J
          f : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun j => ↑s { unop := j } …
        -/
        dsimp
        /-
          J : Type u_1
          C : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.88354, u_1} J
          inst✝ : CategoryTheory.Category.{?u.88869, u_2} C
          F : CategoryTheory.Functor J C
          X : C
          s : ↑(F.op.comp (CategoryTheory.yoneda.obj X)).sections
          j j' : J
          f : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) (↑s { unop := j' })) (Categ …
        -/
        rw [Category.comp_id]
        /-
          J : Type u_1
          C : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.88354, u_1} J
          inst✝ : CategoryTheory.Category.{?u.88869, u_2} C
          F : CategoryTheory.Functor J C
          X : C
          s : ↑(F.op.comp (CategoryTheory.yoneda.obj X)).sections
          j j' : J
          f : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) (↑s { unop := j' })) (↑s {  …
        -/
        exact (s.property f.op) }
        /-
          🎉 no goals
        -/
                                                         /-
                                                           J : Type u_1
                                                           C : Type u_2
                                                           inst✝¹ : CategoryTheory.Category.{?u.88354, u_1} J
                                                           inst✝ : CategoryTheory.Category.{?u.88869, u_2} C
                                                           F : CategoryTheory.Functor J C
                                                           X : C
                                                           τ : Quiver.Hom F ((CategoryTheory.Functor.const J).obj X)
                                                           j j' : Opposite J
                                                           f : Quiver.Hom j j'
                                                           ⊢ Eq ((F.op.comp (CategoryTheory.yoneda.obj X)).map f ((fun j => τ.app (Opposi …
                                                         -/
  invFun τ := ⟨fun j => τ.app j.unop, fun {j j'} f => by simp [τ.naturality f.unop]⟩
                                                         /-
                                                           🎉 no goals
                                                         -/
  left_inv _ := rfl
  right_inv _ := rfl


/-- Sections of `F ⋙ yoneda.obj X` identify to natural
transformations `(const J).obj X ⟶ F`. -/
@[simps]
def compYonedaSectionsEquiv (F : J ⥤ Cᵒᵖ) (X : C) :
    (F ⋙ yoneda.obj X).sections ≃ ((const J).obj (op X) ⟶ F) where
  toFun s :=
    { app := fun j => (s.val j).op
      naturality := fun j j' f => by
        /-
          J : Type u_1
          C : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.91591, u_1} J
          inst✝ : CategoryTheory.Category.{?u.92076, u_2} C
          F : CategoryTheory.Functor J (Opposite C)
          X : C
          s : ↑(F.comp (CategoryTheory.yoneda.obj X)).sections
          j j' : J
          f : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        dsimp
        /-
          J : Type u_1
          C : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.91591, u_1} J
          inst✝ : CategoryTheory.Category.{?u.92076, u_2} C
          F : CategoryTheory.Functor J (Opposite C)
          X : C
          s : ↑(F.comp (CategoryTheory.yoneda.obj X)).sections
          j j' : J
          f : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id { u …
        -/
        rw [Category.id_comp]
        /-
          J : Type u_1
          C : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.91591, u_1} J
          inst✝ : CategoryTheory.Category.{?u.92076, u_2} C
          F : CategoryTheory.Functor J (Opposite C)
          X : C
          s : ↑(F.comp (CategoryTheory.yoneda.obj X)).sections
          j j' : J
          f : Quiver.Hom j j'
          ⊢ Eq (↑s j').op (CategoryTheory.CategoryStruct.comp (↑s j).op (F.map f))
        -/
        exact Quiver.Hom.unop_inj (s.property f).symm }
        /-
          🎉 no goals
        -/
  invFun τ := ⟨fun j => (τ.app j).unop,
                                          /-
                                            J : Type u_1
                                            C : Type u_2
                                            inst✝¹ : CategoryTheory.Category.{?u.91591, u_1} J
                                            inst✝ : CategoryTheory.Category.{?u.92076, u_2} C
                                            F : CategoryTheory.Functor J (Opposite C)
                                            X : C
                                            τ : Quiver.Hom ((CategoryTheory.Functor.const J).obj { unop := X }) F
                                            j j' : J
                                            f : Quiver.Hom j j'
                                            ⊢ Eq (Quiver.Hom.op ((F.comp (CategoryTheory.yoneda.obj X)).map f ((fun j => ( …
                                          -/
    fun {j j'} f => Quiver.Hom.op_inj (by simpa using (τ.naturality f).symm)⟩
                                          /-
                                            🎉 no goals
                                          -/
  left_inv _ := rfl
  right_inv _ := rfl


/-- A cone on `F` with cone point `X` is the same as an element of `lim Hom(X, F·)`. -/
@[simps!]
noncomputable def limitCompCoyonedaIsoCone (F : J ⥤ C) (X : C) :
    limit (F ⋙ coyoneda.obj (op X)) ≅ ((const J).obj X ⟶ F) :=
  ((Types.limitEquivSections _).trans (compCoyonedaSectionsEquiv F X)).toIso


/-- A cone on `F` with cone point `X` is the same as an element of `lim Hom(X, F·)`,
    naturally in `X`. -/
@[simps!]
noncomputable def coyonedaCompLimIsoCones (F : J ⥤ C) :
    coyoneda ⋙ (whiskeringLeft _ _ _).obj F ⋙ lim ≅ F.cones :=
  /-
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    ⊢ ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
  -/
  NatIso.ofComponents (fun X => limitCompCoyonedaIsoCone F X.unop)
  /-
    🎉 no goals
  -/


variable (J) (C) in
/-- A cone on `F` with cone point `X` is the same as an element of `lim Hom(X, F·)`,
    naturally in `F` and `X`. -/
@[simps!]
noncomputable def whiskeringLimYonedaIsoCones : whiskeringLeft _ _ _ ⋙
    (whiskeringRight _ _ _).obj lim ⋙ (whiskeringLeft _ _ _).obj coyoneda ≅ cones J C :=
  /-
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ ∀ {X Y : CategoryTheory.Functor J C} (f : Quiver.Hom X Y), Eq (CategoryTheor …
  -/
  NatIso.ofComponents coyonedaCompLimIsoCones
  /-
    🎉 no goals
  -/


/-- A cocone on `F` with cocone point `X` is the same as an element of `lim Hom(F·, X)`. -/
@[simps!]
noncomputable def limitCompYonedaIsoCocone (F : J ⥤ C) (X : C) :
    limit (F.op ⋙ yoneda.obj X) ≅ (F ⟶ (const J).obj X) :=
  ((Types.limitEquivSections _).trans (opCompYonedaSectionsEquiv F X)).toIso


/-- A cocone on `F` with cocone point `X` is the same as an element of `lim Hom(F·, X)`,
    naturally in `X`. -/
@[simps!]
noncomputable def yonedaCompLimIsoCocones (F : J ⥤ C) :
    yoneda ⋙ (whiskeringLeft _ _ _).obj F.op ⋙ lim ≅ F.cocones :=
  /-
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J C
    ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((C …
  -/
  NatIso.ofComponents (limitCompYonedaIsoCocone F)
  /-
    🎉 no goals
  -/


variable (J) (C) in
/-- A cocone on `F` with cocone point `X` is the same as an element of `lim Hom(F·, X)`,
    naturally in `F` and `X`. -/
@[simps!]
noncomputable def opHomCompWhiskeringLimYonedaIsoCocones : opHom _ _ ⋙ whiskeringLeft _ _ _ ⋙
      (whiskeringRight _ _ _).obj lim ⋙ (whiskeringLeft _ _ _).obj yoneda ≅ cocones J C :=
  /-
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ ∀ {X Y : Opposite (CategoryTheory.Functor J C)} (f : Quiver.Hom X Y), Eq (Ca …
  -/
  NatIso.ofComponents (fun F => yonedaCompLimIsoCocones F.unop)
  /-
    🎉 no goals
  -/



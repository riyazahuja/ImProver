/-- (Implementation)
Given a product diagram in `C/B`, construct the corresponding wide pullback diagram
in `C`.
-/
abbrev widePullbackDiagramOfDiagramOver (B : C) {J : Type w} (F : Discrete J ⥤ Over B) :
    WidePullbackShape J ⥤ C :=
  WidePullbackShape.wideCospan B (fun j => (F.obj ⟨j⟩).left) fun j => (F.obj ⟨j⟩).hom


/-- (Impl) A preliminary definition to avoid timeouts. -/
@[simps]
def conesEquivInverseObj (B : C) {J : Type w} (F : Discrete J ⥤ Over B) (c : Cone F) :
    Cone (widePullbackDiagramOfDiagramOver B F) where
  pt := c.pt.left
  π :=
    { app := fun X => Option.casesOn X c.pt.hom fun j : J => (c.π.app ⟨j⟩).left
      -- `tidy` can do this using `case_bash`, but let's try to be a good `-T50000` citizen:
      naturality := fun X Y f => by
        /-
          J✝ : Type w
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ B : C
          J : Type w
          F : CategoryTheory.Functor (CategoryTheory.Discrete J) (CategoryTheory.Over B)
          c : CategoryTheory.Limits.Cone F
          X Y : CategoryTheory.Limits.WidePullbackShape J
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Cate …
        -/
                                       /-
                                         🎉 no goals
                                       -/
        dsimp; cases X <;> cases Y <;> cases f
          /-
            case none.none.id
            J✝ : Type w
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            X B : C
            J : Type w
            F : CategoryTheory.Functor (CategoryTheory.Discrete J) (CategoryTheory.Over B)
            c : CategoryTheory.Limits.Cone F
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id c.p …
          -/
        · rw [Category.id_comp, Category.comp_id]
          /-
            🎉 no goals
          -/
          /-
            case some.none.term
            J✝ : Type w
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            X B : C
            J : Type w
            F : CategoryTheory.Functor (CategoryTheory.Discrete J) (CategoryTheory.Over B)
            c : CategoryTheory.Limits.Cone F
            val✝ : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id c.p …
          -/
        · rw [Over.w, Category.id_comp]
          /-
            🎉 no goals
          -/
          /-
            case some.some.id
            J✝ : Type w
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            X B : C
            J : Type w
            F : CategoryTheory.Functor (CategoryTheory.Discrete J) (CategoryTheory.Over B)
            c : CategoryTheory.Limits.Cone F
            val✝ : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id c.p …
          -/
        · rw [Category.id_comp, Category.comp_id] }
          /-
            🎉 no goals
          -/


/-- (Impl) A preliminary definition to avoid timeouts. -/
@[simps]
def conesEquivInverse (B : C) {J : Type w} (F : Discrete J ⥤ Over B) :
    Cone F ⥤ Cone (widePullbackDiagramOfDiagramOver B F) where
  obj := conesEquivInverseObj B F
  map f :=
    { hom := f.hom.left
      w := fun j => by
        /-
          J✝ : Type w
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X B : C
          J : Type w
          F : CategoryTheory.Functor (CategoryTheory.Discrete J) (CategoryTheory.Over B)
          X✝ Y✝ : CategoryTheory.Limits.Cone F
          f : Quiver.Hom X✝ Y✝
          j : CategoryTheory.Limits.WidePullbackShape J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom.left ((CategoryTheory.Over.Cons …
        -/
        cases' j with j
          /-
            case none
            J✝ : Type w
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            X B : C
            J : Type w
            F : CategoryTheory.Functor (CategoryTheory.Discrete J) (CategoryTheory.Over B)
            X✝ Y✝ : CategoryTheory.Limits.Cone F
            f : Quiver.Hom X✝ Y✝
            ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom.left ((CategoryTheory.Over.Cons …
          -/
        · simp
          /-
            🎉 no goals
          -/
          /-
            case some
            J✝ : Type w
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            X B : C
            J : Type w
            F : CategoryTheory.Functor (CategoryTheory.Discrete J) (CategoryTheory.Over B)
            X✝ Y✝ : CategoryTheory.Limits.Cone F
            f : Quiver.Hom X✝ Y✝
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom.left ((CategoryTheory.Over.Cons …
          -/
        · dsimp
          /-
            case some
            J✝ : Type w
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            X B : C
            J : Type w
            F : CategoryTheory.Functor (CategoryTheory.Discrete J) (CategoryTheory.Over B)
            X✝ Y✝ : CategoryTheory.Limits.Cone F
            f : Quiver.Hom X✝ Y✝
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom.left (Y✝.π.app { as := j }).lef …
          -/
          rw [← f.w ⟨j⟩]
          /-
            case some
            J✝ : Type w
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            X B : C
            J : Type w
            F : CategoryTheory.Functor (CategoryTheory.Discrete J) (CategoryTheory.Over B)
            X✝ Y✝ : CategoryTheory.Limits.Cone F
            f : Quiver.Hom X✝ Y✝
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom.left (Y✝.π.app { as := j }).lef …
          -/
          rfl }
          /-
            🎉 no goals
          -/

-- Porting note: this should help with the additional `naturality` proof we now have to give in
-- `conesEquivFunctor`, but doesn't.
-- attribute [local aesop safe cases (rule_sets := [CategoryTheory])] Discrete


/-- (Impl) A preliminary definition to avoid timeouts. -/
@[simps]
def conesEquivFunctor (B : C) {J : Type w} (F : Discrete J ⥤ Over B) :
    Cone (widePullbackDiagramOfDiagramOver B F) ⥤ Cone F where
  obj c :=
    { pt := Over.mk (c.π.app none)
      π :=
        { app := fun ⟨j⟩ => Over.homMk (c.π.app (some j)) (c.w (WidePullbackShape.Hom.term j))
          -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10888): added proof for `naturality`
                                                /-
                                                  J✝ : Type w
                                                  C : Type u
                                                  inst✝ : CategoryTheory.Category.{v, u} C
                                                  X✝ B : C
                                                  J : Type w
                                                  F : CategoryTheory.Functor (CategoryTheory.Discrete J) (CategoryTheory.Over B)
                                                  c : CategoryTheory.Limits.Cone (CategoryTheory.Over.ConstructProducts.widePull …
                                                  x✝² x✝¹ : CategoryTheory.Discrete J
                                                  X Y : J
                                                  x✝ : Quiver.Hom { as := X } { as := Y }
                                                  f : Eq { as := X }.as { as := Y }.as
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Cate …
                                                -/
          naturality := fun ⟨X⟩ ⟨Y⟩ ⟨⟨f⟩⟩ => by dsimp at f ⊢; aesop_cat } }
                                                              /-
                                                                🎉 no goals
                                                              -/
                    /-
                      J✝ : Type w
                      C : Type u
                      inst✝ : CategoryTheory.Category.{v, u} C
                      X B : C
                      J : Type w
                      F : CategoryTheory.Functor (CategoryTheory.Discrete J) (CategoryTheory.Over B)
                      X✝ Y✝ : CategoryTheory.Limits.Cone (CategoryTheory.Over.ConstructProducts.wide …
                      f : Quiver.Hom X✝ Y✝
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom ((fun c => { pt := CategoryTheo …
                    -/
  map f := { hom := Over.homMk f.hom }
                    /-
                      🎉 no goals
                    -/

-- Porting note: unfortunately `aesop` can't cope with a `cases` rule here for the type synonym
-- `WidePullbackShape`.
-- attribute [local aesop safe cases (rule_sets := [CategoryTheory])] WidePullbackShape
-- If this worked we could avoid the `rintro` in `conesEquivUnitIso`.


/-- (Impl) A preliminary definition to avoid timeouts. -/
@[simp]
def conesEquivUnitIso (B : C) (F : Discrete J ⥤ Over B) :
    𝟭 (Cone (widePullbackDiagramOfDiagramOver B F)) ≅
      conesEquivFunctor B F ⋙ conesEquivInverse B F :=
  /-
    J : Type w
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X B : C
    F : CategoryTheory.Functor (CategoryTheory.Discrete J) (CategoryTheory.Over B)
    ⊢ ∀ {X Y : CategoryTheory.Limits.Cone (CategoryTheory.Over.ConstructProducts.w …
  -/
  NatIso.ofComponents fun _ => Cones.ext
  /-
    🎉 no goals
  -/
        /-
          J : Type w
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X B : C
          F : CategoryTheory.Functor (CategoryTheory.Discrete J) (CategoryTheory.Over B)
          x✝ : CategoryTheory.Limits.Cone (CategoryTheory.Over.ConstructProducts.widePul …
          ⊢ ∀ (j : CategoryTheory.Limits.WidePullbackShape J), Eq (((CategoryTheory.Func …
        -/
                           /-
                             🎉 no goals
                           -/
    { hom := 𝟙 _
                           /-
                             🎉 no goals
                           -/
      inv := 𝟙 _ }
    (by rintro (j | j) <;> aesop_cat)

-- TODO: Can we add `:= by aesop` to the second arguments of `NatIso.ofComponents` and
--       `Cones.ext`?

/-- (Impl) A preliminary definition to avoid timeouts. -/
@[simp]
def conesEquivCounitIso (B : C) (F : Discrete J ⥤ Over B) :
    conesEquivInverse B F ⋙ conesEquivFunctor B F ≅ 𝟭 (Cone F) :=
                               /-
                                 J : Type w
                                 C : Type u
                                 inst✝ : CategoryTheory.Category.{v, u} C
                                 X B : C
                                 F : CategoryTheory.Functor (CategoryTheory.Discrete J) (CategoryTheory.Over B)
                                 x✝ : CategoryTheory.Limits.Cone F
                                 ⊢ ∀ (j : CategoryTheory.Discrete J), Eq ((((CategoryTheory.Over.ConstructProdu …
                               -/
             /-
               J : Type w
               C : Type u
               inst✝ : CategoryTheory.Category.{v, u} C
               X B : C
               F : CategoryTheory.Functor (CategoryTheory.Discrete J) (CategoryTheory.Over B)
               x✝ : CategoryTheory.Limits.Cone F
               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((( …
             -/
                               /-
                                 🎉 no goals
                               -/
             /-
               🎉 no goals
             -/
             /-
               J : Type w
               C : Type u
               inst✝ : CategoryTheory.Category.{v, u} C
               X B : C
               F : CategoryTheory.Functor (CategoryTheory.Discrete J) (CategoryTheory.Over B)
               x✝ : CategoryTheory.Limits.Cone F
               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((C …
             -/
  NatIso.ofComponents fun _ => Cones.ext
             /-
               🎉 no goals
             -/
  /-
    🎉 no goals
  -/
    { hom := Over.homMk (𝟙 _)
      inv := Over.homMk (𝟙 _) }


/-- (Impl) Establish an equivalence between the category of cones for `F` and for the "grown" `F`.
-/
@[simps]
def conesEquiv (B : C) (F : Discrete J ⥤ Over B) :
    Cone (widePullbackDiagramOfDiagramOver B F) ≌ Cone F where
  functor := conesEquivFunctor B F
  inverse := conesEquivInverse B F
  unitIso := conesEquivUnitIso B F
  counitIso := conesEquivCounitIso B F


/-- Use the above equivalence to prove we have a limit. -/
theorem has_over_limit_discrete_of_widePullback_limit {B : C} (F : Discrete J ⥤ Over B)
    [HasLimit (widePullbackDiagramOfDiagramOver B F)] : HasLimit F :=
  HasLimit.mk
    { cone := _
      isLimit := IsLimit.ofRightAdjoint (conesEquiv B F).symm.toAdjunction
        (limit.isLimit (widePullbackDiagramOfDiagramOver B F)) }


/-- Given a wide pullback in `C`, construct a product in `C/B`. -/
theorem over_product_of_widePullback [HasLimitsOfShape (WidePullbackShape J) C] {B : C} :
    HasLimitsOfShape (Discrete J) (Over B) :=
  { has_limit := fun F => has_over_limit_discrete_of_widePullback_limit F }


/-- Given a pullback in `C`, construct a binary product in `C/B`. -/
theorem over_binaryProduct_of_pullback [HasPullbacks C] {B : C} : HasBinaryProducts (Over B) :=
  over_product_of_widePullback


/-- Given all wide pullbacks in `C`, construct products in `C/B`. -/
theorem over_products_of_widePullbacks [HasWidePullbacks.{w} C] {B : C} :
    HasProducts.{w} (Over B) :=
  fun _ => over_product_of_widePullback


/-- Given all finite wide pullbacks in `C`, construct finite products in `C/B`. -/
theorem over_finiteProducts_of_finiteWidePullbacks [HasFiniteWidePullbacks C] {B : C} :
    HasFiniteProducts (Over B) :=
  ⟨fun _ => over_product_of_widePullback⟩


/-- Construct terminal object in the over category. This isn't an instance as it's not typically the
way we want to define terminal objects.
(For instance, this gives a terminal object which is different from the generic one given by
`over_product_of_widePullback` above.)
-/
theorem over_hasTerminal (B : C) : HasTerminal (Over B) where
  has_limit F := HasLimit.mk
    { cone :=
        { pt := Over.mk (𝟙 _)
          π :=
            { app := fun p => p.as.elim } }
      isLimit :=
                           /-
                             C : Type u
                             inst✝ : CategoryTheory.Category.{v, u} C
                             B : C
                             F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) (CategoryTheor …
                             s : CategoryTheory.Limits.Cone F
                             ⊢ Eq (CategoryTheory.CategoryStruct.comp s.pt.hom { pt := CategoryTheory.Over. …
                           -/
        { lift := fun s => Over.homMk s.pt.hom
                           /-
                             🎉 no goals
                           -/
          fac := fun _ j => j.as.elim
          uniq := fun s m _ => by
            /-
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              B : C
              F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) (CategoryTheor …
              s : CategoryTheory.Limits.Cone F
              m : Quiver.Hom s.pt { pt := CategoryTheory.Over.mk (CategoryTheory.CategoryStr …
              x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{1}), Eq (CategoryTheory.CategorySt …
              ⊢ Eq m ((fun s => CategoryTheory.Over.homMk s.pt.hom ⋯) s)
            -/
            simp only
            /-
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              B : C
              F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) (CategoryTheor …
              s : CategoryTheory.Limits.Cone F
              m : Quiver.Hom s.pt { pt := CategoryTheory.Over.mk (CategoryTheory.CategoryStr …
              x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{1}), Eq (CategoryTheory.CategorySt …
              ⊢ Eq m (CategoryTheory.Over.homMk s.pt.hom ⋯)
            -/
            ext
            /-
              case h
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              B : C
              F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) (CategoryTheor …
              s : CategoryTheory.Limits.Cone F
              m : Quiver.Hom s.pt { pt := CategoryTheory.Over.mk (CategoryTheory.CategoryStr …
              x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{1}), Eq (CategoryTheory.CategorySt …
              ⊢ Eq m.left (CategoryTheory.Over.homMk s.pt.hom ⋯).left
            -/
            rw [Over.homMk_left _]
            /-
              case h
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              B : C
              F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) (CategoryTheor …
              s : CategoryTheory.Limits.Cone F
              m : Quiver.Hom s.pt { pt := CategoryTheory.Over.mk (CategoryTheory.CategoryStr …
              x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{1}), Eq (CategoryTheory.CategorySt …
              ⊢ Eq m.left s.pt.hom
            -/
            have := m.w
            /-
              case h
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              B : C
              F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) (CategoryTheor …
              s : CategoryTheory.Limits.Cone F
              m : Quiver.Hom s.pt { pt := CategoryTheory.Over.mk (CategoryTheory.CategoryStr …
              x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{1}), Eq (CategoryTheory.CategorySt …
              this : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id C).m …
              ⊢ Eq m.left s.pt.hom
            -/
            dsimp at this
            /-
              case h
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              B : C
              F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) (CategoryTheor …
              s : CategoryTheory.Limits.Cone F
              m : Quiver.Hom s.pt { pt := CategoryTheory.Over.mk (CategoryTheory.CategoryStr …
              x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{1}), Eq (CategoryTheory.CategorySt …
              this : Eq (CategoryTheory.CategoryStruct.comp m.left (CategoryTheory.CategoryS …
              ⊢ Eq m.left s.pt.hom
            -/
            rwa [Category.comp_id, Category.comp_id] at this } }
            /-
              🎉 no goals
            -/


lemma isPullback_of_binaryFan_isLimit (c : BinaryFan Y Z) (hc : IsLimit c) :
    IsPullback c.fst.left c.snd.left Y.hom Z.hom :=
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X : C
        Y Z : CategoryTheory.Over X
        c : CategoryTheory.Limits.BinaryFan Y Z
        hc : CategoryTheory.Limits.IsLimit c
        ⊢ CategoryTheory.CommSq c.fst.left c.snd.left Y.hom Z.hom
      -/
  ⟨by simp, ⟨((IsLimit.postcomposeHomEquiv (diagramIsoCospan _) _).symm
      /-
        🎉 no goals
      -/
    ((IsLimit.ofConeEquiv (ConstructProducts.conesEquiv X _).symm).symm hc)).ofIsoLimit
    (PullbackCone.isoMk _)⟩⟩


/-- The product of `Y` and `Z` in `Over X` is isomorpic to `Y ×ₓ Z`. -/
noncomputable
def prodLeftIsoPullback :
    (Y ⨯ Z).left ≅ pullback Y.hom Z.hom :=
  (Over.isPullback_of_binaryFan_isLimit _ (prodIsProd Y Z)).isoPullback


@[reassoc (attr := simp)]
lemma prodLeftIsoPullback_hom_fst :
    (prodLeftIsoPullback Y Z).hom ≫ pullback.fst _ _ = (prod.fst (X := Y)).left :=
  IsPullback.isoPullback_hom_fst _


@[reassoc (attr := simp)]
lemma prodLeftIsoPullback_hom_snd :
    (prodLeftIsoPullback Y Z).hom ≫ pullback.snd _ _ = (prod.snd (X := Y)).left :=
  IsPullback.isoPullback_hom_snd _


@[reassoc (attr := simp)]
lemma prodLeftIsoPullback_inv_fst :
    (prodLeftIsoPullback Y Z).inv ≫ (prod.fst (X := Y)).left = pullback.fst _ _ :=
  IsPullback.isoPullback_inv_fst _


@[reassoc (attr := simp)]
lemma prodLeftIsoPullback_inv_snd :
    (prodLeftIsoPullback Y Z).inv ≫ (prod.snd (X := Y)).left = pullback.snd _ _ :=
  IsPullback.isoPullback_inv_snd _


